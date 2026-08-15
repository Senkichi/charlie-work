"""In-process wedge watchdog for the fleet supervisor (issue #728).

The scheduled task's 5-minute trigger detects a **crashed** supervisor — a
dead process leaves no instance, ``MultipleInstancesPolicy=IgnoreNew`` stops
suppressing, and the next tick relaunches. But it cannot detect a **wedged**
one: the process still exists, so ``IgnoreNew`` suppresses every subsequent
trigger; ``ExecutionTimeLimit`` is ``PT0S`` (correctly — a real limit would
kill a healthy long-lived daemon on a timer); and ``LastTaskResult`` reads
``267009`` ("currently running"), indistinguishable from healthy. So the one
failure the old VBS comment claimed was covered ("hung-pass kill") is the one
failure with no coverage at all.

This module closes that gap. It runs as a daemon thread **inside the
``supervise-loop`` wrapper process** — which is alive whenever the child is
alive (it is blocked on ``process.wait()``) and owns the child's ``Popen``
handle, so it can terminate the child directly. It periodically reads
``supervisor-heartbeat.json`` (the same sidecar
``scripts/heartbeat_check.py`` already monitors passively) and, if
``last_beat_at`` is stale beyond a threshold derived from the config knob
stamped in the heartbeat, kills the child. The kill produces a non-``EXIT_RESTART_REQUESTED``
exit code, so the wrapper exits and the scheduled task's next tick relaunches
a fresh daemon — exactly the recovery path that already exists for a crash.

**Why the threshold is keyed on ``max_pass_runtime_seconds``, not
``full_pass_interval_seconds``.** The issue proposed "a generous multiple of
``full_pass_interval_seconds``", but that is the bound on pass *cadence*
(how long between passes), while the heartbeat is refreshed at the top of
every loop *iteration* — and a single healthy pass can run for up to
``max_pass_runtime_seconds`` (default 1800 s) without returning to the top of
the loop to re-beat. Keying on ``full_pass_interval_seconds`` (300 s) would
false-kill a supervisor in the middle of a legitimate 30-minute pass.
``max_pass_runtime_seconds`` is the config knob that actually bounds how long
a healthy daemon can go between heartbeat refreshes, so it is the correct
base. Both knobs are stamped in the heartbeat by ``record_supervisor_started``
and read back here, so the threshold tracks config without the wrapper loading
config itself (the wrapper may be running pre-deploy code; the heartbeat is
written by the fresh child).

**Why this is a kill, not a passive report.** ``heartbeat_check.py`` already
detects staleness passively (``check_supervisor_heartbeat``), but it only
prints ``ANOMALY`` and exits 1 — it does not terminate the wedged process, so
``IgnoreNew`` keeps suppressing the trigger and the daemon stays wedged
indefinitely. The gap this module fills is the *actuation*, not the detection.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from subprocess import Popen
from typing import Any

from .instrumentation import log_event

logger = logging.getLogger(__name__)

# Event kind recorded when the watchdog kills a wedged supervisor child.
# Registered in ``instrumentation._LEVEL_BY_KIND`` as error-level.
WEDGE_KILL_EVENT_KIND = "supervisor_wedged_killed"

# Multiplier on ``max_pass_runtime_seconds`` (read from the heartbeat) that
# yields the staleness threshold. 3x is deliberately generous: the watchdog
# performs an *active kill* (more destructive than the passive
# ``check_supervisor_heartbeat`` report, which uses 2x), so it must be
# strictly harder to trip. With the default ``max_pass_runtime_seconds=1800``
# this gives a 5400 s / 90 min threshold — well beyond a healthy 30-minute
# pass plus the post-pass cooldown and poll sleep, with margin for a slow
# sibling-repo pass. The base is config-derived (stamped in the heartbeat),
# so the threshold tracks config changes; only the multiplier is a constant.
WEDGE_KILL_STALE_MULTIPLIER = 3

# Fallback pass-timeout (seconds) when the heartbeat lacks both
# ``max_pass_runtime_seconds`` and ``full_pass_interval_seconds`` — e.g. a
# heartbeat written by a supervisor older than the fields were added. Matches
# the ``SupervisorConfig.max_pass_runtime_seconds`` default.
WEDGE_KILL_DEFAULT_PASS_TIMEOUT_SECONDS = 1800

# How often the watchdog polls the heartbeat. With a 90-minute threshold, a
# 60-second poll means detection within ~90-91 minutes of the last beat —
# tight enough that the gap is bounded, loose enough that the watchdog's own
# I/O (one small JSON read per minute) is negligible against the supervisor's
# per-pass work.
WEDGE_KILL_POLL_INTERVAL_SECONDS = 60.0

# Grace window (seconds) for the watched child's first heartbeat. The
# heartbeat sidecar is written by the child on startup
# (``record_supervisor_started``), but there is a window between ``Popen``
# returning and that first write: Python interpreter startup, imports,
# config load, and supervisor-lock acquisition. During that window the
# on-disk heartbeat (if any) is stale residue from a *prior* supervisor
# with a different pid. If the watchdog treated that stale heartbeat as a
# liveness signal for the fresh child, it would either false-kill a healthy
# child (stale heartbeat is old) or false-clear a wedged child (stale
# heartbeat is fresh) — reproducing the original fail-open bug from #728.
#
# Instead, while the heartbeat's pid does not match the watched child's
# pid, the watchdog waits. After this grace window expires with no
# matching heartbeat, the child is treated as wedged at startup (a
# crashed child is caught by ``process.poll()``; a child that wedges
# before its first beat is not). 300 s is generous for startup + imports
# + lock acquisition (typically seconds) but bounded well below the
# 90-minute staleness threshold so a startup-wedged child is killed in
# minutes, not hours.
WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS = 300.0


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp (with ``Z`` or ``+00:00`` suffix) to UTC."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class WedgeWatchdog:
    """Daemon-thread watchdog that terminates a wedged supervisor child.

    Created by the ``supervise-loop`` wrapper for each spawned child. The
    main thread calls ``process.wait()``; this watchdog runs concurrently and
    kills the child if its heartbeat goes stale. After the kill,
    ``process.wait()`` returns with a non-``EXIT_RESTART_REQUESTED`` exit
    code, so the wrapper exits and the scheduled task's next tick relaunches
    a fresh daemon.

    All I/O dependencies (heartbeat path, clock, sleep, log, event sink) are
    injected so the watchdog is fully testable without real processes or
    files. The watchdog never raises — a failure inside the thread is logged
    and swallowed so it cannot crash the wrapper or interfere with
    ``process.wait()``.
    """

    def __init__(
        self,
        process: Popen[Any],
        heartbeat_path: Path,
        *,
        stale_multiplier: int = WEDGE_KILL_STALE_MULTIPLIER,
        default_pass_timeout_seconds: int = WEDGE_KILL_DEFAULT_PASS_TIMEOUT_SECONDS,
        poll_interval_seconds: float = WEDGE_KILL_POLL_INTERVAL_SECONDS,
        first_beat_grace_seconds: float = WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS,
        clock: Callable[[], datetime] | None = None,
        log: Callable[[str], None] | None = None,
        sleep_func: Callable[[float], None] | None = None,
        log_event_fn: Callable[..., None] | None = None,
    ) -> None:
        self._process = process
        self._heartbeat_path = heartbeat_path
        self._stale_multiplier = stale_multiplier
        self._default_pass_timeout_seconds = default_pass_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._first_beat_grace_seconds = first_beat_grace_seconds
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self._log = log if log is not None else lambda msg: print(msg, flush=True)
        self._sleep = sleep_func if sleep_func is not None else time.sleep
        self._log_event_fn = log_event_fn if log_event_fn is not None else log_event
        self._killed = False
        # Captured at construction time (right after Popen returns) so the
        # first-beat grace window is measured from the child's actual start,
        # not from the first poll (which is one ``poll_interval_seconds``
        # later). See ``_is_wedged`` for why this matters.
        self._child_started_at = self._clock()

    @property
    def killed(self) -> bool:
        """True if this watchdog terminated the child (for caller diagnostics)."""
        return self._killed

    def start(self) -> threading.Thread:
        """Start the watchdog as a daemon thread and return it.

        Daemon so it never outlives the wrapper process — if the wrapper exits
        (cap reached, uncaught exception), the thread is killed automatically.
        """
        thread = threading.Thread(target=self._run, name="wedge-watchdog", daemon=True)
        thread.start()
        return thread

    def _derive_pass_timeout(self, heartbeat: dict[str, Any]) -> int:
        """Derive the per-pass timeout (seconds) from the heartbeat's config fields.

        ``max_pass_runtime_seconds`` is the primary source (it bounds a single
        pass's wall-clock runtime, which is what gates heartbeat freshness).
        Falls back to ``full_pass_interval_seconds`` for older heartbeats that
        predate the field, then to the module default.
        """
        for key in ("max_pass_runtime_seconds", "full_pass_interval_seconds"):
            raw = heartbeat.get(key)
            try:
                value = int(raw) if raw is not None else None
            except (TypeError, ValueError):
                value = None
            if value is not None and value > 0:
                return value
        return self._default_pass_timeout_seconds

    def _read_heartbeat(self) -> dict[str, Any] | None:
        """Read and parse the heartbeat sidecar, returning ``None`` on any error.

        Missing, unreadable, or malformed files all resolve to ``None`` — the
        caller treats that as "no signal yet" and skips the check, which is
        correct for the brief startup window before the child writes its first
        heartbeat. A child that crashed on startup will be caught by
        ``process.poll()`` on the next iteration, not by this read.
        """
        if not self._heartbeat_path.exists():
            return None
        try:
            data = json.loads(self._heartbeat_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "WedgeWatchdog: heartbeat at %s unreadable (%s); skipping check",
                self._heartbeat_path,
                exc,
            )
            return None
        return data if isinstance(data, dict) else None

    def _elapsed_since_start(self) -> float:
        """Seconds since the watchdog began watching this child."""
        return (self._clock() - self._child_started_at).total_seconds()

    def _is_wedged(self) -> tuple[bool, dict[str, Any] | None]:
        """Return ``(wedged, heartbeat)``.

        ``wedged`` is True only when the child is alive and one of these
        holds:

        - The heartbeat's ``pid`` matches the watched child's pid, the
          heartbeat has a parseable ``last_beat_at``, the heartbeat does
          not record a clean exit (``exited_at`` is null), and the
          heartbeat age exceeds the derived threshold.
        - The heartbeat's ``pid`` does **not** match the watched child
          (or there is no heartbeat file at all), and the first-beat
          grace window has expired — the child has been alive long enough
          without writing its own heartbeat that it is wedged at startup.

        The pid correlation is the fix for the fail-open bug the original
        issue (#728) was about: without it, a stale heartbeat left by a
        *prior* supervisor (different pid) is treated as this child's
        liveness signal. A stale-but-old prior heartbeat false-kills a
        healthy fresh child; a stale-but-fresh prior heartbeat
        false-clears a child that wedged before its first beat. Both are
        eliminated by refusing to act on a heartbeat whose pid is not the
        watched child's pid, with a bounded grace window for the child's
        first beat.

        Returns the heartbeat dict alongside the verdict so the caller
        can include its fields in the kill log/event.
        """
        heartbeat = self._read_heartbeat()
        child_pid = self._process.pid
        if heartbeat is None:
            # No heartbeat file at all. Within the grace window the child
            # may simply not have written it yet. After the grace window,
            # the child has been alive long enough without writing any
            # heartbeat that it is wedged at startup — a crashed child is
            # caught by ``process.poll()``, but a child that wedges before
            # its first beat is not.
            if self._elapsed_since_start() > self._first_beat_grace_seconds:
                return True, None
            return False, None
        # Correlate the heartbeat to the watched child. A heartbeat from a
        # different (prior) pid is stale residue, not a liveness signal for
        # this child. The prior supervisor may have crashed (null
        # ``exited_at``) or exited cleanly (``exited_at`` set) — either
        # way, its heartbeat tells us nothing about *this* child.
        heartbeat_pid = heartbeat.get("pid")
        if heartbeat_pid is not None and child_pid is not None and heartbeat_pid != child_pid:
            if self._elapsed_since_start() > self._first_beat_grace_seconds:
                return True, heartbeat
            return False, heartbeat
        # The heartbeat is from this child (or pid is unknown on either
        # side — conservative fallback to the pre-correlation behavior).
        if heartbeat.get("exited_at"):
            # The supervisor recorded a clean exit; ``process.poll()`` will
            # catch it on the next iteration. Do not kill a self-exiting child.
            return False, heartbeat
        last_beat = _parse_iso(heartbeat.get("last_beat_at"))
        if last_beat is None:
            return False, heartbeat
        age_seconds = (self._clock() - last_beat).total_seconds()
        pass_timeout = self._derive_pass_timeout(heartbeat)
        threshold_seconds = self._stale_multiplier * pass_timeout
        return age_seconds > threshold_seconds, heartbeat

    def _run(self) -> None:
        """Main watchdog loop: poll, check, kill if wedged.

        Never raises — an unexpected exception inside the thread is logged and
        swallowed so the wrapper's ``process.wait()`` is never disturbed.

        If ``_kill`` reports that ``process.kill()`` raised, the loop
        *continues* rather than returning: the child is still alive and
        still wedged, so the next iteration re-checks ``process.poll()``
        (the child may have died on its own) and retries the kill. This
        avoids silently reporting success (``self._killed = True``) for a
        kill that never happened.
        """
        try:
            while True:
                self._sleep(self._poll_interval_seconds)
                if self._process.poll() is not None:
                    # Child exited on its own (clean, crash, or restart
                    # request) — nothing to watch. ``process.wait()`` in the
                    # main thread is about to return.
                    return
                wedged, heartbeat = self._is_wedged()
                if not wedged:
                    continue
                if self._kill(heartbeat):
                    return
                # Kill failed (``process.kill()`` raised). Continue
                # monitoring — the child is still alive and still wedged.
        except Exception:
            logger.exception("WedgeWatchdog: unexpected error in monitor loop")

    def _kill(self, heartbeat: dict[str, Any] | None) -> bool:
        """Terminate the wedged child, log loudly, and record an event.

        ``process.kill()`` is used rather than ``terminate()`` because a
        wedged process may not respond to SIGTERM (POSIX) — the whole point is
        that it is stuck. On Windows ``kill()`` is ``TerminateProcess``,
        which skips Python ``finally`` blocks, so the supervisor's own
        ``record_supervisor_exit`` does not run; the heartbeat keeps
        ``exited_at=null``, and the next supervisor start detects the
        abnormal exit via ``detect_prior_abnormal_exit``. This event is the
        explicit record that the kill was deliberate, not a mystery crash.

        Returns ``True`` only when ``process.kill()`` succeeded (and sets
        ``self._killed``). Returns ``False`` when ``kill()`` raised — the
        caller must continue monitoring rather than treating the child as
        killed. ``heartbeat`` may be ``None`` when the wedge verdict came
        from the first-beat grace window expiring with no heartbeat file
        at all.

        The event payload's ``pid`` is always ``self._process.pid`` — the
        process actually terminated — because that is the only pid known
        with certainty on every kill path (the heartbeat may be absent, or
        carry a *prior* supervisor's pid). The heartbeat's stated pid, when
        present, is preserved separately as ``heartbeat_pid`` so the
        forensic record still shows which (possibly stale) heartbeat drove
        the verdict.
        """
        hb = heartbeat if heartbeat is not None else {}
        heartbeat_pid = hb.get("pid")
        last_beat = hb.get("last_beat_at")
        pass_timeout = self._derive_pass_timeout(hb)
        threshold_seconds = self._stale_multiplier * pass_timeout
        age_seconds: float | None = None
        last_beat_dt = _parse_iso(last_beat)
        if last_beat_dt is not None:
            age_seconds = (self._clock() - last_beat_dt).total_seconds()
        # ``age_seconds`` is ``None`` when there is no heartbeat at all (the
        # first-beat grace window expired with no file) or when the heartbeat
        # lacks a parseable ``last_beat_at``. Formatting ``None`` with ``:.0f``
        # raises ``TypeError`` *before* ``process.kill()`` is reached, silently
        # defeating the exact no-heartbeat wedge case this watchdog exists to
        # kill — so render ``unknown`` instead of formatting the value.
        age_display = f"{age_seconds:.0f}s" if age_seconds is not None else "unknown"
        message = (
            f"wedge-watchdog: supervisor child pid={self._process.pid} heartbeat stale "
            f"({last_beat}); age={age_display} exceeds threshold="
            f"{threshold_seconds:.0f}s ({self._stale_multiplier}x "
            f"max_pass_runtime_seconds={pass_timeout}s). Terminating so the "
            f"scheduled task's next tick relaunches a fresh daemon."
        )
        self._log(message)
        try:
            self._log_event_fn(
                self._heartbeat_path,
                WEDGE_KILL_EVENT_KIND,
                {
                    "pid": self._process.pid,
                    "heartbeat_pid": heartbeat_pid,
                    "last_beat_at": last_beat,
                    "age_seconds": age_seconds,
                    "threshold_seconds": threshold_seconds,
                    "stale_multiplier": self._stale_multiplier,
                    "pass_timeout_seconds": pass_timeout,
                },
                repo="fleet",
            )
        except Exception:
            logger.exception("WedgeWatchdog: failed to record kill event")
        try:
            self._process.kill()
        except Exception:
            logger.exception("WedgeWatchdog: failed to kill process pid=%s", self._process.pid)
            return False
        self._killed = True
        return True
