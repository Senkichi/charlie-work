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
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self._log = log if log is not None else lambda msg: print(msg, flush=True)
        self._sleep = sleep_func if sleep_func is not None else time.sleep
        self._log_event_fn = log_event_fn if log_event_fn is not None else log_event
        self._killed = False

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

    def _is_wedged(self) -> tuple[bool, dict[str, Any] | None]:
        """Return ``(wedged, heartbeat)``.

        ``wedged`` is True only when the child is alive, the heartbeat has a
        parseable ``last_beat_at``, the heartbeat does not record a clean exit
        (``exited_at`` is null — a clean-exiting child will be reaped by
        ``process.poll()`` momentarily), and the heartbeat age exceeds the
        derived threshold. Returns the heartbeat dict alongside the verdict so
        the caller can include its fields in the kill log/event.
        """
        heartbeat = self._read_heartbeat()
        if heartbeat is None:
            return False, None
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
                if not wedged or heartbeat is None:
                    continue
                self._kill(heartbeat)
                return
        except Exception:
            logger.exception("WedgeWatchdog: unexpected error in monitor loop")

    def _kill(self, heartbeat: dict[str, Any]) -> None:
        """Terminate the wedged child, log loudly, and record an event.

        ``process.kill()`` is used rather than ``terminate()`` because a
        wedged process may not respond to SIGTERM (POSIX) — the whole point is
        that it is stuck. On Windows ``kill()`` is ``TerminateProcess``,
        which skips Python ``finally`` blocks, so the supervisor's own
        ``record_supervisor_exit`` does not run; the heartbeat keeps
        ``exited_at=null``, and the next supervisor start detects the
        abnormal exit via ``detect_prior_abnormal_exit``. This event is the
        explicit record that the kill was deliberate, not a mystery crash.
        """
        pid = heartbeat.get("pid")
        last_beat = heartbeat.get("last_beat_at")
        pass_timeout = self._derive_pass_timeout(heartbeat)
        threshold_seconds = self._stale_multiplier * pass_timeout
        age_seconds: float | None = None
        last_beat_dt = _parse_iso(last_beat)
        if last_beat_dt is not None:
            age_seconds = (self._clock() - last_beat_dt).total_seconds()
        message = (
            f"wedge-watchdog: supervisor child pid={pid} heartbeat stale "
            f"({last_beat}); age={age_seconds:.0f}s exceeds threshold="
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
                    "pid": pid,
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
            logger.exception("WedgeWatchdog: failed to kill process pid=%s", pid)
        self._killed = True
