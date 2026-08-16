"""Tests for the in-process wedge watchdog (issue #728).

Covers the ``WedgeWatchdog`` class that detects a wedged supervisor child by
heartbeat staleness and terminates it, plus the wiring through
``run_fleet_supervise_loop`` / ``_spawn_supervise_child``.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from charlie_work.fleet_dispatch import (
    _spawn_supervise_child,
    run_fleet_supervise_loop,
)
from charlie_work.wedge_watchdog import (
    WEDGE_KILL_DEFAULT_PASS_TIMEOUT_SECONDS,
    WEDGE_KILL_EVENT_KIND,
    WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS,
    WEDGE_KILL_STALE_MULTIPLIER,
    WedgeWatchdog,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeProcess:
    """A minimal ``Popen``-shaped fake with controllable ``poll`` and ``kill``.

    ``poll_results`` is consumed left-to-right; when exhausted, ``poll``
    returns ``0`` (child exited). ``kill`` records the call and pushes a
    non-``None`` poll result so the watchdog's next ``poll()`` sees a dead
    child (mirroring real ``Popen`` semantics where ``kill`` is followed by
    ``poll`` returning the exit code).

    ``pid`` defaults to ``12345`` to match the heartbeat's pid in
    ``_write_heartbeat``, so existing tests where the heartbeat and process
    belong to the same supervisor pass the pid-correlation check. Set a
    different ``pid`` to simulate a stale heartbeat from a prior supervisor.

    ``kill_raises`` makes ``kill()`` raise ``OSError`` instead of
    terminating, simulating a kill failure (e.g. the process already died
    and was reaped by another thread, or a permission error).
    """

    def __init__(
        self,
        poll_results: list[int | None] | None = None,
        *,
        block_until_killed: bool = False,
        pid: int = 12345,
        kill_raises: bool = False,
    ) -> None:
        self._poll_results = list(poll_results) if poll_results else [None]
        self.killed = False
        self.kill_count = 0
        self._wait_return: int = 0
        self._block_until_killed = block_until_killed
        self._killed_event = threading.Event()
        self.pid = pid
        self._kill_raises = kill_raises

    def wait(self) -> int:
        if self._block_until_killed:
            self._killed_event.wait(timeout=10.0)
        return self._wait_return

    def poll(self) -> int | None:
        if self._poll_results:
            return self._poll_results.pop(0)
        return 0

    def kill(self) -> None:
        if self._kill_raises:
            raise OSError("simulated kill failure")
        self.killed = True
        self.kill_count += 1
        self._killed_event.set()
        # After kill, poll should report the child dead.
        if not self._poll_results or self._poll_results[-1] is None:
            self._poll_results.append(1)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_clock(start: datetime, late: datetime) -> Callable[[], datetime]:
    """Return a clock that yields ``start`` once then ``late`` forever.

    The first call (in ``WedgeWatchdog.__init__``) captures the child's
    start time; subsequent calls (in ``_elapsed_since_start``) return
    ``late`` so the grace window is exceeded.
    """
    state = {"first": True}

    def _clock() -> datetime:
        if state["first"]:
            state["first"] = False
            return start
        return late

    return _clock


def _write_heartbeat(
    path: Path,
    *,
    last_beat_at: str | None,
    max_pass_runtime_seconds: int | None = 300,
    full_pass_interval_seconds: int | None = 300,
    exited_at: str | None = None,
    pid: int = 12345,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "pid": pid,
        "started_at": last_beat_at,
        "last_beat_at": last_beat_at,
        "pass_number": 1,
        "exited_at": exited_at,
        "exit_code": None,
    }
    if max_pass_runtime_seconds is not None:
        payload["max_pass_runtime_seconds"] = max_pass_runtime_seconds
    if full_pass_interval_seconds is not None:
        payload["full_pass_interval_seconds"] = full_pass_interval_seconds
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit tests for WedgeWatchdog._is_wedged / _derive_pass_timeout
# ---------------------------------------------------------------------------


def test_is_wedged_true_when_heartbeat_stale(tmp_path: Path) -> None:
    """A heartbeat older than ``multiplier * max_pass_runtime_seconds`` is wedged."""
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    # max_pass_runtime_seconds=300, multiplier=3 → threshold=900s.
    # last_beat 1000s ago → wedged.
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=1000)),
        max_pass_runtime_seconds=300,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat = wd._is_wedged()
    assert wedged is True
    assert heartbeat is not None
    assert heartbeat["pid"] == 12345


def test_is_wedged_false_when_heartbeat_fresh(tmp_path: Path) -> None:
    """A heartbeat within the threshold is not wedged."""
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    # max_pass_runtime_seconds=300, multiplier=3 → threshold=900s.
    # last_beat 60s ago → healthy.
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=60)),
        max_pass_runtime_seconds=300,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, _heartbeat = wd._is_wedged()
    assert wedged is False


def test_is_wedged_false_when_no_heartbeat(tmp_path: Path) -> None:
    """No heartbeat file → not wedged (child may not have written it yet)."""
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat = wd._is_wedged()
    assert wedged is False
    assert heartbeat is None


def test_is_wedged_false_when_supervisor_recorded_clean_exit(tmp_path: Path) -> None:
    """A heartbeat with ``exited_at`` set is a self-exiting child — do not kill."""
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
        exited_at=_iso(now),
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, _heartbeat = wd._is_wedged()
    assert wedged is False


def test_is_wedged_false_when_last_beat_unparseable(tmp_path: Path) -> None:
    """A heartbeat with a garbage ``last_beat_at`` is not a kill signal."""
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at="not-a-timestamp",
        max_pass_runtime_seconds=300,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, _heartbeat = wd._is_wedged()
    assert wedged is False


def test_derive_pass_timeout_uses_max_pass_runtime_seconds(tmp_path: Path) -> None:
    """``max_pass_runtime_seconds`` is the primary source, not ``full_pass_interval_seconds``."""
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(datetime.now(UTC)),
        max_pass_runtime_seconds=1800,
        full_pass_interval_seconds=300,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    heartbeat = json.loads(hb_path.read_text(encoding="utf-8"))
    assert wd._derive_pass_timeout(heartbeat) == 1800


def test_derive_pass_timeout_falls_back_to_full_pass_interval(tmp_path: Path) -> None:
    """Older heartbeats without ``max_pass_runtime_seconds`` fall back to the interval."""
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(datetime.now(UTC)),
        max_pass_runtime_seconds=None,
        full_pass_interval_seconds=300,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    heartbeat = json.loads(hb_path.read_text(encoding="utf-8"))
    assert wd._derive_pass_timeout(heartbeat) == 300


def test_derive_pass_timeout_falls_back_to_default(tmp_path: Path) -> None:
    """When both config fields are absent, the module default is used."""
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(datetime.now(UTC)),
        max_pass_runtime_seconds=None,
        full_pass_interval_seconds=None,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    heartbeat = json.loads(hb_path.read_text(encoding="utf-8"))
    assert wd._derive_pass_timeout(heartbeat) == WEDGE_KILL_DEFAULT_PASS_TIMEOUT_SECONDS


def test_threshold_does_not_false_kill_on_a_healthy_long_pass(tmp_path: Path) -> None:
    """A pass running for exactly ``max_pass_runtime_seconds`` must NOT be wedged.

    This is the core reason the threshold is keyed on
    ``max_pass_runtime_seconds`` rather than ``full_pass_interval_seconds``: a
    single healthy pass can run for up to ``max_pass_runtime_seconds`` without
    updating the heartbeat. Keying on ``full_pass_interval_seconds`` (300s)
    would false-kill at the 900s threshold while the pass is still legit.
    """
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    # max_pass_runtime_seconds=1800, multiplier=3 → threshold=5400s.
    # last_beat 1800s ago (exactly one max-length pass) → healthy.
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=1800)),
        max_pass_runtime_seconds=1800,
        full_pass_interval_seconds=300,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, _heartbeat = wd._is_wedged()
    assert wedged is False


# ---------------------------------------------------------------------------
# PID-correlation unit tests (issue #728 rework: stale heartbeat from a
# prior supervisor pid must not be treated as a liveness signal for the
# watched child)
# ---------------------------------------------------------------------------


def test_is_wedged_false_when_heartbeat_from_prior_pid_within_grace(tmp_path: Path) -> None:
    """A stale heartbeat from a *different* pid is not this child's signal.

    Within the first-beat grace window the current child may not have
    written its own heartbeat yet, so the on-disk heartbeat (from a prior
    supervisor with a null ``exited_at`` — i.e. the prior one was killed)
    must not trigger a kill.
    """
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
        pid=11111,  # prior supervisor
    )
    process = FakeProcess(pid=22222)  # current child — different pid
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat = wd._is_wedged()
    assert wedged is False
    assert heartbeat is not None
    assert heartbeat["pid"] == 11111  # the stale heartbeat is returned for diagnostics


def test_is_wedged_false_when_heartbeat_from_prior_pid_clean_exit_within_grace(
    tmp_path: Path,
) -> None:
    """A prior supervisor's clean-exit heartbeat (``exited_at`` set) is also
    not a liveness signal for the current child.

    The ``exited_at`` check must not fire before the pid-correlation check:
    the prior supervisor exited cleanly, but that tells us nothing about the
    current child, which may not have written its own heartbeat yet.
    """
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
        exited_at=_iso(now),
        pid=11111,  # prior supervisor, clean exit
    )
    process = FakeProcess(pid=22222)  # current child — different pid
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, _heartbeat = wd._is_wedged()
    assert wedged is False


def test_is_wedged_true_when_heartbeat_from_prior_pid_after_grace(tmp_path: Path) -> None:
    """After the grace window, a child that never wrote its own heartbeat is wedged.

    The on-disk heartbeat is still from a prior pid, but the grace window
    has expired — the current child has been alive long enough without
    writing a matching heartbeat that it is wedged at startup.
    """
    start = datetime.now(UTC)
    late = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS + 1)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
        pid=11111,  # prior supervisor
    )
    process = FakeProcess(pid=22222)  # current child — different pid
    # First clock() call (in __init__) returns start; subsequent calls
    # return late so _elapsed_since_start exceeds the grace window.
    clock = _make_clock(start, late)
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=clock,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat = wd._is_wedged()
    assert wedged is True
    assert heartbeat is not None
    assert heartbeat["pid"] == 11111  # stale heartbeat returned for kill diagnostics


def test_is_wedged_true_when_prior_pid_heartbeat_clean_exit_after_grace(
    tmp_path: Path,
) -> None:
    """After grace, a prior pid's clean-exit heartbeat does not protect the child.

    The ``exited_at`` on the stale heartbeat is from the *prior* supervisor's
    clean exit. The pid-correlation check must fire *before* the ``exited_at``
    check: after the grace window, the current child (different pid) has been
    alive long enough without its own heartbeat, so it is wedged — regardless
    of what the prior supervisor's exit record says. The original pre-rework
    code checked ``exited_at`` first and returned False, falsely clearing the
    current child based on a different process's exit.
    """
    start = datetime.now(UTC)
    late = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS + 1)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
        exited_at=_iso(start),
        pid=11111,  # prior supervisor, clean exit
    )
    process = FakeProcess(pid=22222)  # current child — different pid
    clock = _make_clock(start, late)
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=clock,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat = wd._is_wedged()
    assert wedged is True
    assert heartbeat is not None
    assert heartbeat["pid"] == 11111


def test_is_wedged_true_when_no_heartbeat_after_grace(tmp_path: Path) -> None:
    """No heartbeat file at all after the grace window → wedged at startup.

    A child that crashes on startup is caught by ``process.poll()``; a child
    that wedges before writing its first heartbeat is not. The grace window
    bounds how long we wait before treating the absence as a wedge.
    """
    start = datetime.now(UTC)
    late = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS + 1)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    process = FakeProcess(pid=22222)
    clock = _make_clock(start, late)
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=clock,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat = wd._is_wedged()
    assert wedged is True
    assert heartbeat is None  # no file at all


def test_is_wedged_uses_matching_pid_heartbeat_for_staleness(tmp_path: Path) -> None:
    """When the heartbeat pid matches the child pid, normal staleness applies.

    This confirms the pid-correlation check does not short-circuit the
    existing staleness logic for a matching heartbeat.
    """
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=1000)),
        max_pass_runtime_seconds=300,
        pid=22222,  # matches the process
    )
    process = FakeProcess(pid=22222)
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat = wd._is_wedged()
    assert wedged is True
    assert heartbeat is not None
    assert heartbeat["pid"] == 22222


# ---------------------------------------------------------------------------
# Integration tests for WedgeWatchdog._run (the daemon-thread loop)
# ---------------------------------------------------------------------------


def test_watchdog_kills_wedged_child_and_logs_event(tmp_path: Path) -> None:
    """The full loop detects a stale heartbeat, kills the child, and records an event."""
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
    )
    process = FakeProcess(poll_results=[None])  # alive on first poll
    log_messages: list[str] = []
    event_calls: list[tuple[Any, ...]] = []

    def fake_log_event(state_path: Any, kind: str, payload: Any, **kwargs: Any) -> None:
        event_calls.append((state_path, kind, payload, kwargs))

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        poll_interval_seconds=0.01,
        clock=lambda: now,
        log=log_messages.append,
        sleep_func=lambda _: None,
        log_event_fn=fake_log_event,
    )
    thread = wd.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "watchdog thread did not terminate"

    assert process.killed is True
    assert wd.killed is True
    # The kill was logged loudly.
    assert any("wedge-watchdog" in m for m in log_messages)
    assert any("Terminating" in m for m in log_messages)
    # An event was recorded with the registered kind.
    assert len(event_calls) == 1
    _path, kind, payload, kwargs = event_calls[0]
    assert kind == WEDGE_KILL_EVENT_KIND
    assert kwargs.get("repo") == "fleet"
    assert payload["pid"] == 12345  # the killed process's pid
    assert payload["heartbeat_pid"] == 12345  # matched heartbeat's pid
    assert payload["stale_multiplier"] == WEDGE_KILL_STALE_MULTIPLIER


def test_watchdog_does_not_kill_healthy_child(tmp_path: Path) -> None:
    """A fresh heartbeat → the watchdog loops without killing, then exits when the child does."""
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=10)),
        max_pass_runtime_seconds=300,
    )
    # Alive for two polls, then the child exits on its own.
    process = FakeProcess(poll_results=[None, None, 0])
    log_messages: list[str] = []

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        poll_interval_seconds=0.01,
        clock=lambda: now,
        log=log_messages.append,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    thread = wd.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert process.killed is False
    assert wd.killed is False
    assert log_messages == []


def test_watchdog_does_not_kill_when_no_heartbeat(tmp_path: Path) -> None:
    """No heartbeat file → the watchdog skips and exits when the child does."""
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    process = FakeProcess(poll_results=[None, 0])

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        poll_interval_seconds=0.01,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    thread = wd.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert process.killed is False
    assert wd.killed is False


def test_watchdog_stops_immediately_if_child_already_exited(tmp_path: Path) -> None:
    """If ``process.poll()`` returns non-None on the first check, the watchdog exits."""
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
    )
    process = FakeProcess(poll_results=[0])  # already dead

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        poll_interval_seconds=0.01,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    thread = wd.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert process.killed is False
    assert wd.killed is False


# ---------------------------------------------------------------------------
# Integration tests for PID correlation and kill failure (issue #728 rework)
# ---------------------------------------------------------------------------


def test_watchdog_does_not_kill_when_heartbeat_from_prior_pid_within_grace(
    tmp_path: Path,
) -> None:
    """Full loop: a stale heartbeat from a prior pid does not kill a fresh child.

    The heartbeat on disk is from pid 11111 (prior supervisor, killed — null
    ``exited_at``), the watched child is pid 22222. Within the grace window
    the watchdog must not kill — the child hasn't written its own heartbeat
    yet, and the prior heartbeat is not its liveness signal.
    """
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
        pid=11111,
    )
    # Alive for two polls, then exits on its own.
    process = FakeProcess(poll_results=[None, None, 0], pid=22222)

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        poll_interval_seconds=0.01,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    thread = wd.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert process.killed is False
    assert wd.killed is False


def test_watchdog_does_not_kill_when_prior_pid_heartbeat_has_clean_exit(
    tmp_path: Path,
) -> None:
    """Full loop: a prior supervisor's clean-exit heartbeat does not kill.

    The ``exited_at`` on the stale heartbeat is from the *prior* supervisor's
    clean exit. The pid-correlation check must fire before the ``exited_at``
    check so the current child is not falsely cleared by a different
    process's exit record.
    """
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
        exited_at=_iso(now),
        pid=11111,
    )
    process = FakeProcess(poll_results=[None, None, 0], pid=22222)

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        poll_interval_seconds=0.01,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    thread = wd.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert process.killed is False
    assert wd.killed is False


def test_watchdog_kills_when_heartbeat_from_prior_pid_after_grace(tmp_path: Path) -> None:
    """Full loop: after the grace window, a child with no matching heartbeat is killed."""
    start = datetime.now(UTC)
    late = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS + 1)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
        pid=11111,
    )
    process = FakeProcess(poll_results=[None], pid=22222, block_until_killed=True)
    process._wait_return = 1
    event_calls: list[tuple[Any, ...]] = []

    def fake_log_event(state_path: Any, kind: str, payload: Any, **kwargs: Any) -> None:
        event_calls.append((state_path, kind, payload, kwargs))

    clock = _make_clock(start, late)

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        poll_interval_seconds=0.01,
        clock=clock,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=fake_log_event,
    )
    thread = wd.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert process.killed is True
    assert wd.killed is True
    # The kill event's ``pid`` is the process actually terminated
    # (``self._process.pid``), not the stale prior-pid heartbeat — the
    # heartbeat's pid is preserved separately as ``heartbeat_pid``.
    assert len(event_calls) == 1
    _path, kind, payload, _kwargs = event_calls[0]
    assert kind == WEDGE_KILL_EVENT_KIND
    assert payload["pid"] == 22222
    assert payload["heartbeat_pid"] == 11111


def test_watchdog_kills_when_no_heartbeat_at_all_after_grace(tmp_path: Path) -> None:
    """Full loop: a child that never writes a heartbeat is killed after grace.

    This is the exact path the round-2 review found was silently broken:
    ``_kill`` formatted ``age_seconds`` with ``:.0f`` even when it was
    ``None`` (no heartbeat file → no ``last_beat_at`` → ``age_seconds``
    stays ``None``), raising ``TypeError`` *before* ``process.kill()`` was
    reached. The watchdog's ``_run`` loop swallows the exception, so the
    wedged child was never killed and no event was recorded — the very
    capability (killing a child that wedges before its first heartbeat)
    this PR was built to add.

    Here the heartbeat path points at a file that is never written. The
    clock advances past ``WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS`` so
    ``_is_wedged`` returns ``(True, None)``, and ``_kill`` must format
    ``age=None`` as ``unknown`` (not crash), reach ``process.kill()``, set
    ``_killed``, and record the event with ``pid`` equal to the killed
    process's pid (``self._process.pid``) and ``heartbeat_pid=None``.
    """
    start = datetime.now(UTC)
    late = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS + 1)
    # No heartbeat file is written at all — the path simply does not exist.
    hb_path = tmp_path / "supervisor-heartbeat.json"
    process = FakeProcess(poll_results=[None], pid=22222, block_until_killed=True)
    process._wait_return = 1
    log_messages: list[str] = []
    event_calls: list[tuple[Any, ...]] = []

    def fake_log_event(state_path: Any, kind: str, payload: Any, **kwargs: Any) -> None:
        event_calls.append((state_path, kind, payload, kwargs))

    clock = _make_clock(start, late)

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        poll_interval_seconds=0.01,
        clock=clock,
        log=log_messages.append,
        sleep_func=lambda _: None,
        log_event_fn=fake_log_event,
    )
    thread = wd.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "watchdog thread did not terminate"

    # The child was actually killed — the TypeError used to abort _kill
    # before process.kill() was ever called.
    assert process.killed is True
    assert process.kill_count == 1
    assert wd.killed is True
    # The kill was logged loudly, with ``age=unknown`` (not a crash).
    assert any("wedge-watchdog" in m for m in log_messages)
    assert any("age=unknown" in m for m in log_messages)
    assert any("Terminating" in m for m in log_messages)
    # An event was recorded with the registered kind. ``pid`` is the
    # killed process's pid (always known), and ``heartbeat_pid`` is None
    # (no heartbeat file → no stated pid).
    assert len(event_calls) == 1
    _path, kind, payload, kwargs = event_calls[0]
    assert kind == WEDGE_KILL_EVENT_KIND
    assert kwargs.get("repo") == "fleet"
    assert payload["pid"] == 22222
    assert payload["heartbeat_pid"] is None
    assert payload["age_seconds"] is None
    assert payload["last_beat_at"] is None


def test_watchdog_kill_failure_does_not_set_killed_and_continues(tmp_path: Path) -> None:
    """When ``process.kill()`` raises, ``_killed`` stays False and the loop continues.

    The watchdog attempts to kill, ``kill()`` raises ``OSError``, the loop
    continues monitoring. On the next poll the child has exited on its own,
    so the watchdog returns without setting ``_killed``.

    The ``supervisor_wedged_killed`` event must NOT be recorded for a kill
    that did not happen — recording it before ``process.kill()`` succeeds
    would log a failed kill attempt as a completed kill, undermining the
    forensic-accuracy invariant the event exists to serve. The kill-failure
    is still diagnosed via ``logger.exception`` (not asserted here); the
    assertion is that no event was emitted.
    """
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
    )
    # Alive on first poll (triggers kill attempt), then exits on its own.
    process = FakeProcess(poll_results=[None, 0], kill_raises=True)
    event_calls: list[tuple[Any, ...]] = []

    def fake_log_event(state_path: Any, kind: str, payload: Any, **kwargs: Any) -> None:
        event_calls.append((state_path, kind, payload, kwargs))

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        poll_interval_seconds=0.01,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=fake_log_event,
    )
    thread = wd.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert process.kill_count == 0  # kill() raised, never succeeded
    assert wd.killed is False  # kill did not succeed
    # No kill event recorded — the kill did not happen, so it must not be
    # logged as a completed kill.
    assert event_calls == []


def test_watchdog_retries_kill_after_failure(tmp_path: Path) -> None:
    """After a failed kill, the watchdog retries and succeeds on the second attempt.

    The ``supervisor_wedged_killed`` event is recorded exactly once — on the
    successful retry. The failed first attempt must not emit an event (it did
    not kill anything), confirming the event is gated on ``process.kill()``
    succeeding rather than on the kill being attempted.
    """
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
    )
    # First kill raises, second succeeds. Three polls: alive, alive (retry), then
    # kill pushes a dead poll.
    process = FakeProcess(poll_results=[None, None], kill_raises=True)
    # Make the second kill succeed by clearing kill_raises after the first attempt.
    original_kill = process.kill

    def kill_then_succeed() -> None:
        if process._kill_raises:
            process._kill_raises = False
            raise OSError("simulated kill failure")
        original_kill()

    process.kill = kill_then_succeed  # type: ignore[method-assign]
    event_calls: list[tuple[Any, ...]] = []

    def fake_log_event(state_path: Any, kind: str, payload: Any, **kwargs: Any) -> None:
        event_calls.append((state_path, kind, payload, kwargs))

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        poll_interval_seconds=0.01,
        clock=lambda: now,
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=fake_log_event,
    )
    thread = wd.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert process.kill_count == 1  # second kill succeeded
    assert wd.killed is True
    # Exactly one event — on the successful retry, not the failed first attempt.
    assert len(event_calls) == 1
    _path, kind, payload, _kwargs = event_calls[0]
    assert kind == WEDGE_KILL_EVENT_KIND
    assert payload["pid"] == 12345  # the killed process's pid


# ---------------------------------------------------------------------------
# Wiring tests: run_fleet_supervise_loop / _spawn_supervise_child
# ---------------------------------------------------------------------------


def test_run_fleet_supervise_loop_default_watchdog_factory_is_on() -> None:
    """The default ``wedge_watchdog_factory`` creates a real WedgeWatchdog.

    Verifies the sentinel default resolves to ``_default_wedge_watchdog`` rather
    than ``None`` (disabled). Uses an injected ``spawn`` so no real process is
    started — the assertion is on the factory resolution, not the spawn path.
    """
    import inspect

    from charlie_work import fleet_dispatch

    sig = inspect.signature(run_fleet_supervise_loop)
    param = sig.parameters["wedge_watchdog_factory"]
    # The default is the sentinel, not None — meaning the watchdog is ON.
    assert param.default is fleet_dispatch._USE_DEFAULT_WATCHDOG


def test_spawn_supervise_child_starts_watchdog_from_factory(tmp_path: Path) -> None:
    """``_spawn_supervise_child`` calls the factory and starts the returned watchdog."""
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
    )

    fake_process = FakeProcess(poll_results=[None], block_until_killed=True)
    fake_process._wait_return = 1
    created_watchdogs: list[WedgeWatchdog] = []

    def factory(process: Any) -> WedgeWatchdog:
        wd = WedgeWatchdog(
            process,
            hb_path,
            poll_interval_seconds=0.01,
            clock=lambda: now,
            log=lambda _: None,
            sleep_func=lambda _: None,
            log_event_fn=lambda *a, **k: None,
        )
        created_watchdogs.append(wd)
        return wd

    with patch("charlie_work.fleet_dispatch.subprocess.Popen", return_value=fake_process):
        exit_code = _spawn_supervise_child((), wedge_watchdog_factory=factory)

    assert exit_code == 1
    assert len(created_watchdogs) == 1
    assert created_watchdogs[0].killed is True
    assert fake_process.killed is True


def test_spawn_supervise_child_skips_watchdog_when_factory_returns_none(
    tmp_path: Path,
) -> None:
    """A factory that returns ``None`` disables the watchdog (no kill)."""
    fake_process = FakeProcess(poll_results=[None, None, 0])
    fake_process._wait_return = 0

    with patch("charlie_work.fleet_dispatch.subprocess.Popen", return_value=fake_process):
        exit_code = _spawn_supervise_child((), wedge_watchdog_factory=lambda _: None)

    assert exit_code == 0
    assert fake_process.killed is False


def test_spawn_supervise_child_no_watchdog_when_factory_is_none(tmp_path: Path) -> None:
    """``wedge_watchdog_factory=None`` means no watchdog is created at all."""
    fake_process = FakeProcess()
    fake_process._wait_return = 0

    with patch("charlie_work.fleet_dispatch.subprocess.Popen", return_value=fake_process):
        _spawn_supervise_child((), wedge_watchdog_factory=None)

    assert fake_process.killed is False


def test_run_fleet_supervise_loop_with_disabled_watchdog_still_works() -> None:
    """Passing ``wedge_watchdog_factory=lambda _: None`` does not break the loop."""
    result = run_fleet_supervise_loop(
        spawn=lambda _n: 0,
        max_relaunches=3,
        wedge_watchdog_factory=lambda _: None,
    )
    assert result.ok is True
    assert result.data["launches"] == 1
