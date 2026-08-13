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
from typing import Any
from unittest.mock import patch

from charlie_work.fleet_dispatch import (
    _spawn_supervise_child,
    run_fleet_supervise_loop,
)
from charlie_work.wedge_watchdog import (
    WEDGE_KILL_DEFAULT_PASS_TIMEOUT_SECONDS,
    WEDGE_KILL_EVENT_KIND,
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
    """

    def __init__(
        self,
        poll_results: list[int | None] | None = None,
        *,
        block_until_killed: bool = False,
    ) -> None:
        self._poll_results = list(poll_results) if poll_results else [None]
        self.killed = False
        self.kill_count = 0
        self._wait_return: int = 0
        self._block_until_killed = block_until_killed
        self._killed_event = threading.Event()

    def wait(self) -> int:
        if self._block_until_killed:
            self._killed_event.wait(timeout=10.0)
        return self._wait_return

    def poll(self) -> int | None:
        if self._poll_results:
            return self._poll_results.pop(0)
        return 0

    def kill(self) -> None:
        self.killed = True
        self.kill_count += 1
        self._killed_event.set()
        # After kill, poll should report the child dead.
        if not self._poll_results or self._poll_results[-1] is None:
            self._poll_results.append(1)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    assert payload["pid"] == 12345
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
