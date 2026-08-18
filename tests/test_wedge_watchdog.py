"""Tests for the in-process wedge watchdog (issue #728).

Covers the ``WedgeWatchdog`` class that detects a wedged supervisor child by
heartbeat staleness and terminates it, plus the wiring through
``run_fleet_supervise_loop`` / ``_spawn_supervise_child``.

Identity is by spawn time, not pid (#1333): a heartbeat counts as the
watched child's own iff its parseable ``last_beat_at`` is at or after the
clock reading captured at ``WedgeWatchdog`` construction (floored to whole
seconds). The original implementation compared the heartbeat's recorded
``pid`` against ``Popen.pid``, which never matched under a uv/venv
trampoline (``Popen.pid`` is the launcher; the heartbeat writer stamps the
real interpreter's ``os.getpid()``) — every healthy supervisor was killed at
grace expiry. See ``WedgeWatchdog._is_wedged``'s docstring for the full
rationale.
"""

from __future__ import annotations

import json
import threading
import time
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
    WEDGE_REASON_BEAT_STALE,
    WEDGE_REASON_NO_FRESH_BEAT,
    WEDGE_REASON_NO_HEARTBEAT,
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

    ``pid`` defaults to ``12345``. It no longer drives any correlation
    decision (identity is by spawn-time timestamp, not pid — see module
    docstring), but is preserved as a plain field because the kill event's
    ``pid``/``heartbeat_pid`` payload fields are still populated from the
    process and heartbeat respectively, for forensic purposes.

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

    The first call (in ``WedgeWatchdog.__init__``, captured as
    ``_child_started_at``) returns the child's spawn-time reading; every
    subsequent call — in ``_elapsed_since_start``, in the staleness age
    calculation, in ``_kill``'s evidence rendering — returns ``late``. This
    lets a single heartbeat be simultaneously "at spawn" (own, per the
    ``last_beat_at >= _child_started_at`` test) and "old relative to the
    current check" (stale or grace-expired), which is required to exercise
    any branch beyond the degenerate zero-elapsed-time case.
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
    """A heartbeat older than ``multiplier * max_pass_runtime_seconds`` is wedged.

    The heartbeat must be the child's own (``last_beat_at`` at or after the
    construction-time clock reading, floored to the second) for the
    staleness branch to apply at all — an older beat is residue, handled by
    the grace-window branch instead (see the spawn-time identity tests
    below). The beat is written at spawn time (own) and checked 1000s later
    (stale relative to the 900s threshold).
    """
    start = datetime.now(UTC)
    late = start + timedelta(seconds=1000)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    # max_pass_runtime_seconds=300, multiplier=3 -> threshold=900s.
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start),
        max_pass_runtime_seconds=300,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=_make_clock(start, late),
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat, reason = wd._is_wedged()
    assert wedged is True
    assert heartbeat is not None
    assert heartbeat["pid"] == 12345
    assert reason == WEDGE_REASON_BEAT_STALE


def test_is_wedged_false_when_heartbeat_fresh(tmp_path: Path) -> None:
    """A heartbeat within the threshold is not wedged."""
    start = datetime.now(UTC)
    late = start + timedelta(seconds=60)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    # max_pass_runtime_seconds=300, multiplier=3 -> threshold=900s.
    # Beat written at spawn time (own); checked 60s later -> healthy.
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start),
        max_pass_runtime_seconds=300,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=_make_clock(start, late),
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, _heartbeat, reason = wd._is_wedged()
    assert wedged is False
    assert reason is None


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
    wedged, heartbeat, reason = wd._is_wedged()
    assert wedged is False
    assert heartbeat is None
    assert reason is None


def test_is_wedged_false_when_supervisor_recorded_clean_exit(tmp_path: Path) -> None:
    """A heartbeat with ``exited_at`` set is a self-exiting child — do not kill.

    The heartbeat must be the child's OWN (not residue) for ``exited_at`` to
    matter at all: a residue heartbeat's ``exited_at`` no longer suppresses
    anything (the residue check runs first — see the spawn-time identity
    tests below). So this beat is written at spawn time (own) and the clock
    jumps far enough forward that, without the clean-exit record, it would
    exceed the staleness threshold — proving ``exited_at`` is what
    suppresses the kill, not merely a low age.
    """
    start = datetime.now(UTC)
    late = start + timedelta(seconds=99999)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start),
        max_pass_runtime_seconds=300,
        exited_at=_iso(start),
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=_make_clock(start, late),
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, _heartbeat, reason = wd._is_wedged()
    assert wedged is False
    assert reason is None


def test_is_wedged_unparseable_last_beat_is_residue(tmp_path: Path) -> None:
    """An unparseable ``last_beat_at`` is treated as residue, not fail-open.

    Pre-#1333, an unparseable beat was unconditionally "never a kill signal"
    (fail-open). The rework folds it into the same residue path as a
    genuinely stale prior-supervisor heartbeat: within the grace window it
    does not kill (the child may not have written its own beat yet); after
    the grace window with no fresh beat, it DOES kill, with reason
    ``WEDGE_REASON_NO_FRESH_BEAT``. Both halves must hold.
    """
    start = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at="not-a-timestamp",
        max_pass_runtime_seconds=300,
    )

    # Within grace: not wedged.
    within_grace_late = start + timedelta(seconds=1)
    process_a = FakeProcess()
    wd_a = WedgeWatchdog(
        process_a,  # type: ignore[arg-type]
        hb_path,
        clock=_make_clock(start, within_grace_late),
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged_a, heartbeat_a, reason_a = wd_a._is_wedged()
    assert wedged_a is False
    assert heartbeat_a is not None
    assert reason_a is None

    # After grace: wedged, residue reason.
    after_grace_late = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS + 1)
    process_b = FakeProcess()
    wd_b = WedgeWatchdog(
        process_b,  # type: ignore[arg-type]
        hb_path,
        clock=_make_clock(start, after_grace_late),
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged_b, heartbeat_b, reason_b = wd_b._is_wedged()
    assert wedged_b is True
    assert heartbeat_b is not None
    assert reason_b == WEDGE_REASON_NO_FRESH_BEAT


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
    start = datetime.now(UTC)
    late = start + timedelta(seconds=1800)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    # max_pass_runtime_seconds=1800, multiplier=3 -> threshold=5400s.
    # Beat written at spawn time (own); checked 1800s later (exactly one
    # max-length pass) -> healthy.
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start),
        max_pass_runtime_seconds=1800,
        full_pass_interval_seconds=300,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=_make_clock(start, late),
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, _heartbeat, reason = wd._is_wedged()
    assert wedged is False
    assert reason is None


# ---------------------------------------------------------------------------
# Spawn-time identity unit tests (#1333 rework: a heartbeat correlates to the
# watched child by comparing ``last_beat_at`` against the clock reading
# captured at construction, never by pid — see ``_is_wedged``'s docstring for
# why pid correlation was structurally broken under a uv/venv trampoline).
# ---------------------------------------------------------------------------


def test_is_wedged_false_when_residue_heartbeat_within_grace(tmp_path: Path) -> None:
    """An on-disk heartbeat that predates this child's spawn is not its signal.

    Within the first-beat grace window the current child may not have
    written its own heartbeat yet, so residue left by a prior supervisor
    (crashed — null ``exited_at``) must not trigger a kill. The check clock
    is set to one second before grace expiry (not the construction instant)
    so this genuinely probes the boundary rather than only ``elapsed == 0``.
    """
    start = datetime.now(UTC)
    near_grace_end = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS - 1)
    residue_beat = _iso(start - timedelta(seconds=9999))
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=residue_beat,
        max_pass_runtime_seconds=300,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=_make_clock(start, near_grace_end),
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat, reason = wd._is_wedged()
    assert wedged is False
    # the residue heartbeat itself (not some other dict) is returned for
    # diagnostics
    assert heartbeat is not None
    assert heartbeat["last_beat_at"] == residue_beat
    assert reason is None


def test_is_wedged_false_when_residue_heartbeat_has_clean_exit_within_grace(
    tmp_path: Path,
) -> None:
    """A residue heartbeat's ``exited_at`` is irrelevant within the grace window too.

    The residue check runs unconditionally before ``exited_at`` is ever
    inspected (#1333), so a prior supervisor's clean-exit record neither
    helps nor hurts here — the grace window alone decides. The original
    pre-rework code checked ``exited_at`` first, which (after grace expiry —
    see the after-grace regression test below) falsely cleared the current
    child based on a different process's exit. The check clock is set to one
    second before grace expiry so this probes the boundary, not just
    ``elapsed == 0``.
    """
    start = datetime.now(UTC)
    near_grace_end = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS - 1)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
        exited_at=_iso(start),
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=_make_clock(start, near_grace_end),
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, _heartbeat, reason = wd._is_wedged()
    assert wedged is False
    assert reason is None


def test_is_wedged_true_when_residue_heartbeat_after_grace(tmp_path: Path) -> None:
    """After the grace window, residue with no fresh beat means wedged.

    The on-disk heartbeat is still older than this child's spawn time, but
    the grace window has expired — the current child has been alive long
    enough without writing a matching heartbeat that it is treated as wedged
    at startup.
    """
    start = datetime.now(UTC)
    late = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS + 1)
    residue_beat = _iso(start - timedelta(seconds=9999))
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=residue_beat,
        max_pass_runtime_seconds=300,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=_make_clock(start, late),
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat, reason = wd._is_wedged()
    assert wedged is True
    # stale heartbeat returned for kill diagnostics — confirm it's the
    # residue record, not some other dict
    assert heartbeat is not None
    assert heartbeat["last_beat_at"] == residue_beat
    assert reason == WEDGE_REASON_NO_FRESH_BEAT


def test_residue_heartbeat_with_exited_at_after_grace_is_wedged_not_suppressed(
    tmp_path: Path,
) -> None:
    """Regression (#1333): a RESIDUE heartbeat's clean exit must not suppress the grace kill.

    The residue check (comparing ``last_beat_at`` against the spawn-time
    floor) runs before ``exited_at`` is ever inspected. So even though the
    on-disk heartbeat records a clean exit, that exit belongs to a *prior*
    supervisor — it says nothing about the current child, which has been
    alive past the grace window with no beat of its own. The original
    pre-#1333 pid-based code checked ``exited_at`` first and returned False,
    falsely clearing the current child based on a different process's exit.
    """
    start = datetime.now(UTC)
    late = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS + 1)
    residue_beat = _iso(start - timedelta(seconds=9999))
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=residue_beat,
        max_pass_runtime_seconds=300,
        exited_at=_iso(start),
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=_make_clock(start, late),
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat, reason = wd._is_wedged()
    assert wedged is True
    # confirm the RESIDUE record (with its clean exited_at) is what came
    # back, not a different heartbeat
    assert heartbeat is not None
    assert heartbeat["last_beat_at"] == residue_beat
    assert reason == WEDGE_REASON_NO_FRESH_BEAT


def test_is_wedged_true_when_no_heartbeat_after_grace(tmp_path: Path) -> None:
    """No heartbeat file at all after the grace window → wedged at startup.

    A child that crashes on startup is caught by ``process.poll()``; a child
    that wedges before writing its first heartbeat is not. The grace window
    bounds how long we wait before treating the absence as a wedge.
    """
    start = datetime.now(UTC)
    late = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS + 1)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=_make_clock(start, late),
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat, reason = wd._is_wedged()
    assert wedged is True
    assert heartbeat is None  # no file at all
    assert reason == WEDGE_REASON_NO_HEARTBEAT


def test_is_wedged_same_second_beat_counts_as_own_not_residue(tmp_path: Path) -> None:
    """A beat at the same whole second as spawn counts as the child's own.

    ``last_beat_at`` has whole-second resolution, so the spawn anchor is
    floored to the second (see ``_is_wedged``'s docstring on
    ``spawn_anchor``). Construction happens at ``12:00:00.400``; the beat is
    stamped ``12:00:00Z`` — earlier in wall-clock terms, but the SAME
    floored second, so it must count as this child's own. The clock then
    advances past the grace window but stays well under the staleness
    threshold, which is what discriminates the two classifications: if this
    beat were misclassified as residue, ``_elapsed_since_start`` (~350s)
    would exceed the 300s grace window and report ``wedged=True``; correctly
    classified as own, only the (much looser) 900s staleness threshold
    applies, and 350s does not exceed it.
    """
    start = datetime(2026, 1, 1, 12, 0, 0, 400000, tzinfo=UTC)
    late = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS + 50)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start),  # floors to 12:00:00Z — same second as spawn
        max_pass_runtime_seconds=300,
    )
    process = FakeProcess()
    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=_make_clock(start, late),
        log=lambda _: None,
        sleep_func=lambda _: None,
        log_event_fn=lambda *a, **k: None,
    )
    wedged, heartbeat, reason = wd._is_wedged()
    assert wedged is False
    assert heartbeat is not None
    assert reason is None


# ---------------------------------------------------------------------------
# Kill-message branch rendering (#1333: the message and event must name the
# branch that actually fired, not always the age-vs-threshold text — the
# original implementation's self-contradictory "age=12s exceeds
# threshold=5400s" on a grace-window kill actively misdirected diagnosis).
# ---------------------------------------------------------------------------


def test_kill_message_for_no_heartbeat_reason_names_grace_not_threshold(
    tmp_path: Path,
) -> None:
    """A no-heartbeat kill's message says so, and does not claim a threshold was exceeded."""
    hb_path = tmp_path / "supervisor-heartbeat.json"
    process = FakeProcess(pid=22222)
    log_messages: list[str] = []
    event_calls: list[tuple[Any, ...]] = []

    def fake_log_event(state_path: Any, kind: str, payload: Any, **kwargs: Any) -> None:
        event_calls.append((state_path, kind, payload, kwargs))

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        log=log_messages.append,
        sleep_func=lambda _: None,
        log_event_fn=fake_log_event,
    )
    result = wd._kill(None, WEDGE_REASON_NO_HEARTBEAT)

    assert result is True
    assert len(log_messages) == 1
    message = log_messages[0]
    assert "no heartbeat file appeared" in message
    assert f"wedged [{WEDGE_REASON_NO_HEARTBEAT}]" in message
    assert "exceeds threshold" not in message
    assert len(event_calls) == 1
    _path, _kind, payload, _kwargs = event_calls[0]
    assert payload["reason"] == WEDGE_REASON_NO_HEARTBEAT


def test_kill_message_for_no_fresh_beat_reason_names_residue_not_threshold(
    tmp_path: Path,
) -> None:
    """A residue (grace-expired) kill's message names the predates-spawn evidence."""
    hb_path = tmp_path / "supervisor-heartbeat.json"
    process = FakeProcess(pid=22222)
    log_messages: list[str] = []
    event_calls: list[tuple[Any, ...]] = []

    def fake_log_event(state_path: Any, kind: str, payload: Any, **kwargs: Any) -> None:
        event_calls.append((state_path, kind, payload, kwargs))

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        log=log_messages.append,
        sleep_func=lambda _: None,
        log_event_fn=fake_log_event,
    )
    heartbeat = {"pid": 11111, "last_beat_at": "2020-01-01T00:00:00Z"}
    result = wd._kill(heartbeat, WEDGE_REASON_NO_FRESH_BEAT)

    assert result is True
    assert len(log_messages) == 1
    message = log_messages[0]
    assert "predates this child" in message
    assert "no fresh beat" in message
    assert f"wedged [{WEDGE_REASON_NO_FRESH_BEAT}]" in message
    assert "exceeds threshold" not in message
    assert len(event_calls) == 1
    _path, _kind, payload, _kwargs = event_calls[0]
    assert payload["reason"] == WEDGE_REASON_NO_FRESH_BEAT


def test_kill_message_for_beat_stale_reason_names_threshold(tmp_path: Path) -> None:
    """A staleness kill's message reports age exceeding the derived threshold."""
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    process = FakeProcess(pid=22222)
    log_messages: list[str] = []
    event_calls: list[tuple[Any, ...]] = []

    def fake_log_event(state_path: Any, kind: str, payload: Any, **kwargs: Any) -> None:
        event_calls.append((state_path, kind, payload, kwargs))

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        clock=lambda: now,
        log=log_messages.append,
        sleep_func=lambda _: None,
        log_event_fn=fake_log_event,
    )
    heartbeat = {
        "pid": 22222,
        "last_beat_at": _iso(now - timedelta(seconds=1000)),
        "max_pass_runtime_seconds": 300,
    }
    result = wd._kill(heartbeat, WEDGE_REASON_BEAT_STALE)

    assert result is True
    assert len(log_messages) == 1
    message = log_messages[0]
    assert "exceeds threshold" in message
    assert f"wedged [{WEDGE_REASON_BEAT_STALE}]" in message
    assert len(event_calls) == 1
    _path, _kind, payload, _kwargs = event_calls[0]
    assert payload["reason"] == WEDGE_REASON_BEAT_STALE


# ---------------------------------------------------------------------------
# Integration tests for WedgeWatchdog._run (the daemon-thread loop)
# ---------------------------------------------------------------------------


def test_watchdog_kills_wedged_child_and_logs_event(tmp_path: Path) -> None:
    """The full loop detects a stale heartbeat, kills the child, and records an event."""
    start = datetime.now(UTC)
    late = start + timedelta(seconds=1000)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    # Beat written at spawn time (own); by the time the watchdog checks it
    # (1000s later) it exceeds the 900s threshold (300 * multiplier 3).
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start),
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
        clock=_make_clock(start, late),
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
    assert payload["heartbeat_pid"] == 12345  # this child's own heartbeat pid
    assert payload["stale_multiplier"] == WEDGE_KILL_STALE_MULTIPLIER
    assert payload["reason"] == WEDGE_REASON_BEAT_STALE


def test_watchdog_does_not_kill_healthy_child(tmp_path: Path) -> None:
    """A fresh, own heartbeat → the watchdog loops without killing, then exits when the child does."""
    start = datetime.now(UTC)
    late = start + timedelta(seconds=10)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start),
        max_pass_runtime_seconds=300,
    )
    # Alive for two polls, then the child exits on its own.
    process = FakeProcess(poll_results=[None, None, 0])
    log_messages: list[str] = []

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        poll_interval_seconds=0.01,
        clock=_make_clock(start, late),
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


def test_fresh_heartbeat_with_foreign_pid_is_not_residue_trampoline_regression(
    tmp_path: Path,
) -> None:
    """Regression for #1333: pid mismatch alone must never drive the verdict.

    Reproduces the production shape that caused the incident: a uv/venv
    trampoline means ``Popen.pid`` (the launcher) never equals the
    heartbeat's stamped ``os.getpid()`` (the real interpreter) — so any pid
    check here would treat a healthy child as wedged forever. The fix
    correlates by spawn-time timestamp instead: this heartbeat's pid is
    wildly different from the watched process's pid, but its
    ``last_beat_at`` stays fresh relative to the current check time, and
    that alone must keep the child alive — even long after the first-beat
    grace window has expired, when a pid-based (or grace-based) check would
    have killed it. ``kill`` must never be called.
    """
    start = datetime.now(UTC)
    # Ten grace windows past spawn — a pid check would have killed this
    # child long ago.
    late = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS * 10)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        # Refreshed just before the check — the real interpreter beating
        # normally, unrelated to the launcher pid on the process below.
        last_beat_at=_iso(late - timedelta(seconds=10)),
        max_pass_runtime_seconds=300,
        pid=999999,  # heartbeat's real-interpreter pid
    )
    process = FakeProcess(poll_results=[None, None, 0], pid=555)  # launcher pid
    log_messages: list[str] = []

    wd = WedgeWatchdog(
        process,  # type: ignore[arg-type]
        hb_path,
        poll_interval_seconds=0.01,
        clock=_make_clock(start, late),
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
# Integration tests for spawn-time identity and kill failure (#1333 rework)
# ---------------------------------------------------------------------------


def test_watchdog_does_not_kill_when_residue_heartbeat_within_grace(
    tmp_path: Path,
) -> None:
    """Full loop: residue from a prior supervisor does not kill a fresh child within grace.

    The heartbeat on disk predates this child's spawn (prior supervisor,
    crashed — null ``exited_at``). Within the grace window the watchdog
    must not kill — the child hasn't written its own heartbeat yet, and the
    prior heartbeat is not its liveness signal.
    """
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
    )
    # Alive for two polls, then exits on its own.
    process = FakeProcess(poll_results=[None, None, 0])

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


def test_watchdog_does_not_kill_when_residue_heartbeat_has_clean_exit(
    tmp_path: Path,
) -> None:
    """Full loop: a prior supervisor's clean-exit heartbeat does not kill within grace.

    The residue check runs unconditionally before ``exited_at`` is ever
    inspected, so a prior supervisor's clean-exit record doesn't change the
    outcome here — the grace window alone decides.
    """
    now = datetime.now(UTC)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(now - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
        exited_at=_iso(now),
    )
    process = FakeProcess(poll_results=[None, None, 0])

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


def test_watchdog_kills_when_residue_heartbeat_after_grace(tmp_path: Path) -> None:
    """Full loop: after the grace window, a child with no matching heartbeat is killed."""
    start = datetime.now(UTC)
    late = start + timedelta(seconds=WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS + 1)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start - timedelta(seconds=9999)),
        max_pass_runtime_seconds=300,
        pid=11111,  # residue from a prior supervisor
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
    # (``self._process.pid``); the heartbeat's stated pid — irrelevant to
    # the verdict now that correlation is by timestamp, not pid — is
    # preserved separately as ``heartbeat_pid`` for forensic purposes.
    assert len(event_calls) == 1
    _path, kind, payload, _kwargs = event_calls[0]
    assert kind == WEDGE_KILL_EVENT_KIND
    assert payload["pid"] == 22222
    assert payload["heartbeat_pid"] == 11111
    assert payload["reason"] == WEDGE_REASON_NO_FRESH_BEAT


def test_watchdog_kills_when_no_heartbeat_at_all_after_grace(tmp_path: Path) -> None:
    """Full loop: a child that never writes a heartbeat is killed after grace.

    Historical context: an earlier revision of ``_kill`` formatted
    ``age_seconds`` with ``:.0f`` even when it was ``None`` (no heartbeat
    file → no ``last_beat_at`` → ``age_seconds`` stays ``None``), raising
    ``TypeError`` *before* ``process.kill()`` was reached — silently
    defeating the exact no-heartbeat wedge case this watchdog exists to
    kill. That crash is long fixed (``age_display`` renders ``unknown``);
    the #1333 rework additionally changed what the kill message SAYS for
    this reason — it no longer renders the age-vs-threshold text at all
    (misleading for a no-heartbeat kill, since no age exists), instead
    naming the grace-window branch that actually fired.

    Here the heartbeat path points at a file that is never written. The
    clock advances past ``WEDGE_KILL_FIRST_BEAT_GRACE_SECONDS`` so
    ``_is_wedged`` returns ``(True, None, WEDGE_REASON_NO_HEARTBEAT)``, and
    ``_kill`` must reach ``process.kill()``, set ``_killed``, and record the
    event with ``pid`` equal to the killed process's pid
    (``self._process.pid``), ``heartbeat_pid=None``, and
    ``reason=WEDGE_REASON_NO_HEARTBEAT``.
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
    # The kill was logged loudly, naming the branch that fired — not the
    # age-vs-threshold text (which does not apply when there is no beat at
    # all to measure the age of).
    assert any("wedge-watchdog" in m for m in log_messages)
    assert any("no heartbeat file appeared" in m for m in log_messages)
    assert not any("exceeds threshold" in m for m in log_messages)
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
    assert payload["reason"] == WEDGE_REASON_NO_HEARTBEAT


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
    start = datetime.now(UTC)
    late = start + timedelta(seconds=1000)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    # Beat written at spawn time (own) and stale by the time it's checked —
    # must be wedged for _kill to even be attempted.
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start),
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
        clock=_make_clock(start, late),
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
    start = datetime.now(UTC)
    late = start + timedelta(seconds=1000)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start),
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
        clock=_make_clock(start, late),
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
    assert payload["reason"] == WEDGE_REASON_BEAT_STALE


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
    start = datetime.now(UTC)
    late = start + timedelta(seconds=1000)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start),
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
            clock=_make_clock(start, late),
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


def test_spawn_supervise_child_joins_watchdog_thread_for_delayed_event_write(
    tmp_path: Path,
) -> None:
    """Regression for #1333: the wrapper waits for a slow event write.

    During the #1333 incident the wrapper returned (and the process exited)
    while the watchdog's daemon thread was still mid-write on its
    ``supervisor_wedged_killed`` event, tearing it down — 144 kills, 0
    recorded events. The fix keeps the watchdog thread handle and, when
    ``watchdog.killed`` is True after ``process.wait()`` returns, joins it
    with a bounded timeout so the event write completes first.

    Here ``log_event_fn`` sleeps briefly (simulating a slow write) before
    recording the call — a REAL delay on a REAL daemon thread, not a fake
    one, so this exercises the actual join wait rather than asserting a
    mock was configured correctly. ``process.wait()`` unblocks as soon as
    ``process.kill()`` is called (well before the delayed write even
    starts), so without the join this event would very likely still be
    missing by the time ``_spawn_supervise_child`` returns.
    """
    start = datetime.now(UTC)
    late = start + timedelta(seconds=1000)
    hb_path = tmp_path / "supervisor-heartbeat.json"
    _write_heartbeat(
        hb_path,
        last_beat_at=_iso(start),
        max_pass_runtime_seconds=300,
    )

    fake_process = FakeProcess(poll_results=[None], block_until_killed=True)
    fake_process._wait_return = 1
    event_calls: list[tuple[Any, ...]] = []

    def slow_log_event(state_path: Any, kind: str, payload: Any, **kwargs: Any) -> None:
        time.sleep(0.2)
        event_calls.append((state_path, kind, payload, kwargs))

    def factory(process: Any) -> WedgeWatchdog:
        return WedgeWatchdog(
            process,
            hb_path,
            poll_interval_seconds=0.01,
            clock=_make_clock(start, late),
            log=lambda _: None,
            sleep_func=lambda _: None,
            log_event_fn=slow_log_event,
        )

    with patch("charlie_work.fleet_dispatch.subprocess.Popen", return_value=fake_process):
        exit_code = _spawn_supervise_child((), wedge_watchdog_factory=factory)

    assert exit_code == 1
    assert len(event_calls) == 1
    _path, kind, payload, _kwargs = event_calls[0]
    assert kind == WEDGE_KILL_EVENT_KIND
    assert payload["reason"] == WEDGE_REASON_BEAT_STALE


def test_spawn_supervise_child_joins_watchdog_thread_with_10s_timeout_when_killed(
    tmp_path: Path,
) -> None:
    """The join uses a bounded 10s timeout, and only fires when the watchdog killed.

    A stub watchdog isolates the join call itself from the real detection
    logic (covered by the test above and by the ``WedgeWatchdog`` unit
    tests): this asserts ``_spawn_supervise_child`` calls
    ``thread.join(timeout=10.0)`` exactly when ``watchdog.killed`` is True.
    """

    class _StubThread:
        def __init__(self) -> None:
            self.join_calls: list[float | None] = []

        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(timeout)

    class _StubWatchdog:
        def __init__(self, killed: bool) -> None:
            self._killed = killed
            self.thread = _StubThread()

        def start(self) -> _StubThread:
            return self.thread

        @property
        def killed(self) -> bool:
            return self._killed

    fake_process = FakeProcess(poll_results=[None])
    fake_process._wait_return = 1
    stub = _StubWatchdog(killed=True)

    with patch("charlie_work.fleet_dispatch.subprocess.Popen", return_value=fake_process):
        exit_code = _spawn_supervise_child((), wedge_watchdog_factory=lambda _p: stub)  # type: ignore[arg-type]

    assert exit_code == 1
    assert stub.thread.join_calls == [10.0]
