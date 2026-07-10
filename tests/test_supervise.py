"""Tests for charlie_work.supervise — supervised infill loop.

FakeApp: minimal stand-in for OrchestratorApp — only the attributes that
run_supervised touches.  Kept local to this test file.

Injected sleep/clock: record sleep args; monotonically advancing fake clock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.config import OrchestratorConfig, SupervisorConfig
from charlie_work.supervise import (
    has_delta,
    run_supervised,
    should_exit,
    take_snapshot,
    try_acquire_supervisor_lock,
)
from charlie_work.workflow import CommandResult


# ---------------------------------------------------------------------------
# FakeApp
# ---------------------------------------------------------------------------


class _FakePaths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.prs = root / "prs"


class FakeApp:
    """Minimal OrchestratorApp stand-in for run_supervised tests.

    ``results`` is a list of CommandResult objects returned by successive
    ``loop()`` calls (cycles).
    """

    def __init__(
        self,
        tmp_path: Path,
        results: list[CommandResult],
        *,
        supervisor_cfg: SupervisorConfig | None = None,
    ) -> None:
        cfg_supervisor = supervisor_cfg if supervisor_cfg is not None else SupervisorConfig()
        self.config = OrchestratorConfig(supervisor=cfg_supervisor)
        self.paths = _FakePaths(tmp_path / ".var" / "charlie-work")
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self._sessions_dir = tmp_path / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._results = list(results)
        self._call_count = 0

    def _resolve(self, path_str: str) -> Path:
        """Resolve a config path string — returns sessions_dir for any input."""
        return self._sessions_dir

    def loop(self, limit: Any = None, *, merge: Any = None) -> CommandResult:
        if self._call_count < len(self._results):
            result = self._results[self._call_count]
        else:
            # Default drained result when list exhausted
            result = _drained_result()
        self._call_count += 1
        return result


def _drained_result() -> CommandResult:
    """A fully drained pass result (nothing dispatched/merged, no open PRs)."""
    return CommandResult(
        True,
        "loop complete",
        {
            "dispatch": {"selected_count": 0},
            "dispatch_rework": {"selected_count": 0},
            "merges": [],
            "reviews": [],
            "errors": [],
            "open_tracked_prs": 0,
            "skipped_reviews": 0,
        },
    )


def _active_result(
    *,
    dispatched: int = 0,
    rework: int = 0,
    merged: int = 0,
    open_prs: int = 0,
) -> CommandResult:
    """A pass result with some activity."""
    return CommandResult(
        True,
        "loop complete",
        {
            "dispatch": {"selected_count": dispatched},
            "dispatch_rework": {"selected_count": rework},
            "merges": [{"pr": i} for i in range(merged)],
            "reviews": [],
            "errors": [],
            "open_tracked_prs": open_prs,
            "skipped_reviews": 0,
        },
    )


# ---------------------------------------------------------------------------
# Fake sleep + clock infrastructure
# ---------------------------------------------------------------------------


class FakeClock:
    """Monotonically advancing fake clock.

    Advances by ``auto_advance`` on each ``sleep()`` call.
    """

    def __init__(self, start: float = 0.0, auto_advance: float = 0.0) -> None:
        self._now = start
        self._auto_advance = auto_advance
        self.sleep_calls: list[float] = []

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self._now += self._auto_advance if self._auto_advance else seconds


# ---------------------------------------------------------------------------
# should_exit tests
# ---------------------------------------------------------------------------


def test_should_exit_drained_returns_true() -> None:
    result = _drained_result()
    assert should_exit(result, live_count=0) is True


def test_should_exit_live_workers_returns_false() -> None:
    result = _drained_result()
    assert should_exit(result, live_count=2) is False


def test_should_exit_open_prs_returns_false() -> None:
    result = _active_result(open_prs=1)
    assert should_exit(result, live_count=0) is False


def test_should_exit_dispatched_returns_false() -> None:
    result = _active_result(dispatched=1)
    assert should_exit(result, live_count=0) is False


def test_should_exit_rework_dispatched_returns_false() -> None:
    result = _active_result(rework=1)
    assert should_exit(result, live_count=0) is False


def test_should_exit_merged_returns_false() -> None:
    result = _active_result(merged=1)
    assert should_exit(result, live_count=0) is False


# ---------------------------------------------------------------------------
# has_delta / take_snapshot tests
# ---------------------------------------------------------------------------


def test_has_delta_no_change_returns_false(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()
    snap1 = take_snapshot(sessions, prs)
    snap2 = take_snapshot(sessions, prs)
    assert has_delta(snap1, snap2) is False


def test_has_delta_live_count_change_returns_true(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()
    snap1 = take_snapshot(sessions, prs)
    (sessions / "issue-1.json").write_text("{}", encoding="utf-8")
    snap2 = take_snapshot(sessions, prs)
    assert has_delta(snap1, snap2) is True


def test_has_delta_sidecar_mtime_change_returns_true(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()
    sidecar = sessions / "issue-1.json"
    sidecar.write_text("{}", encoding="utf-8")
    snap1 = take_snapshot(sessions, prs)
    # Touch with a different mtime
    import os

    os.utime(sidecar, (sidecar.stat().st_atime + 1, sidecar.stat().st_mtime + 1))
    snap2 = take_snapshot(sessions, prs)
    assert has_delta(snap1, snap2) is True


def test_has_delta_new_verdict_file_returns_true(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()
    snap1 = take_snapshot(sessions, prs)
    pr_dir = prs / "pr-1"
    pr_dir.mkdir()
    (pr_dir / "review-decision.json").write_text('{"decision": "approved"}', encoding="utf-8")
    snap2 = take_snapshot(sessions, prs)
    assert has_delta(snap1, snap2) is True


# ---------------------------------------------------------------------------
# run_supervised tests
# ---------------------------------------------------------------------------


def test_run_supervised_exits_when_drained_first_pass(tmp_path: Path) -> None:
    """First pass drains everything → loop exits immediately."""
    app = FakeApp(tmp_path, [_drained_result()])
    fc = FakeClock()
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=5,
    )
    assert result.ok is True
    assert app._call_count == 1


def test_run_supervised_infill_freed_slot_triggers_prompt_pass(tmp_path: Path) -> None:
    """A sidecar disappearing (worker exited) triggers a delta → prompt pass which dispatches."""
    # Plant a sidecar that will vanish between first poll and first pass
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sidecar = sessions_dir / "issue-1.json"
    sidecar.write_text("{}", encoding="utf-8")

    # Pass 1: still-live (open_prs=1 keeps loop alive), Pass 2: dispatches 1, Pass 3: drained
    results = [
        _active_result(open_prs=1),
        _active_result(dispatched=1),
        _drained_result(),
    ]
    app = FakeApp(tmp_path, results)
    app._sessions_dir = sessions_dir

    fc = FakeClock(auto_advance=1.0)

    def sleeping(seconds: float) -> None:
        fc.sleep(seconds)
        # After first sleep, remove the sidecar to simulate worker exit
        if len(fc.sleep_calls) == 1 and sidecar.exists():
            sidecar.unlink()

    result = run_supervised(
        app,
        clock=fc.now,
        sleep=sleeping,
        max_passes=5,
    )
    assert result.ok is True
    assert app._call_count >= 2


def test_run_supervised_verdict_file_triggers_pass_while_live_zero(tmp_path: Path) -> None:
    """A verdict file appearing while live=0 keeps the loop alive and triggers a merge pass."""
    prs_dir = tmp_path / ".var" / "charlie-work" / "prs"
    prs_dir.mkdir(parents=True, exist_ok=True)
    pr_dir = prs_dir / "pr-456"
    pr_dir.mkdir()

    # Pass 1: no workers, no dispatched, but open_prs=1 → stay alive
    # Pass 2: merges=1 after verdict written, open_prs=0 → drained
    results = [
        _active_result(open_prs=1),
        _active_result(merged=1),
        _drained_result(),
    ]
    app = FakeApp(tmp_path, results)

    fc = FakeClock(auto_advance=1.0)

    def sleeping(seconds: float) -> None:
        fc.sleep(seconds)
        # Write verdict after first sleep
        if len(fc.sleep_calls) == 1:
            verdict = pr_dir / "review-decision.json"
            verdict.write_text('{"decision": "approved"}', encoding="utf-8")

    result = run_supervised(
        app,
        clock=fc.now,
        sleep=sleeping,
        max_passes=5,
    )
    assert result.ok is True
    assert app._call_count >= 2


def test_run_supervised_fallback_timer_fires_with_no_delta(tmp_path: Path) -> None:
    """If no delta fires, the fallback timer still triggers a pass."""
    app = FakeApp(
        tmp_path,
        [_drained_result()],
        supervisor_cfg=SupervisorConfig(
            full_pass_interval_seconds=10,
            poll_interval_seconds=5,
        ),
    )
    # Advance clock to force fallback on first iteration
    fc = FakeClock(start=0.0, auto_advance=0.0)
    # last_full_pass_at will be -10, so first iteration fallback_due = True
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=5,
    )
    assert result.ok is True
    assert app._call_count == 1


def test_run_supervised_active_cooldown_sleep_after_dispatch(tmp_path: Path) -> None:
    """After a pass that dispatches, sleep time equals active_cooldown_seconds."""
    cfg = SupervisorConfig(
        poll_interval_seconds=20,
        active_cooldown_seconds=7,
        full_pass_interval_seconds=300,
    )
    # Pass 1: dispatches 1 (stays alive), Pass 2: drained
    results = [_active_result(dispatched=1), _drained_result()]
    app = FakeApp(tmp_path, results, supervisor_cfg=cfg)

    fc = FakeClock(auto_advance=1.0)
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=5,
    )
    assert result.ok is True
    # First sleep after dispatching pass should be active_cooldown_seconds (7)
    assert fc.sleep_calls[0] == 7.0


def test_run_supervised_poll_interval_sleep_when_idle(tmp_path: Path) -> None:
    """After a pass with no dispatch/merge, sleep time equals poll_interval_seconds."""
    cfg = SupervisorConfig(
        poll_interval_seconds=15,
        active_cooldown_seconds=7,
        full_pass_interval_seconds=300,
    )
    # Pass 1: open_prs=1 (stay alive), Pass 2: drained
    results = [_active_result(open_prs=1), _drained_result()]
    app = FakeApp(tmp_path, results, supervisor_cfg=cfg)

    fc = FakeClock(auto_advance=1.0)
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=5,
    )
    assert result.ok is True
    # First sleep after idle pass (open_prs=1 but no dispatch/merge)
    assert fc.sleep_calls[0] == 15.0


def test_run_supervised_max_passes_exits(tmp_path: Path) -> None:
    """max_passes cap causes exit before draining."""
    # Supply 10 non-draining results
    results = [_active_result(open_prs=1)] * 10
    app = FakeApp(tmp_path, results)
    fc = FakeClock(auto_advance=1.0)
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=3,
    )
    assert result.ok is True
    assert app._call_count == 3


def test_run_supervised_max_runtime_exits(tmp_path: Path) -> None:
    """max_runtime_override cap stops the loop after the wall-clock expires."""
    results = [_active_result(open_prs=1)] * 100
    app = FakeApp(tmp_path, results)

    # Clock advances 70 seconds per sleep call (= >1 minute)
    fc = FakeClock(start=0.0, auto_advance=70.0)
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_runtime_override=1,  # 1 minute
        max_passes=100,
    )
    assert result.ok is True
    # Should have run fewer than 100 passes
    assert app._call_count < 100


def test_run_supervised_keyboard_interrupt_returns_ok(tmp_path: Path) -> None:
    """KeyboardInterrupt is caught; result is ok=True with summary."""
    call_count = [0]

    class InterruptApp(FakeApp):
        def loop(self, limit: Any = None, *, merge: Any = None) -> CommandResult:
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt
            return _active_result(open_prs=1)

    app = InterruptApp(tmp_path, [])
    fc = FakeClock(auto_advance=1.0)
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=10,
    )
    assert result.ok is True
    assert "supervised loop complete" in result.message


def test_run_supervised_second_instance_lock_returns_false(tmp_path: Path) -> None:
    """Second invocation while lock is held returns ok=False."""
    app = FakeApp(tmp_path, [_drained_result()])

    # Acquire the lock externally before calling run_supervised
    lock_path = app.paths.root / "supervisor.lock"
    lock = try_acquire_supervisor_lock(lock_path)
    assert lock is not None, "pre-requisite: should acquire lock in test setup"

    try:
        result = run_supervised(app, max_passes=1)
        assert result.ok is False
        assert "supervisor already running" in result.message
    finally:
        lock.release()


def test_run_supervised_lock_released_after_run(tmp_path: Path) -> None:
    """After run_supervised finishes, the lock is released (second call succeeds)."""
    app = FakeApp(tmp_path, [_drained_result()])
    fc = FakeClock(auto_advance=0.0)
    result1 = run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=5)
    assert result1.ok is True

    # Should be able to acquire again after first run
    lock_path = app.paths.root / "supervisor.lock"
    lock = try_acquire_supervisor_lock(lock_path)
    assert lock is not None, "lock should be free after run_supervised exits"
    lock.release()
