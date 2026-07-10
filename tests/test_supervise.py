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
    merge_failed: int = 0,
    open_prs: int = 0,
) -> CommandResult:
    """A pass result with some activity.

    ``merged`` produces that many successful merge entries ("merged": True).
    ``merge_failed`` produces that many failed merge ATTEMPT entries
    ("merged": False) -- mirrors merge_ready() appending one entry per
    approved PR regardless of outcome (workflow.py merge_ready). Both land in
    the same "merges" list.
    """
    merges = [{"pr": i, "merged": True} for i in range(merged)]
    merges += [{"pr": 1000 + i, "merged": False} for i in range(merge_failed)]
    return CommandResult(
        True,
        "loop complete",
        {
            "dispatch": {"selected_count": dispatched},
            "dispatch_rework": {"selected_count": rework},
            "merges": merges,
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


def test_should_exit_provider_throttled_dispatch_returns_false() -> None:
    """Regression for finding #2: dispatch() defers with selected_count=0 and
    deferred_reason="provider_throttled" while queued issues are still
    waiting on the throttle cooldown to clear. With zero live workers and no
    open PRs, the old should_exit() misread this as fully drained.
    """
    result = CommandResult(
        False,
        "dispatch deferred: provider throttled until 2026-07-10T00:00:00Z",
        {
            "dispatch": {"selected_count": 0, "deferred_reason": "provider_throttled"},
            "dispatch_rework": {"selected_count": 0},
            "merges": [],
            "open_tracked_prs": 0,
        },
    )
    assert should_exit(result, live_count=0) is False


def test_should_exit_failed_merge_attempts_only_still_blocked_by_open_prs() -> None:
    """Regression: merge_ready() appends one "merges" entry per approved PR
    regardless of outcome. All attempts failing (can_merge=False) must not be
    misread as merge "activity" that would let should_exit ignore the still-
    open PRs -- open_tracked_prs > 0 keeps the loop alive on its own, and the
    honest (successes-only) merged count must not accidentally short-circuit
    that.
    """
    result = _active_result(merge_failed=3, open_prs=2)
    assert should_exit(result, live_count=0) is False


def test_should_exit_all_failed_merge_attempts_no_other_activity_exits() -> None:
    """The honest fix: failed merge attempts alone (no live workers, no
    dispatches, no open PRs) are not "activity" -- should_exit returns True.
    The old implementation (len(data["merges"]) as the merged count) would
    have kept the loop alive here since 3 failed-attempt entries still made
    len(merges) == 3 look nonzero.
    """
    result = _active_result(merge_failed=3, open_prs=0)
    assert should_exit(result, live_count=0) is True


def test_should_exit_provider_throttled_rework_returns_false() -> None:
    """Same as above but the throttle hits dispatch_rework instead of dispatch."""
    result = CommandResult(
        False,
        "rework dispatch deferred: provider throttled until 2026-07-10T00:00:00Z",
        {
            "dispatch": {"selected_count": 0},
            "dispatch_rework": {"selected_count": 0, "deferred_reason": "provider_throttled"},
            "merges": [],
            "open_tracked_prs": 0,
        },
    )
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


def test_take_snapshot_verdict_mtimes_keys_on_pr_parent_not_filename(tmp_path: Path) -> None:
    """Regression for finding #1: verdict_mtimes must key on the PR-unique
    parent directory name ("pr-N"), not path.name (always the constant
    "review-decision.json"). Two PRs whose verdict files happen to share an
    identical mtime must still produce two distinct snapshot entries --
    keying on path.name alone would collapse them into a single set element
    (sets dedup identical tuples), silently erasing one PR's presence from
    the delta signal and letting a later rewrite go unnoticed.
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()
    pr1_dir = prs / "pr-1"
    pr1_dir.mkdir()
    pr2_dir = prs / "pr-2"
    pr2_dir.mkdir()
    verdict1 = pr1_dir / "review-decision.json"
    verdict2 = pr2_dir / "review-decision.json"
    verdict1.write_text('{"decision": "approved"}', encoding="utf-8")
    verdict2.write_text('{"decision": "approved"}', encoding="utf-8")

    import os

    shared_mtime = 1_700_000_000.0
    os.utime(verdict1, (shared_mtime, shared_mtime))
    os.utime(verdict2, (shared_mtime, shared_mtime))

    snap1 = take_snapshot(sessions, prs)
    assert len(snap1.verdict_mtimes) == 2, (
        "two PRs with identical verdict mtimes collapsed into one entry -- "
        "verdict_mtimes is keyed on path.name instead of the PR parent dir"
    )

    # Rewrite PR-1's verdict only; PR-2 is left untouched.
    os.utime(verdict1, (shared_mtime + 5, shared_mtime + 5))
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
# Pass summary line: merged count reflects actual merges, not attempts
# ---------------------------------------------------------------------------


def test_pass_summary_reports_zero_merged_for_all_failed_attempts(
    tmp_path: Path, capsys: Any
) -> None:
    """Regression: today's production run printed "merged 3" for a pass where
    all 3 merge attempts had can_merge=False and zero PRs actually merged.
    The summary line must report the real (successful) count -- with the
    attempt count surfaced (0/3) since it diverges from successes.
    """
    app = FakeApp(tmp_path, [_active_result(merge_failed=3, open_prs=3)])
    fc = FakeClock(auto_advance=1.0)
    run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=1)

    out = capsys.readouterr().out
    assert "merged 0/3" in out
    assert "merged 3" not in out.replace("merged 0/3", "")


def test_pass_summary_reports_plain_count_when_all_attempts_succeed(
    tmp_path: Path, capsys: Any
) -> None:
    """When attempts == successes, the line stays in the compact plain form
    ("merged N"), not "N/N" -- keeps the common case stable for callers/tests
    that grep the plain form.
    """
    app = FakeApp(tmp_path, [_active_result(merged=2)])
    fc = FakeClock(auto_advance=1.0)
    run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=1)

    out = capsys.readouterr().out
    assert "merged 2" in out
    assert "merged 2/2" not in out


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
    """A sidecar disappearing (worker exited) triggers a delta → prompt pass
    which dispatches.

    Bounded proof (finding #12): ``full_pass_interval_seconds`` is set far out
    of reach and ``max_passes=2`` caps the run, so pass 2 can ONLY fire because
    ``has_delta()`` detected the vanished sidecar -- not because the fallback
    timer eventually fires (the old assertion, ``call_count >= 2``, would have
    passed even with ``has_delta`` hard-coded to ``False``, since the fallback
    would eventually force a pass).
    """
    # Plant a sidecar that will vanish between first poll and first pass
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sidecar = sessions_dir / "issue-1.json"
    sidecar.write_text("{}", encoding="utf-8")

    cfg = SupervisorConfig(
        poll_interval_seconds=5,
        full_pass_interval_seconds=100_000,
        active_cooldown_seconds=9,
    )
    # Pass 1: still-live (open_prs=1 keeps loop alive), Pass 2: dispatches 1
    # (dispatched > 0 keeps the loop alive one more cooldown before
    # max_passes=2 stops it -- that trailing sleep is expected and doesn't
    # weaken the bound on how pass 2 itself was triggered).
    results = [
        _active_result(open_prs=1),
        _active_result(dispatched=1),
    ]
    app = FakeApp(tmp_path, results, supervisor_cfg=cfg)
    app._sessions_dir = sessions_dir

    fc = FakeClock(auto_advance=1.0)

    def sleeping(seconds: float) -> None:
        fc.sleep(seconds)
        # After first sleep, remove the sidecar to simulate worker exit
        if len(fc.sleep_calls) == 1 and sidecar.exists():
            sidecar.unlink()
        assert len(fc.sleep_calls) <= 2, (
            "pass 2 should have fired after exactly one poll interval via "
            "has_delta(); the fallback timer is set far out of reach so "
            "further sleeps mean delta detection did not fire it"
        )

    result = run_supervised(
        app,
        clock=fc.now,
        sleep=sleeping,
        max_passes=2,
    )
    assert result.ok is True
    assert app._call_count == 2
    # Bound the proof: pass 2 fires after exactly ONE poll-interval sleep
    # (5.0s, not the 100_000s fallback); the trailing active-cooldown sleep
    # (9.0s) follows pass 2 itself, per should_exit keeping the loop alive
    # while dispatched > 0.
    assert fc.sleep_calls == [5.0, 9.0]


def test_run_supervised_verdict_file_triggers_pass_while_live_zero(tmp_path: Path) -> None:
    """A verdict file appearing while live=0 keeps the loop alive and triggers
    a merge pass.

    Bounded proof (finding #12): ``full_pass_interval_seconds`` is set far out
    of reach and ``max_passes=2`` caps the run, so pass 2 can ONLY be
    explained by ``has_delta()`` picking up the new verdict file -- not by the
    fallback timer (the old assertion, ``call_count >= 2``, would have passed
    even with ``has_delta`` hard-coded to ``False``).
    """
    prs_dir = tmp_path / ".var" / "charlie-work" / "prs"
    prs_dir.mkdir(parents=True, exist_ok=True)
    pr_dir = prs_dir / "pr-456"
    pr_dir.mkdir()

    cfg = SupervisorConfig(
        poll_interval_seconds=5,
        full_pass_interval_seconds=100_000,
        active_cooldown_seconds=9,
    )
    # Pass 1: no workers, no dispatched, but open_prs=1 → stay alive
    # Pass 2: merges=1 after verdict written
    results = [
        _active_result(open_prs=1),
        _active_result(merged=1),
    ]
    app = FakeApp(tmp_path, results, supervisor_cfg=cfg)

    fc = FakeClock(auto_advance=1.0)

    def sleeping(seconds: float) -> None:
        fc.sleep(seconds)
        # Write verdict after first sleep
        if len(fc.sleep_calls) == 1:
            verdict = pr_dir / "review-decision.json"
            verdict.write_text('{"decision": "approved"}', encoding="utf-8")
        assert len(fc.sleep_calls) <= 2, (
            "pass 2 should have fired after exactly one poll interval via "
            "has_delta(); the fallback timer is set far out of reach so "
            "further sleeps mean delta detection did not fire it"
        )

    result = run_supervised(
        app,
        clock=fc.now,
        sleep=sleeping,
        max_passes=2,
    )
    assert result.ok is True
    assert app._call_count == 2
    # Bound the proof: pass 2 fires after exactly ONE poll-interval sleep
    # (5.0s, not the 100_000s fallback); the trailing active-cooldown sleep
    # (9.0s) follows pass 2 itself, per should_exit keeping the loop alive
    # while merged > 0.
    assert fc.sleep_calls == [5.0, 9.0]


def test_run_supervised_fallback_timer_fires_with_no_delta(tmp_path: Path) -> None:
    """After a pass with no local-signal delta, the fallback timer still
    forces a subsequent pass once ``full_pass_interval_seconds`` genuinely
    elapses.

    This is distinct from first-pass priming (finding #12): the OLD test only
    proved pass 1 fires, which is trivially true on iteration 1 regardless of
    whether the fallback timer's elapsed-time math works at all (priming sets
    ``last_full_pass_at`` behind "now" specifically to force iteration 1).
    Here the clock is advanced PAST the interval only after pass 1 completes,
    with nothing else changing in sessions/prs, so pass 2 can only be
    explained by the fallback timer noticing real elapsed time.
    """
    cfg = SupervisorConfig(
        full_pass_interval_seconds=10,
        poll_interval_seconds=5,
        active_cooldown_seconds=5,
    )
    results = [_active_result(open_prs=1), _drained_result()]
    app = FakeApp(tmp_path, results, supervisor_cfg=cfg)

    fc = FakeClock(start=0.0, auto_advance=0.0)

    def sleeping(seconds: float) -> None:
        fc.sleep(seconds)
        if len(fc.sleep_calls) == 1:
            # Push the clock past the fallback threshold only AFTER pass 1
            # has completed with no file changes.
            fc.advance(cfg.full_pass_interval_seconds)

    result = run_supervised(
        app,
        clock=fc.now,
        sleep=sleeping,
        max_passes=5,
    )
    assert result.ok is True
    # Pass 1 fires from priming; pass 2 fires from the fallback timer once
    # real elapsed time (not priming) crosses full_pass_interval_seconds.
    assert app._call_count == 2


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


def test_run_supervised_exception_returns_ok_false_and_releases_lock(tmp_path: Path) -> None:
    """Regression for finding #3: a raw exception from app.loop() must not
    propagate past run_supervised (errors-as-values invariant) -- it comes
    back as CommandResult(ok=False, ...) with the pass number in the
    message, and the supervisor lock is still released afterward.
    """
    call_count = [0]

    class RaisingApp(FakeApp):
        def loop(self, limit: Any = None, *, merge: Any = None) -> CommandResult:
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("boom")
            return _active_result(open_prs=1)

    app = RaisingApp(tmp_path, [])
    fc = FakeClock(auto_advance=1.0)
    result = run_supervised(
        app,
        clock=fc.now,
        sleep=fc.sleep,
        max_passes=10,
    )
    assert result.ok is False
    assert "pass 2" in result.message
    assert "boom" in result.message

    # Lock must be released even though the loop aborted via exception --
    # a fresh acquire must succeed.
    lock_path = app.paths.root / "supervisor.lock"
    lock = try_acquire_supervisor_lock(lock_path)
    assert lock is not None, "lock should be released after an aborted pass"
    lock.release()


def test_try_acquire_supervisor_lock_zero_byte_existing_file_succeeds(tmp_path: Path) -> None:
    """Regression for finding #8: a pre-existing 0-byte lock file (e.g. left
    over from an older touch()-based implementation) must not permanently
    block acquisition -- msvcrt.locking raises EACCES on a 0-byte file even
    for a non-blocking attempt, so the acquire path must top up the file
    before locking, not just on first creation.
    """
    lock_path = tmp_path / "supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")  # simulate the old touch()-created 0-byte file
    assert lock_path.stat().st_size == 0

    lock = try_acquire_supervisor_lock(lock_path)
    assert lock is not None, "0-byte pre-existing lock file should still be acquirable"
    lock.release()


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
