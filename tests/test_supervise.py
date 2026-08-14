"""Tests for charlie_work.supervise — supervised infill loop.

FakeApp: minimal stand-in for OrchestratorApp — only the attributes that
run_supervised touches.  Kept local to this test file.

Injected sleep/clock: record sleep args; monotonically advancing fake clock.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest

from charlie_work.config import OrchestratorConfig, SupervisorConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import resolved_layout
from charlie_work import layout
from charlie_work.subprocess_runner import RunResult, run_captured
from charlie_work.supervise import (
    PullBlockerRepair,
    SelfDeployResult,
    _BLOCKER_NAMES_IN_MESSAGE,
    _check_venv,
    _command_failure_message,
    _pending_sync_marker_path,
    _pull_ci_fleet_sibling,
    _record_self_deploy_failure_streak,
    _repair_lossless_pull_blockers,
    _repairable_blocker_path,
    _self_deploy_failure_counter_path,
    _self_deploy_state_path,
    _zero_pass_streak_counter_path,
    has_delta,
    orchestrator_root,
    record_zero_pass_streak,
    run_supervised,
    self_deploy,
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
        # Public layout contract mirroring OrchestratorApp.layout. Override
        # sessions_dir back onto the stub's own `_sessions_dir` (rather than
        # whatever resolved_layout derives from config.devin.sessions_dir) so
        # fixtures that write session files into that exact directory keep
        # working -- the rest of the resolved layout is unused by
        # run_supervised today but kept real (not stubbed) so this stays a
        # faithful stand-in.
        self.layout = replace(
            resolved_layout(self.config, tmp_path), sessions_dir=self._sessions_dir
        )
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
    warnings: list[str] | None = None,
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
            "warnings": warnings if warnings is not None else [],
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


def test_should_exit_fleet_lock_held_returns_false() -> None:
    """Dispatch deferred because another repo holds the fleet lock is not drained."""
    result = CommandResult(
        True,
        "dispatch deferred: fleet lock held",
        {
            "dispatch": {"selected_count": 0, "deferred_reason": "fleet_lock_held"},
            "dispatch_rework": {"selected_count": 0},
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


def test_take_snapshot_excludes_launch_failure_sidecar(tmp_path: Path) -> None:
    """Issue #266: a launch-failure sidecar (pid=None, error set) is not counted as live."""
    import json
    from charlie_work.devin_shell import SessionRecord

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()

    sidecar = sessions / "issue-1.json"
    sidecar.write_text(
        json.dumps(
            SessionRecord(
                issue_number=1,
                branch="agent/issue-1-x",
                worktree_path="/tmp/worktree",
                prompt_path="/tmp/prompt.md",
                command=("devin",),
                pid=None,
                started_at="2024-01-01T00:00:00Z",
                log_path="/tmp/issue-1.log",
                error="worktree path already exists",
            ).to_dict()
        ),
        encoding="utf-8",
    )

    snap = take_snapshot(sessions, prs)
    assert snap.live_count == 0
    assert len(snap.sidecar_mtimes) == 1


def test_take_snapshot_counts_alive_workers_not_sidecar_files(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Issue #266: live_count reflects actual live workers, not raw file count."""
    import json
    from charlie_work.devin_shell import SessionRecord

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    prs = tmp_path / "prs"
    prs.mkdir()

    sidecar = sessions / "issue-1.json"
    sidecar.write_text(
        json.dumps(
            SessionRecord(
                issue_number=1,
                branch="agent/issue-1-x",
                worktree_path="/tmp/worktree",
                prompt_path="/tmp/prompt.md",
                command=("devin",),
                pid=12345,
                started_at="2024-01-01T00:00:00Z",
                log_path="/tmp/issue-1.log",
            ).to_dict()
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    snap = take_snapshot(sessions, prs)
    assert snap.live_count == 1


def test_run_supervised_exits_with_launch_failure_sidecar(tmp_path: Path) -> None:
    """Issue #266: loop exits when only a launch-failure sidecar is present."""
    import json
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.config import SupervisorConfig

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    sidecar = sessions / "issue-1.json"
    sidecar.write_text(
        json.dumps(
            SessionRecord(
                issue_number=1,
                branch="agent/issue-1-x",
                worktree_path="/tmp/worktree",
                prompt_path="/tmp/prompt.md",
                command=("devin",),
                pid=None,
                started_at="2024-01-01T00:00:00Z",
                log_path="/tmp/issue-1.log",
                error="devin binary not found",
            ).to_dict()
        ),
        encoding="utf-8",
    )

    cfg = SupervisorConfig(
        full_pass_interval_seconds=999,
        active_cooldown_seconds=0,
        max_runtime_minutes=999,
    )
    app = FakeApp(tmp_path, results=[], supervisor_cfg=cfg)
    app._sessions_dir = sessions

    def _remove_sidecar_and_drain(limit: Any = None, *, merge: Any = None) -> CommandResult:
        sidecar.unlink(missing_ok=True)
        return _drained_result()

    app.loop = _remove_sidecar_and_drain

    result = run_supervised(
        app,
        sleep=FakeClock().sleep,
        clock=FakeClock().now,
        max_passes=2,
    )
    assert result.ok is True
    assert not sidecar.exists()


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


def test_pass_summary_reports_warnings_count(tmp_path: Path, capsys: Any) -> None:
    """Issue #254: the summary line counts pass warnings (e.g. merge alarms)."""
    app = FakeApp(
        tmp_path,
        [_active_result(warnings=["PR #456 approved but unmergeable for 3 passes"])],
    )
    fc = FakeClock(auto_advance=1.0)
    run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=1)

    out = capsys.readouterr().out
    assert "warnings 1" in out


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
    over from an older touch()-based implementation) must remain acquirable.

    On the deployed runtime (Python 3.13.5, Windows 11), ``msvcrt.locking``
    with ``LK_NBLCK`` succeeds on a genuine 0-byte file, so the lock helper
    does not need to pad the file before locking.
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


def test_run_supervised_summary_uses_fleet_live_count(tmp_path: Path, capfd: Any) -> None:
    """The 'live ~N' summary line uses the dispatch-scoped fleet-wide count, not the local snapshot."""
    result = CommandResult(
        True,
        "loop complete",
        {
            "dispatch": {
                "selected_count": 0,
                "fleet_live_session_count": 2,
                "live_session_count": 1,
            },
            "dispatch_rework": {"selected_count": 0},
            "merges": [],
            "reviews": [],
            "errors": [],
            "open_tracked_prs": 0,
            "skipped_reviews": 0,
        },
    )
    app = FakeApp(tmp_path, [result])
    fc = FakeClock(auto_advance=0.0)
    run_supervised(app, clock=fc.now, sleep=fc.sleep, max_passes=1)

    out = capfd.readouterr().out
    assert "live ~2" in out, "summary should report fleet-wide live count"
    assert "live ~1" not in out, (
        "summary should not report local snapshot count when fleet count is available"
    )


# ---------------------------------------------------------------------------
# self_deploy unit tests
# ---------------------------------------------------------------------------


def _make_fake_runner(
    responses: list[RunResult],
) -> tuple[Callable[..., RunResult], list[tuple[list[str], Path, int]]]:
    """Return a callable that consumes ``responses`` and records its calls."""
    calls: list[tuple[list[str], Path, int]] = []

    def runner(command: list[str], *, cwd: Path, timeout_seconds: int) -> RunResult:
        calls.append((command, cwd, timeout_seconds))
        return responses.pop(0)

    return runner, calls


@pytest.fixture
def no_fleet_live_sessions(monkeypatch: Any) -> None:
    """Patch fleet live-session counting to zero so self_deploy tests stay hermetic."""
    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (0, []),
    )


def test_self_deploy_code_only_change_does_not_sync(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A pull that changes only source files triggers no uv sync."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "def456\n", ""),  # after HEAD
            RunResult(0, "src/foo.py\nREADME.md\n", ""),  # diff
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result == SelfDeployResult(
        ok=True,
        pulled=True,
        changed=True,
        synced=False,
        head_changed=True,
        from_sha="abc123",
        to_sha="def456",
        message="code-only update: def456",
    )
    assert len(calls) == 4
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["git", "pull", "--ff-only", "origin", "main"],
        ["git", "rev-parse", "HEAD"],
        ["git", "diff", "--name-only", "abc123..def456"],
    ]


def test_self_deploy_dependency_change_triggers_uv_sync(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A pull touching pyproject.toml/uv.lock runs uv sync and reports success."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "", ""),
            RunResult(0, "def456\n", ""),
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),
            RunResult(0, "", ""),
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result.ok is True
    assert result.pulled is True
    assert result.changed is True
    assert result.synced is True
    assert result.from_sha == "abc123"
    assert result.to_sha == "def456"
    assert "updated and synced" in result.message
    assert calls[-1][0] == ["uv", "sync"]


def test_self_deploy_pull_failure_is_non_fatal(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A diverged/dirty tree causes the pull to fail; self_deploy returns but does not raise.

    On failure, ``_self_deploy_attempt`` now calls ``_repair_lossless_pull_blockers``
    before giving up, which re-reads HEAD and ``origin/main`` and, in a genuinely
    diverged tree, bails out at the ``merge-base --is-ancestor`` check -- three
    extra canned responses (HEAD, origin/main, the ancestor check failing) beyond
    the original two.
    """
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(1, "", "fatal: Not possible to fast-forward, aborting."),
            RunResult(0, "abc123\n", ""),  # repair: rev-parse HEAD
            RunResult(0, "def999\n", ""),  # repair: rev-parse origin/main (diverged)
            RunResult(1, "", ""),  # repair: merge-base --is-ancestor fails
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result.ok is False
    assert result.pulled is False
    assert result.changed is False
    assert result.synced is False
    assert result.from_sha == "abc123"
    assert "fast-forward" in (result.error or "")
    assert len(calls) == 5


def test_self_deploy_already_up_to_date(tmp_path: Path, no_fleet_live_sessions: None) -> None:
    """When the pull succeeds but HEAD does not move, no sync is attempted."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "Already up to date.\n", ""),
            RunResult(0, "abc123\n", ""),
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result == SelfDeployResult(
        ok=True,
        pulled=True,
        changed=False,
        synced=False,
        head_changed=False,
        from_sha="abc123",
        to_sha="abc123",
        message="already up to date",
    )
    assert len(calls) == 3
    assert all(c[0] != ["uv", "sync"] for c in calls)


def test_self_deploy_uv_sync_failure_is_non_fatal(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """If uv sync fails after a dependency-changing pull, self_deploy reports the error."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "", ""),
            RunResult(0, "def456\n", ""),
            RunResult(0, "uv.lock\n", ""),
            RunResult(1, "", "failed to install"),
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result.ok is False
    assert result.pulled is True
    assert result.changed is True
    assert result.synced is False
    assert result.to_sha == "def456"
    assert "failed to install" in (result.error or "")


def test_self_deploy_defers_sync_when_fleet_runners_active(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A dependency-changing pull defers uv sync while fleet live sessions are active."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "def456\n", ""),  # after HEAD
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # uv sync (should not be reached)
        ]
    )

    def _fake_count(_fleet_dir_override: str | None) -> tuple[int, list[str]]:
        return 2, []

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_sessions", _fake_count)

    result = self_deploy(tmp_path, run_command=runner)
    assert result == SelfDeployResult(
        ok=True,
        pulled=True,
        changed=True,
        synced=False,
        head_changed=True,
        from_sha="abc123",
        to_sha="def456",
        message="sync deferred: 2 runners active",
        deferred=True,
    )
    assert all(c[0] != ["uv", "sync"] for c in calls)
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["git", "pull", "--ff-only", "origin", "main"],
        ["git", "rev-parse", "HEAD"],
        ["git", "diff", "--name-only", "abc123..def456"],
    ]

    marker_path = _pending_sync_marker_path(layout.default_state_root(tmp_path))
    assert marker_path.exists()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker == {"from_sha": "abc123", "to_sha": "def456"}


def test_self_deploy_honors_state_root_override(tmp_path: Path, monkeypatch: Any) -> None:
    """Issue #720: a configured state_root moves the pending-sync marker out of default."""
    custom_state_root = tmp_path / ".var" / "devin-orchestrator"

    def _fake_count(_fleet_dir_override: str | None) -> tuple[int, list[str]]:
        return 1, []

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_sessions", _fake_count)

    runner, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "def456\n", ""),  # after HEAD
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # uv sync (not reached)
        ]
    )

    result = self_deploy(tmp_path, state_root=custom_state_root, run_command=runner)
    assert result.deferred is True
    assert _pending_sync_marker_path(custom_state_root).exists()
    assert not _pending_sync_marker_path(layout.default_state_root(tmp_path)).exists()


def test_self_deploy_proceeds_when_zero_fleet_runners(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A dependency-changing pull runs uv sync when no fleet live sessions are active."""
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "def456\n", ""),  # after HEAD
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # uv sync ok
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)
    assert result.ok is True
    assert result.pulled is True
    assert result.changed is True
    assert result.synced is True
    assert result.head_changed is True
    assert result.from_sha == "abc123"
    assert result.to_sha == "def456"
    assert "updated and synced" in result.message
    assert calls[-1][0] == ["uv", "sync"]
    assert not _pending_sync_marker_path(layout.default_state_root(tmp_path)).exists()


def test_self_deploy_retries_sync_after_deferral(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deferred dependency sync is retried on the next pass once runners are idle.

    Regression for the pass-after-deferral convergence bug: the original code
    returned "already up to date" on the next pass (because HEAD did not move)
    before checking the deferred sync, so uv sync never ran.
    """
    live_counts = iter([2, 0])

    def _fake_count(_fleet_dir_override: str | None) -> tuple[int, list[str]]:
        return next(live_counts), []

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_sessions", _fake_count)

    # Pass N: dependency-changing pull, two active runners -> defer and write marker.
    first_runner, first_calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "def456\n", ""),  # after HEAD
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(0, "", ""),  # uv sync (not reached)
        ]
    )

    first = self_deploy(tmp_path, run_command=first_runner)
    assert first.synced is False
    assert first.message == "sync deferred: 2 runners active"

    marker_path = _pending_sync_marker_path(layout.default_state_root(tmp_path))
    assert marker_path.exists()

    # Pass N+1: no new commits, runners now idle -> sync from marker and clear it.
    second_runner, second_calls = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # before HEAD
            RunResult(0, "Already up to date.\n", ""),  # pull
            RunResult(0, "def456\n", ""),  # after HEAD (unchanged)
            RunResult(0, "", ""),  # uv sync ok
        ]
    )

    second = self_deploy(tmp_path, run_command=second_runner)
    assert second == SelfDeployResult(
        ok=True,
        pulled=True,
        changed=True,
        synced=True,
        from_sha="abc123",
        to_sha="def456",
        message="updated and synced: def456",
    )
    assert second_calls[-1][0] == ["uv", "sync"]
    assert not marker_path.exists()


def test_self_deploy_loud_warning_on_repeated_deferral(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """When a pending-sync marker survives repeated passes, a warning is printed.

    Also the outage regression guard: HEAD did not move on *this* attempt
    (before == after == "def456") even though a marker from an earlier
    deferral is still present and live workers are still active. Callers
    must see ``head_changed is False`` here -- gating a watchdog restart on
    ``from_sha != to_sha`` instead (the marker's original range, from
    "abc123" to "def456") previously caused the supervisor to exit and
    relaunch every single pass without ever reaching zero live workers to
    complete the deferred sync (the total-fleet-outage bug this test guards
    against).
    """
    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (3, []),
    )

    # Create marker from a previous deferral.
    marker_path = _pending_sync_marker_path(layout.default_state_root(tmp_path))
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps({"from_sha": "abc123", "to_sha": "def456"}), encoding="utf-8"
    )

    runner, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # before HEAD
            RunResult(0, "Already up to date.\n", ""),  # pull
            RunResult(0, "def456\n", ""),  # after HEAD (unchanged)
            RunResult(0, "", ""),  # uv sync (not reached)
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)
    assert result.synced is False
    assert result.head_changed is False
    assert "3 runners active" in result.message
    assert marker_path.exists()

    out = capsys.readouterr().out
    assert "WARNING: pending dependency sync still deferred" in out
    assert "3 runners active" in out
    assert "abc123..def456" in out


def test_self_deploy_pull_failure_surfaces_stderr_over_generic_error(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """Issue #817 item 3: a realistic failed RunResult -- as ``run_captured``
    actually produces on any non-zero exit, with ``.error`` always populated
    with the generic ``"command exited N"`` rather than left at the
    dataclass default ``None`` -- must still surface git's specific stderr
    (which names the colliding path) instead of the uninformative generic
    message.

    ``test_self_deploy_pull_failure_is_non_fatal`` above never caught the
    old ``result.error or result.stderr`` bug because it constructs
    ``RunResult(1, "", "fatal: ...")`` without ``.error``, leaving it at the
    ``None`` default -- under the old fallback chain that made ``.stderr``
    win "by accident" (None is falsy), masking the real production
    shadowing where ``.error`` is always truthy.

    Three extra canned responses beyond the original two account for
    ``_repair_lossless_pull_blockers`` (invoked on every pull failure since
    this feature landed) short-circuiting at the diverged-tree check -- see
    the sibling test above for the same shape.
    """
    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(
                returncode=1,
                stdout="",
                stderr=(
                    "error: Your local changes to the following files would be "
                    "overwritten by merge:\n\tsrc/charlie_work/config.py\n"
                    "Please commit your changes or stash them before you merge."
                ),
                error="command exited 1",
            ),
            RunResult(0, "abc123\n", ""),  # repair: rev-parse HEAD
            RunResult(0, "def999\n", ""),  # repair: rev-parse origin/main (diverged)
            RunResult(1, "", ""),  # repair: merge-base --is-ancestor fails
        ]
    )
    result = self_deploy(tmp_path, run_command=runner)
    assert result.ok is False
    assert result.error is not None
    assert "src/charlie_work/config.py" in result.error
    assert "command exited 1" not in result.error
    assert result.error.startswith("git pull --ff-only origin main: ")
    assert len(calls) == 5


def test_command_failure_message_falls_back_to_error_then_fallback() -> None:
    """``_command_failure_message`` prefers stderr, then .error, then fallback."""
    stderr_result = RunResult(1, "", "  stderr detail  ")
    assert _command_failure_message(["git", "pull"], stderr_result, "pull failed") == (
        "git pull: stderr detail"
    )

    error_only_result = RunResult(returncode=None, stdout="", stderr="", error="boom")
    assert _command_failure_message(["uv", "sync"], error_only_result, "sync failed") == (
        "uv sync: boom"
    )

    empty_result = RunResult(returncode=1, stdout="", stderr="")
    assert _command_failure_message(["git", "diff"], empty_result, "diff failed") == (
        "git diff: diff failed"
    )


def test_self_deploy_records_events_db_outcome_for_every_pass(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """Issue #817 item 4: every real self_deploy pass -- success, skip, and
    failure -- is durably recorded to events.db, queryable via
    ``query_events``. Before this fix, self_deploy had zero events.db
    instrumentation; the live fleet accumulated 121 consecutive real deploy
    failures with zero rows to show for it.
    """
    state_path = _self_deploy_state_path(tmp_path)

    # Pass 1: code-only success.
    runner1, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "", ""),
            RunResult(0, "def456\n", ""),
            RunResult(0, "src/foo.py\n", ""),
        ]
    )
    self_deploy(tmp_path, run_command=runner1)

    # Pass 2: already up to date -- ok, but nothing changed (a skip).
    runner2, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),
            RunResult(0, "Already up to date.\n", ""),
            RunResult(0, "def456\n", ""),
        ]
    )
    self_deploy(tmp_path, run_command=runner2)

    # Pass 3: pull failure. On failure, ``_self_deploy_attempt`` now calls
    # ``_repair_lossless_pull_blockers`` before giving up, which re-reads HEAD
    # and ``origin/main`` and, in a genuinely diverged tree, bails out at the
    # ``merge-base --is-ancestor`` check -- three extra canned responses
    # (HEAD, origin/main, the ancestor check failing) beyond the original two.
    runner3, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),
            RunResult(
                returncode=1,
                stdout="",
                stderr="fatal: could not read from remote repository.",
                error="command exited 1",
            ),
            RunResult(0, "def456\n", ""),  # repair: rev-parse HEAD
            RunResult(0, "xyz789\n", ""),  # repair: rev-parse origin/main (diverged)
            RunResult(1, "", ""),  # repair: merge-base --is-ancestor fails
        ]
    )
    self_deploy(tmp_path, run_command=runner3)

    succeeded = query_events(state_path, kind="self_deploy_succeeded")
    skipped = query_events(state_path, kind="self_deploy_skipped")
    failed = query_events(state_path, kind="self_deploy_failed")
    assert len(succeeded) == 1
    assert len(skipped) == 1
    assert len(failed) == 1
    assert failed[0]["level"] == "error"
    assert "could not read from remote" in failed[0]["payload"]["error"]

    # query_events(level="error") -- the general-purpose alerting query
    # already used elsewhere in the codebase -- surfaces the failure without
    # any self-deploy-specific query infrastructure.
    errors = query_events(state_path, level="error")
    assert any(e["kind"] == "self_deploy_failed" for e in errors)


def test_self_deploy_preview_does_not_record_events_db(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """A ``dry_run`` preview touches nothing, including events.db (item 4)."""
    state_path = _self_deploy_state_path(tmp_path)
    runner, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "abc123\n", ""),
        ]
    )
    result = self_deploy(tmp_path, run_command=runner, dry_run=True)
    assert result.previewed is True
    assert query_events(state_path, kind="self_deploy_succeeded") == []
    assert query_events(state_path, kind="self_deploy_skipped") == []
    assert query_events(state_path, kind="self_deploy_failed") == []


def _fake_pull_failure_runner() -> Callable[..., RunResult]:
    runner, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(
                returncode=1,
                stdout="",
                stderr="fatal: unable to access remote",
                error="command exited 1",
            ),
        ]
    )
    return runner


def test_self_deploy_failure_streak_fires_alarm_once_at_threshold(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """Issue #817 item 5: three consecutive self_deploy failures cross the
    default threshold (3) and fire exactly one ``self_deploy_alarm`` event --
    not one per failure past the threshold -- and a subsequent success
    resets the counter so a later failure streak starts counting from zero
    again instead of re-alarming immediately.
    """
    state_path = _self_deploy_state_path(tmp_path)
    counter_path = _self_deploy_failure_counter_path(tmp_path)

    self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=3)
    self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=3)
    assert query_events(state_path, kind="self_deploy_alarm") == []

    self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=3)
    alarms = query_events(state_path, kind="self_deploy_alarm")
    assert len(alarms) == 1
    assert alarms[0]["payload"]["consecutive_failures"] == 3
    assert alarms[0]["level"] == "error"

    # A fourth consecutive failure must not fire a second alarm.
    self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=3)
    assert len(query_events(state_path, kind="self_deploy_alarm")) == 1

    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 4

    # A success resets the streak.
    success_runner, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),
            RunResult(0, "Already up to date.\n", ""),
            RunResult(0, "abc123\n", ""),
        ]
    )
    self_deploy(tmp_path, run_command=success_runner, failure_alarm_threshold=3)
    counter_after_success = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter_after_success["consecutive_failures"] == 0

    # Two more failures after the reset must not re-fire the alarm yet
    # (streak restarted at 0, threshold is 3).
    self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=3)
    self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=3)
    assert len(query_events(state_path, kind="self_deploy_alarm")) == 1


def _deferred_result() -> SelfDeployResult:
    """A genuine deferral: a pending sync was postponed because a fleet
    worker was live (issue #858). ``deferred=True`` is what the streak
    recorder actually keys on -- ``ok=True, synced=False`` alone is *not*
    sufficient, since "already up to date" and "code-only update" share that
    shape without anything pending to sync."""
    return SelfDeployResult(
        ok=True,
        pulled=True,
        changed=True,
        synced=False,
        message="sync deferred: 1 runner active",
        deferred=True,
    )


def _failed_result() -> SelfDeployResult:
    return SelfDeployResult(
        ok=False, pulled=False, changed=False, synced=False, error="uv sync failed"
    )


def _synced_result() -> SelfDeployResult:
    return SelfDeployResult(
        ok=True, pulled=True, changed=True, synced=True, message="updated and synced: abc123"
    )


def test_self_deploy_deferred_sync_does_not_reset_failure_streak(tmp_path: Path) -> None:
    """Issue #858: a deferred sync (``ok=True, synced=False``) carries no
    information about whether the pending sync would have succeeded, so it
    must leave the consecutive-failure streak unchanged -- neither reset
    (today's bug: any ``ok`` result resets to 0, so an interleaved deferral
    silently rewinds the counter and the alarm threshold is never reached)
    nor incremented.

    Replays the live sequence from the issue: fail, fail, defer, fail with
    threshold=3. Before the fix this fires zero alarms, because the deferral
    between the second and third failure resets the counter to 0. The fix
    must fire exactly one alarm, on the final failure.
    """
    state_path = _self_deploy_state_path(tmp_path)
    counter_path = _self_deploy_failure_counter_path(tmp_path)

    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 2
    assert query_events(state_path, kind="self_deploy_alarm") == []

    # A deferral interleaves here -- it must not touch the streak.
    _record_self_deploy_failure_streak(tmp_path, _deferred_result(), threshold=3)
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 2, (
        "a deferred sync (ok=True, synced=False) must leave the streak "
        "unchanged, not reset it to 0"
    )
    assert query_events(state_path, kind="self_deploy_alarm") == []

    # Third genuine failure must cross the threshold and fire exactly once.
    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 3
    alarms = query_events(state_path, kind="self_deploy_alarm")
    assert len(alarms) == 1
    assert alarms[0]["payload"]["consecutive_failures"] == 3


def test_self_deploy_pure_deferral_run_never_alarms_or_resets(tmp_path: Path) -> None:
    """Issue #858 acceptance criterion 5: a long run of pure deferrals never
    fires the alarm and never resets a pre-existing nonzero streak -- a
    healthy fleet that defers for hours (because workers are always live)
    must not have its outage history silently erased.
    """
    state_path = _self_deploy_state_path(tmp_path)
    counter_path = _self_deploy_failure_counter_path(tmp_path)

    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 2

    for _ in range(50):
        _record_self_deploy_failure_streak(tmp_path, _deferred_result(), threshold=3)

    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 2
    assert query_events(state_path, kind="self_deploy_alarm") == []


def test_self_deploy_genuine_sync_resets_failure_streak(tmp_path: Path) -> None:
    """A genuinely synced result (``ok=True, synced=True``) still resets the
    streak to 0 -- unchanged from today's behavior, and distinct from a
    deferral, which must not reset it.
    """
    counter_path = _self_deploy_failure_counter_path(tmp_path)

    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    _record_self_deploy_failure_streak(tmp_path, _failed_result(), threshold=3)
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 2

    _record_self_deploy_failure_streak(tmp_path, _synced_result(), threshold=3)
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 0


def test_self_deploy_streak_survives_a_real_deferral_end_to_end(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Issue #858, driven through the public ``self_deploy()`` entry point
    rather than hand-built ``SelfDeployResult`` values -- this is the wiring
    check: it proves the *real* deferral branch in ``_self_deploy_attempt``
    (live_count > 0, ~supervise.py:877) actually produces a result the
    streak recorder classifies as deferred, not just that the recorder's
    arithmetic is correct for a hand-built one.

    Replays the live incident shape: a failing ``uv sync`` interleaved with a
    genuine deferral once fleet workers come up, threshold=3.

    1. dep-change pull, 0 runners, uv sync fails -> streak 1
    2. marker still pending, 0 runners, uv sync fails again -> streak 2
    3. marker still pending, 2 runners live -> real deferral -> streak stays 2
    4. marker still pending, 0 runners, uv sync fails again -> streak 3, one alarm
    """
    state_path = _self_deploy_state_path(tmp_path)
    counter_path = _self_deploy_failure_counter_path(tmp_path)
    live_counts = iter([0, 0, 2, 0])

    def _fake_count(_fleet_dir_override: str | None) -> tuple[int, list[str]]:
        return next(live_counts), []

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_sessions", _fake_count)

    # Pass 1: dependency-changing pull, 0 runners -> attempts sync, sync fails.
    runner1, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "def456\n", ""),  # after HEAD
            RunResult(0, "pyproject.toml\nuv.lock\n", ""),  # diff
            RunResult(1, "", "uv sync failed: resolution error"),  # uv sync fails
        ]
    )
    result1 = self_deploy(tmp_path, run_command=runner1, failure_alarm_threshold=3)
    assert result1.ok is False
    assert result1.deferred is False
    assert json.loads(counter_path.read_text(encoding="utf-8"))["consecutive_failures"] == 1
    assert _pending_sync_marker_path(layout.default_state_root(tmp_path)).exists()

    # Pass 2: no new commits, marker still pending, 0 runners -> retries sync, fails again.
    runner2, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # before HEAD
            RunResult(0, "Already up to date.\n", ""),  # pull
            RunResult(0, "def456\n", ""),  # after HEAD (unchanged)
            RunResult(1, "", "uv sync failed: resolution error"),  # uv sync fails
        ]
    )
    result2 = self_deploy(tmp_path, run_command=runner2, failure_alarm_threshold=3)
    assert result2.ok is False
    assert result2.deferred is False
    assert json.loads(counter_path.read_text(encoding="utf-8"))["consecutive_failures"] == 2
    assert query_events(state_path, kind="self_deploy_alarm") == []

    # Pass 3: no new commits, marker still pending, 2 runners now live -> real deferral.
    runner3, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # before HEAD
            RunResult(0, "Already up to date.\n", ""),  # pull
            RunResult(0, "def456\n", ""),  # after HEAD (unchanged)
        ]
    )
    result3 = self_deploy(tmp_path, run_command=runner3, failure_alarm_threshold=3)
    assert result3.ok is True
    assert result3.deferred is True
    assert json.loads(counter_path.read_text(encoding="utf-8"))["consecutive_failures"] == 2, (
        "a real deferral through the public entry point must not reset or increment the streak"
    )
    assert query_events(state_path, kind="self_deploy_alarm") == []

    # Pass 4: no new commits, marker still pending, 0 runners -> retries sync, fails
    # a third time -> crosses threshold=3, fires exactly one alarm.
    runner4, _ = _make_fake_runner(
        [
            RunResult(0, "def456\n", ""),  # before HEAD
            RunResult(0, "Already up to date.\n", ""),  # pull
            RunResult(0, "def456\n", ""),  # after HEAD (unchanged)
            RunResult(1, "", "uv sync failed: resolution error"),  # uv sync fails
        ]
    )
    result4 = self_deploy(tmp_path, run_command=runner4, failure_alarm_threshold=3)
    assert result4.ok is False
    assert json.loads(counter_path.read_text(encoding="utf-8"))["consecutive_failures"] == 3
    alarms = query_events(state_path, kind="self_deploy_alarm")
    assert len(alarms) == 1
    assert alarms[0]["payload"]["consecutive_failures"] == 3


def test_record_self_deploy_failure_streak_creates_state_dir_when_absent(
    tmp_path: Path,
) -> None:
    """``_record_self_deploy_failure_streak`` must create its own state
    directory before acquiring ``state_lock`` -- it must not rely on
    ``_log_self_deploy_outcome``'s ``log_event()`` call (which shares the same
    parent directory) having already created it as a side effect. Calling it
    directly, with no prior state-dir-creating call in this process, isolates
    the bug: without the pre-lock ``mkdir``, ``state_lock`` tries to create a
    sibling ``.lock`` file in a nonexistent directory and raises, which would
    violate ``self_deploy``'s documented never-raises contract.
    """
    counter_path = _self_deploy_failure_counter_path(tmp_path)
    assert not counter_path.parent.exists()

    failure = SelfDeployResult(ok=False, pulled=False, changed=False, synced=False, error="boom")
    _record_self_deploy_failure_streak(tmp_path, failure, threshold=3)  # must not raise

    assert counter_path.exists()
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_failures"] == 1


def test_self_deploy_failure_alarm_threshold_zero_disables(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """``failure_alarm_threshold<=0`` disables the alarm entirely, matching
    ``AutoMergeConfig.failed_attempt_alarm``'s "0 disables" convention.
    """
    state_path = _self_deploy_state_path(tmp_path)
    for _ in range(5):
        self_deploy(tmp_path, run_command=_fake_pull_failure_runner(), failure_alarm_threshold=0)
    assert query_events(state_path, kind="self_deploy_alarm") == []


def test_zero_pass_streak_fires_alarm_once_at_threshold(tmp_path: Path) -> None:
    """Issue #855: three consecutive zero-repo-pass cycles (with at least one
    repo configured) cross the default threshold (3) and fire exactly one
    ``supervisor_zero_pass_alarm`` event -- not one per cycle past the
    threshold -- and a subsequent cycle with repo_passes > 0 resets the
    counter so a later 0-pass streak starts counting from zero again instead
    of re-alarming immediately. Mirrors
    test_self_deploy_failure_streak_fires_alarm_once_at_threshold exactly.
    """
    state_path = _self_deploy_state_path(tmp_path)
    counter_path = _zero_pass_streak_counter_path(tmp_path)

    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)
    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)
    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []

    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)
    alarms = query_events(state_path, kind="supervisor_zero_pass_alarm")
    assert len(alarms) == 1
    assert alarms[0]["payload"]["consecutive_zero_pass_cycles"] == 3
    assert alarms[0]["payload"]["threshold"] == 3
    assert alarms[0]["level"] == "error"

    # A fourth consecutive zero-pass cycle must not fire a second alarm.
    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)
    assert len(query_events(state_path, kind="supervisor_zero_pass_alarm")) == 1

    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_zero_pass_cycles"] == 4

    # A cycle with repo_passes > 0 resets the streak.
    record_zero_pass_streak(tmp_path, repo_passes=2, repos_configured=True, threshold=3)
    counter_after_reset = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter_after_reset["consecutive_zero_pass_cycles"] == 0

    # Two more zero-pass cycles after the reset must not re-fire the alarm
    # yet (streak restarted at 0, threshold is 3).
    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)
    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)
    assert len(query_events(state_path, kind="supervisor_zero_pass_alarm")) == 1


def test_zero_pass_streak_never_fires_when_repos_not_configured(tmp_path: Path) -> None:
    """Issue #855 acceptance criterion 4: a fleet with zero configured repos
    is a configuration state, not an incident. Any number of zero-repo-pass
    cycles with ``repos_configured=False`` must never fire the alarm, and
    must not move the counter in either direction (the counter file is not
    even created).
    """
    state_path = _self_deploy_state_path(tmp_path)
    counter_path = _zero_pass_streak_counter_path(tmp_path)

    for _ in range(10):
        record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=False, threshold=3)

    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []
    assert not counter_path.exists()


def test_zero_pass_streak_never_fires_when_repo_passes_positive(tmp_path: Path) -> None:
    """A cycle that performs repo work (repo_passes > 0) never emits the
    alarm, however many times it repeats.
    """
    state_path = _self_deploy_state_path(tmp_path)

    for _ in range(10):
        record_zero_pass_streak(tmp_path, repo_passes=1, repos_configured=True, threshold=3)

    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []


def test_record_zero_pass_streak_creates_state_dir_when_absent(tmp_path: Path) -> None:
    """``record_zero_pass_streak`` must create its own state directory
    before acquiring ``state_lock`` -- it must not rely on some other call
    having already created it as a side effect. Calling it directly, with no
    prior state-dir-creating call in this process, isolates the bug: without
    the pre-lock ``mkdir``, ``state_lock`` tries to create a sibling
    ``.lock`` file in a nonexistent directory and raises, which would
    propagate out of run_fleet_supervise's post-loop bookkeeping. Mirrors
    test_record_self_deploy_failure_streak_creates_state_dir_when_absent.
    """
    counter_path = _zero_pass_streak_counter_path(tmp_path)
    assert not counter_path.parent.exists()

    # must not raise
    record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=3)

    assert counter_path.exists()
    counter = json.loads(counter_path.read_text(encoding="utf-8"))
    assert counter["consecutive_zero_pass_cycles"] == 1


def test_zero_pass_alarm_threshold_zero_disables(tmp_path: Path) -> None:
    """``threshold<=0`` disables the alarm entirely, matching
    ``self_deploy_failure_alarm``'s "0 disables" convention.
    """
    state_path = _self_deploy_state_path(tmp_path)
    for _ in range(5):
        record_zero_pass_streak(tmp_path, repo_passes=0, repos_configured=True, threshold=0)
    assert query_events(state_path, kind="supervisor_zero_pass_alarm") == []


def test_orchestrator_root_contains_pyproject_toml() -> None:
    """orchestrator_root() resolves to the orchestrator source tree root."""
    root = orchestrator_root()
    assert (root / "pyproject.toml").is_file()
    # It should be the directory that holds the source tree, not a subpackage.
    assert (root / "src" / "charlie_work" / "supervise.py").is_file()


# ---------------------------------------------------------------------------
# Orchestrator venv editable .pth self-heal tests (issue #447)
# ---------------------------------------------------------------------------


def _setup_fake_venv(
    repo_root: Path,
    *,
    wrong_target: Path | None = None,
) -> Path:
    """Create a fake venv under ``repo_root/.venv`` with one editable .pth file.

    If ``wrong_target`` is provided, the .pth points there (mismatch).  If
    ``None``, it points at ``repo_root/src`` (healthy).
    """
    site_packages = repo_root / ".venv" / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    pth = site_packages / "_editable_charlie_work.pth"
    target = wrong_target if wrong_target is not None else repo_root / "src"
    pth.write_text(str(target.resolve()) + "\n", encoding="utf-8")
    init_path = repo_root / "src" / "charlie_work" / "__init__.py"
    init_path.parent.mkdir(parents=True)
    init_path.write_text("", encoding="utf-8")
    return pth


def test_check_venv_noop_when_no_venv_found(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """_check_venv is a no-op when _find_venv_path cannot locate a venv."""
    monkeypatch.setattr(
        "charlie_work.supervise._find_venv_path",
        lambda _repo_root: None,
    )

    result = _check_venv(tmp_path)

    assert result == SelfDeployResult(
        ok=True,
        pulled=False,
        changed=False,
        synced=False,
        message="no orchestrator venv found; pth check skipped",
    )


def test_self_deploy_repairs_venv_pth_mismatch(
    tmp_path: Path,
) -> None:
    """A poisoned editable .pth is atomically rewritten to repo_root/src."""
    wrong_target = tmp_path / "wrong" / "src"
    pth_path = _setup_fake_venv(tmp_path, wrong_target=wrong_target)

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "abc123\n", ""),  # after HEAD (no change)
        ],
    )

    result = self_deploy(tmp_path, run_command=runner)

    assert result.ok is True
    assert result.venv_repaired is True
    assert result.pulled is True
    assert result.synced is False
    assert result.from_sha == "abc123"
    assert result.to_sha == "abc123"
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["git", "pull", "--ff-only", "origin", "main"],
        ["git", "rev-parse", "HEAD"],
    ]
    assert pth_path.read_text(encoding="utf-8").strip() == str((tmp_path / "src").resolve())


def test_self_deploy_repairs_venv_pth_mismatch_with_runners_active(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A .pth rewrite is not exe-locked and succeeds while charlie.exe is held open."""
    msvcrt = pytest.importorskip("msvcrt")

    wrong_target = tmp_path / "wrong" / "src"
    pth_path = _setup_fake_venv(tmp_path, wrong_target=wrong_target)

    # Simulate a live orchestrator process image by holding an exclusive byte-range
    # lock on a charlie.exe stand-in in the venv. The .pth rewrite must still succeed.
    charlie_exe = tmp_path / ".venv" / "Scripts" / "charlie.exe"
    charlie_exe.parent.mkdir(parents=True, exist_ok=True)
    charlie_exe.write_bytes(b"MZ fake executable content")
    handle = charlie_exe.open("r+b", encoding=None)

    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (2, []),
    )

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "abc123\n", ""),  # after HEAD (no change)
        ],
    )

    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        result = self_deploy(tmp_path, run_command=runner)

        assert result.ok is True
        assert result.venv_repaired is True
        assert result.pulled is True
        assert result.synced is False
        assert pth_path.read_text(encoding="utf-8").strip() == str((tmp_path / "src").resolve())
        assert [c[0] for c in calls] == [
            ["git", "rev-parse", "HEAD"],
            ["git", "pull", "--ff-only", "origin", "main"],
            ["git", "rev-parse", "HEAD"],
        ]
    finally:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def test_self_deploy_venv_repair_failure_is_non_fatal(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A failed .pth repair is returned as a non-fatal error value."""
    wrong_target = tmp_path / "wrong" / "src"
    _setup_fake_venv(tmp_path, wrong_target=wrong_target)
    monkeypatch.setattr(
        "charlie_work.supervise._repair_venv_pth",
        lambda _repo_root, _venv_path: (False, "Access is denied"),
    )

    runner, calls = _make_fake_runner([RunResult(0, "abc123\n", "")])

    result = self_deploy(tmp_path, run_command=runner)

    assert result.ok is False
    assert result.venv_repaired is False
    assert result.pulled is False
    assert result.changed is False
    assert result.synced is False
    assert result.error is not None
    assert "Access is denied" in result.error
    assert not calls


# ---------------------------------------------------------------------------
# _repair_lossless_pull_blockers -- real git repos, no mocked git
#
# Every test here builds a real ``origin`` repo and a real ``clone`` parked
# behind it, then fetches (never pulls) so ``origin/main`` genuinely points
# past HEAD before the repair function is asked anything. That mirrors the
# actual precondition the function itself checks (a real pending
# fast-forward), not just a return-value stand-in for one.
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_origin(origin_root: Path, *, autocrlf: bool = False) -> None:
    origin_root.mkdir(parents=True, exist_ok=True)
    _git(origin_root, "init", "--initial-branch=main")
    _git(origin_root, "config", "user.email", "test@example.test")
    _git(origin_root, "config", "user.name", "Test User")
    _git(origin_root, "config", "core.autocrlf", "true" if autocrlf else "false")
    (origin_root / "README.md").write_bytes(b"hello\n")
    _git(origin_root, "add", "README.md")
    _git(origin_root, "commit", "-m", "initial commit")


def _clone(origin_root: Path, clone_root: Path, *, autocrlf: bool = False) -> None:
    _git(origin_root.parent, "clone", str(origin_root), str(clone_root))
    _git(clone_root, "config", "user.email", "test@example.test")
    _git(clone_root, "config", "user.name", "Test User")
    _git(clone_root, "config", "core.autocrlf", "true" if autocrlf else "false")


def _counting_git_runner() -> tuple[Callable[..., RunResult], list[list[str]]]:
    """Wrap the real ``run_captured`` so calls are recorded but still actually
    executed -- proves a precondition short-circuited before doing
    unnecessary work, without mocking git itself."""
    calls: list[list[str]] = []

    def runner(command: list[str], *, cwd: Path, timeout_seconds: int) -> RunResult:
        calls.append(command)
        return run_captured(command, cwd=cwd, timeout_seconds=timeout_seconds)

    return runner, calls


def test_repair_lossless_pull_blockers_clears_both_classes_and_pull_succeeds(
    tmp_path: Path,
) -> None:
    """One blocker of each refusal class, both byte-identical to what the
    incoming commit would write: both are cleared, retained is empty, and a
    real ``git pull --ff-only`` immediately after succeeds -- the actual
    outcome this feature exists to restore.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    (origin_root / "tracked.txt").write_bytes(b"original content\n")
    _git(origin_root, "add", "tracked.txt")
    _git(origin_root, "commit", "-m", "add tracked.txt")
    _clone(origin_root, clone_root)

    # Commit B on origin: modifies tracked.txt and adds a new file.
    (origin_root / "tracked.txt").write_bytes(b"incoming content\n")
    (origin_root / "new_untracked.txt").write_bytes(b"new content\n")
    _git(origin_root, "add", "tracked.txt", "new_untracked.txt")
    _git(origin_root, "commit", "-m", "advance origin")
    _git(clone_root, "fetch", "origin")

    # Local blockers whose content already matches what B would write.
    (clone_root / "new_untracked.txt").write_bytes(b"new content\n")  # untracked shadow
    (clone_root / "tracked.txt").write_bytes(b"incoming content\n")  # modified, unstaged

    repair = _repair_lossless_pull_blockers(clone_root, run_command=run_captured, timeout=30)

    assert repair.cleared == ("new_untracked.txt", "tracked.txt")
    assert repair.retained == ()
    assert repair.acted is True

    pull = subprocess.run(
        ["git", "pull", "--ff-only", "origin", "main"],
        cwd=clone_root,
        capture_output=True,
        text=True,
    )
    assert pull.returncode == 0, pull.stderr
    assert (clone_root / "tracked.txt").read_bytes() == b"incoming content\n"
    assert (clone_root / "new_untracked.txt").read_bytes() == b"new content\n"


def test_repair_lossless_pull_blockers_mixed_clears_lossless_retains_genuine_conflict(
    tmp_path: Path,
) -> None:
    """A lossless untracked blocker and a lossless modified blocker are
    cleared; a modified blocker whose local content genuinely differs from
    both HEAD and the incoming commit is retained, untouched, on disk.
    Partial clearing is the documented behaviour, not a bug to fix.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    (origin_root / "tracked.txt").write_bytes(b"original content\n")
    (origin_root / "divergent.txt").write_bytes(b"original divergent\n")
    _git(origin_root, "add", "tracked.txt", "divergent.txt")
    _git(origin_root, "commit", "-m", "add tracked files")
    _clone(origin_root, clone_root)

    (origin_root / "tracked.txt").write_bytes(b"incoming content\n")
    (origin_root / "divergent.txt").write_bytes(b"incoming divergent v2\n")
    (origin_root / "new_untracked.txt").write_bytes(b"new content\n")
    _git(origin_root, "add", "tracked.txt", "divergent.txt", "new_untracked.txt")
    _git(origin_root, "commit", "-m", "advance origin")
    _git(clone_root, "fetch", "origin")

    (clone_root / "new_untracked.txt").write_bytes(b"new content\n")
    (clone_root / "tracked.txt").write_bytes(b"incoming content\n")
    # Genuinely different from BOTH HEAD's version and the incoming version.
    (clone_root / "divergent.txt").write_bytes(b"local unique edit, not incoming\n")

    repair = _repair_lossless_pull_blockers(clone_root, run_command=run_captured, timeout=30)

    assert repair.cleared == ("new_untracked.txt", "tracked.txt")
    assert repair.retained == ("divergent.txt",)
    assert (clone_root / "divergent.txt").read_bytes() == b"local unique edit, not incoming\n"


def test_repair_lossless_pull_blockers_diverged_clears_nothing(tmp_path: Path) -> None:
    """A local commit origin lacks means this is not a pending fast-forward at
    all -- the precondition must refuse before touching anything, even a
    blocker that would otherwise be provably lossless. The call trace pins
    that the refusal happens specifically at the ancestor check (not, say,
    an accidental early return that would also mask a real bug), and the
    blocker's untouched content proves ``git checkout HEAD --`` never ran:
    if it had, it would restore HEAD's ("local-only" commit's inherited)
    version, not leave our injected content in place.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    (origin_root / "tracked.txt").write_bytes(b"original content\n")
    _git(origin_root, "add", "tracked.txt")
    _git(origin_root, "commit", "-m", "add tracked.txt")
    _clone(origin_root, clone_root)

    # Origin advances...
    (origin_root / "tracked.txt").write_bytes(b"incoming content\n")
    _git(origin_root, "add", "tracked.txt")
    _git(origin_root, "commit", "-m", "advance origin")
    _git(clone_root, "fetch", "origin")

    # ...while the clone independently commits something origin never saw.
    (clone_root / "local_only.txt").write_bytes(b"local commit content\n")
    _git(clone_root, "add", "local_only.txt")
    _git(clone_root, "commit", "-m", "local-only commit")

    # A blocker that WOULD be lossless if the precondition were skipped.
    (clone_root / "tracked.txt").write_bytes(b"incoming content\n")

    runner, calls = _counting_git_runner()
    repair = _repair_lossless_pull_blockers(clone_root, run_command=runner, timeout=30)

    assert repair == PullBlockerRepair()
    assert len(calls) == 3
    assert calls[0] == ["git", "rev-parse", "HEAD"]
    assert calls[1] == ["git", "rev-parse", "origin/main"]
    assert calls[2][:3] == ["git", "merge-base", "--is-ancestor"]
    # Not reverted to HEAD's inherited content -- proves checkout never ran.
    assert (clone_root / "tracked.txt").read_bytes() == b"incoming content\n"


def test_repair_lossless_pull_blockers_up_to_date_clears_nothing(tmp_path: Path) -> None:
    """HEAD already at ``origin/main`` -- there is no pending fast-forward, so
    the precondition refuses immediately. The call trace pins that only the
    two rev-parse calls happen: no ancestor check, no blocker computation at
    all, proving this is the equality short-circuit specifically and not
    some other path that happens to also land on an empty result.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    _clone(origin_root, clone_root)

    runner, calls = _counting_git_runner()
    repair = _repair_lossless_pull_blockers(clone_root, run_command=runner, timeout=30)

    assert repair == PullBlockerRepair()
    assert calls == [
        ["git", "rev-parse", "HEAD"],
        ["git", "rev-parse", "origin/main"],
    ]


def test_repair_lossless_pull_blockers_crlf_worktree_matches_lf_blob(tmp_path: Path) -> None:
    """With ``core.autocrlf=true`` and CRLF bytes in the worktree, a file whose
    blob is LF-only must still be recognised as lossless. ``git hash-object``
    must run with filters ON (the default, never ``--no-filters``) or this
    sweep goes silently inert on every Windows checkout with autocrlf enabled
    -- a real trap on this machine, not a hypothetical one.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root, autocrlf=False)
    _clone(origin_root, clone_root, autocrlf=True)

    # Origin adds a file with LF-only line endings.
    (origin_root / "crlf_target.txt").write_bytes(b"line one\nline two\n")
    _git(origin_root, "add", "crlf_target.txt")
    _git(origin_root, "commit", "-m", "advance origin with LF file")
    _git(clone_root, "fetch", "origin")

    # The clone's untracked copy has the SAME content but CRLF line endings --
    # what a Windows checkout with autocrlf=true would actually produce.
    (clone_root / "crlf_target.txt").write_bytes(b"line one\r\nline two\r\n")

    repair = _repair_lossless_pull_blockers(clone_root, run_command=run_captured, timeout=30)

    assert repair.cleared == ("crlf_target.txt",)
    assert repair.retained == ()
    assert not (clone_root / "crlf_target.txt").exists()


def test_repair_lossless_pull_blockers_staged_unique_work_is_retained(tmp_path: Path) -> None:
    """Worktree content that happens to equal the incoming blob is not enough
    when the INDEX holds something else: staging is where the unique work
    actually lives, and ``git checkout HEAD --`` would destroy it along with
    the worktree copy. Must be retained, and the staged content must survive
    untouched.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    (origin_root / "staged.txt").write_bytes(b"ancestor\n")
    _git(origin_root, "add", "staged.txt")
    _git(origin_root, "commit", "-m", "add staged.txt")
    _clone(origin_root, clone_root)

    (origin_root / "staged.txt").write_bytes(b"incoming staged content\n")
    _git(origin_root, "add", "staged.txt")
    _git(origin_root, "commit", "-m", "advance origin")
    _git(clone_root, "fetch", "origin")

    # Stage a locally-authored, unique change...
    (clone_root / "staged.txt").write_bytes(b"staged unique work\n")
    _git(clone_root, "add", "staged.txt")
    # ...then overwrite the WORKTREE (not the index) to match what origin
    # would write -- content-identical worktree, divergent index.
    (clone_root / "staged.txt").write_bytes(b"incoming staged content\n")

    repair = _repair_lossless_pull_blockers(clone_root, run_command=run_captured, timeout=30)

    assert repair.cleared == ()
    assert repair.retained == ("staged.txt",)
    staged_index_content = subprocess.run(
        ["git", "show", ":staged.txt"],
        cwd=clone_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert staged_index_content == "staged unique work\n"


def test_repairable_blocker_path_refuses_venv_and_escaping_paths(tmp_path: Path) -> None:
    """The path gate every candidate runs through before content is even
    considered: ``.venv/*`` is refused unconditionally (repair must never
    touch the running interpreter), a path escaping ``repo_root`` is
    refused, and an ordinary contained path is the positive control proving
    the gate isn't just returning ``False`` for everything.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    assert _repairable_blocker_path(repo_root, ".venv/pyvenv.cfg") is False
    assert _repairable_blocker_path(repo_root, ".venv/Lib/site-packages/foo.py") is False
    assert _repairable_blocker_path(repo_root, "../outside.txt") is False
    assert _repairable_blocker_path(repo_root, "ordinary.txt") is True


def test_repair_lossless_pull_blockers_refuses_venv_path_even_when_lossless(
    tmp_path: Path,
) -> None:
    """A ``.venv/...`` blocker that is byte-identical to the incoming blob
    must still be retained and left on disk -- content-identity is not
    sufficient license to touch the orchestrator's own interpreter tree.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    _clone(origin_root, clone_root)

    (origin_root / ".venv").mkdir()
    (origin_root / ".venv" / "marker.txt").write_bytes(b"venv marker\n")
    _git(origin_root, "add", ".venv/marker.txt")
    _git(origin_root, "commit", "-m", "advance origin with venv marker")
    _git(clone_root, "fetch", "origin")

    (clone_root / ".venv").mkdir()
    (clone_root / ".venv" / "marker.txt").write_bytes(b"venv marker\n")  # byte-identical

    repair = _repair_lossless_pull_blockers(clone_root, run_command=run_captured, timeout=30)

    assert repair.cleared == ()
    assert repair.retained == (".venv/marker.txt",)
    assert (clone_root / ".venv" / "marker.txt").read_bytes() == b"venv marker\n"


def test_pull_blocker_repair_describe_and_acted() -> None:
    """``.acted`` reflects whether anything was cleared (drives the
    self-deploy retry decision); ``describe()`` names cleared and retained
    paths for the escalation message an operator reads.
    """
    empty = PullBlockerRepair()
    assert empty.acted is False
    assert empty.describe() == ""

    cleared_only = PullBlockerRepair(cleared=("a.txt", "b.txt"))
    assert cleared_only.acted is True
    description = cleared_only.describe()
    assert "a.txt" in description
    assert "b.txt" in description
    assert "auto-cleared" in description

    mixed = PullBlockerRepair(cleared=("a.txt",), retained=("c.txt",))
    assert mixed.acted is True
    mixed_description = mixed.describe()
    assert "a.txt" in mixed_description
    assert "c.txt" in mixed_description
    assert "still need a human" in mixed_description

    # Retained-only: something needs a human, but nothing on disk changed --
    # `.acted` must track "did disk change", not "is there anything to report".
    retained_only = PullBlockerRepair(retained=("d.txt",))
    assert retained_only.acted is False
    assert "d.txt" in retained_only.describe()


def test_pull_blocker_repair_describe_bounds_long_path_lists() -> None:
    """``describe()`` names at most ``_BLOCKER_NAMES_IN_MESSAGE`` paths per class.

    The string reaches ``SelfDeployResult.error`` and is re-rendered into a
    ``self_deploy_failed`` payload once per retry for the whole duration of a
    wedge, so an unbounded join is a slow leak and an unreadable operator
    message. The exact COUNT must survive truncation -- that is the number an
    operator acts on -- so it is asserted separately from the names.
    """
    many = tuple(f"f{i:02d}.txt" for i in range(25))
    description = PullBlockerRepair(cleared=many).describe()

    assert "25" in description, "the full count must survive truncation"
    assert "f00.txt" in description
    assert f"f{_BLOCKER_NAMES_IN_MESSAGE - 1:02d}.txt" in description
    assert f"f{_BLOCKER_NAMES_IN_MESSAGE:02d}.txt" not in description, "not truncated"
    assert f"+{25 - _BLOCKER_NAMES_IN_MESSAGE} more" in description

    # Boundary: exactly at the limit is rendered in full, with no "+N more".
    exact = tuple(f"g{i:02d}.txt" for i in range(_BLOCKER_NAMES_IN_MESSAGE))
    exact_description = PullBlockerRepair(retained=exact).describe()
    assert f"g{_BLOCKER_NAMES_IN_MESSAGE - 1:02d}.txt" in exact_description
    assert "more" not in exact_description


def test_self_deploy_end_to_end_repairs_lossless_blocker_and_retries_pull(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """Wiring test through the public ``self_deploy`` entry point: a real
    clone with exactly one lossless blocker ends up ``ok`` via the single
    retry, rather than surfacing the original pull failure -- the actual
    outcome this feature exists to produce end to end, not just at the
    ``_repair_lossless_pull_blockers`` unit level.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    _clone(origin_root, clone_root)

    (origin_root / "new_untracked.txt").write_bytes(b"new content\n")
    _git(origin_root, "add", "new_untracked.txt")
    _git(origin_root, "commit", "-m", "advance origin")
    _git(clone_root, "fetch", "origin")

    (clone_root / "new_untracked.txt").write_bytes(b"new content\n")

    result = self_deploy(clone_root, run_command=run_captured)

    assert result.ok is True
    assert result.pulled is True
    assert (clone_root / "new_untracked.txt").read_bytes() == b"new content\n"

    # A repair that succeeds leaves an ordinary `self_deploy_succeeded` behind
    # and is otherwise indistinguishable from a pass where nothing was wrong.
    # The audit record is the only thing that keeps a RECURRING cause visible,
    # so assert it on the success path specifically -- the failure path already
    # names the paths in the error string.
    cleared_events = query_events(
        _self_deploy_state_path(clone_root), kind="self_deploy_blockers_cleared"
    )
    assert len(cleared_events) == 1, "successful repair must still be recorded"
    payload = cleared_events[0]["payload"]
    assert payload["cleared"] == ["new_untracked.txt"]
    assert payload["retained"] == []
    assert payload["pull_ok_after_retry"] is True


def test_self_deploy_records_nothing_cleared_when_no_repair_was_needed(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """Negative control for the audit record above.

    Without this, the assertion that a repair emits exactly one
    ``self_deploy_blockers_cleared`` is equally consistent with the event
    firing on *every* deploy. An ordinary clean fast-forward must emit none.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    _clone(origin_root, clone_root)

    (origin_root / "clean.txt").write_bytes(b"clean\n")
    _git(origin_root, "add", "clean.txt")
    _git(origin_root, "commit", "-m", "advance origin")
    _git(clone_root, "fetch", "origin")

    result = self_deploy(clone_root, run_command=run_captured)

    assert result.ok is True
    assert (
        query_events(_self_deploy_state_path(clone_root), kind="self_deploy_blockers_cleared")
        == []
    )


# ---------------------------------------------------------------------------
# _pull_ci_fleet_sibling unit tests (issue #552 deploy-clone half)
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_log_events(monkeypatch: Any) -> list[tuple[Path, str, dict[str, Any]]]:
    """Capture (state_path, kind, payload) tuples instead of writing to events.db."""
    calls: list[tuple[Path, str, dict[str, Any]]] = []

    def _fake_log_event(
        state_path: Path, kind: str, payload: dict[str, Any], **_kwargs: Any
    ) -> None:
        calls.append((state_path, kind, payload))

    monkeypatch.setattr("charlie_work.supervise.log_event", _fake_log_event)
    return calls


def _fake_declared_root(monkeypatch: Any, sibling_src: Path | None) -> None:
    """Patch declared_ci_fleet_root at its source module.

    ``_pull_ci_fleet_sibling`` imports it lazily inside its body
    (``from .ci_fleet_anchor import declared_ci_fleet_root``), so it must be
    patched at ``charlie_work.ci_fleet_anchor``, not at ``charlie_work.supervise``.
    """
    monkeypatch.setattr("charlie_work.ci_fleet_anchor.declared_ci_fleet_root", lambda: sibling_src)


def test_pull_ci_fleet_sibling_happy_path(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """Clean main sibling, sha moves -> one ok/changed events.db entry."""
    sibling = tmp_path / "ci-fleet"
    _fake_declared_root(monkeypatch, sibling / "src")

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "main\n", ""),  # branch
            RunResult(0, "", ""),  # status --porcelain (clean)
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "def456\n", ""),  # after HEAD
        ]
    )

    outcome = _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    assert outcome is None
    assert len(captured_log_events) == 1
    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is True
    assert payload["changed"] is True
    assert payload["from_sha"] == "abc123"
    assert payload["to_sha"] == "def456"
    assert payload["from_sha"] != payload["to_sha"]
    assert len(calls) == 5
    assert all(c[1] == sibling for c in calls)


def test_pull_ci_fleet_sibling_unchanged_pull(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """Pull succeeds but the sha does not move -> ok True, changed False."""
    sibling = tmp_path / "ci-fleet"
    _fake_declared_root(monkeypatch, sibling / "src")

    runner, _ = _make_fake_runner(
        [
            RunResult(0, "main\n", ""),
            RunResult(0, "", ""),
            RunResult(0, "abc123\n", ""),
            RunResult(0, "Already up to date.\n", ""),
            RunResult(0, "abc123\n", ""),
        ]
    )

    _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is True
    assert payload["changed"] is False
    assert payload["from_sha"] == payload["to_sha"] == "abc123"


def test_pull_ci_fleet_sibling_skips_non_main_branch(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """A sibling not on main is skipped -- pull is never issued.

    This pins the fail-safe precondition: only two canned responses (branch,
    status) are supplied, so if the branch check were deleted the function
    would try to consume a third response and the fake runner's ``pop(0)``
    would raise ``IndexError``, failing the test.
    """
    sibling = tmp_path / "ci-fleet"
    _fake_declared_root(monkeypatch, sibling / "src")

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "feature-branch\n", ""),  # branch
            RunResult(0, "", ""),  # status --porcelain
        ]
    )

    _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is False
    assert "feature-branch" in payload["skipped_reason"]
    assert all(c[0] != ["git", "pull", "--ff-only", "origin", "main"] for c in calls)
    assert len(calls) == 2


def test_pull_ci_fleet_sibling_skips_dirty_tree(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """A dirty sibling tree is skipped -- pull is never issued.

    Only two canned responses are supplied, pinning the precondition the same
    way as the non-main-branch case above.
    """
    sibling = tmp_path / "ci-fleet"
    _fake_declared_root(monkeypatch, sibling / "src")

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "main\n", ""),
            RunResult(0, " M some_file.py\n", ""),  # dirty
        ]
    )

    _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is False
    assert payload["skipped_reason"] == "sibling tree is dirty"
    assert all(c[0] != ["git", "pull", "--ff-only", "origin", "main"] for c in calls)
    assert len(calls) == 2


def test_pull_ci_fleet_sibling_skips_when_no_declared_root(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """No declared ci-fleet source -- skipped, no git commands issued at all."""
    _fake_declared_root(monkeypatch, None)

    runner, calls = _make_fake_runner([])

    _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is False
    assert "no declared" in payload["skipped_reason"]
    assert calls == []


def test_pull_ci_fleet_sibling_pull_failure_is_non_fatal(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """A failed pull is recorded with an error, event still emitted, no raise."""
    sibling = tmp_path / "ci-fleet"
    _fake_declared_root(monkeypatch, sibling / "src")

    runner, _ = _make_fake_runner(
        [
            RunResult(0, "main\n", ""),
            RunResult(0, "", ""),
            RunResult(0, "abc123\n", ""),
            RunResult(1, "", "fatal: could not read from remote repository."),
            RunResult(0, "abc123\n", ""),
        ]
    )

    outcome = _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    assert outcome is None
    assert len(captured_log_events) == 1
    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is False
    assert payload.get("error")


def test_pull_ci_fleet_sibling_declared_root_raises_is_caught(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """An exception from declared_ci_fleet_root is caught, logged, never propagates."""

    def _boom() -> Path | None:
        raise RuntimeError("boom")

    monkeypatch.setattr("charlie_work.ci_fleet_anchor.declared_ci_fleet_root", _boom)

    runner, calls = _make_fake_runner([])

    outcome = _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    assert outcome is None
    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is False
    assert "RuntimeError" in payload["error"]
    assert calls == []


def test_self_deploy_does_not_pull_ci_fleet_sibling_by_default(
    tmp_path: Path, monkeypatch: Any, no_fleet_live_sessions: None
) -> None:
    """self_deploy's default pull_ci_fleet=False never touches the sibling machinery.

    Patches ``_pull_ci_fleet_sibling`` with a fail-if-called stub and drives
    ``self_deploy`` (no ``pull_ci_fleet`` kwarg -> default False) with a
    successful, code-only pull -- confirming the sibling path is gated even
    when the orchestrator's own pull succeeds.
    """

    def _fail_if_called(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("_pull_ci_fleet_sibling must not be called when pull_ci_fleet=False")

    monkeypatch.setattr("charlie_work.supervise._pull_ci_fleet_sibling", _fail_if_called)

    runner, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "def456\n", ""),  # after HEAD
            RunResult(0, "src/foo.py\n", ""),  # diff (code-only)
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)

    assert result.ok is True
    assert result.pulled is True
    assert result.changed is True
