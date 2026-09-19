"""Dispatch recovery, reaping, and provider-throttle deferrals.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the recovery seam of ``dispatch()`` -- dead-worker recovery and aborts,
stale-orphan flag clearing, blocked-environment reap counters, failed-retry
escalation, stall detection, and throttle-window deferrals. Shared fakes and
helpers in ``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from _dispatch_fixtures import _make_stalled_sidecar
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import (
    _blocked_env_timestamps,
    _wg,
)
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import (
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    set_throttled_until,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp


def test_dispatch_recovers_dead_worker_without_open_pr(tmp_path: Path) -> None:
    """Issue #5: a dead worker with no open PR becomes dispatchable again."""
    from charlie_work.devin_shell import SessionRecord

    config = OrchestratorConfig(devin=DevinConfig())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Mark the default PR as closed so the issue is considered dispatchable.
    fake_gh.prs[0]["state"] = "CLOSED"
    # Simulate a prior dispatch that crashed before PR opened
    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "title": "Test issue",
        "url": "https://github.com/test/repo/issues/123",
    }
    save_state(paths.state_file, seed)
    # Create a session record with a dead PID
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    # Spawn and immediately wait for a short-lived process to get a dead PID
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process.wait()  # Ensure it's dead
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123",
        worktree_path="/tmp/wt/issue-123",
        prompt_path="p.md",
        command=("x",),
        pid=process.pid,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
    )
    # Write the session record manually (mirrors internal _write_json pattern)
    sidecar_path = sessions_dir / f"issue-{123}.json"
    tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    tmp.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    tmp.replace(sidecar_path)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=3)

    # The issue should be re-dispatched since the worker is dead and there's no open PR
    assert result.data["attempted_count"] == 1
    assert result.data["selected_count"] == 1
    assert 123 in [s["issue_number"] for s in result.data["sessions"]]


def test_dispatch_does_not_recover_dead_worker_with_open_pr(tmp_path: Path) -> None:
    """Issue #5: a dead worker with an open PR is NOT re-dispatched (mid-review)."""
    from charlie_work.devin_shell import SessionRecord

    config = OrchestratorConfig(devin=DevinConfig())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Override prs to return an open PR for this issue
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix issue 123",
            "headRefName": "agent/issue-123",
            "url": "https://github.com/test/repo/pull/456",
            "state": "OPEN",
            "isCrossRepository": False,
        }
    ]
    # Simulate a prior dispatch that crashed after PR opened
    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "title": "Test issue",
        "url": "https://github.com/test/repo/issues/123",
    }
    save_state(paths.state_file, seed)
    # Create a session record with a dead PID
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    # Spawn and immediately wait for a short-lived process to get a dead PID
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process.wait()  # Ensure it's dead
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123",
        worktree_path="/tmp/wt/issue-123",
        prompt_path="p.md",
        command=("x",),
        pid=process.pid,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
    )
    # Write the session record manually (mirrors internal _write_json pattern)
    sidecar_path = sessions_dir / f"issue-{123}.json"
    tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    tmp.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    tmp.replace(sidecar_path)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "OPEN"
    result = app.dispatch(limit=3)

    # The issue should NOT be re-dispatched since there's an open PR
    assert result.data["attempted_count"] == 0


def test_dispatch_recovery_aborts_for_live_worker_and_restores_in_progress(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #282: a recovery redispatch that detects a live worker must abort
    and restore the in-progress label, not clobber the worktree."""

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=4242,
            started_at="2026-07-02T00:00:00Z",
            log_path=str(tmp_path / "log"),
            error="pid_alive",
            failure_kind="live_worker_redispatch_averted",
            process_start_time=1_234_567.0,
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    # Issue #523: the live-worker slot count now verifies the recorded PID is
    # actually alive at the OS level (is_pid_alive + process_start_time).
    # Stub is_pid_alive so the dispatch-side result PID is treated as live,
    # and stub _worker_pid_alive so the state.json worker_pid does not block
    # candidate selection (the issue must be selectable to reach dispatch).
    monkeypatch.setattr("charlie_work.workflow.is_pid_alive", lambda pid, start: True)
    monkeypatch.setattr("charlie_work.workflow._worker_pid_alive", lambda entry: False)
    config = OrchestratorConfig(worker=WorkerRoleConfig(harness="claude-code"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.pr_list = lambda: []

    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "branch_name": "agent/issue-123-fix-search",
        "worker_pid": 4242,
        "worker_process_start_time": 1_234_567.0,
        "title": "Fix search",
        "url": "https://example.test/issues/123",
    }
    save_state(paths.state_file, seed)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"
    assert (123, "agent:in-progress") in fake_gh.labels_added
    assert any(
        event["kind"] == "live_worker_redispatch_averted"
        and event["payload"]["issue_number"] == 123
        for event in state.get("events", [])
    )


def test_dispatch_survives_worker_census_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #646 regression guard: ``_log_worker_census`` is invoked
    unconditionally as the first statement of ``dispatch()`` so every dispatch
    path logs it. That diagnostic must never be the reason a whole dispatch
    pass aborts -- a torn sidecar read (or any other unexpected failure in the
    census sweep) is *more* likely, not less, during the exact
    high-concurrency moment this census exists to diagnose.

    This behavior already regressed once earlier in this branch's own commit
    history (the call was unguarded before commit f891866), so pin it with a
    test: force the census to raise, then assert dispatch still runs to
    completion and selects a worker, and that the failure is surfaced as a
    warning rather than propagating or being silently swallowed.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Make the issue dispatchable (default fixture ships an open tracked PR).
    app.gh.prs[0]["state"] = "CLOSED"

    def _boom(_sessions_dir: Path) -> None:
        raise OSError("simulated torn sidecar read")

    monkeypatch.setattr("charlie_work.workflow._log_worker_census", _boom)

    with caplog.at_level(logging.WARNING, logger="charlie_work.workflow"):
        result = app.dispatch(limit=1)

    # dispatch() must not have raised: it returns a normal ok result and, more
    # importantly, actually did its work (selected a worker, wrote the prompt)
    # -- proving it continued past the failing census call rather than aborting
    # at the first statement.
    assert result.ok is True
    assert result.data["selected_count"] == 1
    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    assert prompt_path.exists()
    assert (123, "agent:queued") in fake_gh.labels_added

    # The census failure must be surfaced as a warning (not silently swallowed
    # and not propagated), so an operator scanning logs can still see it.
    assert any("worker census failed" in record.getMessage() for record in caplog.records), [
        r.getMessage() for r in caplog.records
    ]


def test_dispatch_clears_stale_orphan_flagged_at(tmp_path: Path) -> None:
    """Issue #259 review: a fresh dispatch must clear a stale orphan flag."""
    config = OrchestratorConfig(devin=DevinConfig())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Close the default PR so the issue is dispatchable.
    fake_gh.prs[0]["state"] = "CLOSED"
    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "orphan_flagged_at": "2024-01-01T00:00:00Z",
        "title": "Test issue",
        "url": "https://github.com/test/repo/issues/123",
    }
    save_state(paths.state_file, seed)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch()

    assert result.data["attempted_count"] == 1
    assert result.data["selected_count"] == 1
    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry.get("status") == "manifest_written"
    assert "orphan_flagged_at" not in entry


def test_dispatch_fresh_worktree_foreign_writer_does_not_increment_dispatch_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1393: a fresh dispatch that fails at launch with
    worktree_foreign_writer must NOT increment the dispatch_failed counter.
    Instead it uses blocked_environment_at and escalates with
    dispatch_blocked_environment after the cap.
    """
    from charlie_work.adapters import SessionDispatchResult

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Avoid the open-PR exclusion by closing the default fixture PR.
    fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=False,
                error="worktree C:\\wt is a foreign checkout",
                failure_kind="worktree_foreign_writer",
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    # First blocked launch: blocked_environment_at grows, dispatch_failed_at stays empty.
    result1 = app.dispatch(limit=1)
    assert result1.ok is False
    assert result1.data["failed_count"] == 1
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatch_failed"
    assert state["issues"]["123"].get("dispatch_failed_at") is None
    assert len(state["issues"]["123"].get("blocked_environment_at", [])) == 1

    # Second blocked launch: still under the cap.
    result2 = app.dispatch(limit=1)
    assert result2.ok is False
    state = load_state(paths.state_file)
    assert state["issues"]["123"].get("dispatch_failed_at") is None
    assert len(state["issues"]["123"].get("blocked_environment_at", [])) == 2

    # Third blocked launch: exceeds the cap (max_auto_redispatch=2).
    result3 = app.dispatch(limit=1)
    assert result3.ok is False
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "dispatch_blocked_environment"
    assert state["issues"]["123"].get("dispatch_failed_at") is None


def test_dispatch_fresh_blocked_environment_reap_resets_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1423: at the fresh-dispatch blocked-environment cap exhaustion,
    a successful foreign-writer reap resets ``blocked_environment_at`` and
    records the reap in ``foreign_writer_reaps`` instead of escalating."""
    from charlie_work.adapters import SessionDispatchResult

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        watchdog=WatchdogConfig(
            max_auto_redispatch=2, redispatch_window_minutes=240, max_foreign_writer_reaps=2
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Pre-seed: two prior blocked passes (at the cap), so this pass's failure
    # pushes len(blocked_environment_at) to 3 > max_auto_redispatch(2) and
    # enters the reap branch.
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "dispatch_failed",
            "blocked_environment_at": _blocked_env_timestamps(2),
        }
        save_state(paths.state_file, state)

    wt_path = tmp_path / "wt-foreign"

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=False,
                error="worktree has a live foreign writer",
                failure_kind="worktree_foreign_writer",
                pid=1234,
                worktree_path=str(wt_path),
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)
    reap_calls: list[int] = []

    def _fake_reap(failed_result, _config, _state_file, issue_number, _sessions_dir=None):
        reap_calls.append(issue_number)
        return True

    monkeypatch.setattr("charlie_work.workflow._try_reap_blocked_foreign_writer", _fake_reap)

    result = app.dispatch(limit=1)
    assert result.ok is False
    state = load_state(paths.state_file)
    # Reap succeeded: status is dispatch_failed (NOT escalated), counter reset.
    assert state["issues"]["123"]["status"] == "dispatch_failed"
    assert state["issues"]["123"]["blocked_environment_at"] == []
    # The reap was recorded so a persistently-blocked worktree eventually escalates.
    assert len(state["issues"]["123"].get("foreign_writer_reaps", [])) == 1
    assert reap_calls == [123]


def test_dispatch_fresh_blocked_environment_reap_cap_escalates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1423 review: once ``foreign_writer_reaps`` reaches
    ``max_foreign_writer_reaps``, the cap-exhaustion site escalates instead of
    reaping again — a persistently-blocked worktree cannot loop forever
    between reap and redispatch."""
    from charlie_work.adapters import SessionDispatchResult

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        watchdog=WatchdogConfig(
            max_auto_redispatch=2, redispatch_window_minutes=240, max_foreign_writer_reaps=2
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Pre-seed: blocked_environment_at at the cap AND foreign_writer_reaps
    # already at max_foreign_writer_reaps(2). The reap must NOT be attempted;
    # the issue must escalate.
    paths.root.mkdir(parents=True, exist_ok=True)
    reap_ts = _blocked_env_timestamps(2)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "dispatch_failed",
            "blocked_environment_at": _blocked_env_timestamps(2),
            "foreign_writer_reaps": reap_ts,
        }
        save_state(paths.state_file, state)

    wt_path = tmp_path / "wt-foreign"

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=False,
                error="worktree has a live foreign writer",
                failure_kind="worktree_foreign_writer",
                pid=1234,
                worktree_path=str(wt_path),
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    def _reap_must_not_run(
        _failed_result, _config, _state_file, _issue_number, _sessions_dir=None
    ):
        raise AssertionError("reap must not be attempted once the reap cap is reached")

    monkeypatch.setattr(
        "charlie_work.workflow._try_reap_blocked_foreign_writer", _reap_must_not_run
    )

    result = app.dispatch(limit=1)
    assert result.ok is False
    state = load_state(paths.state_file)
    # Escalated, NOT reaped.
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "dispatch_blocked_environment"
    # The reap counter is unchanged (no new reap recorded).
    assert state["issues"]["123"]["foreign_writer_reaps"] == reap_ts


def test_dispatch_failed_retries_are_capped_and_escalate(tmp_path: Path) -> None:
    """Issue #461: repeated dispatch failures are capped and then escalated."""
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(7)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=1),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Avoid the open-PR exclusion by closing the default fixture PR.
    fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # First dispatch failure is recorded normally.
    result1 = app.dispatch(limit=1)
    assert result1.ok is False
    assert result1.data["failed_count"] == 1
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatch_failed"
    assert len(state["issues"]["123"]["dispatch_failed_at"]) == 1

    # Second failure exceeds the cap and escalates the issue.
    result2 = app.dispatch(limit=1)
    assert result2.ok is False
    assert result2.data["failed_count"] == 1
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "dispatch_failed_cap_exceeded"
    assert len(state["issues"]["123"]["dispatch_failed_at"]) == 2
    # Issue #1266: repeated dispatch failure is a mechanical escalation.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added

    # Third dispatch no longer selects the escalated issue.
    result3 = app.dispatch(limit=1)
    assert result3.ok is True
    assert result3.data["selected_count"] == 0


def test_dispatch_defers_after_stall_reap_sets_throttled_until(tmp_path: Path) -> None:
    """Issue #246: after the stall watchdog reaps a worker that hit a live
    provider rate limit, the very next dispatch pass must defer instead of
    launching a replacement worker into the same throttle window.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        dispatch=DispatchConfig(default_limit=3),
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    _make_stalled_sidecar(
        sessions_dir,
        4055,
        log_text=(
            "Error: Reached overall message rate limit. Please try again "
            "later. Your limit will reset in 9 minutes.\n"
        ),
    )

    with (
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.write_gate.kill_process_tree", return_value=[99999]),
        patch("charlie_work.dead_worker_reap.sweep_orphan_processes", return_value=[]),
    ):
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        _detect_and_handle_stalled_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )

        # dispatch() re-runs the stall reaper unconditionally at its top (workflow.py
        # dispatch():~1180) before checking is_throttled — keep the same mocks active
        # so this second reap pass over the already-classified sidecar stays cheap
        # and deterministic instead of shelling out to real process/PowerShell calls.
        app.gh.prs[0]["state"] = "CLOSED"
        result = app.dispatch()

    assert result.ok is False
    assert result.data["deferred_reason"] == "provider_throttled"
    assert result.data["throttled_until"] is not None
    assert result.data["selected_count"] == 0


def test_dispatch_stall_detection_called_once_per_dispatch(tmp_path: Path, monkeypatch) -> None:
    """Regression test for issue #158: _detect_and_handle_stalled_sessions should be called exactly once per dispatch() call, not twice (was duplicated in _apply_concurrency_governor)."""
    # Mock _detect_and_handle_stalled_sessions to track call count
    stall_detection_calls = []

    def mock_stall_detection(sessions_dir, state_file, config, *, write_gate):
        stall_detection_calls.append(1)
        return []  # No stalled sessions

    monkeypatch.setattr(
        "charlie_work.workflow._detect_and_handle_stalled_sessions", mock_stall_detection
    )

    # Mock _count_live_sessions to return 0 (no live sessions)
    def mock_count_live(sessions_dir, state_file=None):
        return 0

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Call dispatch() with max_concurrent_sessions > 0
    app.gh.prs[0]["state"] = "CLOSED"
    app.dispatch()

    # Verify stall detection was called exactly once
    assert len(stall_detection_calls) == 1, (
        f"_detect_and_handle_stalled_sessions was called {len(stall_detection_calls)} times, expected 1"
    )


def test_dispatch_defers_when_provider_throttled(tmp_path: Path) -> None:
    """When provider throttle window is active, dispatch should defer and report why."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(default_limit=3),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Set a throttle window in the future
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        # Set throttled_until to 1 hour in the future
        from datetime import UTC, datetime, timedelta

        future_time = datetime.now(UTC) + timedelta(hours=1)
        throttled_until = future_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        state = set_throttled_until(state, throttled_until)
        save_state(paths.state_file, state)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch()

    # Should defer with provider_throttled reason
    assert result.ok is False
    assert result.data["deferred_reason"] == "provider_throttled"
    assert result.data["throttled_until"] is not None
    assert result.data["selected_count"] == 0
    assert result.data["attempted_count"] == 0


def test_dispatch_proceeds_when_throttle_window_expired(tmp_path: Path) -> None:
    """When provider throttle window has passed, dispatch should proceed normally."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(default_limit=3),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Set a throttle window in the past
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        from datetime import UTC, datetime, timedelta

        past_time = datetime.now(UTC) - timedelta(hours=1)
        throttled_until = past_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        state = set_throttled_until(state, throttled_until)
        save_state(paths.state_file, state)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch()

    # Should proceed normally (not throttled)
    assert result.ok is True
    assert result.data["selected_count"] == 1  # One ready issue
    assert "deferred_reason" not in result.data
