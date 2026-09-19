"""Dispatch candidate selection, ordering, and exclusion rules.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the selection seam of ``dispatch()`` -- ready-issue selection, oldest/newest
ordering, explicit-subset selection, stalled-session and open-PR exclusions,
and duplicate-worker guards. Shared fakes and helpers in
``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from _fakes_github import FakeGitHub
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
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_dispatch_selects_ready_issue_without_operator_queue_label(tmp_path: Path) -> None:
    """Positive control for the operator_queue dispatch-exclusion test below:
    the exact same fixture (ready label, no open tracked PR) but with no
    ``operator_queue`` label present must dispatch normally. Without this
    control, a dispatch-exclusion assertion of ``selected_count == 0`` would
    be equally consistent with "the terminal check works" and with "this
    fixture never dispatches for an unrelated reason" (see issue #257 above --
    the default fixture's open tracked PR is exactly that trap)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Clear the default fixture's open tracked PR so it cannot mask the
    # terminal-label check under test.
    fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert (123, "agent:queued") in fake_gh.labels_added


def test_dispatch_oldest_first_by_default(tmp_path: Path) -> None:
    """Test that dispatch selects oldest issues first by default (issue #151)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake GitHub with issues created out of order
    fake_gh = FakeGitHub()
    # Override issue_list to return issues with different creation dates
    fake_gh.issues = [
        {
            "number": 792,
            "title": "crash-fix",
            "url": "https://github.com/test/repo/issues/792",
            "body": "Fix crash",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-01T00:00:00Z",  # Oldest
            "updatedAt": "2026-07-01T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 808,
            "title": "e2e-test",
            "url": "https://github.com/test/repo/issues/808",
            "body": "E2E test",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-06T00:00:00Z",  # Newest
            "updatedAt": "2026-07-06T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 793,
            "title": "data-model",
            "url": "https://github.com/test/repo/issues/793",
            "body": "Data model",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-02T00:00:00Z",  # Middle
            "updatedAt": "2026-07-02T00:00:00Z",
            "state": "OPEN",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Dispatch 2 issues - should select oldest first (792, then 793)
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=2)

    assert result.ok is True
    assert result.data["selected_count"] == 2
    # Should select oldest issues: 792 (oldest), 793 (middle)
    selected_numbers = [s["issue_number"] for s in result.data["sessions"]]
    assert selected_numbers == [792, 793]


def test_dispatch_newest_first_with_config(tmp_path: Path) -> None:
    """Test that dispatch selects newest issues first when configured (issue #151)."""
    config = OrchestratorConfig(dispatch=DispatchConfig(order="newest"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake GitHub with issues created out of order
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 792,
            "title": "crash-fix",
            "url": "https://github.com/test/repo/issues/792",
            "body": "Fix crash",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-01T00:00:00Z",  # Oldest
            "updatedAt": "2026-07-01T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 808,
            "title": "e2e-test",
            "url": "https://github.com/test/repo/issues/808",
            "body": "E2E test",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-06T00:00:00Z",  # Newest
            "updatedAt": "2026-07-06T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 793,
            "title": "data-model",
            "url": "https://github.com/test/repo/issues/793",
            "body": "Data model",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-02T00:00:00Z",  # Middle
            "updatedAt": "2026-07-02T00:00:00Z",
            "state": "OPEN",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Dispatch 2 issues - should select newest first (808, then 793)
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=2)

    assert result.ok is True
    assert result.data["selected_count"] == 2
    # Should select newest issues: 808 (newest), 793 (middle)
    selected_numbers = [s["issue_number"] for s in result.data["sessions"]]
    assert selected_numbers == [808, 793]


def test_dispatch_sorts_by_out_degree_blocked_dependents(tmp_path: Path) -> None:
    """Test that dispatch sorts by out-degree (number of blocked dependents) per issue #152."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake GitHub with dependency relationships:
    # - Issue X (100) has 0 blocked dependents
    # - Issue Y (200) has 3 blocked dependents (300, 400, 500)
    # - Issues 300, 400, 500 are blocked by Y (have open blocker Y)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 100,
            "title": "issue-x",
            "url": "https://github.com/test/repo/issues/100",
            "body": "Issue X with no dependents",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-01T00:00:00Z",
            "updatedAt": "2026-07-01T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 200,
            "title": "issue-y",
            "url": "https://github.com/test/repo/issues/200",
            "body": "Issue Y with 3 dependents",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-02T00:00:00Z",
            "updatedAt": "2026-07-02T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 300,
            "title": "dependent-1",
            "url": "https://github.com/test/repo/issues/300",
            "body": "Blocked by #200",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-03T00:00:00Z",
            "updatedAt": "2026-07-03T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 400,
            "title": "dependent-2",
            "url": "https://github.com/test/repo/issues/400",
            "body": "Blocked by #200",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-04T00:00:00Z",
            "updatedAt": "2026-07-04T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 500,
            "title": "dependent-3",
            "url": "https://github.com/test/repo/issues/500",
            "body": "Blocked by #200",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-05T00:00:00Z",
            "updatedAt": "2026-07-05T00:00:00Z",
            "state": "OPEN",
        },
    ]

    # Mock issue_list to return all ready issues for out-degree computation
    original_issue_list = fake_gh.issue_list

    def mock_issue_list(labels=None, state=None):
        if labels and "automated-ready" in labels:
            return fake_gh.issues
        return original_issue_list(labels=labels, state=state)

    fake_gh.issue_list = mock_issue_list

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Dispatch 2 issues - should select Y (3 dependents) before X (0 dependents)
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=2)

    assert result.ok is True
    assert result.data["selected_count"] == 2
    selected_numbers = [s["issue_number"] for s in result.data["sessions"]]
    # Y (200) should be selected first due to higher out-degree
    assert selected_numbers == [200, 100]


def test_dispatch_only_issues_selects_explicit_subset(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Numbers not among the dispatchable candidates are skipped; only the
    # explicit, dispatchable match is selected (dependency-ordered waves).
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(only_issues="999, 123")

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert (123, "agent:queued") in fake_gh.labels_added


def test_dispatch_only_issues_preserves_mixed_fresh_recovery_order(
    tmp_path: Path,
) -> None:
    """Issue #506: --only-issues preserves operator order for mixed fresh/recovery.

    When an operator explicitly lists recovery candidates before fresh work,
    the requested order is honored, but recovery retries are still capped at
    one per pass so a backlog of stuck recovery issues cannot consume the
    budget.
    """
    from charlie_work.state import empty_state, save_state

    config = OrchestratorConfig(
        dispatch=DispatchConfig(default_limit=3),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    state = empty_state()
    state["issues"]["1317"] = {
        "number": 1317,
        "status": "dispatched",
        "branch_name": "agent/issue-1317-fix-stuck",
    }
    state["issues"]["1323"] = {
        "number": 1323,
        "status": "dispatched",
        "branch_name": "agent/issue-1323-fix-stuck-too",
    }
    save_state(paths.state_file, state)

    class MixedOrderGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 1317,
                    "title": "Fix stuck",
                    "url": "https://example.test/issues/1317",
                    "body": "Stuck",
                    "labels": [{"name": config.labels.ready}],
                    "state": "OPEN",
                    "createdAt": "2026-07-19T00:00:00Z",
                },
                {
                    "number": 1323,
                    "title": "Fix stuck too",
                    "url": "https://example.test/issues/1323",
                    "body": "Also stuck",
                    "labels": [{"name": config.labels.ready}],
                    "state": "OPEN",
                    "createdAt": "2026-07-19T00:01:00Z",
                },
                {
                    "number": 1322,
                    "title": "Fix fresh",
                    "url": "https://example.test/issues/1322",
                    "body": "Fresh",
                    "labels": [{"name": config.labels.ready}],
                    "state": "OPEN",
                    "createdAt": "2026-07-19T00:02:00Z",
                },
            ]
            self.prs = []

    fake_gh = MixedOrderGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(only_issues="1317,1323,1322")

    assert result.ok is True
    assert result.data["selected_count"] == 2
    selected_numbers = [s["issue_number"] for s in result.data["sessions"]]
    assert selected_numbers == [1317, 1322]
    assert result.data["deferred_by_concurrency"] == [1323]
    assert result.data["skipped_issue_numbers"] == []


def test_dispatch_fresh_candidates_take_priority_over_recovery_retry(
    tmp_path: Path,
) -> None:
    """Issue #506: a stuck recovery-retry candidate must not starve fresh candidates.

    With dispatch_limit=1, the fresh candidate must dispatch and the recovery
    retry must not consume the only slot.
    """
    from charlie_work.state import empty_state, save_state

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=1, default_limit=1),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    state = empty_state()
    state["issues"]["1317"] = {
        "number": 1317,
        "status": "dispatched",
        "branch_name": "agent/issue-1317-fix-stuck",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
    }
    save_state(paths.state_file, state)

    class FreshAndStuckGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 1317,
                    "title": "Fix stuck",
                    "url": "https://example.test/issues/1317",
                    "body": "Stuck",
                    "labels": [{"name": config.labels.ready}],
                    "state": "OPEN",
                    "createdAt": "2026-07-19T00:00:00Z",
                },
                {
                    "number": 1322,
                    "title": "Fix fresh",
                    "url": "https://example.test/issues/1322",
                    "body": "Fresh",
                    "labels": [{"name": config.labels.ready}],
                    "state": "OPEN",
                    "createdAt": "2026-07-19T01:00:00Z",
                },
            ]
            self.prs = []

    fake_gh = FreshAndStuckGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 1
    sessions = result.data["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["issue_number"] == 1322
    assert sessions[0].get("recovery") is None


def test_dispatch_excludes_issue_with_open_tracked_pr(tmp_path: Path) -> None:
    """Issue #257: a labeled issue with an open tracked PR must never be a
    dispatch candidate, even with no state.json entry (label drift after
    manual salvage or escalation churn) — GitHub's open-PR set is the
    ground truth, not labels or state."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # The default FakeGitHub fixture is exactly the hazard case: issue 123 is
    # labeled ready and has NO state entry, while open PR 456 tracks it.
    assert app.gh.prs[0]["state"] == "OPEN"
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    assert not prompt_path.exists()
    assert (123, "agent:queued") not in fake_gh.labels_added
    assert (123, "agent:in-progress") not in fake_gh.labels_added


def test_dispatch_excludes_issue_with_operator_queue_label(tmp_path: Path) -> None:
    """Issue #1266: an issue carrying ``agent:operator-queue`` (a mechanical
    escalation awaiting operator triage) must never be selected for dispatch,
    even though it still carries ``automated-ready`` -- exactly the same
    invariant ``human_needed`` already has via ``LabelConfig.terminal``.
    Drives the real ``OrchestratorApp.dispatch`` -> ``_is_dispatchable`` path,
    not the label-set membership in isolation (that is
    ``test_label_config_operator_queue_in_terminal_set`` in test_config.py).
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    fake_gh.issues[0]["labels"] = [
        {"name": config.labels.ready},
        {"name": config.labels.operator_queue},
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    assert not prompt_path.exists()
    assert (123, "agent:queued") not in fake_gh.labels_added
    assert (123, "agent:in-progress") not in fake_gh.labels_added


def test_dispatch_excludes_stalled_session_dry_run(tmp_path: Path) -> None:
    """Test that dispatch excludes issues with stalled sessions (dry-run path)."""
    from datetime import UTC, datetime, timedelta
    from charlie_work.devin_shell import SessionRecord

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Create a session record for issue 123 with a live PID
    sessions_dir = app._layout.sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_file = sessions_dir / "issue-123.json"
    log_file = sessions_dir / "issue-123.log"

    # Write a log file with old mtime (stalled)
    log_file.write_text("working on issue\nmaking progress\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a session record with a fake PID (we'll mock liveness check)
    session_record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(
            tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
        ),
        command=("devin", "--prompt-file", "{prompt_path}"),
        pid=99999,  # Fake PID - we'll mock liveness to return True
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
        process_start_time=time.time(),
    )
    session_file.write_text(json.dumps(session_record.to_dict()), encoding="utf-8")

    # Mock the liveness check to return True (simulating a live but stalled process).
    # Patch target is charlie_work.worker (not devin_shell): the stalled-detection
    # path goes through worker.WorkerView.is_alive(), which holds its own
    # already-bound reference to is_session_alive from its module-level import —
    # patching devin_shell's attribute would not reach that call site.
    from unittest.mock import patch

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        app.gh.prs[0]["state"] = "CLOSED"
        result = app.dispatch(limit=1)

    # The stalled issue should be excluded from dispatch
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["stalled"] == [{"issue": 123, "pid": 99999, "health": "STALLED"}]


def test_dispatch_excludes_stalled_session_real(tmp_path: Path) -> None:
    """Test that dispatch excludes issues with stalled sessions (real dispatch path)."""
    from datetime import UTC, datetime, timedelta
    from charlie_work.devin_shell import SessionRecord

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a session record for issue 123 with a live PID
    sessions_dir = app._layout.sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_file = sessions_dir / "issue-123.json"
    log_file = sessions_dir / "issue-123.log"

    # Write a log file with old mtime (stalled)
    log_file.write_text("working on issue\nmaking progress\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a session record with a fake PID (we'll mock liveness check)
    session_record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(
            tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
        ),
        command=("devin", "--prompt-file", "{prompt_path}"),
        pid=99999,  # Fake PID - we'll mock liveness to return True
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
        process_start_time=time.time(),
    )
    session_file.write_text(json.dumps(session_record.to_dict()), encoding="utf-8")

    # Mock the liveness check to return True (simulating a live but stalled process)
    from unittest.mock import patch

    with patch("charlie_work.devin_shell.is_session_alive", return_value=True):
        app.gh.prs[0]["state"] = "CLOSED"
        result = app.dispatch(limit=1)

    # The stalled issue should be excluded from dispatch
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["stalled"] == [{"issue": 123, "pid": 99999}]


def test_dispatch_issues_reports_skipped(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(only_issues="123,999")

    assert result.data["skipped_issue_numbers"] == [999]
    assert "999" in result.message


def test_dispatch_guard_blocks_second_worker_for_live_dispatched_issue(tmp_path: Path) -> None:
    """A live dispatched issue is not re-dispatched even if label write failed."""
    from charlie_work.devin_shell import SessionRecord

    config = OrchestratorConfig(
        devin=DevinConfig(shell_command=(sys.executable, "-c", "import sys; sys.exit(0)")),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Simulate a prior dispatch that launched a worker but whose label write
    # failed: state says "dispatched" but the issue still lacks active labels.
    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {"number": 123, "status": "dispatched"}
    save_state(paths.state_file, seed)
    # Create a genuinely live worker session record
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    # Spawn a short-lived process
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
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

        assert result.data["attempted_count"] == 0  # not re-dispatched
    finally:
        process.kill()
        process.wait()
