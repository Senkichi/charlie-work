"""Rework-dispatch caps: no-op and death-loop escalation, rework cap.

Split out of ``tests/test_charlie_work.py`` (issue #1547, Track-1 wave 1/8):
the ``test_dispatch_rework_*`` seam's cap half -- repeated/deterministic
failure escalation, no-op vs worker-death classification, the death loop
with stranded-commit salvage, and the ``test_rework_cap_*`` operator-queue
cap pair. Shared fakes and helpers in
``tests/_rework_dispatch_fixtures.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest
from _fakes_github import FakeGitHub
from _helpers import _init_git_repo
from _rework_dispatch_fixtures import (
    _init_repo_with_remote_inline,
)
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp


def test_dispatch_rework_escalates_after_repeated_failures(tmp_path: Path) -> None:
    """Issue #515: repeated failed rework-dispatch attempts must count toward the
    redispatch cap and escalate instead of retrying forever.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    for i in range(2):
        result = app.dispatch_rework()
        assert result.ok is False
        assert result.data["failed_count"] == 1
        state = load_state(paths.state_file)
        assert state["issues"]["123"]["status"] == "rework_requested"
        assert len(state["issues"]["123"].get("redispatch_at", [])) == i + 1

    # Third failure exceeds the cap and escalates. Issue #1266:
    # redispatch_cap_exceeded is mechanical, so this lands
    # agent:operator-queue, not agent:human-needed.
    result = app.dispatch_rework()
    assert result.ok is False
    assert result.data["failed_count"] == 1
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "redispatch_cap_exceeded"
    assert len(state["issues"]["123"]["redispatch_at"]) == 3
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    assert (123, config.labels.needs_rework) in fake_gh.labels_removed


def test_dispatch_rework_deterministic_failure_kind_escalates_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rework dispatch failure with a deterministic failure_kind (e.g.
    rework_branch_conflict) must escalate immediately on the first occurrence,
    not loop rework_requested → dispatch → fail until the redispatch cap is hit.
    """
    from charlie_work.adapters import SessionDispatchResult

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=3, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=False,
                error="rework branch conflicts with origin/main",
                failure_kind="rework_branch_conflict",
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    result = app.dispatch_rework()
    assert result.ok is False

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "rework_branch_conflict"
    # Issue #1266: a deterministic failure_kind escalation is mechanical, so
    # it lands agent:operator-queue, not agent:human-needed.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


def test_dispatch_rework_no_op_rework_cap_escalates(tmp_path: Path) -> None:
    """A rework candidate whose PR head hasn't moved since the last
    request_changes verdict, and whose redispatch_at count is already at the
    cap, must be escalated immediately during candidate filtering instead of
    dispatching another worker that will produce no changes.
    """
    from datetime import UTC, datetime

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    fake_gh = ReworkGitHub()
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": [now_iso, now_iso],
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch_rework()
    assert result.ok is True
    assert 123 in result.data.get("no_op_rework_escalated", [])

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "redispatch_cap_exceeded"
    # Issue #1266: no_op_rework_escalated is mechanical, so it lands
    # agent:operator-queue, not agent:human-needed.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


def test_dispatch_rework_worker_deaths_dont_count_as_no_op(tmp_path: Path) -> None:
    """Issue #1134: a rework candidate whose redispatch_at count is at the cap
    but whose redispatches all ended in worker deaths must NOT escalate as
    no_op_rework_cap_exceeded.  A death is not a no-op — the worker may have
    completed its work but died before pushing.  When the death count itself
    reaches the cap, the issue escalates with worker_death_loop instead.
    """
    from datetime import UTC, datetime

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    fake_gh = ReworkGitHub()
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": [now_iso, now_iso],
            "worker_death_at": [now_iso, now_iso],
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch_rework()
    assert result.ok is True
    # Must NOT be in no_op_rework_escalated — deaths are not no-ops.
    assert 123 not in result.data.get("no_op_rework_escalated", [])
    # Must be in worker_death_escalated instead.
    assert 123 in result.data.get("worker_death_escalated", [])

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "worker_death_loop"
    # Issue #1266: worker_death_loop is mechanical, so it lands
    # agent:operator-queue, not agent:human-needed.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


def test_dispatch_rework_mixed_deaths_and_no_ops_no_op_dominates(tmp_path: Path) -> None:
    """Issue #1134: when there are both deaths and genuine no-ops, the no-op
    count (total redispatches minus deaths) determines the no-op cap.  With
    3 redispatches, 1 death, and cap=2, the no-op count is 2 — enough to
    escalate as no_op_rework_cap_exceeded (the no-ops dominate).
    """
    from datetime import UTC, datetime

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    fake_gh = ReworkGitHub()
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": [now_iso, now_iso, now_iso],
            "worker_death_at": [now_iso],
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch_rework()
    assert result.ok is True
    # no_op_count = 3 - 1 = 2 >= cap(2) → no-op escalation.
    assert 123 in result.data.get("no_op_rework_escalated", [])
    assert 123 not in result.data.get("worker_death_escalated", [])

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "redispatch_cap_exceeded"


def test_dispatch_rework_deaths_below_cap_still_dispatched(tmp_path: Path) -> None:
    """Issue #1134: when both the no-op count and the death count are below
    the cap, the issue must NOT be escalated — it remains a legitimate
    dispatch candidate.  With 2 redispatches, 1 death, and cap=2, the no-op
    count is 1 and the death count is 1 — both below cap.
    """
    from datetime import UTC, datetime

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    fake_gh = ReworkGitHub()
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": [now_iso, now_iso],
            "worker_death_at": [now_iso],
            "branch_name": "agent/issue-123-fix-search",
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    from charlie_work.adapters import SessionDispatchResult

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=True,
                pid=99999,
                process_start_time=datetime.now(UTC).isoformat(),
            )
            for request in requests
        ]

    import charlie_work.workflow as workflow_module

    original = workflow_module.dispatch_sessions
    workflow_module.dispatch_sessions = fake_dispatch_sessions
    try:
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)
        result = app.dispatch_rework()
    finally:
        workflow_module.dispatch_sessions = original

    # Not escalated — both counts below cap.
    assert 123 not in result.data.get("no_op_rework_escalated", [])
    assert 123 not in result.data.get("worker_death_escalated", [])
    # The issue was dispatched (not filtered out).
    assert result.data.get("selected_count", 0) >= 1


def test_dispatch_rework_worker_death_loop_includes_stranded_commits(
    tmp_path: Path,
) -> None:
    """Issue #1134: when a worker death loop escalates, the escalation payload
    must include ``stranded_commits`` — the count of commits in the worktree
    that are ahead of the PR head (salvageable work the worker completed but
    never pushed).
    """
    from datetime import UTC, datetime

    from charlie_work.paths import resolved_layout
    from charlie_work.worktree import worktree_path_for_branch

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)

    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )

    # Create a branch with one commit (the "PR head").
    run(["git", "branch", "agent/issue-123-fix-search"])
    pr_head_sha = run(["git", "rev-parse", "main"]).stdout.strip()

    # Create a worktree at the expected path and add a commit to it
    # (simulating a worker that completed work but died before pushing).
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    layout = resolved_layout(config, repo_root)
    wt_path = worktree_path_for_branch(repo_root, "agent/issue-123-fix-search", layout.worktrees)
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", str(wt_path), "agent/issue-123-fix-search"])
    (wt_path / "fix.txt").write_text("fixed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "fix.txt"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "completed rework (died before push)"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )

    paths = runtime_paths(repo_root, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.repo_root = repo_root
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]
            self.prs[0]["headRefOid"] = pr_head_sha

    fake_gh = ReworkGitHub()
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": [now_iso, now_iso],
            "worker_death_at": [now_iso, now_iso],
            "branch_name": "agent/issue-123-fix-search",
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": pr_head_sha,
        }
        save_state(paths.state_file, state)

    app = OrchestratorApp(repo_root, paths, config, fake_gh)

    result = app.dispatch_rework()
    assert result.ok is True
    assert 123 in result.data.get("worker_death_escalated", [])

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "worker_death_loop"
    # stranded_commits must be present and reflect the 1 unpushed commit.
    assert state["issues"]["123"]["stranded_commits"] == 1


def test_dispatch_rework_death_loop_salvages_stranded_commits(
    tmp_path: Path,
) -> None:
    """Issue #1239: when a rework worker death-loop reaches the cap, the
    death-loop gate must attempt to salvage-push stranded commits from the
    dead worker's worktree BEFORE escalating.  A successful push means the
    worker completed the rework and died at the final push step — the PR
    head moves past the request_changes verdict, so the issue is routed to
    review instead of escalated.
    """
    from datetime import UTC, datetime

    from charlie_work.paths import resolved_layout
    from charlie_work.worktree import push_branch, worktree_path_for_branch

    remote, repo_root = _init_repo_with_remote_inline(tmp_path)
    branch = "agent/issue-123-fix-search"

    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )

    # Create a branch from main and a worktree at the expected orchestrator path.
    run(["git", "branch", branch])
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    layout = resolved_layout(config, repo_root)
    wt_path = worktree_path_for_branch(repo_root, branch, layout.worktrees)
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", str(wt_path), branch])

    # Push the branch so the remote branch exists at the PR head sha.
    ok, error = push_branch(repo_root, branch, worktree_path=wt_path)
    assert ok, error
    pr_head_sha = run(["git", "rev-parse", branch]).stdout.strip()

    # Add a stranded commit to the worktree (worker completed work but died
    # before pushing).
    (wt_path / "fix.txt").write_text("fixed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "fix.txt"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "completed rework (died before push)"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    local_tip = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    paths = runtime_paths(repo_root, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.repo_root = repo_root
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]
            self.prs[0]["headRefOid"] = pr_head_sha

    fake_gh = ReworkGitHub()
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": [now_iso, now_iso],
            "worker_death_at": [now_iso, now_iso],
            "branch_name": branch,
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": pr_head_sha,
        }
        save_state(paths.state_file, state)

    app = OrchestratorApp(repo_root, paths, config, fake_gh)

    result = app.dispatch_rework()
    assert result.ok is True
    assert 123 in result.data.get("salvaged_to_review", [])
    assert 123 not in result.data.get("worker_death_escalated", [])

    # The remote branch head must have advanced to the worktree's local tip.
    remote_sha = subprocess.run(
        ["git", "rev-parse", branch],
        cwd=remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert remote_sha == local_tip

    state = load_state(paths.state_file)
    # The issue must NOT be escalated (routed to review or stays rework_requested).
    assert state["issues"]["123"]["status"] != "escalated"
    # worker_death_at must NOT have been extended (still 2 entries, not 3).
    assert len(state["issues"]["123"].get("worker_death_at", [])) == 2

    # An event with kind "rework_stranded_commits_salvaged" must exist.
    events = state.get("events", [])
    salvage_events = [e for e in events if e.get("kind") == "rework_stranded_commits_salvaged"]
    assert len(salvage_events) >= 1


def test_dispatch_rework_death_loop_empty_death_still_escalates(
    tmp_path: Path,
) -> None:
    """Issue #1239: a death-loop where the worktree has NO stranded commits
    (the worker died without completing any work) must still escalate with
    ``worker_death_loop`` — the salvage push is only a reprieve when there
    is actual completed work to publish.
    """
    from datetime import UTC, datetime

    from charlie_work.paths import resolved_layout
    from charlie_work.worktree import push_branch, worktree_path_for_branch

    remote, repo_root = _init_repo_with_remote_inline(tmp_path)
    branch = "agent/issue-123-fix-search"

    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )

    run(["git", "branch", branch])
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    layout = resolved_layout(config, repo_root)
    wt_path = worktree_path_for_branch(repo_root, branch, layout.worktrees)
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", str(wt_path), branch])

    # Push the branch — but do NOT add any commit (worktree is at the same
    # sha as the remote branch, so there are no stranded commits to salvage).
    ok, error = push_branch(repo_root, branch, worktree_path=wt_path)
    assert ok, error
    pr_head_sha = run(["git", "rev-parse", branch]).stdout.strip()

    paths = runtime_paths(repo_root, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.repo_root = repo_root
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]
            self.prs[0]["headRefOid"] = pr_head_sha

    fake_gh = ReworkGitHub()
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": [now_iso, now_iso],
            "worker_death_at": [now_iso, now_iso],
            "branch_name": branch,
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": pr_head_sha,
        }
        save_state(paths.state_file, state)

    app = OrchestratorApp(repo_root, paths, config, fake_gh)

    result = app.dispatch_rework()
    assert result.ok is True
    assert 123 in result.data.get("worker_death_escalated", [])
    assert 123 not in result.data.get("salvaged_to_review", [])

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "worker_death_loop"


def test_rework_cap_escalates_to_operator_queue(tmp_path: Path) -> None:
    config = OrchestratorConfig()  # max_rework_cycles = 2
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # First request_changes (count = 1, head = "sha-1")
    fake_gh.pr_head_shas[456] = "sha-1"
    first = app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Second request_changes (count = 2, head = "sha-2")
    fake_gh.pr_head_shas[456] = "sha-2"
    second = app.record_review(
        456, "request_changes", summary="fix B", verdict_provenance="fresh_llm_review"
    )

    # Third request_changes (count stays at 2, escalated, head = "sha-3")
    fake_gh.pr_head_shas[456] = "sha-3"
    third = app.record_review(
        456, "request_changes", summary="fix C", verdict_provenance="fresh_llm_review"
    )

    assert first.data["escalated"] is False and first.data["rework_path"]
    assert second.data["escalated"] is False and second.data["rework_path"]
    assert third.data["escalated"] is True
    assert third.data["rework_path"] is None  # no third rework prompt
    assert fake_gh.labels_added.count((123, "agent:needs-rework")) == 2
    # Issue #1266: max_rework_cycles_exceeded is a mechanical escalation.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


def test_rework_cap_survives_event_log_truncation(tmp_path: Path) -> None:
    # The P0: the counter used to derive from state["events"], which
    # append_event truncates to the last 200 - evicting a PR's earlier
    # request_changes and silently resetting the cap. The durable per-PR
    # counter must escalate regardless of how many unrelated events churn.
    from charlie_work.state import append_event as _append

    config = OrchestratorConfig()  # max_rework_cycles = 2
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # First request_changes (count = 1, head = "sha-1")
    fake_gh.pr_head_shas[456] = "sha-1"
    app.record_review(456, "request_changes", summary="a", verdict_provenance="fresh_llm_review")

    # Second request_changes (count = 2, head = "sha-2")
    fake_gh.pr_head_shas[456] = "sha-2"
    app.record_review(456, "request_changes", summary="b", verdict_provenance="fresh_llm_review")

    # Flood the event log so any record_review events for 456 are evicted.
    state = load_state(paths.state_file)
    for i in range(300):
        state = _append(state, "review_packet", {"pr_number": 90000 + i}, max_size=200)
    save_state(paths.state_file, state)
    assert not any(  # prove the earlier request_changes events are gone
        e.get("kind") == "record_review" for e in load_state(paths.state_file)["events"]
    )

    # Third request_changes (count stays at 2, escalated, head = "sha-3")
    fake_gh.pr_head_shas[456] = "sha-3"
    third = app.record_review(
        456, "request_changes", summary="c", verdict_provenance="fresh_llm_review"
    )

    assert third.data["escalated"] is True
    assert third.data["rework_path"] is None
    # Issue #1266: max_rework_cycles_exceeded is a mechanical escalation.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
