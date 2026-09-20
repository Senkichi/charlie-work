"""Salvage paths: empty-diff fetch failure, already-landed salvage, stranded-commit skip, phantom-live sidecar preservation.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from _dead_session_fixtures import _git
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _init_repo_with_remote_inline
from _worktree_fixtures import (
    _init_bare_remote_and_clone,
    _setup_completed_worktree,
)
from charlie_work import github as github_module
from charlie_work.claude_code import ClaudeWorkerRecord
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
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_phantom_live_worker_preserves_sidecar_for_dirty_worktree_with_commits(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #1130: a phantom live worker whose worktree is PARTIAL (dirty
    working tree but commits ahead of base) must NOT have its sidecar reaped.
    The committed work is salvageable; reaping the sidecar would destroy the
    salvage path. This guards the relaxed ``ahead_count > 0`` preserve
    condition against the previous ``COMPLETED``-only check."""

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 1130, dirty=True)

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree_path),
            prompt_path=str(worktree_path / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=8282,
            started_at="2026-08-10T11:15:39Z",
            log_path=str(tmp_path / "log"),
            error="probe_error",
            failure_kind="live_worker_redispatch_averted",
            process_start_time=5_678_901.0,
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    monkeypatch.setattr("charlie_work.workflow.is_pid_alive", lambda pid, start: False)

    config = OrchestratorConfig(
        devin=DevinConfig(), worker=WorkerRoleConfig(harness="claude-code")
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.pr_list = lambda: []
    _original_issue_view = fake_gh.issue_view

    def _patched_issue_view(number: int):
        issue = _original_issue_view(number)
        return {
            **issue,
            "labels": [
                {"name": "automated-ready"},
                {"name": "agent:in-progress"},
            ],
        }

    fake_gh.issue_view = _patched_issue_view

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sessions_dir / "issue-123.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 123,
                "branch": branch,
                "worktree_path": str(worktree_path),
                "prompt_path": "",
                "command": ["claude", "-p"],
                "pid": 8282,
                "started_at": "2026-08-10T11:15:39Z",
                "log_path": str(tmp_path / "log"),
                "error": "probe_error",
                "failure_kind": "live_worker_redispatch_averted",
                "process_start_time": 5_678_901.0,
                "session_id": "test-session-1130",
            }
        ),
        encoding="utf-8",
    )

    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "branch_name": branch,
        "worker_pid": 8282,
        "worker_process_start_time": 5_678_901.0,
        "title": "Fix search",
        "url": "https://example.test/issues/123",
    }
    save_state(paths.state_file, seed)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    # The phantom live worker is detected but the sidecar is PRESERVED.
    assert result.data["phantom_live_worker_count"] == 1
    assert sidecar_path.exists(), (
        "Sidecar must not be reaped so the reaper lane can salvage the committed work"
    )

    # Labels are NOT stripped.
    assert (123, "agent:in-progress") not in fake_gh.labels_removed
    assert (123, "automated-ready") not in fake_gh.labels_removed

    state = load_state(paths.state_file)
    preserve_events = [
        e
        for e in state.get("events", [])
        if e["kind"] == "session_failed_relabeled"
        and e["payload"]["issue_number"] == 123
        and e["payload"]["reason"] == "phantom_live_worker_completed_work_preserved"
    ]
    assert len(preserve_events) == 1


def test_salvage_branch_empty_diff_returns_false_on_fetch_failure(
    tmp_path: Path,
) -> None:
    """Issue #1221 (check 3 fail-safe): ``salvage_branch_empty_diff`` returns
    False (do not skip salvage) when ``git fetch origin <base>`` fails -- a
    transient network error falls back to opening the PR, which a human reviews
    anyway. This is the fail-safe branch the design relies on but had no test.
    """
    from charlie_work.worktree import salvage_branch_empty_diff

    # A git repo with NO origin remote: ``git fetch origin main`` fails.
    repo_root = tmp_path / "no-remote-clone"
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "Test User")
    _git(repo_root, "config", "commit.gpgSign", "false")
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo_root, "add", "README.md")
    _git(repo_root, "commit", "-m", "initial commit")

    # No origin remote configured -- fetch will fail.
    result = salvage_branch_empty_diff(repo_root, "agent/issue-1221", "main")
    assert result is False


def test_salvage_already_landed_proceeds_when_empty_diff_fetch_fails(
    tmp_path: Path,
) -> None:
    """Issue #1221 (check 3 fail-safe, integration): when the git fetch inside
    ``salvage_branch_empty_diff`` fails, the function returns False and
    ``_salvage_already_landed`` returns ``(False, None)`` -- salvage proceeds
    (does not skip) instead of treating the git error as evidence the work
    already landed. A human reviews salvage PRs anyway.
    """
    from charlie_work.workflow import _salvage_already_landed

    # A git repo with NO origin remote: ``git fetch origin main`` fails.
    repo_root = tmp_path / "no-remote-clone"
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "Test User")
    _git(repo_root, "config", "commit.gpgSign", "false")
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo_root, "add", "README.md")
    _git(repo_root, "commit", "-m", "initial commit")

    config = OrchestratorConfig()

    class FakeGitHubEmptyMergeSearch(FakeGitHub):
        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            return github_module._MergedPRSearchResult([], ok=True)

    gh = FakeGitHubEmptyMergeSearch()

    # Issue is OPEN, no merged PR binds to it, and the fetch inside
    # salvage_branch_empty_diff fails (no origin remote). The fail-safe
    # must let salvage proceed: _salvage_already_landed returns (False, None).
    already_landed, reason = _salvage_already_landed(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch="agent/issue-1221",
        base_ref="main",
        issue_number=1221,
        issue={"state": "OPEN"},
    )
    assert already_landed is False
    assert reason is None


def test_salvage_rework_stranded_commits_skips_when_status_not_rework_requested(
    tmp_path: Path,
) -> None:
    """Issue #1239 round-3: the death-loop escalation gate's salvage call site
    (``_salvage_rework_stranded_commits``) receives ``issue_entry`` from the
    ``head_check_state`` snapshot loaded at the top of ``dispatch_rework``'s
    candidate loop.  Between that snapshot and this call the issue's status may
    have already moved off ``rework_requested`` (e.g. a concurrent loop pass
    dispatched it, escalated it, or the issue was closed).  In that case the
    salvage push to the shared origin remote MUST NOT be attempted — an
    unaudited push for a stale/no-longer-rework_requested issue leaves no event
    trail if it succeeds.  A fresh ``status == "rework_requested"`` precondition
    check (short state_lock scope, before any network push) gates the salvage,
    mirroring the ``_reap_restore_rework_requested`` precondition (which checks
    ``status == "dispatched"`` because that lane handles already-dispatched
    workers).
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
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
        worker=WorkerRoleConfig(harness="command"),
    )
    layout = resolved_layout(config, repo_root)
    wt_path = worktree_path_for_branch(repo_root, branch, layout.worktrees)
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", str(wt_path), branch])

    # Push the branch so the remote branch exists at the PR head sha.
    ok, error = push_branch(repo_root, branch, worktree_path=wt_path)
    assert ok, error
    pr_head_sha = run(["git", "rev-parse", branch]).stdout.strip()

    # Add a stranded commit to the worktree — the salvage WOULD push this if
    # the precondition check were absent.
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
    # The issue's status has ALREADY moved off "rework_requested" — a
    # concurrent loop pass dispatched it (status is now "dispatched").  This
    # is the head_check_state/state.json decoupling the round-3 review flagged:
    # the snapshot still says rework_requested, but state.json has advanced.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "dispatched",
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

    # The snapshot issue_entry still says rework_requested (stale) — this is
    # what dispatch_rework's candidate loop would pass.  The fresh check inside
    # _salvage_rework_stranded_commits must catch that state.json has moved.
    stale_issue_entry = {
        "number": 123,
        "status": "rework_requested",
        "branch_name": branch,
    }
    pr_data = fake_gh.prs[0]

    result = app._salvage_rework_stranded_commits(123, pr_data, stale_issue_entry)

    # The salvage must report no push.
    assert result is False

    # The remote branch head MUST NOT have advanced — no salvage push.
    remote_sha = subprocess.run(
        ["git", "rev-parse", branch],
        cwd=remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert remote_sha == pr_head_sha, (
        f"salvage push attempted for a non-rework_requested issue: "
        f"remote {remote_sha} != pr head {pr_head_sha}"
    )

    # The issue status must be unchanged (still dispatched, not reset).
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"

    # No salvage event must have been recorded.
    events = state.get("events", [])
    salvage_events = [e for e in events if e.get("kind") == "rework_stranded_commits_salvaged"]
    assert len(salvage_events) == 0
