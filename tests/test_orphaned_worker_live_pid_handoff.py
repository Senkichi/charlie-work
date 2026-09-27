"""PID-independent worker-handoff finalize (issue #1867).

Regression coverage for the incident where a worker pushed its branch and
wrote a complete ``.worker-outcome.json`` (``push_succeeded: true`` /
``pr_created: false``) but its process stayed alive ~2h: the dead-PID
recovery lanes could not see the completed handoff, so the PR was never
opened until the stall watchdog eventually reaped the process.

These tests exercise ``_detect_and_handle_orphaned_workers``'s live-PID
finalize lane: when the outcome file is older than
``watchdog.worker_outcome_finalize_minutes`` the orchestrator opens the PR
from the worker's drafted title/body without waiting for the PID to exit.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from charlie_work.config import WorkerRoleConfig


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _make_repo_with_pushed_branch(tmp_path: Path, branch: str) -> Path:
    """Create a real repo + bare origin with ``branch`` pushed (one commit ahead)."""
    remote_repo = tmp_path / "remote"
    remote_repo.mkdir(parents=True, exist_ok=True)
    _git(["git", "init", "--bare", str(remote_repo)], cwd=tmp_path)

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(["git", "init", "--initial-branch=main", str(repo_root)], cwd=tmp_path)
    _git(["git", "config", "user.email", "test@example.test"], cwd=repo_root)
    _git(["git", "config", "user.name", "Test User"], cwd=repo_root)
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["git", "add", "README.md"], cwd=repo_root)
    _git(["git", "commit", "-m", "initial"], cwd=repo_root)
    _git(["git", "remote", "add", "origin", str(remote_repo)], cwd=repo_root)
    _git(["git", "push", "-u", "origin", "main"], cwd=repo_root)

    _git(["git", "checkout", "-b", branch], cwd=repo_root)
    (repo_root / "fix.txt").write_text("fix\n", encoding="utf-8")
    _git(["git", "add", "fix.txt"], cwd=repo_root)
    _git(["git", "commit", "-m", "fix"], cwd=repo_root)
    _git(["git", "push", "-u", "origin", branch], cwd=repo_root)
    _git(["git", "checkout", "main"], cwd=repo_root)
    return repo_root


def _write_outcome(worktree_path: Path, outcome: dict, *, age_seconds: float) -> Path:
    """Write ``.worker-outcome.json`` into ``worktree_path`` with a backdated mtime."""
    import json

    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / ".worker-outcome.json"
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    old = time.time() - age_seconds
    os.utime(outcome_path, (old, old))
    return outcome_path


def _seed_dispatched_issue(paths, *, issue_number: int, branch: str, pid: int = 99999) -> None:
    from charlie_work.state import load_state, save_state

    state = load_state(paths.state_file)
    state["issues"][str(issue_number)] = {
        "status": "dispatched",
        "worker_pid": pid,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "branch_name": branch,
    }
    save_state(paths.state_file, state)


def _fake_gh(repo_root: Path, *, issue_number: int, in_progress: str, pr_create_return):
    class _Fake(FakeGitHub):
        def __init__(self) -> None:
            super().__init__(repo_root=repo_root, dry_run=False)
            self.issues = [
                {
                    "number": issue_number,
                    "title": "Worker finished but its PID never exited",
                    "url": f"https://example.test/issues/{issue_number}",
                    "body": "Repro of the stalled handoff.",
                    "labels": [{"name": in_progress}],
                    "state": "OPEN",
                }
            ]
            self.prs = []
            self.pr_create_return = pr_create_return

    return _Fake()


def _config(*, finalize_minutes: int = 15):
    from charlie_work.config import DevinConfig, OrchestratorConfig, WatchdogConfig

    return OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(
            enabled=True,
            stall_minutes=20,
            worker_outcome_finalize_minutes=finalize_minutes,
        ),
    )


def test_live_pid_stale_outcome_opens_pr_without_waiting_for_exit(tmp_path: Path) -> None:
    """Issue #1867 (swole #163): a worker that pushed its branch and wrote a
    complete ``.worker-outcome.json`` has finished the handoff contract --
    the orchestrator must open the PR from the drafted title/body even while
    the recorded worker PID is still alive.
    """
    from charlie_work.paths import resolved_layout, runtime_paths
    from charlie_work.state import PASSIVE_OPEN_STATUS, load_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers
    from charlie_work.worktree import worktree_path_for_branch

    config = _config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    branch = "agent/issue-1867-stale-outcome-live-pid"
    repo_root = _make_repo_with_pushed_branch(tmp_path, branch)
    _seed_dispatched_issue(paths, issue_number=1867, branch=branch)

    drafted_body = "Closes #1867\n\nRan `pytest -q` -- all passed.\nNo risks identified."
    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    _write_outcome(
        worktree_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "pr_title": "fix: finalize stale worker outcome without waiting for PID exit",
            "pr_body": drafted_body,
        },
        age_seconds=3600,  # ~1h old -- well past the 15-minute finalize threshold
    )

    fake_gh = _fake_gh(
        repo_root,
        issue_number=1867,
        in_progress=config.labels.in_progress,
        pr_create_return=9001,
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # The incident's whole point: the PID is still alive, so the dead-PID
    # lanes must never see this issue -- yet the PR still gets opened.
    with patch("charlie_work.workflow._worker_pid_alive", return_value=True):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1867"]
    assert entry.get("status") == PASSIVE_OPEN_STATUS
    assert entry.get("pr_number") == 9001

    assert len(fake_gh.prs_created) == 1
    created = fake_gh.prs_created[0]
    assert created["head"] == branch
    assert created["title"] == "fix: finalize stale worker outcome without waiting for PID exit"
    assert created["body"] == drafted_body
    assert (1867, config.labels.in_progress) in fake_gh.labels_removed
    assert (1867, config.labels.pr_open) in fake_gh.labels_added

    events = state.get("events", [])
    handoff_events = [e for e in events if e.get("kind") == "worker_handoff_pr_opened"]
    assert len(handoff_events) == 1
    payload = handoff_events[0]["payload"]
    # The "log line noting the PID was still running" the issue asks for.
    assert payload["worker_pid_still_running"] is True
    assert payload["worker_pid"] == 99999
    assert payload["pr_number"] == 9001
    assert payload["branch_name"] == branch
    assert [e for e in events if e.get("kind") == "orphaned_worker_opened_pr"] == []

    # A completed handoff is not a worker death: none of the dead-PID
    # bookkeeping may be touched for a still-running process.
    assert "worker_death_at" not in entry
    assert "orphan_redispatch_at" not in entry
    assert "orphan_drift_at" not in entry


def test_live_pid_fresh_outcome_does_not_finalize(tmp_path: Path) -> None:
    """A just-written outcome file belongs to a worker that may still be
    finishing its exit -- only an outcome older than the finalize threshold
    is treated as a stale handoff."""
    from charlie_work.paths import resolved_layout, runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers
    from charlie_work.worktree import worktree_path_for_branch

    config = _config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    branch = "agent/issue-1867-stale-outcome-live-pid"
    repo_root = _make_repo_with_pushed_branch(tmp_path, branch)
    _seed_dispatched_issue(paths, issue_number=1867, branch=branch)

    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    _write_outcome(
        worktree_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "pr_title": "fix: fresh outcome",
            "pr_body": "Closes #1867",
        },
        age_seconds=60,  # 1 minute old -- inside the 15-minute grace window
    )

    fake_gh = _fake_gh(
        repo_root,
        issue_number=1867,
        in_progress=config.labels.in_progress,
        pr_create_return=9001,
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=True):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1867"]
    assert entry.get("status") == "dispatched"
    assert "pr_number" not in entry
    assert fake_gh.prs_created == []
    assert [
        e for e in state.get("events", []) if e.get("kind") == "worker_handoff_pr_opened"
    ] == []


def test_live_pid_stale_outcome_disabled_by_config(tmp_path: Path) -> None:
    """``worker_outcome_finalize_minutes: 0`` disables the PID-independent
    finalize (the config kill switch), reverting to wait-for-PID-exit."""
    from charlie_work.paths import resolved_layout, runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers
    from charlie_work.worktree import worktree_path_for_branch

    config = _config(finalize_minutes=0)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    branch = "agent/issue-1867-stale-outcome-live-pid"
    repo_root = _make_repo_with_pushed_branch(tmp_path, branch)
    _seed_dispatched_issue(paths, issue_number=1867, branch=branch)

    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    _write_outcome(
        worktree_path,
        {"push_succeeded": True, "pr_created": False},
        age_seconds=3600,
    )

    fake_gh = _fake_gh(
        repo_root,
        issue_number=1867,
        in_progress=config.labels.in_progress,
        pr_create_return=9001,
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=True):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    assert state["issues"]["1867"].get("status") == "dispatched"
    assert fake_gh.prs_created == []


def test_live_pid_stale_non_handoff_outcome_does_not_finalize(tmp_path: Path) -> None:
    """An outcome file that does not confirm ``push_succeeded``/
    ``pr_created: false`` -- e.g. the ``blocked`` shape -- is not a completed
    handoff and must never trigger PR creation."""
    from charlie_work.paths import resolved_layout, runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers
    from charlie_work.worktree import worktree_path_for_branch

    config = _config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    branch = "agent/issue-1867-stale-outcome-live-pid"
    repo_root = _make_repo_with_pushed_branch(tmp_path, branch)
    _seed_dispatched_issue(paths, issue_number=1867, branch=branch)

    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    _write_outcome(
        worktree_path,
        {"outcome": "blocked", "reason_kind": "ambiguous_scope", "detail": "needs human"},
        age_seconds=3600,
    )

    fake_gh = _fake_gh(
        repo_root,
        issue_number=1867,
        in_progress=config.labels.in_progress,
        pr_create_return=9001,
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=True):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    assert state["issues"]["1867"].get("status") == "dispatched"
    assert fake_gh.prs_created == []
    assert [
        e for e in state.get("events", []) if e.get("kind") == "worker_handoff_pr_opened"
    ] == []


def test_live_pid_stale_outcome_skipped_when_pr_already_exists(tmp_path: Path) -> None:
    """Idempotency: if a PR already exists for the issue, the finalize lane
    must not open a duplicate."""
    from charlie_work.paths import resolved_layout, runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers
    from charlie_work.worktree import worktree_path_for_branch

    config = _config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    branch = "agent/issue-1867-stale-outcome-live-pid"
    repo_root = _make_repo_with_pushed_branch(tmp_path, branch)
    _seed_dispatched_issue(paths, issue_number=1867, branch=branch)

    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    _write_outcome(
        worktree_path,
        {"push_succeeded": True, "pr_created": False},
        age_seconds=3600,
    )

    fake_gh = _fake_gh(
        repo_root,
        issue_number=1867,
        in_progress=config.labels.in_progress,
        pr_create_return=9001,
    )
    fake_gh.prs = [
        {
            "number": 7777,
            "title": "Existing PR",
            "headRefName": branch,
            "state": "OPEN",
            "body": "Closes #1867",
            "isCrossRepository": False,
        }
    ]
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=True):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    assert fake_gh.prs_created == []
    state = load_state(paths.state_file)
    assert [
        e for e in state.get("events", []) if e.get("kind") == "worker_handoff_pr_opened"
    ] == []


def test_live_pid_stale_outcome_pr_create_failed_emits_stranded_drift_once(
    tmp_path: Path,
) -> None:
    """A failed ``gh pr create`` on the live-PID finalize lane surfaces as
    the fingerprinted ``pr_create_failed_branch_stranded`` drift -- emitted
    once per unchanged finding, retried on the next sweep, status stays
    ``dispatched`` so the next pass re-attempts."""
    from charlie_work.paths import resolved_layout, runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers
    from charlie_work.worktree import worktree_path_for_branch

    config = _config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    branch = "agent/issue-1867-stale-outcome-live-pid"
    repo_root = _make_repo_with_pushed_branch(tmp_path, branch)
    _seed_dispatched_issue(paths, issue_number=1867, branch=branch)

    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    _write_outcome(
        worktree_path,
        {"push_succeeded": True, "pr_created": False},
        age_seconds=3600,
    )

    fake_gh = _fake_gh(
        repo_root,
        issue_number=1867,
        in_progress=config.labels.in_progress,
        pr_create_return=None,  # PR create fails
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=True):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1867"]
    assert entry.get("status") == "dispatched"
    assert "pr_number" not in entry

    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "pr_create_failed_branch_stranded"]
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["reason"] == "live_worker_handoff_pr_create_failed"
    assert drift_events[0]["payload"]["worker_pid_still_running"] is True
    assert drift_events[0]["payload"]["branch_name"] == branch


def test_live_pid_without_outcome_file_does_not_finalize(tmp_path: Path) -> None:
    """A live worker with no outcome file is still legitimately working --
    the lane must leave it entirely alone (the stall watchdog owns
    liveness)."""
    from charlie_work.paths import resolved_layout, runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers
    from charlie_work.worktree import worktree_path_for_branch

    config = _config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    branch = "agent/issue-1867-stale-outcome-live-pid"
    repo_root = _make_repo_with_pushed_branch(tmp_path, branch)
    _seed_dispatched_issue(paths, issue_number=1867, branch=branch)

    # Worktree exists (worker is running in it) but no outcome file.
    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path_for_branch(repo_root, branch, worktrees_dir).mkdir(parents=True)

    fake_gh = _fake_gh(
        repo_root,
        issue_number=1867,
        in_progress=config.labels.in_progress,
        pr_create_return=9001,
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=True):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1867"]
    assert entry.get("status") == "dispatched"
    assert "pr_number" not in entry
    assert fake_gh.prs_created == []
    assert state.get("events", []) == []


def test_live_pid_no_stale_outcome_never_calls_pr_list(tmp_path: Path) -> None:
    """Round-2 review: the common case for an active fleet -- every
    dispatched worker's PID alive, none with a stale/handoff-confirmed
    outcome file -- must not pay for ``gh.pr_list()`` (network) or the
    ``state_lock`` below it. Relaxing the early return to cover "any live
    PID" instead of "an actual finalize candidate" would make both fire on
    nearly every pass of an active fleet.
    """
    from charlie_work.paths import resolved_layout, runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers
    from charlie_work.worktree import worktree_path_for_branch

    config = _config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    branch = "agent/issue-1867-live-worker-still-working"
    repo_root = _make_repo_with_pushed_branch(tmp_path, branch)
    _seed_dispatched_issue(paths, issue_number=1867, branch=branch)

    # Worktree exists (worker is running in it) but no outcome file at all --
    # the overwhelmingly common case: a live worker still mid-task.
    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path_for_branch(repo_root, branch, worktrees_dir).mkdir(parents=True)

    fake_gh = _fake_gh(
        repo_root,
        issue_number=1867,
        in_progress=config.labels.in_progress,
        pr_create_return=9001,
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch.object(fake_gh, "pr_list", wraps=fake_gh.pr_list) as pr_list_spy,
        patch("charlie_work.workflow._worker_pid_alive", return_value=True),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    pr_list_spy.assert_not_called()

    state = load_state(paths.state_file)
    entry = state["issues"]["1867"]
    assert entry.get("status") == "dispatched"
    assert "pr_number" not in entry
    assert fake_gh.prs_created == []
    assert state.get("events", []) == []


def test_live_pid_finalize_honors_workflow_patch_of_open_pr_helper(tmp_path: Path) -> None:
    """Issue #1867 round 3: the live-handoff lane reaches
    ``_open_pr_for_orphaned_branch`` through the ``charlie_work.workflow``
    module object, so patching the name there (as every orphan test does)
    intercepts this lane too instead of silently running the real helper.
    """
    from charlie_work.paths import resolved_layout, runtime_paths
    from charlie_work.workflow import _detect_and_handle_orphaned_workers
    from charlie_work.worktree import worktree_path_for_branch

    config = _config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    branch = "agent/issue-1867-patched-open-pr-helper"
    repo_root = _make_repo_with_pushed_branch(tmp_path, branch)
    _seed_dispatched_issue(paths, issue_number=1867, branch=branch)
    worktrees_dir = resolved_layout(config, repo_root).worktrees
    _write_outcome(
        worktree_path_for_branch(repo_root, branch, worktrees_dir),
        {"push_succeeded": True, "pr_created": False, "pr_title": "t", "pr_body": "Closes #1867"},
        age_seconds=3600,
    )
    fake_gh = _fake_gh(
        repo_root, issue_number=1867, in_progress=config.labels.in_progress, pr_create_return=9001
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=True),
        patch(
            "charlie_work.workflow._open_pr_for_orphaned_branch",
            return_value=(None, "stubbed failure", None),
        ) as open_pr,
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    assert open_pr.call_count == 1
    assert open_pr.call_args.kwargs["branch"] == branch
    assert fake_gh.prs_created == []
