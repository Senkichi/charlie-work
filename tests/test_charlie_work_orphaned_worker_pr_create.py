"""Orphaned-worker PR creation: pushed-branch PR open and the distinct pr_create_failed stranded-drift signals (emission and dedup).

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

from pathlib import Path
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from charlie_work.config import WorkerRoleConfig


def test_orphaned_worker_pushed_branch_opens_pr(tmp_path: Path) -> None:
    """Issue #935: a dead worker that pushed a branch but could not open a PR
    should have its PR opened by the orchestrator instead of being re-dispatched.
    """
    import subprocess
    from unittest.mock import patch

    from charlie_work.config import DevinConfig, OrchestratorConfig, WatchdogConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Set up a real git repo with an origin and a pushed worker branch.
    remote_repo = tmp_path / "remote"
    remote_repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--bare", str(remote_repo)], check=True, capture_output=True, text=True
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    for cmd in (
        ["git", "config", "user.email", "test@example.test"],
        ["git", "config", "user.name", "Test User"],
    ):
        subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True, text=True)
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"], cwd=repo_root, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(remote_repo)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    branch = "agent/issue-935-workers-push-a-finished-branch-but-cannot-open-t"
    subprocess.run(
        ["git", "checkout", "-b", branch],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo_root / "fix.txt").write_text("fix\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "fix.txt"], cwd=repo_root, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "fix"], cwd=repo_root, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "push", "-u", "origin", branch],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "checkout", "main"], cwd=repo_root, check=True, capture_output=True, text=True
    )

    state = load_state(paths.state_file)
    state["issues"]["935"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "branch_name": branch,
    }
    save_state(paths.state_file, state)

    in_progress = config.labels.in_progress
    pr_open = config.labels.pr_open

    class FakeGitHubForPushedBranch(FakeGitHub):
        def __init__(self) -> None:
            super().__init__(repo_root=repo_root, dry_run=False)
            self.issues = [
                {
                    "number": 935,
                    "title": "Workers push a finished branch but cannot open the PR",
                    "url": "https://example.test/issues/935",
                    "body": "Workers cannot open PRs because gh is unauthenticated.",
                    "labels": [{"name": in_progress}],
                    "state": "OPEN",
                }
            ]
            self.prs = []
            self.pr_create_return = 9001

        def pr_list(self):
            return []

    fake_gh = FakeGitHubForPushedBranch()
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["935"]
    assert entry.get("status") == PASSIVE_OPEN_STATUS
    assert entry.get("pr_number") == 9001

    events = state.get("events", [])
    open_pr_events = [e for e in events if e.get("kind") == "orphaned_worker_opened_pr"]
    assert len(open_pr_events) == 1
    assert open_pr_events[0]["payload"]["reason"] == "dead_worker_branch_pushed_no_pr"
    assert open_pr_events[0]["payload"]["pr_number"] == 9001
    assert open_pr_events[0]["payload"]["branch_name"] == branch

    assert len(fake_gh.prs_created) == 1
    assert fake_gh.prs_created[0]["head"] == branch
    assert (935, in_progress) in fake_gh.labels_removed
    assert (935, pr_open) in fake_gh.labels_added


def test_orphaned_worker_reported_push_pr_create_failed_emits_distinct_drift(
    tmp_path: Path,
) -> None:
    """Issue #935: a worker-reported push with a PR-create failure must not be
    treated as a no-open-PR orphan; it emits a distinct drift and stays
    dispatched so a human sees the real error.
    """
    from unittest.mock import patch

    from charlie_work.config import DevinConfig, OrchestratorConfig, WatchdogConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.process_utils import (
        worker_terminal_status_path,
        write_worker_terminal_status,
    )
    from charlie_work.state import load_state, save_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    branch = "agent/issue-935-workers-push-a-finished-branch-but-cannot-open-t"
    state = load_state(paths.state_file)
    state["issues"]["935"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "branch_name": branch,
    }
    save_state(paths.state_file, state)

    # The worker wrote a terminal outcome claiming push succeeded but PR failed.
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = worker_terminal_status_path(sessions_dir, 935, "claude")
    write_worker_terminal_status(
        terminal_path,
        pid=1234,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:05:00Z",
        duration_seconds=300.0,
        worker_outcome={
            "push_succeeded": True,
            "pr_created": False,
            "error": "gh unauthenticated",
        },
    )

    in_progress = config.labels.in_progress

    class FakeGitHubForFailedPr(FakeGitHub):
        def __init__(self) -> None:
            super().__init__(repo_root=tmp_path / "not-a-repo")
            self.issues = [
                {
                    "number": 935,
                    "title": "Workers push a finished branch but cannot open the PR",
                    "url": "https://example.test/issues/935",
                    "body": "Workers cannot open PRs because gh is unauthenticated.",
                    "labels": [{"name": in_progress}],
                    "state": "OPEN",
                }
            ]
            self.prs = []
            self.pr_create_return = None  # PR create fails

        def pr_list(self):
            return []

    fake_gh = FakeGitHubForFailedPr()

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["935"]
    assert entry.get("status") == "dispatched"
    assert "pr_number" not in entry

    events = state.get("events", [])
    # cw#1273: this specific reason now emits its own kind
    # (pr_create_failed_branch_stranded) instead of the generic
    # orphaned_worker_drift, after the bounded outer retry (3 attempts)
    # exhausted -- see workflow.py's orphan-reap sweep.
    drift_events = [e for e in events if e.get("kind") == "pr_create_failed_branch_stranded"]
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["reason"] == "dead_worker_branch_pushed_pr_create_failed"
    assert drift_events[0]["payload"]["pr_create_error"] is not None
    assert drift_events[0]["payload"]["worker_reported"] is True
    assert drift_events[0]["payload"]["branch_name"] == branch

    # Must NOT emit the generic no-open-PR drift, which would cause re-dispatch.
    no_pr_events = [
        e
        for e in events
        if e.get("kind") == "orphaned_worker_drift"
        and e["payload"].get("reason") == "dead_worker_no_open_pr"
    ]
    assert not no_pr_events


def test_orphaned_worker_pr_create_failed_stranded_drift_dedups_on_repeat_sweep(
    tmp_path: Path,
) -> None:
    """cw#1273 AC4: a repeated ``pr_create_failed_branch_stranded`` terminal
    for the same branch/error within one ``orphan_drift_fingerprint`` window
    must emit only once -- proven by running the orphan-reap sweep twice
    against state that still has no PR number and the same failing
    ``gh pr create``, and asserting the event count stays at 1. This rides
    the pre-existing ``_drift_fingerprint``/``orphan_drift_fingerprint``
    dedup mechanism unchanged (see workflow.py's orphan-reap sweep) rather
    than inventing a second dedup layer for the new event kind.
    """
    from unittest.mock import patch

    from charlie_work.config import DevinConfig, OrchestratorConfig, WatchdogConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.process_utils import (
        worker_terminal_status_path,
        write_worker_terminal_status,
    )
    from charlie_work.state import load_state, save_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    branch = "agent/issue-935-workers-push-a-finished-branch-but-cannot-open-t"
    state = load_state(paths.state_file)
    state["issues"]["935"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "branch_name": branch,
    }
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = worker_terminal_status_path(sessions_dir, 935, "claude")
    write_worker_terminal_status(
        terminal_path,
        pid=1234,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:05:00Z",
        duration_seconds=300.0,
        worker_outcome={
            "push_succeeded": True,
            "pr_created": False,
            "error": "gh unauthenticated",
        },
    )

    in_progress = config.labels.in_progress

    class FakeGitHubForFailedPr(FakeGitHub):
        def __init__(self) -> None:
            super().__init__(repo_root=tmp_path / "not-a-repo")
            self.issues = [
                {
                    "number": 935,
                    "title": "Workers push a finished branch but cannot open the PR",
                    "url": "https://example.test/issues/935",
                    "body": "Workers cannot open PRs because gh is unauthenticated.",
                    "labels": [{"name": in_progress}],
                    "state": "OPEN",
                }
            ]
            self.prs = []
            self.pr_create_return = None  # PR create fails, every attempt, forever.

        def pr_list(self):
            return []

    fake_gh = FakeGitHubForFailedPr()

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["935"]
    assert entry.get("status") == "dispatched"
    assert "orphan_drift_fingerprint" in entry

    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "pr_create_failed_branch_stranded"]
    assert len(drift_events) == 1, (
        f"expected exactly one pr_create_failed_branch_stranded event across two "
        f"sweep passes within the same fingerprint window, got {len(drift_events)}"
    )
    assert drift_events[0]["payload"]["reason"] == "dead_worker_branch_pushed_pr_create_failed"
    assert drift_events[0]["payload"]["branch_name"] == branch

    # Positive control: the assertion above is only meaningful if pass 2
    # actually reached the emit site and was suppressed BY THE FINGERPRINT --
    # not because some earlier state mutation from pass 1 (status, sidecar
    # consumption, an unrelated early `continue`) made pass 2 exit before
    # ever getting there. Clear `orphan_drift_fingerprint` -- the same
    # "force re-observation" pattern this file already uses elsewhere (e.g.
    # `test_orphaned_worker_drift_fingerprint_cleared_on_redispatch`) -- and
    # sweep again: if the mechanism under test is real, this MUST produce a
    # second event, since nothing else changed.
    state = load_state(paths.state_file)
    state["issues"]["935"].pop("orphan_drift_fingerprint", None)
    save_state(paths.state_file, state)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "pr_create_failed_branch_stranded"]
    assert len(drift_events) == 2, (
        f"expected a second pr_create_failed_branch_stranded event once the "
        f"fingerprint was cleared (proving the sweep re-reaches the emit site "
        f"and the first assertion's count==1 was real dedup, not an unrelated "
        f"early exit), got {len(drift_events)}"
    )
    assert drift_events[1]["payload"]["branch_name"] == branch
