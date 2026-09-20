"""Orphaned-worker salvage push: stranded-commit recovery before classification, dry-run threading, failure preserving classification, up-to-date no-event, and cross-repository PR skip.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch
from _dead_session_fixtures import _write_flat_review_decision
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
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
)


def test_orphaned_worker_salvage_push_recovers_stranded_commits_before_classification(
    tmp_path: Path,
) -> None:
    """A successful pre-lock salvage push refreshes the PR snapshot's
    headRefOid before classification runs. A dead worker whose
    request_changes review matched the PRE-push head must NOT be
    misclassified as a no-op crash (reason=dead_worker_with_request_changes,
    which also burns a worker_death_at entry) -- it takes the head-changed
    path instead, and the salvage push itself is recorded.
    """
    import charlie_work.workflow as workflow_module
    from charlie_work.worktree import SalvagePushResult

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1248"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["500"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 500, "request_changes", "abc123")

    class FakeGitHubForSalvage(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 500,
                    "headRefOid": "abc123",  # pre-push: unchanged since review
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-1248",
                }
            ]

    fake_gh = FakeGitHubForSalvage(repo_root=tmp_path)
    fake_gh.issues.append(
        {
            "number": 1248,
            "title": "Test issue",
            "url": "https://example.test/issues/1248",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    salvage_calls: list[tuple[Any, ...]] = []

    def fake_salvage(repo_root, branch, worktree_path, *, base_ref="", dry_run=False):
        salvage_calls.append((repo_root, branch, worktree_path, base_ref))
        return SalvagePushResult(
            pushed=True,
            old_remote_sha="abc123",
            new_remote_sha="newsha123",
            commit_count=1,
        )

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch.object(workflow_module, "salvage_push_stranded_commits", fake_salvage),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    assert len(salvage_calls) == 1
    assert salvage_calls[0][1] == "agent/issue-1248"

    state = load_state(paths.state_file)
    entry = state["issues"]["1248"]

    # NOT the no-op-crash death path: no worker_death_at entry was recorded.
    assert not entry.get("worker_death_at")

    events = state.get("events", [])
    recovered_events = [
        e
        for e in events
        if e.get("kind") == "orphaned_worker_recovered"
        and e.get("payload", {}).get("reason") == "dead_worker_with_request_changes"
    ]
    assert recovered_events == []

    # Instead, the head-changed drift path fired (no review_callback wired
    # into this call, so it surfaces as orphaned_worker_drift rather than a
    # review route).
    drift_events = [
        e
        for e in events
        if e.get("kind") == "orphaned_worker_drift"
        and e.get("payload", {}).get("reason") == "dead_worker_with_head_change"
    ]
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["live_head_sha"] == "newsha123"
    assert drift_events[0]["payload"]["reviewed_head_sha"] == "abc123"

    # The salvage push itself is recorded in the same event stream.
    salvage_events = [e for e in events if e.get("kind") == "salvage_pushed_stranded_commits"]
    assert len(salvage_events) == 1
    assert salvage_events[0]["payload"]["issue_number"] == 1248
    assert salvage_events[0]["payload"]["pr_number"] == 500
    assert salvage_events[0]["payload"]["new_remote_sha"] == "newsha123"


def test_orphaned_worker_salvage_push_threads_dry_run_to_salvage_push_stranded_commits(
    tmp_path: Path,
) -> None:
    """Issue #1326 regression: ``_detect_and_handle_orphaned_workers`` must
    thread ``write_gate.dry_run`` into ``salvage_push_stranded_commits`` (the
    higher-traffic call site at workflow.py:2218-2224). Under a dry-run
    WriteGate the salvage stand-in must receive ``dry_run=True`` -- the
    existing ``fake_salvage`` stand-ins in this file were only patched for
    signature compatibility and never asserted the value, so a regression
    that drops the kwarg would pass silently. This test pins the value.
    """
    import charlie_work.workflow as workflow_module
    from charlie_work.worktree import SalvagePushResult

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1326"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)

    class FakeGitHubForSalvage(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 500,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-1326",
                }
            ]

    fake_gh = FakeGitHubForSalvage(repo_root=tmp_path)
    # Issue #1229: the branch-issue validator in _detect_and_handle_orphaned_workers
    # rejects branch-name bindings to issues absent from the open-issue list, so
    # #1326 must be seeded as an OPEN issue for the pr_by_issue binding to survive
    # and the salvage loop to fire.
    fake_gh.issues.append(
        {
            "number": 1326,
            "title": "Test issue",
            "url": "https://example.test/issues/1326",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    salvage_calls: list[dict[str, Any]] = []

    def fake_salvage(repo_root, branch, worktree_path, *, base_ref="", dry_run=False):
        salvage_calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "worktree_path": worktree_path,
                "base_ref": base_ref,
                "dry_run": dry_run,
            }
        )
        return SalvagePushResult(
            pushed=True,
            old_remote_sha="abc123",
            new_remote_sha="newsha123",
            commit_count=1,
        )

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch.object(workflow_module, "salvage_push_stranded_commits", fake_salvage),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            fake_gh,
            write_gate=_wg(paths.state_file, dry_run=True),
        )

    # The salvage stand-in must have been called exactly once for the orphaned
    # issue's branch, and -- the point of this test -- dry_run must be True.
    assert len(salvage_calls) == 1
    assert salvage_calls[0]["branch"] == "agent/issue-1326"
    assert salvage_calls[0]["dry_run"] is True


def test_orphaned_worker_salvage_push_failure_preserves_existing_classification(
    tmp_path: Path,
) -> None:
    """A failed salvage push must not change downstream classification: the
    PR snapshot's headRefOid is left untouched (push never succeeded), so the
    pre-existing dead_worker_with_request_changes -> rework_requested reset
    (with its worker_death_at counter) fires exactly as it would without the
    feature. The failure itself is recorded as a separate event.
    """
    import charlie_work.workflow as workflow_module
    from charlie_work.worktree import SalvagePushResult

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1248"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["500"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 500, "request_changes", "abc123")

    class FakeGitHubForSalvage(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 500,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-1248",
                }
            ]

    fake_gh = FakeGitHubForSalvage(repo_root=tmp_path)
    fake_gh.issues.append(
        {
            "number": 1248,
            "title": "Test issue",
            "url": "https://example.test/issues/1248",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    def fake_salvage(repo_root, branch, worktree_path, *, base_ref="", dry_run=False):
        return SalvagePushResult(
            pushed=False,
            error="push_failed: network timeout",
            old_remote_sha="abc123",
            commit_count=2,
        )

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch.object(workflow_module, "salvage_push_stranded_commits", fake_salvage),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1248"]

    # Same classification as without the feature: head unchanged since
    # review -> reset to rework_requested, counted as a worker death.
    assert entry.get("status") == "rework_requested"
    assert entry.get("dispatched_at") is None
    assert len(entry.get("worker_death_at") or []) == 1

    events = state.get("events", [])
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 1
    assert recovered_events[0]["payload"]["reason"] == "dead_worker_with_request_changes"

    salvage_events = [e for e in events if e.get("kind") == "salvage_push_failed"]
    assert len(salvage_events) == 1
    assert salvage_events[0]["payload"]["issue_number"] == 1248
    assert salvage_events[0]["payload"]["error"] == "push_failed: network timeout"
    # No success event was also recorded.
    assert [e for e in events if e.get("kind") == "salvage_pushed_stranded_commits"] == []


def test_orphaned_worker_salvage_push_up_to_date_emits_no_event(tmp_path: Path) -> None:
    """skip_reason="up_to_date" is silent by design (issue #1248 docstring):
    no salvage event is recorded and classification is identical to running
    without the feature at all.
    """
    import charlie_work.workflow as workflow_module
    from charlie_work.worktree import SalvagePushResult

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1248"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["500"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 500, "request_changes", "abc123")

    class FakeGitHubForSalvage(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 500,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-1248",
                }
            ]

    fake_gh = FakeGitHubForSalvage(repo_root=tmp_path)
    fake_gh.issues.append(
        {
            "number": 1248,
            "title": "Test issue",
            "url": "https://example.test/issues/1248",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    def fake_salvage(repo_root, branch, worktree_path, *, base_ref="", dry_run=False):
        return SalvagePushResult(pushed=False, skip_reason="up_to_date", old_remote_sha="abc123")

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch.object(workflow_module, "salvage_push_stranded_commits", fake_salvage),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1248"]

    # Identical behavior to pre-feature: head unchanged -> rework_requested.
    assert entry.get("status") == "rework_requested"
    assert len(entry.get("worker_death_at") or []) == 1

    events = state.get("events", [])
    salvage_events = [
        e
        for e in events
        if e.get("kind") in ("salvage_pushed_stranded_commits", "salvage_push_failed")
    ]
    assert salvage_events == []


def test_orphaned_worker_salvage_push_skips_cross_repository_pr(tmp_path: Path) -> None:
    """A fork PR (isCrossRepository=True) must never have its branch pushed
    to from this checkout -- the pre-lock salvage loop must skip calling
    salvage_push_stranded_commits entirely for that issue.
    """
    import charlie_work.workflow as workflow_module
    from charlie_work.worktree import SalvagePushResult

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1249"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)

    class FakeGitHubForSalvage(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 501,
                    "headRefOid": "abc999",
                    "isCrossRepository": True,
                    "headRepository": {"owner": {"login": "fork-owner"}, "name": "repo"},
                    "headRefName": "agent/issue-1249",
                }
            ]

    fake_gh = FakeGitHubForSalvage(repo_root=tmp_path)

    salvage_calls: list[tuple[Any, ...]] = []

    def fake_salvage(repo_root, branch, worktree_path, *, base_ref="", dry_run=False):
        salvage_calls.append((repo_root, branch, worktree_path, base_ref))
        return SalvagePushResult(pushed=False, skip_reason="should_never_be_called")

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch.object(workflow_module, "salvage_push_stranded_commits", fake_salvage),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    assert salvage_calls == []
