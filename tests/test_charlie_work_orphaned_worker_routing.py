"""Orphaned-worker classification routing: merge-conflict and stale-empty-checks rework routes, head-advanced review route, unsafe-to-auto-reset drift, and approved-rework dead-worker reset.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

import json
from pathlib import Path
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
from charlie_work.workflow import CommandResult


def test_orphaned_worker_routes_merge_conflict_to_rework(tmp_path: Path) -> None:
    """Issue #439: a dead worker with a CONFLICTING open PR is routed to rework."""
    from datetime import UTC, datetime

    from charlie_work.config import AutoMergeConfig
    from charlie_work.state import load_state, save_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(required_checks=("Tests passed", "Lint & Format", "Pre-commit"))
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub(repo_root=tmp_path)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    fake_gh.prs = [
        {
            "number": 1,
            "title": "Fix #42: search",
            "url": "https://example.test/pull/1",
            "headRefName": "agent/issue-42-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeable": "CONFLICTING",
            "mergeStateStatus": "DIRTY",
            "body": "Closes #42",
            "state": "OPEN",
            "labels": [],
            "isCrossRepository": False,
            "updatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "statusCheckRollup": [],
        }
    ]

    state = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "issues": {
            "42": {
                "number": 42,
                "status": "dispatched",
                "worker_pid": 9999999,
                "worker_process_start_time": 1234567890.0,
                "redispatch_at": [],
            }
        },
        "prs": {},
        "events": [],
    }
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    _detect_and_handle_orphaned_workers(
        sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
    )

    state = load_state(paths.state_file)
    assert state["issues"]["42"]["status"] == "rework_requested"
    assert state["issues"]["42"]["pre_review_rework_reason"] == "merge_conflict"
    assert state["issues"]["42"]["worker_pid"] == 9999999
    assert state["issues"]["42"]["worker_process_start_time"] == 1234567890.0
    assert state["prs"]["1"]["status"] == "rework_requested"
    assert (42, config.labels.needs_rework) in fake_gh.labels_added
    assert (42, config.labels.in_progress) in fake_gh.labels_removed

    prompt_path = paths.prs / "pr-1" / "rework-prompt.md"
    assert prompt_path.exists()
    assert "merge conflict" in prompt_path.read_text(encoding="utf-8").lower()


def test_orphaned_worker_routes_stale_empty_checks_to_rework(tmp_path: Path) -> None:
    """Issue #439: a dead worker with an old PR and empty statusCheckRollup is routed to rework."""
    from datetime import UTC, datetime, timedelta

    from charlie_work.config import AutoMergeConfig
    from charlie_work.state import load_state, save_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(required_checks=("Tests passed", "Lint & Format", "Pre-commit"))
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub(repo_root=tmp_path)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    old_updated = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    fake_gh.prs = [
        {
            "number": 1,
            "title": "Fix #42: search",
            "url": "https://example.test/pull/1",
            "headRefName": "agent/issue-42-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #42",
            "state": "OPEN",
            "labels": [],
            "isCrossRepository": False,
            "updatedAt": old_updated,
            "statusCheckRollup": [],
        }
    ]

    state = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "issues": {
            "42": {
                "number": 42,
                "status": "dispatched",
                "worker_pid": 9999999,
                "worker_process_start_time": 1234567890.0,
                "redispatch_at": [],
            }
        },
        "prs": {},
        "events": [],
    }
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    _detect_and_handle_orphaned_workers(
        sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
    )

    state = load_state(paths.state_file)
    assert state["issues"]["42"]["status"] == "rework_requested"
    assert state["issues"]["42"]["pre_review_rework_reason"] == "stale_empty_checks"
    assert state["issues"]["42"]["worker_pid"] == 9999999
    assert state["issues"]["42"]["worker_process_start_time"] == 1234567890.0
    assert state["prs"]["1"]["status"] == "rework_requested"
    assert (42, config.labels.needs_rework) in fake_gh.labels_added
    assert (42, config.labels.in_progress) in fake_gh.labels_removed

    prompt_path = paths.prs / "pr-1" / "rework-prompt.md"
    assert prompt_path.exists()
    prompt_text = prompt_path.read_text(encoding="utf-8").lower()
    assert "no ci checks" in prompt_text


def test_orphaned_worker_head_advanced_routes_to_review(tmp_path: Path) -> None:
    """Issue #457: dead worker with request_changes and an advanced head is routed
    to the review-pending path instead of being re-emitted as drift."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["457"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 100, "request_changes", "abc123")

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "def456",  # Changed since request_changes
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-457",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues.append(
        {
            "number": 457,
            "title": "Test issue",
            "url": "https://example.test/issues/457",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    def fake_review(pr_number: int):
        return CommandResult(True, "review packet generated", {"pr_number": pr_number})

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            fake_gh,
            review_callback=fake_review,
            write_gate=_wg(paths.state_file),
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["457"]

    # Should transition to the review-pending state, not stay dispatched.
    assert entry.get("status") == "reviewing"

    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 0

    routed_events = [e for e in events if e.get("kind") == "orphaned_worker_routed_to_review"]
    assert len(routed_events) == 1
    assert routed_events[0]["payload"]["issue_number"] == 457
    assert routed_events[0]["payload"]["pr_number"] == 100
    assert routed_events[0]["payload"]["review_ok"] is True
    assert routed_events[0]["payload"]["routed"] is True


def test_orphaned_worker_head_advanced_review_failure_emits_drift_once(tmp_path: Path) -> None:
    """Issue #457: if routing to review fails, the head-advance finding is emitted
    as a single drift event and not re-emitted on subsequent passes."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["457"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 100, "request_changes", "abc123")

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "def456",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-457",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues.append(
        {
            "number": 457,
            "title": "Test issue",
            "url": "https://example.test/issues/457",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    def fake_review(pr_number: int):
        return CommandResult(False, "janitor gate blocked review", {"pr_number": pr_number})

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            fake_gh,
            review_callback=fake_review,
            write_gate=_wg(paths.state_file),
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["457"]

    # Status should remain dispatched; the finding is tracked as drift.
    assert entry.get("status") == "dispatched"
    assert "orphan_drift_fingerprint" in entry

    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["reason"] == "dead_worker_with_head_change"

    routed_events = [e for e in events if e.get("kind") == "orphaned_worker_routed_to_review"]
    assert len(routed_events) == 0

    # Second pass must not re-emit the drift.
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            fake_gh,
            review_callback=fake_review,
            write_gate=_wg(paths.state_file),
        )

    state = load_state(paths.state_file)
    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1, "drift must not be re-emitted for the same fingerprint"


def test_orphaned_worker_unsafe_to_auto_reset_drift_emits_once(tmp_path: Path) -> None:
    """Issue #457: non-request_changes dead workers emit a drift finding once and
    are not re-emitted on every subsequent pass."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["457"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    # Last decision is "approved" instead of "request_changes".
    state["prs"]["100"] = {
        "decision": "approved",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 100, "approved", "abc123")

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-457",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues.append(
        {
            "number": 457,
            "title": "Test issue",
            "url": "https://example.test/issues/457",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(3):
            _detect_and_handle_orphaned_workers(
                sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
            )

    state = load_state(paths.state_file)
    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["reason"] == "dead_worker_unsafe_to_auto_reset"


def test_orphaned_worker_approved_rework_dead_worker_auto_resets(tmp_path: Path) -> None:
    """Issue #1109: a dead worker on an approved PR whose PR state carries
    ``status="rework_requested"`` (evidence the post-approval rework lane
    dispatched this worker) and whose head is unchanged since review must be
    auto-reset to ``rework_requested`` -- not wedged in ``dispatched`` via
    ``dead_worker_unsafe_to_auto_reset``.

    This is the post-approval CI-failure rework lane (#674 -> PR #685): the
    PR's decision stays ``approved`` while a rework worker is dispatched to
    fix failing checks. If that worker dies at launch (crash wave, reboot,
    OOM) without pushing, the sweep must reset so the normal redispatch lane
    can retry, subject to the same death counter and redispatch caps as the
    request_changes branch.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1109"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    # Approved PR whose post-approval rework lane set status=rework_requested
    # (via _route_to_rework) and preserved the approved decision + reviewed head.
    state["prs"]["100"] = {
        "decision": "approved",
        "status": "rework_requested",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 100, "approved", "abc123")

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",  # Unchanged since approved review
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-1109",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues.append(
        {
            "number": 1109,
            "title": "Test issue",
            "url": "https://example.test/issues/1109",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1109"]

    # Status must be reset to rework_requested so the redispatch lane can retry.
    assert entry.get("status") == "rework_requested"
    assert entry.get("dispatched_at") is None

    # Worker PID preserved for recovery-path verification (issue #282).
    assert entry["worker_pid"] == 99999
    assert entry["worker_process_start_time"] == 1234567890.0

    # Worker death recorded (issue #1134 death counter).
    assert isinstance(entry.get("worker_death_at"), list)
    assert len(entry["worker_death_at"]) == 1

    events = state.get("events", [])
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 1
    payload = recovered_events[0]["payload"]
    assert payload["issue_number"] == 1109
    assert payload["pr_number"] == 100
    assert payload["reason"] == "dead_worker_with_approved_rework"
    assert payload["decision"] == "approved"
    assert payload["pr_state_status"] == "rework_requested"
    assert payload["pid"] == 99999
    assert payload["exit_code"] is None
    assert payload["duration_seconds"] is None

    # No drift event -- this is a recovery, not an unclassifiable finding.
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert drift_events == []


def test_orphaned_worker_approved_rework_clean_exit_no_op_drift(tmp_path: Path) -> None:
    """Issue #1109: a dead worker on an approved+rework_requested PR that
    exited cleanly (exit code 0) without pushing must surface as
    ``dead_worker_clean_exit_no_op`` drift, not auto-reset -- mirroring the
    #773 clean-exit-no-op sub-case of the request_changes branch so a benign
    no-op worker does not burn redispatch attempts.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1109"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["100"] = {
        "decision": "approved",
        "status": "rework_requested",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 100, "approved", "abc123")

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-1109",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues.append(
        {
            "number": 1109,
            "title": "Test issue",
            "url": "https://example.test/issues/1109",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = sessions_dir / "issue-1109.claude.terminal.json"
    terminal_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 0,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:05:00Z",
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1109"]

    # Must NOT be reset -- clean exit with no push is a no-op, not a crash.
    assert entry.get("status") == "dispatched"
    assert entry.get("dispatched_at") == "2024-01-01T00:00:00Z"

    events = state.get("events", [])
    assert [e for e in events if e.get("kind") == "orphaned_worker_recovered"] == []
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    payload = drift_events[0]["payload"]
    assert payload["reason"] == "dead_worker_clean_exit_no_op"
    assert payload["decision"] == "approved"
    assert payload["pr_state_status"] == "rework_requested"
    assert payload["exit_code"] == 0
    assert payload["duration_seconds"] == 300.0
