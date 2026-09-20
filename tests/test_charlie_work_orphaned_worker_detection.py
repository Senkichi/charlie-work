"""Orphaned-worker sweep detection: liveness probes, PID recycling, bulk-sweep batching, and watchdog-disabled sweep entry.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
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
    PASSIVE_OPEN_STATUS,
    append_event,
    load_state,
    save_state,
)


def test_orphaned_worker_detection_with_request_changes_and_unchanged_head(tmp_path: Path) -> None:
    """Regression test for issue #207: dead worker with request_changes and unchanged head should reset to rework_requested."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create initial state with a dispatched issue and dead worker PID
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,  # Dead PID
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 100, "request_changes", "abc123")

    # Mock GitHub to return an open PR for the issue
    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",  # Unchanged since request_changes
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues.append(
        {
            "number": 207,
            "title": "Test issue",
            "url": "https://example.test/issues/207",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    # Mock PID liveness check to return False (dead PID)
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    # Load state and verify the transition
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # Status should be reset to rework_requested
    assert entry.get("status") == "rework_requested"
    assert entry.get("dispatched_at") is None

    # Worker PID should be preserved for recovery-path verification (issue #282)
    assert entry["worker_pid"] == 99999
    assert entry["worker_process_start_time"] == 1234567890.0

    # Verify the event was logged
    events = state.get("events", [])
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 1
    assert recovered_events[0]["payload"]["issue_number"] == 207
    assert recovered_events[0]["payload"]["pr_number"] == 100
    assert recovered_events[0]["payload"]["reason"] == "dead_worker_with_request_changes"

    # Issue #773 measurement-first requirement: even the legacy no-terminal-
    # record fallback path now reports pid/exit_code/duration_seconds on the
    # event (exit_code/duration None since no terminal record exists).
    assert recovered_events[0]["payload"]["pid"] == 99999
    assert recovered_events[0]["payload"]["exit_code"] is None
    assert recovered_events[0]["payload"]["duration_seconds"] is None


def test_orphaned_worker_detection_with_head_change(tmp_path: Path) -> None:
    """Regression test for issue #207: dead worker with head change should emit drift event, not auto-reset."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create initial state with a dispatched issue and dead worker PID
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,  # Dead PID
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",  # Old head
    }
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 100, "request_changes", "abc123")

    # Mock GitHub to return an open PR with changed head
    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "def456",  # Changed since request_changes
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues.append(
        {
            "number": 207,
            "title": "Test issue",
            "url": "https://example.test/issues/207",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    # Mock PID liveness check to return False (dead PID)
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    # Load state and verify NO auto-reset
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # Status should NOT be reset (still dispatched)
    assert entry.get("status") == "dispatched"

    # Worker PID should be preserved for recovery-path verification (issue #282)
    assert entry["worker_pid"] == 99999
    assert entry["worker_process_start_time"] == 1234567890.0

    # Verify drift event was logged
    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["issue_number"] == 207
    assert drift_events[0]["payload"]["reason"] == "dead_worker_with_head_change"


def test_orphaned_worker_detection_with_live_pid(tmp_path: Path) -> None:
    """Regression test for issue #207: live worker with matching start time should be untouched."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create initial state with a dispatched issue and live worker PID
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,  # Live PID
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)

    # Mock GitHub to return an open PR
    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()

    # Mock PID liveness check to return True (live PID) with matching start time
    with patch("charlie_work.workflow._worker_pid_alive", return_value=True):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    # Load state and verify NO changes
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # Status should remain dispatched
    assert entry.get("status") == "dispatched"

    # Worker PID should still be present
    assert entry.get("worker_pid") == 99999
    assert entry.get("worker_process_start_time") == 1234567890.0

    # Verify NO events were logged
    events = state.get("events", [])
    orphaned_events = [
        e
        for e in events
        if e.get("kind") in ("orphaned_worker_recovered", "orphaned_worker_drift")
    ]
    assert len(orphaned_events) == 0


def test_orphaned_worker_detection_with_pid_recycled(tmp_path: Path) -> None:
    """Regression test for issue #207: PID recycled (start-time mismatch) should be treated as dead."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create initial state with a dispatched issue and recycled PID
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,  # Recycled PID
        "worker_process_start_time": 1234567890.0,  # Old start time
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 100, "request_changes", "abc123")

    # Mock GitHub to return an open PR
    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",  # Unchanged
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues.append(
        {
            "number": 207,
            "title": "Test issue",
            "url": "https://example.test/issues/207",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    # Mock the helper to simulate PID recycling (alive check returns False due to start-time mismatch)
    def mock_worker_pid_alive(entry):
        # Simulate start-time mismatch by returning False even though PID is set
        return False

    with patch("charlie_work.workflow._worker_pid_alive", side_effect=mock_worker_pid_alive):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    # Load state and verify it was treated as dead
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # Status should be reset to rework_requested
    assert entry.get("status") == "rework_requested"

    # Worker PID should be preserved for recovery-path verification (issue #282)
    assert entry["worker_pid"] == 99999
    assert entry["worker_process_start_time"] == 1234567890.0

    # Verify recovered event was logged
    events = state.get("events", [])
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 1


def test_orphaned_worker_detection_no_open_pr(tmp_path: Path) -> None:
    """Regression test for issue #207: dead worker with no open PR should emit drift event (not auto-reset status)."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create initial state with a dispatched issue and dead worker PID
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,  # Dead PID
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)

    # Mock GitHub to return NO open PRs
    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubForOrphan()

    # Mock PID liveness check to return False (dead PID)
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    # Load state and verify NO status auto-reset
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # Status should NOT be reset (still dispatched)
    assert entry.get("status") == "dispatched"

    # Worker PID should be preserved for recovery-path verification (issue #282)
    assert entry["worker_pid"] == 99999
    assert entry["worker_process_start_time"] == 1234567890.0

    # Verify drift event was logged (not recovered)
    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["issue_number"] == 207
    assert drift_events[0]["payload"]["reason"] == "dead_worker_no_open_pr"

    # Verify NO recovered event
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 0

    # Issue #259: the entry should be marked so it is not re-flagged every pass.
    assert "orphan_flagged_at" in entry


def test_orphaned_worker_detection_no_open_pr_emits_once(tmp_path: Path) -> None:
    """Issue #259: sweep must emit only one drift event per zombie across N passes."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["259"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubForOrphan()

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(3):
            _detect_and_handle_orphaned_workers(
                sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
            )

    state = load_state(paths.state_file)
    drift_events = [e for e in state.get("events", []) if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1, (
        f"Expected exactly one orphaned_worker_drift event, got {len(drift_events)}"
    )
    assert drift_events[0]["payload"]["issue_number"] == 259
    assert drift_events[0]["payload"]["reason"] == "dead_worker_no_open_pr"

    entry = state["issues"]["259"]
    assert entry.get("status") == "dispatched"
    assert "orphan_flagged_at" in entry


def test_orphaned_worker_detection_bulk_sweep_excludes_pre_flagged(tmp_path: Path) -> None:
    """Issue #275 review: a sweep must aggregate only newly-flagged orphans.

    Pre-flagged entries (from #290's orphan_flagged_at guard) are suppressed
    before aggregation. A fresh bulk sweep of the remaining orphans is emitted
    as a single aggregated event.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    pre_flagged = {1, 2, 3}
    fresh = {4, 5, 6}
    for issue_number in pre_flagged | fresh:
        state["issues"][str(issue_number)] = {
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "dispatched_at": "2024-01-01T00:00:00Z",
        }
    for issue_number in pre_flagged:
        state["issues"][str(issue_number)]["orphan_flagged_at"] = "2024-01-01T00:00:00Z"
    save_state(paths.state_file, state)

    class FakeGitHubNoOrphanPrs(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubNoOrphanPrs()

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    events = state.get("events", [])

    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 0, "pre-flagged orphans must not emit individual drift events"

    sweep_events = [e for e in events if e.get("kind") == "orphaned_worker_drift_sweep"]
    assert len(sweep_events) == 1
    assert sweep_events[0]["payload"]["count"] == len(fresh)
    assert set(sweep_events[0]["payload"]["issue_numbers"]) == fresh

    for issue_number in pre_flagged | fresh:
        entry = state["issues"][str(issue_number)]
        assert entry.get("status") == "dispatched"
        assert "orphan_flagged_at" in entry


def test_orphaned_worker_detection_bulk_sweep_does_not_flood_event_buffer(tmp_path: Path) -> None:
    """Regression test for issue #275: a single bulk reap sweep must not evict unrelated diagnostic events.

    A 500-issue orphan sweep would previously emit 500 ``orphaned_worker_drift``
    events and overrun the 200-entry event buffer. The sweep now aggregates
    same-kind events into one summary event, so prior diagnostic events survive.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    diagnostic_count = 199
    for seq in range(diagnostic_count):
        state = append_event(state, "diagnostic_event", {"seq": seq})
    for issue_number in range(1, 501):
        state["issues"][str(issue_number)] = {
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "dispatched_at": "2024-01-01T00:00:00Z",
        }
    save_state(paths.state_file, state)

    class FakeGitHubNoPrs(FakeGitHub):
        def __init__(self):
            super().__init__()
            self.prs = []

    fake_gh = FakeGitHubNoPrs()

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    events = state["events"]

    # The 200-entry cap must not be exceeded and the buffer must not be flooded
    assert len(events) <= 200
    diagnostic_events = [e for e in events if e.get("kind") == "diagnostic_event"]
    assert len(diagnostic_events) == diagnostic_count

    # A single sweep summary should represent all 500 orphan drifts
    sweep_events = [e for e in events if e.get("kind") == "orphaned_worker_drift_sweep"]
    assert len(sweep_events) == 1
    assert sweep_events[0]["payload"]["count"] == 500
    assert set(sweep_events[0]["payload"]["issue_numbers"]) == set(range(1, 501))


def test_orphaned_worker_sweep_runs_with_watchdog_disabled(tmp_path: Path) -> None:
    """Issue #1122: ``_detect_and_handle_orphaned_workers`` must run even when
    ``watchdog.enabled=False``. The watchdog flag controls log-mtime stall
    detection, not the dead-pid state-keyed recovery (#935 pushed-branch
    salvage, #417 label reclaim, orphan drift diagnostics). A deployment that
    disables watchdog must not lose these backstops.
    """
    from unittest.mock import patch

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    # Set up a real git repo with a pushed worker branch.
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

    branch = "agent/issue-1122-phantom-pid-dispatch-route"
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

    # watchdog disabled — the sweep must still run.
    config = OrchestratorConfig(
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=False, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1122"] = {
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
                    "number": 1122,
                    "title": "Phantom-pid dispatch route reaps completed session",
                    "url": "https://example.test/issues/1122",
                    "body": "",
                    "labels": [{"name": in_progress}],
                    "state": "OPEN",
                }
            ]
            self.prs = []
            self.pr_create_return = 8001

        def pr_list(self):
            return []

    fake_gh = FakeGitHubForPushedBranch()
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    # The #935 salvage backstop fired despite watchdog being disabled: a PR was
    # opened for the pushed branch and the issue moved to open_passive.
    state = load_state(paths.state_file)
    entry = state["issues"]["1122"]
    assert entry.get("status") == PASSIVE_OPEN_STATUS
    assert entry.get("pr_number") == 8001

    events = state.get("events", [])
    open_pr_events = [e for e in events if e.get("kind") == "orphaned_worker_opened_pr"]
    assert len(open_pr_events) == 1
    assert open_pr_events[0]["payload"]["reason"] == "dead_worker_branch_pushed_no_pr"
    assert open_pr_events[0]["payload"]["pr_number"] == 8001

    assert len(fake_gh.prs_created) == 1
    assert fake_gh.prs_created[0]["head"] == branch
    assert (1122, in_progress) in fake_gh.labels_removed
    assert (1122, pr_open) in fake_gh.labels_added
