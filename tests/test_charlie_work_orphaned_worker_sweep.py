"""Orphaned-worker sweep bookkeeping: death-at recording, request-changes recovery, clean-exit and terminal-record recovery, and bulk issue-list skipping.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from _dead_session_fixtures import _write_flat_review_decision
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from charlie_work import rework_outcome
from charlie_work.config import (
    WORKER_OUTCOME_FILENAME,
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
from charlie_work.worktree import worktree_path_for_branch


def test_orphaned_worker_sweep_records_worker_death_at_in_state(tmp_path: Path) -> None:
    """Issue #1134: ``_detect_and_handle_orphaned_workers`` must write
    ``worker_death_at`` into the issue's state entry (and the
    ``orphaned_worker_recovered`` event payload) when it recovers a dead
    rework worker whose PR head has not moved.  The death timestamp is what
    the no-op cap check in ``_dispatch_rework_impl`` subtracts from the
    redispatch count to separate genuine no-ops from worker deaths — without
    it, every death counts as a no-op and produces a false
    ``no_op_rework_cap_exceeded`` escalation.

    This test exercises the *production* side (the sweep itself writing the
    timestamp), complementing the existing ``test_dispatch_rework_*`` tests
    which only seed ``worker_death_at`` directly to exercise the consumption
    side.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["207"] = {
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
                    "headRefOid": "abc123",
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

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = sessions_dir / "issue-207.claude.terminal.json"
    terminal_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 1,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:05Z",
                "duration_seconds": 5.0,
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
    entry = state["issues"]["207"]
    assert entry.get("status") == "rework_requested"

    # The sweep must have recorded a worker_death_at timestamp.
    death_at = entry.get("worker_death_at")
    assert isinstance(death_at, list)
    assert len(death_at) == 1
    assert isinstance(death_at[0], str)
    # The timestamp must be a valid ISO 8601 string.
    from datetime import datetime

    datetime.fromisoformat(death_at[0].replace("Z", "+00:00"))

    # The event payload must also carry the death timestamp.
    events = state.get("events", [])
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 1
    payload = recovered_events[0]["payload"]
    assert payload["reason"] == "dead_worker_with_request_changes"
    assert payload.get("worker_death_at") == death_at[0]


def test_orphaned_worker_request_changes_recovered_with_watchdog_disabled(
    tmp_path: Path,
) -> None:
    """Issue #1108: the dead-pid orphan recovery sweep must reset a
    ``status=dispatched`` issue with a dead PID and an open PR carrying a
    ``request_changes`` verdict (head unchanged) to ``rework_requested`` even
    when ``watchdog.enabled=False``.

    The ``watchdog.enabled`` flag controls log-mtime stall detection
    (``_detect_stalled_sessions`` / ``_detect_and_handle_stalled_sessions``),
    not the dead-pid state-keyed recovery in
    ``_detect_and_handle_orphaned_workers``. A deployment that disables
    watchdog (e.g. job-cannon, to work around shim log-mtime blindness) must
    not lose the request_changes → rework_requested transition — without it,
    issues with dead workers and open PRs sit wedged in ``dispatched``
    indefinitely with no path to redispatch (the exact 8+ hour stall observed
    2026-08-09/10 on jc #1358, #1479, et al.).
    """
    from unittest.mock import patch

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    # watchdog disabled — the sweep must still run.
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=False, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1108"] = {
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
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-1108",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues.append(
        {
            "number": 1108,
            "title": "Test issue",
            "url": "https://example.test/issues/1108",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1108"]

    # The sweep fired despite watchdog being disabled: the issue was reset to
    # rework_requested so dispatch_rework can re-select it.
    assert entry.get("status") == "rework_requested"
    assert entry.get("dispatched_at") is None

    events = state.get("events", [])
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 1
    assert recovered_events[0]["payload"]["issue_number"] == 1108
    assert recovered_events[0]["payload"]["pr_number"] == 100
    assert recovered_events[0]["payload"]["reason"] == "dead_worker_with_request_changes"


def test_orphaned_worker_clean_exit_not_reset_to_rework(tmp_path: Path) -> None:
    """Issue #773: a worker that exited 0 (clean, no-op) must not be reset to
    rework_requested or burn a redispatch attempt, even though its dead PID and
    unchanged head otherwise look identical to a crash under
    ``_worker_pid_alive`` alone.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["207"] = {
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

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    # Durable terminal-status record a real worker's watcher thread would have
    # written at exit (process_utils.start_terminal_status_watcher): exit
    # code 0 means the worker completed cleanly rather than crashing.
    terminal_path = sessions_dir / "issue-207.claude.terminal.json"
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
    entry = state["issues"]["207"]

    # Must NOT be reset to rework_requested -- that would burn a redispatch
    # attempt on a worker that never had anything to change.
    assert entry.get("status") == "dispatched"
    assert entry.get("dispatched_at") == "2024-01-01T00:00:00Z"
    assert entry["worker_pid"] == 99999

    events = state.get("events", [])
    assert [e for e in events if e.get("kind") == "orphaned_worker_recovered"] == []
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    payload = drift_events[0]["payload"]
    assert payload["reason"] == "dead_worker_clean_exit_no_op"
    assert payload["pid"] == 99999
    assert payload["exit_code"] == 0
    assert payload["duration_seconds"] == 300.0


def test_orphaned_worker_with_flag_and_open_pr_request_changes_recovered(tmp_path: Path) -> None:
    """Issue #259 review: orphan suppression must not block open-PR recovery paths."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "orphan_flagged_at": "2024-01-01T00:00:00Z",
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
                    "headRefOid": "abc123",
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

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # With an open PR and request_changes with unchanged head, recovery should
    # run regardless of the orphan_flagged_at suppression.
    assert entry.get("status") == "rework_requested"
    assert entry.get("dispatched_at") is None
    # Worker PID should be preserved for recovery-path verification (issue #282)
    assert entry["worker_pid"] == 99999
    assert entry["worker_process_start_time"] == 1234567890.0

    events = state.get("events", [])
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 1
    assert recovered_events[0]["payload"]["reason"] == "dead_worker_with_request_changes"


def test_orphaned_worker_crash_with_terminal_record_still_recovered(tmp_path: Path) -> None:
    """Issue #773: a non-zero exit code recorded in the terminal-status file
    must still take the pre-#773 recovery path (reset to rework_requested) --
    the fix only special-cases a confirmed clean (exit code 0) exit, never a
    confirmed abnormal one.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["207"] = {
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
                    "headRefOid": "abc123",
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

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = sessions_dir / "issue-207.claude.terminal.json"
    terminal_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 1,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:05Z",
                "duration_seconds": 5.0,
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
    entry = state["issues"]["207"]
    assert entry.get("status") == "rework_requested"
    assert entry.get("dispatched_at") is None

    events = state.get("events", [])
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 1
    payload = recovered_events[0]["payload"]
    assert payload["reason"] == "dead_worker_with_request_changes"
    assert payload["pid"] == 99999
    assert payload["exit_code"] == 1
    assert payload["duration_seconds"] == 5.0


def test_orphaned_worker_no_pr_orphans_skips_bulk_issue_list(tmp_path: Path) -> None:
    """Regression test for issue #996.

    The #417 ground-truth label-reclaim sweep only calls the bounded-cost
    ``gh.issue_list(state="open")`` when ``no_pr_orphans`` (dead-PID orphans
    with no linked open PR) is non-empty -- that guard is also what makes the
    later ``issues_by_number.get(...)`` read reachable. This pins the other
    side of that invariant: when the only orphan already has a linked open
    PR, ``no_pr_orphans`` is empty and ``issue_list`` must never be called.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["207"] = {
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

    # Issue #1362 Stage 1: the reader is now file-first, so the decision
    # must exist on disk (not just in state.json) for `.missing` to be False.
    pr_decision_dir = paths.prs / "pr-100"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "abc123"}),
        encoding="utf-8",
    )

    class FakeGitHubForOrphan(FakeGitHub):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.issue_list_calls = 0

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

        def issue_list(self, labels=None, state=None):
            self.issue_list_calls += 1
            return super().issue_list(labels=labels, state=state)

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

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    # Issue 207 has a linked open PR (number 100), so no_pr_orphans is empty
    # and the bulk issue-list sweep must not run. The single issue_list call
    # is the branch-issue validator's own open-issue fetch (issue #1229), not
    # the bulk reclaim sweep.
    assert fake_gh.issue_list_calls == 1


# ---------------------------------------------------------------------------
# Issue #1911: a dead devin rework session never gets a terminal-status record
# (only claude_code's launch path runs start_terminal_status_watcher), so
# ``terminal_exit_code`` is None even when the session completed, pushed, and
# wrote a fresh, on-target .worker-outcome.json. Before crediting a worker
# death, the sweep must route that outcome into the #1877 apply path.
# ---------------------------------------------------------------------------


def _dead_worker_rework_bed(tmp_path: Path) -> tuple[Any, Any, Any, str]:
    """Seed the shared #1911 scenario.

    Returns ``(config, paths, fake_gh, dispatched_at)``: a dead-PID dispatched
    issue 207 whose open PR 100 carries a ``request_changes`` verdict on the
    unchanged live head ``abc123``, with ``gh.repo_root`` pointed at
    ``tmp_path`` so the sweep resolves a real worktrees dir.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    dispatched_at = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": dispatched_at,
        "branch_name": "agent/issue-207",
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
                    "headRefOid": "abc123",  # Unchanged since request_changes
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan(repo_root=tmp_path)
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
    return config, paths, fake_gh, dispatched_at


def _write_outcome(
    paths: Any, tmp_path: Path, payload: dict[str, Any], *, mtime: datetime | None = None
) -> Path:
    worktree_path = worktree_path_for_branch(tmp_path, "agent/issue-207", paths.worktrees)
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / WORKER_OUTCOME_FILENAME
    outcome_path.write_text(json.dumps(payload), encoding="utf-8")
    if mtime is not None:
        ts = mtime.timestamp()
        os.utime(outcome_path, (ts, ts))
    return outcome_path


def _run_orphan_sweep(tmp_path: Path, paths: Any, config: Any, fake_gh: Any) -> None:
    from unittest.mock import patch

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )


def test_orphaned_worker_fresh_outcome_without_terminal_record_is_not_a_death(
    tmp_path: Path,
) -> None:
    """Acceptance: dead worker, no terminal record, fresh on-target outcome.

    The sweep must apply the outcome through the #1877 path (the PR body the
    worker drafted lands on the PR) and surface once via fingerprinted drift --
    not reset to ``rework_requested`` and credit a worker death, which is what
    drove the swole#198 no_op_rework_attempts_cap_exceeded loop.
    """
    from unittest.mock import patch

    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",
            "pr_body": "Closes #207\n\nCorrected PR body per review.",
        },
    )
    # No terminal-status file in sessions_dir -- the devin launch path never
    # writes one.

    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    # Not credited as a death: no reset, no worker_death_at.
    assert entry.get("status") == "dispatched"
    assert entry.get("worker_death_at") is None
    events = state.get("events", [])
    assert [e for e in events if e.get("kind") == "orphaned_worker_recovered"] == []
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    payload = drift_events[0]["payload"]
    assert payload["reason"] == "dead_worker_completed_outcome"
    assert payload["exit_code"] is None
    assert payload["worker_outcome_head_sha"] == "abc123"

    # The outcome reached the #1877 apply path: the drafted PR body landed.
    assert len(fake_gh.pr_edits) == 1
    assert fake_gh.pr_edits[0][0] == 100
    assert "Corrected PR body per review" in fake_gh.pr_edits[0][1]
    applied = [e for e in events if e.get("kind") == "rework_outcome_applied"]
    assert len(applied) == 1
    assert state.get("rework_outcome_applied_heads", {}).get("207") == "abc123"


def test_orphaned_worker_stale_outcome_without_terminal_record_is_a_death(
    tmp_path: Path,
) -> None:
    """An outcome file from a PREVIOUS dispatch (mtime before dispatched_at)
    must not suppress the death credit -- otherwise a redispatch would read
    the prior round's leftover outcome as proof of completion forever."""
    config, paths, fake_gh, dispatched_at = _dead_worker_rework_bed(tmp_path)
    stale_mtime = datetime.fromisoformat(dispatched_at.replace("Z", "+00:00")) - timedelta(
        minutes=5
    )
    _write_outcome(
        paths,
        tmp_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",
            "pr_body": "stale leftover",
        },
        mtime=stale_mtime,
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "rework_requested"
    assert isinstance(entry.get("worker_death_at"), list)
    events = state.get("events", [])
    recovered = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered) == 1
    assert recovered[0]["payload"]["reason"] == "dead_worker_with_request_changes"
    assert fake_gh.pr_edits == []


def test_orphaned_worker_outcome_at_other_head_is_a_death(tmp_path: Path) -> None:
    """A fresh outcome pinning a head other than the live head describes a
    different remote state -- not proof this dispatch completed."""
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "other-head-sha",
            "pr_body": "different head",
        },
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "rework_requested"
    assert isinstance(entry.get("worker_death_at"), list)
    events = state.get("events", [])
    recovered = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered) == 1
    assert fake_gh.pr_edits == []


def test_orphaned_worker_recorded_crash_with_fresh_outcome_is_a_death(
    tmp_path: Path,
) -> None:
    """Boundary pin: a terminal record carrying a confirmed non-zero exit is
    still a crash even when a fresh, on-target outcome file exists -- the
    #1911 recovery only fills the *missing* terminal-record case
    (``terminal_exit_code is None``), matching the issue's suggested fix."""
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",
            "pr_body": "finished then crashed",
        },
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "issue-207.devin.terminal.json").write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 1,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:05:00Z",
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "rework_requested"
    assert isinstance(entry.get("worker_death_at"), list)
    events = state.get("events", [])
    recovered = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered) == 1
    assert recovered[0]["payload"]["exit_code"] == 1
    assert fake_gh.pr_edits == []
