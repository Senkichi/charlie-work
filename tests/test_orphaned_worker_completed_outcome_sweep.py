"""Orphan-sweep recovery for a dead worker's completed ``.worker-outcome.json`` (issue #1911).

Split out of ``tests/test_charlie_work_orphaned_worker_sweep.py`` during the
#1911 rework: that file crossed the 800-line file-size cap. A dead devin
rework session never gets a terminal-status record (only claude_code's
launch path runs ``start_terminal_status_watcher``), so
``terminal_exit_code`` is None even when the session completed, pushed, and
wrote a fresh, on-target ``.worker-outcome.json``. Before crediting a worker
death, the sweep must route that outcome into the #1877 apply path
(``rework_outcome.apply_collected_rework_outcomes``).
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


# ---------------------------------------------------------------------------
# Issue #1911: a dead devin rework session never gets a terminal-status record
# (only claude_code's launch path runs start_terminal_status_watcher), so
# ``terminal_exit_code`` is None even when the session completed, pushed, and
# wrote a fresh, on-target .worker-outcome.json. Before crediting a worker
# death, the sweep must route that outcome into the #1877 apply path.
# ---------------------------------------------------------------------------


def _dead_worker_rework_bed(
    tmp_path: Path,
    *,
    decision: str = "request_changes",
    pr_state_status: str | None = None,
) -> tuple[Any, Any, Any, str]:
    """Seed the shared #1911 scenario.

    Returns ``(config, paths, fake_gh, dispatched_at)``: a dead-PID dispatched
    issue 207 whose open PR 100 carries a ``decision`` verdict on the
    unchanged live head ``abc123``, with ``gh.repo_root`` pointed at
    ``tmp_path`` so the sweep resolves a real worktrees dir.
    ``pr_state_status`` sets the PR state entry's ``status`` field -- the
    post-approval rework lanes mark it ``rework_requested`` while leaving
    ``decision="approved"`` (issue #1109).
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
        "decision": decision,
        "reviewed_head_sha": "abc123",
    }
    if pr_state_status is not None:
        state["prs"]["100"]["status"] = pr_state_status
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 100, decision, "abc123")

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


def test_orphaned_worker_fresh_outcome_approved_rework_is_not_a_death(
    tmp_path: Path,
) -> None:
    """Approved + PR-state ``rework_requested`` branch of the same recovery.

    The post-approval rework lane (#1109) dispatches workers on PRs whose
    ``decision`` is ``approved`` and whose PR-state ``status`` is
    ``rework_requested``. A dead devin worker there gets the identical
    #1911 recovery -- no death credit, no reset -- and the drift payload
    records the branch's decision/status provenance.
    """
    from unittest.mock import patch

    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="approved", pr_state_status="rework_requested"
    )
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

    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "dispatched"
    assert entry.get("worker_death_at") is None
    events = state.get("events", [])
    assert [e for e in events if e.get("kind") == "orphaned_worker_recovered"] == []
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    payload = drift_events[0]["payload"]
    assert payload["reason"] == "dead_worker_completed_outcome"
    assert payload["decision"] == "approved"
    assert payload["pr_state_status"] == "rework_requested"
    assert payload["exit_code"] is None
    assert payload["worker_outcome_head_sha"] == "abc123"
    assert len(fake_gh.pr_edits) == 1
    applied = [e for e in events if e.get("kind") == "rework_outcome_applied"]
    assert len(applied) == 1


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
