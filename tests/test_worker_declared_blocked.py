"""Tests for the worker-declared blocked outcome channel (issue #1453).

A worker that deliberately concludes it CANNOT do the task writes a ``blocked``
outcome in ``.worker-outcome.json`` instead of exiting PR-less with no signal.
The orphan sweep reads this file and routes the issue directly to the operator
queue on the FIRST sweep pass -- no redispatch, no cap burn.

These tests drive the real ``_detect_and_handle_orphaned_workers`` sweep
function, not an isolated helper, so a regression that drops the outcome
check would silently reintroduce the redispatch-cap burn with every unit
test green.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from charlie_work.config import DevinConfig, OrchestratorConfig, WatchdogConfig
from charlie_work.paths import resolved_layout, runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.worktree import worktree_path_for_branch
from charlie_work.write_gate import WriteGate

from _fakes_github import FakeGitHub


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


def _write_blocked_outcome(worktree_path: Path, reason_kind: str, detail: str) -> None:
    """Write a ``.worker-outcome.json`` with a ``blocked`` outcome."""
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome = {"outcome": "blocked", "reason_kind": reason_kind, "detail": detail}
    (worktree_path / ".worker-outcome.json").write_text(json.dumps(outcome), encoding="utf-8")


def test_blocked_outcome_routes_to_operator_queue_on_first_pass(
    tmp_path: Path,
) -> None:
    """A worktree containing a ``blocked`` worker outcome and no PR is routed
    to the operator queue on the FIRST sweep pass, with zero redispatches.

    Asserted by faking the outcome file in the worktree directory the sweep
    resolves for the issue's branch, then driving the real
    ``_detect_and_handle_orphaned_workers``.
    """
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    issue_number = 1453
    branch = "agent/issue-1453-test"
    state = load_state(paths.state_file)
    state["issues"][str(issue_number)] = {
        "status": "dispatched",
        "dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "branch_name": branch,
    }
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create the worktree directory the sweep resolves for this branch and
    # plant the blocked outcome file there.
    worktrees_dir = resolved_layout(config, tmp_path).worktrees
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    _write_blocked_outcome(
        worktree_path,
        reason_kind="cross_repo_scope",
        detail="The fix targets job-cannon, not this repo.",
    )

    class FakeGitHubNoPR(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubNoPR(repo_root=tmp_path)
    fake_gh.issues = [
        {
            "number": issue_number,
            "title": "test issue",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch("charlie_work.workflow.remote_branch_head_sha", return_value=None),
        patch("charlie_work.workflow.remote_branch_ahead_count", return_value=(0, None)),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            fake_gh,
            write_gate=_wg(paths.state_file),
        )

    st = load_state(paths.state_file)
    entry = st["issues"][str(issue_number)]

    # Escalated on the FIRST pass -- no redispatch cap burn.
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worker_declared_blocked"

    # Active labels removed, human-needed added (pre-lock fallback), ready NOT
    # added (no redispatch).
    assert (issue_number, config.labels.in_progress) in fake_gh.labels_removed
    assert (issue_number, config.labels.human_needed) in fake_gh.labels_added
    assert (issue_number, config.labels.ready) not in fake_gh.labels_added

    # Post-lock transition() actually applies the operator-queue label edge
    # the issue/PR title promises -- the pre-lock human_needed add above is a
    # fallback, the durable routing target is config.labels.operator_queue.
    # The operator_queued edge (resolved by _escalation_edge("escalated",
    # "mechanical")) adds operator_queue and removes every other workflow
    # label, including the pre-lock human_needed fallback, so verifying both
    # directions catches a regression that drops the post-lock transition
    # while leaving the transient pre-lock label in place.
    assert (issue_number, config.labels.operator_queue) in fake_gh.labels_added
    assert (issue_number, config.labels.human_needed) in fake_gh.labels_removed

    # The worker_declared_blocked event carries reason_kind and detail so the
    # operator queue entry is actionable without reading the worktree.
    blocked_events = [
        e for e in st.get("events", []) if e.get("kind") == "worker_declared_blocked"
    ]
    assert len(blocked_events) == 1
    payload = blocked_events[0]["payload"]
    assert payload["reason_kind"] == "cross_repo_scope"
    assert payload["detail"] == "The fix targets job-cannon, not this repo."
    assert payload["reason"] == "worker_declared_blocked"

    # Zero redispatch attempts -- the issue was escalated on the first pass,
    # not redispatched.  The orphan_redispatch_at list must have at most one
    # entry (this pass's first observation), never enough to hit the cap.
    redispatch_at = entry.get("orphan_redispatch_at", [])
    assert len(redispatch_at) <= 1

    # No orphan_sweep_redispatch_escalated event -- the cap was never reached.
    cap_events = [
        e for e in st.get("events", []) if e.get("kind") == "orphan_sweep_redispatch_escalated"
    ]
    assert len(cap_events) == 0

    # No orphaned_worker_drift event with dead_worker_no_open_pr -- the
    # blocked check fires before that classification path.
    drift_events = [
        e
        for e in st.get("events", [])
        if e.get("kind") == "orphaned_worker_drift"
        and e.get("payload", {}).get("reason") == "dead_worker_no_open_pr"
    ]
    assert len(drift_events) == 0


def test_no_outcome_file_keeps_redispatch_behavior(tmp_path: Path) -> None:
    """Regression guard: a worktree with NO outcome file and no PR keeps
    today's redispatch behavior -- the issue is NOT escalated on the first
    pass, and the redispatch counter is seeded.
    """
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    issue_number = 1454
    branch = "agent/issue-1454-test"
    state = load_state(paths.state_file)
    state["issues"][str(issue_number)] = {
        "status": "dispatched",
        "dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "branch_name": branch,
    }
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create the worktree directory but do NOT write an outcome file.
    worktrees_dir = resolved_layout(config, tmp_path).worktrees
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)

    class FakeGitHubNoPR(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubNoPR(repo_root=tmp_path)
    fake_gh.issues = [
        {
            "number": issue_number,
            "title": "test issue",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch("charlie_work.workflow.remote_branch_head_sha", return_value=None),
        patch("charlie_work.workflow.remote_branch_ahead_count", return_value=(0, None)),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            fake_gh,
            write_gate=_wg(paths.state_file),
        )

    st = load_state(paths.state_file)
    entry = st["issues"][str(issue_number)]

    # NOT escalated -- the redispatch behavior is preserved.
    assert entry["status"] == "dispatched"
    assert entry.get("escalation_reason") is None

    # No worker_declared_blocked event.
    blocked_events = [
        e for e in st.get("events", []) if e.get("kind") == "worker_declared_blocked"
    ]
    assert len(blocked_events) == 0

    # The redispatch counter is seeded (first observation), not escalated.
    redispatch_at = entry.get("orphan_redispatch_at", [])
    assert len(redispatch_at) == 1
