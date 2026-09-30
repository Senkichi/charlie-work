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
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
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
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
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
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
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


def test_leftover_terminal_outcome_older_than_dispatch_is_not_laundered_as_fresh(
    tmp_path: Path,
) -> None:
    """B5 (wf-review-opus.md), rule 1: the terminal-status watcher copies
    whatever ``.worker-outcome.json`` sits in the worktree at process exit,
    stamped with a fresh ``ended_at`` -- the moment the watcher observed
    exit, not when that outcome file was actually written. A worktree reused
    across dispatches can still hold a PREVIOUS dispatch's leftover outcome
    if the new dispatch died before writing its own, and nothing deletes the
    file at dispatch time.

    Scenario: dispatch #1 writes a ``blocked`` outcome and dies. Dispatch #2
    starts later, reuses the worktree, and also dies -- without ever writing
    its own outcome -- so the terminal-status watcher copies dispatch #1's
    stale ``blocked`` outcome into dispatch #2's terminal record with a
    fresh ``ended_at``. Both the worktree file itself (mtime before
    dispatch #2 started) and the terminal record's own
    ``worker_outcome_written_at`` (the same, honest mtime -- the additive
    field ``process_utils.write_worker_terminal_status`` now populates)
    correctly mark this evidence as older than ``dispatched_at``. Before the
    fix, only ``ended_at`` was available as the outcome's freshness anchor,
    so the leftover passed rule 1's ``written_at > dispatched_at`` gate and
    the issue was escalated as blocked on a dispatch that made no such
    declaration.
    """
    from charlie_work.process_utils import write_worker_terminal_status
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    issue_number = 1455
    branch = "agent/issue-1455-test"

    leftover_written_at = "2026-09-29T00:00:00Z"
    leftover_mtime_epoch = datetime(2026, 9, 29, 0, 0, 0, tzinfo=UTC).timestamp()
    dispatch_2_started_at = "2026-09-29T00:10:00Z"
    watcher_ended_at = "2026-09-29T00:20:00Z"

    state = load_state(paths.state_file)
    state["issues"][str(issue_number)] = {
        "status": "dispatched",
        "dispatched_at": dispatch_2_started_at,
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "branch_name": branch,
    }
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch #1's leftover outcome file, still sitting in the reused
    # worktree, its mtime pinned BEFORE dispatch #2 started.
    worktrees_dir = resolved_layout(config, tmp_path).worktrees
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    _write_blocked_outcome(
        worktree_path,
        reason_kind="cross_repo_scope",
        detail="dispatch #1's stale declaration",
    )
    outcome_path = worktree_path / ".worker-outcome.json"
    os.utime(outcome_path, (leftover_mtime_epoch, leftover_mtime_epoch))

    # Dispatch #2's terminal record: watcher-observed exit is fresh
    # (after dispatched_at), but it copied dispatch #1's leftover outcome
    # file, and worker_outcome_written_at honestly reports that file's own
    # (stale) mtime.
    terminal_path = sessions_dir / f"issue-{issue_number}.devin.terminal.json"
    write_worker_terminal_status(
        terminal_path,
        pid=99999,
        exit_code=1,
        started_at=dispatch_2_started_at,
        ended_at=watcher_ended_at,
        duration_seconds=600.0,
        worker_outcome={
            "outcome": "blocked",
            "reason_kind": "cross_repo_scope",
            "detail": "dispatch #1's stale declaration",
        },
        worker_outcome_written_at=leftover_written_at,
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

    # NOT escalated -- dispatch #1's leftover outcome is correctly stale
    # relative to dispatch #2's dispatched_at, so it must not decide
    # dispatch #2's fate.
    assert entry["status"] == "dispatched", (
        f"leftover outcome laundered as fresh: status={entry['status']!r}"
    )
    assert entry.get("escalation_reason") is None

    blocked_events = [
        e for e in st.get("events", []) if e.get("kind") == "worker_declared_blocked"
    ]
    assert len(blocked_events) == 0


def test_stale_leftover_outcome_emits_worker_evidence_stale_event(tmp_path: Path) -> None:
    """B6 (wf-review-opus.md), design doc §5: rule 1 dropping a stale
    candidate must emit a ``worker_evidence_stale`` warning event so the
    operator has a signal in events.db -- before this fix, ``FateBasis.stale``
    was populated but nothing read it and no event existed at all.

    Same leftover-outcome-via-terminal-record scenario as
    ``test_leftover_terminal_outcome_older_than_dispatch_is_not_laundered_as_fresh``
    (B5): both the worktree file and the terminal record's copy of it are
    correctly recognized as stale, which is exactly what must now surface a
    signal instead of silently vanishing.
    """
    from charlie_work.instrumentation import query_events
    from charlie_work.process_utils import write_worker_terminal_status
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    issue_number = 1456
    branch = "agent/issue-1456-test"

    leftover_written_at = "2026-09-29T00:00:00Z"
    leftover_mtime_epoch = datetime(2026, 9, 29, 0, 0, 0, tzinfo=UTC).timestamp()
    dispatch_2_started_at = "2026-09-29T00:10:00Z"
    watcher_ended_at = "2026-09-29T00:20:00Z"

    state = load_state(paths.state_file)
    state["issues"][str(issue_number)] = {
        "status": "dispatched",
        "dispatched_at": dispatch_2_started_at,
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "branch_name": branch,
        "adapter": "devin",
    }
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    worktrees_dir = resolved_layout(config, tmp_path).worktrees
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    _write_blocked_outcome(
        worktree_path,
        reason_kind="cross_repo_scope",
        detail="dispatch #1's stale declaration",
    )
    outcome_path = worktree_path / ".worker-outcome.json"
    os.utime(outcome_path, (leftover_mtime_epoch, leftover_mtime_epoch))

    terminal_path = sessions_dir / f"issue-{issue_number}.devin.terminal.json"
    write_worker_terminal_status(
        terminal_path,
        pid=99999,
        exit_code=1,
        started_at=dispatch_2_started_at,
        ended_at=watcher_ended_at,
        duration_seconds=600.0,
        worker_outcome={
            "outcome": "blocked",
            "reason_kind": "cross_repo_scope",
            "detail": "dispatch #1's stale declaration",
        },
        worker_outcome_written_at=leftover_written_at,
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

    stale_events = query_events(paths.state_file, kind="worker_evidence_stale")
    assert len(stale_events) >= 1, "no worker_evidence_stale signal reached events.db"
    for event in stale_events:
        assert event["level"] == "warning"
        raw_payload = event["payload"]
        payload = json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
        assert payload["issue_number"] == issue_number
        assert payload["reason"] == "older_than_dispatch"
        assert payload["adapter"] == "devin"
        assert payload["dispatched_at"] == dispatch_2_started_at

    # Dedup marker persisted so a second pass over the same still-dead,
    # still-stale worker does not re-emit the same candidates.
    st = load_state(paths.state_file)
    entry = st["issues"][str(issue_number)]
    assert entry.get("stale_evidence_reported"), "dedup marker was never persisted"
    reported_before = list(entry["stale_evidence_reported"])
    count_before = len(query_events(paths.state_file, kind="worker_evidence_stale"))

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

    st_after = load_state(paths.state_file)
    assert st_after["issues"][str(issue_number)]["stale_evidence_reported"] == reported_before
    count_after = len(query_events(paths.state_file, kind="worker_evidence_stale"))
    assert count_after == count_before, "same stale candidate re-emitted on the next pass"


def test_permission_denial_blocked_outcome_is_not_operator_escalated(tmp_path: Path) -> None:
    """Issue #2010: a ``blocked`` outcome that is the headless permission-denial
    signature must not escalate to the operator as a blocked task."""
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    issue_number = 2010
    branch = "agent/issue-2010-test"
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

    worktrees_dir = resolved_layout(config, tmp_path).worktrees
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    _write_blocked_outcome(
        worktree_path,
        reason_kind="other",
        detail="Bash was denied. If you approve command execution, I can finish these steps.",
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

    st = load_state(paths.state_file)
    entry = st["issues"][str(issue_number)]
    assert entry.get("escalation_reason") != "worker_declared_blocked"
    assert entry["status"] != "escalated"
    assert not [e for e in st.get("events", []) if e.get("kind") == "worker_declared_blocked"]
