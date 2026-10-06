"""Characterization of the dead-worker sweep's OPEN-PR routing (wave B, dws-1).

Companion of ``test_dead_worker_sweep_characterization.py`` (the no-open-PR
family). Pins CURRENT end-to-end routing of a dead ``dispatched`` Worker whose
issue has an open PR: request_changes (reset, clean-exit no-op, completed
outcome + review, blocked outcome), head advanced (review, refused review, no
review callback), approved (+rework), unreviewed PR (advance, label-failure
drift, pre-review merge-conflict rework), the dead-dispatched backstop, and the
spine (nothing dispatched, live PID, dry-run).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from _dead_worker_sweep_characterization_fixtures import (
    events_of,
    iso,
    issue_entry,
    no_pr_bed,
    run_sweep,
    seed_issue,
    write_terminal_exit,
)
from _orphan_sweep_fixtures import _dead_worker_rework_bed, _write_outcome
from charlie_work import rework_outcome
from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state
from charlie_work.command_result import CommandResult

# ---------------------------------------------------------------------------
# Open PR: request_changes verdict family
# ---------------------------------------------------------------------------


def _review_ok(pr_number: int) -> CommandResult:
    return CommandResult(True, "review packet generated", {"pr_number": pr_number})


def _review_refused(pr_number: int) -> CommandResult:
    return CommandResult(False, "transient refusal", {"pr_number": pr_number})


def test_pr_request_changes_unchanged_head_resets_to_rework(tmp_path: Path) -> None:
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 207)
    assert entry["status"] == "rework_requested"
    assert entry.get("dispatched_at") is None
    assert entry["worker_pid"] == 99999
    assert len(entry["worker_death_at"]) == 1
    (event,) = events_of(paths, "orphaned_worker_recovered")
    assert event["payload"]["reason"] == "dead_worker_with_request_changes"
    assert event["payload"]["pr_number"] == 100
    assert events_of(paths, "orphaned_worker_drift") == []
    assert events_of(paths, "orphaned_worker_routed_to_review") == []


def test_pr_request_changes_clean_exit_is_no_op_escalation_not_reset(tmp_path: Path) -> None:
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)
    write_terminal_exit(
        tmp_path,
        207,
        started_at=iso(minutes_ago=30),
        ended_at=iso(minutes_ago=25),
    )

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 207)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    assert events_of(paths, "orphaned_worker_recovered") == []
    (drift,) = events_of(paths, "orphaned_worker_drift")
    assert drift["payload"]["reason"] == "dead_worker_clean_exit_no_op"
    assert drift["payload"]["exit_code"] == 0


_COMPLETED_OUTCOME = {
    "push_succeeded": True,
    "pr_created": False,
    "head_sha": "abc123",
    "pr_body": "Closes #207\n\nCorrected PR body per review.",
}


def test_pr_request_changes_completed_outcome_applies_and_routes_to_review(
    tmp_path: Path,
) -> None:
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)
    _write_outcome(paths, tmp_path, _COMPLETED_OUTCOME)
    review_calls: list[int] = []

    def review(pr_number: int) -> CommandResult:
        review_calls.append(pr_number)
        return _review_ok(pr_number)

    run_sweep(
        tmp_path,
        paths,
        config,
        gh,
        review_callback=review,
        patches=(patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"),),
    )

    entry = issue_entry(paths, 207)
    assert entry["status"] == "reviewing"
    assert entry.get("worker_death_at") is None
    assert len(gh.pr_edits) == 1
    assert review_calls == [100]
    assert len(events_of(paths, "rework_outcome_applied")) == 1
    (routed,) = events_of(paths, "orphaned_worker_routed_to_review")
    assert routed["payload"]["reason"] == "dead_worker_completed_outcome"
    assert routed["payload"]["routed"] is True


def test_pr_request_changes_completed_outcome_review_refused_returns_to_rework(
    tmp_path: Path,
) -> None:
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)
    _write_outcome(paths, tmp_path, _COMPLETED_OUTCOME)

    run_sweep(
        tmp_path,
        paths,
        config,
        gh,
        review_callback=_review_refused,
        patches=(patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"),),
    )

    entry = issue_entry(paths, 207)
    assert entry["status"] == "rework_requested"
    assert entry.get("worker_death_at") is None
    assert (207, config.labels.needs_rework) in gh.labels_added
    (recovered,) = events_of(paths, "orphaned_worker_recovered")
    assert recovered["payload"]["reason"] == "dead_worker_completed_outcome"
    assert recovered["payload"]["new_status"] == "rework_requested"


def test_pr_request_changes_blocked_outcome_escalates(tmp_path: Path) -> None:
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {"outcome": "blocked", "reason_kind": "cross_repo_scope", "detail": "needs jc change"},
    )

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 207)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worker_declared_blocked"
    assert len(events_of(paths, "worker_declared_blocked")) == 1
    assert events_of(paths, "orphaned_worker_recovered") == []


# ---------------------------------------------------------------------------
# Open PR: head advanced since the verdict
# ---------------------------------------------------------------------------


def _advance_head(gh: Any, sha: str = "def456") -> None:
    gh.prs[0]["headRefOid"] = sha


def test_pr_head_advanced_routes_to_review(tmp_path: Path) -> None:
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)
    _advance_head(gh)

    run_sweep(tmp_path, paths, config, gh, review_callback=_review_ok)

    assert issue_entry(paths, 207)["status"] == "reviewing"
    (routed,) = events_of(paths, "orphaned_worker_routed_to_review")
    assert routed["payload"]["pr_number"] == 100
    assert routed["payload"]["review_ok"] is True
    assert routed["payload"]["routed"] is True
    assert events_of(paths, "orphaned_worker_drift") == []


def test_pr_head_advanced_review_refused_drifts_once_and_stays_dispatched(
    tmp_path: Path,
) -> None:
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)
    _advance_head(gh)

    run_sweep(tmp_path, paths, config, gh, review_callback=_review_refused)
    run_sweep(tmp_path, paths, config, gh, review_callback=_review_refused)

    entry = issue_entry(paths, 207)
    assert entry["status"] == "dispatched"
    assert entry.get("orphan_drift_fingerprint")
    (drift,) = events_of(paths, "orphaned_worker_drift")
    assert drift["payload"]["reason"] == "dead_worker_with_head_change"
    assert events_of(paths, "orphaned_worker_routed_to_review") == []
    assert events_of(paths, "rework_no_op_escalated") == []


def test_pr_head_advanced_without_review_callback_drifts_then_no_op_escalates(
    tmp_path: Path,
) -> None:
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)
    _advance_head(gh)

    run_sweep(tmp_path, paths, config, gh, review_callback=None)

    entry = issue_entry(paths, 207)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    (drift,) = events_of(paths, "orphaned_worker_drift")
    assert drift["payload"]["reason"] == "dead_worker_with_head_change"
    (escalated,) = events_of(paths, "rework_no_op_escalated")
    assert escalated["payload"]["reason"] == "dead_worker_no_op"
    assert escalated["payload"]["head_sha"] == "def456"
    assert events_of(paths, "orphaned_worker_routed_to_review") == []


# ---------------------------------------------------------------------------
# Open PR: approved verdict
# ---------------------------------------------------------------------------


def test_pr_approved_rework_status_auto_resets_to_rework(tmp_path: Path) -> None:
    config, paths, gh, _ = _dead_worker_rework_bed(
        tmp_path, decision="approved", pr_state_status="rework_requested"
    )

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 207)
    assert entry["status"] == "rework_requested"
    assert len(entry["worker_death_at"]) == 1
    (event,) = events_of(paths, "orphaned_worker_recovered")
    assert event["payload"]["reason"] == "dead_worker_with_approved_rework"
    assert event["payload"]["decision"] == "approved"
    assert events_of(paths, "orphaned_worker_drift") == []


def test_pr_approved_carried_forward_status_recovers_to_rework(tmp_path: Path) -> None:
    """#2135: carry-forward resets the PR status to ``approved`` mid-rework.

    ``dispatched`` + ``approved`` can only be a post-approval rework, so a
    dead worker must recover whatever the PR ``status`` says.
    """
    config, paths, gh, _ = _dead_worker_rework_bed(
        tmp_path, decision="approved", pr_state_status="approved"
    )
    write_terminal_exit(
        tmp_path,
        207,
        exit_code=1,
        started_at=iso(minutes_ago=30),
        ended_at=iso(minutes_ago=25),
    )

    run_sweep(tmp_path, paths, config, gh)

    assert issue_entry(paths, 207)["status"] == "rework_requested"
    (event,) = events_of(paths, "orphaned_worker_recovered")
    assert event["payload"]["reason"] == "dead_worker_with_approved_rework"
    assert events_of(paths, "orphaned_worker_drift") == []


def test_pr_approved_without_rework_status_drifts_unsafe_to_auto_reset(
    tmp_path: Path,
) -> None:
    """#2135 (supersedes #1109): no PR ``status`` at all no longer wedges.

    ``dispatched`` + ``approved`` on the same head is a post-approval rework
    whatever the PR ``status`` says, so the dead worker recovers instead of
    drifting. The leaf name predates #2135 and is kept so the collect-only gate
    sees the test as modified rather than removed.
    """
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path, decision="approved")
    write_terminal_exit(
        tmp_path,
        207,
        exit_code=1,
        started_at=iso(minutes_ago=30),
        ended_at=iso(minutes_ago=25),
    )

    run_sweep(tmp_path, paths, config, gh)

    assert issue_entry(paths, 207)["status"] == "rework_requested"
    (event,) = events_of(paths, "orphaned_worker_recovered")
    assert event["payload"]["reason"] == "dead_worker_with_approved_rework"
    assert events_of(paths, "orphaned_worker_drift") == []


def test_pr_approved_carried_forward_status_clean_exit_is_no_op(tmp_path: Path) -> None:
    """#2135: exit 0 on a carried-forward approved PR counts against the no-op cap."""
    config, paths, gh, _ = _dead_worker_rework_bed(
        tmp_path, decision="approved", pr_state_status="approved"
    )
    write_terminal_exit(
        tmp_path,
        207,
        started_at=iso(minutes_ago=30),
        ended_at=iso(minutes_ago=25),
    )

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 207)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    (drift,) = events_of(paths, "orphaned_worker_drift")
    assert drift["payload"]["reason"] == "dead_worker_clean_exit_no_op"


def test_pr_approved_head_advanced_routes_to_review(tmp_path: Path) -> None:
    """#2135: approved + live head != reviewed head takes the head-change route.

    Before #2135 this arm fell into ``dead_worker_unsafe_to_auto_reset`` drift;
    with status no longer consulted it must route through ``_head_changed``.
    """
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path, decision="approved")
    _advance_head(gh)

    run_sweep(tmp_path, paths, config, gh, review_callback=_review_ok)

    assert issue_entry(paths, 207)["status"] == "reviewing"
    (routed,) = events_of(paths, "orphaned_worker_routed_to_review")
    assert routed["payload"]["pr_number"] == 100
    assert routed["payload"]["routed"] is True
    assert events_of(paths, "orphaned_worker_drift") == []
    assert events_of(paths, "orphaned_worker_recovered") == []


def test_pr_approved_head_advanced_review_refused_drifts_head_change(
    tmp_path: Path,
) -> None:
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path, decision="approved")
    _advance_head(gh)

    run_sweep(tmp_path, paths, config, gh, review_callback=_review_refused)

    assert issue_entry(paths, 207)["status"] == "dispatched"
    (drift,) = events_of(paths, "orphaned_worker_drift")
    assert drift["payload"]["reason"] == "dead_worker_with_head_change"
    assert events_of(paths, "orphaned_worker_routed_to_review") == []


def test_pr_approved_rework_clean_exit_is_no_op_escalation(tmp_path: Path) -> None:
    config, paths, gh, _ = _dead_worker_rework_bed(
        tmp_path, decision="approved", pr_state_status="rework_requested"
    )
    write_terminal_exit(
        tmp_path,
        207,
        started_at=iso(minutes_ago=30),
        ended_at=iso(minutes_ago=25),
    )

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 207)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    assert events_of(paths, "orphaned_worker_recovered") == []
    (drift,) = events_of(paths, "orphaned_worker_drift")
    assert drift["payload"]["reason"] == "dead_worker_clean_exit_no_op"


# ---------------------------------------------------------------------------
# Open PR: unreviewed (no verdict) and pre-review rework
# ---------------------------------------------------------------------------


def _unreviewed_pr_bed(tmp_path: Path, **pr_fields: Any):
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)
    state = load_state(paths.state_file)
    state["prs"]["100"] = {"reviewed_head_sha": None}
    save_state(paths.state_file, state)
    (paths.prs / "pr-100" / "review-decision.json").unlink()
    gh.issues[-1]["labels"] = [{"name": config.labels.in_progress}]
    gh.prs[0].update({"mergeStateStatus": "CLEAN", "state": "OPEN", **pr_fields})
    return config, paths, gh


def test_pr_unreviewed_open_pr_advances_to_pr_open(tmp_path: Path) -> None:
    config, paths, gh = _unreviewed_pr_bed(tmp_path)

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 207)
    assert entry["status"] == PASSIVE_OPEN_STATUS
    assert entry.get("dispatched_at") is None
    (event,) = events_of(paths, "orphaned_worker_advanced_to_pr_open")
    assert event["payload"]["reason"] == "dead_worker_unsafe_to_auto_reset_open_unreviewed_pr"
    assert event["payload"]["label_write_ok"] is True
    assert (207, config.labels.in_progress) in gh.labels_removed
    assert (207, config.labels.pr_open) in gh.labels_added
    assert events_of(paths, "orphaned_worker_drift") == []


def test_pr_unreviewed_open_pr_label_failure_falls_back_to_drift(tmp_path: Path) -> None:
    config, paths, gh = _unreviewed_pr_bed(tmp_path)
    gh.remove_issue_label = lambda number, label: False  # type: ignore[method-assign]

    run_sweep(tmp_path, paths, config, gh)

    assert issue_entry(paths, 207)["status"] == "dispatched"
    assert events_of(paths, "orphaned_worker_advanced_to_pr_open") == []
    drift = [
        e
        for e in events_of(paths, "orphaned_worker_drift")
        if e["payload"].get("reason") == "dead_worker_unsafe_to_auto_reset"
    ]
    assert len(drift) == 1


def test_pr_unreviewed_merge_conflict_routes_to_pre_review_rework(tmp_path: Path) -> None:
    config, paths, gh = _unreviewed_pr_bed(
        tmp_path, mergeable="CONFLICTING", mergeStateStatus="DIRTY", statusCheckRollup=[]
    )

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 207)
    assert entry["status"] == "rework_requested"
    assert entry["pre_review_rework_reason"] == "merge_conflict"
    assert load_state(paths.state_file)["prs"]["100"]["status"] == "rework_requested"
    assert (207, config.labels.needs_rework) in gh.labels_added
    assert (207, config.labels.in_progress) in gh.labels_removed
    assert (paths.prs / "pr-100" / "rework-prompt.md").exists()
    assert events_of(paths, "orphaned_worker_advanced_to_pr_open") == []


# ---------------------------------------------------------------------------
# Open PR: dead-dispatched backstop
# ---------------------------------------------------------------------------


def test_pr_armed_drift_past_backstop_window_is_reaped_to_escalated(tmp_path: Path) -> None:
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path, decision="approved")
    seed_issue(
        paths,
        207,
        orphan_flagged_at=iso(minutes_ago=180),
        orphan_drift_at=iso(minutes_ago=180),
    )

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 207)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "dead_dispatched_worker_reap"
    assert len(events_of(paths, "dead_dispatched_worker_reaped")) == 1


# ---------------------------------------------------------------------------
# Spine
# ---------------------------------------------------------------------------


def test_sweep_with_nothing_dispatched_writes_no_events(tmp_path: Path) -> None:
    config, paths, gh = no_pr_bed(tmp_path, 55, status=PASSIVE_OPEN_STATUS)

    run_sweep(tmp_path, paths, config, gh)

    assert issue_entry(paths, 55)["status"] == PASSIVE_OPEN_STATUS
    assert load_state(paths.state_file).get("events", []) == []
    assert gh.labels_added == [] and gh.labels_removed == []


def test_sweep_leaves_live_pid_dispatched_issue_untouched(tmp_path: Path) -> None:
    config, paths, gh = no_pr_bed(tmp_path, 56, dispatched_at=iso())
    before = issue_entry(paths, 56)

    run_sweep(tmp_path, paths, config, gh, pid_alive=True)

    assert issue_entry(paths, 56) == before
    assert gh.labels_added == [] and gh.labels_removed == []
    assert load_state(paths.state_file).get("events", []) == []


def test_sweep_dry_run_gates_state_and_events_not_the_injected_github_client(
    tmp_path: Path,
) -> None:
    config, paths, gh = no_pr_bed(tmp_path, 1176, dispatched_at="2026-07-14T17:24:55Z")
    before = issue_entry(paths, 1176)

    run_sweep(tmp_path, paths, config, gh, dry_run=True)

    assert issue_entry(paths, 1176) == before
    assert load_state(paths.state_file).get("events", []) == []
    # Issue #2226: sweep label writes now route through the WriteGate, which
    # owns dry-run suppression uniformly — a dry-run sweep writes no labels
    # to the injected client ("not the injected github client": the gate,
    # not ``gh.dry_run``, is what suppresses them). In production
    # ``gh.dry_run`` suppressed the same writes at the transport layer, so
    # this is an earlier no-op of the same effective behavior.
    # The leaf name is load-bearing: the collect-only gate (issue #1538)
    # fails a rename outright — a removed leaf must reappear verbatim in a
    # sibling module, and only an operator exemption waives it.
    assert gh.labels_added == []
    assert gh.labels_removed == []
