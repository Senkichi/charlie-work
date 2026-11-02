"""The merge lane's conflict-rework-cap stop (issue #2900).

``merge_ready`` must skip a PR the conflict lane already escalated for
``conflict_rework_attempts_cap_exceeded``, for as long as the PR's head is
the one the escalation stamped. Characterization harness shared with
``test_merge_path_characterization_gates``; the stop's decision tables live
in ``test_merge_path_decide_branch``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from _merge_path_characterization_harness import (
    _ISSUE,
    _PR,
    _assert_dry_run_inert,
    _cfg,
    _conflicted_gh,
    _pair,
    _seed_issue_entry,
    _seed_pr_entry,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

# ===========================================================================
# Conflict-rework cap stop (issue #2900)
# ===========================================================================


_LEVEL = "conflict_rework_attempts_cap_exceeded"
_CAP_STOP_MSG = (
    "PR #456 conflict-rework cap exceeded; escalated — merge lane skipping "
    "until the head moves or the operator unescalates"
)
_CAP_STOP_PREVIEW_MSG = (
    "PR #456 conflict-rework cap exceeded; escalated — merge lane would "
    "skip until the head moves or the operator unescalates"
)
_CAP_STOP_KEYS = {
    "pr",
    "issue",
    "can_merge",
    "merged",
    "review_decision",
    "merge_conflict",
    "consecutive_failed_merge_attempts",
    "merge_attempt_alarm",
    "merge_attempt_warning",
    "conflict_cap_stop",
}


def _seed_conflict_cap_escalation(paths: Any, escalations: dict[str, Any]) -> None:
    """The pair after ``_route_janitor_gate_failure_to_rework``'s cap branch:
    escalated on both records, this lane's reason in both escalation
    histories, the head stamped as the skip anchor."""
    _seed_pr_entry(
        paths,
        status="escalated",
        escalation_reason=_LEVEL,
        **escalations,
    )
    _seed_issue_entry(paths, status="escalated", escalation_reason=_LEVEL)


def test_conflict_cap_escalated_pr_on_unchanged_head_skips_the_merge_lane(
    tmp_path: Path,
) -> None:
    """Issue #2900: past the conflict lane's rework cap, the merge lane must
    stop re-gating the PR on the head it escalated at -- the shape the
    branch gate yields instead carries no ``merge_ready`` row (the bug was
    3,340 of them from one PR), no re-review, and no attempt movement."""

    def pre(_app: OrchestratorApp, paths: Any, _gh: FakeGitHub) -> None:
        _seed_conflict_cap_escalation(
            paths,
            {
                "escalation_reasons_seen": [_LEVEL],
                "conflict_rework_attempts": 3,
                "conflict_rework_attempts_escalated_head": "sha-abc123",
            },
        )

    pair = _pair(tmp_path, _cfg(alarm=1), _conflicted_gh, pre=pre, merge=False)

    live, dry = pair.live, pair.dry
    assert live.result.ok is True
    assert live.result.message == _CAP_STOP_MSG
    assert set(live.data) == _CAP_STOP_KEYS
    assert live.data["can_merge"] is False
    assert live.data["merged"] is False
    assert live.data["merge_conflict"] is True
    assert live.data["conflict_cap_stop"] is True
    assert live.data["merge_attempt_alarm"] is False
    # Zero events: no per-pass merge_ready row, no rework routing, no alarm.
    assert live.new_event_kinds() == []
    assert live.state_after["issues"]["123"]["status"] == "escalated"
    assert live.state_after["prs"]["456"]["conflict_rework_attempts"] == 3
    # No base-currency / sync / merge actions were attempted.
    assert live.gh.merged == []
    assert live.gh.pr_update_branch_calls == []

    # Dry-run parity (same facts, no write): the preview reports the same
    # stop instead of projecting a remediation that will never happen.
    assert dry.result.ok is True
    assert dry.result.message == _CAP_STOP_PREVIEW_MSG
    assert set(dry.data) == _CAP_STOP_KEYS | {"dry_run"}
    _assert_dry_run_inert(dry)


def test_conflict_cap_escalated_pr_stays_quiescent_until_head_moves(
    tmp_path: Path,
) -> None:
    """Every repeated pass on the unchanged head emits nothing (AC 1); the
    head moving -- with a patch that no longer carries the verdict -- puts
    the PR back through the merge lane as a head-moved re-review (AC 2). The
    stale head was the observed 3,340-row shape."""
    config = _cfg(alarm=1)
    root = tmp_path / "live"
    paths = runtime_paths(root, config.runtime.state_dir)
    gh = _conflicted_gh()
    # The verdict's patch-id belongs to the old head; the new head carries a
    # different cumulative diff so the carry-forward cannot re-approve it.
    gh.diffs[_PR] = "diff --git a/old b/old\n--- a/old\n+++ b/old\n@@\n"
    app = OrchestratorApp(root, paths, config, gh)
    app.record_review(_PR, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"][str(_PR)] = {
            **state["prs"].get(str(_PR), {}),
            "status": "escalated",
            "escalation_reason": _LEVEL,
            "escalation_reasons_seen": [_LEVEL],
            "conflict_rework_attempts": 3,
            "conflict_rework_attempts_escalated_head": "sha-abc123",
        }
        state["issues"][str(_ISSUE)] = {
            **state["issues"].get(str(_ISSUE), {}),
            "status": "escalated",
            "escalation_reason": _LEVEL,
            "escalation_reasons_seen": [_LEVEL],
        }
        save_state(paths.state_file, state)

    for recorder in (gh.labels_added, gh.labels_removed, gh.pr_update_branch_calls):
        recorder.clear()

    baseline = len(load_state(paths.state_file).get("events", []))
    for _ in range(3):
        result = app.merge_ready(_PR, merge=False)
        assert result.data["conflict_cap_stop"] is True
        state = load_state(paths.state_file)
        assert len(state.get("events", [])) == baseline
        assert state["prs"]["456"]["conflict_rework_attempts"] == 3

    # A new head lifts the stop and the lane runs its head-moved re-review.
    gh.prs[0]["headRefOid"] = "sha-pushed"
    gh.diffs[_PR] = "diff --git a/new b/new\n--- a/new\n+++ b/new\n@@\n"
    moved = app.merge_ready(_PR, merge=False)
    assert moved.data.get("conflict_cap_stop") is None
    assert moved.data["head_moved"] is True
    state = load_state(paths.state_file)
    assert len(state.get("events", [])) == baseline  # still no merge_ready row
    assert state["issues"]["123"]["status"] == "escalated"  # #833: no review stamping

    # ...and the run was not inert on the head change: no merge/sync actions.
    assert gh.merged == []
    assert gh.pr_update_branch_calls == []


# ===========================================================================
