"""Decision table for ``decide_merge`` (merge-path stage 4), via its public interface."""

from __future__ import annotations

import itertools
from typing import Any

import _merge_path_facts as mf
import pytest
from charlie_work.merge_path import FactNotGathered, Hold, PlanKind, decide_merge

BLOCKED_GATE = mf.gate(summary_ready=False)
FAILED = mf.summary(passed=(), failed=("Lint & Format",))
ALL_STATUSES = (
    None,
    "pr_open",
    "escalated",
    "blocked",
    "dispatched",
    "dispatch_pending",
    "manifest_written",
    "rework_requested",
)
CHECK_ROUTE_EXCLUDED = (
    "escalated",
    "blocked",
    "dispatched",
    "dispatch_pending",
    "manifest_written",
    "rework_requested",
)
DEBOUNCED = mf.persisted(failed_attempts=2)  # prior + 1 >= alarm (3)


def _conflict(**over: Any) -> Any:
    """Readiness for an approved PR with a merge conflict and a blocked gate."""
    base: dict[str, Any] = {
        "gate": BLOCKED_GATE,
        "branch": mf.branch_gate(merge_conflict=True),
        "summary": mf.summary(passed=(), missing=mf.REQUIRED),
    }
    return mf.readiness(**{**base, **over})


def _checks_failed(**over: Any) -> Any:
    base: dict[str, Any] = {"gate": BLOCKED_GATE, "summary": FAILED}
    return mf.readiness(**{**base, **over})


# --------------------------------------------------------------------------- #
# Action matrix
# --------------------------------------------------------------------------- #


def test_self_merge_when_no_mergequeue_label_is_configured() -> None:
    plan = decide_merge(mf.readiness(), mf.hold_facts())
    assert plan.kind == PlanKind.MERGE
    assert (plan.action_merge, plan.action_hand_off, plan.action_human_merge) == (
        True,
        False,
        False,
    )
    assert plan.read_merge_hold is False
    assert plan.holds == frozenset()


def test_no_action_when_the_pass_is_not_a_merge_pass() -> None:
    plan = decide_merge(mf.readiness(), mf.hold_facts(should_merge=False))
    assert plan.kind == PlanKind.NONE
    assert (plan.action_merge, plan.action_hand_off) == (False, False)


@pytest.mark.parametrize(
    ("hold", "unavailable", "kind", "holds"),
    [
        pytest.param(False, False, PlanKind.HAND_OFF, frozenset(), id="free-to-hand-off"),
        pytest.param(True, False, PlanKind.HOLD, frozenset({Hold.MERGE_HOLD}), id="merge-hold"),
        pytest.param(
            False,
            True,
            PlanKind.HOLD,
            frozenset({Hold.MERGE_HOLD_UNAVAILABLE}),
            id="merge-hold-read-failed",
        ),
    ],
)
def test_mergequeue_hand_off_reads_the_merge_hold(
    hold: bool, unavailable: bool, kind: PlanKind, holds: frozenset[Hold]
) -> None:
    facts = mf.hold_facts(
        config=mf.cfg(mergequeue_label=mf.LABEL),
        merge_hold=None if unavailable else hold,
        merge_hold_unavailable=unavailable,
    )
    plan = decide_merge(mf.readiness(), facts)
    assert plan.kind == kind
    assert plan.read_merge_hold is True
    assert plan.action_hand_off is (kind == PlanKind.HAND_OFF)
    assert plan.action_merge is False
    assert plan.holds == holds


def test_hand_off_without_the_merge_hold_fact_fails_loudly() -> None:
    facts = mf.hold_facts(config=mf.cfg(mergequeue_label=mf.LABEL), merge_hold=None)
    with pytest.raises(FactNotGathered):
        decide_merge(mf.readiness(), facts)


def test_merge_hold_is_not_needed_when_no_hand_off_can_happen() -> None:
    facts = mf.hold_facts(config=mf.cfg(mergequeue_label=mf.LABEL), should_merge=False)
    plan = decide_merge(mf.readiness(), facts)
    assert (plan.read_merge_hold, plan.kind) == (False, PlanKind.NONE)


@pytest.mark.parametrize(
    ("reverted", "reset"),
    [
        pytest.param(False, True, id="fresh-hand-off-resets-counter"),
        pytest.param(True, False, id="reapply-while-reverted-keeps-counter"),
    ],
)
def test_counter_reset_on_hand_off_respects_a_reverted_label(reverted: bool, reset: bool) -> None:
    readiness = mf.readiness(
        branch=mf.branch_gate(admission=mf.admission(mergequeue_label_reverted=reverted))
    )
    facts = mf.hold_facts(config=mf.cfg(mergequeue_label=mf.LABEL), merge_hold=False)
    assert decide_merge(readiness, facts).reset_failed_attempts_on_hand_off is reset


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"gate": mf.gate(summary_ready=False)}, id="checks-not-green"),
        pytest.param({"gate": mf.gate(sync_failed=True)}, id="sync-failed"),
        pytest.param({"gate": mf.gate(approved=False)}, id="unapproved"),
    ],
)
def test_no_merge_action_without_can_merge(over: dict[str, Any]) -> None:
    plan = decide_merge(mf.readiness(**over), mf.hold_facts())
    assert (plan.action_merge, plan.action_hand_off, plan.action_human_merge) == (False,) * 3


def test_unapproved_pr_may_merge_when_approval_is_not_required() -> None:
    gate = mf.gate(approved=False, require_approved_review=False)
    plan = decide_merge(mf.readiness(gate=gate), mf.hold_facts())
    assert plan.kind == PlanKind.MERGE


# --------------------------------------------------------------------------- #
# Holds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"pr_escalated": True}, id="pr-escalated"),
        pytest.param({"issue_escalated": True}, id="issue-escalated"),
    ],
)
def test_escalated_hold_blocks_actions_but_leaves_can_merge_intact(over: dict[str, Any]) -> None:
    readiness = mf.readiness()
    plan = decide_merge(readiness, mf.hold_facts(**over))
    assert plan.kind == PlanKind.HOLD
    assert plan.escalated_merge_hold is True
    assert plan.holds == frozenset({Hold.ESCALATED})
    assert (plan.action_merge, plan.action_hand_off, plan.action_human_merge) == (False,) * 3
    # #840 / #777b: the hold must not rewrite the gate the merge_ready event reports.
    assert plan.readiness.gate.can_merge is True
    assert plan.readiness.gate == readiness.gate


def test_escalation_is_not_a_merge_hold_when_the_pr_could_not_merge_anyway() -> None:
    plan = decide_merge(mf.readiness(gate=BLOCKED_GATE), mf.hold_facts(pr_escalated=True))
    assert plan.escalated_merge_hold is False
    assert plan.kind == PlanKind.NONE


def test_human_merge_label_escalates_instead_of_merging() -> None:
    plan = decide_merge(mf.readiness(human_merge_hold=True), mf.hold_facts())
    assert plan.kind == PlanKind.ESCALATE
    assert plan.action_human_merge is True
    assert (plan.action_merge, plan.action_hand_off) == (False, False)
    assert Hold.HUMAN_MERGE in plan.holds


def test_human_merge_hand_off_is_suppressed_while_escalated() -> None:
    plan = decide_merge(mf.readiness(human_merge_hold=True), mf.hold_facts(issue_escalated=True))
    assert plan.action_human_merge is False
    assert plan.kind == PlanKind.HOLD
    assert plan.holds == frozenset({Hold.ESCALATED, Hold.HUMAN_MERGE})


def test_human_merge_label_outside_a_merge_pass_only_holds() -> None:
    plan = decide_merge(mf.readiness(human_merge_hold=True), mf.hold_facts(should_merge=False))
    assert (plan.kind, plan.action_human_merge) == (PlanKind.HOLD, False)


def test_human_merge_label_does_not_escalate_a_pr_that_cannot_merge() -> None:
    plan = decide_merge(mf.readiness(human_merge_hold=True, gate=BLOCKED_GATE), mf.hold_facts())
    assert (plan.kind, plan.action_human_merge) == (PlanKind.HOLD, False)


def test_unreadable_human_merge_labels_block_every_action() -> None:
    plan = decide_merge(
        mf.readiness(human_merge_check_unavailable=True),
        mf.hold_facts(config=mf.cfg(mergequeue_label=mf.LABEL)),
    )
    assert plan.kind == PlanKind.HOLD
    assert plan.holds == frozenset({Hold.HUMAN_MERGE_UNAVAILABLE})
    assert plan.read_merge_hold is False


def test_undetermined_revert_and_unavailable_checks_surface_as_holds() -> None:
    plan = decide_merge(
        mf.readiness(
            gate=mf.gate(sync_failed=True),
            cross_pr_revert_undetermined=True,
            checks_unavailable=True,
        ),
        mf.hold_facts(),
    )
    assert plan.kind == PlanKind.HOLD
    assert plan.holds == frozenset({Hold.REVERT_UNDETERMINED, Hold.CHECKS_UNAVAILABLE})


@pytest.mark.parametrize(
    "kind", [PlanKind.REQUEST_REWORK, PlanKind.RERUN_OR_ESCALATE], ids=["stall", "infra"]
)
def test_a_stage_three_terminal_kind_is_carried_through(kind: PlanKind) -> None:
    plan = decide_merge(mf.readiness(kind=kind, gate=BLOCKED_GATE), mf.hold_facts())
    assert plan.kind == kind


# --------------------------------------------------------------------------- #
# Conflict-rework route (#776) and its debounce
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("prior", "alarm", "routed"),
    [
        pytest.param(0, 3, False, id="first-pass"),
        pytest.param(1, 3, False, id="second-pass"),
        pytest.param(2, 3, True, id="prior-plus-one-reaches-alarm"),
        pytest.param(5, 3, True, id="beyond-alarm"),
        pytest.param(0, 1, True, id="alarm-one-routes-immediately"),
        pytest.param(9, 0, False, id="alarm-zero-never-routes"),
    ],
)
def test_conflict_route_is_debounced_by_the_failed_attempt_alarm(
    prior: int, alarm: int, routed: bool
) -> None:
    facts = mf.hold_facts(
        config=mf.cfg(failed_attempt_alarm=alarm), persisted=mf.persisted(failed_attempts=prior)
    )
    plan = decide_merge(_conflict(), facts)
    assert plan.conflict_rework is routed
    assert plan.kind == (PlanKind.REQUEST_REWORK if routed else PlanKind.NONE)


@pytest.mark.parametrize("status", ALL_STATUSES, ids=str)
def test_conflict_route_status_exclusions(status: str | None) -> None:
    plan = decide_merge(_conflict(), mf.hold_facts(persisted=DEBOUNCED, issue_status=status))
    assert plan.conflict_rework is (status not in ("manifest_written", "blocked"))


def test_conflict_route_still_fires_for_an_escalated_issue() -> None:
    facts = mf.hold_facts(persisted=DEBOUNCED, issue_status="escalated", issue_escalated=True)
    assert decide_merge(_conflict(), facts).conflict_rework is True


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"issue_number": None}, id="no-issue"),
        pytest.param({"gate": mf.gate(approved=False, summary_ready=False)}, id="unapproved"),
        pytest.param({"pending_only": True}, id="pending-only"),
        pytest.param({"branch": mf.branch_gate(merge_conflict=False)}, id="no-conflict"),
    ],
)
def test_conflict_route_preconditions(over: dict[str, Any]) -> None:
    plan = decide_merge(_conflict(**over), mf.hold_facts(persisted=DEBOUNCED))
    assert plan.conflict_rework is False


def test_conflict_route_with_no_persisted_entry_counts_from_zero() -> None:
    plan = decide_merge(_conflict(), mf.hold_facts(persisted=None))
    assert plan.conflict_rework is False


# --------------------------------------------------------------------------- #
# Check-failure route
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", ALL_STATUSES, ids=str)
def test_check_failure_route_status_exclusions(status: str | None) -> None:
    plan = decide_merge(_checks_failed(), mf.hold_facts(persisted=DEBOUNCED, issue_status=status))
    assert plan.check_failure_rework is (status not in CHECK_ROUTE_EXCLUDED)


@pytest.mark.parametrize(
    ("prior", "alarm", "routed"),
    [(0, 3, False), (2, 3, True), (9, 0, False)],
    ids=["first-pass", "debounced", "alarm-zero"],
)
def test_check_failure_route_debounce(prior: int, alarm: int, routed: bool) -> None:
    facts = mf.hold_facts(
        config=mf.cfg(failed_attempt_alarm=alarm), persisted=mf.persisted(failed_attempts=prior)
    )
    plan = decide_merge(_checks_failed(), facts)
    assert (plan.check_failure_rework, plan.kind == PlanKind.REQUEST_REWORK) == (routed, routed)


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"issue_number": None}, id="no-issue"),
        pytest.param({"summary": mf.summary(passed=(), pending=mf.REQUIRED)}, id="nothing-failed"),
        pytest.param({"branch": mf.branch_gate(merge_conflict=True)}, id="conflict-owns-it"),
        pytest.param({"cross_pr_revert_detected": True}, id="revert-owns-it"),
        pytest.param({"gate": mf.gate(approved=False, summary_ready=False)}, id="unapproved"),
    ],
)
def test_check_failure_route_preconditions(over: dict[str, Any]) -> None:
    plan = decide_merge(_checks_failed(**over), mf.hold_facts(persisted=DEBOUNCED))
    assert plan.check_failure_rework is False


# --------------------------------------------------------------------------- #
# Exclusivity invariant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("summary_ready", "approved", "sync_failed", "conflict", "failed", "human", "should_merge"),
    list(itertools.product([True, False], repeat=7)),
)
def test_rework_routes_exclude_merge_and_hand_off_actions(
    summary_ready: bool,
    approved: bool,
    sync_failed: bool,
    conflict: bool,
    failed: bool,
    human: bool,
    should_merge: bool,
) -> None:
    readiness = mf.readiness(
        gate=mf.gate(summary_ready=summary_ready, approved=approved, sync_failed=sync_failed),
        branch=mf.branch_gate(merge_conflict=conflict),
        summary=FAILED if failed else mf.summary(),
        human_merge_hold=human,
    )
    for label in (None, mf.LABEL):
        facts = mf.hold_facts(
            config=mf.cfg(mergequeue_label=label),
            persisted=DEBOUNCED,
            should_merge=should_merge,
            merge_hold=False,
        )
        plan = decide_merge(readiness, facts)  # must never trip the internal assertion
        routed = plan.conflict_rework or plan.check_failure_rework
        acted = plan.action_merge or plan.action_hand_off or plan.action_human_merge
        assert not (routed and acted)
        assert not (plan.action_merge and plan.action_hand_off)
