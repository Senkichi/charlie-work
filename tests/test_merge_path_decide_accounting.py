"""Decision table for ``decide_accounting`` (merge-path stage 5), via its public interface."""

from __future__ import annotations

from typing import Any

import _merge_path_facts as mf
import pytest
from charlie_work.merge_path import EffectResults, MergePlan, PlanKind, decide_accounting

BLOCKED_GATE = mf.gate(summary_ready=False)
PENDING = mf.summary(passed=(), pending=mf.REQUIRED)
FAILED = mf.summary(passed=(), failed=("Lint & Format",))
MISSING = mf.summary(passed=(), missing=("Tests passed",))


def _plan(kind: PlanKind = PlanKind.NONE, **readiness_over: Any) -> MergePlan:
    return MergePlan(kind=kind, readiness=mf.readiness(**readiness_over))


def _blocked(summary: Any = MISSING, **over: Any) -> MergePlan:
    return _plan(gate=BLOCKED_GATE, summary=summary, **over)


def _facts(prior: int = 0, **over: Any) -> Any:
    locked = over.pop("locked", mf.persisted(failed_attempts=prior))
    return mf.accounting_facts(locked=locked, **over)


def _kinds(acc: Any) -> list[str]:
    return [e.kind for e in acc.events]


# --------------------------------------------------------------------------- #
# Failed-attempt counter and alarm
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("prior", "alarm_at", "attempts", "alarm"),
    [
        pytest.param(0, 3, 1, False, id="first-failure"),
        pytest.param(1, 3, 2, False, id="second-failure"),
        pytest.param(2, 3, 3, True, id="alarm-fires-exactly-at-threshold"),
        pytest.param(3, 3, 4, False, id="past-threshold-does-not-refire"),
        pytest.param(40, 3, 4, False, id="counter-clamps-at-threshold-plus-one"),
        pytest.param(0, 1, 1, True, id="threshold-one"),
        pytest.param(5, 0, 6, False, id="alarm-zero-never-fires-and-never-clamps"),
    ],
)
def test_unmergeable_approved_pr_increments_and_clamps(
    prior: int, alarm_at: int, attempts: int, alarm: bool
) -> None:
    acc = decide_accounting(
        _blocked(), EffectResults(), _facts(prior, config=mf.cfg(failed_attempt_alarm=alarm_at))
    )
    assert (acc.failed_attempts, acc.alarm) == (attempts, alarm)
    assert ("merge_failed_attempt_alarm" in _kinds(acc)) is alarm
    assert (acc.warning is not None) is alarm


def test_spent_deadline_leaves_the_counter_untouched() -> None:
    acc = decide_accounting(_blocked(), EffectResults(), _facts(2, deadline_spent=True))
    assert (acc.failed_attempts, acc.alarm, acc.warning) == (2, False, None)
    assert "merge_failed_attempt_alarm" not in _kinds(acc)


def test_spent_deadline_does_not_reset_a_mergeable_pr_either() -> None:
    acc = decide_accounting(_plan(), EffectResults(), _facts(2, deadline_spent=True))
    assert acc.failed_attempts == 2


def test_pending_only_is_not_a_failed_attempt() -> None:
    acc = decide_accounting(_blocked(PENDING, pending_only=True), EffectResults(), _facts(2))
    assert (acc.failed_attempts, acc.alarm) == (2, False)


def test_unapproved_pr_is_not_a_failed_attempt() -> None:
    plan = _plan(gate=mf.gate(summary_ready=False, approved=False), summary=MISSING)
    assert decide_accounting(plan, EffectResults(), _facts(2)).failed_attempts == 2


def test_mergeable_pr_resets_the_counter() -> None:
    acc = decide_accounting(_plan(PlanKind.MERGE), EffectResults(merge_output="ok"), _facts(2))
    assert acc.failed_attempts == 0


def test_stale_base_deferrals_always_reset() -> None:
    facts = _facts(locked=mf.persisted(stale_base_deferrals=4))
    assert decide_accounting(_plan(), EffectResults(), facts).stale_base_deferrals == 0


# --------------------------------------------------------------------------- #
# Hand-off failure
# --------------------------------------------------------------------------- #

HANDOFF_CFG = mf.cfg(mergequeue_label=mf.LABEL)


@pytest.mark.parametrize(
    ("label_applied", "reverted", "self_revoked", "can_merge", "failed"),
    [
        pytest.param(False, False, False, True, True, id="apply-failed"),
        pytest.param(True, False, False, True, False, id="applied"),
        pytest.param(None, False, False, True, False, id="not-attempted"),
        pytest.param(None, True, False, True, True, id="reverted-counts-as-failed"),
        pytest.param(None, True, True, True, False, id="self-revoked-stale-head-is-not-a-failure"),
        pytest.param(None, True, False, False, False, id="reverted-but-cannot-merge"),
    ],
)
def test_handoff_failed(
    label_applied: bool | None, reverted: bool, self_revoked: bool, can_merge: bool, failed: bool
) -> None:
    adm = mf.admission(mergequeue_label_reverted=reverted, self_revoked_stale_head=self_revoked)
    gate = mf.gate() if can_merge else BLOCKED_GATE
    plan = _plan(gate=gate, branch=mf.branch_gate(admission=adm))
    acc = decide_accounting(
        plan,
        EffectResults(mergequeue_label_applied=label_applied),
        _facts(config=HANDOFF_CFG),
    )
    assert acc.handoff_failed is failed
    assert acc.failed_attempts == (1 if (failed or not can_merge) else 0)


def test_failed_apply_without_a_configured_label_is_not_a_handoff_failure() -> None:
    acc = decide_accounting(
        _plan(), EffectResults(mergequeue_label_applied=False), _facts(config=mf.cfg())
    )
    assert acc.handoff_failed is False


def test_reverted_label_keeps_the_counter_climbing_across_passes() -> None:
    adm = mf.admission(mergequeue_label_reverted=True)
    plan = _plan(branch=mf.branch_gate(admission=adm))
    facts = _facts(1, config=HANDOFF_CFG)
    assert (
        decide_accounting(
            plan, EffectResults(mergequeue_label_applied=True), facts
        ).failed_attempts
        == 2
    )


# --------------------------------------------------------------------------- #
# Alarm warning text priority
# --------------------------------------------------------------------------- #


def _alarm_warning(plan: MergePlan, results: EffectResults | None = None, **over: Any) -> str:
    facts = _facts(2, **over)
    acc = decide_accounting(plan, results or EffectResults(), facts)
    assert acc.alarm and acc.warning
    return acc.warning


def test_warning_conflict_outranks_everything() -> None:
    plan = _blocked(
        FAILED,
        branch=mf.branch_gate(merge_conflict=True),
        cross_pr_revert_detected=True,
    )
    text = _alarm_warning(plan, EffectResults(conflict_routed=True))
    assert "merge conflict" in text
    assert text.endswith("rework dispatched")


@pytest.mark.parametrize(
    ("results", "detail"),
    [
        pytest.param(
            EffectResults(conflict_escalated=True), "escalated to a human", id="escalated"
        ),
        pytest.param(EffectResults(), "rework not routed", id="not-routed"),
        pytest.param(
            EffectResults(issue_status_after="rework_requested"),
            "rework already requested",
            id="already-requested",
        ),
        pytest.param(
            EffectResults(conflict_routed=True, rework_label_error={"outcome": "boom"}),
            "label update failed: boom",
            id="label-error",
        ),
    ],
)
def test_warning_conflict_detail(results: EffectResults, detail: str) -> None:
    text = _alarm_warning(_blocked(branch=mf.branch_gate(merge_conflict=True)), results)
    assert detail in text


def test_warning_conflict_without_an_issue() -> None:
    plan = _blocked(branch=mf.branch_gate(merge_conflict=True), issue_number=None)
    text = _alarm_warning(plan, issue_number=None)
    assert "no linked issue, cannot route to rework" in text


def test_warning_cross_pr_revert() -> None:
    plan = _blocked(cross_pr_revert_detected=True)
    text = _alarm_warning(plan, EffectResults(cross_pr_revert_routed=True))
    assert "cross-PR revert" in text


def test_warning_handoff_failure() -> None:
    plan = _plan()
    text = _alarm_warning(plan, EffectResults(mergequeue_label_applied=False), config=HANDOFF_CFG)
    assert "failed to apply" in text
    assert "never handed off" in text
    assert repr(mf.LABEL) in text


def test_warning_failed_checks() -> None:
    text = _alarm_warning(_blocked(FAILED), EffectResults(check_failure_routed=True))
    assert "required check(s) failed (Lint & Format)" in text
    assert text.endswith("rework dispatched")


def test_warning_falls_back_to_the_generic_check_summary() -> None:
    text = _alarm_warning(_blocked(MISSING), mergeable="MERGEABLE", merge_state_status="CLEAN")
    assert text.startswith("PR #7 approved but unmergeable for 3 passes: ")
    assert "required checks missing" in text


def test_alarm_event_payload() -> None:
    acc = decide_accounting(_blocked(), EffectResults(), _facts(2, mergeable="CONFLICTING"))
    ev = next(e for e in acc.events if e.kind == "merge_failed_attempt_alarm")
    assert ev.payload["pr_number"] == mf.PR
    assert ev.payload["issue_number"] == mf.ISSUE
    assert (ev.payload["attempts"], ev.payload["threshold"]) == (3, 3)
    assert ev.payload["mergeable"] == "CONFLICTING"
    assert ev.payload["message"] == acc.warning


# --------------------------------------------------------------------------- #
# Merge outcome, status and events
# --------------------------------------------------------------------------- #


def test_self_merge_marks_merged_and_emits_merge_succeeded() -> None:
    acc = decide_accounting(
        _plan(PlanKind.MERGE),
        EffectResults(merge_output="merged", merged_at="2026-01-01T12:00:01+00:00"),
        _facts(locked=mf.persisted(status="approved")),
    )
    assert (acc.merged, acc.pr_status) == (True, "merged")
    assert _kinds(acc) == ["merge_ready", "merge_succeeded"]
    ev = acc.events[-1]
    assert ev.payload["actor"] == "fleet"
    assert ev.payload["merge_method"] == "squash"
    assert ev.payload["merged_at"] == "2026-01-01T12:00:01+00:00"
    assert (ev.payload["pr_number"], ev.payload["issue_number"]) == (mf.PR, mf.ISSUE)


def test_hand_off_never_writes_merged_or_emits_merge_succeeded() -> None:
    # ADR-0003: the merge queue, not us, merges.
    acc = decide_accounting(
        _plan(PlanKind.HAND_OFF),
        EffectResults(mergequeue_label_applied=True),
        _facts(config=HANDOFF_CFG, locked=mf.persisted(status="mergequeue")),
    )
    assert (acc.merged, acc.pr_status) == (False, None)
    assert "merge_succeeded" not in _kinds(acc)
    assert _kinds(acc) == ["merge_ready"]


def test_merge_ready_payload_carries_the_gate_keys_and_flags() -> None:
    plan = MergePlan(
        kind=PlanKind.HOLD,
        readiness=mf.readiness(human_merge_hold=True, human_merge_check_unavailable=True),
        merge_hold=True,
        merge_hold_unavailable=True,
    )
    acc = decide_accounting(
        plan,
        EffectResults(mergequeue_label_applied=True, cancel_results={"cancelled": 1}),
        _facts(),
    )
    payload = acc.events[-1].payload
    assert acc.events[-1].kind == "merge_ready"
    for key in ("summary_ready", "approved", "require_approved_review", "sync_failed"):
        assert key in payload
    assert payload["can_merge"] is True
    assert payload["merged"] is False
    assert payload["merge_hold"] is True
    assert payload["merge_hold_check_unavailable"] is True
    assert payload["human_merge_hold"] is True
    assert payload["human_merge_check_unavailable"] is True
    assert payload["mergequeue_label_applied"] is True
    assert payload["cancel_superseded_runs_results"] == {"cancelled": 1}


def test_escalated_hold_reports_the_true_gate_in_the_event() -> None:
    plan = MergePlan(kind=PlanKind.HOLD, readiness=mf.readiness(), escalated_merge_hold=True)
    payload = decide_accounting(plan, EffectResults(), _facts()).events[-1].payload
    assert payload["can_merge"] is True


@pytest.mark.parametrize(
    ("over", "plan_over", "ok"),
    [
        pytest.param({}, {}, True, id="mergeable-and-not-merged"),
        pytest.param({"merge_output": "merged"}, {}, False, id="merged"),
        pytest.param({}, {"gate": BLOCKED_GATE}, False, id="cannot-merge"),
        pytest.param({}, {"gate": mf.gate(approved=False)}, False, id="unapproved"),
        pytest.param({}, {"human_merge_hold": True}, False, id="human-hold"),
        pytest.param({}, {"human_merge_check_unavailable": True}, False, id="human-unavailable"),
        pytest.param({"mergequeue_label_applied": False}, {}, False, id="handoff-failed"),
    ],
)
def test_merge_alert_ok(over: dict[str, Any], plan_over: dict[str, Any], ok: bool) -> None:
    acc = decide_accounting(_plan(**plan_over), EffectResults(**over), _facts(config=HANDOFF_CFG))
    assert acc.merge_alert_ok is ok


def test_merge_alert_is_not_ok_when_the_merge_hold_read_failed() -> None:
    plan = MergePlan(kind=PlanKind.HOLD, readiness=mf.readiness(), merge_hold_unavailable=True)
    assert decide_accounting(plan, EffectResults(), _facts()).merge_alert_ok is False


# --------------------------------------------------------------------------- #
# mergequeue_since / mergequeue_head_sha stamps
# --------------------------------------------------------------------------- #

SINCE = "2026-01-01T10:00:00+00:00"


@pytest.mark.parametrize(
    ("locked", "live_head", "since", "head"),
    [
        pytest.param(
            mf.persisted(status="mergequeue", mergequeue_since=SINCE, mergequeue_head_sha=mf.HEAD),
            mf.HEAD,
            SINCE,
            mf.HEAD,
            id="same-head-keeps-stamps",
        ),
        pytest.param(
            mf.persisted(status="mergequeue", mergequeue_since=SINCE, mergequeue_head_sha=mf.HEAD),
            mf.NEW_HEAD,
            mf.NOW_ISO,
            mf.NEW_HEAD,
            id="head-moved-restamps",
        ),
        pytest.param(
            mf.persisted(status="mergequeue"),
            mf.HEAD,
            mf.NOW_ISO,
            mf.HEAD,
            id="first-hand-off-stamps",
        ),
        pytest.param(
            mf.persisted(status="mergequeue", mergequeue_since=None, mergequeue_head_sha=mf.HEAD),
            mf.HEAD,
            mf.NOW_ISO,
            mf.HEAD,
            id="missing-since-restamps",
        ),
        pytest.param(
            mf.persisted(status="mergequeue", mergequeue_since=SINCE, mergequeue_head_sha=mf.HEAD),
            None,
            None,
            None,
            id="unknown-live-head-clears",
        ),
        pytest.param(
            mf.persisted(status="approved", mergequeue_since=SINCE, mergequeue_head_sha=mf.HEAD),
            mf.HEAD,
            None,
            None,
            id="status-left-mergequeue-clears",
        ),
    ],
)
def test_mergequeue_stamps(locked: Any, live_head: str | None, since: Any, head: Any) -> None:
    acc = decide_accounting(
        _plan(PlanKind.HAND_OFF),
        EffectResults(mergequeue_label_applied=True),
        _facts(locked=locked, live_head_sha=live_head),
    )
    assert (acc.mergequeue_since, acc.mergequeue_head_sha) == (since, head)


def test_a_merged_pr_drops_the_mergequeue_stamps() -> None:
    locked = mf.persisted(status="mergequeue", mergequeue_since=SINCE, mergequeue_head_sha=mf.HEAD)
    acc = decide_accounting(
        _plan(PlanKind.MERGE), EffectResults(merge_output="merged"), _facts(locked=locked)
    )
    assert (acc.mergequeue_since, acc.mergequeue_head_sha) == (None, None)


# --------------------------------------------------------------------------- #
# mergequeue requeue counter and cap event (issue #2743)
# --------------------------------------------------------------------------- #


def _reverted_plan(**over: Any) -> MergePlan:
    admission = mf.admission(mergequeue_label_reverted=True)
    base: dict[str, Any] = {"branch": mf.branch_gate(admission=admission)}
    return _plan(**{**base, **over})


def _requeue_locked(requeues: int, head: str | None = mf.HEAD, **over: Any) -> Any:
    return mf.persisted(
        status="mergequeue",
        mergequeue_requeues=requeues,
        mergequeue_requeues_head_sha=head,
        **over,
    )


def test_a_counted_revert_increments_the_requeue_counter_at_the_live_head() -> None:
    acc = decide_accounting(
        _reverted_plan(),
        EffectResults(mergequeue_label_applied=True),
        _facts(config=HANDOFF_CFG, locked=_requeue_locked(1), live_head_sha=mf.HEAD),
    )
    assert acc.mergequeue_requeues == 2
    assert acc.mergequeue_requeues_head_sha == mf.HEAD


def test_a_revert_at_a_new_head_restarts_the_count() -> None:
    acc = decide_accounting(
        _reverted_plan(),
        EffectResults(mergequeue_label_applied=True),
        _facts(
            config=HANDOFF_CFG,
            locked=_requeue_locked(5, head=mf.NEW_HEAD),
            live_head_sha=mf.HEAD,
        ),
    )
    assert acc.mergequeue_requeues == 1
    assert acc.mergequeue_requeues_head_sha == mf.HEAD


@pytest.mark.parametrize(
    ("reverted", "self_revoked", "can_merge", "live_head"),
    [
        pytest.param(True, True, True, mf.HEAD, id="self-revocation-is-not-counted"),
        pytest.param(True, False, False, mf.HEAD, id="failed-pr-checks-own-the-revert"),
        pytest.param(True, False, True, None, id="unknown-head-cannot-anchor"),
        pytest.param(False, False, True, mf.HEAD, id="no-revert-nothing-to-count"),
    ],
)
def test_uncounted_reverts_leave_the_requeue_fields_alone(
    reverted: bool, self_revoked: bool, can_merge: bool, live_head: str | None
) -> None:
    admission = mf.admission(
        mergequeue_label_reverted=reverted, self_revoked_stale_head=self_revoked
    )
    gate = mf.gate() if can_merge else BLOCKED_GATE
    plan = _plan(branch=mf.branch_gate(admission=admission), gate=gate)
    acc = decide_accounting(
        plan,
        EffectResults(mergequeue_label_applied=True),
        _facts(config=HANDOFF_CFG, locked=_requeue_locked(1), live_head_sha=live_head),
    )
    assert acc.mergequeue_requeues is None
    assert acc.mergequeue_requeues_head_sha is None


def test_reaching_the_cap_emits_mergequeue_requeue_capped_once() -> None:
    acc = decide_accounting(
        _reverted_plan(),
        EffectResults(mergequeue_label_applied=None),
        _facts(config=HANDOFF_CFG, locked=_requeue_locked(2), live_head_sha=mf.HEAD),
    )
    events = [e for e in acc.events if e.kind == "mergequeue_requeue_capped"]
    assert len(events) == 1
    payload = events[0].payload
    assert payload["pr_number"] == mf.PR
    assert payload["issue_number"] == mf.ISSUE
    assert payload["head_sha"] == mf.HEAD
    assert payload["requeues"] == 3
    assert payload["cap"] == 3
    assert payload["mergequeue_label"] == mf.LABEL


def test_past_the_cap_does_not_reemit_the_event() -> None:
    acc = decide_accounting(
        _reverted_plan(),
        EffectResults(mergequeue_label_applied=None),
        _facts(config=HANDOFF_CFG, locked=_requeue_locked(3), live_head_sha=mf.HEAD),
    )
    assert acc.mergequeue_requeues == 4
    assert "mergequeue_requeue_capped" not in _kinds(acc)


def test_cap_zero_still_counts_but_never_emits() -> None:
    cfg = mf.cfg(mergequeue_label=mf.LABEL, mergequeue_requeue_cap=0)
    acc = decide_accounting(
        _reverted_plan(),
        EffectResults(mergequeue_label_applied=True),
        _facts(config=cfg, locked=_requeue_locked(9), live_head_sha=mf.HEAD),
    )
    assert acc.mergequeue_requeues == 10
    assert "mergequeue_requeue_capped" not in _kinds(acc)


def test_a_self_revoked_revert_neither_counts_nor_emits() -> None:
    admission = mf.admission(mergequeue_label_reverted=True, self_revoked_stale_head=True)
    plan = _plan(branch=mf.branch_gate(admission=admission))
    acc = decide_accounting(
        plan,
        EffectResults(mergequeue_label_applied=True),
        _facts(config=HANDOFF_CFG, locked=_requeue_locked(2), live_head_sha=mf.HEAD),
    )
    assert acc.mergequeue_requeues is None
    assert "mergequeue_requeue_capped" not in _kinds(acc)
