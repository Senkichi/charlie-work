"""Decision table for ``decide_branch`` (merge-path stage 2), via its public interface."""

from __future__ import annotations

from typing import Any

import _merge_path_facts as mf
import pytest
from charlie_work.merge_path import FactNotGathered, PlanKind, SyncOutcome, decide_branch
from charlie_work.merge_path.model import UNAVAILABLE, BranchStop, StageKind


def test_unapproved_short_circuits_with_no_reads() -> None:
    f = mf.branch_facts(
        admission=mf.admission(approved=False),
        merge_conflict=True,
        train_head_param=999,
        base_currency_gated=None,
    )
    gate = decide_branch(f)
    assert gate.kind == StageKind.PROCEED
    assert (gate.merge_conflict, gate.sync_failed, gate.request_sync) == (False, False, False)


@pytest.mark.parametrize(
    ("issue_status", "kind", "stop", "sync_failed"),
    [
        pytest.param(
            "dispatched", PlanKind.WAIT, BranchStop.CONFLICT_IN_FLIGHT, False, id="dispatched"
        ),
        pytest.param(
            "dispatch_pending", PlanKind.WAIT, BranchStop.CONFLICT_IN_FLIGHT, False, id="pending"
        ),
        pytest.param(
            "manifest_written", PlanKind.WAIT, BranchStop.CONFLICT_IN_FLIGHT, False, id="manifest"
        ),
        pytest.param("blocked", PlanKind.HOLD, BranchStop.CONFLICT_BLOCKED, False, id="blocked"),
        pytest.param("escalated", StageKind.PROCEED, None, True, id="escalated-still-routes-776"),
        pytest.param("rework_requested", StageKind.PROCEED, None, True, id="rework-requested"),
        pytest.param(None, StageKind.PROCEED, None, True, id="no-issue-status"),
    ],
)
def test_merge_conflict_by_issue_status(
    issue_status: str | None, kind: object, stop: object, sync_failed: bool
) -> None:
    gate = decide_branch(mf.branch_facts(merge_conflict=True, issue_status=issue_status))
    assert (gate.kind, gate.stop, gate.sync_failed) == (kind, stop, sync_failed)
    assert gate.merge_conflict is True


def test_conflict_precedes_the_train_head_check() -> None:
    f = mf.branch_facts(merge_conflict=True, issue_status="dispatched", train_head_param=999)
    assert decide_branch(f).stop == BranchStop.CONFLICT_IN_FLIGHT


@pytest.mark.parametrize(
    ("over", "kind", "sync_failed"),
    [
        pytest.param({"train_head_param": mf.PR}, StageKind.PROCEED, False, id="param-is-this-pr"),
        pytest.param({"train_head_param": 999}, PlanKind.WAIT, False, id="param-other-pr"),
        pytest.param(
            {"train_head_param": None, "train_head": None}, StageKind.PROCEED, False, id="no-head"
        ),
        pytest.param(
            {"train_head_param": None, "train_head": mf.PR},
            StageKind.PROCEED,
            False,
            id="list-head-is-this-pr",
        ),
        pytest.param(
            {"train_head_param": None, "train_head": 999},
            PlanKind.WAIT,
            False,
            id="list-head-other-pr",
        ),
        pytest.param(
            {"train_head_param": None, "train_head": UNAVAILABLE},
            StageKind.PROCEED,
            True,
            id="pr-list-unavailable",
        ),
        pytest.param(
            {"train_head_param": 999, "config": mf.cfg(update_branch_strategy="broadcast")},
            StageKind.PROCEED,
            False,
            id="broadcast-ignores-train",
        ),
        pytest.param(
            {"train_head_param": 999, "config": mf.cfg(update_branch_strategy="off")},
            StageKind.PROCEED,
            False,
            id="off-ignores-train",
        ),
    ],
)
def test_train_head(over: dict[str, Any], kind: object, sync_failed: bool) -> None:
    gate = decide_branch(mf.branch_facts(**over))
    assert (gate.kind, gate.sync_failed) == (kind, sync_failed)
    if kind == PlanKind.WAIT:
        assert gate.stop == BranchStop.NOT_TRAIN_HEAD


def test_off_strategy_never_reads_the_base_gate() -> None:
    f = mf.branch_facts(config=mf.cfg(update_branch_strategy="off"), base_currency_gated=None)
    assert decide_branch(f).kind == StageKind.PROCEED


def test_unread_gated_fact_fails_loudly() -> None:
    with pytest.raises(FactNotGathered):
        decide_branch(mf.branch_facts(base_currency_gated=None))


def test_base_gate_off_for_this_base_ref_proceeds() -> None:
    assert decide_branch(mf.branch_facts(base_currency_gated=False)).kind == StageKind.PROCEED


@pytest.mark.parametrize(
    ("base_current", "reason"),
    [
        pytest.param(None, "compare_unavailable", id="compare-unavailable"),
        pytest.param(False, "base_stale", id="stale"),
    ],
)
def test_gated_base_not_current_defers(base_current: bool | None, reason: str) -> None:
    f = mf.branch_facts(
        base_currency_gated=True, base_current_read=True, base_current=base_current
    )
    gate = decide_branch(f, SyncOutcome.NOT_NEEDED)
    assert (gate.kind, gate.stop, gate.stale_base_reason) == (
        PlanKind.WAIT,
        BranchStop.STALE_BASE,
        reason,
    )


def test_gated_current_base_proceeds() -> None:
    f = mf.branch_facts(base_currency_gated=True, base_current_read=True, base_current=True)
    gate = decide_branch(f)
    assert (gate.kind, gate.request_sync, gate.sync_failed) == (StageKind.PROCEED, False, False)


def test_gated_without_a_read_base_fact_fails_loudly() -> None:
    f = mf.branch_facts(base_currency_gated=True, base_current_read=False)
    with pytest.raises(FactNotGathered):
        decide_branch(f)


@pytest.mark.parametrize("strategy", ["front_of_train", "broadcast"])
def test_stale_base_requests_a_sync_before_deferring(strategy: str) -> None:
    f = mf.branch_facts(
        config=mf.cfg(update_branch_strategy=strategy),
        base_currency_gated=True,
        base_current_read=True,
        base_current=False,
        should_update_branch=True,
    )
    gate = decide_branch(f)
    assert (gate.kind, gate.request_sync) == (StageKind.PROCEED, True)
    assert gate.stop is None


def test_sync_not_wanted_falls_through_to_the_gate() -> None:
    f = mf.branch_facts(
        base_currency_gated=True,
        base_current_read=True,
        base_current=False,
        should_update_branch=False,
    )
    assert decide_branch(f).stale_base_reason == "base_stale"


def test_sync_request_needs_the_should_update_fact() -> None:
    f = mf.branch_facts(
        base_currency_gated=True,
        base_current_read=True,
        base_current=False,
        should_update_branch=None,
    )
    with pytest.raises(FactNotGathered):
        decide_branch(f)


def test_queued_pr_suppresses_our_sync_but_not_the_freshness_read() -> None:
    f = mf.branch_facts(
        persisted=mf.persisted(status="mergequeue"),
        base_currency_gated=True,
        base_current_read=True,
        base_current=False,
        should_update_branch=True,
    )
    gate = decide_branch(f)
    assert gate.request_sync is False
    assert gate.already_in_mergequeue is True
    assert (gate.kind, gate.stale_base_reason) == (PlanKind.WAIT, "base_stale")


def test_reverted_hand_off_is_not_treated_as_queued() -> None:
    f = mf.branch_facts(
        admission=mf.admission(mergequeue_label_reverted=True),
        persisted=mf.persisted(status="mergequeue"),
        base_currency_gated=True,
        base_current_read=True,
        base_current=False,
        should_update_branch=True,
    )
    gate = decide_branch(f)
    assert gate.already_in_mergequeue is False
    assert gate.request_sync is True


@pytest.mark.parametrize(
    ("outcome", "sync_failed"),
    [
        pytest.param(SyncOutcome.FAILED, True, id="update-refused-or-unverified"),
        pytest.param(SyncOutcome.SAME_HEAD, False, id="same-head"),
        pytest.param(SyncOutcome.NEW_HEAD, False, id="new-head-fresh-base"),
        pytest.param(SyncOutcome.NOT_ATTEMPTED, False, id="preview-not-attempted"),
        pytest.param(SyncOutcome.SKIPPED_QUEUED, False, id="skipped-queued"),
    ],
)
def test_second_call_with_sync_outcome(outcome: SyncOutcome, sync_failed: bool) -> None:
    f = mf.branch_facts(
        base_currency_gated=True,
        base_current_read=True,
        base_current=True,
        should_update_branch=True,
    )
    gate = decide_branch(f, outcome)
    assert (gate.kind, gate.sync_failed, gate.request_sync) == (
        StageKind.PROCEED,
        sync_failed,
        False,
    )


def test_failed_sync_skips_the_freshness_gate() -> None:
    f = mf.branch_facts(
        base_currency_gated=True,
        base_current_read=True,
        base_current=False,
        should_update_branch=True,
    )
    gate = decide_branch(f, SyncOutcome.FAILED)
    assert (gate.kind, gate.sync_failed) == (StageKind.PROCEED, True)


def test_new_head_with_a_stale_reread_still_defers() -> None:
    f = mf.branch_facts(
        base_currency_gated=True,
        base_current_read=True,
        base_current=False,
        should_update_branch=True,
    )
    assert decide_branch(f, SyncOutcome.NEW_HEAD).stale_base_reason == "base_stale"


def test_non_sync_strategy_reads_the_gate_directly_without_a_sync_request() -> None:
    # Only reachable with a hand-built config (validation admits front_of_train,
    # broadcast, off): the gate is applied for any strategy that is not "off".
    f = mf.branch_facts(
        config=mf.cfg(update_branch_strategy="custom"),
        base_currency_gated=True,
        base_current_read=True,
        base_current=False,
        should_update_branch=True,
    )
    gate = decide_branch(f)
    assert (gate.request_sync, gate.stale_base_reason) == (False, "base_stale")


def test_conflict_blocks_the_base_gate_entirely() -> None:
    f = mf.branch_facts(merge_conflict=True, issue_status=None, base_currency_gated=None)
    gate = decide_branch(f)
    assert (gate.kind, gate.sync_failed, gate.merge_conflict) == (StageKind.PROCEED, True, True)
