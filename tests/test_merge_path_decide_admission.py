"""Decision table for ``decide_admission`` (merge-path stage 1), via its public interface."""

from __future__ import annotations

from typing import Any

import _merge_path_facts as mf
import pytest
from charlie_work.merge_path import FactNotGathered, PlanKind, decide_admission
from charlie_work.merge_path.model import StageKind


@pytest.mark.parametrize(
    ("over", "kind"),
    [
        pytest.param(
            {"persisted": mf.persisted(status="merged")}, PlanKind.SKIP, id="already-merged"
        ),
        pytest.param({"pr_found": False}, PlanKind.NOT_FOUND, id="pr-not-found"),
        pytest.param(
            {"persisted": mf.persisted(status="merged"), "pr_found": False},
            PlanKind.SKIP,
            id="merged-wins-over-not-found",
        ),
        pytest.param({"persisted": None}, StageKind.PROCEED, id="no-persisted-entry"),
        pytest.param(
            {"verdict": mf.verdict(approved=False, reviewed_head_sha=None)},
            StageKind.PROCEED,
            id="unapproved-proceeds",
        ),
    ],
)
def test_terminal_and_trivial_kinds(over: dict[str, Any], kind: object) -> None:
    assert decide_admission(mf.admission_facts(**over)).kind == kind


def test_unapproved_never_needs_head_or_carry_facts() -> None:
    f = mf.admission_facts(
        verdict=mf.verdict(approved=False, reviewed_head_sha=None),
        live_head_sha=mf.NEW_HEAD,
        carry_forward=None,
    )
    adm = decide_admission(f)
    assert adm.kind == StageKind.PROCEED
    assert adm.approved is False
    assert adm.head_moved is False


def test_unchanged_head_proceeds_without_carry_fact() -> None:
    adm = decide_admission(mf.admission_facts(carry_forward=None))
    assert (adm.kind, adm.approved, adm.head_moved, adm.carry_forward_needed) == (
        StageKind.PROCEED,
        True,
        False,
        False,
    )


@pytest.mark.parametrize(
    ("reviewed", "live"),
    [
        pytest.param(mf.HEAD, mf.NEW_HEAD, id="head-differs"),
        pytest.param(None, mf.HEAD, id="no-reviewed-head"),
    ],
)
def test_moved_head_without_carry_forward_requests_re_review(
    reviewed: str | None, live: str
) -> None:
    f = mf.admission_facts(
        verdict=mf.verdict(reviewed_head_sha=reviewed), live_head_sha=live, carry_forward=False
    )
    adm = decide_admission(f)
    assert adm.kind == PlanKind.REQUEST_REWORK
    assert adm.head_moved is True
    assert (adm.stamp_reviewing, adm.transition_review_started, adm.record_head_moved) == (
        True,
        True,
        True,
    )


@pytest.mark.parametrize(
    ("over", "stamp", "transition", "record"),
    [
        pytest.param({}, True, True, True, id="plain"),
        pytest.param({"pr_escalated": True}, False, False, False, id="pr-escalated"),
        pytest.param({"issue_escalated": True}, False, False, False, id="issue-escalated"),
        pytest.param(
            {"config": mf.cfg(review_dispatch_enabled=False)},
            False,
            False,
            True,
            id="dispatch-disabled-still-records-head-moved",
        ),
        pytest.param({"issue_number": None}, True, False, True, id="no-issue-no-transition"),
    ],
)
def test_re_review_bookkeeping(
    over: dict[str, Any], stamp: bool, transition: bool, record: bool
) -> None:
    f = mf.admission_facts(live_head_sha=mf.NEW_HEAD, carry_forward=False, **over)
    adm = decide_admission(f)
    assert adm.kind == PlanKind.REQUEST_REWORK
    assert (adm.stamp_reviewing, adm.transition_review_started, adm.record_head_moved) == (
        stamp,
        transition,
        record,
    )


def test_escalated_re_review_is_reported_but_flagged() -> None:
    f = mf.admission_facts(live_head_sha=mf.NEW_HEAD, carry_forward=False, issue_escalated=True)
    adm = decide_admission(f)
    assert adm.kind == PlanKind.REQUEST_REWORK
    assert adm.escalated is True


def test_moved_head_with_carry_forward_asks_shell_then_proceeds_on_observation() -> None:
    f = mf.admission_facts(live_head_sha=mf.NEW_HEAD, carry_forward=True)
    first = decide_admission(f)
    assert (first.kind, first.carry_forward_needed) == (StageKind.PROCEED, True)
    second = decide_admission(f, observed_carry_forward=True)
    assert (second.kind, second.carry_forward_needed) == (StageKind.PROCEED, False)


def test_observed_carry_forward_does_not_recheck_the_head() -> None:
    # Today's behaviour on a refetch race: no second head-moved check after a carry-forward.
    f = mf.admission_facts(live_head_sha=mf.NEW_HEAD, carry_forward=None)
    assert decide_admission(f, observed_carry_forward=True).kind == StageKind.PROCEED


def test_moved_head_with_unread_carry_fact_fails_loudly() -> None:
    f = mf.admission_facts(live_head_sha=mf.NEW_HEAD, carry_forward=None)
    with pytest.raises(FactNotGathered):
        decide_admission(f)


@pytest.mark.parametrize(
    ("preview_flag", "kind"),
    [
        pytest.param(False, PlanKind.REQUEST_REWORK, id="live-re-reviews"),
        pytest.param(True, StageKind.PROCEED, id="legacy-preview-proceeds"),
    ],
)
def test_moved_head_with_unknown_live_head(preview_flag: bool, kind: object) -> None:
    # Known divergence: the legacy dry-run skips the carry-forward check and proceeds
    # when the live head is unknown; the live path re-reviews.
    f = mf.admission_facts(live_head_sha=None, carry_forward=None)
    adm = decide_admission(f, preview_unknown_live_head_proceeds=preview_flag)
    assert adm.kind == kind


@pytest.mark.parametrize(
    ("label", "status", "live_labels", "reason", "reverted", "self_revoked"),
    [
        pytest.param(mf.LABEL, "mergequeue", frozenset(), None, True, False, id="label-gone"),
        pytest.param(
            mf.LABEL,
            "mergequeue",
            frozenset(),
            "stale_head_pending_carry_forward",
            True,
            True,
            id="label-gone-self-revoked",
        ),
        pytest.param(
            mf.LABEL,
            "mergequeue",
            frozenset(),
            "not_approved",
            True,
            False,
            id="label-gone-not-approved",
        ),
        pytest.param(
            mf.LABEL,
            "mergequeue",
            frozenset({mf.LABEL}),
            None,
            False,
            False,
            id="label-still-on-pr",
        ),
        pytest.param(mf.LABEL, "approved", frozenset(), None, False, False, id="never-queued"),
        pytest.param(
            None, "mergequeue", frozenset(), None, False, False, id="no-label-configured"
        ),
        pytest.param(
            mf.LABEL,
            "approved",
            frozenset(),
            "stale_head_pending_carry_forward",
            False,
            False,
            id="reason-without-revert-is-not-self-revoked",
        ),
    ],
)
def test_mergequeue_revert_detection(
    label: str | None,
    status: str,
    live_labels: frozenset[str],
    reason: str | None,
    reverted: bool,
    self_revoked: bool,
) -> None:
    f = mf.admission_facts(
        config=mf.cfg(mergequeue_label=label),
        persisted=mf.persisted(status=status, mergequeue_revoked_reason=reason),
        live_labels=live_labels,
    )
    adm = decide_admission(f)
    assert (adm.mergequeue_label_reverted, adm.self_revoked_stale_head) == (reverted, self_revoked)


def test_revert_flags_survive_an_unapproved_verdict() -> None:
    f = mf.admission_facts(
        config=mf.cfg(mergequeue_label=mf.LABEL),
        persisted=mf.persisted(status="mergequeue"),
        verdict=mf.verdict(approved=False),
    )
    adm = decide_admission(f)
    assert adm.mergequeue_label_reverted is True
    assert adm.approved is False
