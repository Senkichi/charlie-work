"""Decision table for ``decide_readiness`` (merge-path stage 3), via its public interface."""

from __future__ import annotations

import itertools
from typing import Any

import _merge_path_facts as mf
import pytest
from charlie_work.merge_path import PlanKind, RevertStatus, decide_readiness
from charlie_work.merge_path.model import StageKind
from charlie_work.merge_path.rules import readiness_no_ci_stall

OLD = "2026-01-01T11:00:00Z"  # 60 minutes before mf.NOW
STALLED: dict[str, Any] = {
    "checks": mf.summary(passed=(), missing=mf.REQUIRED),
    "check_names_seen": frozenset(),
    "pr_updated_at": OLD,
}
INFRA: dict[str, Any] = {"checks": mf.summary(passed=(), infra_failed=("Tests passed",))}


def _branch(**over: Any) -> Any:
    return mf.branch_gate(**over)


# --------------------------------------------------------------------------- #
# Cross-PR revert
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("over", "route"),
    [
        pytest.param({}, True, id="bound-issue-no-status"),
        pytest.param({"issue_status": "pr_open"}, True, id="bound-issue-pr-open"),
        pytest.param({"issue_status": "escalated"}, False, id="already-escalated"),
        pytest.param({"issue_status": "blocked"}, False, id="blocked"),
        pytest.param({"issue_status": "dispatched"}, False, id="dispatched"),
        pytest.param({"issue_status": "dispatch_pending"}, False, id="dispatch-pending"),
        pytest.param({"issue_status": "manifest_written"}, False, id="manifest-written"),
        pytest.param({"issue_status": "rework_requested"}, False, id="rework-requested"),
        pytest.param({"issue_number": None}, False, id="no-issue"),
    ],
)
def test_detected_revert_routes_only_when_the_issue_is_free(
    over: dict[str, Any], route: bool
) -> None:
    r = decide_readiness(
        mf.readiness_facts(revert=RevertStatus.DETECTED, revert_reason="reverts #3", **over)
    )
    assert r.cross_pr_revert_detected is True
    assert r.route_cross_pr_revert is route
    assert r.cross_pr_revert_reason == "reverts #3"
    assert r.gate.sync_failed is True
    assert r.gate.can_merge is False


def test_undetermined_revert_fails_closed_and_never_routes() -> None:
    r = decide_readiness(mf.readiness_facts(revert=RevertStatus.UNDETERMINED))
    assert (r.cross_pr_revert_undetermined, r.cross_pr_revert_detected) == (True, False)
    assert r.route_cross_pr_revert is False
    assert r.gate.sync_failed is True


@pytest.mark.parametrize(
    "revert", [RevertStatus.DETECTED, RevertStatus.UNDETERMINED], ids=["detected", "undetermined"]
)
def test_revert_is_ignored_for_an_unapproved_pr(revert: RevertStatus) -> None:
    r = decide_readiness(
        mf.readiness_facts(branch=_branch(admission=mf.admission(approved=False)), revert=revert)
    )
    assert (r.cross_pr_revert_detected, r.cross_pr_revert_undetermined) == (False, False)
    assert r.gate.sync_failed is False


def test_revert_is_ignored_when_the_branch_sync_already_failed() -> None:
    r = decide_readiness(
        mf.readiness_facts(branch=_branch(sync_failed=True), revert=RevertStatus.DETECTED)
    )
    assert (r.cross_pr_revert_detected, r.route_cross_pr_revert) == (False, False)
    assert r.gate.sync_failed is True


# --------------------------------------------------------------------------- #
# Gate formula
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("ready", "approved", "require", "sync_failed"),
    list(itertools.product([True, False], repeat=4)),
    ids=lambda v: "T" if v else "F",
)
def test_gate_inputs_follow_the_can_merge_formula(
    ready: bool, approved: bool, require: bool, sync_failed: bool
) -> None:
    checks = mf.summary() if ready else mf.summary(passed=(), failed=("Lint & Format",))
    r = decide_readiness(
        mf.readiness_facts(
            config=mf.cfg(require_approved_review=require),
            branch=_branch(admission=mf.admission(approved=approved), sync_failed=sync_failed),
            checks=checks,
        )
    )
    assert r.gate.summary_ready is ready
    assert r.gate.approved is approved
    assert r.gate.require_approved_review is require
    assert r.gate.sync_failed is sync_failed
    assert r.gate.can_merge is (ready and (approved or not require) and not sync_failed)


def test_gate_payload_carries_the_four_issue_1060_keys() -> None:
    r = decide_readiness(mf.readiness_facts())
    assert r.gate.as_payload() == {
        "summary_ready": True,
        "approved": True,
        "require_approved_review": True,
        "sync_failed": False,
    }


# --------------------------------------------------------------------------- #
# Readiness-no-CI stall
# --------------------------------------------------------------------------- #


def test_stall_routes_an_approved_pr_whose_required_checks_never_started() -> None:
    r = decide_readiness(mf.readiness_facts(**STALLED))
    assert (r.kind, r.readiness_stall) == (PlanKind.REQUEST_REWORK, True)


@pytest.mark.parametrize(
    ("over", "why"),
    [
        pytest.param({"checks_unavailable": True}, "checks-unavailable", id="checks-unavailable"),
        pytest.param({"issue_number": None}, "no-issue", id="no-issue"),
        pytest.param({"pr_updated_at": "2026-01-01T11:55:00Z"}, "recent", id="updated-recently"),
        pytest.param({"pr_updated_at": None}, "no-updated-at", id="no-updated-at"),
        pytest.param(
            {"check_names_seen": frozenset({"Tests passed"})}, "started", id="check-started"
        ),
        pytest.param({"config": mf.cfg(readiness_no_ci_minutes=0)}, "off", id="detection-off"),
        pytest.param(
            {"config": mf.cfg(required_checks=())}, "no-required", id="no-required-checks"
        ),
        pytest.param(
            {"checks": mf.summary(passed=(), pending=mf.REQUIRED)}, "pending", id="pending-only"
        ),
        pytest.param(
            {"branch": _branch(admission=mf.admission(approved=False))},
            "unapproved",
            id="unapproved",
        ),
        pytest.param({"branch": _branch(sync_failed=True)}, "sync-failed", id="sync-failed"),
        pytest.param({"checks": mf.summary()}, "ready", id="summary-ready"),
    ],
)
def test_stall_is_not_declared(over: dict[str, Any], why: str) -> None:
    r = decide_readiness(mf.readiness_facts(**{**STALLED, **over}))
    assert r.readiness_stall is False, why
    assert r.kind != PlanKind.REQUEST_REWORK, why


@pytest.mark.parametrize(
    "status",
    [
        "escalated",
        "blocked",
        "dispatched",
        "dispatch_pending",
        "manifest_written",
        "rework_requested",
    ],
)
def test_stall_is_suppressed_for_statuses_with_rework_routed(status: str) -> None:
    r = decide_readiness(mf.readiness_facts(issue_status=status, **STALLED))
    assert (r.readiness_stall, r.kind) == (False, StageKind.PROCEED)


def test_revert_failure_suppresses_the_stall() -> None:
    r = decide_readiness(mf.readiness_facts(revert=RevertStatus.UNDETERMINED, **STALLED))
    assert r.readiness_stall is False


@pytest.mark.parametrize(
    ("names", "updated_at", "minutes", "required", "expected"),
    [
        pytest.param(set(), OLD, 15, mf.REQUIRED, True, id="never-started-and-old"),
        pytest.param({"other"}, OLD, 15, mf.REQUIRED, True, id="only-unrelated-checks"),
        pytest.param({"Lint & Format"}, OLD, 15, mf.REQUIRED, False, id="one-required-seen"),
        pytest.param(set(), OLD, 0, mf.REQUIRED, False, id="zero-minutes"),
        pytest.param(set(), OLD, 15, (), False, id="no-required"),
        pytest.param(set(), "2026-01-01T11:50:00Z", 15, mf.REQUIRED, False, id="inside-window"),
        pytest.param(set(), None, 15, mf.REQUIRED, False, id="no-timestamp"),
    ],
)
def test_fact_based_stall_predicate(
    names: set[str],
    updated_at: str | None,
    minutes: int,
    required: tuple[str, ...],
    expected: bool,
) -> None:
    assert (
        readiness_no_ci_stall(
            check_names_seen=names,
            updated_at=updated_at,
            now=mf.NOW,
            required_checks=required,
            minutes=minutes,
        )
        is expected
    )


# --------------------------------------------------------------------------- #
# Infra remediation (sole-blocker predicate)
# --------------------------------------------------------------------------- #


def test_infra_failure_as_sole_blocker_is_eligible() -> None:
    r = decide_readiness(mf.readiness_facts(**INFRA))
    assert (r.infra_eligible, r.kind) == (True, PlanKind.RERUN_OR_ESCALATE)


def test_pending_checks_do_not_stop_infra_eligibility() -> None:
    checks = mf.summary(passed=(), infra_failed=("Tests passed",), pending=("Lint & Format",))
    assert decide_readiness(mf.readiness_facts(checks=checks)).infra_eligible is True


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"is_draft": True}, id="draft"),
        pytest.param({"pr_escalated": True}, id="pr-escalated"),
        pytest.param({"issue_escalated": True}, id="issue-escalated"),
        pytest.param({"issue_number": None}, id="no-issue"),
        pytest.param({"branch": _branch(admission=mf.admission(approved=False))}, id="unapproved"),
        pytest.param({"branch": _branch(sync_failed=True)}, id="sync-failed"),
        pytest.param({"revert": RevertStatus.UNDETERMINED}, id="revert-undetermined"),
        pytest.param(
            {
                "checks": mf.summary(
                    passed=(), infra_failed=("Tests passed",), failed=("Lint & Format",)
                )
            },
            id="also-failed",
        ),
        pytest.param(
            {
                "checks": mf.summary(
                    passed=(), infra_failed=("Tests passed",), missing=("Lint & Format",)
                )
            },
            id="also-missing",
        ),
        pytest.param(
            {
                "checks": mf.summary(
                    passed=(), infra_failed=("Tests passed",), unavailable=("Lint & Format",)
                )
            },
            id="also-unavailable",
        ),
        pytest.param(
            {
                "checks": mf.summary(
                    passed=(), infra_failed=("Tests passed",), infra_blocked=("Lint & Format",)
                )
            },
            id="also-infra-blocked",
        ),
        pytest.param({"checks": mf.summary()}, id="no-infra-failure"),
    ],
)
def test_infra_is_not_eligible(over: dict[str, Any]) -> None:
    r = decide_readiness(mf.readiness_facts(**{**INFRA, **over}))
    assert r.infra_eligible is False
    assert r.kind == StageKind.PROCEED


def test_infra_eligibility_when_approval_is_not_required() -> None:
    r = decide_readiness(
        mf.readiness_facts(
            config=mf.cfg(require_approved_review=False),
            branch=_branch(admission=mf.admission(approved=False)),
            **INFRA,
        )
    )
    assert r.infra_eligible is True


def test_stall_takes_precedence_over_infra() -> None:
    checks = mf.summary(passed=(), missing=mf.REQUIRED, infra_failed=("x",))
    r = decide_readiness(mf.readiness_facts(**{**STALLED, "checks": checks}))
    assert r.kind == PlanKind.REQUEST_REWORK


# --------------------------------------------------------------------------- #
# De-escalation and pass-through facts
# --------------------------------------------------------------------------- #

DEESC: dict[str, Any] = {
    "config": mf.cfg(human_merge_labels=("human-merge",)),
    "issue_status": "escalated",
    "issue_reason_class": "policy",
}


def test_policy_escalation_deescalates_once_the_human_hold_is_lifted() -> None:
    assert decide_readiness(mf.readiness_facts(**DEESC)).deescalate is True


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"issue_reason_class": "infra"}, id="other-reason-class"),
        pytest.param({"issue_reason_class": None}, id="no-reason-class"),
        pytest.param({"issue_status": "blocked"}, id="not-escalated"),
        pytest.param({"human_merge_hold": True}, id="hold-still-on"),
        pytest.param({"human_merge_check_unavailable": True}, id="hold-read-failed"),
        pytest.param({"issue_number": None}, id="no-issue"),
        pytest.param({"config": mf.cfg(human_merge_labels=())}, id="feature-off"),
    ],
)
def test_no_deescalation(over: dict[str, Any]) -> None:
    assert decide_readiness(mf.readiness_facts(**{**DEESC, **over})).deescalate is False


def test_hold_facts_and_warnings_pass_through() -> None:
    r = decide_readiness(
        mf.readiness_facts(
            human_merge_hold=True,
            human_merge_check_unavailable=True,
            containment_warnings=("w1",),
            checks_unavailable=True,
        )
    )
    assert (r.human_merge_hold, r.human_merge_check_unavailable) == (True, True)
    assert r.containment_warnings == ("w1",)
    assert r.checks_unavailable is True


@pytest.mark.parametrize(
    ("checks", "pending_only"),
    [
        pytest.param(mf.summary(passed=(), pending=mf.REQUIRED), True, id="pending"),
        pytest.param(
            mf.summary(passed=(), pending=("a",), failed=("b",)), False, id="pending-and-failed"
        ),
        pytest.param(mf.summary(), False, id="ready"),
    ],
)
def test_pending_only_flag(checks: Any, pending_only: bool) -> None:
    assert decide_readiness(mf.readiness_facts(checks=checks)).pending_only is pending_only
