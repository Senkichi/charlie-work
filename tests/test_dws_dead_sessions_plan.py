"""Decision tables for the dead-session lane's per-session plans (issue #2111).

Each test calls a pure plan function with plain facts: no ``gh``, filesystem, clock or
state fixture. One block per arm of ``_reap_launch_failed`` / ``_reap_dead`` /
reclaim. (Named ``test_dws_*`` so the dormant-module guard does not mistake this for
a one-module-one-test file.)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from charlie_work.dead_worker_sweep.decide_dead_sessions_plan import (
    DEAD_REAP_PERSIST_SOURCE,
    LAUNCH_FAILURE_PERSIST_SOURCE,
    DeadClassification,
    EmitBackgroundExit,
    EmitProviderSuspended,
    EscalateLaunchFailure,
    NoPrGate,
    OpenPrRoute,
    PersistFailure,
    ReapSidecar,
    ReclaimOrRoute,
    Reclaim,
    RestoreRework,
    WarnLiteralTmp,
    no_pr_gate,
    open_pr_candidate_route,
    open_pr_route,
    plan_dead_classification,
    plan_dead_reap,
    plan_launch_escalation,
    plan_launch_failed,
    plan_reclaim_commit,
    reclaim_route,
    scope_adjusted_kind,
    wants_publish_salvage,
    wants_salvage_from_unsafe,
)

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
STAMP = "2026-09-30T12:00:00Z"


# -- _reap_launch_failed ----------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "has_open_pr", "expected"),
    [
        # classified, deterministic, no PR: persist, escalate, reap, restore (in order)
        (
            "worker_blocked",
            False,
            (
                PersistFailure(LAUNCH_FAILURE_PERSIST_SOURCE),
                EscalateLaunchFailure(),
                ReapSidecar(),
                RestoreRework(),
            ),
        ),
        # deterministic but a PR is open: never escalates
        (
            "worker_blocked",
            True,
            (PersistFailure(LAUNCH_FAILURE_PERSIST_SOURCE), ReapSidecar(), RestoreRework()),
        ),
        # ordinary kind: persisted, not escalated
        (
            "launch_failed",
            False,
            (PersistFailure(LAUNCH_FAILURE_PERSIST_SOURCE), ReapSidecar(), RestoreRework()),
        ),
        # unclassified: nothing to persist, nothing to escalate
        (None, False, (ReapSidecar(), RestoreRework())),
        (None, True, (ReapSidecar(), RestoreRework())),
    ],
)
def test_launch_failed_plan_orders_persist_escalate_reap_restore(kind, has_open_pr, expected):
    assert plan_launch_failed(kind, has_open_pr=has_open_pr) == expected


def test_launch_escalation_always_counts_itself_and_names_the_kind():
    plan = plan_launch_escalation(
        ["a"], "worker_blocked", now=NOW, active_labels={"z-label", "a-label"}
    )
    assert plan.reason == "worker_blocked"
    assert plan.reason_class == "mechanical"
    assert plan.redispatch_at == ("a", STAMP)
    assert plan.removed_labels == ("a-label", "z-label")


def test_launch_escalation_of_a_judgment_kind_is_judgment_class():
    plan = plan_launch_escalation(
        (), "worktree_unsafe_local_commits", now=NOW, active_labels={"agent:in-progress"}
    )
    assert plan.reason_class == "judgment"


@pytest.mark.parametrize(("ahead", "expected"), [(0, False), (1, True), (7, True)])
def test_unsafe_launch_failure_salvages_only_with_commits_ahead(ahead, expected):
    assert wants_salvage_from_unsafe(ahead_count=ahead) is expected


# -- _reap_dead: classification --------------------------------------------------


@pytest.mark.parametrize(
    ("is_completed", "unknown", "expected"),
    [
        # completed: classify first (post-mortem after), log-tail matching skipped
        (True, False, DeadClassification(False, True, "unpublished_work")),
        (True, True, DeadClassification(False, True, "unpublished_work")),
        # otherwise post-mortem first so worker_blocked can still escalate
        (False, False, DeadClassification(True, False, "stalled")),
        (False, True, DeadClassification(True, False, None)),
    ],
)
def test_dead_classification_arms(is_completed, unknown, expected):
    assert (
        plan_dead_classification(is_completed=is_completed, worktree_unknown=unknown) == expected
    )


def test_dead_classification_background_exit_overrides_stalled_not_completed_or_unknown():
    bg = "worker_exited_with_background_work"
    assert plan_dead_classification(
        is_completed=False, worktree_unknown=False, background_exit=True
    ) == DeadClassification(True, False, bg)
    assert (
        plan_dead_classification(
            is_completed=False, worktree_unknown=True, background_exit=True
        ).fallback_kind
        is None
    )
    assert (
        plan_dead_classification(
            is_completed=True, worktree_unknown=False, background_exit=True
        ).fallback_kind
        == "unpublished_work"
    )


# -- _reap_dead: reap steps -------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (
            "stalled",
            (
                PersistFailure(DEAD_REAP_PERSIST_SOURCE),
                ReapSidecar(),
                WarnLiteralTmp(),
                ReclaimOrRoute(),
            ),
        ),
        (
            "provider_suspended",
            (
                PersistFailure(DEAD_REAP_PERSIST_SOURCE),
                ReapSidecar(),
                WarnLiteralTmp(),
                EmitProviderSuspended(),
                ReclaimOrRoute(),
            ),
        ),
        (
            "worker_exited_with_background_work",
            (
                PersistFailure(DEAD_REAP_PERSIST_SOURCE),
                ReapSidecar(),
                WarnLiteralTmp(),
                EmitBackgroundExit(),
                ReclaimOrRoute(),
            ),
        ),
        (None, (ReapSidecar(), WarnLiteralTmp(), ReclaimOrRoute())),
    ],
)
def test_dead_reap_plan_orders_persist_reap_warn_suspend_reclaim(kind, expected):
    assert plan_dead_reap(kind) == expected


def test_dead_reap_always_reaps_before_it_signals_or_reclaims():
    for kind in (None, "stalled", "provider_suspended", "worker_blocked"):
        steps = plan_dead_reap(kind)
        assert steps.index(ReapSidecar()) < steps.index(WarnLiteralTmp())
        assert steps[-1] == ReclaimOrRoute()


# -- reclaim ---------------------------------------------------------------------


def test_reclaim_route_follows_open_pr_presence():
    assert reclaim_route(has_open_pr=False) is Reclaim.NO_OPEN_PR
    assert reclaim_route(has_open_pr=True) is Reclaim.OPEN_PR


@pytest.mark.parametrize(
    ("found", "active", "expected"),
    [
        (False, {"agent:in-progress"}, NoPrGate.SKIP),
        (False, set(), NoPrGate.SKIP),
        (True, set(), NoPrGate.PARK),
        (True, {"agent:in-progress"}, NoPrGate.PROCEED),
    ],
)
def test_no_pr_gate_arms(found, active, expected):
    assert no_pr_gate(issue_found=found, active_labels=active) is expected


@pytest.mark.parametrize(
    ("ahead", "root", "expected"),
    [(0, True, False), (3, False, False), (3, True, True), (0, False, False)],
)
def test_publish_salvage_needs_commits_and_a_repo_root(ahead, root, expected):
    assert wants_publish_salvage(ahead_count=ahead, has_repo_root=root) is expected


def test_a_cross_repo_scope_hop_overrides_the_failure_kind():
    assert scope_adjusted_kind("stalled", scope_passed=True) == "stalled"
    assert scope_adjusted_kind(None, scope_passed=True) is None
    assert scope_adjusted_kind("stalled", scope_passed=False) == "cross_repo_hop"
    assert scope_adjusted_kind(None, scope_passed=False) == "cross_repo_hop"


def _commit(
    windowed=(), kind: str | None = "stalled", cap=3, active=("agent:in-progress",), ready=False
):
    return plan_reclaim_commit(
        windowed,
        kind,
        now=NOW,
        max_auto_redispatch=cap,
        active_labels=set(active),
        ready_label_present=ready,
    )


def test_reclaim_relabels_to_ready_and_records_the_redispatch():
    plan = _commit(["a"])
    assert plan.escalate is False
    assert plan.redispatch_at == ("a", STAMP)
    assert plan.removed_labels == ("agent:in-progress",)
    assert plan.add_ready is True


def test_reclaim_does_not_re_add_a_ready_label_already_present():
    assert _commit(ready=True).add_ready is False


def test_reclaim_escalates_past_the_cap():
    plan = _commit(["a", "b", "c"], cap=3)
    assert (plan.escalate, plan.reason, plan.reason_class) == (
        True,
        "redispatch_cap_exceeded",
        "mechanical",
    )


def test_reclaim_escalates_a_cross_repo_hop_on_first_occurrence():
    plan = _commit(kind=scope_adjusted_kind("stalled", scope_passed=False))
    assert (plan.escalate, plan.reason) == (True, "cross_repo_hop")


def test_reclaim_removes_every_active_label_sorted():
    assert _commit(active=("z", "a")).removed_labels == ("a", "z")


@pytest.mark.parametrize(
    ("completed", "has_pr", "expected"),
    [
        (True, True, OpenPrRoute.SKIP),
        (True, False, OpenPrRoute.SKIP),
        (False, False, OpenPrRoute.RESTORE),
        (False, True, OpenPrRoute.INSPECT_PR),
    ],
)
def test_open_pr_route_arms(completed, has_pr, expected):
    assert open_pr_route(is_completed=completed, has_rework_pr=has_pr) is expected


def test_open_pr_candidate_route_arms():
    assert open_pr_candidate_route(is_candidate=True) is OpenPrRoute.ROUTE_PRE_REVIEW
    assert open_pr_candidate_route(is_candidate=False) is OpenPrRoute.RESTORE


def test_persist_source_constants_match_the_shell_audit_literals():
    # dead_sessions.py passes these as string literals (the throttle-source audit
    # guard rejects a variable); this keeps the plan's constants honest.
    assert LAUNCH_FAILURE_PERSIST_SOURCE == "dead_sessions_launch_failure"
    assert DEAD_REAP_PERSIST_SOURCE == "dead_sessions_reap"
