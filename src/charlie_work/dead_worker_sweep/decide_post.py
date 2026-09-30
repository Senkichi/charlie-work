"""The post-lock phase: outcome apply, the review drain, no-op drain, label edges.

Ports ``orphaned_worker_review_drain.drain_orphaned_worker_review_routes``. Each
disposition is a ``GuardedUpdate`` *request*: the shell re-checks ``require_status``
and ``require_pr_reviewed_head`` against a fresh load under a short lock, so a
concurrent transition that landed while ``review()`` ran still wins -- and the
answer comes back to the flow. The event, the recovery (and so the label edge)
and the no-op route are recorded only for a write that committed, which is what
the original's single in-lock check/flip/event guaranteed. When the first
disposition is refused the chain falls through to the next, as its ``elif`` did.
"""

from __future__ import annotations

from .decide_common import Flow, emit
from .model import (
    ApplyOutcomes,
    DrainNoOp,
    GuardedUpdate,
    LockOutcome,
    NoOpRoute,
    PreOutcome,
    ReadAppliedHeads,
    ReportStaleEvidence,
    Review,
    ReviewRoute,
    SweepFacts,
    TransitionLabel,
)


def _route_flow(
    facts: SweepFacts, route: ReviewRoute, no_op: list[NoOpRoute], recovered: list[int]
) -> Flow:
    number = route.issue_number
    if route.reason == "dead_worker_completed_outcome":
        applied = yield ReadAppliedHeads(number)
        if not (isinstance(applied, dict) and applied.get(str(number)) == route.live_head_sha):
            return
    result = yield Review(number, route.pr_number, route.reason)
    if result.raised_error is not None:
        # events.db only, as the original ``write_gate.log_event`` wrote it: a
        # persistently raising review() must not churn the capped state ring.
        yield emit(
            "orphaned_worker_review_route_failed",
            {
                "issue_number": number,
                "pr_number": route.pr_number,
                "reason": route.reason,
                "error": result.raised_error,
            },
            level="warning",
            audit_only=True,
        )
        return

    dispatched = result.entry_status == "dispatched"
    unchanged = result.pr_reviewed_head_sha == route.reviewed_head_sha
    blocked_route = result.routed_to_rework or result.closed_unmerged_converged
    routed = False
    rework_requested = False
    taken = False
    if (
        result.ok
        and not blocked_route
        and not result.escalation_deferred_live_worker
        and unchanged
        and dispatched
    ):
        routed = taken = yield GuardedUpdate(
            number,
            (("status", "reviewing"),),
            require_status="dispatched",
            require_pr_reviewed_head=route.reviewed_head_sha,
        )
    if (
        not taken
        and not result.ok
        and not blocked_route
        and unchanged
        and route.reason == "dead_worker_completed_outcome"
        and dispatched
    ):
        rework_requested = taken = yield GuardedUpdate(
            number,
            (("status", "rework_requested"), ("dispatched_at", None), ("orphan_drift_at", None)),
            require_status="dispatched",
            require_pr_reviewed_head=route.reviewed_head_sha,
        )
        if rework_requested:
            recovered.append(number)
    if not taken and not result.ok and not result.routed_to_rework and dispatched:
        drifted = yield GuardedUpdate(
            number,
            (("orphan_drift_fingerprint", route.fingerprint),),
            stamp_fields=("orphan_drift_at",),  # anchored when written, after review() returned
            require_status="dispatched",
        )
        if drifted and route.reason == "dead_worker_with_head_change" and result.is_no_op_rework:
            no_op.append(
                NoOpRoute(
                    issue_number=number,
                    pr_number=route.pr_number,
                    live_head_sha=route.live_head_sha,
                    reason=route.reason,
                    branch=result.entry_branch,
                )
            )

    if rework_requested:
        yield emit(
            "orphaned_worker_recovered",
            {
                "issue_number": number,
                "pr_number": route.pr_number,
                "previous_status": "dispatched",
                "new_status": "rework_requested",
                "reason": route.reason,
            },
        )
        return
    yield emit(
        "orphaned_worker_routed_to_review"
        if result.ok and not result.escalation_deferred_live_worker
        else "orphaned_worker_drift",
        {
            "issue_number": number,
            "pr_number": route.pr_number,
            "review_ok": result.ok,
            "routed": routed,
            "live_head_sha": route.live_head_sha,
            "reviewed_head_sha": route.reviewed_head_sha,
            "reason": route.reason,
        },
    )


def post_flow(facts: SweepFacts, pre: PreOutcome, lock: LockOutcome) -> Flow:
    yield ReportStaleEvidence("swept")
    if lock.outcome_apply_routes:
        yield ApplyOutcomes(lock.outcome_apply_routes)

    no_op = list(lock.no_op_routes)
    recovered: list[int] = []
    if facts.review_available:
        for route in lock.review_routes:
            yield from _route_flow(facts, route, no_op, recovered)
    for number in recovered:
        yield TransitionLabel(number, "rework_requested", persist_label_error=True)
    if no_op:
        yield DrainNoOp(tuple(no_op))
    for number in lock.reap_escalations:
        yield TransitionLabel(number, "escalated")
    return None
