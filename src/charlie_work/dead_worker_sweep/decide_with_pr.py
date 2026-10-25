"""The in-lock arm for a dead worker that still has an open PR.

Ports ``orphaned_worker_sweep.handle_dead_worker_with_pr`` and
``handle_dead_worker_completed_outcome``: completed-outcome recovery (#1911),
declared-blocked escalation, clean-exit no-op (#773), death reset to
``rework_requested`` (#1134), head-advanced review routing (#339), ``pr-open``
advance for an unverdicted PR (#1128), and the shared drift fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .constants import APPLIED_HEADS_KEY, NO_OP_DEFERRED_HEAD_KEY, PASSIVE_OPEN_STATUS
from .decide_common import Draft, Flow, LockAcc, drift_fingerprint, emit, flush
from .model import (
    AdvanceToPrOpen,
    CreditDeadWorker,
    Escalate,
    ExemptThrottleCleanExit,
    NoOpRoute,
    PreOutcome,
    ReadBlockedOutcome,
    ReadCompletedOutcome,
    ReadReviewDecision,
    ReadTerminal,
    ReviewRoute,
    SweepFacts,
)


@dataclass(frozen=True)
class _Ctx:
    number: int
    pr: Mapping[str, Any]
    pr_number: int
    reviewed: str | None
    live: str | None
    pid: Any
    exit_code: Any
    duration: Any

    def base(self) -> dict[str, Any]:
        return {
            "issue_number": self.number,
            "pr_number": self.pr_number,
            "previous_status": "dispatched",
        }

    def proc(self) -> dict[str, Any]:
        return {"pid": self.pid, "exit_code": self.exit_code, "duration_seconds": self.duration}


def _no_op_route(ctx: _Ctx, entry: Mapping[str, Any]) -> NoOpRoute:
    return NoOpRoute(
        issue_number=ctx.number,
        pr_number=ctx.pr_number,
        live_head_sha=ctx.live,
        reason="dead_worker_no_op",
        branch=ctx.pr.get("headRefName") or entry.get("branch_name"),
    )


def _resolve_outcome(facts: SweepFacts, draft: Draft, ctx: _Ctx) -> Flow:
    """The fresh, on-target completed outcome for this dead worker, or ``None``.

    A request is keyed by value, so the second caller on a pass (the same-head
    route re-resolves it for routing) is served from ``observed`` -- one read.
    """
    if not ctx.live:
        return None
    branch = ctx.pr.get("headRefName") or draft.work.get("branch_name")
    if not branch or not facts.repo.repo_root_is_path or not facts.repo.has_worktrees:
        return None
    return (yield ReadCompletedOutcome(ctx.number, ctx.pr_number, ctx.live, branch))


def _queue_apply(facts: SweepFacts, acc: LockAcc, ctx: _Ctx, outcome: Mapping[str, Any]) -> None:
    """Queue the post-lock apply unless this outcome's head is already recorded applied."""
    applied = (facts.locked or {}).get(APPLIED_HEADS_KEY) or {}
    if applied.get(str(ctx.number)) != outcome["head_sha"]:
        acc.outcome_apply_routes.append((ctx.number, ctx.pr_number))


def _completed_outcome(
    facts: SweepFacts, draft: Draft, acc: LockAcc, ctx: _Ctx, extra: Mapping[str, Any]
) -> Flow:
    """True when a fresh, on-target completed outcome exists (and was routed).

    Routing only: with ``apply_rework_outcome_on_every_route`` the apply was already
    queued ahead of the route fan-out (issue #2826), so queueing here too would
    double-queue it. With the switch off this is the only queueing site, as before.
    """
    entry = draft.work
    outcome = yield from _resolve_outcome(facts, draft, ctx)
    if outcome is None:
        return False
    if not facts.config.watchdog.apply_rework_outcome_on_every_route:
        _queue_apply(facts, acc, ctx, outcome)
    fingerprint = drift_fingerprint(
        reason="dead_worker_completed_outcome", reviewed_head_sha=ctx.reviewed
    )
    if entry.get("orphan_drift_fingerprint") == fingerprint:
        return True
    if facts.review_available:
        if entry.get("orphan_drift_at") is None:
            entry["orphan_drift_at"] = facts.stamp
        acc.review_routes.append(
            ReviewRoute(
                ctx.number,
                ctx.pr_number,
                ctx.reviewed,
                ctx.live,
                fingerprint,
                "dead_worker_completed_outcome",
            )
        )
        return True
    entry["orphan_drift_fingerprint"] = fingerprint
    entry["orphan_drift_at"] = facts.stamp
    yield emit(
        "orphaned_worker_drift",
        {
            **ctx.base(),
            "reason": "dead_worker_completed_outcome",
            **ctx.proc(),
            "worker_outcome_head_sha": outcome["head_sha"],
            **extra,
        },
    )
    return True


def _same_head(
    facts: SweepFacts,
    draft: Draft,
    acc: LockAcc,
    ctx: _Ctx,
    *,
    recovered_reason: str,
    extra: Mapping[str, Any],
) -> Flow:
    """Reviewed head == live head: completed, blocked, clean-exit no-op, or death."""
    entry = draft.work
    number = ctx.number
    if (yield from _completed_outcome(facts, draft, acc, ctx, extra)):
        return
    blocked = yield ReadBlockedOutcome(number)
    if blocked is not None:
        yield from flush(draft)
        yield Escalate(
            number,
            reason="worker_declared_blocked",
            reason_class="mechanical",
            pr_number=ctx.pr_number,
            issue_extra={"dispatched_at": None},
        )
        draft.invalidate()
        acc.reap_escalations.append(number)
        yield emit(
            "worker_declared_blocked",
            {
                **ctx.base(),
                "reason": "worker_declared_blocked",
                **extra,
                "reason_kind": str(blocked.get("reason_kind") or "unknown"),
                "detail": str(blocked.get("detail") or ""),
                **ctx.proc(),
            },
        )
        return
    if ctx.exit_code == 0:
        # Issue #2286: an exit-0 session whose log ENDS in a provider throttle is
        # a provider death, not a no-op -- restore it like one. The handler's
        # terminal-line check (never the whole transcript, per #656) decides;
        # it also refunds the dispatch stamp, flags the PR, and arms the
        # cooldown, so the fall-through path sees an unexempted exit as a real
        # no-op.
        throttle = yield ExemptThrottleCleanExit(
            number, dispatched_at=entry.get("dispatched_at"), pr_number=ctx.pr_number
        )
        if throttle.failure_kind is not None:
            entry["status"] = "rework_requested"
            entry["dispatched_at"] = None
            yield from flush(draft)
            acc.throttled_until = throttle.throttled_until
            yield emit(
                "orphaned_worker_recovered",
                {
                    **ctx.base(),
                    "new_status": "rework_requested",
                    "reason": "dead_worker_clean_exit_throttle",
                    **extra,
                    **ctx.proc(),
                    "worker_death_at": facts.stamp,
                    "failure_kind": throttle.failure_kind,
                },
            )
            return
        fingerprint = drift_fingerprint(
            reason="dead_worker_clean_exit_no_op", reviewed_head_sha=ctx.reviewed
        )
        acc.no_op_routes.append(_no_op_route(ctx, entry))
        if entry.get("orphan_drift_fingerprint") == fingerprint:
            return
        entry["orphan_drift_fingerprint"] = fingerprint
        entry["orphan_drift_at"] = facts.stamp
        yield emit(
            "orphaned_worker_drift",
            {**ctx.base(), "reason": "dead_worker_clean_exit_no_op", **extra, **ctx.proc()},
        )
        return
    dispatched_at = entry.get("dispatched_at")
    entry["status"] = "rework_requested"
    entry["dispatched_at"] = None
    yield from flush(draft)
    credit = yield CreditDeadWorker(number, dispatched_at=dispatched_at, pr_number=ctx.pr_number)
    acc.throttled_until = credit.throttled_until
    yield emit(
        "orphaned_worker_recovered",
        {
            **ctx.base(),
            "new_status": "rework_requested",
            "reason": recovered_reason,
            **extra,
            **ctx.proc(),
            "worker_death_at": facts.stamp,
            "failure_kind": credit.failure_kind,
        },
    )


def _head_changed(
    facts: SweepFacts,
    draft: Draft,
    acc: LockAcc,
    ctx: _Ctx,
    last_decision: str | None,
    *,
    carries_apply: bool,
) -> Flow:
    """Reviewed head != live head. ``carries_apply``: a completed outcome was queued
    for apply, so the review route must hold until its head is recorded applied."""
    entry = draft.work
    fingerprint = drift_fingerprint(
        reason="dead_worker_with_head_change",
        reviewed_head_sha=ctx.reviewed,
        live_head_sha=ctx.live,
    )
    if entry.get(NO_OP_DEFERRED_HEAD_KEY) == ctx.live:
        acc.no_op_routes.append(_no_op_route(ctx, entry))
        return
    if entry.get("orphan_drift_fingerprint") == fingerprint:
        return
    if facts.review_available:
        if carries_apply and entry.get("orphan_drift_at") is None:
            # A held route writes no drift of its own, so anchor the #654 backstop
            # here (as the completed-outcome route does): an apply that can never
            # land must end in the dead-dispatched reap, not hold the issue forever.
            entry["orphan_drift_at"] = facts.stamp
        acc.review_routes.append(
            ReviewRoute(
                ctx.number,
                ctx.pr_number,
                ctx.reviewed,
                ctx.live,
                fingerprint,
                "dead_worker_with_head_change",
                awaits_applied_head=carries_apply,
            )
        )
        return
    acc.no_op_routes.append(_no_op_route(ctx, entry))
    entry["orphan_drift_fingerprint"] = fingerprint
    entry["orphan_drift_at"] = facts.stamp
    yield emit(
        "orphaned_worker_drift",
        {
            **ctx.base(),
            "last_decision": last_decision,
            "reviewed_head_sha": ctx.reviewed,
            "live_head_sha": ctx.live,
            "reason": "dead_worker_with_head_change",
            **ctx.proc(),
        },
    )


def _advance_unreviewed(
    facts: SweepFacts, pre: PreOutcome, draft: Draft, acc: LockAcc, ctx: _Ctx
) -> Flow:
    """#1128: move an unverdicted PR's issue to pr-open. True when it advanced."""
    active = pre.unreviewed.get(ctx.number)
    if active is None:
        return False
    if not (yield AdvanceToPrOpen(ctx.number)):
        return False
    entry = draft.work
    entry["status"] = PASSIVE_OPEN_STATUS
    entry["dispatched_at"] = None
    entry["orphan_drift_fingerprint"] = None
    entry["orphan_drift_at"] = None
    yield from flush(draft)
    credit = yield CreditDeadWorker(ctx.number, classify_log=False)
    acc.throttled_until = credit.throttled_until
    yield emit(
        "orphaned_worker_advanced_to_pr_open",
        {
            **ctx.base(),
            "new_status": PASSIVE_OPEN_STATUS,
            "reason": "dead_worker_unsafe_to_auto_reset_open_unreviewed_pr",
            "removed_labels": sorted(active),
            **ctx.proc(),
            "label_write_ok": True,
            "worker_death_at": facts.stamp,
            "failure_kind": credit.failure_kind,
        },
    )
    return True


def with_pr_flow(
    facts: SweepFacts, pre: PreOutcome, draft: Draft, pr: Mapping[str, Any], acc: LockAcc
) -> Flow:
    number = draft.issue
    entry = draft.work
    pr_number = int(pr["number"])
    live = pre.heads.get(number)
    pr_state = ((facts.locked or {}).get("prs") or {}).get(str(pr_number)) or {}
    resolved = yield ReadReviewDecision(pr_number, live, "lock")
    terminal = yield ReadTerminal(number)
    ctx = _Ctx(
        number=number,
        pr=pr,
        pr_number=pr_number,
        reviewed=resolved.reviewed_head_sha,
        live=live,
        pid=entry.get("worker_pid"),
        exit_code=terminal.exit_code,
        duration=terminal.duration_seconds,
    )
    last_decision = resolved.decision

    # Issue #2826: a rework worker's ``pr_body``/``pr_comment`` is applied whenever it
    # left a fresh outcome, not only on the routes that happened to read one. Resolved
    # once here, ahead of the fan-out, so a branch added later cannot skip the apply.
    outcome = None
    if facts.config.watchdog.apply_rework_outcome_on_every_route:
        outcome = yield from _resolve_outcome(facts, draft, ctx)
        if outcome is not None:
            _queue_apply(facts, acc, ctx, outcome)

    # An issue can only be ``dispatched`` while its PR's verdict is ``approved`` if
    # that dispatch is a post-approval rework (check failure or conflict), so the
    # PR ``status`` is not consulted: carry-forward (#2135) can rewrite it back to
    # ``approved`` mid-rework, and enumerating writers would rot.
    if last_decision in ("request_changes", "approved") and ctx.reviewed and ctx.live:
        if ctx.reviewed != ctx.live:
            yield from _head_changed(
                facts, draft, acc, ctx, last_decision, carries_apply=outcome is not None
            )
        elif last_decision == "approved":
            yield from _same_head(
                facts,
                draft,
                acc,
                ctx,
                recovered_reason="dead_worker_with_approved_rework",
                extra={"decision": "approved", "pr_state_status": pr_state.get("status")},
            )
        else:
            yield from _same_head(
                facts,
                draft,
                acc,
                ctx,
                recovered_reason="dead_worker_with_request_changes",
                extra={},
            )
        return

    if last_decision is None or last_decision == "pending":
        if (yield from _advance_unreviewed(facts, pre, draft, acc, ctx)):
            return
    fingerprint = drift_fingerprint(
        reason="dead_worker_unsafe_to_auto_reset",
        last_decision=last_decision or "",
        pr_number=pr_number,
    )
    if entry.get("orphan_drift_fingerprint") != fingerprint:
        entry["orphan_drift_fingerprint"] = fingerprint
        entry["orphan_drift_at"] = facts.stamp
        yield emit(
            "orphaned_worker_drift",
            {
                **ctx.base(),
                "last_decision": last_decision,
                "reason": "dead_worker_unsafe_to_auto_reset",
                **ctx.proc(),
            },
        )
