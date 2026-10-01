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


def _completed_outcome(
    facts: SweepFacts, draft: Draft, acc: LockAcc, ctx: _Ctx, extra: Mapping[str, Any]
) -> Flow:
    """True when a fresh, on-target completed outcome exists (and was routed)."""
    entry = draft.work
    if not ctx.live:
        return False
    branch = ctx.pr.get("headRefName") or entry.get("branch_name")
    if not branch or not facts.repo.repo_root_is_path or not facts.repo.has_worktrees:
        return False
    outcome = yield ReadCompletedOutcome(ctx.number, ctx.pr_number, ctx.live, branch)
    if outcome is None:
        return False
    applied = (facts.locked or {}).get(APPLIED_HEADS_KEY) or {}
    if applied.get(str(ctx.number)) != outcome["head_sha"]:
        acc.outcome_apply_routes.append((ctx.number, ctx.pr_number))
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
    entry["status"] = "rework_requested"
    entry["dispatched_at"] = None
    yield from flush(draft)
    credit = yield CreditDeadWorker(number)
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
    facts: SweepFacts, draft: Draft, acc: LockAcc, ctx: _Ctx, last_decision: str | None
) -> Flow:
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
        acc.review_routes.append(
            ReviewRoute(
                ctx.number,
                ctx.pr_number,
                ctx.reviewed,
                ctx.live,
                fingerprint,
                "dead_worker_with_head_change",
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

    # An issue can only be ``dispatched`` while its PR's verdict is ``approved`` if
    # that dispatch is a post-approval rework (check failure or conflict), so the
    # PR ``status`` is not consulted: carry-forward (#2135) can rewrite it back to
    # ``approved`` mid-rework, and enumerating writers would rot.
    if last_decision in ("request_changes", "approved") and ctx.reviewed and ctx.live:
        if ctx.reviewed != ctx.live:
            yield from _head_changed(facts, draft, acc, ctx, last_decision)
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
