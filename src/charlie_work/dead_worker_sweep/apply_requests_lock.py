"""Request handlers for the lock and post phases, plus the ``serve`` dispatcher.

Lock-phase handlers read the state copy the shell loaded under the lock
(``ctx.state``); post-phase handlers run with no lock held and take
``state_lock`` themselves where they read persisted state.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import (
    blocked_worker_escalation,
    dead_worker_classification,
    orphaned_worker_no_op_drain,
    rework_outcome,
    worker_fate,
)
from ..process_utils import find_worker_terminal_status
from ..worktree import worktree_path_for_branch
from . import apply_requests_pre as pre_handlers
from .apply_context import SweepContext
from .decide_common import label_names
from .model import (
    CloseVerifiedNoChanges,
    CollectLiveHandoff,
    FetchOpenIssues,
    FetchOpenPrs,
    FetchPrView,
    ParkBackstop,
    ParkOrReclaim,
    ProbeCrossRepoScope,
    ProbeRemoteBranch,
    ProbeWorktreeHead,
    ProbeZeroArtifact,
    ReadClock,
    ReadReviewDecision,
    ResolveFate,
    SalvagePush,
    StripAndFlag,
    AdvanceToPrOpen,
    ApplyOutcomes,
    CreditDeadWorker,
    CreditResult,
    DrainNoOp,
    ExemptThrottleCleanExit,
    GuardedUpdate,
    OpenPrForBranch,
    PrOpenResult,
    ReadAppliedHeads,
    ReadBlockedOutcome,
    ReadCompletedOutcome,
    ReadTerminal,
    Review,
    ReviewResult,
    TerminalFacts,
)

logger = logging.getLogger(__name__)


def _parsed_dispatched_at(ctx: SweepContext, entry: dict[str, Any]) -> Any:
    return ctx.ports.parse_iso_timestamp(entry.get("dispatched_at"))


def open_pr_for_branch(ctx: SweepContext, req: OpenPrForBranch) -> PrOpenResult:
    number = req.issue
    issue = (ctx.issues or {}).get(number) or {}
    labels = label_names(issue)
    if req.source == "pushed_orphan":
        outcome = ctx.worker_outcomes.get(number)
    else:
        outcome = (ctx.live_candidates.get(number) or {}).get("worker_outcome")
    pr_number, pr_error, _closing = ctx.ports.open_pr_for_orphaned_branch(
        gh=ctx.gh,
        config=ctx.config,
        repo_root=ctx.repo_root if isinstance(ctx.repo_root, Path) else None,
        branch=req.branch,
        base_ref=ctx.config.dispatch.base_ref,
        issue_number=number,
        active_labels=labels & ctx.config.labels.active,
        issue_labels=labels,
        issue_title=issue.get("title"),
        state_file=ctx.state_file,
        worker_outcome=outcome,
    )
    return PrOpenResult(pr_number=pr_number, error=pr_error)


def advance_to_pr_open(ctx: SweepContext, req: AdvanceToPrOpen) -> bool:
    """#1128: strip the active labels and add ``pr-open`` for an unverdicted PR."""
    number = req.issue
    labels = label_names((ctx.issues or {}).get(number) or {})
    # Issue #2226: route through the WriteGate's canonical seam so the
    # PR-open transition lands in events.db bound to this repo; the
    # conditional add preserves the no-redundant-write contract.
    result = ctx.write_gate.apply_issue_labels(
        ctx.gh,
        ctx.config.labels,
        number,
        add=(ctx.config.labels.pr_open,) if ctx.config.labels.pr_open not in labels else (),
        remove=sorted(labels & ctx.config.labels.active),
        to_state="pr_open",
        cause="advance_to_pr_open",
    )
    return result.ok


def credit_dead_worker(ctx: SweepContext, req: CreditDeadWorker) -> CreditResult:
    kind = dead_worker_classification.classify_and_credit_dead_worker(
        ctx.issue_entry(req.issue),
        ctx.sessions_dir,
        req.issue,
        ctx.state,
        ctx.config,
        write_gate=ctx.write_gate,
        at=ctx.stamp,
        classify_log=req.classify_log,
        dispatched_at=req.dispatched_at,
        pr_number=req.pr_number,
    )
    return CreditResult(failure_kind=kind, throttled_until=ctx.state.get("throttled_until"))


def exempt_throttle_clean_exit(ctx: SweepContext, req: ExemptThrottleCleanExit) -> CreditResult:
    kind = dead_worker_classification.exempt_clean_exit_throttle_death(
        ctx.issue_entry(req.issue),
        ctx.sessions_dir,
        req.issue,
        ctx.state,
        ctx.config,
        write_gate=ctx.write_gate,
        dispatched_at=req.dispatched_at,
        pr_number=req.pr_number,
        now=ctx.now,
    )
    return CreditResult(failure_kind=kind, throttled_until=ctx.state.get("throttled_until"))


def read_terminal(ctx: SweepContext, req: ReadTerminal) -> TerminalFacts:
    entry = ctx.issue_entry(req.issue)
    record = worker_fate.fresh_terminal_record(
        find_worker_terminal_status(ctx.sessions_dir, req.issue),
        _parsed_dispatched_at(ctx, entry),
        issue_number=req.issue,
        on_fate=ctx.collector("swept"),
    )
    if not record:
        return TerminalFacts(exit_code=None, duration_seconds=None)
    return TerminalFacts(
        exit_code=record.get("exit_code"), duration_seconds=record.get("duration_seconds")
    )


def read_completed_outcome(ctx: SweepContext, req: ReadCompletedOutcome) -> dict[str, Any] | None:
    path = (
        worktree_path_for_branch(ctx.repo_root, req.branch, ctx.worktrees_dir)
        if ctx.repo_root is not None and ctx.worktrees_dir is not None
        else None
    )
    return rework_outcome.fresh_completed_worker_outcome(
        path,
        issue_number=req.issue,
        live_head_sha=req.live_head_sha,
        dispatched_at=_parsed_dispatched_at(ctx, ctx.issue_entry(req.issue)),
        pr_number=req.pr_number,
        on_fate=ctx.collector("swept"),
    )


def read_blocked_outcome(ctx: SweepContext, req: ReadBlockedOutcome) -> dict[str, Any] | None:
    return blocked_worker_escalation.dead_worker_blocked_outcome(
        pr_data=ctx.pr_by_issue[req.issue],
        entry=ctx.issue_entry(req.issue),
        issue_number=req.issue,
        sessions_dir=ctx.sessions_dir,
        repo_root=ctx.repo_root,
        worktrees_dir=ctx.worktrees_dir,
        on_fate=ctx.collector("swept"),
    )


def apply_outcomes(ctx: SweepContext, req: ApplyOutcomes) -> bool:
    rework_outcome.apply_collected_rework_outcomes(
        ctx.gh,
        outcome_apply_routes=list(req.routes),
        repo_root=ctx.repo_root,
        worktrees_dir=ctx.worktrees_dir,
        sessions_dir=ctx.sessions_dir,
        state_file=ctx.state_file,
        write_gate=ctx.write_gate,
    )
    return True


def read_applied_heads(ctx: SweepContext, _req: ReadAppliedHeads) -> dict[str, Any]:
    with ctx.ports.state_lock(ctx.state_file):
        state = ctx.ports.load_state(ctx.state_file)
    heads = state.get(rework_outcome.APPLIED_HEADS_KEY) or {}
    return dict(heads) if isinstance(heads, dict) else {}


def review(ctx: SweepContext, req: Review) -> ReviewResult:
    """Run the review callback outside any lock, then read the fresh disposition facts."""
    ctx.review_prs[req.issue] = req.pr_number
    callback = ctx.review_callback
    if callback is None:  # decide never yields Review without a callback; stay total
        return ReviewResult(
            False, False, False, False, False, "no review callback", None, None, None
        )
    try:
        outcome = callback(req.pr_number)
    except Exception as exc:
        logger.exception(
            "review_callback escaped for orphaned issue %s / pr %s", req.issue, req.pr_number
        )
        return ReviewResult(
            False, False, False, False, False, f"{type(exc).__name__}: {exc}", None, None, None
        )
    data = outcome.data
    with ctx.ports.state_lock(ctx.state_file):
        state = ctx.ports.load_state(ctx.state_file)
    pr_state = (state.get("prs") or {}).get(str(req.pr_number), {})
    entry = (state.get("issues") or {}).get(str(req.issue), {})
    entry = entry if isinstance(entry, dict) else {}
    return ReviewResult(
        ok=bool(outcome.ok),
        routed_to_rework=bool(data.get("routed_to_rework")),
        closed_unmerged_converged=bool(data.get("closed_unmerged_converged")),
        escalation_deferred_live_worker=bool(data.get("escalation_deferred_live_worker")),
        is_no_op_rework=bool(data.get("is_no_op_rework")),
        raised_error=None,
        entry_status=entry.get("status"),
        pr_reviewed_head_sha=pr_state.get("reviewed_head_sha"),
        entry_branch=entry.get("branch_name"),
    )


def guarded_update(ctx: SweepContext, req: GuardedUpdate) -> bool:
    """Re-check the guards on a fresh load under one short lock, then merge.

    Returns whether the merge was written: ``False`` means a concurrent writer
    changed the issue's status or the PR's reviewed head first, and the flow
    must not record anything that presumes the write (ADR-0001: merge, never
    replace the entry). ``req.event`` is appended in this same lock window."""
    with ctx.ports.state_lock(ctx.state_file):
        state = ctx.ports.load_state(ctx.state_file)
        entry = (state.get("issues") or {}).get(str(req.issue))
        if not isinstance(entry, dict):
            return False
        if req.require_status is not None and entry.get("status") != req.require_status:
            return False
        if req.require_pr_reviewed_head is not None:
            pr_number = ctx.review_prs.get(req.issue, entry.get("pr_number"))
            pr_state = (state.get("prs") or {}).get(str(pr_number), {})
            if pr_state.get("reviewed_head_sha") != req.require_pr_reviewed_head:
                return False
        entry.update(copy.deepcopy(dict(req.set_items)))
        if req.stamp_fields:
            stamp = ctx.ports.utc_now()
            for key in req.stamp_fields:
                entry[key] = stamp
        if req.event is not None:
            kind, payload = req.event
            # An append_event failure propagates before save_state: status and
            # event are both absent, never status without its audit row (#2113).
            state = ctx.write_gate.append_event(
                state,
                # event-consumer: audit-only -- pass-through of the kind ``decide*`` chose
                kind,
                dict(payload),
                ctx.config.runtime.event_ring_size,
            )
        ctx.write_gate.save_state(state)
    return True


def drain_no_op(ctx: SweepContext, req: DrainNoOp) -> bool:
    routes = [
        orphaned_worker_no_op_drain.NoOpReworkRoute(
            issue_number=r.issue_number,
            pr_number=r.pr_number,
            live_head_sha=r.live_head_sha,
            reason=r.reason,
            branch=r.branch,
        )
        for r in req.routes
    ]
    orphaned_worker_no_op_drain.drain_no_op_rework_routes(
        routes,
        gh=ctx.gh,
        config=ctx.config,
        state_file=ctx.state_file,
        write_gate=ctx.write_gate,
        sessions_dir=ctx.sessions_dir,
        repo_root=ctx.repo_root,
        worktrees_dir=ctx.worktrees_dir,
        review_callback=ctx.review_callback,
        record_review_callback=ctx.record_review_callback,
        enrich_checks_callback=ctx.enrich_checks_callback,
    )
    return True


HANDLERS: dict[type, Callable[[SweepContext, Any], Any]] = {
    ReadClock: pre_handlers.read_clock,
    CollectLiveHandoff: pre_handlers.collect_live_handoff,
    FetchOpenPrs: pre_handlers.fetch_open_prs,
    FetchOpenIssues: pre_handlers.fetch_open_issues,
    ResolveFate: pre_handlers.resolve_fate,
    StripAndFlag: pre_handlers.strip_and_flag,
    CloseVerifiedNoChanges: pre_handlers.close_verified_no_changes,
    ProbeZeroArtifact: pre_handlers.probe_zero_artifact,
    ProbeCrossRepoScope: pre_handlers.probe_cross_repo_scope,
    ParkOrReclaim: pre_handlers.park_or_reclaim,
    ParkBackstop: pre_handlers.park_backstop,
    SalvagePush: pre_handlers.salvage_push,
    ProbeRemoteBranch: pre_handlers.probe_remote_branch,
    ProbeWorktreeHead: pre_handlers.probe_worktree_head,
    ReadReviewDecision: pre_handlers.read_review_decision,
    FetchPrView: pre_handlers.fetch_pr_view,
    OpenPrForBranch: open_pr_for_branch,
    AdvanceToPrOpen: advance_to_pr_open,
    CreditDeadWorker: credit_dead_worker,
    ExemptThrottleCleanExit: exempt_throttle_clean_exit,
    ReadTerminal: read_terminal,
    ReadCompletedOutcome: read_completed_outcome,
    ReadBlockedOutcome: read_blocked_outcome,
    ApplyOutcomes: apply_outcomes,
    ReadAppliedHeads: read_applied_heads,
    Review: review,
    GuardedUpdate: guarded_update,
    DrainNoOp: drain_no_op,
}


def serve(ctx: SweepContext, request: Any) -> Any:
    """Run one request against the world and return its result value."""
    return HANDLERS[type(request)](ctx, request)
