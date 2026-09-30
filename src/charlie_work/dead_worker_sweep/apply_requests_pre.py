"""Request handlers for the pre-lock phase: network reads, fate resolution, parking.

Every handler takes the :class:`SweepContext` and one frozen request from
``model.py`` and returns that request's result value. Module functions are called
through their modules (``module.func(...)``), never ``from`` -imported, so a suite
patch on the source module stays live exactly as it did against the original sweep.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import (
    blocked_worker_escalation,
    escalation,
    fleet_registry,
    local_work_park,
    no_pr_orphan_fate,
    worker_fate,
)
from . import effects_pr, effects_sessions, live_handoff
from ..cross_repo_gate import cross_repo_scope_gate
from ..github import build_branch_issue_validator
from ..process_utils import find_worker_terminal_status
from ..review_decision import review_decision
from ..worktree import read_worker_outcome, worktree_path_for_branch
from .apply_context import SweepContext
from .decide_common import label_names
from .model import (
    BackstopResult,
    CollectLiveHandoff,
    FateResult,
    FetchOpenIssues,
    FetchOpenPrs,
    FetchPrView,
    IssuesResult,
    LabelWrite,
    LiveHandoffFound,
    ParkBackstop,
    ParkOrReclaim,
    ProbeCrossRepoScope,
    ProbeRemoteBranch,
    ProbeWorktreeHead,
    ProbeZeroArtifact,
    ReadClock,
    ReadReviewDecision,
    RemoteProbe,
    ResolveFate,
    ReviewDecisionFacts,
    SalvagePush,
    SalvagePushResult,
    ScopeResult,
    StripAndFlag,
)


def _worktree(ctx: SweepContext, branch: str) -> Path | None:
    if ctx.repo_root is None or ctx.worktrees_dir is None:
        return None
    return worktree_path_for_branch(ctx.repo_root, branch, ctx.worktrees_dir)


def _issue(ctx: SweepContext, number: int) -> dict[str, Any]:
    return (ctx.issues or {}).get(number) or {}


def read_clock(ctx: SweepContext, _req: ReadClock) -> datetime:
    ctx.now = datetime.now(UTC)
    return ctx.now


def collect_live_handoff(ctx: SweepContext, _req: CollectLiveHandoff) -> LiveHandoffFound:
    live_entries = {
        int(key): entry
        for key, entry in (ctx.state.get("issues") or {}).items()
        if isinstance(entry, dict)
        and entry.get("status") == "dispatched"
        and ctx.pid_alive.get(int(key)) is True
    }
    candidates = live_handoff.collect_stale_live_handoff_pids(
        live_entries,
        worker_outcome_finalize_minutes=ctx.config.watchdog.worker_outcome_finalize_minutes,
        repo_root=ctx.repo_root,
        worktrees_dir=ctx.worktrees_dir,
        now=ctx.now,
        sessions_dir=ctx.sessions_dir,
        on_fate=ctx.collector("live_handoff"),
    )
    ctx.live_candidates = candidates
    return LiveHandoffFound(candidates=candidates)


def fetch_open_prs(ctx: SweepContext, _req: FetchOpenPrs) -> dict[int, dict[str, Any]]:
    validator = build_branch_issue_validator(ctx.gh)
    for pr in ctx.gh.pr_list():
        linked = ctx.ports.linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=ctx.config.dispatch.branch_prefix,
            branch_issue_validator=validator,
        )
        if linked is not None:
            ctx.pr_by_issue[linked] = pr
    return copy.deepcopy(ctx.pr_by_issue)


def fetch_open_issues(ctx: SweepContext, _req: FetchOpenIssues) -> IssuesResult:
    if ctx.issues is None:
        found: dict[int, dict[str, Any]] = {}
        for issue in ctx.gh.issue_list(state="open"):
            number = issue.get("number")
            if number is not None:
                found[int(number)] = issue
        ctx.issues = found
    return IssuesResult(issues_by_number=copy.deepcopy(ctx.issues))


def fetch_pr_view(ctx: SweepContext, req: FetchPrView) -> Any:
    try:
        view = ctx.gh.pr_view(req.pr_number)
    except Exception:
        view = None
    ctx.pr_views[req.pr_number] = view
    return copy.deepcopy(view)


def resolve_fate(ctx: SweepContext, req: ResolveFate) -> FateResult:
    number = req.issue
    entry = ctx.issue_entry(number)
    if req.stage == "pushed":
        pushed = no_pr_orphan_fate.resolve_pushed_orphan_fate(
            issue_number=number,
            entry=entry,
            precompute=ctx.fates[number],
            remote_head_sha=req.remote_head_sha,
            ahead_count=req.ahead_count,
            now=ctx.now,
            sessions_dir=ctx.sessions_dir,
        )
        worker_fate.collect_fate(ctx.buckets["no_pr"], pushed)
        return FateResult(
            branch=ctx.branches[number],
            kind=type(pushed).__name__,
            blocked_escalatable=False,
            pushed_without_pr=isinstance(pushed, worker_fate.PushedWithoutPr),
            worker_outcome=ctx.worker_outcomes.get(number),
        )

    issue = _issue(ctx, number)
    branch = entry.get("branch_name")
    if not branch:
        slug = ctx.ports.slugify(str(issue.get("title") or "work"))
        branch = f"{ctx.config.dispatch.branch_prefix}-{number}-{slug}"
    ctx.branches[number] = branch
    worktree_path = _worktree(ctx, branch)
    fate = no_pr_orphan_fate.resolve_no_pr_orphan_fate(
        issue_number=number,
        entry=entry,
        terminal=find_worker_terminal_status(ctx.sessions_dir, number),
        worktree_path=worktree_path,
        worktree_outcome_raw=(
            read_worker_outcome(worktree_path) if worktree_path is not None else None
        ),
        now=ctx.now,
    )
    ctx.fates[number] = fate
    resolved = fate.basis.outcome
    outcome = dict(resolved.raw) if resolved is not None else None
    ctx.worker_outcomes[number] = outcome
    worker_fate.collect_fate(ctx.buckets["no_pr"], fate)
    escalatable = (
        isinstance(fate, worker_fate.Blocked)
        and blocked_worker_escalation.escalatable_blocked_outcome(
            outcome, sessions_dir=ctx.sessions_dir, issue_number=number
        )
        is not None
    )
    return FateResult(
        branch=branch,
        kind=type(fate).__name__,
        blocked_escalatable=escalatable,
        blocked_reason_kind=str((outcome or {}).get("reason_kind") or "unknown"),
        blocked_detail=str((outcome or {}).get("detail") or ""),
        throttled=worker_fate.throttle_failure(fate) is not None,
        worker_outcome=copy.deepcopy(outcome),
    )


def strip_and_flag(ctx: SweepContext, req: StripAndFlag) -> LabelWrite:
    issue_labels = label_names(_issue(ctx, req.issue))
    active = issue_labels & ctx.config.labels.active
    ok = escalation._strip_active_and_flag_human_needed(
        ctx.gh, ctx.config, req.issue, active, issue_labels
    )
    ctx.escalations[req.kind][req.issue] = {"label_write_ok": ok}
    return LabelWrite(ok=bool(ok), removed_labels=tuple(sorted(active)))


def probe_zero_artifact(ctx: SweepContext, req: ProbeZeroArtifact) -> bool:
    return bool(effects_sessions._is_zero_artifact_dispatch_loop(ctx.sessions_dir, req.issue))


def probe_cross_repo_scope(ctx: SweepContext, req: ProbeCrossRepoScope) -> ScopeResult:
    if ctx.scope_context is None:
        repo_name = (
            effects_pr._dispatching_repo_name(ctx.gh, ctx.repo_root)
            if ctx.repo_root is not None
            else ""
        )
        ctx.scope_context = (
            frozenset(fleet_registry.managed_repo_names(ctx.fleet_dir_override)),
            repo_name,
        )
    fleet_repos, repo_name = ctx.scope_context
    issue = _issue(ctx, req.issue)
    verdict = cross_repo_scope_gate(
        str(issue.get("title") or ""), str(issue.get("body") or ""), repo_name, fleet_repos
    )
    return ScopeResult(passed=bool(verdict.passed), reason=verdict.reason)


def park_or_reclaim(ctx: SweepContext, req: ParkOrReclaim) -> bool:
    number = req.issue
    issue = _issue(ctx, number)
    labels = label_names(issue)
    return bool(
        local_work_park.park_or_reclaim_local_orphan(
            gh=ctx.gh,
            config=ctx.config,
            repo_root=ctx.repo_root,
            worktrees_dir=ctx.worktrees_dir,
            state=ctx.state,
            issue_number=number,
            issue=issue,
            active_labels=labels & ctx.config.labels.active,
            issue_labels=labels,
            state_file=ctx.state_file,
            worker_outcome=ctx.worker_outcomes.get(number),
            write_gate=ctx.write_gate,
            reclaim_results=ctx.reclaim_results,
            park_verdicts=ctx.park_verdicts,
            now=ctx.now,
        )
    )


def park_backstop(ctx: SweepContext, req: ParkBackstop) -> BackstopResult:
    deferred = local_work_park.park_backstop_due_local_orphans(
        gh=ctx.gh,
        config=ctx.config,
        repo_root=ctx.repo_root,
        worktrees_dir=ctx.worktrees_dir,
        state=ctx.state,
        state_file=ctx.state_file,
        no_pr_orphans=list(req.orphans),
        issues_by_number=ctx.issues or {},
        worker_outcomes=ctx.worker_outcomes,
        reclaim_results=ctx.reclaim_results,
        escalations=(
            ctx.escalations["worker_declared_blocked"],
            ctx.escalations["zero_artifact"],
            ctx.escalations["cross_repo_scope"],
        ),
        park_verdicts=ctx.park_verdicts,
        dead_dispatched_reap_minutes=ctx.config.watchdog.dead_dispatched_reap_minutes,
        now=ctx.now,
        write_gate=ctx.write_gate,
    )
    ctx.local_park_deferred = dict(deferred or {})
    return BackstopResult(
        deferred=dict(ctx.local_park_deferred),
        reclaim_results=copy.deepcopy(ctx.reclaim_results),
    )


def salvage_push(ctx: SweepContext, req: SalvagePush) -> SalvagePushResult:
    result = ctx.ports.salvage_push_stranded_commits(
        ctx.repo_root,
        req.branch,
        _worktree(ctx, req.branch),
        base_ref=ctx.config.dispatch.base_ref,
        dry_run=ctx.write_gate.dry_run,
    )
    return SalvagePushResult(
        pushed=bool(result.pushed),
        old_remote_sha=result.old_remote_sha,
        new_remote_sha=result.new_remote_sha,
        commit_count=result.commit_count,
        skip_reason=result.skip_reason,
        error=result.error,
    )


def probe_remote_branch(ctx: SweepContext, req: ProbeRemoteBranch) -> RemoteProbe:
    ahead_count, ahead_error = ctx.ports.remote_branch_ahead_count(
        ctx.repo_root, req.branch, ctx.config.dispatch.base_ref
    )
    head = ctx.ports.remote_branch_head_sha(ctx.repo_root, req.branch)
    return RemoteProbe(ahead_count=ahead_count, ahead_error=ahead_error, head_sha=head)


def probe_worktree_head(ctx: SweepContext, req: ProbeWorktreeHead) -> str | None:
    path = _worktree(ctx, req.branch)
    return ctx.ports.worktree_head_sha(path) if path is not None else None


def read_review_decision(ctx: SweepContext, req: ReadReviewDecision) -> ReviewDecisionFacts:
    resolved = review_decision(
        ctx.state_file.parent / "prs" / f"pr-{req.pr_number}", None, req.head_sha
    )
    return ReviewDecisionFacts(
        decision=resolved.decision,
        stale=bool(resolved.stale),
        missing=bool(resolved.missing),
        reviewed_head_sha=resolved.reviewed_head_sha,
    )
