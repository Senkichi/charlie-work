"""The dead-session lane's second half: what happens to the issue once its worker is reaped.

Two routes, chosen by whether the dead worker's issue still has an open PR:

* no open PR (issue #118): reclaim the issue. Park a labelless local session
  (#1971), salvage committed-but-unpublished work (#252, #1130), trip the
  cross-repo scope wire (#1244), then either escalate past the redispatch cap or
  relabel to ``ready`` (#165, #417). Every decision is ``decide_dead_sessions``.
* an open PR (issue #295, #315): route a rework session back to its owning lane.

Effects run in the order the lane always ran them; see ``dead_sessions`` for the
collaborators' lookup discipline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import state as state_mod
from ..cross_repo_gate import cross_repo_scope_gate
from ..dispatch_selection import _windowed_redispatch_at
from ..escalation import _escalate_issue, _escalation_edge
from ..github import label_names
from ..host import current as _host_current
from ..local_work_park import park_labelless_dead_local_session
from ..worker import WorkerView
from ..worktree import read_worker_outcome
from .decide_dead_sessions_plan import (
    NoPrGate,
    OpenPrRoute,
    Reclaim,
    no_pr_gate,
    open_pr_candidate_route,
    open_pr_route,
    plan_reclaim_commit,
    reclaim_route,
    scope_adjusted_kind,
    wants_publish_salvage,
)
from .effects_pr import _attempt_salvage
from .effects_rework import (
    _is_pre_review_rework_candidate,
    _reap_restore_rework_requested,
    _rework_pr_for_worker,
    _route_dead_worker_to_pre_review_rework,
)
from .effects_sessions import _emit_session_failed_relabeled

if TYPE_CHECKING:
    from .dead_sessions import SessionPass


def reclaim_or_route_dead_session(
    ctx: SessionPass,
    w: WorkerView,
    *,
    failure_kind: str | None,
    inspection: Any,
    is_completed: bool,
    worktree_path: Path,
) -> None:
    """Issue #118 reclaim when no PR is open; issue #295 rework routing when one is."""
    if reclaim_route(has_open_pr=w.issue_number in ctx.open_prs_by_issue) is Reclaim.NO_OPEN_PR:
        _reclaim_no_open_pr(
            ctx, w, failure_kind=failure_kind, inspection=inspection, worktree_path=worktree_path
        )
    else:
        _route_open_pr(ctx, w, failure_kind=failure_kind, is_completed=is_completed)


def _reclaim_no_open_pr(
    ctx: SessionPass,
    w: WorkerView,
    *,
    failure_kind: str | None,
    inspection: Any,
    worktree_path: Path,
) -> None:
    gh, config, write_gate, state_file = ctx.gh, ctx.config, ctx.write_gate, ctx.state_file
    try:
        issue = gh.issue_view(w.issue_number)
    except Exception:
        # Issue may have been deleted or we lack access.
        issue = None
    issue_labels = label_names(issue) if issue is not None else set()
    active_labels = issue_labels & config.labels.active
    # Gate the WHOLE reclaim on an active label actually being present, matching
    # reconcile.py's issue_active_label_no_open_pr pattern so all three sites agree.
    # An issue with no active label (e.g. one carrying only a terminal label like
    # agent:human-needed/agent:done) has nothing here to reclaim; it must never get
    # `ready` added back just because it also has a stale dispatched/no-PR entry.
    gate = no_pr_gate(issue_found=issue is not None, active_labels=active_labels)
    if gate is NoPrGate.SKIP:
        return  # skip relabel
    if gate is NoPrGate.PARK:
        # Issue #1971: a ``dispatched`` entry on a no-PR backend with commits is
        # finished work -- park, don't leave for the timed backstop reap.
        park_labelless_dead_local_session(
            gh=gh,
            config=config,
            repo_root=ctx.repo_root,
            issue_labels=issue_labels,
            worker=w,
            write_gate=write_gate,
        )
        return
    needs_ready = config.labels.ready not in issue_labels

    # Issue #252: completed-but-unpublished work takes the salvage path (push + PR)
    # instead of re-dispatching. Issue #1130: the trigger is ``ahead_count > 0``
    # (ahead, regardless of working-tree dirt): a worker that dies mid-push leaves a
    # committed-but-unpushed branch, and shim dirt can read as worker-authored. Salvage
    # pushes the branch ref, not the working tree, so the dirt is irrelevant to it.
    salvage_error: str | None = None
    has_salvageable_commits = inspection.ahead_count > 0
    if wants_publish_salvage(
        ahead_count=inspection.ahead_count, has_repo_root=ctx.repo_root is not None
    ):
        salvaged, salvage_error = _attempt_salvage(
            gh=gh,
            config=config,
            repo_root=ctx.repo_root,
            worktree_path=worktree_path,
            branch=w.branch,
            base_ref=inspection.resolved_base_ref or "",
            issue_number=w.issue_number,
            active_labels=active_labels,
            issue_labels=issue_labels,
            state_file=state_file,
            failure_kind=failure_kind,
            issue_title=issue.get("title") if issue else None,
            issue=issue,
            # cw#1771: ``worktree_path`` here is the worker's own recorded worktree
            # dir (no repo_root fallback), so reading the outcome from it is safe.
            worker_outcome=read_worker_outcome(worktree_path),
            write_gate=write_gate,
        )
        if salvaged:
            return
        # Salvage failed: fall through to the normal relabel path below.

    # Issue #1244: cross-repo scope tripwire. Before relabeling to ready for another
    # redispatch, check whether the issue's title names another managed repo. A dead
    # worker whose scope targets a sibling repo hopped to that repo's worktree;
    # redispatching repeats the hop forever, so override the failure kind and let the
    # deterministic-escalation check below escalate on the first occurrence.
    scope_result = cross_repo_scope_gate(
        str(issue.get("title") or ""),
        str(issue.get("body") or ""),
        ctx.dispatching_repo_name,
        ctx.fleet_repos,
    )
    failure_kind = scope_adjusted_kind(failure_kind, scope_passed=scope_result.passed)

    # Track the redispatch count for the escalation cap (issue #165); this
    # relabel-to-ready path is a redispatch event.
    with state_mod.state_lock(state_file):
        state = state_mod.load_state(state_file)
        entry = state["issues"].get(str(w.issue_number), {})
        plan = plan_reclaim_commit(
            _windowed_redispatch_at(
                entry, window_minutes=config.watchdog.redispatch_window_minutes
            ),
            failure_kind,
            now=_host_current().clock.now(),
            max_auto_redispatch=config.watchdog.max_auto_redispatch,
            active_labels=active_labels,
            ready_label_present=not needs_ready,
        )
        redispatch_at = list(plan.redispatch_at)
        if plan.escalate:
            # Escalate to human review instead of relabeling to ready.
            state = _escalate_issue(
                state,
                w.issue_number,
                reason=plan.reason,
                reason_class=plan.reason_class,
                issue_extra={"redispatch_at": redispatch_at},
            )
            # Issue #282: the PID is already verified dead here, but the liveness
            # fingerprint stays for the recovery probe to cross-check.
            write_gate.save_state(state)
            write_gate.transition(
                gh,
                config.labels,
                w.issue_number,
                _escalation_edge("redispatch_escalated", plan.reason_class),
            )
            state = write_gate.append_event(
                state,
                "session_failed_escalated",
                {
                    "issue_number": w.issue_number,
                    "failure_kind": failure_kind,
                    "removed_labels": sorted(active_labels),
                    "redispatch_count": len(redispatch_at),
                },
            )
            write_gate.save_state(state)
            return
        entry["redispatch_at"] = redispatch_at
        state["issues"][str(w.issue_number)] = entry
        write_gate.save_state(state)

    # Remove all active labels and ensure the ready label is present. Issue
    # #2226: the write routes through the WriteGate's canonical label seam so
    # the return to ``ready`` is recorded as a ``lifecycle_transition`` in
    # events.db bound to this repo; the conditional add preserves the
    # no-redundant-write contract (ready is only re-added when absent).
    # Issue #417: record the outcome instead
    # of discarding it -- a PARTIAL_FAILURE means this pass's label swap did
    # not fully land, and the orphan sweep's no-open-PR lane finishes the
    # reclaim later (it re-derives "does this still need fixing" from
    # GitHub's live labels and never touches ``redispatch_at``, so a retry
    # cannot double-count).
    label_result = write_gate.apply_issue_labels(
        gh,
        config.labels,
        w.issue_number,
        add=(config.labels.ready,) if needs_ready else (),
        remove=sorted(active_labels),
        to_state="ready",
        cause="session_failed_relabeled",
    )
    label_write_ok = label_result.ok
    # Record the relabel event.
    with state_mod.state_lock(state_file):
        state = state_mod.load_state(state_file)
        # Issue #282: the liveness fingerprint survives so the recovery path can verify
        # the worker is dead before removing the worktree.
        state = _emit_session_failed_relabeled(
            state,
            issue_number=w.issue_number,
            reason="dead_worker_no_open_pr",
            failure_kind=failure_kind,
            removed_labels=sorted(active_labels),
            added_ready=needs_ready,
            label_write_ok=label_write_ok,
            salvage_failed=has_salvageable_commits,
            salvage_error=salvage_error,
            state_path=state_file,
            write_gate=write_gate,
        )
        write_gate.save_state(state)


def _route_open_pr(
    ctx: SessionPass,
    w: WorkerView,
    *,
    failure_kind: str | None,
    is_completed: bool,
) -> None:
    """Issue #295: an open PR with request_changes (or a rework prompt) goes back to rework.

    Issue #315 finding 1: a completed worktree proves the worker finished, even if this
    pass's PR-list snapshot has not caught up to a fresh push, so it is never rolled
    back to ``rework_requested`` just because a classifier ran on its dead sidecar.
    """
    pr_data = _rework_pr_for_worker(ctx.open_prs_by_issue, w)
    route = open_pr_route(is_completed=is_completed, has_rework_pr=pr_data is not None)
    if route is OpenPrRoute.SKIP:
        return
    gh, config, state_file = ctx.gh, ctx.config, ctx.state_file

    def restore() -> None:
        _reap_restore_rework_requested(
            state_file,
            gh,
            config,
            ctx.open_prs_by_issue,
            w,
            failure_kind=failure_kind,
            repo_root=ctx.repo_root,
            write_gate=ctx.write_gate,
        )

    if route is OpenPrRoute.RESTORE:
        restore()
        return
    assert pr_data is not None
    try:
        pr_view = gh.pr_view(int(pr_data["number"]))
    except Exception:
        pr_view = None
    enriched = pr_view if pr_view else pr_data
    is_candidate, reason = _is_pre_review_rework_candidate(enriched, config, ctx.now_for_health)
    if open_pr_candidate_route(is_candidate=is_candidate) is OpenPrRoute.ROUTE_PRE_REVIEW:
        _route_dead_worker_to_pre_review_rework(
            state_file,
            gh,
            config,
            enriched,
            w.issue_number,
            reason,
            failure_kind=failure_kind,
            write_gate=ctx.write_gate,
        )
    else:
        restore()
