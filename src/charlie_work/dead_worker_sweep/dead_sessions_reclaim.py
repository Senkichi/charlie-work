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

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import state as state_mod
from ..cross_repo_gate import cross_repo_scope_gate
from ..dead_worker_reap import (
    _attempt_salvage,
    _emit_session_failed_relabeled,
    _is_pre_review_rework_candidate,
    _reap_restore_rework_requested,
    _rework_pr_for_worker,
    _route_dead_worker_to_pre_review_rework,
)
from ..dispatch_selection import _windowed_redispatch_at
from ..escalation import _escalate_issue, _escalation_edge
from ..github import label_names
from ..local_work_park import park_labelless_dead_local_session
from ..worker import WorkerView
from ..worktree import read_worker_outcome
from .decide_dead_sessions import redispatch_verdict

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
    if w.issue_number not in ctx.open_prs_by_issue:
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
        # Issue may have been deleted or we lack access; skip relabel.
        return
    issue_labels = label_names(issue)
    active_labels = issue_labels & config.labels.active
    # Gate the WHOLE reclaim on an active label actually being present, matching
    # reconcile.py's issue_active_label_no_open_pr pattern so all three sites agree.
    # An issue with no active label (e.g. one carrying only a terminal label like
    # agent:human-needed/agent:done) has nothing here to reclaim; it must never get
    # `ready` added back just because it also has a stale dispatched/no-PR entry.
    if not active_labels:
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
    if has_salvageable_commits and ctx.repo_root is not None:
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
    if not scope_result.passed:
        failure_kind = "cross_repo_hop"

    # Track the redispatch count for the escalation cap (issue #165); this
    # relabel-to-ready path is a redispatch event.
    with state_mod.state_lock(state_file):
        state = state_mod.load_state(state_file)
        entry = state["issues"].get(str(w.issue_number), {})
        now = datetime.now(UTC)
        verdict = redispatch_verdict(
            _windowed_redispatch_at(
                entry, window_minutes=config.watchdog.redispatch_window_minutes
            ),
            failure_kind,
            now=now,
            max_auto_redispatch=config.watchdog.max_auto_redispatch,
        )
        redispatch_at = list(verdict.redispatch_at)
        if verdict.escalate:
            # Escalate to human review instead of relabeling to ready.
            state = _escalate_issue(
                state,
                w.issue_number,
                reason=verdict.reason,
                reason_class=verdict.reason_class,
                issue_extra={"redispatch_at": redispatch_at},
            )
            # Issue #282: the PID is already verified dead here, but the liveness
            # fingerprint stays for the recovery probe to cross-check.
            write_gate.save_state(state)
            write_gate.transition(
                gh,
                config.labels,
                w.issue_number,
                _escalation_edge("redispatch_escalated", verdict.reason_class),
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

    # Remove all active labels and ensure the ready label is present. Issue #417:
    # record the bool returns instead of discarding them -- a False means this pass's
    # label swap did not fully land, and the orphan sweep's no-open-PR lane finishes
    # the reclaim later (it re-derives "does this still need fixing" from GitHub's live
    # labels and never touches ``redispatch_at``, so a retry cannot double-count).
    label_write_ok = True
    for label in sorted(active_labels):
        if not gh.remove_issue_label(w.issue_number, label):
            label_write_ok = False
    if needs_ready:
        if not gh.add_issue_label(w.issue_number, config.labels.ready):
            label_write_ok = False
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
    if is_completed:
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

    pr_data = _rework_pr_for_worker(ctx.open_prs_by_issue, w)
    if pr_data is None:
        restore()
        return
    try:
        pr_view = gh.pr_view(int(pr_data["number"]))
    except Exception:
        pr_view = None
    enriched = pr_view if pr_view else pr_data
    is_candidate, reason = _is_pre_review_rework_candidate(enriched, config, ctx.now_for_health)
    if is_candidate:
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
