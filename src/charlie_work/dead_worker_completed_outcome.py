"""Completed-outcome recovery for the orphaned-worker sweep (#1911).

Extracted from ``orphaned_worker_sweep`` during the issue #1971 rework: the
merge of the #1993 provider-throttle rearm pushed the sweep module past the
repo's 800-line cap, and this function -- the #1911 completed-outcome
recovery for a dead dispatched worker that still has an open PR -- is a
self-contained concern the sweep calls into. Verbatim move; the sweep's
``handle_dead_worker_with_pr`` calls it at both of its original sites.

Workflow-module names the moved code resolved through ``workflow``'s module
namespace (``utc_now``, ``_parse_iso_timestamp``) are reached through a
function-local ``import charlie_work.workflow as _wf`` -- a top-level import
would cycle (``workflow`` imports this module transitively), and the
module-object seam keeps suite patches on ``charlie_work.workflow.<name>``
interceptable. Every other free name is imported directly from its defining
module.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .orphaned_worker_review_drain import OrphanedWorkerReviewRoute
from .rework_outcome import (
    APPLIED_HEADS_KEY,
    fresh_completed_worker_outcome,
)
from .worktree import worktree_path_for_branch


def handle_dead_worker_completed_outcome(
    *,
    state: dict[str, Any],
    sweep_events: list[tuple[str, dict[str, Any]]],
    entry: dict[str, Any],
    issue_number: int,
    pr_number: int,
    pr_data: dict[str, Any],
    reviewed_head_sha: str | None,
    live_head_sha: str | None,
    terminal_pid: Any,
    terminal_exit_code: int | None,
    terminal_duration_seconds: Any,
    repo_root: Any,
    worktrees_dir: Path | None,
    outcome_apply_routes: list[tuple[int, int]],
    review_routes: list[OrphanedWorkerReviewRoute],
    review_callback: Callable[[int], Any] | None,
    drift_fingerprint: Callable[..., str],
    extra_payload: dict[str, Any] | None = None,
) -> bool:
    """Recover a dead worker that provably completed its handoff (#1911).

    Returns ``True`` only when there is no terminal record
    (``terminal_exit_code is None``) AND the worktree holds a fresh,
    on-target ``.worker-outcome.json`` -- written after this dispatch's
    ``dispatched_at`` (a previous session's leftover does not count),
    reporting ``push_succeeded``, and pinning ``head_sha`` to the live
    head. In that case the outcome is queued for the post-lock #1877
    apply pass (idempotent: ``APPLIED_HEADS_KEY`` dedups on the reported
    head, so a transient failure retries next pass while an
    already-applied outcome is never re-queued) and -- when a review
    callback is available -- the issue is also queued for the post-lock
    ``review_routes`` drain (issue #1915): the drain re-checks that the
    outcome applied, then calls ``review()`` and flips
    ``dispatched`` -> ``reviewing`` on a fresh packet or, when review
    cannot produce one, returns the issue to ``rework_requested`` -- still
    without a ``worker_death_at`` credit. False-crediting a worker death
    here while the issue sat ``dispatched`` forever was the two-step
    failure that drove the swole#198 0-commit
    ``no_op_rework_attempts_cap_exceeded`` loop. The
    drift fingerprint is deliberately NOT marked on the routed path: the
    drain is what resolves the finding, so an unapplied/skipped route must
    re-collect cleanly on the next pass. Without a review callback the
    finding surfaces once via the same fingerprinted drift the clean-exit
    (#773) branch uses, and ``orphan_drift_at`` arms the #654 time-based
    reap backstop either way so a route that never resolves still
    converges.

    Every negative answer (a recorded exit code -- zero handled by the
    caller's own branch, non-zero being a confirmed crash -- a missing
    branch/worktree/timestamp, a stale or off-target outcome) returns
    ``False`` and leaves the caller's worker-death path untouched.
    """

    # Deferred: workflow.py imports this module top-level, so a top-level
    # import here would cycle. Attribute access through the module object
    # also keeps suite patches on ``charlie_work.workflow.<name>`` live.
    import charlie_work.workflow as _wf

    if terminal_exit_code is not None or not live_head_sha:
        return False
    branch = pr_data.get("headRefName") or entry.get("branch_name")
    if not branch or not isinstance(repo_root, Path) or worktrees_dir is None:
        return False
    outcome = fresh_completed_worker_outcome(
        worktree_path_for_branch(repo_root, branch, worktrees_dir),
        live_head_sha=live_head_sha,
        dispatched_at=_wf._parse_iso_timestamp(entry.get("dispatched_at")),
    )
    if outcome is None:
        return False
    outcome_head_sha = outcome.get("head_sha")
    applied_heads = state.get(APPLIED_HEADS_KEY, {})
    if not (
        isinstance(applied_heads, dict)
        and applied_heads.get(str(issue_number)) == outcome_head_sha
    ):
        outcome_apply_routes.append((issue_number, pr_number))
    fingerprint = drift_fingerprint(
        reason="dead_worker_completed_outcome",
        reviewed_head_sha=reviewed_head_sha,
    )
    if entry.get("orphan_drift_fingerprint") == fingerprint:
        return True
    if review_callback is not None:
        # Issue #1915: applying the outcome is only half the recovery -- the
        # issue must also leave ``dispatched`` or it dead-ends until the
        # 60-minute ``dead_dispatched_worker_reap`` backstop escalates it
        # (swole#198/PR#348). Queue a review route like the head-advanced
        # branch below: post-lock, once the outcome-apply drain has run,
        # ``review()`` either produces a fresh packet (the drain flips the
        # issue to ``reviewing``) or cannot (the drain returns it to
        # ``rework_requested``, still without a death credit, so the
        # ordinary dispatch loop owns the still-outstanding rework). The
        # drift fingerprint is deliberately NOT marked here -- the drain is
        # what resolves the finding, so a route whose apply has not landed
        # yet must re-collect cleanly next pass -- but ``orphan_drift_at``
        # still arms so the #654 backstop stays the terminal for a route
        # that never resolves (e.g. an apply that can never succeed).
        if entry.get("orphan_drift_at") is None:
            entry["orphan_drift_at"] = _wf.utc_now()
        review_routes.append(
            OrphanedWorkerReviewRoute(
                issue_number=issue_number,
                pr_number=pr_number,
                reviewed_head_sha=reviewed_head_sha,
                live_head_sha=live_head_sha,
                fingerprint=fingerprint,
                reason="dead_worker_completed_outcome",
            )
        )
        return True
    entry["orphan_drift_fingerprint"] = fingerprint
    entry["orphan_drift_at"] = _wf.utc_now()
    sweep_events.append(
        (
            "orphaned_worker_drift",
            {
                "issue_number": issue_number,
                "pr_number": pr_number,
                "previous_status": "dispatched",
                "reason": "dead_worker_completed_outcome",
                "pid": terminal_pid,
                "exit_code": terminal_exit_code,
                "duration_seconds": terminal_duration_seconds,
                "worker_outcome_head_sha": outcome_head_sha,
                **(extra_payload or {}),
            },
        )
    )
    return True
