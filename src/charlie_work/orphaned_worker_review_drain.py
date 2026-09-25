"""Post-lock review-route drain for the orphaned-worker sweep.

Extracted from ``workflow._detect_and_handle_orphaned_workers`` during the
issue #1915 rework: ``workflow.py`` sits over its file-size ratchet mark,
and this drain was the growth the reviewer asked to move. The sweep's
in-lock classification (``orphaned_worker_sweep``) collects
:class:`OrphanedWorkerReviewRoute` findings; once the outcome-apply drain
has run, this module drives each route through ``review()`` outside the
state lock and resolves the issue's disposition:

* a fresh packet flips the still-``dispatched`` issue to ``reviewing``;
* a completed-outcome route whose ``review()`` cannot produce a packet
  returns the issue to ``rework_requested`` (issue #1915) -- still without
  a ``worker_death_at`` credit, the #1911 invariant -- so the ordinary
  dispatch loop owns the still-outstanding rework, bounded by
  ``dispatch_rework``'s ``redispatch_at`` accounting;
* any other failed review marks the finding's drift fingerprint so the
  identical finding is not re-emitted every pass.

Workflow-module names the moved code resolved through ``workflow``'s module
namespace (``state_lock``, ``load_state``, ``utc_now``, ``log_event``,
``TransitionOutcome``) are reached through a function-local
``import charlie_work.workflow as _wf`` -- a top-level import would cycle
(``workflow`` imports this module), and the module-object seam keeps suite
patches on ``charlie_work.workflow.<name>`` interceptable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from .rework_outcome import APPLIED_HEADS_KEY

if TYPE_CHECKING:
    from .config import OrchestratorConfig
    from .github import GitHubLike
    from .write_gate import WriteGate

logger = logging.getLogger(__name__)


class OrphanedWorkerReviewRoute(NamedTuple):
    """One post-lock ``review()`` routing collected by the orphan sweep.

    ``reviewed_head_sha`` is the head the recorded verdict was taken
    against (None when unverdicted); ``live_head_sha`` is the PR's live
    head at collection time; ``fingerprint`` is the finding's drift
    fingerprint (marked only when the drain resolves the route to drift);
    ``reason`` is the sweep branch's reason string -- the drain uses it
    for event attribution and for the completed-outcome (#1915) apply
    gate / ``rework_requested`` fallback.
    """

    issue_number: int
    pr_number: int
    reviewed_head_sha: str | None
    live_head_sha: str
    fingerprint: str
    reason: str


def drain_orphaned_worker_review_routes(
    review_routes: Sequence[OrphanedWorkerReviewRoute],
    *,
    review_callback: Callable[[int], Any] | None,
    gh: GitHubLike,
    config: OrchestratorConfig,
    state_file: Path,
    write_gate: WriteGate,
) -> None:
    """Drain collected orphan-sweep review routes through ``review()``.

    Runs after ``rework_outcome.apply_collected_rework_outcomes`` so a
    completed-outcome route reviews the PR only once the worker's edits
    have demonstrably landed. Every disposition shares one guard: a route
    is only resolved while the recorded verdict is unchanged
    (``decision_unchanged``) and the issue is still ``dispatched`` -- any
    concurrent transition wins and the route simply re-collects next pass
    (the finding's drift fingerprint stays unmarked on every deferral).
    """
    import charlie_work.workflow as _wf

    # Issue #1915: completed-outcome issues whose post-apply review() could
    # not produce a fresh packet are returned to ``rework_requested`` in the
    # drain below; the matching label edge is applied post-loop (transition()
    # is network I/O and never runs under the state lock).
    rework_requested_recoveries: list[int] = []

    # Route head-advanced request_changes findings -- and, since issue
    # #1915, completed-outcome findings whose apply pass landed -- to the
    # review lane outside the state lock. review() generates the packet,
    # fires the review_started label transition, and returns ok when a
    # fresh packet is produced. We then flip the issue status to
    # "reviewing" so it is not re-detected as an orphan on every
    # subsequent pass. If review() fails, we record a drift fingerprint
    # so the identical finding is not re-emitted every pass -- except on
    # the completed-outcome route, where the issue instead returns to
    # ``rework_requested`` so the ordinary dispatch loop owns the
    # still-outstanding rework.
    for route in review_routes:
        issue_number = route.issue_number
        pr_number = route.pr_number
        reviewed_head_sha_before = route.reviewed_head_sha
        live_head_sha = route.live_head_sha
        fingerprint = route.fingerprint
        reason = route.reason
        if review_callback is None:
            continue
        if reason == "dead_worker_completed_outcome":
            # Issue #1915: a completed-outcome route only reviews once its
            # outcome demonstrably applied -- the outcome-apply drain ran
            # just above, and review() builds its packet from the live PR
            # body, so a still-pending apply (transient edit failure or a
            # live-head move) must defer the review rather than flip the
            # issue out of "dispatched" before the worker's edits land.
            # The finding's drift fingerprint stayed unmarked in the
            # sweep, so a skipped route is re-collected next pass.
            with _wf.state_lock(state_file):
                _route_state = _wf.load_state(state_file)
                _applied_heads = _route_state.get(APPLIED_HEADS_KEY, {})
            if not (
                isinstance(_applied_heads, dict)
                and _applied_heads.get(str(issue_number)) == live_head_sha
            ):
                continue
        try:
            review_result = review_callback(pr_number)
        except Exception as exc:
            # Per-route guard mirroring apply_collected_rework_outcomes: one
            # throwing review() must not starve the remaining routes or the
            # post-drain transition loops. The finding's drift fingerprint
            # stays unmarked, so the route re-collects next pass.
            logger.exception(
                "review_callback escaped for orphaned issue %s / pr %s",
                issue_number,
                pr_number,
            )
            write_gate.log_event(
                "orphaned_worker_review_route_failed",
                {
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": reason,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                level="warning",
            )
            continue
        routed = False
        rework_requested = False
        # See _route_rework_candidate_to_review's matching comment: review()
        # can return ok=True for the janitor-gate conflict/no-op-rework route
        # (no packet, no review_started transition) as well as for a real
        # packet. Only a real packet should flip this orphaned-but-dispatched
        # issue to "reviewing".
        routed_to_rework = bool(review_result.data.get("routed_to_rework"))
        # Issue #558: review() also returns ok=True when it converges a
        # CLOSED-unmerged PR's state entry to "closed" at the janitor gate.
        # That is not a fresh packet -- the PR is dead, not transiently
        # blocked -- so it must NOT flip this issue to "reviewing" (an
        # ACTIVE_STATE_STATUS no reconcile rule clears while the GitHub
        # issue itself stays open: issue_active_label_no_open_pr sees the
        # closed PR still links to the issue, issue_active_label_with_open_pr
        # sees no OPEN PR, and the unknown-status recompute sweep skips
        # "reviewing" because it is a VALID_ISSUE_STATUSES member). The
        # issue's disposition is left to the existing closed-unmerged
        # issue-side handling (closed_unmerged_pr_active_labels). Neither
        # the "reviewing" flip nor the transient-block drift fingerprint
        # below applies to a permanently-dead PR.
        closed_unmerged_converged = bool(review_result.data.get("closed_unmerged_converged"))
        with _wf.state_lock(state_file):
            state = _wf.load_state(state_file)
            pr_state = state["prs"].get(str(pr_number), {})
            entry = state["issues"].get(str(issue_number), {})
            decision_unchanged = pr_state.get("reviewed_head_sha") == reviewed_head_sha_before
            if (
                review_result.ok
                and not routed_to_rework
                and not closed_unmerged_converged
                and decision_unchanged
                and isinstance(entry, dict)
                and entry.get("status") == "dispatched"
            ):
                state["issues"][str(issue_number)] = {**entry, "status": "reviewing"}
                routed = True
            elif (
                not review_result.ok
                and not routed_to_rework
                and not closed_unmerged_converged
                and decision_unchanged
                and reason == "dead_worker_completed_outcome"
                and isinstance(entry, dict)
                and entry.get("status") == "dispatched"
            ):
                # Issue #1915: the completed outcome applied but review()
                # cannot produce a fresh packet for its head (e.g. the
                # janitor's unchanged-head no-op gate for a substantive
                # request_changes verdict). The worker's recovered edits
                # are published yet the verdict still stands, so the
                # rework is still outstanding: hand the issue back to the
                # ordinary dispatch loop as "rework_requested" WITHOUT
                # crediting the dead worker's death (no ``worker_death_at``
                # -- the #1911 invariant) and without burning an immediate
                # redispatch. dispatch_rework's own head-unchanged
                # accounting bounds any repeat, and the needs_rework label
                # edge is applied post-lock below. The ``orphan_drift_at``
                # backstop armed at collection time is resolved here --
                # clear it so the marker cannot outlive the finding it
                # was armed for (a later dispatched-phase orphan_drift_at
                # must describe that dispatch, not this one).
                state["issues"][str(issue_number)] = {
                    **entry,
                    "status": "rework_requested",
                    "dispatched_at": None,
                    "orphan_drift_at": None,
                }
                rework_requested = True
                rework_requested_recoveries.append(issue_number)
            elif (
                not review_result.ok
                and not routed_to_rework
                and isinstance(entry, dict)
                and entry.get("status") == "dispatched"
            ):
                # Review failed: mark the drift fingerprint so the next pass
                # does not retry/re-emit for this unchanged head.
                state["issues"][str(issue_number)] = {
                    **entry,
                    "orphan_drift_fingerprint": fingerprint,
                    "orphan_drift_at": _wf.utc_now(),
                }
            if rework_requested:
                state = write_gate.append_event(
                    state,
                    "orphaned_worker_recovered",
                    {
                        "issue_number": issue_number,
                        "pr_number": pr_number,
                        "previous_status": "dispatched",
                        "new_status": "rework_requested",
                        "reason": reason,
                    },
                )
            else:
                state = write_gate.append_event(
                    state,
                    "orphaned_worker_routed_to_review"
                    if review_result.ok
                    else "orphaned_worker_drift",
                    {
                        "issue_number": issue_number,
                        "pr_number": pr_number,
                        "review_ok": review_result.ok,
                        "routed": routed,
                        "live_head_sha": live_head_sha,
                        "reviewed_head_sha": reviewed_head_sha_before,
                        "reason": reason,
                    },
                )
            write_gate.save_state(state)

    # Issue #1915: apply the needs_rework label edge for completed-outcome
    # issues the drain returned to the rework queue. Same post-lock shape
    # as the reap_escalations transitions the caller applies below --
    # transition() is network I/O and never runs under the state lock.
    for issue_number in rework_requested_recoveries:
        try:
            result = write_gate.transition(gh, config.labels, issue_number, "rework_requested")
        except Exception as exc:
            # Same isolation the per-route guard above gives review(): a
            # throwing label call must not starve the remaining recoveries.
            # The status flip already committed, so the marker keeps the
            # failure inspectable and reconcile's label sweep converges it.
            logger.exception(
                "rework_requested label transition escaped for issue %s",
                issue_number,
            )
            result = None
            exception_error = f"{type(exc).__name__}: {exc}"
        else:
            exception_error = None
        if result is not None and result.outcome == _wf.TransitionOutcome.APPLIED:
            continue
        # Same label_error persistence dead_worker_reap uses: the status
        # flip already committed, so a failed edge leaves a stale label
        # that the next sweep's drift/reconcile will still converge --
        # the marker keeps the failure inspectable meanwhile.
        with _wf.state_lock(state_file):
            state = _wf.load_state(state_file)
            entry = state["issues"].get(str(issue_number), {})
            if isinstance(entry, dict):
                entry["label_error"] = (
                    {
                        "edge": "rework_requested",
                        "outcome": result.outcome.value,
                        "add_failures": result.add_failures,
                        "remove_failures": result.remove_failures,
                    }
                    if result is not None
                    else {
                        "edge": "rework_requested",
                        "outcome": "exception",
                        "error": exception_error,
                    }
                )
                state["issues"][str(issue_number)] = entry
                write_gate.save_state(state)
