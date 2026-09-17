"""Rework request / conflict-routing delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L02 batch 3 (issue #1654, parent #1633, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from typing import Any


def _rework_candidate_conflict_blocked(self, pr_data: dict[str, Any], pr_number: int) -> bool:
    """Issue #1349: a patch-id-advanced head on a CONFLICTING/DIRTY PR is
    still a legitimate launch candidate — the head advance did not resolve
    the conflict the rework was requested for. Routing such a PR to
    review() just bounces off the janitor gate's conflict check back to
    rework_requested, deadlocking the issue between dispatch_rework and
    review() forever (the only exit being the #765 stall escalation to a
    human, not a dispatch).

    Returns True when the candidate must be kept as a launch candidate
    (conflict-bypass applies). Shared by the dry-run and live paths so the
    two cannot drift.

    ``pr_list``'s ``mergeable`` can be UNKNOWN (GitHub computes it
    asynchronously); ``mergeStateStatus == "DIRTY"`` is reliable from
    ``pr_list``, but a CONFLICTING reading may only appear on a fresh
    ``pr_view``. When ``pr_list``'s signal is indeterminate (not a definite
    MERGEABLE and not a definite CONFLICTING), re-check with a fresh
    ``pr_view`` before routing to review so a persistently-conflicting PR
    is not misrouted back into the deadlock. A failed ``pr_view`` returns
    ``{}`` (production) which ``_is_merge_conflict`` reads as
    not-conflicting, falling through to the review lane — the same
    fail-closed behavior review() itself uses (issue #1349).
    """
    if self._is_merge_conflict(pr_data):
        return True
    live_mergeable = str(pr_data.get("mergeable") or "").upper()
    if live_mergeable not in ("MERGEABLE", "CONFLICTING"):
        fresh_pr = self.gh.pr_view(pr_number)
        if fresh_pr and self._is_merge_conflict(fresh_pr):
            return True
    return False


def _request_no_op_rework_repair(
    self,
    pr: dict[str, Any],
    issue_number: int,
    decision: dict[str, Any],
    *,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Route a PR whose last rework cycle pushed no actual change to rework.

    The janitor's no-op-rework check (``janitor._check_no_op_rework``)
    only detects the condition (unchanged patch-id/head since the last
    request_changes verdict); nothing previously consumed it
    (pr-lifecycle.md Finding 1's "no-op-rework-never-escalated"
    sub-case). This is that consumer, called from
    ``_route_janitor_gate_failure_to_rework`` and shaped like
    ``_request_merge_conflict_rework``.
    """
    summary = (
        "The previous rework cycle produced no actual content change (the diff or head "
        "matches the last request_changes verdict). Check the branch worktree for "
        "unpushed commits and push the real fix, or explain in the PR body why no "
        "further change was needed."
    )
    return self._route_to_rework(
        pr,
        issue_number,
        decision,
        summary,
        # event-consumer: audit-only -- records a rework repair request already
        # routed to a worker via _route_to_rework (the dispatch IS the action)
        "no_op_rework_repair_requested",
        extra_state=extra_state,
    )


def _request_cross_pr_revert_rework(
    self,
    pr: dict[str, Any],
    issue_number: int,
    decision: dict[str, Any],
    reason: str,
) -> dict[str, Any] | None:
    """Route an approved PR whose branch silently reverts a base commit to rework."""
    summary = (
        f"{reason}. Remove the revert commit (or the merge+revert pair) from the PR "
        "history, or add an explicit 'allow-revert: <reason>' line to the PR body if the "
        "revert is intentional. Then push the corrected branch and re-request review."
    )
    return self._route_to_rework(
        pr, issue_number, decision, summary, "cross_pr_revert_rework_requested"
    )
