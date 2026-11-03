"""Operator-queue route for a PR held by an operator-only required check (issue #2920).

A required check can fail until the *operator* acts (adds an approval label, signs
off) and not until code changes. Review's CI-failure gate cannot tell the two
apart: it recorded ``request_changes`` ("CI failed on <check>; push a fix"), the
issue moved to ``agent:needs-rework`` and a worker session ran for a failure no
push can clear -- one wasted worker round per PR through such a gate, ending on
the operator queue where it should have started.

``dispatch.operator_gated_checks`` names those checks. :func:`is_operator_gated_failure`
is the single predicate and :func:`route_operator_gated_pr` the single writer of
the exit; :func:`park_operator_gated_review` is the one call ``OrchestratorApp.review``
makes at its CI-failure gate -- before the flake rerun (a rerun cannot clear a check
that waits for an operator), the ``request_changes`` verdict and the rework worker.
The glue lives here rather than in ``workflow.py``, which is over its file-size cap.

The route fires only when **all** of these hold, so every other PR keeps today's
behaviour:

* every failed required check is in ``operator_gated_checks`` (one code-fixable
  failure beside a gated one still needs a worker, so it is not routed here);
* the failure is the PR's sole janitor failure (the caller's
  ``is_check_failure_block``): a body or title the worker can fix is not parked;
* the bound issue carries a ``dispatch.human_merge_labels`` label, which is the
  operator's declaration that a human owns this PR's outcome.

The escalation is ``reason_class="mechanical"`` -- the operator-queue class. Three
properties follow from that choice, each verified rather than assumed:

* ``escalation._escalation_edge`` resolves it to the ``operator_queued`` edge, so
  the issue lands on ``agent:operator-queue``;
* the escalated-label repair sweep leaves it there. A ``policy`` escalation on the
  same label is re-derived to ``agent:human-needed`` by that sweep (it maps every
  non-mechanical class to the ``escalated`` edge), which is why this route is not
  ``policy`` even though the human-merge label is a policy input;
* the mechanical de-escalation sweep clears it once the PR's janitor gate is green
  -- that is, once the operator has cleared the check -- and review resumes. Its
  ``max_auto_deescalations`` cap bounds a check that goes red again.

The event is the existing ``review_dispatch_escalated`` kind with a distinct
``reason`` (the same shape ``review_empty_diff_route`` uses): the dashboard's
escalation milestone and the quality metrics already consume it by ``reason``.

This module imports nothing from ``workflow``.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .escalation import _escalate_issue, _escalation_edge, _record_issue_label_error
from .github import label_names
from .labels import TransitionOutcome
from .state import SINK_STATUSES, load_state, state_lock

if TYPE_CHECKING:
    from .config import DispatchConfig, LabelConfig, OrchestratorConfig
    from .github import GitHubLike
    from .janitor import JanitorVerdict
    from .write_gate import WriteGate

OPERATOR_GATED_ESCALATION_REASON = "operator_gated_check_failed"
"""``escalation_reason`` the route stamps and the ``review_dispatch_escalated`` reason."""


@dataclass(frozen=True)
class OperatorGatedRouteOutcome:
    """What :func:`route_operator_gated_pr` did.

    ``routed`` is ``False`` when the PR or its issue was already in a sink status
    (another lane escalated it between the caller's read and this write): nothing
    was written, so that escalation's reason is not clobbered (the #1461 dedup
    contract). ``label_error`` is the failed/partial ``operator_queued`` label
    edge, ``None`` when it applied cleanly or nothing was routed.
    """

    routed: bool
    label_error: dict[str, Any] | None = None


def is_operator_gated_failure(
    dispatch: DispatchConfig,
    failed_checks: Iterable[str],
    issue_labels: Collection[str] | None,
) -> bool:
    """Is this failure one only the operator can clear, on an issue a human owns?

    ``False`` with an empty ``operator_gated_checks`` (the default), with no failed
    check, with any failed check outside the list, with no
    ``human_merge_labels`` label on the issue, or when the issue's labels could not
    be read (``issue_labels is None``): an unreadable label set must never turn
    into a skipped rework.
    """
    gated = set(dispatch.operator_gated_checks)
    failed = set(failed_checks)
    if not gated or not failed or not failed <= gated or issue_labels is None:
        return False
    return bool(set(dispatch.human_merge_labels) & set(issue_labels))


def route_operator_gated_pr(
    *,
    state_file: Path,
    write_gate: WriteGate,
    gh: GitHubLike,
    labels: LabelConfig,
    pr_number: int,
    issue_number: int,
    failed_checks: Iterable[str],
    head_sha: str | None,
) -> OperatorGatedRouteOutcome:
    """Park the PR's issue on the operator queue instead of requesting rework.

    The state write (escalation, ``review_dispatch_escalated`` event) happens under
    the state lock on freshly loaded state; the ``operator_queued`` label edge is
    applied after the lock is released, because ``transition()`` makes GitHub API
    calls. A label edge that does not apply cleanly is persisted as the issue's
    ``label_error`` so the escalated-label repair sweep retries it.

    Every exit: a sink status already set -> nothing written, ``routed=False``;
    otherwise state and event are written, the edge is attempted once, and
    ``routed=True`` is returned with the edge's failure (or ``None``).
    """
    failed = list(failed_checks)
    with state_lock(state_file):
        state = load_state(state_file)
        state.setdefault("issues", {})
        state.setdefault("prs", {})
        pr_state = state["prs"].get(str(pr_number), {})
        issue_state = state["issues"].get(str(issue_number), {})
        if pr_state.get("status") in SINK_STATUSES or issue_state.get("status") in SINK_STATUSES:
            return OperatorGatedRouteOutcome(routed=False)
        state = _escalate_issue(
            state,
            issue_number,
            reason=OPERATOR_GATED_ESCALATION_REASON,
            reason_class="mechanical",
            pr_number=pr_number,
        )
        state = write_gate.append_event(
            state,
            "review_dispatch_escalated",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                # Parking the PR is not a review attempt; report the counter
                # unchanged so "escalation without attempts" stays queryable.
                "attempt_count": int(pr_state.get("review_dispatch_attempt_count", 0)),
                "reason": OPERATOR_GATED_ESCALATION_REASON,
                "failed_required_checks": failed,
                "head_sha": head_sha,
            },
        )
        write_gate.save_state(state)

    edge = _escalation_edge("escalated", "mechanical")
    result = write_gate.transition(
        gh,
        labels,
        issue_number,
        edge,
        pr_number=pr_number,
        cause=OPERATOR_GATED_ESCALATION_REASON,
    )
    if result.outcome == TransitionOutcome.APPLIED:
        return OperatorGatedRouteOutcome(routed=True)
    _record_issue_label_error(state_file, write_gate, issue_number, edge, result)
    return OperatorGatedRouteOutcome(
        routed=True,
        label_error={
            "edge": edge,
            "outcome": result.outcome.value,
            "add_failures": result.add_failures,
            "remove_failures": result.remove_failures,
        },
    )


@dataclass(frozen=True)
class OperatorGatedReviewExit:
    """The ``review()`` result for a PR parked by :func:`park_operator_gated_review`."""

    message: str
    data: dict[str, Any]


def park_operator_gated_review(
    *,
    config: OrchestratorConfig,
    state_file: Path,
    write_gate: WriteGate,
    gh: GitHubLike,
    pr: dict[str, Any],
    pr_number: int,
    issue: Any,
    issue_number: int | None,
    verdict: JanitorVerdict,
) -> OperatorGatedReviewExit | None:
    """Park the PR if its only failure is an operator-gated check; else ``None``.

    ``None`` -- the caller continues to the flake rerun and the rework verdict
    exactly as before -- unless the PR is bound to an issue, the failure is the
    PR's sole janitor failure (``verdict.is_check_failure_block``) and
    :func:`is_operator_gated_failure` holds. The issue's labels are read from the
    ``issue`` payload ``review()`` already fetched, so no GitHub call is added on any
    path; a payload without a ``labels`` key is unreadable, not label-free, and
    returns ``None``.

    Otherwise :func:`route_operator_gated_pr` runs and its outcome is folded into
    the exit: ``escalated`` is ``False`` when another lane had already escalated
    the PR (nothing written).
    """
    if issue_number is None or not verdict.is_check_failure_block:
        return None
    issue_labels = label_names(issue) if isinstance(issue, dict) and "labels" in issue else None
    if not is_operator_gated_failure(
        config.dispatch, verdict.failed_required_checks, issue_labels
    ):
        return None
    routed = route_operator_gated_pr(
        state_file=state_file,
        write_gate=write_gate,
        gh=gh,
        labels=config.labels,
        pr_number=pr_number,
        issue_number=issue_number,
        failed_checks=verdict.failed_required_checks,
        head_sha=pr.get("headRefOid"),
    )
    checks = ", ".join(verdict.failed_required_checks)
    outcome = "issue parked on the operator queue" if routed.routed else "already escalated"
    return OperatorGatedReviewExit(
        message=f"PR #{pr_number} held by operator-gated check(s) {checks}; {outcome}",
        data={
            "pr": pr_number,
            "issue": issue_number,
            "operator_gated": True,
            "escalated": routed.routed,
            "label_error": routed.label_error,
            "checks_unavailable": False,
        },
    )
