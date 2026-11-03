"""Operator-queue exit route for the review-dispatch empty-diff gate (issue #2899).

``OrchestratorApp.dispatch_reviews`` skips a PR whose review packet's
``diff.patch`` is empty (issue #1251): a zero-file PR must not burn a paid
reviewer session. That skip alone had no exit -- a zero-file PR whose issue
stayed ``agent:pr-open`` produced one warning EVERY pass forever (491 in 48h
on one fleet) with no review, no merge, and no terminal state.

:func:`route_empty_diff_prs` is the single writer of the exit: the first pass
that observes an empty diff for a given packet head routes the linked issue to
the operator queue through the standard mechanical escalation
(``escalation._escalate_issue``, ``reason_class="mechanical"``, which the label
edge in ``dispatch_reviews`` resolves to the ``operator_queued`` edge), with a
reason naming the empty diff.

Durability has two halves, deliberately in two places:

* **This module** records the head-scoped ``empty_diff_operator_queued_head_sha``
  marker on the PR record and stays silent for a head that already carries it.
  The marker is NOT in ``UNESCALATE_PR_RESET_FIELDS``: an operator re-arm of a
  still-empty head must not re-fire the escalation they just cleared for the
  same bytes.
* **The mechanical de-escalation sweep**
  (``orchestration/state_mechanical.py::_deescalate_mechanical_issue``) refuses
  to auto-clear an ``empty_diff_zero_file`` escalation while the PR's live head
  still equals the marker. An open, mergeable, janitor-green zero-diff PR
  satisfies every other clear condition, so without that skip the sweep would
  clear the escalation and the marker would then keep the gate silent forever
  -- no exit, now with no signal. Once the head moves the sweep clears as
  usual, so a regained diff is reviewed and a new empty head re-routes once.

The reason string and marker field live here as the single source of truth for
both halves. The review-dispatch attempt counter is never touched: an empty
diff is not a review attempt and must not walk toward the attempt-cap
escalation.

``workflow.py`` re-exports the public names via a facade import block (the
same pattern as ``escalation.py`` and ``dispatch_selection.py``). This module
imports nothing from ``workflow``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .escalation import _escalate_issue
from .state import SINK_STATUSES

if TYPE_CHECKING:
    from .write_gate import WriteGate

EMPTY_DIFF_ESCALATION_REASON = "empty_diff_zero_file"
"""``escalation_reason`` the route stamps; the sweep's skip branch keys on it."""

EMPTY_DIFF_ROUTED_HEAD_FIELD = "empty_diff_operator_queued_head_sha"
"""PR-record field holding the packet head the route fired for."""


@dataclass(frozen=True)
class EmptyDiffSkip:
    """One PR the empty-diff gate skipped from dispatch this pass.

    ``packet_head_sha`` is the head the (empty) packet was built for; ``None``
    when it could not be read, in which case the route still fires but records
    no dedup marker (an unscoped marker could never be told apart from a new
    head, which is exactly the silent-forever shape this module exists to
    remove -- the sweep's attempt cap bounds the re-route instead).
    """

    pr_number: int
    issue_number: int | None
    packet_head_sha: str | None


@dataclass(frozen=True)
class EmptyDiffRouteOutcome:
    """What :func:`route_empty_diff_prs` did this pass.

    ``state`` is the mutated state (the caller saves it). ``routed_prs`` lists
    every PR whose route fired fresh, in input order. ``label_edges`` is the
    subset that resolved a linked issue and therefore still owes the
    ``operator_queued`` label edge (applied by the caller outside the state
    lock, since ``transition()`` makes GitHub API calls); a PR without a
    resolvable issue (cross-repo/fork) is escalated PR-only and has no edge.
    """

    state: dict[str, Any]
    routed_prs: tuple[int, ...]
    label_edges: tuple[tuple[int, int], ...]


def route_empty_diff_prs(
    state: dict[str, Any],
    skips: Iterable[EmptyDiffSkip],
    *,
    write_gate: WriteGate,
) -> EmptyDiffRouteOutcome:
    """Route each skipped zero-diff PR to the operator queue, once per head.

    Must run under the state lock on a freshly loaded ``state``. Every
    ``continue`` below is a silent path -- no state mutation, no event -- and
    each is pinned by a direct test:

    1. The PR or its issue is already in a sink status (escalated/blocked
       concurrently since the selection snapshot, another lane won the race):
       stay silent rather than clobbering that escalation's reason with ours
       (escalation.py's #1461 dedup contract).
    2. The PR record already carries the marker for this exact head (the
       escalation was cleared -- operator re-arm -- while the head is still
       empty): emit nothing, so there is no escalate/clear ping-pong.

    Otherwise the PR escalates (``issue_number`` may be ``None`` -> PR-only),
    the marker is written when the head is known, and one
    ``review_dispatch_escalated`` event names the empty diff.
    """
    state.setdefault("issues", {})
    state.setdefault("prs", {})
    routed_prs: list[int] = []
    label_edges: list[tuple[int, int]] = []
    for skip in skips:
        pr_number = skip.pr_number
        pr_state = state["prs"].get(str(pr_number), {})
        issue_num = pr_state.get("issue_number") or skip.issue_number
        issue_state = state["issues"].get(str(issue_num), {}) if issue_num is not None else {}
        if pr_state.get("status") in SINK_STATUSES or issue_state.get("status") in SINK_STATUSES:
            continue
        if skip.packet_head_sha and (
            pr_state.get(EMPTY_DIFF_ROUTED_HEAD_FIELD) == skip.packet_head_sha
        ):
            continue
        state = _escalate_issue(
            state,
            issue_num,
            reason=EMPTY_DIFF_ESCALATION_REASON,
            reason_class="mechanical",
            pr_number=pr_number,
        )
        if skip.packet_head_sha:
            state["prs"][str(pr_number)] = {
                **state["prs"].get(str(pr_number), {}),
                "number": pr_number,
                EMPTY_DIFF_ROUTED_HEAD_FIELD: skip.packet_head_sha,
            }
        state = write_gate.append_event(
            state,
            "review_dispatch_escalated",
            {
                "pr_number": pr_number,
                "issue_number": issue_num,
                # The attempt count did not move (an empty diff is not a
                # review attempt, issue #1251); report it so the "escalation
                # without attempts" shape is queryable.
                "attempt_count": int(pr_state.get("review_dispatch_attempt_count", 0)),
                "reason": EMPTY_DIFF_ESCALATION_REASON,
                "packet_head_sha": skip.packet_head_sha,
            },
        )
        if issue_num is not None:
            label_edges.append((int(pr_number), int(issue_num)))
        routed_prs.append(int(pr_number))
    return EmptyDiffRouteOutcome(
        state=state,
        routed_prs=tuple(routed_prs),
        label_edges=tuple(label_edges),
    )
