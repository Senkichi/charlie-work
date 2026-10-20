"""Mergequeue-requeue rework routing delegate for ``OrchestratorApp``.

Issue #2743: the per-head requeue cap is the admission gate; this module is
the lane it hands off to. The ``workflow_delegation`` installer re-attaches
each ``def`` onto the class; the shared ``_route_to_rework`` wrapper it calls
lives in ``state_rework_routing``.
"""

from __future__ import annotations

from typing import Any

import charlie_work.workflow as _wf


def _request_mergequeue_requeue_rework(
    self,
    pr: dict[str, Any],
    issue_number: int,
    decision: dict[str, Any],
    *,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Route an approved PR the merge queue kept reverting at one head to rework.

    Issue #2743: the cap is the admission gate; this is the lane it hands off
    to. The prompt spells out WHY the PR is back -- its own checks stayed
    green the whole time, so a generic "fix the failing CI" prompt would send
    the worker hunting in the wrong place. The failure lives on Aviator's
    queue branch (the ``mq-bot-*`` merge of the PR into base): something about
    the COMBINED tree fails where the PR's own head did not. As with the
    other rework routers the approved verdict stays on disk so the rework
    push is re-confirmed, not re-reviewed from scratch.
    """
    label = self.config.auto_merge.mergequeue_label or "mergequeue"
    summary = (
        f"The merge queue (label {label!r}) kept reverting this PR at its current "
        "head: the queue branch's CI -- the merge of this PR into the base branch "
        "Aviator builds as its mq-bot-* branch -- failed deterministically while "
        "the PR's own checks stayed green, so the fleet stopped re-adding the "
        "label. Reproduce the base-branch merge locally, find what only fails on "
        "the combined tree (typically an interaction with work that merged after "
        "this head, or a check that behaves differently on a merge commit), fix "
        "it, and push. The code changes are already approved; do not re-litigate "
        "the review."
    )
    requested_at = _wf.utc_now()
    # The revert that tripped the cap is persisted by stage-5 accounting AFTER
    # this route, so the count read here is the pre-pass value; the
    # ``mergequeue_requeue_capped`` event carries the authoritative total.
    snapshot = _wf.load_state_locked(self.paths.state_file)
    pr_state = snapshot.get("prs", {}).get(str(int(pr["number"])), {})
    merged_extra_state = {"mergequeue_requeue_rework_requested_at": requested_at}
    if extra_state:
        merged_extra_state.update(extra_state)
    return self._route_to_rework(
        pr,
        issue_number,
        decision,
        summary,
        "mergequeue_requeue_rework_requested",
        extra_payload={
            "mergequeue_requeue_rework_requested_at": requested_at,
            "mergequeue_label": label,
            "consecutive_mergequeue_requeues": pr_state.get("consecutive_mergequeue_requeues"),
        },
        extra_state=merged_extra_state,
    )
