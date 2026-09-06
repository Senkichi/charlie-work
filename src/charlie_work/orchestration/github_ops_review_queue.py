"""Review-queue command delegate moved out of ``OrchestratorApp``.

Track 2 Phase B, L04 batch 2 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Body relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches the top-level ``def``
unwrapped onto ``OrchestratorApp``. ``CommandResult`` and ``linked_issue_number``
are reached through ``_wf.``: ``CommandResult`` is a ``charlie_work.workflow``
module-level class, and ``linked_issue_number`` is patched on
``charlie_work.workflow`` by the suite (Tier D, via the ``workflow_mod`` alias).
``StateLockBusy`` is imported directly (no test patches it on
``charlie_work.workflow``).
"""

from __future__ import annotations

import charlie_work.workflow as _wf
from typing import Any
from charlie_work.state import StateLockBusy


def review_queue(self) -> _wf.CommandResult:
    """Enumerate open agent PRs whose review packet is current and awaiting a verdict.

    When a recorded verdict (``approved``, ``request_changes``, or
    ``blocked``) was made against an older head, this method computes the
    live head's stable patch-id. If it matches the recorded
    ``reviewed_patch_id``, the verdict is carried forward to the new head
    (atomic decision-file + state update) and the PR is not queued as
    stale. In dry-run mode the carry-forward write is skipped; the patch-id
    check still runs so the queue reflects real content changes.

    A PR is queued when:

    - It has a linked issue (same as ``review()``).
    - ``prs/pr-N/review-prompt.md`` exists.
    - The recorded decision is ``missing``/``pending`` and the stored packet
      head OID matches the PR's live ``headRefOid``.
    - The recorded decision is a stale ``request_changes``/``blocked``/
      ``approved`` verdict whose patch-id genuinely differs from the live
      head and the packet head is still current.

    A PR is NOT queued, but has a side-effecting repair applied instead,
    when the recorded decision is an actionable (non-content-free),
    non-escalated ``request_changes`` verdict at the live head whose
    linked issue status does not show it was ever routed to rework
    (issue #784 AC-8, Case 2, ``_reroute_stranded_request_changes``) --
    this re-applies the SAME ``rework_requested`` transition
    ``record_review`` already decided, so it is a repair of a dropped
    transition, not a new decision, and is a no-op once the issue is at
    ``rework_requested`` (or otherwise already spoken for).

    Returns:
        CommandResult with a sorted ``queue`` list keyed by repo. Each
        candidate dict carries ``mergeable`` and ``mergeStateStatus`` from
        ``pr_list()`` so callers (e.g. ``dispatch_reviews``) can detect
        merge-conflicting PRs without an extra ``pr_view`` call (issue #1497).
    """
    prs = self.gh.pr_list()
    queue: list[dict[str, Any]] = []

    # Issue #1229: validate branch-name-derived issue numbers against the
    # actual open-issue set so a stale branch name (e.g. agent/issue-709-…
    # left over from a merged PR #709, reused by an unrelated issue-less
    # PR) cannot bind the PR to a non-existent or closed issue here. This
    # is the same phantom-binding failure class already fixed at the
    # dispatch-claim pr_by_issue construction, the dead-session escalation
    # guard, and the orphaned-worker sweep: review_queue's issue_number
    # feeds _reroute_stranded_request_changes (a real rework-routing state
    # mutation) and _emit_stale_ci_verdict_requeued (a state.json write),
    # so a stale binding would route rework at the wrong issue subject.
    # Built once before the loop so a single issue_list(state="open") call
    # (cached within the pass on the real GitHub client) is shared across
    # every PR in this queue.
    branch_validator = self._make_branch_issue_validator()
    for pr in prs:
        issue_number = _wf.linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=self.config.dispatch.branch_prefix,
            branch_issue_validator=branch_validator,
        )
        if issue_number is None:
            continue

        pr_number = int(pr["number"])
        pr_dir = self.paths.prs / f"pr-{pr_number}"
        prompt_path = pr_dir / "review-prompt.md"
        if not prompt_path.exists():
            continue

        packet_head_sha = self._read_packet_head_oid(pr_number)
        live_head_sha = pr.get("headRefOid")
        if live_head_sha is None:
            continue

        decision = self._review_decision(pr_number)
        decision_value = decision.get("decision")
        reviewed_head_sha = decision.get("reviewed_head_sha")

        if decision_value in ("approved", "request_changes", "blocked"):
            if reviewed_head_sha == live_head_sha:
                if decision_value == "request_changes" and self._is_stale_ci_request_changes(
                    pr_number, decision
                ):
                    # Issue #1111: the verdict's only findings cite
                    # required checks that are all green on this same head
                    # — the failure it describes no longer exists (the
                    # check flipped transiently mid-review, or a rerun
                    # recovered it). Re-driving rework here is a
                    # guaranteed no-op that burns no_op_rework_attempts
                    # toward a manufactured escalation, so instead queue
                    # the PR for a FRESH review despite the unchanged
                    # head. The stale verdict is only ever superseded by
                    # a new recorded verdict, never auto-approved;
                    # repeated request_changes re-verdicts stay bounded
                    # by max_rework_cycles in record_review.
                    if packet_head_sha == live_head_sha and self._packet_template_current(
                        pr_number
                    ):
                        self._emit_stale_ci_verdict_requeued(
                            pr_number,
                            issue_number,
                            reviewed_head_sha,
                            live_head_sha,
                            decision.get("required_changes"),
                        )
                        queue.append(
                            {
                                "pr": pr_number,
                                "issue": issue_number,
                                "packet_head_sha": packet_head_sha,
                                "decision": "stale",
                                "reviewed_head_sha": reviewed_head_sha,
                                "mergeable": pr.get("mergeable"),
                                "mergeStateStatus": pr.get("mergeStateStatus"),
                            }
                        )
                    continue
                # Issue #784 AC-8 (Case 2): "reviewed at live head" only
                # means "nothing to do" if the recorded verdict was
                # actually actioned. A request_changes verdict that
                # record_review decided is within the rework-cycle
                # budget (``escalated`` falsy) must have routed its
                # issue to rework_requested in that same call; if the
                # issue status doesn't show that -- e.g. issue #789's
                # reconcile one-way "closed" gate clobbered it -- re-
                # drive the same target here rather than silently
                # re-confirming a verdict nobody ever acted on.
                if (
                    not self.dry_run
                    and decision_value == "request_changes"
                    and not decision.get("escalated")
                ):
                    self._reroute_stranded_request_changes(pr, issue_number, decision)
                continue

            check = self._check_carry_forward(pr_number, decision)
            if (
                check.carry_forward
                and decision_value == "request_changes"
                and self._is_stale_ci_request_changes(pr_number, decision)
            ):
                # Issue #1111 (head-advanced variant): the diff content is
                # unchanged (carry-forward matched), but the verdict's only
                # findings cite required checks that are all green on the
                # live head — e.g. a no-op rework push after a transient CI
                # failure recovered. Carrying the request_changes verdict
                # forward would re-apply a failure that no longer exists,
                # so skip the carry-forward and fall through to the
                # stale-queue path below: a fresh review supersedes the
                # verdict (never auto-approved).
                self._emit_stale_ci_verdict_requeued(
                    pr_number,
                    issue_number,
                    reviewed_head_sha,
                    live_head_sha,
                    decision.get("required_changes"),
                )
            elif check.carry_forward:
                # In dry-run mode we still run the content check so the queue
                # reflects real changes, but we skip the durable head update.
                if not self.dry_run:
                    try:
                        self._update_approval_head(
                            pr_number,
                            decision,
                            live_head_sha,
                            old_head=reviewed_head_sha,
                            issue_number=issue_number,
                            tier=check.tier or "patch-id",
                            new_patch_id=check.live_patch_id,
                            new_signature=check.live_signature,
                        )
                    except StateLockBusy:
                        # Could not mirror the carry-forward into state.json,
                        # but the decision-file update is the durable source
                        # of truth; proceed as carried-forward.
                        pass
                continue

            # If the packet is stale, we cannot dispatch a new reviewer from
            # it; the merge gate will route the issue to re-review. Only
            # surface as stale when the packet head is still current.
            if packet_head_sha is None or packet_head_sha != live_head_sha:
                continue
            # Issue #592: a template edit makes a same-head packet stale
            # too. Don't queue it for dispatch -- loop() will regenerate it
            # from the current template first, and the next pass queues it.
            if not self._packet_template_current(pr_number):
                continue

            queue.append(
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "packet_head_sha": packet_head_sha,
                    "decision": "stale",
                    "reviewed_head_sha": reviewed_head_sha,
                    "mergeable": pr.get("mergeable"),
                    "mergeStateStatus": pr.get("mergeStateStatus"),
                }
            )
        elif decision_value in ("pending", "missing"):
            if packet_head_sha is None or packet_head_sha != live_head_sha:
                continue
            # Issue #592: a template-stale packet must be regenerated by
            # loop() before it is dispatched, not handed to a reviewer as-is.
            if not self._packet_template_current(pr_number):
                continue

            queue.append(
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "packet_head_sha": packet_head_sha,
                    "decision": decision_value
                    if decision_value in ("pending", "missing")
                    else "missing",
                    "reviewed_head_sha": None,
                    "mergeable": pr.get("mergeable"),
                    "mergeStateStatus": pr.get("mergeStateStatus"),
                }
            )

    queue = self._sort_review_queue_by_dependency_depth(queue)
    return _wf.CommandResult(
        True,
        f"review queue: {len(queue)} PR(s) awaiting verdict",
        {"queue": queue},
    )
