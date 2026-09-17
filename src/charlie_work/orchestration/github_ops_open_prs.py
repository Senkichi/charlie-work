"""Open-agent-PR update delegate moved out of ``OrchestratorApp``.

Track 2 Phase B, L04 batch 1 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each top-level ``def``
unwrapped onto ``OrchestratorApp``. ``linked_issue_number`` and
``_authorized_override_matches`` are reached through ``_wf.``:
``linked_issue_number`` is patched on ``charlie_work.workflow`` by the suite
(Tier D, via the ``workflow_mod`` alias), and ``_authorized_override_matches``
is a ``charlie_work.workflow`` module-level def. All other free names are
imported directly (no test patches them on ``charlie_work.workflow``).
"""

from __future__ import annotations

import charlie_work.workflow as _wf
from typing import Any
from charlie_work.github import GitHubError


def _update_open_agent_prs(self, merged_pr_number: int) -> list[dict[str, Any]]:
    """Update remaining open agent PRs after a successful merge.

    Behavior is controlled by ``auto_merge.update_branch_strategy``:

    - "front_of_train" (default): update only the head of the approved
      queue, so a single merge step causes at most one CI reset on a
      single-runner merge train.
    - "broadcast": update every eligible open tracked PR that is not
      approved-pending-ship and has no required checks in-flight. Intended
      for multi-runner setups. PRs whose current review decision is
      ``request_changes``, escalated, or blocked are skipped.
    - "off": do nothing.

    Per-PR failures (conflicts, network errors) are reported as values and
    never abort the batch operation. A GitHubError from pr_list is also
    reported as a value and never propagates.
    """
    results: list[dict[str, Any]] = []
    mode = self.config.auto_merge.update_branch_strategy
    if mode == "off":
        return results

    if mode == "front_of_train":
        try:
            candidates = self._merge_train_candidates(exclude_pr_number=merged_pr_number)
        except GitHubError as exc:
            return [{"error": f"pr_list failed: {exc}"}]

        if not candidates:
            return results

        # Only the head of the merge-train queue is synced.
        _, pr_number, pr, decision, head = candidates[0]
        base_current = self._is_base_current(pr)
        if base_current is None:
            # Compare API unavailable: report distinctly from "up-to-date" so
            # a GitHub compare-API degradation is visible in telemetry instead
            # of masquerading as every open PR being current.
            return [
                {
                    "pr_number": pr_number,
                    "head_ref": head,
                    "updated": False,
                    "skipped_reason": "compare_unavailable",
                }
            ]
        if not self._should_update_pr_branch(pr, base_current):
            return [
                {
                    "pr_number": pr_number,
                    "head_ref": head,
                    "updated": False,
                    "skipped_reason": "up-to-date",
                }
            ]

        old_head = pr.get("headRefOid")
        if not self.gh.pr_update_branch(pr_number):
            return [
                {
                    "pr_number": pr_number,
                    "head_ref": head,
                    "updated": False,
                    "error": "pr_update_branch failed",
                }
            ]

        new_head = self._verify_synced_head(pr_number, old_head)
        if new_head is None:
            return [
                {
                    "pr_number": pr_number,
                    "head_ref": head,
                    "updated": False,
                    "error": "post-sync head verification failed",
                }
            ]
        if new_head == old_head:
            return [
                {
                    "pr_number": pr_number,
                    "head_ref": head,
                    "updated": False,
                    "skipped_reason": "up-to-date",
                }
            ]

        # Issue #1072: capture the bool return so telemetry does not
        # report success for a carry-forward write that was refused
        # (e.g. _update_approval_head's identity guard rejected a
        # concurrent verdict change). The branch WAS updated, but the
        # approval verdict was not carried forward to the new head —
        # the PR will need re-review. Telemetry-only, no state impact.
        approval_carried = self._update_approval_head(
            pr_number,
            decision,
            new_head,
            old_head=old_head,
            issue_number=_wf.linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=self.config.dispatch.branch_prefix,
            ),
        )
        return [
            {
                "pr_number": pr_number,
                "head_ref": head,
                "updated": True,
                "new_head": new_head,
                "approval_carry_forward": approval_carried,
            }
        ]

    # mode == "broadcast": update every eligible PR.
    try:
        prs = self.gh.pr_list()
    except GitHubError as exc:
        # Report the pr_list failure as a value instead of raising
        return [{"error": f"pr_list failed: {exc}"}]
    branch_prefix = self.config.dispatch.branch_prefix
    required_checks = self.config.auto_merge.required_checks

    for pr in prs:
        pr_number = int(pr.get("number", 0))
        if pr_number == merged_pr_number:
            continue

        # Skip fork PRs
        if pr.get("isCrossRepository"):
            continue

        # Only update PRs with the configured branch prefix
        head = str(pr.get("headRefName") or "")
        if not head.startswith(branch_prefix):
            continue

        # Derive eligibility from the recorded review decision. Never
        # update-branch a PR whose current decision is request_changes,
        # escalated, or blocked — rework or human intervention will replace
        # the head, so the CI run would be guaranteed-wasted time.
        decision = self._review_decision(pr_number)
        decision_value = decision.get("decision")
        if decision_value in {"request_changes", "blocked"} or decision.get("escalated"):
            results.append(
                {
                    "pr_number": pr_number,
                    "head_ref": head,
                    "updated": False,
                    "skipped_reason": "not_approved",
                }
            )
            continue

        # Skip approved-pending-ship PRs to avoid invalidating their approvals.
        # These will get base-updated when they themselves are merged (GitHub
        # merges handle base freshness) or by a later pass after they merge.
        if decision_value == "approved":
            reviewed_head_sha = decision.get("reviewed_head_sha")
            live_head_sha = pr.get("headRefOid")
            if reviewed_head_sha is not None and live_head_sha == reviewed_head_sha:
                # PR is approved and head hasn't moved since approval — skip update
                results.append(
                    {
                        "pr_number": pr_number,
                        "head_ref": head,
                        "updated": False,
                        "skipped_reason": "approved-pending-ship",
                    }
                )
                continue

        # Skip PRs with required checks in PENDING/IN_PROGRESS to avoid cancelling in-flight CI
        # This prevents the wedge described in issue #209 where update-branch cancels
        # matrix jobs and aggregate-gate checks permanently fail against the frozen CANCELLED state.
        status_rollup = pr.get("statusCheckRollup")
        if status_rollup and required_checks:
            # statusCheckRollup is a flat array of check objects (CheckRun or StatusContext)
            # CheckRun uses 'status' field, StatusContext uses 'state' field
            has_pending_required = False
            for check in status_rollup:
                check_name = check.get("name") or check.get("context")
                if check_name in required_checks:
                    # Check if this required check is in a pending/in-progress state
                    # For CheckRuns: status != COMPLETED means in-flight
                    # For StatusContext: state != SUCCESS/FAILURE/ERROR means in-flight
                    status = check.get("status") or check.get("state", "")
                    # Treat any non-terminal status as in-flight (safer than enumerating)
                    # Terminal states: COMPLETED (CheckRun), SUCCESS/FAILURE/ERROR (StatusContext)
                    if status.upper() != "COMPLETED" and status.upper() not in {
                        "SUCCESS",
                        "FAILURE",
                        "ERROR",
                    }:
                        has_pending_required = True
                        break

            if has_pending_required:
                results.append(
                    {
                        "pr_number": pr_number,
                        "head_ref": head,
                        "updated": False,
                        "skipped_reason": "pending-required-checks",
                    }
                )
                continue

        # Whether base freshness is required at all is derived from GitHub
        # branch protection, cached per orchestrator pass (issue #812) --
        # see _is_base_freshness_required for the fail-closed contract.
        # When not required, skip the compare-API read and the
        # pr_update_branch write entirely: this is the update_open_prs
        # half of the churn issue #812 eliminates (merge_ready's own sync
        # block is the other half).
        base_ref = pr.get("baseRefName") or self.config.runners.default_branch
        if not self._is_base_freshness_required(base_ref):
            results.append(
                {
                    "pr_number": pr_number,
                    "head_ref": head,
                    "updated": False,
                    "skipped_reason": "base_freshness_not_required",
                }
            )
            continue

        # Use the same compare-derived base-current signal as the front-of-train
        # path and merge_ready so broadcast mode also skips up-to-date PRs and
        # syncs stale ones even when mergeStateStatus is CLEAN.
        base_current = self._is_base_current(pr)
        if base_current is None:
            # Compare API unavailable: report distinctly from "up-to-date" so
            # a GitHub compare-API degradation is visible in telemetry instead
            # of masquerading as every open PR being current.
            results.append(
                {
                    "pr_number": pr_number,
                    "head_ref": head,
                    "updated": False,
                    "skipped_reason": "compare_unavailable",
                }
            )
            continue
        if not self._should_update_pr_branch(pr, base_current):
            results.append(
                {
                    "pr_number": pr_number,
                    "head_ref": head,
                    "updated": False,
                    "skipped_reason": "up-to-date",
                }
            )
            continue

        # Attempt to update the branch
        success = self.gh.pr_update_branch(pr_number)
        results.append(
            {
                "pr_number": pr_number,
                "head_ref": head,
                "updated": success,
            }
        )

    return results
