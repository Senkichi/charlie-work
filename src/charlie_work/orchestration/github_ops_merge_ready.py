"""Merge-ready dry-run command delegate moved out of ``OrchestratorApp``.

Track 2 Phase B, L04 batch 2 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Body relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches the top-level ``def``
unwrapped onto ``OrchestratorApp``. ``CommandResult``, ``load_state_locked``,
``linked_issue_number`` and ``detect_cross_pr_revert`` are reached through
``_wf.``: ``CommandResult`` is a ``charlie_work.workflow`` module-level class,
and the other three are patched on ``charlie_work.workflow`` by the suite
(Tier D -- ``load_state_locked`` string form, ``linked_issue_number`` via the
``workflow_mod`` alias, ``detect_cross_pr_revert`` via the ``workflow_module``
alias). All other free names are imported directly (no test patches them on
``charlie_work.workflow``).
"""

from __future__ import annotations

import charlie_work.workflow as _wf
from dataclasses import asdict
from typing import Any
from charlie_work.checks import summarize_checks
from charlie_work.cross_pr_revert import CrossPrRevertStatus
from charlie_work.escalation import _escalation_flags
from charlie_work.github import GitHubError, label_names
from charlie_work.janitor import check_operator_containment


def _merge_ready_dry_run(
    self,
    pr_number: int,
    *,
    merge: bool | None = None,
    merge_train_head: int | None = None,
) -> _wf.CommandResult:
    """Dry-run readiness evaluation for ``merge_ready`` (issue #614).

    Mirrors the read-only prefix of :meth:`merge_ready` — fetching the PR,
    review decision, checks, diff, and computing ``can_merge`` — but skips
    every state write, label transition, branch update, merge, and
    mergequeue label add.  Returns the computed readiness verdict so a
    preview can report what *would* happen without fabricating a merge or
    stranding the PR in a terminal state.

    The gate in :meth:`merge_ready` sits above its ``state_lock`` (the
    idempotency short-circuit conditionally writes) and touches nothing,
    mirroring ``_dispatch_impl``'s dry-run gate.  Gating lower would drop
    into the shared failed-attempt-alarm block and increment
    ``consecutive_failed_merge_attempts``, advancing the state machine
    instead of merely previewing it.
    """
    # Read-only state snapshot.  Dry-run never writes, but the locked
    # read helper is still required: ``load_state_locked`` is the single
    # point of enforcement for read-only ``load_state`` calls outside an
    # explicit ``state_lock`` block (issue #310, enforced by
    # ``test_no_unlocked_load_state_in_production_code``).  Holding the
    # advisory lock during the read also prevents a concurrent writer's
    # tmp+replace from racing this read.  The idempotency short-circuit's
    # conditional merge_alert clear is a write and is skipped; the verdict
    # is still accurate because "already merged" is a read-only fact.
    state = _wf.load_state_locked(self.paths.state_file)
    existing_pr_state = state["prs"].get(str(pr_number), {})
    if existing_pr_state.get("status") == "merged":
        return _wf.CommandResult(
            True,
            f"PR #{pr_number} already merged",
            {
                "pr": pr_number,
                "issue": existing_pr_state.get("issue_number"),
                "already_merged": True,
                "merged": True,
                "dry_run": True,
            },
        )

    pr = self.gh.pr_view(pr_number)
    if not pr:
        return _wf.CommandResult(False, f"PR #{pr_number} was not found", {})

    issue_number = _wf.linked_issue_number(
        pr,
        is_cross_repository=pr.get("isCrossRepository"),
        branch_prefix=self.config.dispatch.branch_prefix,
        branch_issue_validator=self._make_branch_issue_validator(),
    )
    decision = self._review_decision(pr_number)
    approved = decision.get("decision") == "approved"
    sync_failed = False
    merge_conflict = False
    cross_pr_revert_detected = False
    cross_pr_revert_undetermined = False
    cross_pr_revert_reason: str | None = None

    if approved:
        reviewed_head_sha = decision.get("reviewed_head_sha")
        live_head_sha = pr.get("headRefOid")
        head_moved = reviewed_head_sha is None or live_head_sha != reviewed_head_sha
        if head_moved and live_head_sha:
            check = self._check_carry_forward(pr_number, decision)
            if not check.carry_forward:
                # Under dry-run the head-moved re-review transition
                # (state write + label transition) is skipped; return the
                # verdict without persisting.
                return _wf.CommandResult(
                    False,
                    "PR head moved since approval — re-review required",
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "can_merge": False,
                        "merged": False,
                        "head_moved": True,
                        "reviewed_head_sha": reviewed_head_sha,
                        "live_head_sha": live_head_sha,
                        "review_decision": decision,
                        "dry_run": True,
                    },
                )
            # Carry-forward would apply; continue with read-only checks.
            # The approval-head write (_update_approval_head) is skipped
            # under dry-run, so reviewed_head_sha stays stale — but
            # ``approved`` is still True and ``can_merge`` depends on
            # checks/sync_failed, not on the head SHA directly.

        # Genuine merge conflict detection (read-only).
        if self._is_merge_conflict(pr):
            merge_conflict = True
            sync_failed = True

        update_branch_strategy = self.config.auto_merge.update_branch_strategy
        if update_branch_strategy == "front_of_train":
            if merge_train_head is not None and merge_train_head != pr_number:
                return self._merge_not_ready_result(
                    pr_number, issue_number, decision, existing_pr_state
                )
            if merge_train_head is None:
                try:
                    prs = self.gh.pr_list()
                except GitHubError:
                    sync_failed = True
                else:
                    head = self._merge_train_head(prs)
                    if head is not None and head != pr_number:
                        return self._merge_not_ready_result(
                            pr_number, issue_number, decision, existing_pr_state
                        )

        # Base freshness check (read-only). The ``pr_update_branch`` WRITE
        # is skipped under dry-run; if the base is stale, report the
        # deferral without persisting the counter increment or event.
        if not sync_failed and update_branch_strategy != "off":
            base_currency_gated = self._is_base_currency_gated(
                pr.get("baseRefName") or self.config.runners.default_branch
            )
            if base_currency_gated and update_branch_strategy in {
                "front_of_train",
                "broadcast",
            }:
                base_current = self._is_base_current(pr)
                if base_current is not True:
                    return _wf.CommandResult(
                        True,
                        f"PR #{pr_number} base is stale; merge deferred until base is current",
                        {
                            "pr": pr_number,
                            "issue": issue_number,
                            "can_merge": False,
                            "auto_merge_enabled": self.config.auto_merge.enabled,
                            "merged": False,
                            "merge_output": None,
                            "branch_deleted": None,
                            "review_decision": decision,
                            "checks": asdict(
                                summarize_checks([], self.config.auto_merge.required_checks)
                            ),
                            "checks_unavailable": False,
                            "label_error": None,
                            "update_open_prs_results": None,
                            "cancel_superseded_runs_results": None,
                            "containment_warnings": [],
                            "stale_base": True,
                            "consecutive_failed_merge_attempts": (
                                existing_pr_state.get("consecutive_failed_merge_attempts", 0)
                            ),
                            "consecutive_stale_base_deferrals": (
                                existing_pr_state.get("consecutive_stale_base_deferrals", 0)
                            ),
                            "merge_attempt_alarm": False,
                            "merge_attempt_warning": None,
                            "merge_conflict": False,
                            "dry_run": True,
                        },
                    )

        # Cross-PR revert detection (read-only). The rework routing write
        # is skipped under dry-run. UNDETERMINED fails closed (issue #1068):
        # the merge is held but no rework is routed. ``blocks_merge`` is the
        # single enforcement point for sync_failed; the status chain is
        # routing/dispatch only (matches the merge_ready path above).
        if not sync_failed:
            cross_pr_revert_verdict = _wf.detect_cross_pr_revert(pr, self.repo_root)
            cross_pr_revert_reason = cross_pr_revert_verdict.reason
            if cross_pr_revert_verdict.blocks_merge:
                sync_failed = True
                if cross_pr_revert_verdict.status is CrossPrRevertStatus.REVERT_DETECTED:
                    cross_pr_revert_detected = True
                elif cross_pr_revert_verdict.status is CrossPrRevertStatus.UNDETERMINED:
                    cross_pr_revert_undetermined = True

    checks = self.gh.pr_checks(pr_number)
    checks_unavailable = checks is None

    if checks_unavailable:
        summary = summarize_checks(None, self.config.auto_merge.required_checks)
        enriched_checks: list[dict[str, Any]] = []
    else:
        # Issue #1383: shared data-boundary enrichment, same as the real
        # merge_ready path -- see _enrich_checks_infra_blocked.
        enriched_checks = self._enrich_checks_infra_blocked(
            checks, self.config.auto_merge.required_checks
        )
        summary = summarize_checks(enriched_checks, self.config.auto_merge.required_checks)

    diff = self.gh.pr_diff(pr_number)
    containment_warnings = check_operator_containment(self.repo_root, diff, pr_number)

    # Issue #1060: mirror the real path's dict-based gate so the two
    # duplicated gates cannot drift -- a future condition added to one but
    # not the other would silently diverge the dry-run preview from the
    # real verdict.
    merge_gate_inputs = {
        "summary_ready": summary.ready,
        "approved": approved,
        "require_approved_review": self.config.auto_merge.require_approved_review,
        "sync_failed": sync_failed,
    }
    can_merge = (
        merge_gate_inputs["summary_ready"]
        and (merge_gate_inputs["approved"] or not merge_gate_inputs["require_approved_review"])
        and not merge_gate_inputs["sync_failed"]
    )
    # Issue #840: mirror the real path's escalation gate so the dry-run
    # preview accurately reports "would be held" instead of "would merge"
    # when the PR or its linked issue is escalated. Reuses the read-only
    # ``state`` snapshot loaded at the top of this dry-run (no mid-call
    # writes happen in dry-run, unlike the real path's re-read).
    _pr_escalated, _issue_escalated = _escalation_flags(
        existing_pr_state,
        state.get("issues", {}).get(str(issue_number), {}) if issue_number is not None else None,
    )
    escalated_merge_hold = can_merge and (_pr_escalated or _issue_escalated)
    should_merge = self.config.auto_merge.enabled if merge is None else merge
    mergequeue_label = self.config.auto_merge.mergequeue_label

    # Issue #1598: mirror the real path's human-merge-labels gate so the
    # dry-run preview accurately reports "would be held for human merge"
    # instead of "would merge" when the bound issue carries a configured
    # human_merge_labels label. Detection is shared with ``merge_ready``
    # via ``_human_merge_hold_check`` so the two paths cannot drift.
    # Read-only (no de-escalation writes in dry-run).
    human_merge_hold, human_merge_check_unavailable = self._human_merge_hold_check(issue_number)

    # Read-only merge-hold check (same condition as the real path).
    merge_hold = False
    merge_hold_check_unavailable = False
    if (
        can_merge
        and should_merge
        and not escalated_merge_hold
        and not human_merge_hold
        and not human_merge_check_unavailable
    ):
        merge_hold = self.config.labels.merge_hold in label_names(pr)
        if not merge_hold and issue_number is not None:
            try:
                issue = self.gh.issue_view(issue_number)
            except (GitHubError, ValueError):
                merge_hold_check_unavailable = True
                issue = None
            if not merge_hold_check_unavailable and (
                not isinstance(issue, dict) or "labels" not in issue
            ):
                merge_hold_check_unavailable = True
                issue = None
            if not merge_hold_check_unavailable:
                issue_labels = label_names(issue) if issue else set()
                merge_hold = self.config.labels.merge_hold in issue_labels

    # Describe what *would* happen so the preview is actionable.
    message = "dry-run: merge readiness evaluated"
    if (
        can_merge
        and should_merge
        and not merge_hold
        and not escalated_merge_hold
        and not human_merge_hold
        and not human_merge_check_unavailable
    ):
        if mergequeue_label:
            message += f" (would hand off to mergequeue label {mergequeue_label!r})"
        else:
            message += " (would merge)"
    elif escalated_merge_hold:
        message += " (escalated — would hold merge while agent:human-needed is up)"
    elif human_merge_hold:
        message += " (human-merge label on issue — would not auto-merge)"
    elif human_merge_check_unavailable:
        message += f" (human-merge label check unavailable for issue #{issue_number})"
    elif merge_hold:
        message += (
            f" (merge-hold label {self.config.labels.merge_hold!r} present — would be left alone)"
        )
    elif merge_conflict:
        message += " (merge conflict — would route to rework on threshold)"
    elif cross_pr_revert_detected:
        message += f" (cross-PR revert: {cross_pr_revert_reason})"
    elif cross_pr_revert_undetermined:
        message += (
            f" (cross-PR revert gate undetermined — would hold merge, fail-closed: "
            f"{cross_pr_revert_reason})"
        )
    elif checks_unavailable:
        message = "dry-run: checks unavailable (gh failure)"
    elif merge_hold_check_unavailable:
        message += f" (merge-hold check unavailable for issue #{issue_number})"

    return _wf.CommandResult(
        not (checks_unavailable or merge_hold_check_unavailable or human_merge_check_unavailable),
        message,
        {
            "pr": pr_number,
            "issue": issue_number,
            "can_merge": can_merge,
            "auto_merge_enabled": self.config.auto_merge.enabled,
            "merged": False,
            "merge_output": None,
            "branch_deleted": None,
            "review_decision": decision,
            "checks": asdict(summary),
            "checks_unavailable": checks_unavailable,
            "label_error": None,
            "update_open_prs_results": None,
            "cancel_superseded_runs_results": None,
            "containment_warnings": list(containment_warnings),
            "consecutive_failed_merge_attempts": existing_pr_state.get(
                "consecutive_failed_merge_attempts", 0
            ),
            "consecutive_stale_base_deferrals": existing_pr_state.get(
                "consecutive_stale_base_deferrals", 0
            ),
            "merge_attempt_alarm": False,
            "merge_attempt_warning": None,
            "merge_conflict": merge_conflict,
            "cross_pr_revert_detected": cross_pr_revert_detected,
            "cross_pr_revert_reason": cross_pr_revert_reason,
            "cross_pr_revert_routed": False,
            "cross_pr_revert_undetermined": cross_pr_revert_undetermined,
            "mergequeue_label_applied": None,
            "merge_hold": merge_hold,
            "merge_hold_check_unavailable": merge_hold_check_unavailable,
            "human_merge_hold": human_merge_hold,
            "human_merge_check_unavailable": human_merge_check_unavailable,
            "escalated_merge_hold": escalated_merge_hold,
            # Issue #1060: surface the gate inputs in the dry-run preview
            # too, for diagnostic parity with the persisted event.
            **merge_gate_inputs,
            "dry_run": True,
        },
    )
