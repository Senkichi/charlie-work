"""Alarm text for the merge path's accounting stage (moved out of ``decide.py``).

``decide.py`` is held under the repo's 800-line module cap
(``tests/test_file_size_ratchet.py``); these two helpers are pure string
builders over the accounting facts, so they live beside it rather than in the
stage file. The bodies are unchanged from where they were defined.
"""

from __future__ import annotations

from typing import Any, Mapping

from .model import AccountingFacts, EffectResults, Readiness
from .rules import format_merge_attempt_alarm_message


def route_detail(
    issue_number: int | None,
    *,
    routed: bool,
    escalated: bool,
    label_error: Mapping[str, Any] | None,
    issue_status_after: str | None,
) -> str:
    if issue_number is None:
        return "no linked issue, cannot route to rework"
    if escalated:
        return "conflict-rework cap exhausted; escalated to a human"
    if routed:
        if label_error:
            outcome = label_error.get("outcome", label_error)
            return f"rework dispatch attempted (label update failed: {outcome})"
        return "rework dispatched"
    if issue_status_after == "rework_requested":
        return "rework already requested"
    return "rework not routed"


def alarm_warning(
    readiness: Readiness,
    results: EffectResults,
    facts: AccountingFacts,
    attempts: int,
    handoff_failed: bool,
) -> str:
    """Alarm text; priority: conflict > cross-PR revert > hand-off > failed checks > generic."""
    summary = readiness.summary
    pr_number = facts.pr_number
    issue_number = facts.issue_number
    pass_str = "pass" if attempts == 1 else "passes"
    lead = f"PR #{pr_number} approved but unmergeable for {attempts} {pass_str}: "
    if readiness.branch.merge_conflict:
        detail = route_detail(
            issue_number,
            routed=results.conflict_routed,
            escalated=results.conflict_escalated,
            label_error=results.rework_label_error,
            issue_status_after=results.issue_status_after,
        )
        return f"{lead}merge conflict — {detail}"
    if readiness.cross_pr_revert_detected:
        detail = route_detail(
            issue_number,
            routed=results.cross_pr_revert_routed,
            escalated=False,
            label_error=results.rework_label_error,
            issue_status_after=results.issue_status_after,
        )
        return f"{lead}cross-PR revert — {detail}"
    if handoff_failed:
        return (
            f"PR #{pr_number} approved and checks green but the mergequeue "
            f"label {facts.config.mergequeue_label!r} failed to "
            f"apply for {attempts} {pass_str} — never handed off to "
            "Aviator"
        )
    if summary.failed:
        detail = route_detail(
            issue_number,
            routed=results.check_failure_routed,
            escalated=False,
            label_error=results.rework_label_error,
            issue_status_after=results.issue_status_after,
        )
        failed_str = ", ".join(summary.failed)
        return f"{lead}required check(s) failed ({failed_str}) — {detail}"
    return format_merge_attempt_alarm_message(
        pr_number,
        attempts,
        summary,
        mergeable=facts.mergeable,
        merge_state_status=facts.merge_state_status,
    )
