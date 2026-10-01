"""Renderers of the **Merge path**: plan values in, ``CommandResult`` out.

Pure apart from building the result through ``MergePathPorts.command_result``
(the workflow module's ``CommandResult``). Message rules are tables here, not
decisions: the order of the suffix rules is rendering, never a verdict. Two
modes keep the payload shapes the legacy bodies produced -- live results carry
the effect outcomes, the preview's carry ``"dry_run": True`` and report effects
it never ran as ``None`` / ``False``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..checks import CheckSummary, summarize_checks
from .model import (
    Accounting,
    EffectResults,
    MergePathConfig,
    MergePlan,
    Readiness,
)
from .ports import MergePathPorts

HEAD_MOVED_MESSAGE = "PR head moved since approval — re-review required"


def _empty_checks(cfg: MergePathConfig) -> dict[str, Any]:
    return asdict(summarize_checks([], list(cfg.required_checks)))


# --------------------------------------------------------------------------- #
# Shared by live and preview
# --------------------------------------------------------------------------- #


def render_not_found(ports: MergePathPorts, pr_number: int) -> Any:
    return ports.command_result(False, f"PR #{pr_number} was not found", {})


def render_train_not_head(
    ports: MergePathPorts,
    cfg: MergePathConfig,
    pr_number: int,
    issue_number: int | None,
    decision: Mapping[str, Any],
    failed_attempts: int,
) -> Any:
    """Approved PR that is not the head of the merge-train queue (no ``dry_run`` key)."""
    return ports.command_result(
        True,
        f"PR #{pr_number} is not the head of the merge-train queue",
        {
            "pr": pr_number,
            "issue": issue_number,
            "can_merge": False,
            "auto_merge_enabled": cfg.auto_merge_enabled,
            "merged": False,
            "merge_output": None,
            "branch_deleted": None,
            "review_decision": decision,
            "checks": _empty_checks(cfg),
            "checks_unavailable": False,
            "label_error": None,
            "update_open_prs_results": None,
            "cancel_superseded_runs_results": None,
            "containment_warnings": [],
            "consecutive_failed_merge_attempts": failed_attempts,
            "merge_attempt_alarm": False,
            "merge_attempt_warning": None,
            "merge_conflict": False,
        },
    )


# --------------------------------------------------------------------------- #
# Preview (dry-run)
# --------------------------------------------------------------------------- #


def render_preview_skip(ports: MergePathPorts, pr_number: int, issue_number: int | None) -> Any:
    return ports.command_result(
        True,
        f"PR #{pr_number} already merged",
        {
            "pr": pr_number,
            "issue": issue_number,
            "already_merged": True,
            "merged": True,
            "dry_run": True,
        },
    )


def _head_moved_data(
    pr_number: int,
    issue_number: int | None,
    reviewed_head_sha: str | None,
    live_head_sha: str | None,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "pr": pr_number,
        "issue": issue_number,
        "can_merge": False,
        "merged": False,
        "head_moved": True,
        "reviewed_head_sha": reviewed_head_sha,
        "live_head_sha": live_head_sha,
        "review_decision": decision,
    }


def render_preview_head_moved(
    ports: MergePathPorts,
    pr_number: int,
    issue_number: int | None,
    reviewed_head_sha: str | None,
    live_head_sha: str | None,
    decision: Mapping[str, Any],
) -> Any:
    data = _head_moved_data(pr_number, issue_number, reviewed_head_sha, live_head_sha, decision)
    return ports.command_result(False, HEAD_MOVED_MESSAGE, {**data, "dry_run": True})


def render_preview_stale_base(
    ports: MergePathPorts,
    cfg: MergePathConfig,
    pr_number: int,
    issue_number: int | None,
    decision: Mapping[str, Any],
    failed_attempts: int,
    stale_base_deferrals: int,
) -> Any:
    return ports.command_result(
        True,
        f"PR #{pr_number} base is stale; merge deferred until base is current",
        {
            "pr": pr_number,
            "issue": issue_number,
            "can_merge": False,
            "auto_merge_enabled": cfg.auto_merge_enabled,
            "merged": False,
            "merge_output": None,
            "branch_deleted": None,
            "review_decision": decision,
            "checks": _empty_checks(cfg),
            "checks_unavailable": False,
            "label_error": None,
            "update_open_prs_results": None,
            "cancel_superseded_runs_results": None,
            "containment_warnings": [],
            "stale_base": True,
            "consecutive_failed_merge_attempts": failed_attempts,
            "consecutive_stale_base_deferrals": stale_base_deferrals,
            "merge_attempt_alarm": False,
            "merge_attempt_warning": None,
            "merge_conflict": False,
            "dry_run": True,
        },
    )


@dataclass(frozen=True)
class PreviewFinal:
    """What the preview observed after the stage-3 reads (nothing was applied)."""

    pr_number: int
    issue_number: int | None
    decision: Mapping[str, Any]
    readiness: Readiness
    cfg: MergePathConfig
    should_merge: bool
    escalated_merge_hold: bool
    merge_hold: bool
    merge_hold_unavailable: bool
    merge_hold_label: str
    failed_attempts: int
    stale_base_deferrals: int


def preview_message(p: PreviewFinal) -> str:
    """Suffix rules, first match wins; (9) checks-unavailable replaces the whole message."""
    r = p.readiness
    can_merge = r.gate.can_merge
    base = "dry-run: merge readiness evaluated"
    if (
        can_merge
        and p.should_merge
        and not p.merge_hold
        and not p.escalated_merge_hold
        and not r.human_merge_hold
        and not r.human_merge_check_unavailable
    ):
        label = p.cfg.mergequeue_label
        if label:
            return base + f" (would hand off to mergequeue label {label!r})"
        return base + " (would merge)"
    if p.escalated_merge_hold:
        return base + " (escalated — would hold merge while agent:human-needed is up)"
    if r.human_merge_hold:
        return base + " (human-merge label on issue — would not auto-merge)"
    if r.human_merge_check_unavailable:
        return base + f" (human-merge label check unavailable for issue #{p.issue_number})"
    if p.merge_hold:
        return base + f" (merge-hold label {p.merge_hold_label!r} present — would be left alone)"
    if r.branch.merge_conflict:
        return base + " (merge conflict — would route to rework on threshold)"
    if r.cross_pr_revert_detected:
        return base + f" (cross-PR revert: {r.cross_pr_revert_reason})"
    if r.cross_pr_revert_undetermined:
        return base + (
            " (cross-PR revert gate undetermined — would hold merge, fail-closed: "
            f"{r.cross_pr_revert_reason})"
        )
    if r.checks_unavailable:
        return "dry-run: checks unavailable (gh failure)"
    if p.merge_hold_unavailable:
        return base + f" (merge-hold check unavailable for issue #{p.issue_number})"
    return base


def render_preview_final(ports: MergePathPorts, p: PreviewFinal) -> Any:
    r = p.readiness
    ok = not (r.checks_unavailable or p.merge_hold_unavailable or r.human_merge_check_unavailable)
    return ports.command_result(
        ok,
        preview_message(p),
        {
            "pr": p.pr_number,
            "issue": p.issue_number,
            "can_merge": r.gate.can_merge,
            "auto_merge_enabled": p.cfg.auto_merge_enabled,
            "merged": False,
            "merge_output": None,
            "branch_deleted": None,
            "review_decision": p.decision,
            "checks": asdict(r.summary),
            "checks_unavailable": r.checks_unavailable,
            "label_error": None,
            "update_open_prs_results": None,
            "cancel_superseded_runs_results": None,
            "containment_warnings": list(r.containment_warnings),
            "consecutive_failed_merge_attempts": p.failed_attempts,
            "consecutive_stale_base_deferrals": p.stale_base_deferrals,
            "merge_attempt_alarm": False,
            "merge_attempt_warning": None,
            "merge_conflict": r.branch.merge_conflict,
            "cross_pr_revert_detected": r.cross_pr_revert_detected,
            "cross_pr_revert_reason": r.cross_pr_revert_reason,
            "cross_pr_revert_routed": False,
            "cross_pr_revert_undetermined": r.cross_pr_revert_undetermined,
            "mergequeue_label_applied": None,
            "merge_hold": p.merge_hold,
            "merge_hold_check_unavailable": p.merge_hold_unavailable,
            "human_merge_hold": r.human_merge_hold,
            "human_merge_check_unavailable": r.human_merge_check_unavailable,
            "escalated_merge_hold": p.escalated_merge_hold,
            **r.gate.as_payload(),
            "dry_run": True,
        },
    )


# --------------------------------------------------------------------------- #
# Live
# --------------------------------------------------------------------------- #


def render_live_skip(ports: MergePathPorts, pr_number: int, issue_number: int | None) -> Any:
    return ports.command_result(
        True,
        f"PR #{pr_number} already merged",
        {
            "pr": pr_number,
            "issue": issue_number,
            "already_merged": True,
            "merged": True,
        },
    )


def render_live_head_moved(
    ports: MergePathPorts,
    pr_number: int,
    issue_number: int | None,
    reviewed_head_sha: str | None,
    live_head_sha: str | None,
    decision: Mapping[str, Any],
    *,
    label_error: Mapping[str, Any] | None,
    escalated: bool,
) -> Any:
    data = _head_moved_data(pr_number, issue_number, reviewed_head_sha, live_head_sha, decision)
    return ports.command_result(
        False,
        HEAD_MOVED_MESSAGE,
        {**data, "label_error": label_error, "escalated": escalated},
    )


def _conflict_data(
    pr_number: int,
    issue_number: int | None,
    decision: Mapping[str, Any],
    failed_attempts: int,
) -> dict[str, Any]:
    return {
        "pr": pr_number,
        "issue": issue_number,
        "can_merge": False,
        "merged": False,
        "review_decision": decision,
        "merge_conflict": True,
        "consecutive_failed_merge_attempts": failed_attempts,
        "merge_attempt_alarm": False,
        "merge_attempt_warning": None,
    }


def render_live_conflict_in_flight(
    ports: MergePathPorts,
    pr_number: int,
    issue_number: int | None,
    decision: Mapping[str, Any],
    failed_attempts: int,
) -> Any:
    return ports.command_result(
        True,
        f"PR #{pr_number} merge conflict is being resolved by a rework worker",
        _conflict_data(pr_number, issue_number, decision, failed_attempts),
    )


def render_live_conflict_blocked(
    ports: MergePathPorts,
    pr_number: int,
    issue_number: int | None,
    decision: Mapping[str, Any],
    failed_attempts: int,
    issue_status: str | None,
) -> Any:
    return ports.command_result(
        True,
        f"PR #{pr_number} merge conflict on issue #{issue_number} "
        f"awaiting human decision ({issue_status}); not rerouted",
        _conflict_data(pr_number, issue_number, decision, failed_attempts),
    )


def render_live_stall(
    ports: MergePathPorts,
    pr_number: int,
    issue_number: int | None,
    decision: Mapping[str, Any],
    summary: CheckSummary,
    *,
    label_error: Mapping[str, Any] | None,
    merge_conflict: bool,
) -> Any:
    return ports.command_result(
        True,
        f"PR #{pr_number} has not started required CI checks; rework requested",
        {
            "pr": pr_number,
            "issue": issue_number,
            "can_merge": False,
            "merged": False,
            "review_decision": decision,
            "checks": asdict(summary),
            "checks_unavailable": False,
            "label_error": label_error,
            "readiness_no_ci_stall": True,
            "merge_conflict": merge_conflict,
            "merge_attempt_alarm": False,
            "merge_attempt_warning": None,
        },
    )


@dataclass(frozen=True)
class LiveFinal:
    """Everything the live final result reports, collected by the shell."""

    pr_number: int
    issue_number: int | None
    decision: Mapping[str, Any]
    cfg: MergePathConfig
    plan: MergePlan
    results: EffectResults
    accounting: Accounting
    label_error: Mapping[str, Any] | None
    merge_hold_label: str


def live_message(f: LiveFinal) -> str:
    """Suffix rules, first match wins; the first three replace the base message."""
    r = f.plan.readiness
    res = f.results
    label = f.cfg.mergequeue_label
    if r.cross_pr_revert_detected:
        return f"cross-PR revert detected: {r.cross_pr_revert_reason}"
    if r.cross_pr_revert_undetermined:
        return (
            "cross-PR revert gate undetermined (merge blocked, fail-closed): "
            f"{r.cross_pr_revert_reason}"
        )
    if r.checks_unavailable:
        return "checks unavailable (gh failure)"
    base = "merge readiness evaluated"
    if f.plan.escalated_merge_hold:
        return base + " (escalated — merge held while agent:human-needed is up)"
    if r.human_merge_hold:
        return base + " (human-merge label on issue — fleet will not auto-merge)"
    if r.human_merge_check_unavailable:
        return base + (
            f" (human-merge label check unavailable for issue #{f.issue_number} — not merged)"
        )
    if f.plan.merge_hold_unavailable:
        return base + (
            f" (merge-hold check unavailable for issue #{f.issue_number} — not handed off to Aviator)"
        )
    if f.plan.merge_hold:
        return base + f" (merge-hold label {f.merge_hold_label!r} present — left alone)"
    if res.mergequeue_label_applied is False:
        return base + (
            f" (mergequeue label {label!r} FAILED to "
            f"apply — not handed off to Aviator; will retry, attempt {f.accounting.failed_attempts})"
        )
    if res.mergequeue_label_applied is True:
        return base + f" (handed off to mergequeue label {label!r})"
    if f.accounting.merged and f.label_error:
        return base + f" (merged; post-merge label/branch cleanup failed: {f.label_error})"
    if f.label_error:
        return base + f" (label update failed: {f.label_error.get('outcome', f.label_error)})"
    return base


def render_live_final(ports: MergePathPorts, f: LiveFinal) -> Any:
    r = f.plan.readiness
    res = f.results
    acc = f.accounting
    data = {
        "pr": f.pr_number,
        "issue": f.issue_number,
        "can_merge": r.gate.can_merge,
        "auto_merge_enabled": f.cfg.auto_merge_enabled,
        "merged": acc.merged,
        "merge_output": res.merge_output,
        "branch_deleted": res.branch_deleted,
        "review_decision": f.decision,
        "checks": asdict(r.summary),
        "checks_unavailable": r.checks_unavailable,
        "label_error": f.label_error,
        "update_open_prs_results": (
            list(res.update_open_prs_results) if res.update_open_prs_results is not None else None
        ),
        "cancel_superseded_runs_results": res.cancel_results,
        "containment_warnings": list(r.containment_warnings),
        "consecutive_failed_merge_attempts": acc.failed_attempts,
        "consecutive_stale_base_deferrals": acc.stale_base_deferrals,
        "merge_attempt_alarm": acc.alarm,
        "merge_attempt_warning": acc.warning,
        "merge_conflict": r.branch.merge_conflict,
        "cross_pr_revert_detected": r.cross_pr_revert_detected,
        "cross_pr_revert_reason": r.cross_pr_revert_reason,
        "cross_pr_revert_routed": res.cross_pr_revert_routed,
        "cross_pr_revert_undetermined": r.cross_pr_revert_undetermined,
        "mergequeue_label_applied": res.mergequeue_label_applied,
        "merge_hold": f.plan.merge_hold,
        "merge_hold_check_unavailable": f.plan.merge_hold_unavailable,
        "human_merge_hold": r.human_merge_hold,
        "human_merge_check_unavailable": r.human_merge_check_unavailable,
        "human_merge_label_error": res.human_merge_label_error,
        "escalated_merge_hold": f.plan.escalated_merge_hold,
        **r.gate.as_payload(),
    }
    ok = not (r.checks_unavailable or f.plan.merge_hold_unavailable)
    return ports.command_result(ok, live_message(f), data)
