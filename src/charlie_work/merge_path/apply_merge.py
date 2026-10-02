"""Stage 4 of the live **Merge path**: the effects ``decide_merge`` planned.

Exactly one of hand-off / self-merge / human-merge hand-off runs (the plan makes
them exclusive), then the conflict and check-failure rework routes, which are
exclusive with all three. Each effect reports back through ``EffectResults``;
nothing here decides. Every state write goes through the ``WriteGate``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from ..escalation import _escalate_issue
from ..labels import TransitionOutcome
from ..merge_finalize import _merged_issue_fields
from ..pass_deadline import pass_deadline_suspended
from ..github import cancel_superseded_runs
from ..write_gate import WriteGate, require_write_gate
from .model import EffectResults, MergePathConfig, MergePlan
from .ports import MergePathPorts

logger = logging.getLogger(__name__)

_HUMAN_MERGE_COMMENT = (
    "This PR is approved and all checks are green, but the linked "
    "issue carries a human-merge label. The fleet will not auto-merge "
    "it — a human merge is required."
)


def label_error_of(edge: str, result: Any) -> dict[str, Any] | None:
    """``None`` for an applied transition, else the failure record the result reports."""
    if result.outcome == TransitionOutcome.APPLIED:
        return None
    return {
        "edge": edge,
        "outcome": result.outcome.value,
        "add_failures": result.add_failures,
        "remove_failures": result.remove_failures,
    }


def apply_merge_plan(
    app: Any,
    ports: MergePathPorts,
    write_gate: WriteGate,
    *,
    pr_number: int,
    pr: dict[str, Any],
    decision: dict[str, Any],
    issue_number: int | None,
    cfg: MergePathConfig,
    plan: MergePlan,
    results: EffectResults,
) -> tuple[EffectResults, dict[str, Any] | None]:
    """Run the planned effects; returns the updated results and the post-merge ``label_error``."""
    write_gate = require_write_gate(write_gate)
    label_error: dict[str, Any] | None = None
    if plan.action_hand_off:
        results = _hand_off(app, ports, write_gate, pr_number, issue_number, cfg, plan, results)
    elif plan.action_merge:
        results, label_error = _self_merge(
            app, ports, write_gate, pr, pr_number, issue_number, results
        )
    if plan.action_human_merge:
        results = _human_merge(app, ports, write_gate, pr_number, issue_number, results)
    results = _route_rework(app, pr, decision, issue_number, cfg, plan, results)
    return results, label_error


def _hand_off(
    app: Any,
    ports: MergePathPorts,
    write_gate: WriteGate,
    pr_number: int,
    issue_number: int | None,
    cfg: MergePathConfig,
    plan: MergePlan,
    results: EffectResults,
) -> EffectResults:
    """Aviator hand-off: add the mergequeue label, and on success mark the PR queued (ADR-0003)."""
    applied = app.gh.add_pr_label(pr_number, cfg.mergequeue_label)
    if applied:
        state_file = app.paths.state_file
        with ports.state_lock(state_file):
            state = ports.load_state(state_file)
            update: dict[str, Any] = {
                **state["prs"].get(str(pr_number), {}),
                "number": pr_number,
                "issue_number": issue_number,
                "status": "mergequeue",
                "mergequeue_revoked_reason": None,
            }
            if plan.reset_failed_attempts_on_hand_off:
                update["consecutive_failed_merge_attempts"] = 0
            state["prs"][str(pr_number)] = update
            if issue_number is not None:
                key = str(issue_number)
                state["issues"][key] = {**state["issues"].get(key, {}), "merge_alert": "OK"}
            write_gate.save_state(state)
    return replace(results, mergequeue_label_applied=applied)


def _self_merge(
    app: Any,
    ports: MergePathPorts,
    write_gate: WriteGate,
    pr: dict[str, Any],
    pr_number: int,
    issue_number: int | None,
    results: EffectResults,
) -> tuple[EffectResults, dict[str, Any] | None]:
    auto = app.config.auto_merge
    # ``merge_pr`` raises on failure by design: the merge is irreversible and a
    # caller must see it.
    merge_output = app.gh.merge_pr(
        pr_number, auto.strategy, admin=auto.admin, merge_flags=auto.merge_flags
    )
    merged_at = ports.utc_now()
    state_file = app.paths.state_file
    with ports.state_lock(state_file):
        state = ports.load_state(state_file)
        state["prs"][str(pr_number)] = {
            **state["prs"].get(str(pr_number), {}),
            "number": pr_number,
            "issue_number": issue_number,
            "status": "merged",
            "merged": True,
            "merged_at": merged_at,
            "consecutive_failed_merge_attempts": 0,
        }
        if issue_number is not None:
            key = str(issue_number)
            state["issues"][key] = _merged_issue_fields(state["issues"].get(key, {}), issue_number)
        write_gate.save_state(state)
    results = replace(results, merge_output=merge_output, merged_at=merged_at)
    label_error: dict[str, Any] | None = None
    # Cleanup is best-effort and must finish even when the pass budget is spent.
    with pass_deadline_suspended(app.gh):
        if issue_number is not None:
            transitioned = write_gate.transition(
                app.gh,
                app.config.labels,
                issue_number,
                "merged",
                pr_number=pr_number,
                cause="merge_finalize",
            )
            label_error = label_error_of("merged", transitioned)
            app.gh.close_issue(issue_number)
        if auto.delete_branch:
            head_ref = str(pr.get("headRefName") or "")
            results = replace(
                results, branch_deleted=app.gh.delete_branch(head_ref) if head_ref else False
            )
        if auto.update_branch_strategy in {"broadcast", "front_of_train"}:
            results = replace(
                results, update_open_prs_results=tuple(app._update_open_agent_prs(pr_number))
            )
        runners = app.config.runners
        if runners.enabled and runners.cancel_superseded_main_runs:
            results = replace(
                results,
                cancel_results=cancel_superseded_runs(
                    app.gh, runners.default_branch, runners.workflow_name
                ),
            )
    return results, label_error


def _human_merge(
    app: Any,
    ports: MergePathPorts,
    write_gate: WriteGate,
    pr_number: int,
    issue_number: int | None,
    results: EffectResults,
) -> EffectResults:
    """Human-merge hand-off (issue #1598): escalate as policy, label, and comment once."""
    state_file = app.paths.state_file
    with ports.state_lock(state_file):
        state = ports.load_state(state_file)
        already_posted = bool(
            state["prs"].get(str(pr_number), {}).get("human_merge_comment_posted")
        )
        state = _escalate_issue(
            state,
            issue_number,
            reason="human_merge_required",
            reason_class="policy",
            pr_number=pr_number,
            pr_extra={"human_merge_comment_posted": True},
        )
        state = write_gate.record_event(
            state,
            "human_merge_required",  # event-consumer: audit-only -- records the human-merge hand-off (issue #1598); consumed by tests/test_human_merge_labels_1598.py.
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "comment_posted": not already_posted,
            },
        )
        write_gate.save_state(state)
    if issue_number is not None:
        transitioned = write_gate.transition(
            app.gh,
            app.config.labels,
            issue_number,
            "human_merge_required",
            pr_number=pr_number,
        )
        results = replace(
            results,
            human_merge_label_error=label_error_of("human_merge_required", transitioned),
        )
    if not already_posted and issue_number is not None:
        try:
            app._comment_pr(pr_number, _HUMAN_MERGE_COMMENT)
        except Exception:
            logging.getLogger(__name__).warning(
                "human_merge comment post failed pr=%d", pr_number, exc_info=True
            )
    return results


def _route_rework(
    app: Any,
    pr: dict[str, Any],
    decision: dict[str, Any],
    issue_number: int | None,
    cfg: MergePathConfig,
    plan: MergePlan,
    results: EffectResults,
) -> EffectResults:
    """Conflict and check-failure rework routes (exclusive with every merge action)."""
    summary = plan.readiness.summary
    if plan.conflict_rework:
        routed = app._route_janitor_gate_failure_to_rework(
            pr,
            issue_number,
            attempts_key="conflict_rework_attempts",
            max_attempts=cfg.max_conflict_rework_attempts,
            reason="merge_conflict",
            router=app._request_merge_conflict_rework,
        )
        if routed is not None:
            results = replace(
                results,
                conflict_routed=bool(routed.data.get("routed_to_rework")),
                conflict_escalated=bool(routed.data.get("escalated")),
                rework_label_error=routed.data.get("label_error"),
            )
    if plan.check_failure_rework:
        results = replace(
            results,
            check_failure_routed=True,
            rework_label_error=app._request_check_failure_rework(
                pr, issue_number, decision, summary
            ),
        )
    return results
