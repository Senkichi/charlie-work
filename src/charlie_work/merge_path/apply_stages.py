"""Stage 1-3 effects of the live **Merge path**.

Each function applies one effect a decision asked for and returns what the next
gather needs (a refetched ``Opening``, a sync outcome, a rework label error). All
state writes go through the ``WriteGate``; helpers that already own their writes
(``_update_approval_head``, the ``_request_*_rework`` routers, the infra
remediation) are called unchanged as app methods.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..escalation import _reset_linked_pr_status_to_passive_open
from ..state import PASSIVE_OPEN_STATUS, clear_escalation, clear_escalation_on_issue_prs
from ..write_gate import WriteGate, require_write_gate
from .apply_merge import label_error_of
from .decide import decide_admission
from .gather import BranchRead, Opening, regather_after_sync
from .model import BranchGate, EffectResults, Readiness, SyncOutcome
from .ports import MergePathPorts
from .render import render_live_head_moved


def carry_forward(
    app: Any, ports: MergePathPorts, write_gate: WriteGate, opening: Opening
) -> Opening:
    """Approved, head moved, patch-id/signature carried: re-stamp the approval, refetch.

    Returns the opening with the refetched PR and verdict and the admission the
    second ``decide_admission`` call (``observed_carry_forward``) answers.
    """
    write_gate = require_write_gate(write_gate)
    check = opening.carry_check
    pr_number = opening.facts.pr_number
    issue_number = opening.issue_number
    app._update_approval_head(
        pr_number,
        opening.decision,
        opening.pr.get("headRefOid"),
        old_head=opening.facts.verdict.reviewed_head_sha,
        issue_number=issue_number,
        tier=check.tier or "patch-id",
        new_patch_id=check.live_patch_id,
        new_signature=check.live_signature,
    )
    pr = app.gh.pr_view(pr_number) or opening.pr
    decision = app._review_decision(pr_number)
    state_file = app.paths.state_file
    with ports.state_lock(state_file):
        state = ports.load_state(state_file)
        state["prs"][str(pr_number)] = {
            **state["prs"].get(str(pr_number), {}),
            "number": pr_number,
            "issue_number": issue_number,
            "status": "approved",
            "head_moved": False,
            "reviewed_head_sha": decision.get("reviewed_head_sha"),
            "reviewed_patch_id": check.live_patch_id,
            "carry_forward_tier": check.tier,
            "carried_forward_from": decision.get("carried_forward_from", []),
            "live_head_sha": pr.get("headRefOid"),
            "consecutive_failed_merge_attempts": 0,
            "consecutive_stale_base_deferrals": 0,
        }
        write_gate.save_state(state)
    admission = decide_admission(opening.facts, observed_carry_forward=True)
    return replace(opening, pr=pr, decision=decision, admission=admission)


def re_review(app: Any, ports: MergePathPorts, write_gate: WriteGate, opening: Opening) -> Any:
    """Head moved since approval and not carried: stamp the PR for re-review, report it."""
    write_gate = require_write_gate(write_gate)
    adm = opening.admission
    facts = opening.facts
    pr_number = facts.pr_number
    issue_number = opening.issue_number
    reviewed = facts.verdict.reviewed_head_sha
    live = facts.live_head_sha
    label_error: dict[str, Any] | None = None
    if adm.transition_review_started:
        label_error = label_error_of(
            "review_started",
            write_gate.transition(app.gh, app.config.labels, issue_number, "review_started"),
        )
    if adm.record_head_moved:
        state_file = app.paths.state_file
        with ports.state_lock(state_file):
            state = ports.load_state(state_file)
            state["prs"][str(pr_number)] = {
                **state["prs"].get(str(pr_number), {}),
                "number": pr_number,
                "issue_number": issue_number,
                **({"status": "reviewing"} if adm.stamp_reviewing else {}),
                "head_moved": True,
                "reviewed_head_sha": reviewed,
                "live_head_sha": live,
                "consecutive_failed_merge_attempts": 0,
                "consecutive_stale_base_deferrals": 0,
            }
            if issue_number is not None:
                key = str(issue_number)
                state["issues"][key] = {**state["issues"].get(key, {}), "merge_alert": "OK"}
            state = write_gate.record_event(
                state,
                "head_moved",
                {"pr_number": pr_number, "reviewed_head_sha": reviewed, "live_head_sha": live},
            )
            write_gate.save_state(state)
    return render_live_head_moved(
        ports,
        pr_number,
        issue_number,
        reviewed,
        live,
        opening.decision,
        label_error=label_error,
        escalated=adm.escalated,
    )


def sync_branch(app: Any, read: BranchRead, opening: Opening) -> tuple[BranchGate, Opening]:
    """Apply the branch sync the gate requested; re-run the branch decision with its outcome."""
    gate = read.gate
    if not gate.request_sync:
        return gate, opening
    pr = opening.pr
    pr_number = opening.facts.pr_number
    live_head = pr.get("headRefOid")
    pr_after = pr
    decision = opening.decision
    outcome = SyncOutcome.FAILED
    if app.gh.pr_update_branch(pr_number):
        new_head = app._verify_synced_head(pr_number, live_head)
        if new_head is None:
            outcome = SyncOutcome.FAILED
        elif new_head != live_head:
            app._update_approval_head(
                pr_number,
                decision,
                new_head,
                old_head=live_head,
                issue_number=opening.issue_number,
            )
            pr_after = app.gh.pr_view(pr_number) or pr
            decision = app._review_decision(pr_number)
            outcome = SyncOutcome.NEW_HEAD
        else:
            outcome = SyncOutcome.SAME_HEAD
    gate = regather_after_sync(app, read, pr_before=pr, pr_after=pr_after, outcome=outcome)
    return gate, replace(opening, pr=pr_after, decision=decision)


def route_cross_pr_revert(
    app: Any, opening: Opening, readiness: Readiness, results: EffectResults
) -> EffectResults:
    """Cross-PR revert detected on a bound, not-yet-routed issue: request rework."""
    if not readiness.route_cross_pr_revert:
        return results
    return replace(
        results,
        cross_pr_revert_routed=True,
        rework_label_error=app._request_cross_pr_revert_rework(
            opening.pr, opening.issue_number, opening.decision, readiness.cross_pr_revert_reason
        ),
    )


def record_containment(
    app: Any,
    ports: MergePathPorts,
    write_gate: WriteGate,
    pr_number: int,
    warnings: tuple[str, ...],
) -> None:
    """Report-only audit event for operator-containment warnings in the diff."""
    write_gate = require_write_gate(write_gate)
    if not warnings:
        return
    state_file = app.paths.state_file
    with ports.state_lock(state_file):
        state = ports.load_state(state_file)
        state = write_gate.record_event(
            state,
            "containment_check",  # event-consumer: audit-only -- report-only per issue directive, not a blocking gate
            {"pr_number": pr_number, "warnings": list(warnings)},
        )
        write_gate.save_state(state)


def deescalate_human_merge(
    app: Any,
    ports: MergePathPorts,
    write_gate: WriteGate,
    pr_number: int,
    issue_number: int,
) -> None:
    """The human-merge label is gone: lift the policy escalation it caused (issue #1598).

    The decision saw a snapshot; the write re-checks under the lock, so a
    concurrent writer that already changed the entry wins.
    """
    write_gate = require_write_gate(write_gate)
    state_file = app.paths.state_file
    key = str(issue_number)
    with ports.state_lock(state_file):
        state = ports.load_state(state_file)
        entry = state["issues"].get(key, {})
        if not (
            isinstance(entry, dict)
            and entry.get("status") == "escalated"
            and entry.get("reason_class") == "policy"
        ):
            return
        entry["status"] = PASSIVE_OPEN_STATUS
        clear_escalation(entry)
        entry.pop("label_error", None)
        state["issues"][key] = entry
        clear_escalation_on_issue_prs(state, issue_number)
        _reset_linked_pr_status_to_passive_open(state, pr_number)
        state = write_gate.record_event(
            state,
            "human_merge_label_removed",  # event-consumer: audit-only -- records the policy de-escalation when an operator removes a human-merge label (issue #1598); consumed by tests/test_human_merge_labels_1598.py.
            {"pr_number": pr_number, "issue_number": issue_number},
        )
        write_gate.save_state(state)
