"""Live driver of the **Merge path**: gather -> decide -> apply, stage by stage.

``run_merge_ready`` is ``OrchestratorApp.merge_ready`` minus the dry-run gate
(which stays in the facade, above any lock). It runs the same five stages the
dry-run preview runs, but applies each stage's effects before the next gather:

    admission -> branch -> readiness -> merge plan -> accounting

Two stages are decided twice because their own effect changes a fact: admission
after the approval head is carried forward, the branch gate after the branch is
synced. Every state write goes through the ``WriteGate`` (Convention B).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..merge_finalize import _merged_issue_fields
from ..write_gate import WriteGate, require_write_gate
from .apply_accounting import settle_accounting
from .apply_merge import apply_merge_plan
from .apply_stages import (
    carry_forward,
    deescalate_human_merge,
    record_containment,
    re_review,
    route_cross_pr_revert,
    sync_branch,
)
from .decide import (
    decide_merge,
    merge_hold_read_needed,
)
from .gather import (
    BranchRead,
    Opening,
    config_slice,
    decide_readiness_lazily,
    escalation_from,
    gather_branch,
    gather_checks,
    gather_merge_hold,
    gather_opening,
    gather_revert,
    gather_skip,
    holds_from_snapshot,
    proceed,
    readiness_facts,
)
from .readiness_gates import decide_revert, deescalation_read_needed, with_human_merge
from .model import BranchGate, BranchStop, EffectResults, MergePathConfig, PlanKind
from .ports import MergePathPorts, ports_from_workflow
from .render import (
    LiveFinal,
    render_live_conflict_blocked,
    render_live_conflict_in_flight,
    render_live_final,
    render_live_head_moved,
    render_live_skip,
    render_live_stall,
    render_not_found,
    render_train_not_head,
)


def run_merge_ready(
    app: Any,
    pr_number: int,
    *,
    merge: bool | None = None,
    merge_train_head: int | None = None,
    write_gate: WriteGate,
    ports: MergePathPorts | None = None,
) -> Any:
    """Evaluate one PR and act on the verdict (merge, hand off, route rework, or hold)."""
    write_gate = require_write_gate(write_gate)
    ports = ports or ports_from_workflow()
    cfg = config_slice(app.config)
    state_file = app.paths.state_file

    with ports.state_lock(state_file):
        state = ports.load_state(state_file)
        entry = dict(state["prs"].get(str(pr_number), {}))
        if gather_skip(pr_number, cfg, entry) is not None:
            _converge_merged_issue(write_gate, state, entry)
            return render_live_skip(ports, pr_number, entry.get("issue_number"))

    opening = gather_opening(
        app,
        ports,
        pr_number,
        cfg,
        entry,
        escalation=lambda issue: escalation_from(
            ports.load_state_locked(state_file), pr_number, issue
        ),
    )
    if opening.admission.kind is PlanKind.NOT_FOUND:
        return render_not_found(ports, pr_number)
    if opening.admission.carry_forward_needed:
        carried = carry_forward(app, ports, write_gate, opening)
        if carried is None:
            return _approval_refused(ports, opening)
        opening = carried
    if opening.admission.kind is PlanKind.REQUEST_REWORK:
        return re_review(app, ports, write_gate, opening)

    read = gather_branch(app, ports, opening, train_head_param=merge_train_head, mode="live")
    stopped = _branch_result(app, ports, cfg, opening, read, read.gate)
    if stopped is not None:
        return stopped
    synced = sync_branch(app, read, opening)
    if synced is None:
        return _approval_refused(ports, opening)
    gate, opening = synced
    stopped = _branch_result(app, ports, cfg, opening, read, gate)
    if stopped is not None:
        return stopped

    return _readiness_and_merge(app, ports, write_gate, cfg, opening, gate, merge=merge)


def _approval_refused(ports: MergePathPorts, opening: Opening) -> Any:
    """``_update_approval_head`` refused a stale approval (issue #2205): end the pass, no merge.

    A newer verdict landed after the opening read ``approved``; nothing is
    re-stamped and the next pass re-reads the verdict from disk.
    """
    facts = opening.facts
    return render_live_head_moved(
        ports,
        facts.pr_number,
        opening.issue_number,
        facts.verdict.reviewed_head_sha,
        facts.live_head_sha,
        opening.decision,
        label_error=None,
        escalated=False,
    )


def _converge_merged_issue(
    write_gate: WriteGate, state: dict[str, Any], entry: dict[str, Any]
) -> None:
    """A PR already recorded as merged: make its issue entry reach the finalized shape."""
    issue_number = entry.get("issue_number")
    if issue_number is None:
        return
    key = str(issue_number)
    issue_entry = state["issues"].get(key, {})
    finalized = _merged_issue_fields(issue_entry, issue_number)
    if finalized != issue_entry:
        state["issues"][key] = finalized
        write_gate.save_state(state)


def _branch_result(
    app: Any,
    ports: MergePathPorts,
    cfg: MergePathConfig,
    opening: Opening,
    read: BranchRead,
    gate: BranchGate,
) -> Any | None:
    """The terminal result for a branch-gate stop, or ``None`` to keep going."""
    if gate.stop is None:
        return None
    pr_number = opening.facts.pr_number
    issue_number = opening.issue_number
    decision = opening.decision
    failed_attempts = opening.facts.persisted.failed_attempts
    if gate.stop is BranchStop.CONFLICT_IN_FLIGHT:
        return render_live_conflict_in_flight(
            ports, pr_number, issue_number, decision, failed_attempts
        )
    if gate.stop is BranchStop.CONFLICT_BLOCKED:
        return render_live_conflict_blocked(
            ports, pr_number, issue_number, decision, failed_attempts, read.facts.issue_status
        )
    if gate.stop is BranchStop.NOT_TRAIN_HEAD:
        return render_train_not_head(
            ports, cfg, pr_number, issue_number, decision, failed_attempts
        )
    pr = opening.pr
    return app._merge_deferred_stale_base_result(
        pr_number,
        issue_number,
        decision,
        pr.get("baseRefName"),
        pr.get("headRefOid"),
        gate.stale_base_reason,
    )


def _readiness_and_merge(
    app: Any,
    ports: MergePathPorts,
    write_gate: WriteGate,
    cfg: MergePathConfig,
    opening: Opening,
    gate: BranchGate,
    *,
    merge: bool | None,
) -> Any:
    """Stages 3-5: readiness effects, merge plan and its effects, accounting, final result."""
    state_file = app.paths.state_file
    pr_number = opening.facts.pr_number
    issue_number = opening.issue_number

    def read_issue_status() -> tuple[str | None, str | None]:
        if issue_number is None:
            return None, None
        issue = ports.load_state_locked(state_file).get("issues", {}).get(str(issue_number), {})
        if not isinstance(issue, dict):
            return None, None
        return issue.get("status"), issue.get("reason_class")

    # Legacy read/write order: the revert route is persisted BEFORE ``pr_checks``,
    # and the human-merge label read comes only after the stall and infra exits,
    # so a refused or failing read in between cannot lose or add a side effect.
    revert = gather_revert(app, ports, opening, gate)
    revert_args = {
        "approved": opening.admission.approved,
        "sync_failed": gate.sync_failed,
        "revert": revert.status,
        "issue_number": issue_number,
    }
    status, reason_class = None, None
    verdict = decide_revert(**revert_args, issue_status=None)
    if verdict.route:
        status, reason_class = read_issue_status()
        verdict = decide_revert(**revert_args, issue_status=status)
    results = route_cross_pr_revert(app, opening, verdict, revert.reason, EffectResults())

    checks = gather_checks(app, pr_number)
    facts = readiness_facts(
        opening, gate, revert, checks, issue_status=status, issue_reason_class=reason_class
    )
    readiness, _ = decide_readiness_lazily(facts, read_issue_status)

    pr = opening.pr
    decision = opening.decision
    record_containment(app, ports, write_gate, pr_number, readiness.containment_warnings)
    if readiness.readiness_stall:
        label_error = app._request_readiness_no_ci_rework(
            pr, issue_number, decision, readiness.summary.missing
        )
        return render_live_stall(
            ports,
            pr_number,
            issue_number,
            decision,
            readiness.summary,
            label_error=label_error,
            merge_conflict=readiness.branch.merge_conflict,
        )
    if readiness.kind is PlanKind.RERUN_OR_ESCALATE:
        remediated = app._merge_ready_infra_remediation(
            pr_number,
            pr,
            issue_number,
            decision,
            checks.enriched,
            readiness.summary,
            approved=opening.admission.approved,
            sync_failed=readiness.gate.sync_failed,
            merge_conflict=readiness.branch.merge_conflict,
        )
        if remediated is not None:
            return remediated
    readiness = proceed(readiness)
    human_merge = app._human_merge_hold_check(issue_number)
    if deescalation_read_needed(cfg, issue_number, *human_merge):
        status, reason_class = read_issue_status()
    else:
        status, reason_class = None, None
    readiness = with_human_merge(readiness, cfg, human_merge, status, reason_class)
    if readiness.deescalate:
        deescalate_human_merge(app, ports, write_gate, pr_number, issue_number)

    should_merge = cfg.auto_merge_enabled if merge is None else merge
    holds = holds_from_snapshot(
        ports.load_state_locked(state_file),
        cfg,
        pr_number,
        issue_number,
        should_merge=should_merge,
    )
    if merge_hold_read_needed(readiness, holds):
        merge_hold, unavailable = gather_merge_hold(app, pr, issue_number)
        holds = replace(holds, merge_hold=merge_hold, merge_hold_unavailable=unavailable)
    plan = decide_merge(readiness, holds)

    results, label_error = apply_merge_plan(
        app,
        ports,
        write_gate,
        pr_number=pr_number,
        pr=pr,
        decision=decision,
        issue_number=issue_number,
        cfg=cfg,
        plan=plan,
        results=replace(results, issue_status_after=holds.issue_status),
    )
    if results.rework_label_error is not None:
        label_error = results.rework_label_error

    accounting = settle_accounting(
        app,
        ports,
        write_gate,
        pr_number=pr_number,
        issue_number=issue_number,
        cfg=cfg,
        plan=plan,
        results=results,
        pr=pr,
    )
    return render_live_final(
        ports,
        LiveFinal(
            pr_number=pr_number,
            issue_number=issue_number,
            decision=decision,
            cfg=cfg,
            plan=plan,
            results=results,
            accounting=accounting,
            label_error=label_error,
            merge_hold_label=app.config.labels.merge_hold,
        ),
    )
