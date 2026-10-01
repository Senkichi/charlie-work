"""Dry-run consumer of the **Merge path**: gather -> decide -> render, never apply.

Runs the same stages as ``run_merge_ready`` with the preview gathers
(``mode="preview"``). It performs no write, no label transition, no branch
sync, no merge and no hand-off, and never enters the accounting stage, so the
failed-attempt counters it reports are the persisted ones and the alarm is
always off. Preview-only behaviour that exists because effects are not applied
(conflict and stall continue instead of routing, no de-escalation, escalation
read from the opening snapshot) lives here, not in the decisions.
"""

from __future__ import annotations

from typing import Any

from .decide import decide_readiness, merge_entry
from .gather import (
    config_slice,
    gather_branch,
    gather_checks,
    gather_merge_hold,
    gather_opening,
    gather_revert,
    holds_from_snapshot,
    readiness_facts,
)
from .model import BranchStop, PlanKind
from .ports import MergePathPorts, ports_from_workflow
from .render import (
    PreviewFinal,
    render_not_found,
    render_preview_final,
    render_preview_head_moved,
    render_preview_skip,
    render_preview_stale_base,
    render_train_not_head,
)


def preview_merge_ready(
    app: Any,
    pr_number: int,
    *,
    merge: bool | None = None,
    merge_train_head: int | None = None,
    ports: MergePathPorts | None = None,
) -> Any:
    """Read-only readiness verdict for one PR (issue #614)."""
    ports = ports or ports_from_workflow()
    cfg = config_slice(app.config)
    # ``load_state_locked`` is the single read point outside a ``state_lock``
    # (issue #310); the snapshot is reused for the escalation flags because
    # nothing in a preview writes.
    snapshot = ports.load_state_locked(app.paths.state_file)
    entry = snapshot["prs"].get(str(pr_number), {})

    opening = gather_opening(app, ports, pr_number, cfg, entry, preview=True)
    kind = opening.admission.kind
    if kind is PlanKind.SKIP:
        return render_preview_skip(ports, pr_number, entry.get("issue_number"))
    if kind is PlanKind.NOT_FOUND:
        return render_not_found(ports, pr_number)
    issue_number = opening.issue_number
    decision = opening.decision
    persisted = opening.facts.persisted
    if kind is PlanKind.REQUEST_REWORK:
        return render_preview_head_moved(
            ports,
            pr_number,
            issue_number,
            opening.facts.verdict.reviewed_head_sha,
            opening.facts.live_head_sha,
            decision,
        )

    branch = gather_branch(app, ports, opening, train_head_param=merge_train_head, mode="preview")
    gate = branch.gate
    if gate.stop is BranchStop.NOT_TRAIN_HEAD:
        return render_train_not_head(
            ports, cfg, pr_number, issue_number, decision, persisted.failed_attempts
        )
    if gate.stop is BranchStop.STALE_BASE:
        return render_preview_stale_base(
            ports,
            cfg,
            pr_number,
            issue_number,
            decision,
            persisted.failed_attempts,
            persisted.stale_base_deferrals,
        )

    revert = gather_revert(app, ports, opening, gate)
    checks = gather_checks(app, pr_number)
    human_merge = app._human_merge_hold_check(issue_number)
    readiness = decide_readiness(
        readiness_facts(opening, gate, revert, checks, human_merge=human_merge)
    )

    should_merge = cfg.auto_merge_enabled if merge is None else merge
    holds = holds_from_snapshot(snapshot, cfg, pr_number, issue_number, should_merge=should_merge)
    escalated_merge_hold, enter = merge_entry(readiness, holds)
    merge_hold, merge_hold_unavailable = (
        gather_merge_hold(app, opening.pr, issue_number) if enter else (False, False)
    )
    return render_preview_final(
        ports,
        PreviewFinal(
            pr_number=pr_number,
            issue_number=issue_number,
            decision=decision,
            readiness=readiness,
            cfg=cfg,
            should_merge=should_merge,
            escalated_merge_hold=escalated_merge_hold,
            merge_hold=merge_hold,
            merge_hold_unavailable=merge_hold_unavailable,
            merge_hold_label=app.config.labels.merge_hold,
            failed_attempts=persisted.failed_attempts,
            stale_base_deferrals=persisted.stale_base_deferrals,
        ),
    )
