"""Pure event -> fact-row derivation for the dashboard rollup (events recon section 3).

``derive_event`` maps one ``events`` row to ``(table, columns)`` pairs; it does no I/O.
Kinds without a handler yield nothing, so the handler registry doubles as the
allow-list the rollup selects from. ``KNOWN_IGNORED`` names every deliberately
un-interpreted kind with a one-line reason; a kind in neither table is
*unclassified* and counted per source so it surfaces instead of silently dropping
(issue #2269).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# The kind is spelled via the constant, never a literal: the #2262 salvage-seam
# AST guard (tests/test_dead_worker_salvage_seam.py) counts any bare
# "session_failed_relabeled" constant as a dead-worker requeue locus, which this
# read-side classification is not.
from ..dead_worker_sweep.decide_common import SESSION_FAILED_RELABELED
from .rollup_common import Row, _dict, _flag, _int, _ints, _milestone, _refs, reason_group
from .rollup_flow_handlers import FLOW_HANDLERS, escalation
from .rollup_schema import JOB_TABLE

# Kinds the rollup deliberately does not interpret, with the one-line reason each is
# skipped. A kind in an events DB that is in neither ``HANDLERS`` nor this table is
# *unclassified*: the rollup counts it per source (``SourceResult.unclassified``) and
# warns when the set changes (issue #2269). An emitted kind missing from both fails
# ``test_every_emitted_kind_is_rollup_classified``, so a new writer kind cannot land
# without someone deciding what it means for the dashboard.
KNOWN_IGNORED: dict[str, str] = {
    # -- ci-fleet host kinds (host I/O, leak drain): host-level signals with no per-issue flow fact.
    "host_reboot_due": "ci-fleet host-level event; no per-issue rollup fact",
    "host_leak_warn": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_fallback": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_hold": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_exclusions_applied": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_exclusions_failed": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_exclusions_pending": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_verify_failed": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_unprovisioned": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_invalid": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_converged": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_reverted": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_converge_failed": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_rollout_complete": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_finalize_requested": "ci-fleet host-level event; no per-issue rollup fact",
    "host_io_ab_result": "ci-fleet host-level event; no per-issue rollup fact",
    "host_drain_started": "ci-fleet host-level event; no per-issue rollup fact",
    "host_drained": "ci-fleet host-level event; no per-issue rollup fact",
    "host_drain_cleared": "ci-fleet host-level event; no per-issue rollup fact",
    "host_probe_stale": "ci-fleet host-level event; no per-issue rollup fact",
    # -- Reaper / sweep bookkeeping: recovery detail (and the <kind>_sweep batch
    #    summaries it is folded into); the corrective transitions are their own events.
    "orphaned_worker_drift": "dead-worker sweep drift record; the corrective transition is its own event",
    "orphaned_worker_drift_sweep": "batch form of orphaned_worker_drift",
    "review_dispatch_stalled": "a stalled review claim was reset for retry; recovery bookkeeping",
    "review_dispatch_stalled_sweep": "batch form of review_dispatch_stalled",
    "review_dispatch_lifecycle_reaped": "names a merged/closed PR only; deliberately not merge evidence (see rollup_flow_handlers docstring)",
    "review_dispatch_lifecycle_reaped_sweep": "batch form of review_dispatch_lifecycle_reaped",
    "fleet_reap_sweep": "fleet-level reap summary; per-issue effects carry their own kinds",
    SESSION_FAILED_RELABELED: "a dead session's failure kind was reclassified after the fact",
    "session_salvaged": "a dead worker's salvageable work was recovered",
    "superseded_worker_reaped": "a superseded worker was reaped at the rework trigger",
    "superseded_worker_reap_failed": "the superseded-worker reap failed",
    "foreign_writer_reaped": "a foreign writer's worktree state was cleaned up",
    "foreign_issue_ref_cleared": "a foreign issue reference was cleared",
    "orphan_processes_killed": "orphan-process sweep bookkeeping",
    "closed_unmerged_pr_state_converged": "a closed-unmerged PR's state was converged by reconcile",
    "escalated_label_repaired": "a stale escalated label was repaired",
    "rework_stranded_commits_salvaged": "stranded-commit salvage record for a rework worker",
    "worktree_foreign_adopted": "worktree hygiene: a foreign checkout was adopted",
    "worktree_foreign_writer": "worktree hygiene: a foreign writer was detected",
    "worktree_local_commits_archived": "worktree hygiene: local commits were archived on a no-remote repo",
    "worktree_rescue_captured": "worktree hygiene: a rescue capture was taken",
    "worktrees_reclaimed": "worktree hygiene: merged worktrees were reclaimed",
    # -- Loop-pass / supervisor lifecycle: pass cadence and duration already come
    #    from the loop_passes table; liveness is a Now concern, not History.
    "loop_started": "pass cadence/duration comes from the loop_passes table",
    "loop_completed": "pass cadence/duration comes from the loop_passes table",
    "pass_skipped_locked": "a lock-skipped pass produces no work to chart",
    "loop_pass_deadline_deferred": "in-pass deadline deferral bookkeeping",
    "fleet_pass_completed": "fleet pass summary; per-repo pass data comes from loop_passes",
    "fleet_pass_config_error": "supervisor-side config fault record",
    "fleet_pass_deadline_deferred": "fleet pass deadline bookkeeping",
    "fleet_lane_completed": "fleet lane timing bookkeeping",
    "fleet_lane_overrun": "fleet lane timing bookkeeping",
    "supervisor_started": "supervisor lifecycle; liveness is read live by Now",
    "supervisor_exited": "supervisor lifecycle; liveness is read live by Now",
    "supervisor_wedge_loop": "watchdog tripwire record",
    "supervisor_zero_pass_alarm": "alarm record; heartbeat/Now surface it",
    "supervisor_restart_watchdog_disabled": "watchdog configuration record",
    "supervise_relaunch_cap_reached": "relaunch-cap tripwire record",
    "notify_digest_stale": "notify-digest freshness record; notify supervision consumes it",
    "fleet_paused": "operator control record",
    "fleet_resumed": "operator control record",
    "fleet_stop_requested": "operator control record",
    "fleet_registry_stale_entry": "registry hygiene record",
    "fleet_canary": "synthetic liveness probe; not pipeline data",
    "token_unusable": "auth diagnostic",
    # -- GitHub transport / network diagnostics: throttles and capped_demand carry
    #    the chartable signal.
    "github_error": "transport diagnostic",
    "github_not_found_error": "transport diagnostic",
    "github_transport_fallback": "transport diagnostic",
    "github_circuit_opened": "circuit-breaker state record",
    "github_circuit_closed": "circuit-breaker state record",
    "github_issue_state_partial_fallback": "partial-fallback diagnostic",
    "git_network_retry": "per-call retry diagnostic",
    # -- Runner / CI infrastructure: capacity metrics read runner_samples and
    #    job_observations; these are the incident records around them.
    "runner_health": "host-health record",
    "runner_health_alert": "host-health record",
    "runner_health_desktop_pressure": "host-health record",
    "runner_health_stuck_window": "host-health record",
    "runner_keepalive": "liveness heartbeat, not pipeline data",
    "runner_unregistered": "provisioning record; slot state comes from runner_samples",
    "runner_allocation_refused": "allocation decision record; runner_samples carries the outcome",
    "runner_allocation_skipped": "allocation decision record; runner_samples carries the outcome",
    "runner_capacity_recovered": "capacity bookkeeping",
    "ci_fleet_provenance": "ci_fleet provenance guard record",
    "ci_fleet_worktree_dirty": "ci_fleet provenance guard record",
    "ci_run_never_created": "CI-run creation failure record",
    "workflow_no_jobs": "workflow discovery record",
    "main_ci_reclaim_cancelled": "main-branch CI reclaim bookkeeping",
    "main_ci_reclaim_failed": "main-branch CI reclaim bookkeeping",
    "ci_retrigger_skipped_conflicting": "stale-checks retrigger bookkeeping",
    "ci_retriggered_stale_checks": "stale-checks retrigger bookkeeping",
    "stale_checks_retrigger_exhausted": "stale-checks retrigger bookkeeping",
    "flake_rerun_triggered": "flake-rerun bookkeeping",
    "flake_rerun_failed": "flake-rerun bookkeeping",
    "infra_rerun_triggered": "infra-rerun bookkeeping",
    "infra_rerun_failed": "infra-rerun bookkeeping",
    "infra_rerun_escalated": "infra-rerun escalation record",
    # -- Config / host housekeeping.
    "config_key_deprecated_read": "deprecated-config diagnostic (issue #1976)",
    "config_key_retirement_armed": "config-retirement bookkeeping",
    "config_key_retirement_regressed": "config-retirement bookkeeping",
    "config_retirement_sweep": "config-retirement bookkeeping",
    "containment_check": "report-only merge-path diagnostic per its emit site",
    "label_filter_fallback": "label-filter diagnostic",
    "venv_editable_anchor_violation": "venv-anchor diagnostic",
    "venv_pth_mismatch": "venv-anchor diagnostic",
    "venv_pth_repair_failed": "venv-anchor repair record",
    "venv_pth_repaired": "venv-anchor repair record",
    "test_slot_wait_timeout": "test-slot pool bookkeeping",
    "markdown_guard_disagreement": "guard-comparison diagnostic",
    "outbound_body_secret_refused": "outbound secret-guard refusal record",
    "self_deploy_alarm": "self-deploy alarm record; deploys rows cover outcomes",
    "self_deploy_blockers_cleared": "self-deploy bookkeeping",
    "self_deploy_ci_fleet_pull": "self-deploy bookkeeping",
    "self_deploy_skipped": "a skipped self-deploy produces nothing to chart",
    "self_deploy_sync_starved": "self-deploy starvation record",
    # -- Dispatch-lane decision records: the dispatched milestone and pass_samples
    #    carry what shipped; these record the why-not / decision detail.
    "intake": "intake-lane bookkeeping",
    "intake_failed": "intake-lane bookkeeping",
    "intake_prose_only_deps": "intake-lane bookkeeping",
    "blocker_cycle": "dependency-graph diagnostic",
    "dispatch_skip_blocked": "dispatch-skip record; dispatchable counts come from pass_samples",
    "dispatch_skip_operator_claimed": "dispatch-skip record; dispatchable counts come from pass_samples",
    "dispatch_stale": "a paused fleet's stale-dispatch signal",
    "dispatch_blocked_chain_dead": "dispatch-skip record: a blocked chain went dead",
    "dispatch_blocked_environment": "environment-block dispatch record",
    "dispatch_blocked_environment_reaped": "environment-block reap record",
    "dispatch_citation_drift_flagged": "citation-drift diagnostic",
    "dispatch_closed_unmerged_ready_stripped": "closed-PR ready-state bookkeeping",
    "dispatch_cross_repo_gate_overridden": "cross-repo gate decision record",
    "dispatch_merged_pr_mention_flagged": "merged-PR mention bookkeeping",
    "dispatch_merged_pr_mention_rearmed": "merged-PR mention bookkeeping",
    "escalation_deferred_live_worker": "deferral record: a live worker still owns the issue",
    "live_worker_redispatch_averted": "deferral record: redispatch averted by a live worker",
    "launch_failed": "per-launch failure record; the launch_failures metric reads verdict_missed",
    "role_fallback_selected": "model-chain selection record (audit-only per its emit site)",
    "worker_model_tier_selected": "model-tier routing record (audit-only per its emit site)",
    "worker_model_tier_fallback": "model-tier fallback record (audit-only per its emit site)",
    "rescue_dispatched": "rescue-tier dispatch record; dispatch_rework carries the rework milestone",
    "rescue_review_escalated": "rescue-tier escalation record",
    "rework_dispatch_blocked_environment": "environment-block rework record",
    "rework_dispatch_blocked_environment_reaped": "environment-block reap record",
    "rework_issue_fetch_skipped": "rework-issue bookkeeping",
    "attempt_resumed": "attempt-resume seed record (audit-only per its emit site; issue #2289)",
    "attempt_resume_failed": "attempt-resume fallback record (audit-only per its emit site; issue #2289)",
    # -- Review-lane bookkeeping: verdict milestones and review_samples carry the
    #    lane; these record packet / janitor / verdict detail around it.
    "review_packet": "review-packet bookkeeping",
    "review_packet_template_stale": "review-packet bookkeeping",
    "review_reap_invoked": "review-reap bookkeeping",
    "review_checkout_removal_failed": "review-checkout bookkeeping",
    "review_verdict_reconciled": "verdict-reconcile bookkeeping",
    "review_decision_reclassified_blocked": "verdict-reclassification bookkeeping",
    "review_dispatch_skipped_ci_red": "review-skip record: CI already red",
    "review_exec_rejection_resumed": "exec-rejection resume record",
    "review_exec_rejection_resume_failed": "exec-rejection resume record",
    "review_interrupted_by_deploy": "deploy-interruption record",
    "quota_probe_succeeded": "quota-probe bookkeeping",
    "verdict_carried_forward_clean_rebase": "verdict carry-forward bookkeeping",
    "verdict_carried_forward_line_content": "verdict carry-forward bookkeeping",
    "verdict_carried_forward_verified_sync": "verdict carry-forward bookkeeping",
    "verdict_force_voided": "verdict carry-forward bookkeeping",
    "stale_ci_verdict_gate_pass": "stale-verdict bookkeeping",
    "stale_ci_verdict_requeued": "stale-verdict bookkeeping",
    "required_changes_vacuous": "required-changes diagnostic",
    "request_changes_body_changed_requeued": "requeue bookkeeping",
    "no_op_rework_repair_requested": "rework-routing record",
    "janitor_gate": "janitor gate decision record",
    "janitor_rework_cycle_failed": "janitor rework-loop bookkeeping",
    "janitor_rework_stalled": "janitor rework-loop bookkeeping",
    "pr_body_closing_keyword_autofix_failed": "PR-body autofix bookkeeping",
    "pr_body_closing_keyword_autofixed": "PR-body autofix bookkeeping",
    "pr_closing_ref_rewritten": "closing-reference bookkeeping",
    "pr_closing_ref_unlinked": "closing-reference bookkeeping",
    # -- Rework routing records: the rework_dispatched milestone carries the
    #    dispatch; these record the route taken and the skips.
    "check_failure_rework_requested": "rework-routing record",
    "cross_pr_revert_rework_requested": "rework-routing record",
    "merge_conflict_rework_requested": "rework-routing record",
    "readiness_no_ci_rework_requested": "rework-routing record",
    "stranded_request_changes_rework_requested": "rework-routing record",
    "stranded_request_changes_skipped_issue_closed": "rework-routing record",
    "pre_review_rework_routed": "rework-routing record",
    "rework_already_pushed": "rework-routing record",
    "rework_attempt_exempted_provider_throttle": "rework-attempt bookkeeping",
    "rework_brief_regenerated": "rework-packet bookkeeping",
    "rework_no_op_ci_rework_requested": "rework-routing record",
    "rework_no_op_deferred": "rework-routing record",
    "rework_no_op_escalated": "rework-routing record (audit-only per its emit site)",
    "rework_no_op_rebuttal_review": "rework-routing record",
    "rework_outcome_applied": "rework-outcome bookkeeping",
    "rework_outcome_skipped": "rework-outcome bookkeeping",
    "rework_requeued": "rework-routing record",
    # -- Merge-path bookkeeping: the merged milestone carries the outcome; these
    #    record aborts, deferrals, de-escalations and operator hand-offs.
    "merge_ready": "merge-path accounting record; the merged milestone carries the outcome",
    "merge_authorized": "operator merge authorization record",
    "head_moved": "merge-path abort record: the base head moved mid-merge",
    "merge_deferred_stale_base": "stale-base deferral record",
    "merge_deferred_stale_base_alarm": "stale-base deferral alarm record",
    "merge_failed_attempt_alarm": "merge-failure alarm record",
    "human_merge_required": "human-merge hand-off record (audit-only per its emit site)",
    "human_merge_label_removed": "human-merge bookkeeping",
    "unauthorized_merge_queue_sync_covered": "repeated every pass for the same PR; recon section 4 noise",
    "unauthorized_merge_acknowledged": "unauthorized-merge bookkeeping",
    "unauthorized_merge_baseline_armed": "unauthorized-merge bookkeeping",
    "unauthorized_merge_check_skipped": "unauthorized-merge bookkeeping",
    "unauthorized_merge_detected": "unauthorized-merge bookkeeping",
    "draft_pr_blocked": "draft-PR readiness bookkeeping",
    "draft_pr_ready_failed": "draft-PR readiness bookkeeping",
    "draft_pr_ready_held": "draft-PR readiness bookkeeping",
    "draft_pr_ready_triggered": "draft-PR readiness bookkeeping",
    "deescalation_cap_exhausted": "deescalation bookkeeping",
    "deescalation_cleared": "deescalation bookkeeping",
    "deescalation_pass_completed": "deescalation bookkeeping",
    "deescalation_reason_class_backfilled": "deescalation bookkeeping",
    "deescalation_recurrence_promoted": "deescalation bookkeeping",
    "reconcile_pass_completed": "reconcile pass bookkeeping; merge milestones come from reconcile",
    "reconcile_pass_deferred": "reconcile pass bookkeeping",
    "reconcile_pass_failed": "reconcile pass bookkeeping",
    "reconcile_pass_skipped": "reconcile pass bookkeeping",
    "salvage_skipped_already_landed": "salvage skipped: the work already landed",
    "salvage_skipped_superseded": "salvage skipped: the work was superseded",
    # -- Operator queue / local (no-remote) lane.
    "operator_claim": "operator-claim bookkeeping",
    "operator_claim_released": "operator-claim bookkeeping",
    "operator_queue_impact": "edge-triggered operator-queue signal; Now reads the queue live",
    "throttle_window_set": "operator throttle record; throttles rows cover the refusal kinds",
    "local_blocker_satisfied_by_patch_equivalence": "local (no-remote) lane bookkeeping",
    "local_lane_kill_switch_stalled": "local (no-remote) lane bookkeeping",
    "local_no_op_rework_rearmed": "local (no-remote) lane bookkeeping",
    "local_review_adopted": "local (no-remote) lane bookkeeping",
    "local_suite_failed": "local (no-remote) lane bookkeeping",
    "local_work_ready": "local (no-remote) lane bookkeeping",
    "worktree_unsafe_stranded_salvaged": "stranded-commit salvage record",
    "worktree_unsafe_stranded_salvage_failed": "stranded-commit salvage record",
    # -- Registered kinds the static emit scan cannot see (dynamic or allow-listed
    #    emit sites, or kinds retained for older events DBs): same bookkeeping class.
    "api_worker_provider_suspended": "provider-suspension record for API workers",
    "check_infra_blocked": "infra-block gate record",
    "ci_headroom_unavailable": "CI headroom diagnostic",
    "coverage_probe_flagged": "diff-coverage probe diagnostic",
    "dead_dispatched_throttle_rearmed": "dead-dispatch throttle bookkeeping",
    "dead_dispatched_worker_reaped": "a dead dispatched worker was reaped; recovery bookkeeping",
    "dispatch_failed": "dispatch failure record",
    "host_load_unavailable": "host-load diagnostic",
    "infra_blocked_escalated": "infra-block escalation record",
    "local_merge_deferred": "local (no-remote) lane bookkeeping",
    "local_merge_failed": "local (no-remote) lane bookkeeping",
    "local_merge_rework_escalated": "local (no-remote) lane bookkeeping",
    "local_review_adopt_failed": "local (no-remote) lane bookkeeping",
    "local_review_packet_failed": "local (no-remote) lane bookkeeping",
    "local_suite_launched": "local (no-remote) lane bookkeeping",
    "local_suite_ok": "local (no-remote) lane bookkeeping",
    "local_suite_result": "local (no-remote) lane bookkeeping",
    "loop_refused_preflight": "preflight refusal record",
    "merge_blocked": "merge-path failure record; the merged milestone carries the outcome",
    "merge_failed": "merge-path failure record; the merged milestone carries the outcome",
    "notify_resolution": "notify-digest resolution record; notify supervision consumes it",
    "operator_claim_failed": "operator-claim bookkeeping",
    "orphan_sweep_redispatch_escalated": "orphan-sweep escalation record",
    "orphaned_worker_recovered": "dead-worker sweep recovery record; the corrective transition is its own event",
    "orphaned_worker_routed_to_review": "dead-worker sweep routing record; the corrective transition is its own event",
    "pr_create_failed_branch_stranded": "PR-creation failure record",
    "pr_unlinked_resolved": "unlinked-PR marker bookkeeping (audit-only per its emit site)",
    "pr_unlinked_skipped": "unlinked-PR marker bookkeeping (audit-only per its emit site)",
    "preflight_config_stale": "preflight diagnostic",
    "preflight_warning": "preflight diagnostic",
    "review_dispatch_skipped_empty_diff": "review-skip record: nothing to review",
    "review_stale_claim_recovery_skipped": "stale-claim recovery bookkeeping",
    "review_verdict_reconcile_failed": "verdict-reconcile bookkeeping",
    "rework_outcome_apply_failed": "rework-outcome bookkeeping",
    "runner_capacity_starvation_escalation": "capacity-starvation escalation record",
    "salvage_push_failed": "stranded-commit salvage record",
    "session_budget_exceeded": "session-budget record",
    "session_stalled": "session-stall record; worker exits carry the fate",
    "spec_review": "spec-review bookkeeping",
    "spec_review_failed": "spec-review bookkeeping",
    "supervisor_wedged_killed": "watchdog kill record",
    "unwired_symbol": "unwired-symbol diagnostic (advisory-only per its level file)",
    "worker_attachment_budget_failed": "worker-outcome bookkeeping",
    "worker_declared_blocked": "worker-outcome bookkeeping",
    "worker_evidence_stale": "worker-outcome bookkeeping",
    "worker_literal_tmp_path": "post-hoc /tmp-misuse signal (issue #1780)",
    "worker_module_map_failed": "worker-outcome bookkeeping",
    "worker_verified_no_changes": "worker-outcome bookkeeping",
    "worker_verified_no_changes_ignored": "worker-outcome bookkeeping",
}
# Written to the global DB and (also) to per-repo DBs: the global DB is authoritative.
GLOBAL_ONLY_KINDS = frozenset({"fleet_canary", "runner_allocation", "fleet_job_observations"})

_VERDICTS = {
    "approved": "verdict_approved",
    "request_changes": "verdict_request_changes",
    "blocked": "verdict_blocked",
}


def _dispatch(ev: dict) -> list[Row]:
    p = ev["payload"]
    gov, br = _dict(p.get("concurrency_governor")), _dict(p.get("backlog_reachability"))
    issues = _ints(p.get("issue_numbers"))
    sample: Row = (
        "pass_samples",
        {
            "live_sessions": _int(gov.get("live_session_count")),
            "fleet_live_sessions": _int(gov.get("fleet_live_session_count")),
            "concurrency_limit": _int(gov.get("concurrency_limit")),
            "fleet_concurrency_limit": _int(gov.get("fleet_concurrency_limit")),
            "available_slots": _int(gov.get("available_slots")),
            "dispatch_limit": _int(gov.get("dispatch_limit")),
            "clamped": _flag(gov.get("clamped")),
            "deferred_by_concurrency": _int(p.get("deferred_by_concurrency_count")),
            "launched": len(issues),
            "open_total": _int(br.get("open_total")),
            "dispatchable": _int(br.get("dispatchable")),
            "active_label": _int(br.get("active_label")),
            "missing_ready": _int(br.get("missing_ready")),
            "terminal_label": _int(br.get("terminal_label")),
            "blocked_by_open_dependency": _int(br.get("blocked_by_open_dependency")),
            "operator_claimed": _int(br.get("operator_claimed")),
        },
    )
    return [sample, *(_milestone(ev, "dispatched", i, None) for i in issues)]


def _dispatch_rework(ev: dict) -> list[Row]:
    """One row per reworked issue. A batch event carries ONE ``pr_number``: the first
    *selected* issue's PR (state_dispatch_rework), which need not be the first listed one,
    so no batch member can be told it is theirs -- only a lone issue gets the PR. Tagging
    every member with it merged another issue's PR into their timelines (W3-02)."""
    issues = _ints(ev["payload"].get("issue_numbers")) or [ev["issue_number"]]
    pr = ev["payload"].get("pr_number", ev["pr_number"]) if len(issues) == 1 else None
    return [_milestone(ev, "rework_dispatched", i, pr) for i in issues]


def _pr_opened(milestone: str, approx: bool) -> Callable[[dict], list[Row]]:
    def handler(ev: dict) -> list[Row]:
        issue, pr = _refs(ev)
        return [_milestone(ev, milestone, issue, pr, approx)]

    return handler


def _review_claim(ev: dict) -> list[Row]:
    prs = _ints(ev["payload"].get("pr_numbers"))
    return [_milestone(ev, "review_claimed", None, pr) for pr in prs]


def _review_dispatch(ev: dict) -> list[Row]:
    p = ev["payload"]
    row = {
        "available_slots": _int(p.get("fleet_available_review_slots")),
        "live_reviews": _int(p.get("fleet_live_review_count")),
        "review_limit": _int(p.get("fleet_review_concurrency_limit")),
        "launched": len(p.get("launched") or []),
        "failed": len(p.get("failed") or []),
        "quota_hit": _flag(p.get("quota_hit")),
    }
    return [("review_samples", row)]


def _record_review(ev: dict) -> list[Row]:
    p = ev["payload"]
    issue, pr = _refs(ev)
    rows: list[Row] = []
    if p.get("decision") in _VERDICTS:
        rows.append(_milestone(ev, _VERDICTS[p["decision"]], issue, pr))
    if p.get("escalated"):
        rows.extend(escalation(ev, "escalated", "review_verdict_escalated"))
    return rows


def _ready_observed(ev: dict) -> list[Row]:
    issue, pr = _refs(ev)
    return [_milestone(ev, "ready_observed", issue, pr)]


def _session_exited(ev: dict) -> list[Row]:
    p = ev["payload"]
    row = {
        "issue": _int(_refs(ev)[0]),
        "failure_kind": p.get("failure_kind"),
        "worker_health": p.get("worker_health"),
    }
    return [("worker_exits", row)]


def _verdict_missed(ev: dict) -> list[Row]:
    p = ev["payload"]
    issue, pr = _refs(ev)
    row = {
        "issue": _int(issue),
        "pr": _int(pr),
        "reason": p.get("reason") if isinstance(p.get("reason"), str) else None,
        "reason_group": reason_group(p.get("reason")),
        "exit_code": _int(_dict(p.get("cause")).get("exit_code")),
        "turn_count": _int(p.get("turn_count")),
        "tool_call_count": _int(p.get("tool_call_count")),
    }
    return [("verdict_missed", row)]


def _runner_allocation(ev: dict) -> list[Row]:
    p = ev["payload"]
    rows: list[Row] = []
    for t in p.get("targets") or []:
        if not isinstance(t, dict) or not t.get("repo"):
            continue
        row = {
            "target_repo": str(t["repo"]),
            "capacity": _int(t.get("capacity")),
            "demand": _int(t.get("demand")),
            "running": _int(t.get("running")),
            "target": _int(t.get("target")),
            "budget": _int(p.get("budget")),
            "oldest_queued_seconds": t.get("oldest_queued_seconds"),
        }
        rows.append(("runner_samples", row))
    return rows


def _measured(durations: dict[str, Any], name: str) -> float | None:
    m = _dict(durations.get(name))
    return m.get("seconds") if m.get("kind") == "measured" else None


def _job_observations(ev: dict) -> list[Row]:
    rows: list[Row] = []
    for job in ev["payload"].get("jobs") or []:
        if not isinstance(job, dict) or job.get("status") != "completed" or not job.get("job_id"):
            continue
        d = _dict(job.get("durations"))
        row = {
            "job_id": str(job["job_id"]),
            "name": job.get("name"),
            "status": job.get("status"),
            "queue_wait_seconds": _measured(d, "queue_wait"),
            "execution_seconds": _measured(d, "execution"),
            "wall_seconds": _measured(d, "wall"),
        }
        rows.append((JOB_TABLE, row))
    return rows


def _deploy(ok: bool) -> Callable[[dict], list[Row]]:
    def handler(ev: dict) -> list[Row]:
        p = ev["payload"]
        err = p.get("error")
        row = {
            "ok": int(ok),
            "changed": _flag(p.get("changed")),
            "from_sha": p.get("from_sha"),
            "to_sha": p.get("to_sha"),
            "error": str(err)[:200] if err else None,
        }
        return [("deploys", row)]

    return handler


def _throttle(until_key: str | None, detail_key: str | None) -> Callable[[dict], list[Row]]:
    def handler(ev: dict) -> list[Row]:
        p = ev["payload"]
        until = p.get(until_key) if until_key else None
        detail = p.get(detail_key) if detail_key else None
        row = {
            "event_kind": ev["kind"],
            "until": until if isinstance(until, str) else None,
            "detail": str(detail) if detail is not None else None,
        }
        return [("throttles", row)]

    return handler


def _capped(
    requested: str | None, granted: str | None, reason: str | None, starved_repo: bool = False
) -> Callable[[dict], list[Row]]:
    def handler(ev: dict) -> list[Row]:
        p = ev["payload"]
        why = p.get(reason) if reason else None
        # The global DB records a starved repo's event; the payload names who starved.
        named = p.get("repo") if starved_repo and isinstance(p.get("repo"), str) else None
        row = {
            **({"repo": named} if named else {}),
            "event_kind": ev["kind"],
            "requested": _int(p.get(requested)) if requested else None,
            "granted": _int(p.get(granted)) if granted else None,
            "reason": str(why) if why is not None else None,
        }
        return [("capped_demand", row)]

    return handler


# Handlers whose rows depend only on the event's issue/PR refs, never on other payload
# fields: exactly the ones a ``<kind>_sweep`` summary (numbers only) can be expanded into.
REF_ONLY: dict[str, Callable[[dict], list[Row]]] = {
    "worker_handoff_pr_opened": _pr_opened("pr_opened_by_worker", False),
    "orphaned_worker_opened_pr": _pr_opened("pr_opened_by_salvage", True),
    "salvage_pushed_stranded_commits": _pr_opened("pr_opened_by_salvage", False),
    # The worker died after opening its PR; the sweep moved the issue to PR open. The PR
    # predates this event, so the stage boundary is approximate.
    "orphaned_worker_advanced_to_pr_open": _pr_opened("pr_open_after_dead_worker", True),
}

SWEEP_SUFFIX = "_sweep"


def _sweep(handler: Callable[[dict], list[Row]]) -> Callable[[dict], list[Row]]:
    """Expand a ``<kind>_sweep`` summary into the per-issue rows of ``<kind>``.

    The reaper folds several same-kind events of one pass into a single summary that keeps
    only ``issue_numbers`` (or ``pr_numbers``) plus a count (stalled_review_reap), so the
    per-event rows would otherwise be lost for every issue in the batch.
    """

    def expand(ev: dict) -> list[Row]:
        payload = ev["payload"]
        out: list[Row] = []
        for issue in _ints(payload.get("issue_numbers")):
            out += handler({**ev, "issue_number": issue, "pr_number": None, "payload": {}})
        for pr in _ints(payload.get("pr_numbers")):
            out += handler({**ev, "issue_number": None, "pr_number": pr, "payload": {}})
        return out

    return expand


HANDLERS: dict[str, Callable[[dict], list[Row]]] = {
    "dispatch": _dispatch,
    "dispatch_rework": _dispatch_rework,
    **REF_ONLY,
    **{kind + SWEEP_SUFFIX: _sweep(handler) for kind, handler in REF_ONLY.items()},
    "review_dispatch_claim": _review_claim,
    "review_dispatch": _review_dispatch,
    "record_review": _record_review,
    "ready_observed": _ready_observed,
    "session_exited": _session_exited,
    "review_verdict_missed": _verdict_missed,
    "runner_allocation": _runner_allocation,
    "fleet_job_observations": _job_observations,
    "self_deploy_succeeded": _deploy(True),
    "self_deploy_failed": _deploy(False),
    "session_rate_limit_deferred": _throttle("defer_until", None),
    "graphql_rate_limit_deferred": _throttle(None, "phase"),
    "review_quota_exhausted": _throttle("throttled_until", "source"),
    "quota_probe_failed": _throttle(None, None),
    "operator_throttle_set": _throttle("throttled_until", "reason"),
    "api_budget_refused": _throttle(None, "reason"),
    "dispatch_backpressure": _capped("requested_limit", "clamped_limit", "clamped_by"),
    "dispatch_deferred": _capped(None, None, "deferred_reason"),
    "dispatch_starved": _capped(None, None, "lane"),
    "runner_capacity_starved": _capped("demand", "capacity", None, starved_repo=True),
}
HANDLERS.update(FLOW_HANDLERS)
assert (
    not KNOWN_IGNORED.keys() & HANDLERS.keys()
)  # a kind is either interpreted or ignored, never both


def derive_event(source: str, ev: dict) -> list[Row]:
    """Map one decoded event row to fact rows, stamped with the shared key columns.

    ``seq`` numbers each table's rows within the event. ``source`` (the DB the row came
    from) is the repo; the event's own ``repo`` column is never consulted.
    """
    handler = HANDLERS.get(ev["kind"])
    if handler is None:
        return []
    seqs: dict[str, int] = {}
    out: list[Row] = []
    for table, cols in handler(ev):
        base: dict[str, Any] = {"source": source, "src_id": ev["id"], "ts": ev["ts"]}
        if table != JOB_TABLE:
            seq = seqs.get(table, 0)
            seqs[table] = seq + 1
            base.update(seq=seq, repo=source)
        out.append((table, {**base, **cols}))
    return out
