"""Nothing-to-dispatch results for the dispatch lanes (issue #3139).

A lane whose lock-free scan finds nothing to launch returns through here
without requesting the launch permit, so the pending ``FleetLaunchLock``
handle is never realized -- ending the hold (observed at 183 s) that
starved the other lanes of the shared fleet lock. Two shapes, each
mirroring the lane's existing empty-path tail exactly:

* ``nothing_to_dispatch`` -- the fresh lane, whose ``dispatch`` event is
  unconditional (it is where ``backlog_reachability`` /
  ``dispatch_staleness`` telemetry lives), emitted with an exact ``0``
  hold, an empty phase map, and the governor-only fields at the values a
  deferral before the governor reports (``clamped`` false,
  ``dispatch_limit`` unclamped).
* ``nothing_to_dispatch_rework`` -- the rework lane's ``not selected``
  tail: "no rework candidates found" and no ``dispatch_rework`` event
  (that path emits none either).

Both run the same lock-free gates the permit would have run --
:func:`~charlie_work.worker_launch_gates.lock_free_launch_gates`, then
the fleet provider-throttle window -- so an empty pass on a paused or
throttled repo reports the same deferral it always did instead of a clean
"nothing to dispatch" that would mask it.

These are plain functions, deliberately not in
``charlie_work.orchestration``: the delegation installer attaches every
top-level ``def`` of a delegate module onto ``OrchestratorApp``, and this
internal plumbing is not app surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from charlie_work import worker_fate
from charlie_work.attention_digest import (
    _emit_attention_digest,
    _emit_stalled_session_digest,
)
from charlie_work.ci_findings import check_dispatch_staleness
from charlie_work.command_result import CommandResult
import charlie_work.fleet_provider_throttle as fleet_provider_throttle
from charlie_work.state import (
    arm_dispatch_stale_alert,
    backfill_dispatch_baseline,
    clear_dispatch_stale_alert,
    load_state,
    operator_claimed_issues,
    record_non_empty_dispatch,
    state_lock,
)
from charlie_work.worker_launch_gate import FleetLaunchLock
from charlie_work.worker_launch_gates import (
    REASON_PROVIDER_THROTTLED,
    WorkerLaunchDeferral,
    lock_free_launch_gates,
)

if TYPE_CHECKING:
    from charlie_work.workflow import OrchestratorApp, _MergedPRListOutcome


def _workflow() -> Any:
    # ``charlie_work.workflow`` imports the orchestration delegates, which
    # import this module -- resolve it at call time, not import time.
    import charlie_work.workflow as wf

    return wf


def _dispatch_staleness_bookkeeping(
    app: OrchestratorApp,
    state: dict[str, Any],
    *,
    successful_issue_numbers: set[int],
    backlog_reachability: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Issue #946/#1769 cadence bookkeeping, shared by both dispatch outcomes.

    Runs the durable-baseline marker, the staleness check, and the
    edge-triggered ``dispatch_stale`` alert -- the same sequence the launch
    path runs inside the state lock before appending its ``dispatch`` event
    -- and returns ``(state, dispatch_staleness_for_event)``: the updated
    state plus the ``should_emit``-stripped staleness payload the caller
    embeds in that event. The #3139 never-realized-lock path calls this too
    so a nothing-to-dispatch pass reports staleness identically.
    """
    # Issue #1769: a single frozen clock read for this whole cluster --
    # the durable baseline marker, the staleness check, and the alert
    # cadence marker all timestamp against the same instant, mirroring
    # the #828/#838 single-frozen-clock-per-pass invariant.
    dispatch_cadence_now = app.host.clock.now()
    dispatch_cadence_now_iso = (
        dispatch_cadence_now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    if successful_issue_numbers:
        # Durable baseline marker (issue #1769): persisted here,
        # whenever a pass actually dispatches, instead of re-derived by
        # scanning events.db -- a durable field cannot fall out of a
        # rolling window the way a count-bounded lookback can.
        state = record_non_empty_dispatch(
            state, dispatch_cadence_now_iso, sorted(successful_issue_numbers)
        )
    # Issue #1769 review follow-up: a repo that was already mid-stall (or
    # already existed) before the durable marker was introduced never
    # takes the write path above on its own -- a stalled repo dispatches
    # nothing by definition. One-time recovery from events.db history so
    # that case does not report `no_baseline` forever. No-ops instantly
    # once a baseline exists or a prior attempt already ran.
    state = backfill_dispatch_baseline(state, app.paths.state_file)
    dispatch_staleness = check_dispatch_staleness(
        state,
        app.config.dispatch,
        backlog_reachability,
        recent_issue_numbers=sorted(successful_issue_numbers),
        now=dispatch_cadence_now,
    )
    # `should_emit` is this method's own control-flow answer, not a fact
    # worth persisting: it is `True` on every `dispatch_stale` row by
    # construction (the event is only ever recorded when it's `True`),
    # so embedding it in the payload/ring entry can never discriminate
    # anything a reader doesn't already know (review finding #7).
    dispatch_staleness_for_event = {
        key: value for key, value in dispatch_staleness.items() if key != "should_emit"
    }
    if dispatch_staleness["stale"]:
        # Issue #1769: edge-triggered + bounded low-rate reminder, not
        # an unconditional re-fire every pass the condition holds (the
        # design artifact's section 6 policy). `should_emit` already
        # encodes both the stall-onset edge and the reminder cadence.
        if dispatch_staleness["should_emit"]:
            state = arm_dispatch_stale_alert(state, dispatch_cadence_now_iso)
            state = app._record_event(state, "dispatch_stale", dispatch_staleness_for_event)
    else:
        # Reset the alert marker only on a genuine resolution -- an
        # "unknown" reading (backlog_not_observed, no_baseline,
        # threshold_disabled) must not be mistaken for one, or a
        # transient blip resets the edge and the very next real
        # observation re-fires immediately (review finding #4).
        state = clear_dispatch_stale_alert(state, reason=dispatch_staleness["reason"])
    return state, dispatch_staleness_for_event


def _dispatch_deferral_result(
    deferral: WorkerLaunchDeferral,
    *,
    merged_prs_for_tripwire: list[dict[str, Any]] | None,
    launch_lock: FleetLaunchLock,
) -> CommandResult:
    """The fresh lane's deferral result -- the permit branch and the #3139
    nothing-to-dispatch preflight build the same payload.

    Issue #3139: the lock fields ride the deferral payload too
    (``records_deferral`` copies ``data`` into the ``dispatch_deferred``
    event) so a deferral that happened after the governor ran reports its
    ``governor`` phase.
    """
    data = {
        "selected_count": 0,
        "attempted_count": 0,
        "failed_count": 0,
        "skipped_issue_numbers": [],
        "label_errors": [],
        "sessions": [],
        "dispatch_results": [],
        "merged_prs": merged_prs_for_tripwire,
        "deferred_reason": deferral.reason,
        **deferral.report_fields(),
        **launch_lock.event_fields(),
    }
    if deferral.reason == REASON_PROVIDER_THROTTLED:
        message = f"dispatch deferred: provider throttled until {deferral.throttled_until}"
    else:
        message = f"dispatch deferred: {deferral.reason}"
    return CommandResult(deferral.ok, message, data)


def _rework_deferral_result(
    app: OrchestratorApp,
    deferral: WorkerLaunchDeferral,
    *,
    operator_claimed_skipped: list[int],
    launch_lock: FleetLaunchLock,
) -> CommandResult:
    """The rework lane's deferral result -- same payload whether the
    deferral came from the permit or the #3139 empty-pass preflight."""
    data = {
        "adapter": app.config.worker.harness,
        "selected_count": 0,
        "deferred_reason": deferral.reason,
        "operator_claimed_skipped": sorted(operator_claimed_skipped),
        "worker_death_escalated": [],
        **deferral.report_fields(),
        **launch_lock.event_fields(),
    }
    if deferral.reason == REASON_PROVIDER_THROTTLED:
        message = f"rework dispatch deferred: provider throttled until {deferral.throttled_until}"
    else:
        message = f"rework dispatch deferred: {deferral.reason}"
    return CommandResult(deferral.ok, message, data)


def nothing_to_dispatch(
    app: OrchestratorApp,
    *,
    issues: list[dict[str, Any]],
    dispatch_limit: int,
    merged_prs: _MergedPRListOutcome | None,
    resolved_merged_prs: list[dict[str, Any]],
    backlog_reachability: dict[str, Any],
    stalled_entries: list[dict[str, int]] | None,
    launch_lock: FleetLaunchLock,
) -> CommandResult:
    """Issue #3139: the fresh lane's "nothing to dispatch" result.

    The lock-free gates the permit would have run still run: an empty pass
    on a paused or throttled repo reports the same deferral it always did,
    never a clean "nothing to dispatch" that would mask it.
    """
    _wf = _workflow()
    merged_prs_for_tripwire: list[dict[str, Any]] | None = (
        merged_prs.items
        if merged_prs is not None and merged_prs.called and merged_prs.error is None
        else None
    )
    selection, preflight = lock_free_launch_gates(app)
    if preflight is not None:
        return _dispatch_deferral_result(
            preflight,
            merged_prs_for_tripwire=merged_prs_for_tripwire,
            launch_lock=launch_lock,
        )
    resume = fleet_provider_throttle.decide_for_app(app, selection=selection)
    if resume.deferred:
        return resume.deferred_result(
            "dispatch",
            {
                "selected_count": 0,
                "attempted_count": 0,
                "failed_count": 0,
                "skipped_issue_numbers": [],
                "label_errors": [],
                "sessions": [],
                "dispatch_results": [],
                "merged_prs": merged_prs_for_tripwire,
            },
        )
    manifest_path = app._layout.session_manifest
    results_path = app._layout.session_results
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        # Same audit event the launch path emits for issues the
        # operator-claimed filter kept out of the candidate set -- a CLOSED
        # ready issue can carry the marker, and its presence still gets
        # reported even though nothing here can launch.
        operator_claimed_ready = sorted(
            operator_claimed_issues(state) & {int(issue["number"]) for issue in issues}
        )
        if operator_claimed_ready:
            state = app.write_gate.append_event(
                state,
                # event-consumer: audit-only -- records a skip that the
                # candidate filter already enforced; the skip itself
                # already happened.
                "dispatch_skip_operator_claimed",
                {"issue_numbers": operator_claimed_ready},
            )
        dispatch_failure_map = _wf._build_failure_map(
            [], set(), [], dispatch_limit, extra_failures={}
        )
        state, dispatch_staleness_for_event = _dispatch_staleness_bookkeeping(
            app,
            state,
            successful_issue_numbers=set(),
            backlog_reachability=backlog_reachability,
        )
        state = app.write_gate.append_event(
            state,
            "dispatch",
            {
                "issue_numbers": [],
                "live_worker_issue_numbers": [],
                "phantom_live_worker_issue_numbers": [],
                "failed_issue_numbers": [],
                "foreign_writer_issue_numbers": [],
                "cross_repo_escalated_issue_numbers": [],
                "cross_repo_override_issue_numbers": [],
                "label_errors": [],
                "skipped_issue_numbers": [],
                "deferred_by_concurrency": [],
                "deferred_by_concurrency_count": 0,
                "merged_pr_referenced_issue_numbers": [],
                "merged_pr_closed_issue_numbers": [],
                "merged_pr_flagged_issue_numbers": [],
                "merged_pr_mention_rearmed_issue_numbers": [],
                "failures": dispatch_failure_map,
                **launch_lock.event_fields(),
                "backlog_reachability": backlog_reachability,
                "dispatch_staleness": dispatch_staleness_for_event,
                # The governor never ran, so its subfields take their
                # pre-governor-deferral values -- absent. `clamped` and
                # `dispatch_limit` stay in the shape so readers never
                # re-derive the shape difference.
                "concurrency_governor": {
                    "clamped": False,
                    "dispatch_limit": dispatch_limit,
                },
            },
        )
        app.write_gate.save_state(state)
    worker_fate.report_stale_evidence(app.paths.state_file, {}, write_gate=app.write_gate)

    # Emit notification digest if there are health transitions (stalled sessions)
    # This will be enhanced by #165 to include RUNAWAY/DEAD/escalated transitions
    _emit_stalled_session_digest(app, stalled_entries)

    # Emit dispatch-alert digest -- a never-launched pass has no transitions,
    # so this no-ops exactly as the empty-input call on the launch path does.
    _emit_attention_digest(app, {}, state_field="dispatch_alert")

    data = {
        "selected_count": 0,
        "attempted_count": 0,
        "failed_count": 0,
        "failures": dispatch_failure_map,
        "live_worker_count": 0,
        "phantom_live_worker_count": 0,
        "phantom_live_worker_issue_numbers": [],
        "foreign_writer_count": 0,
        "cross_repo_escalated_issue_numbers": [],
        "cross_repo_override_issue_numbers": [],
        "skipped_issue_numbers": [],
        "deferred_by_concurrency": [],
        "deferred_by_concurrency_count": 0,
        "merged_prs": resolved_merged_prs,
        "merged_pr_referenced_issue_numbers": [],
        "merged_pr_closed_issue_numbers": [],
        "merged_pr_flagged_issue_numbers": [],
        "merged_pr_mention_rearmed_issue_numbers": [],
        "merged_pr_mention_details": {},
        "label_errors": [],
        "session_manifest": str(manifest_path),
        "session_results": str(results_path),
        "sessions": [],
        "dispatch_results": [],
        "live_worker_redispatch_averted": [],
        "stalled": stalled_entries,
        "blocked": [],
        "operator_claimed_ready": operator_claimed_ready,
    }
    return CommandResult(True, "dispatch complete", data)


def nothing_to_dispatch_rework(
    app: OrchestratorApp,
    *,
    rework_limit: int,
    operator_claimed_skipped: list[int],
    launch_lock: FleetLaunchLock,
) -> CommandResult:
    """Issue #3139: the rework lane's "no candidates" result.

    Returned when the scan found no rework candidate (``rework_requested``
    + open PR + not operator-claimed), so the pending fleet lock handle was
    never realized. Mirrors the ``not selected`` tail: the same data keys
    with their empty-pass values, no ``dispatch_rework`` event (that path
    emits none either), and none of the governor report fields the governor
    never produced.
    """
    _wf = _workflow()
    selection, preflight = lock_free_launch_gates(app)
    if preflight is not None:
        return _rework_deferral_result(
            app,
            preflight,
            operator_claimed_skipped=operator_claimed_skipped,
            launch_lock=launch_lock,
        )
    resume = fleet_provider_throttle.decide_for_app(app, selection=selection)
    if resume.deferred:
        data = {"adapter": app.config.worker.harness, "selected_count": 0}
        data["operator_claimed_skipped"] = sorted(operator_claimed_skipped)
        data["worker_death_escalated"] = []
        return resume.deferred_result("rework dispatch", data)
    data = {
        "adapter": app.config.worker.harness,
        "selected_count": 0,
        "failures": _wf._build_failure_map([], set(), [], rework_limit),
        "deferred_by_concurrency": [],
        "deferred_by_concurrency_count": 0,
        "routed_to_review": [],
        "skipped_head_indeterminate": [],
        "review_blocked_retry": [],
        "operator_claimed_skipped": sorted(operator_claimed_skipped),
        "no_op_rework_escalated": [],
        "worker_death_escalated": [],
        "salvaged_to_review": [],
        "blocked_environment_escalated": [],
        "escalation_deferred_live_worker": [],
    }
    return CommandResult(True, "no rework candidates found", data)
