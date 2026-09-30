"""Live-worker cap-escalation deferral (issue #2051).

A rework cap lane (the ``dispatch_rework`` redispatch/death-loop/
blocked-environment caps and the janitor conflict/no-op attempts caps in
``_route_janitor_gate_failure_to_rework``) must not escalate while a
genuinely live worker still holds the issue -- escalation hands the branch
to the operator queue while the worker is still writing it (the #2006
incident: the conflict router re-set ``rework_requested`` over a live
worker, and two cap escalations then handed the issue to
``agent:operator-queue`` while it kept committing). ``rework_requested``
is not evidence nobody holds the issue.

This module is the shared decision+recording layer both call sites use:

- ``_live_worker_cap_escalation_deferrals`` probes every cap candidate
  once through ``issue_worker_liveness`` -- the same single-authority
  predicate ``unescalate`` uses, so "live" cannot mean something different
  at the escalation decision than at the operator re-arm door. The probe
  touches the filesystem (sidecars, sessions.db) and the process table, so
  callers run it OUTSIDE ``state_lock``.
- ``_filter_cap_escalation_lanes_for_live_workers`` is the call-site
  helper both ``_dispatch_rework_impl`` paths (dry-run partition and live
  escalation) share: probe every lane once, return each lane's list minus
  its deferred issues plus the deferral map.
- ``_defer_cap_escalation_for_live_worker`` writes the per-issue deferral
  marker + ``escalation_deferred_live_worker`` event against already-loaded
  state (the caller holds the lock).
- ``_record_cap_escalation_deferrals`` is the batch write path: lock,
  load, apply one deferral per issue, save.
- ``_defer_janitor_cap_escalation_for_live_worker`` is the janitor lane's
  combined probe-and-record step.

The helpers are Convention-B WriteGate consumers: every write goes through
the injected ``write_gate`` (``record_event``/``save_state``), so deferral
recording is dry-run-safe by construction and adds no raw primitive call
sites to the gated-mutator-layer ratchet (issue #1264).

``workflow.py`` re-exports every public-for-the-facade name here via its
facade import block so existing ``charlie_work.workflow.<name>`` import
paths and monkeypatch targets resolve unchanged (the #1283 pattern).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .state import load_state, state_lock, utc_now
from .write_gate import WriteGate, require_write_gate

if TYPE_CHECKING:
    from .config import OrchestratorConfig
    from .worker import IssueWorkerLiveness


def _live_worker_cap_escalation_deferrals(
    lanes: Iterable[tuple[Iterable[int], str]],
    *,
    issues: Mapping[str, Any],
    sessions_dir: Path,
    config: OrchestratorConfig,
    now: datetime,
) -> dict[int, tuple[str, IssueWorkerLiveness]]:
    """Partition cap-escalation candidates by worker liveness (issue #2051).

    ``lanes`` is a sequence of ``(issue_numbers, reason)`` pairs -- one pair
    per cap lane -- where ``reason`` is the cap reason that lane would report
    (e.g. ``"no_op_rework_cap_exceeded"``, the ``session_failed_escalated``
    payload's reason label for the no-op lane, whose persisted
    ``escalation_reason`` is the shared ``"redispatch_cap_exceeded"``; or
    ``"conflict_rework_attempts_cap_exceeded"``, the janitor lane's true
    escalation reason). Every candidate is probed
    once through ``issue_worker_liveness`` -- the same single-authority
    predicate ``unescalate`` uses, so "live" cannot mean something different
    at the escalation decision than at the operator re-arm door -- and a
    ``live`` verdict maps the issue to ``(reason, verdict)`` in the returned
    dict. Callers drop those issues from their escalation lists entirely:
    the deferral leaves counters, status, and labels untouched, so the next
    pass re-evaluates and escalates as soon as the worker exits or wedges
    (the verdict already folds in the watchdog stall standard).

    Read-only probe: touches sidecars/sessions.db and the process table, so
    callers run it OUTSIDE ``state_lock`` and re-load state for the marker
    write via ``_record_cap_escalation_deferrals``. A worker that exits
    between probe and write defers once and escalates next pass --
    self-correcting, at worst one pass late.
    """
    from .worker import issue_worker_liveness

    deferrals: dict[int, tuple[str, IssueWorkerLiveness]] = {}
    for lane, reason in lanes:
        for issue_number in lane:
            if issue_number in deferrals:
                continue
            entry = issues.get(str(issue_number), {})
            verdict = issue_worker_liveness(
                issue_number,
                entry if isinstance(entry, dict) else {},
                sessions_dir,
                config,
                now,
            )
            if verdict.live:
                deferrals[issue_number] = (reason, verdict)
    return deferrals


def _filter_cap_escalation_lanes_for_live_workers(
    lanes: Iterable[tuple[Iterable[int], str]],
    *,
    issues: Mapping[str, Any],
    sessions_dir: Path,
    config: OrchestratorConfig,
    now: datetime,
) -> tuple[list[list[int]], dict[int, tuple[str, IssueWorkerLiveness]]]:
    """Probe every cap lane once and return each lane minus its deferrals.

    Issue #2051 call-site helper for ``_dispatch_rework_impl``: the dry-run
    partition and the live escalation path run the identical three-lane
    probe+filter, so the shape lives here once instead of being spelled
    out twice in the over-ratchet call-site file. Returns
    ``(filtered_lanes, deferrals)``: ``filtered_lanes`` mirrors the input
    lane order with every deferred issue dropped, and ``deferrals`` is the
    ``_live_worker_cap_escalation_deferrals`` result for the caller to
    report (the dry-run partition) or persist (``_record_cap_escalation_
    deferrals``).

    Read-only, like the probe it wraps -- recording the deferral is the
    caller's decision, so the dry-run partition can use this helper
    without any writes.
    """
    lane_lists = [(list(lane), reason) for lane, reason in lanes]
    deferrals = _live_worker_cap_escalation_deferrals(
        lane_lists,
        issues=issues,
        sessions_dir=sessions_dir,
        config=config,
        now=now,
    )
    filtered = [
        [number for number in lane if number not in deferrals] for lane, _reason in lane_lists
    ]
    return filtered, deferrals


def _defer_cap_escalation_for_live_worker(
    state: dict[str, Any],
    issue_number: int,
    *,
    verdict: IssueWorkerLiveness,
    reason: str,
    write_gate: WriteGate,
) -> dict[str, Any]:
    """Record a cap-escalation deferral while a live worker holds the issue.

    Issue #2051. When a rework cap lane decides ``issue_worker_liveness``
    reports a live worker (see ``_live_worker_cap_escalation_deferrals``),
    the escalation does not fire -- no status/label/counter mutation -- and
    the caller instead stamps an ``escalation_deferred_live_worker`` marker
    on the issue record and emits the same-named event so the held-up cap
    is operator-visible instead of silent. The next pass re-runs the cap
    check: nothing was consumed, so escalation fires as soon as the worker
    exits or wedges.

    ``reason`` is the cap reason the lane WOULD have reported (see
    ``_live_worker_cap_escalation_deferrals``) -- recorded for
    diagnosability only; it is never written to ``escalation_reason`` or
    ``escalation_reasons_seen`` (no escalation happened, and those fields
    feed the per-lane dedup guards).

    The event is emitted once per worker identity: the marker stores
    ``(worker_source, worker_pid, session_started_at)``, and a repeat
    deferral for the same identity rewrites nothing and emits nothing -- a
    worker that stays live for days must not fire every pass. A new worker
    (new session start, or a recycled PID carrying a different recorded
    start) is a different identity and emits again. The marker is
    per-episode bookkeeping: ``charlie unescalate`` clears it via
    ``UNESCALATE_ISSUE_RESET_FIELDS`` so a re-armed issue reports a fresh
    deferral for its next worker.

    Must be called inside ``state_lock`` with freshly loaded state. Writes
    go through ``write_gate`` (Convention B) so the deferral is dry-run-safe
    by construction.
    """
    write_gate = require_write_gate(write_gate)
    state.setdefault("issues", {})
    issue_key = str(issue_number)
    prior = state["issues"].get(issue_key)
    entry = dict(prior) if isinstance(prior, dict) else {}
    marker = entry.get("escalation_deferred_live_worker")
    same_worker = (
        isinstance(marker, dict)
        and marker.get("worker_source") == verdict.source
        and marker.get("worker_pid") == verdict.pid
        and marker.get("session_started_at") == verdict.session_started_at
    )
    if same_worker:
        return state
    entry["number"] = issue_number
    entry["escalation_deferred_live_worker"] = {
        "worker_source": verdict.source,
        "worker_pid": verdict.pid,
        "session_started_at": verdict.session_started_at,
        "deferred_at": utc_now(),
        "reason": reason,
    }
    state["issues"][issue_key] = entry
    return write_gate.record_event(
        state,
        "escalation_deferred_live_worker",
        {
            "issue_number": issue_number,
            "reason": reason,
            "worker_source": verdict.source,
            "worker_pid": verdict.pid,
            "session_started_at": verdict.session_started_at,
            "last_activity_at": verdict.last_activity_at,
            "last_activity_source": verdict.last_activity_source,
            "liveness_reason": verdict.reason,
        },
    )


def _record_cap_escalation_deferrals(
    deferrals: Mapping[int, tuple[str, IssueWorkerLiveness]],
    *,
    write_gate: WriteGate,
) -> None:
    """Persist one ``escalation_deferred_live_worker`` marker+event per issue.

    The batch write path for cap lanes that probed with
    ``_live_worker_cap_escalation_deferrals`` outside the state lock: takes
    ``state_lock`` on the gate's auto-bound ``state_path``, re-loads fresh
    state (the probe ran unlocked, so the locked view may be newer), applies
    ``_defer_cap_escalation_for_live_worker`` per issue, and saves once.
    Under dry-run the gate suppresses both the event and the save.
    """
    write_gate = require_write_gate(write_gate)
    if not deferrals:
        return
    with state_lock(write_gate.state_path):
        state = load_state(write_gate.state_path)
        for issue_number in sorted(deferrals):
            reason, verdict = deferrals[issue_number]
            state = _defer_cap_escalation_for_live_worker(
                state,
                issue_number,
                verdict=verdict,
                reason=reason,
                write_gate=write_gate,
            )
        write_gate.save_state(state)


def _defer_janitor_cap_escalation_for_live_worker(
    issue_number: int,
    issue_state: Mapping[str, Any],
    *,
    attempts_key: str,
    attempts: int,
    max_attempts: int,
    sessions_dir: Path,
    config: OrchestratorConfig,
    write_gate: WriteGate,
    now: datetime,
) -> IssueWorkerLiveness | None:
    """Janitor cap-exceeded gate: defer while a live worker holds the issue.

    Issue #2051, site B. ``_route_janitor_gate_failure_to_rework`` calls
    this after computing the would-be attempt count and BEFORE the
    rescue/escalation branches, so a live verdict defers the whole cap
    decision -- the attempt counter is never persisted, the one-shot
    ``rescue_attempted`` slot is not burned, and no label transitions. The
    next pass re-evaluates and escalates (or rescues) once the worker exits
    or wedges.

    Returns the ``IssueWorkerLiveness`` verdict when the cap decision must
    defer (cap exceeded AND a live worker holds the issue, in which case
    the marker+event are recorded here); ``None`` otherwise -- the cap is
    not exceeded, or the worker is dead/wedged and the caller proceeds with
    its existing rescue/escalation path.
    """
    write_gate = require_write_gate(write_gate)
    if not (max_attempts > 0 and attempts > max_attempts):
        return None
    deferrals = _live_worker_cap_escalation_deferrals(
        (([issue_number], f"{attempts_key}_cap_exceeded"),),
        issues={str(issue_number): issue_state},
        sessions_dir=sessions_dir,
        config=config,
        now=now,
    )
    if issue_number not in deferrals:
        return None
    _record_cap_escalation_deferrals(deferrals, write_gate=write_gate)
    return deferrals[issue_number][1]
