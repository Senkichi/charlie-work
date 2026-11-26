"""Pure event -> fact-row derivation for the dashboard rollup (events recon section 3).

``derive_event`` maps one ``events`` row to ``(table, columns)`` pairs; it does no I/O.
Kinds without a handler yield nothing, so the handler registry doubles as the
allow-list the rollup selects from. A kind with a level file in the Event level
registry is ignored by default; ``KNOWN_IGNORED`` is the residual list of
un-interpreted kinds without one, each with a one-line reason. Any other kind is
*unclassified* and counted per source so it surfaces instead of silently dropping
(issues #2269, #3093).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# The kind is spelled via the constant, never a literal: the #2262 salvage-seam
# AST guard (tests/test_dead_worker_salvage_seam.py) counts any bare
# "session_failed_relabeled" constant as a dead-worker requeue locus, which this
# read-side classification is not.
from ..dead_worker_sweep.decide_common import SESSION_FAILED_RELABELED
from ..instrumentation import has_level_entry
from .rollup_common import Row, _dict, _flag, _int, _ints, _milestone, _refs, reason_group
from .rollup_flow_handlers import FLOW_HANDLERS, escalation
from .rollup_schema import JOB_TABLE

# Residual list: kinds the rollup deliberately does not interpret that have NO entry in
# the Event level registry (``event_levels/<kind>.level``), each with its one-line
# reason. A kind with a level file and no handler is ignored by default, so it needs no
# entry here (issue #3093); this list holds only kinds without a level file
# (``test_known_ignored_holds_only_kinds_without_a_level_file``). A kind in an events DB
# that is handled, level-registered, listed here, or the ``_sweep`` form of an ignored
# kind is *classified*; any other is *unclassified*: the rollup counts it per source
# (``SourceResult.unclassified``) and warns when the set changes (issue #2269). An emitted
# kind that is none of these fails ``test_every_emitted_kind_is_rollup_classified``.
KNOWN_IGNORED: dict[str, str] = {
    # -- Batch (<kind>_sweep) kinds with their own name but no level file.
    "orphaned_worker_drift_sweep": "batch form of orphaned_worker_drift",
    "review_dispatch_stalled_sweep": "batch form of review_dispatch_stalled",
    "review_dispatch_lifecycle_reaped_sweep": "batch form of review_dispatch_lifecycle_reaped",
    # -- Retired, legacy, and bookkeeping kinds with no level file.
    "cross_family_regen_not_reached": "retired cross-family auto-gate surface (deleted in role-config phase 2); kept for older events DBs",
    "cross_family_report_regen_exhausted": "retired cross-family auto-gate surface (deleted in role-config phase 2); kept for older events DBs",
    "cross_family_report_regen_forced": "retired cross-family auto-gate surface (deleted in role-config phase 2); kept for older events DBs",
    "cross_family_verdict_abandoned": "retired cross-family auto-gate surface (deleted in role-config phase 2); kept for older events DBs",
    "cross_family_verdict_head_indeterminate": "retired cross-family auto-gate surface (deleted in role-config phase 2); kept for older events DBs",
    "cross_family_verdict_unparseable": "retired cross-family auto-gate surface (deleted in role-config phase 2); kept for older events DBs",
    "label_ensure_ok": "label-ensure diagnostic; label state is read from GitHub, not this event",
    "label_ensure_incomplete": "label-ensure diagnostic; label state is read from GitHub, not this event",
    "operator_local_park": "operator action audit record (legacy kind); no flow fact",
    "operator_orphan_push_completed": "operator recovery audit record (legacy kind); no flow fact",
    "operator_orphan_requeue": "operator recovery audit record (legacy kind); no flow fact",
    "operator_probe_advanced": "operator probe audit record (legacy kind); no flow fact",
    "operator_queue_depth": "retired per-pass operator-queue gauge (#1768); replaced by operator_queue_impact",
    "operator_reviewer_quota_cleared": "operator quota-clear audit record (legacy kind); no flow fact",
    "operator_rework_rearmed": "operator rework-rearm audit record (legacy kind); no flow fact",
    "operator_state_correction": "operator state-correction audit record (legacy kind); no flow fact",
    "review_packet_discarded_head_moved": "review-packet bookkeeping: PR head moved, packet re-made",
    "rework_label_skipped_issue_closed": "rework-label bookkeeping: issue already closed",
    "unauthorized_merge_ack_revoked": "unauthorized-merge bookkeeping",
    "worker_token_missing": "retired worker-token gate (#1877); kept for older events DBs",
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
            "parked_unready": _int(br.get("parked_unready")),
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


# Spelled through an alias of the constant: the #2262 salvage-seam AST guard
# counts the constant (or a bare literal) passed as a call argument as a
# dead-worker requeue locus, which this read-side milestone is not.
_RELABELED_MILESTONE = SESSION_FAILED_RELABELED


def _session_failed_relabeled(ev: dict) -> list[Row]:
    """A dead worker's issue was handed back to dispatch: the approx
    ``in_progress`` stage ends here (issue #2473). The sweep-batch summary
    expands to the same per-issue milestone."""
    issue, pr = _refs(ev)
    return [_milestone(ev, _RELABELED_MILESTONE, issue, pr)]


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
    cause = _dict(p.get("cause"))
    api_status = _int(cause.get("api_error_status"))
    raw_cause = cause.get("cause")
    if api_status is not None:
        cause_label = f"api_error:{api_status}"
    elif isinstance(raw_cause, str) and raw_cause:
        cause_label = raw_cause
    else:
        cause_label = None
    reason = p.get("reason") if isinstance(p.get("reason"), str) else None
    group = reason_group(reason)
    detail = p.get("detail") if isinstance(p.get("detail"), str) else None
    if detail is None and reason is not None and reason != group:
        # Pre-#2476 events kept the free-text message in ``reason``; keep it
        # reachable as ``detail`` so the token-vs-message split holds for every
        # row regardless of when it was written.
        detail = reason
    row = {
        "issue": _int(issue),
        "pr": _int(pr),
        "reason": reason,
        "reason_group": group,
        "detail": detail,
        "cause": cause_label,
        "exit_code": _int(cause.get("exit_code")),
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
    SESSION_FAILED_RELABELED: _session_failed_relabeled,
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


def _is_ignored(kind: str) -> bool:
    """True when ``kind`` is not handled but is level-registered or in the residual list."""
    return kind not in HANDLERS and (has_level_entry(kind) or kind in KNOWN_IGNORED)


def is_classified(kind: str) -> bool:
    """True when the rollup handles ``kind`` or deliberately ignores it (issues #2269, #3093).

    A kind is ignored by default when it has an entry in the Event level registry
    (``has_level_entry``) and no handler, so a new kind with a level file needs no
    dashboard edit; ``KNOWN_IGNORED`` holds only the kinds without a level file. If the
    level registry failed to load it is empty and those kinds surface as unclassified
    (noisy, not silent).

    A ``<kind>_sweep`` batch summary (minted by the reaper from ``sweep_events.append``)
    inherits an *ignored* sibling's classification: it is the same record folded into one
    event, so it needs no entry of its own (issue #2459). A handled kind's ``_sweep`` does
    not inherit: handled sweeps are explicit in ``HANDLERS`` because their per-issue
    expansion is handler-specific.
    """
    if kind in HANDLERS or _is_ignored(kind):
        return True
    return kind.endswith(SWEEP_SUFFIX) and _is_ignored(kind[: -len(SWEEP_SUFFIX)])


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
