"""Pure event -> fact-row derivation for the dashboard rollup (events recon section 3).

``derive_event`` maps one ``events`` row to ``(table, columns)`` pairs; it does no I/O.
Kinds without a handler (including the noise kinds) yield nothing, so the handler
registry doubles as the allow-list the rollup selects from.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .rollup_common import Row, _dict, _flag, _int, _ints, _milestone, _refs, reason_group
from .rollup_flow_handlers import FLOW_HANDLERS, escalation
from .rollup_schema import JOB_TABLE

# Excluded from every rollup (recon section 4): ~25% of cw rows / repeated every pass.
# ``unauthorized_merge_queue_sync_covered`` has no handler; ``reconcile`` is handled but
# only its ``merged_outside_orchestrator`` sub-kind emits (``terminal_state_stale`` never).
NOISE_KINDS = frozenset({"unauthorized_merge_queue_sync_covered"})
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
assert not NOISE_KINDS & HANDLERS.keys()  # a noise kind must never gain a handler


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
