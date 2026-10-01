"""The stalled-session lane's pure decision: ``decide_stalled(facts, observed)``.

One worker at a time. A worker whose PID is alive but whose agent stopped
progressing (``WorkerHealth.STALLED``) -- or whose process is already gone
(``DEAD``) -- is either deferred (a provider rate limit explains the silence),
reaped because it blew its api budget, or reaped as stalled/exited. The flow
below is the decision table; every effect is a request whose result it reads or
a commit whose content is already decided. Nothing here reads a clock, the
filesystem, GitHub or state: ``facts.now`` is the pass's single clock sample.

The history the original function carried (issues #246, #247, #261, #338, #484,
#873, #1325, #1917) is kept on the branch that each one shaped.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..worker import WorkerHealth
from .decide_common import Flow, run_flow
from .stalled_model import (
    STALLED_COMMIT_TYPES,
    FailureStamp,
    KillOrphans,
    KillTree,
    MarkBudgetExceeded,
    ProbeCompletedHandoff,
    ProbeHealth,
    ProbeRateLimitDefer,
    ReadAdapterProfile,
    ReadLogTail,
    RecordFailure,
    RecordPostMortem,
    RecordRoleQuota,
    StalledFacts,
    StalledPlan,
    StampSidecar,
    StateTxn,
    ThrottleArm,
)

REAP_SOURCE = "stalled_sessions_reap"
DEFER_SOURCE = "stalled_sessions_rate_limit_defer"


def _parse_defer_until(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    except (ValueError, TypeError):
        return None


def _is_deferred(facts: StalledFacts) -> bool:
    """True if the worker has a stored defer deadline that has not passed."""
    defer_until = _parse_defer_until(facts.worker.rate_limit_defer_until)
    if defer_until is None:
        return False
    return facts.now < defer_until


def _pairs(**fields: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(fields.items())


def _kill(facts: StalledFacts) -> Flow:
    """Tree kill, then orphan sweep; returns ``(killed_pids, orphan_pids)``."""
    issue = facts.worker.issue_number
    tree = yield KillTree(issue)
    orphans = yield KillOrphans(issue)
    return [*tree, *orphans], list(orphans)


def _budget_flow(facts: StalledFacts) -> Flow:
    """#484: an api worker over its per-session cap is killed and sidecar-marked."""
    w = facts.worker
    killed, orphans = yield from _kill(facts)
    if not facts.dry_run:
        yield MarkBudgetExceeded(w.issue_number)
    yield StateTxn(
        event_kind="session_budget_exceeded",
        event_payload=_pairs(
            issue_number=w.issue_number,
            pid=w.pid,
            process_start_time=w.process_start_time,
            killed_pids=killed,
            orphan_pids=orphans if orphans else None,
            provider=w.provider,
        ),
    )


def _reap_flow(facts: StalledFacts, health: WorkerHealth, probe: Any, adapter: Any) -> Flow:
    """Kill, classify the sidecar, arm any throttle, and record the reap event."""
    w = facts.worker
    issue = w.issue_number
    killed, orphans = yield from _kill(facts)
    if not facts.dry_run:
        yield RecordPostMortem(issue)
    failure_kind: str | None = None
    throttled_until: str | None = None
    # A DEAD worker that already handed off a fresh completed outcome did not
    # stall: the "stalled" fallback would be a false label (#2104), so it is not
    # classified at all. Only a live, non-progressing (STALLED) worker earns it.
    handed_off = health is WorkerHealth.DEAD and (yield ProbeCompletedHandoff(issue))
    if not handed_off and not facts.dry_run and adapter.can_record_failure:
        failure_kind, throttled_until = yield RecordFailure(issue)
    if failure_kind and throttled_until:
        yield StateTxn(
            throttle=ThrottleArm(throttled_until, REAP_SOURCE, failure_kind, w.adapter_kind)
        )
    tail = yield ReadLogTail(issue)
    probe_payload = (
        probe.to_payload()
        if probe is not None
        else {"sources": [], "latest_timestamp": None, "latest_source": "probe unavailable"}
    )
    # #873: STALLED (a live process that stopped progressing) is a fault; DEAD
    # (the process is already gone -- also the normal end of a finished worker)
    # is only a warning-level ``session_exited``.
    event_kind = "session_stalled" if health is WorkerHealth.STALLED else "session_exited"
    yield StateTxn(
        stamp=(
            FailureStamp(issue, failure_kind, w.adapter_kind) if failure_kind is not None else None
        ),
        event_kind=event_kind,
        event_payload=_pairs(
            issue_number=issue,
            pid=w.pid,
            process_start_time=w.process_start_time,
            worker_health=health.name,
            log_mtime=tail.mtime,
            last_log_line=tail.last_line,
            killed_pids=killed,
            orphan_pids=orphans if orphans else None,
            failure_kind=failure_kind,
            activity_sources=probe_payload.get("sources", []),
            latest_real_activity_at=probe_payload.get("latest_timestamp"),
            latest_real_activity_source=probe_payload.get("latest_source"),
        ),
    )


def stalled_flow(facts: StalledFacts) -> Flow:
    """One worker's decision; returns the ``{"issue", "pid"}`` row, or ``None``."""
    w = facts.worker
    if w.pid is None or w.error is not None:
        return None
    issue = w.issue_number
    entry = {"issue": issue, "pid": w.pid}

    adapter = yield ReadAdapterProfile(issue)
    # #1325: sidecar writes are suppressed under dry-run; detection reads only
    # the pre-fetched ``WorkerView`` so skipping them leaves this pass intact.
    if not facts.dry_run:
        yield StampSidecar(issue)
    observed_health = yield ProbeHealth(issue)
    health = observed_health.health
    # #338: this lane is the sole writer of the inconclusive-probe counter.
    if not facts.dry_run:
        yield StampSidecar(
            issue,
            _pairs(inconclusive_probe_deferred_count=observed_health.next_inconclusive_count),
        )

    if adapter.over_budget:
        yield from _budget_flow(facts)
        return entry

    # Still inside a stored rate-limit defer window: leave it alone.
    if health == WorkerHealth.STALLED and _is_deferred(facts):
        return None

    if health not in (WorkerHealth.STALLED, WorkerHealth.DEAD):
        return None

    # #247: before killing a stalled-looking worker, look for a rate-limit
    # signature in the log tail; if found, record a defer deadline and skip.
    if (
        health == WorkerHealth.STALLED
        and facts.config.watchdog.rate_limit_defer_enabled
        and not _is_deferred(facts)
        and w.rate_limit_defer_until is None
    ):
        defer_until = yield ProbeRateLimitDefer(issue)
        if defer_until is not None:
            if not facts.dry_run:
                yield StampSidecar(issue, _pairs(rate_limit_defer_until=defer_until))
                # Issue #2086: the fleet ledger restricts the stamped chain entry.
                yield RecordRoleQuota(issue, defer_until, "rate_limited", DEFER_SOURCE)
            yield StateTxn(
                throttle=ThrottleArm(defer_until, DEFER_SOURCE, "rate_limited", w.adapter_kind),
                event_kind="session_rate_limit_deferred",
                event_payload=_pairs(issue_number=issue, pid=w.pid, defer_until=defer_until),
            )
            return None

    yield from _reap_flow(facts, health, observed_health.probe, adapter)
    return entry


def decide_stalled(facts: StalledFacts, observed: dict[Any, Any]) -> StalledPlan:
    """Replay the worker's flow against ``observed``; see ``run_flow``."""
    value, commits, pending = run_flow(
        stalled_flow(facts), observed, commit_types=STALLED_COMMIT_TYPES
    )
    return StalledPlan(
        requests=() if pending is None else (pending,),
        commits=commits,
        done=pending is None,
        entry=value,
    )
