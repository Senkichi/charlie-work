"""The timed dead-dispatched backstop (#654/#1917/#1971/#1993) as a decision flow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..dead_dispatched_timer import defer_or_expire_local_park
from ..worker_fate import persisted_failure
from .decide_common import Draft, Flow, LockAcc, emit, events_to_emits, flush, reap_due
from .model import Escalate, PreOutcome, ReadTerminal, SweepFacts


def reap_flow(
    facts: SweepFacts,
    pre: PreOutcome,
    draft: Draft,
    pr_data: Mapping[str, Any] | None,
    acc: LockAcc,
) -> Flow:
    """Returns True when the entry was escalated (the caller records it and moves on)."""
    number = draft.issue
    entry = draft.work
    config = facts.config
    now = pre.now
    reap_minutes = config.watchdog.dead_dispatched_reap_minutes
    if not reap_due(
        entry, throttled_until=acc.throttled_until, reap_minutes=reap_minutes, now=now
    ):
        return False

    orphan_drift_at = entry.get("orphan_drift_at")
    failure = persisted_failure(entry)
    rearm_count = int(entry.get("throttle_reap_rearm_count") or 0)
    max_rearms = config.watchdog.max_auto_redispatch
    if failure.is_throttle and pr_data is None and rearm_count < max_rearms:
        entry["orphan_drift_at"] = now.isoformat().replace("+00:00", "Z")
        entry["throttle_reap_rearm_count"] = rearm_count + 1
        yield emit(
            "dead_dispatched_throttle_rearmed",
            {
                "issue_number": number,
                "failure_kind": failure.kind,
                "previous_orphan_drift_at": orphan_drift_at,
                "throttled_until": acc.throttled_until,
                "rearm_count": rearm_count + 1,
                "max_rearms": max_rearms,
            },
        )
        return False

    park_deferral = pre.deferred.get(number)
    if park_deferral is not None:
        events: list[tuple[str, dict[str, Any]]] = []
        deferred = defer_or_expire_local_park(
            entry,
            issue_number=number,
            reason=park_deferral,
            now=now,
            orphan_drift_at=orphan_drift_at,
            sweep_events=events,
        )
        for item in events_to_emits(events):
            yield item
        if deferred:
            return False

    pr_number = int(pr_data["number"]) if pr_data else None
    terminal = yield ReadTerminal(number)
    yield from flush(draft)
    yield Escalate(
        number,
        reason="dead_dispatched_worker_reap",
        reason_class="mechanical",
        pr_number=pr_number,
        issue_extra={
            "dispatched_at": None,
            "orphan_drift_fingerprint": None,
            "orphan_drift_at": None,
            "local_park_defer_count": None,
            "local_park_defer_since": None,
            "local_park_defer_pass_key": None,
            "local_park_defer_reason": park_deferral,
        },
    )
    draft.invalidate()
    yield emit(
        "dead_dispatched_worker_reaped",
        {
            "issue_number": number,
            "pr_number": pr_number,
            "previous_status": "dispatched",
            "reason": "dead_dispatched_worker_reap",
            "orphan_drift_at": orphan_drift_at,
            "reap_minutes": reap_minutes,
            "exit_code": terminal.exit_code,
            "local_park_defer_error": park_deferral,
        },
    )
    return True
