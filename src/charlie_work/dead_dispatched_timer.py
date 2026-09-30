"""The ``dead_dispatched_reap_minutes`` due-check shared by both sides of the
issue #1971 split.

The #654 timed backstop (``orphaned_worker_sweep.maybe_reap_dead_dispatched_
worker``) escalates a dead ``dispatched`` entry once ``orphan_drift_at`` has
been armed longer than ``dead_dispatched_reap_minutes``. Issue #1971 added a
pre-lock salvageable-park drain on no-PR backends
(``local_work_park.park_backstop_due_local_orphans``) that must select exactly
the same "backstop-due" entries the in-lock timer is about to escalate -- the
park itself takes ``state_lock`` and does git/label I/O, so it can never run
inside the locked classification where the timer lives. Housing the predicate
here (rather than in either consumer) keeps the two due-checks provably
identical and avoids an import cycle between the sweep and the park module.

The same module also holds the *bound* on the #1971 deferral both lanes
share: a failed or inconclusive park/probe defers the escalation/reclaim only
while ``LOCAL_PARK_DEFER_MAX_PASSES`` consecutive deferred passes remain --
tracked on the state entry -- after which the lane resolves anyway. Without
the bound a deterministic git failure (a missing base ref, no merge base)
would wedge the entry exactly the way the unbounded pre-#1971 probe did.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# Consecutive deferred passes an issue gets while the salvageable-work probe
# keeps failing before the lane resolves anyway (escalate for the #654
# backstop, reclaim for the active-labeled lane). Deferred passes are counted
# on the state entry so the bound survives restarts and is shared by both
# lanes. 3 mirrors the other convergence bounds the local lane uses
# (``LOCAL_SUITE_GATE_MAX_RESYNCS``/``_MAX_ORPHANS``): a git probe failure that
# outlives three full sweep passes is not transient.
LOCAL_PARK_DEFER_MAX_PASSES = 3

# State-entry fields carrying the deferral bookkeeping (issue #1971). Per
# death episode: cleared wherever the death classification is cleared
# (``state.clear_dead_worker_failure_kind``, the unescalate reset fields, and
# the park's own status flip).
LOCAL_PARK_DEFER_FIELDS = (
    "local_park_defer_count",
    "local_park_defer_since",
    "local_park_defer_pass_key",
    "local_park_defer_reason",
)


def dead_dispatched_reap_due(
    *,
    state: dict[str, Any],
    entry: dict[str, Any],
    dead_dispatched_reap_minutes: float,
    now: datetime,
) -> bool:
    """Whether the #654 timed backstop would escalate this entry this pass.

    The pure-predicate half of
    ``orphaned_worker_sweep.maybe_reap_dead_dispatched_worker``, extracted
    (issue #1971) so the sweep's pre-lock salvageable-park probe can select
    exactly the backstop-due entries without duplicating the drift-expiry or
    provider-throttle exemption logic. Issue #1993 anchored the throttle-death
    grace at ``max(orphan_drift_at, throttled_until)`` once the provider
    window lapses; the bounded re-arm on top of that (``throttle_reap_rearm_
    count``) stays in the caller -- it mutates the entry, so it is not part
    of the read-only predicate.
    """
    # Deferred: workflow.py imports this module's consumers top-level, and
    # worker_fate -> state -> this module would close a second cycle; a
    # top-level import of either would break at module init. Attribute
    # access through the module objects also keeps suite patches on
    # ``charlie_work.workflow.<name>`` / ``charlie_work.worker_fate.<name>``
    # live.
    import charlie_work.worker_fate as _fate
    import charlie_work.workflow as _wf

    orphan_drift_at = entry.get("orphan_drift_at")
    drift_dt = _wf._parse_iso_timestamp(orphan_drift_at) if orphan_drift_at else None
    if drift_dt is None or dead_dispatched_reap_minutes <= 0:
        return False
    grace_anchor = drift_dt
    # A6 (worker_fate §6): the persisted failure kind is read through the
    # single accessor -- the raw ``dead_worker_failure_kind`` key is confined
    # to state.py/worker_fate.py by the AST seam guard.
    if _fate.persisted_failure(entry).is_throttle:
        throttled_until_dt = _wf._parse_iso_timestamp(state.get("throttled_until"))
        if throttled_until_dt is not None:
            if throttled_until_dt > now:
                return False
            grace_anchor = max(drift_dt, throttled_until_dt)
    return (now - grace_anchor).total_seconds() / 60 >= dead_dispatched_reap_minutes


def note_local_park_deferral(entry: dict[str, Any], *, now: datetime, reason: str) -> int:
    """Record one deferred pass on ``entry``; return the consecutive count.

    Idempotent within a sweep pass: the reclaim lane (pre-lock, via
    ``local_work_park``) and the in-lock backstop can both observe the same
    pass's deferral -- the pass key (the sweep's ``now``) makes the second
    note a no-op, so one pass never consumes two deferral passes. On the
    first deferral of a run ``local_park_defer_since`` is stamped, giving
    the forensic record a start timestamp for the wedge.
    """
    pass_key = now.isoformat()
    if entry.get("local_park_defer_pass_key") == pass_key:
        return int(entry.get("local_park_defer_count") or 0)
    count = int(entry.get("local_park_defer_count") or 0) + 1
    entry["local_park_defer_count"] = count
    entry["local_park_defer_pass_key"] = pass_key
    entry["local_park_defer_reason"] = reason
    if count == 1:
        entry["local_park_defer_since"] = pass_key.replace("+00:00", "Z")
    return count


def defer_or_expire_local_park(
    entry: dict[str, Any],
    *,
    issue_number: int,
    reason: str,
    now: datetime,
    orphan_drift_at: Any,
    sweep_events: list[tuple[str, dict[str, Any]]],
) -> bool:
    """Bounded #1971 deferral for the in-lock backstop.

    Counts this pass's deferral (``note_local_park_deferral``) and, while
    the budget remains, surfaces it once per distinct reason through the
    same fingerprinted ``orphaned_worker_drift`` audit shape the drift
    branches use -- returns True (defer). Once
    ``LOCAL_PARK_DEFER_MAX_PASSES`` consecutive deferred passes are spent,
    returns False so the caller proceeds to the escalation it would have
    run had the park probe proved the branch empty; the probe error rides
    along on the entry (``local_park_defer_reason``) and the caller's
    escalation payload.
    """
    defer_count = note_local_park_deferral(entry, now=now, reason=reason)
    if defer_count >= LOCAL_PARK_DEFER_MAX_PASSES:
        return False
    fingerprint = f"local_park_reap_deferred:{reason}"
    if entry.get("orphan_drift_fingerprint") != fingerprint:
        entry["orphan_drift_fingerprint"] = fingerprint
        sweep_events.append(
            (
                "orphaned_worker_drift",
                {
                    "issue_number": issue_number,
                    "previous_status": "dispatched",
                    "reason": "dead_dispatched_local_park_deferred",
                    "detail": reason,
                    "orphan_drift_at": orphan_drift_at,
                    "defer_count": defer_count,
                    "defer_max": LOCAL_PARK_DEFER_MAX_PASSES,
                },
            )
        )
    return True


def clear_local_park_deferral(entry: dict[str, Any]) -> None:
    """Drop the #1971 deferral bookkeeping from ``entry`` in place."""
    for field_name in LOCAL_PARK_DEFER_FIELDS:
        entry.pop(field_name, None)
