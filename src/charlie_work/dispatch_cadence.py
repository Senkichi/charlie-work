"""Dispatch-cadence bookkeeping (issue #1769; extracted from ``state.py``).

Extracted verbatim from ``state.py``: the durable dispatch-cadence baseline
marker and the edge-triggered stale-alert bookkeeping issue #1769 added
(``_dispatch_cadence``, ``record_non_empty_dispatch``,
``last_non_empty_dispatch``, ``dispatch_baseline_needs_backfill``,
``mark_dispatch_baseline_backfill_attempted``, ``backfill_dispatch_baseline``,
``is_dispatch_stale_alert_due``, ``arm_dispatch_stale_alert``,
``_DISPATCH_STALE_RESOLVED_REASONS``, ``clear_dispatch_stale_alert``).

``state.py`` re-exports every symbol here via a facade import block
(mirroring ``workflow.py``'s re-export of ``ci_findings.py`` and the rest of
the #1283 Phase-A extraction lineage), so existing import paths
(``charlie_work.state.record_non_empty_dispatch`` and friends) and
monkeypatch targets keep working unchanged.

Moved out of ``state.py`` as part of the #1769 review follow-up rather than
left in place: the review's own fixes (the BLOCKER-required one-time
events.db backfill, the MAJOR-required resolved-reason gating on
``clear_dispatch_stale_alert``) grew this cluster past the point where
``state.py`` -- already an over-cap monolith tracked by the file-size
ratchet, issue #1442 -- could absorb it without breaching its recorded
high-water mark (``file_size_ratchet_baseline/``). A byte-identical
extraction shrinks ``state.py`` back under its mark and passes the ratchet
trivially, the same remedy ``tests/test_file_size_ratchet.py`` names for any
over-cap-monolith growth.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def _dispatch_cadence(data: dict[str, Any]) -> dict[str, Any]:
    """Return the dispatch-cadence bookkeeping sub-dict from ``data``.

    Issue #1769: durable, non-count-bounded replacement for
    ``ci_findings._latest_non_empty_dispatch``'s prior events.db
    ``query_events(kind="dispatch", limit=100)`` lookback, which could
    scroll the last real non-empty dispatch out of its window during a
    sustained stall -- the exact failure mode this section exists to make
    structurally impossible: a durable field cannot fall out of a rolling
    window no matter how long the stall runs. Also carries the
    edge-triggered ``dispatch_stale`` alert bookkeeping
    (``last_stale_alert_at``) so the warning fires once at stall onset plus
    a bounded low-rate reminder, never unconditionally every pass while the
    condition still holds.

    Ensures a mutable copy so callers can build new state without mutating
    ``data``, mirroring ``_deescalation_pass``.
    """
    section = data.get("dispatch_cadence")
    if not isinstance(section, dict):
        return {}
    return dict(section)


def record_non_empty_dispatch(
    data: dict[str, Any], at: str, issue_numbers: list[int]
) -> dict[str, Any]:
    """Persist the durable baseline marker for the latest non-empty dispatch.

    Callers record this every time a dispatch pass actually launches one or
    more issues (issue #1769). ``ci_findings.check_dispatch_staleness``
    reads it back directly instead of re-deriving "when did dispatch last do
    something" from a windowed events.db scan -- an O(1) dict read that
    cannot degrade no matter how many empty dispatch passes accumulate
    afterward, unlike the count-bounded lookback it replaces.

    Returns a new state dict; does not mutate ``data``.
    """
    section = _dispatch_cadence(data)
    section["last_non_empty_dispatch_at"] = at
    section["last_non_empty_dispatch_issue_numbers"] = sorted(issue_numbers)
    return {**data, "dispatch_cadence": section}


def last_non_empty_dispatch(data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the durable ``{"ts", "issue_numbers"}`` baseline, or ``None``.

    ``None`` means no non-empty dispatch has ever been recorded through
    :func:`record_non_empty_dispatch` for this repo (a fresh deploy, or a
    repo that has not yet had a dispatch pass since this marker was
    introduced) -- callers treat that the same as the prior ``no_baseline``
    reason, unless :func:`backfill_dispatch_baseline` has already recovered
    a value from ``events.db`` history.
    """
    section = _dispatch_cadence(data)
    at = section.get("last_non_empty_dispatch_at")
    if not at:
        return None
    return {
        "ts": at,
        "issue_numbers": list(section.get("last_non_empty_dispatch_issue_numbers", [])),
    }


def dispatch_baseline_needs_backfill(data: dict[str, Any]) -> bool:
    """True when the durable baseline is absent and has never been backfilled.

    Issue #1769 follow-up (review blocker): ``record_non_empty_dispatch`` is
    only ever written going *forward*, from the moment a dispatch pass
    actually launches something. A repo that is already mid-stall the moment
    this marker is introduced -- which is exactly the reported live
    symptom -- never reaches that write path, so ``last_non_empty_dispatch``
    would return ``None`` forever and the detector would report
    ``no_baseline`` (not ``stale``) permanently, never recovering. This
    predicate gates the one-time ``events.db`` recovery scan in
    :func:`backfill_dispatch_baseline` so it runs at most once per repo,
    regardless of whether that scan finds a match.
    """
    section = _dispatch_cadence(data)
    if section.get("last_non_empty_dispatch_at"):
        return False
    return not section.get("baseline_backfill_attempted", False)


def mark_dispatch_baseline_backfill_attempted(data: dict[str, Any]) -> dict[str, Any]:
    """Record that the one-time ``events.db`` baseline recovery scan has run.

    Set regardless of whether the scan found a matching event, so a repo
    with no non-empty dispatch in its entire history (a genuinely fresh repo)
    does not re-pay the full-table-scan cost on every subsequent pass.
    Returns a new state dict; does not mutate ``data``.
    """
    section = _dispatch_cadence(data)
    section["baseline_backfill_attempted"] = True
    return {**data, "dispatch_cadence": section}


def backfill_dispatch_baseline(data: dict[str, Any], state_path: Path) -> dict[str, Any]:
    """One-time recovery of the durable dispatch baseline from ``events.db``.

    Issue #1769 blocker: without this, a repo that already existed (or was
    already mid-stall) before the durable marker was introduced starts with
    no ``dispatch_cadence`` key and can never populate one on its own -- a
    stalled repo by definition never takes the ``record_non_empty_dispatch``
    write path. This scans the full ``dispatch``-kind history in
    ``events.db`` for the newest event whose ``issue_numbers`` payload is
    non-empty and seeds the marker from it, exactly once per repo
    (:func:`dispatch_baseline_needs_backfill` gates re-entry so the O(events)
    scan is a single one-time cost, not a recurring one -- the old windowed
    ``query_events(limit=100)`` scan this replaces was cheap but count-bounded;
    this is unbounded but pays its cost only once). A repo with no non-empty
    dispatch anywhere in its history is left with no baseline, same as
    today -- this recovers real history, it does not fabricate one.

    No-ops (aside from being idempotent) when a baseline already exists or a
    prior backfill attempt already ran. Returns a new state dict; does not
    mutate ``data``.
    """
    if not dispatch_baseline_needs_backfill(data):
        return data

    from .instrumentation import query_events

    found: dict[str, Any] | None = None
    for event in reversed(query_events(state_path, kind="dispatch")):
        issue_numbers = event.get("payload", {}).get("issue_numbers") or []
        if issue_numbers:
            found = {"ts": event["ts"], "issue_numbers": list(issue_numbers)}
            break

    new_data = mark_dispatch_baseline_backfill_attempted(data)
    if found is not None:
        new_data = record_non_empty_dispatch(new_data, found["ts"], found["issue_numbers"])
    return new_data


def is_dispatch_stale_alert_due(
    data: dict[str, Any], *, now: datetime, reminder_minutes: int
) -> bool:
    """True when a ``dispatch_stale`` warning should be (re-)emitted now.

    Issue #1769 (design artifact section 6 policy): a warning describing a
    *condition* that continues to hold across passes must be edge-triggered,
    never re-fired every pass merely because the condition still holds. An
    absent ``last_stale_alert_at`` means this is a fresh stall onset (the
    edge) -- always due. Otherwise due once ``reminder_minutes`` have
    elapsed since the last alert, a bounded low-rate reminder so a
    long-running stall does not go fully silent between edges. A malformed
    *or timezone-naive* timestamp, or a non-positive ``reminder_minutes``, is
    treated as due, mirroring ``is_reconcile_due``'s "corrupt/unset schedule
    is due immediately" contract -- the comparison below runs inside the
    ``try`` (not after it) specifically so a naive ``last_time`` raises
    ``TypeError`` on the aware/naive comparison and is caught here rather
    than propagating out of the state-lock-held dispatch pass that calls
    this.
    """
    last_at = _dispatch_cadence(data).get("last_stale_alert_at")
    if not last_at or reminder_minutes <= 0:
        return True
    try:
        last_time = datetime.fromisoformat(str(last_at).replace("Z", "+00:00"))
        return now >= last_time + timedelta(minutes=reminder_minutes)
    except (ValueError, TypeError):
        return True


def arm_dispatch_stale_alert(data: dict[str, Any], at: str) -> dict[str, Any]:
    """Record that a ``dispatch_stale`` warning was just emitted at ``at``.

    Returns a new state dict; does not mutate ``data``.
    """
    section = _dispatch_cadence(data)
    section["last_stale_alert_at"] = at
    return {**data, "dispatch_cadence": section}


# Issue #1769 review follow-up: `check_dispatch_staleness`'s non-stale
# `reason` values that mean "we affirmatively know nothing needs an alarm
# right now" -- as opposed to `backlog_not_observed` (a failed fetch --
# unknown, not resolved), `no_baseline` (no dispatch history to judge
# staleness against -- unknown), and `threshold_disabled` (detection turned
# off -- nothing was determined either way). Only these reset the
# alert-cadence marker; see `clear_dispatch_stale_alert`. `empty_backlog` and
# `all_ready_blocked_by_dependencies` are both genuinely-idle, observed
# states (nothing dispatchable and nothing wrong), not unknowns -- the same
# distinction `_backlog_is_non_empty`'s `observed` check draws.
_DISPATCH_STALE_RESOLVED_REASONS = frozenset(
    {
        "current_pass_dispatched",
        "within_threshold",
        "empty_backlog",
        "all_ready_blocked_by_dependencies",
    }
)


def clear_dispatch_stale_alert(
    data: dict[str, Any], *, reason: str | None = None
) -> dict[str, Any]:
    """Reset the alert marker once the backlog is genuinely no longer stale.

    ``reason`` is ``check_dispatch_staleness``'s ``result["reason"]`` for a
    pass that found ``stale`` False. Resetting on *every* non-stale reason
    (the prior behavior) treated "unknown" the same as "resolved": a
    transient ``backlog_not_observed`` reading (e.g. a GitHub API blip)
    would wipe the marker, so the very next pass -- once the real,
    still-ongoing stall is observed again -- is treated as a fresh edge and
    re-fires immediately, defeating the bounded low-rate reminder this
    marker exists to enforce. Only a reason in
    ``_DISPATCH_STALE_RESOLVED_REASONS`` -- a reading that affirmatively
    knows nothing needs an alarm right now -- resets it. ``reason=None``
    (a caller that predates this parameter) is treated conservatively as
    *not* a genuine resolution, the same as the "unknown" reasons.

    Leaves the ``last_non_empty_dispatch_at`` baseline marker untouched in
    every case; only the alert-cadence field is ever reset. Returns a new
    state dict; does not mutate ``data``.
    """
    if reason not in _DISPATCH_STALE_RESOLVED_REASONS:
        return data
    section = _dispatch_cadence(data)
    if section.get("last_stale_alert_at") is None:
        return data
    section["last_stale_alert_at"] = None
    return {**data, "dispatch_cadence": section}
