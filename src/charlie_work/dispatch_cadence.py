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

Issue #1682 added ``_past_dated_state_timestamps`` and
``_dependency_root_blocker_progress`` -- the root-blocker progress scan that
bounds the #1110 dependency-blocked exemption in
``ci_findings.check_dispatch_staleness``. They live here rather than in
``ci_findings.py`` because that module is at the file-size ratchet cap
(issue #1442), and they are staleness bookkeeping of the same kind this
module already holds.
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


# ---------------------------------------------------------------------------
# Issue #1682: dependency-root-blocker progress bookkeeping -- the bounded
# form of ci_findings.check_dispatch_staleness's #1110 exemption. Lives here
# rather than in ci_findings.py because that module is at the file-size
# ratchet cap (issue #1442); these are staleness bookkeeping, same theme as
# the baseline marker and alert-cadence arms above.
# ---------------------------------------------------------------------------


def _past_dated_state_timestamps(entry: dict[str, Any], now: datetime) -> list[datetime]:
    """Every parseable ``*_at``/``*_since`` timestamp on a state entry that is
    not in the future (issue #1682).

    The ``*_at``/``*_since`` naming convention covers the durable lifecycle
    stamps this codebase writes (``dispatched_at``, ``reviewed_at``,
    ``terminal_since``, ``stale_checks_last_retrigger_at``, …) without a
    hardcoded field list that would silently miss the next stamp added. Only
    PAST-dated values count: a ``next_*_at``-style field records a scheduled
    FUTURE event, not a state change, and letting one win would pin
    ``last_change_at`` in the future and silence the idle check forever.
    """
    # Deferred import: ci_findings imports this module at top level for
    # ``_dependency_root_blocker_progress``, so a module-level import of
    # ``_parse_iso_ts`` here would close an import cycle -- the same reason
    # ``backfill_dispatch_baseline`` defers its ``query_events`` import.
    from .ci_findings import _parse_iso_ts

    times: list[datetime] = []
    for key, value in entry.items():
        if not isinstance(value, str):
            continue
        if not (key.endswith("_at") or key.endswith("_since")):
            continue
        ts = _parse_iso_ts(value)
        if ts is not None and ts <= now:
            times.append(ts)
    return times


def _dependency_root_blocker_progress(
    state: dict[str, Any],
    roots: list[Any],
    now: datetime,
) -> list[dict[str, Any]]:
    """Issue #1682: last recorded change time per dependency root blocker.

    ``roots`` is the ``dependency_root_blockers`` list emitted by
    ``classify_backlog_reachability`` (``{"number", "updated_at"}`` entries).
    Returns one ``{"issue", "last_change_at", "idle_seconds",
    "blocking_prs"}`` detail dict per well-formed root, sorted-input order
    preserved.

    "Last change" is the newest of:

    * the root issue's GitHub ``updatedAt`` -- already fetched by the
      classifier, and the only signal that also captures progress the
      orchestrator never writes to state (human review, comments, label
      edits, a force-push), and
    * every past-dated ``*_at``/``*_since`` timestamp on the root's own
      ``state["issues"]`` entry and on each ``state["prs"]`` entry bound to
      it by ``issue_number`` (dispatch claims, review verdicts, janitor
      transitions, CI retriggers -- each one writes a stamp).

    A root with no parseable evidence gets ``last_change_at: None`` /
    ``idle_seconds: None`` -- the caller treats that as idle, because there
    is no recorded progress to bound the wait against (matching the
    ``no_baseline`` precedent's fail-LOUD direction for a detected stall:
    silence about progress is itself the finding).
    """
    # Deferred import for the same cycle reason documented in
    # ``_past_dated_state_timestamps`` above.
    from .ci_findings import _parse_iso_ts

    issues_state = state.get("issues")
    prs_state = state.get("prs")
    details: list[dict[str, Any]] = []
    for root in roots:
        if not isinstance(root, dict) or not isinstance(root.get("number"), int):
            continue
        number = root["number"]
        candidates: list[datetime] = []
        updated_at = root.get("updated_at")
        if isinstance(updated_at, str):
            ts = _parse_iso_ts(updated_at)
            if ts is not None and ts <= now:
                candidates.append(ts)
        entry = issues_state.get(str(number)) if isinstance(issues_state, dict) else None
        if isinstance(entry, dict):
            candidates.extend(_past_dated_state_timestamps(entry, now))
        blocking_prs: list[dict[str, Any]] = []
        if isinstance(prs_state, dict):
            for key, pr_entry in sorted(
                prs_state.items(),
                key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0,
            ):
                if not isinstance(pr_entry, dict):
                    continue
                if str(pr_entry.get("issue_number")) != str(number):
                    continue
                pr_times = _past_dated_state_timestamps(pr_entry, now)
                candidates.extend(pr_times)
                pr_last = max(pr_times) if pr_times else None
                pr_number = pr_entry.get("number")
                blocking_prs.append(
                    {
                        "number": (
                            pr_number
                            if isinstance(pr_number, int)
                            else int(key)
                            if str(key).isdigit()
                            else None
                        ),
                        "status": pr_entry.get("status"),
                        "last_change_at": (
                            pr_last.isoformat().replace("+00:00", "Z") if pr_last else None
                        ),
                    }
                )
        last_change = max(candidates) if candidates else None
        details.append(
            {
                "issue": number,
                "last_change_at": (
                    last_change.isoformat().replace("+00:00", "Z") if last_change else None
                ),
                "idle_seconds": (
                    int((now - last_change).total_seconds()) if last_change else None
                ),
                "blocking_prs": blocking_prs,
            }
        )
    return details
