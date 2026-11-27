"""The wake predicate and deferral bounds (design §7, issue #3122).

Split out of ``pass_run.py`` for the file-size ratchet: ``wake_reason``
decides whether a foreman wants a Turn this pass (lifecycle changed, inbox,
item answer, signal, report cadence, quiet elapsed), ``_deferral_reason``
decides whether a wanted Turn waits (fleet paused, min interval, turns-per-
day bound, stale snapshot, decide deferred). ``pass_run`` re-exports the
public surface so its ``__all__`` and the tests' ``pass_run.WAKE_*`` /
``pass_run.DEFER_*`` spellings keep working.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ..instrumentation import log_event
from ..iso_timestamp import parse_iso_timestamp
from . import items as fitems
from . import journal as fjournal
from . import paths as fpaths
from .collect import Foreman, lifecycle_fingerprint
from .items import STATUS_ANSWERED

EVENT_WAKE_DEFERRED = "foreman_wake_deferred"

#: Wake-predicate reasons (design §7): what made this foreman want a Turn.
WAKE_LIFECYCLE = "lifecycle_changed"
WAKE_INBOX = "inbox"
WAKE_ITEM_ANSWER = "item_answer"
WAKE_SIGNAL = "signal"
WAKE_REPORT_DUE = "report_due"
WAKE_QUIET = "quiet_elapsed"
#: Not a predicate — the resume marker when a pass finishes a crashed Turn.
WAKE_RESUME = "resume"

#: Deferral reasons carried on ``foreman_wake_deferred`` (design §7).
DEFER_FLEET_PAUSED = "fleet_paused"
DEFER_MIN_INTERVAL = "min_turn_interval_minutes"
DEFER_MAX_TURNS = "max_turns_per_day"
DEFER_STALE_SNAPSHOT = "stale_snapshot"
DEFER_DECIDE = "decide_deferred"


def _today(now: datetime) -> str:
    """The UTC date bucket ``turns_today`` counts in."""
    return now.astimezone(UTC).strftime("%Y-%m-%d")


def _new_item_answer(foreman: Foreman) -> bool:
    """An item moved to ``answered`` after the last Turn completed."""
    last = parse_iso_timestamp(foreman.state.last_turn_at)
    for item in fitems.load_items(foreman.root):
        if item.status != STATUS_ANSWERED or not item.answered_at:
            continue
        answered = parse_iso_timestamp(item.answered_at)
        if answered is not None and (last is None or answered > last):
            return True
    return False


def _report_due(foreman: Foreman, now: datetime) -> bool:
    """The daily ``cadence.report_at`` (local HH:MM) passed since the last
    scheduled Turn — one catch-up Turn however many days were missed."""
    cadence = foreman.charter.cadence
    report_at = cadence.report_at if cadence is not None else None
    if not report_at:
        return False
    local_now = now.astimezone()
    due = local_now.replace(
        hour=int(report_at[:2]), minute=int(report_at[3:5]), second=0, microsecond=0
    )
    if due > local_now:
        due -= timedelta(days=1)
    last = parse_iso_timestamp(foreman.state.last_scheduled_turn_at)
    return last is None or last < due


def _quiet_due(foreman: Foreman, now: datetime) -> bool:
    """``bounds.max_quiet_hours`` elapsed since the last Turn."""
    last = parse_iso_timestamp(foreman.state.last_turn_at)
    return last is None or (now - last) >= timedelta(hours=foreman.charter.bounds.max_quiet_hours)


def wake_reason(foreman: Foreman, *, now: datetime) -> str | None:
    """The first true wake condition for ``foreman``, or ``None``."""
    state = foreman.state
    if lifecycle_fingerprint(foreman) != state.lifecycle_fingerprint:
        return WAKE_LIFECYCLE
    if fjournal.read_inbox(foreman.root, after_id=state.inbox_cursor):
        return WAKE_INBOX
    if _new_item_answer(foreman):
        return WAKE_ITEM_ANSWER
    if fjournal.read_signals(foreman.root, after_id=state.signal_cursor):
        return WAKE_SIGNAL
    if _report_due(foreman, now):
        return WAKE_REPORT_DUE
    if _quiet_due(foreman, now):
        return WAKE_QUIET
    return None


def _deferral_reason(foreman: Foreman, *, fleet_paused: bool, now: datetime) -> str | None:
    """The bound that defers a wanted Turn, or ``None``."""
    if fleet_paused:
        return DEFER_FLEET_PAUSED
    bounds = foreman.charter.bounds
    last = parse_iso_timestamp(foreman.state.last_turn_at)
    if last is not None and (now - last) < timedelta(minutes=bounds.min_turn_interval_minutes):
        return DEFER_MIN_INTERVAL
    if (
        foreman.state.turns_today_date == _today(now)
        and foreman.state.turns_today >= bounds.max_turns_per_day
    ):
        return DEFER_MAX_TURNS
    return None


def _emit_deferred(foreman: Foreman, *, reason: str, wake: str, detail: str | None) -> None:
    payload: dict[str, Any] = {"foreman": foreman.name, "reason": reason, "wake": wake}
    if detail is not None:
        payload["detail"] = detail
    # write-gate-exempt(issue=3122): no write_gate receiver on the pass — the event mirrors the deferral decision into the foreman's events.db.
    log_event(  # event-consumer: audit-only — the pass outcome carries the same decision
        fpaths.state_path(foreman.root),
        EVENT_WAKE_DEFERRED,
        payload,
        repo=foreman.repo_key,
    )


__all__ = [
    "DEFER_DECIDE",
    "DEFER_FLEET_PAUSED",
    "DEFER_MAX_TURNS",
    "DEFER_MIN_INTERVAL",
    "DEFER_STALE_SNAPSHOT",
    "EVENT_WAKE_DEFERRED",
    "WAKE_INBOX",
    "WAKE_ITEM_ANSWER",
    "WAKE_LIFECYCLE",
    "WAKE_QUIET",
    "WAKE_REPORT_DUE",
    "WAKE_RESUME",
    "WAKE_SIGNAL",
    "wake_reason",
]
