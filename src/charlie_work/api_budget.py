"""API worker spend ledger — atomic settlement and budget status (issue #480).

The paid ``api`` worker kind (#478) needs spend accounting: USD computed from
stream-json token usage times configured provider pricing
(``ApiProviderConfig``). Claude Code's own self-reported dollar cost is WRONG
against non-Anthropic endpoints; token counts are correct — so this ledger
derives cost from tokens and pricing.

Design: ``docs/design/api-worker-adapter.md`` §6.

Scope of this module (the ledger, math, status, and settlement ONLY — no
enforcement):

* Value types are frozen dataclasses; every computation returns a NEW value
  and never mutates its inputs (CLAUDE.md invariant).
* All ledger writes go through the temp-file + ``replace()`` atomic pattern
  (same as ``state.save_state`` / ``claude_code._write_json_atomic``).
* Corrupt / unparsable ledger files are moved aside to a timestamped
  ``.corrupt`` sibling BEFORE recovering as empty — accounting data is never
  silently destroyed.
* No network calls anywhere in this module.

Token-usage parsing reuses ``claude_code.iter_claude_events`` (the shared
JSONL parsing primitive) — the file-reading logic is implemented once in
``claude_code`` and consumed here; it is NOT re-implemented. ``usage_from_events``
is a pure accumulator over already-parsed event dicts.

Cost model (matches the three rates on ``ApiProviderConfig``):

* ``input_tokens`` are billed at ``input_usd_per_mtok``. This includes
  ``cache_creation_input_tokens`` (fresh input written to the cache), because
  the config exposes no separate cache-creation write premium — cache writes
  are fresh input billed at the input rate.
* ``cached_tokens`` (``cache_read_input_tokens`` — input read from the cache)
  are billed at ``cached_input_usd_per_mtok``.
* ``output_tokens`` are billed at ``output_usd_per_mtok``.

Per-Mtok pricing: ``usd = tokens / 1_000_000 * usd_per_mtok``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .config import ApiBudgetConfig, ApiProviderConfig
from .routing import BudgetStatus

logger = logging.getLogger(__name__)

LEDGER_FILENAME = "api-budget.json"

# Idempotence identity for a settled session: a session is the same settlement
# unit when ALL THREE of (issue, started_at, session_id) match an existing
# ledger entry. Re-settling an already-recorded session is a no-op.
_SETTLE_KEY: tuple[str, ...] = ("issue", "started_at", "session_id")


@dataclass(frozen=True)
class Usage:
    """Token usage accumulated from a session's events.jsonl.

    ``input_tokens`` includes cache-creation tokens (fresh input written to
    the cache, billed at the input rate — see module docstring). ``cached_tokens``
    is cache-read input (billed at the cached-input rate).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


@dataclass(frozen=True)
class DayBucket:
    """Per-UTC-day spend aggregate."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    usd: float = 0.0


@dataclass(frozen=True)
class SessionEntry:
    """Per-session detail record stored under ``ledger.sessions``.

    Per-session detail is load-bearing: the operator reviews these entries to
    calibrate ``budget.max_usd_per_session`` after the first trial sessions,
    so completeness beats compactness here.
    """

    issue: int
    session_id: str
    provider: str
    model: str
    started_at: str
    ended_at: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    usd: float
    duration_s: float
    outcome: str


@dataclass(frozen=True)
class Ledger:
    """Spend ledger value type.

    ``days`` maps a UTC date (``YYYY-MM-DD``) to that day's aggregate
    ``DayBucket``. ``sessions`` is the per-session detail history. Both are
    immutable views: settlement returns a new ``Ledger`` with updated copies.
    """

    days: Mapping[str, DayBucket] = field(default_factory=lambda: MappingProxyType({}))
    lifetime_usd: float = 0.0
    sessions: tuple[SessionEntry, ...] = ()


# ---------------------------------------------------------------------------
# Pure computation
# ---------------------------------------------------------------------------


def _to_int(value: Any, default: int = 0) -> int:
    """Coerce a JSON-deserialized usage field to a non-negative int."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    try:
        return max(0, int(value))
    except (OverflowError, ValueError):
        return default


def _usage_from_usage_dict(usage: Mapping[str, Any]) -> Usage:
    """Build a ``Usage`` from one ``message.usage`` / result ``usage`` dict.

    ``input_tokens`` folds in ``cache_creation_input_tokens`` (fresh input
    written to the cache, billed at the input rate — no separate write
    premium is configured). ``cached_tokens`` is ``cache_read_input_tokens``.
    """
    input_tokens = _to_int(usage.get("input_tokens")) + _to_int(
        usage.get("cache_creation_input_tokens")
    )
    output_tokens = _to_int(usage.get("output_tokens"))
    cached_tokens = _to_int(usage.get("cache_read_input_tokens"))
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )


def usage_from_events(events: Iterable[dict[str, Any]]) -> Usage:
    """Accumulate input/output/cached token counts from parsed events.jsonl entries.

    ``events`` is an iterable of already-parsed event dicts (obtained from
    ``claude_code.iter_claude_events`` — the shared JSONL parsing primitive;
    the file-reading logic is NOT re-implemented here).

    Claude Code's stream-json format carries token usage in two places:

    * ``assistant`` events carry ``message.usage`` — the per-API-call (per-turn)
      usage. Claude Code writes one JSONL line per content block of an
      assistant message; the lines of one API call share ``message.id`` and
      have identical ``input_tokens`` / cache fields, with ``output_tokens``
      growing monotonically across the lines. Counting one row per line
      over-counts ~2.4x, so per-``message.id`` rows are de-duplicated keeping
      the entry with the largest ``output_tokens``.
    * The terminal ``result`` event carries a top-level ``usage`` — the
      cumulative session total.

    When a ``result`` event is present (the normal case for a reaped/completed
    session), its cumulative ``usage`` is authoritative and returned directly.
    When no ``result`` event exists (a session killed before emitting one),
    the per-turn ``assistant`` usages are summed (de-duplicated by
    ``message.id``) to reconstruct the total.
    """
    result_usage: Mapping[str, Any] | None = None
    assistant_by_msg_id: dict[str, Mapping[str, Any]] = {}
    assistant_no_id: list[Mapping[str, Any]] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "result":
            usage = event.get("usage")
            if isinstance(usage, dict):
                # Last result wins (there is normally exactly one; if a
                # stream emitted several, the final cumulative is authoritative).
                result_usage = usage
        elif etype == "assistant":
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict):
                continue
            msg_id = message.get("id")
            if msg_id:
                existing = assistant_by_msg_id.get(msg_id)
                if existing is None or _to_int(usage.get("output_tokens")) > _to_int(
                    existing.get("output_tokens")
                ):
                    assistant_by_msg_id[msg_id] = usage
            else:
                assistant_no_id.append(usage)

    if result_usage is not None:
        return _usage_from_usage_dict(result_usage)

    total_in = 0
    total_out = 0
    total_cached = 0
    for usage in (*assistant_by_msg_id.values(), *assistant_no_id):
        u = _usage_from_usage_dict(usage)
        total_in += u.input_tokens
        total_out += u.output_tokens
        total_cached += u.cached_tokens
    return Usage(total_in, total_out, total_cached)


def cost_usd(usage: Usage, provider_config: ApiProviderConfig) -> float:
    """Compute USD from token counts and per-Mtok provider pricing.

    ``usd = (tokens / 1_000_000) * usd_per_mtok`` for each token class, with
    cached input billed at the cached-input rate. Rounding to 8 decimal places
    keeps the on-disk ledger compact and deterministic without losing
    cent-level precision at any realistic token volume.
    """
    usd = (
        (usage.input_tokens / 1_000_000.0) * provider_config.input_usd_per_mtok
        + (usage.output_tokens / 1_000_000.0) * provider_config.output_usd_per_mtok
        + (usage.cached_tokens / 1_000_000.0) * provider_config.cached_input_usd_per_mtok
    )
    return round(usd, 8)


def _daily_reserve(budget_config: ApiBudgetConfig) -> float:
    """Reserve used for the daily headroom preflight check.

    ``max_usd_per_session`` when set (> 0); otherwise ``preflight_reserve_usd``
    (a conservative headroom estimate during the calibration window).
    """
    if budget_config.max_usd_per_session > 0:
        return budget_config.max_usd_per_session
    return budget_config.preflight_reserve_usd


def budget_status(ledger: Ledger, budget_config: ApiBudgetConfig, today: str) -> BudgetStatus:
    """Produce the ``routing.BudgetStatus`` value for the api preflight.

    ``today`` is a UTC date string ``YYYY-MM-DD`` (passed in so this function
    stays pure — no clock access inside).

    Daily headroom follows the issue's explicit check
    ``spent_today + reserve <= max_usd_per_day``: a launch whose reserve
    exactly fits the remaining daily budget is allowed (headroom True), while
    spend already at the daily cap is exhausted (with a positive reserve,
    ``cap + reserve <= cap`` is False).

    Lifetime headroom uses a strict ``lifetime_spent < lifetime_usd`` cap so
    that spend exactly at the lifetime cap is exhausted (acceptance criterion:
    exactly-at-cap is exhausted).
    """
    spent_today = ledger.days.get(today, DayBucket()).usd
    reserve = _daily_reserve(budget_config)
    daily_headroom = (spent_today + reserve) <= budget_config.max_usd_per_day
    lifetime_headroom = ledger.lifetime_usd < budget_config.lifetime_usd
    return BudgetStatus(
        spent_today_usd=spent_today,
        lifetime_spent_usd=ledger.lifetime_usd,
        daily_headroom=daily_headroom,
        lifetime_headroom=lifetime_headroom,
    )


# ---------------------------------------------------------------------------
# Settlement (idempotent)
# ---------------------------------------------------------------------------


def _session_key(entry: SessionEntry) -> tuple[int, str, str]:
    """Idempotence identity for a session entry."""
    return (entry.issue, entry.started_at, entry.session_id)


def _utc_date_from_iso(iso_ts: str) -> str:
    """Extract the UTC ``YYYY-MM-DD`` date from an ISO-8601 timestamp.

    Naive timestamps are treated as UTC. Malformed timestamps fall back to an
    empty string (the caller decides how to handle a missing day bucket key).
    """
    if not iso_ts:
        return ""
    try:
        parsed = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    return parsed.strftime("%Y-%m-%d")


def settle_session(ledger: Ledger, session_entry: SessionEntry) -> Ledger:
    """Return a new ledger with ``session_entry`` settled.

    Idempotent: settling the same session twice (identified by
    ``issue`` + ``started_at`` + ``session_id``) is a no-op the second time —
    the entry is not re-appended and the day bucket / lifetime total are not
    bumped again.

    Does not mutate ``ledger``.
    """
    new_key = _session_key(session_entry)
    for existing in ledger.sessions:
        if _session_key(existing) == new_key:
            return ledger

    day_key = _utc_date_from_iso(session_entry.started_at)
    new_days = dict(ledger.days)
    if day_key:
        current = new_days.get(day_key, DayBucket())
        new_days[day_key] = DayBucket(
            input_tokens=current.input_tokens + session_entry.input_tokens,
            output_tokens=current.output_tokens + session_entry.output_tokens,
            cached_tokens=current.cached_tokens + session_entry.cached_tokens,
            usd=round(current.usd + session_entry.usd, 8),
        )
    return Ledger(
        days=MappingProxyType(new_days),
        lifetime_usd=round(ledger.lifetime_usd + session_entry.usd, 8),
        sessions=(*ledger.sessions, session_entry),
    )


# ---------------------------------------------------------------------------
# Persistence (atomic + corrupt recovery)
# ---------------------------------------------------------------------------


def _quarantine_ledger(path: Path, exc: BaseException) -> None:
    """Move an unrecoverable ledger file aside to a timestamped ``.corrupt`` sibling.

    The corrupt original is preserved on disk for forensics; a loud log line
    makes a silent accounting wipe visible. Mirrors ``state._quarantine_state``.
    """
    quarantine = path.with_name(
        f"{path.name}.corrupt-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    logger.error(
        "API budget ledger %s is unrecoverable (%s: %s); quarantining to %s",
        path,
        type(exc).__name__,
        exc,
        quarantine,
    )
    try:
        path.replace(quarantine)
    except OSError as move_err:
        logger.error("Failed to quarantine ledger %s: %s", path, move_err)


def _ledger_from_dict(data: Any) -> Ledger:
    """Build a ``Ledger`` from a JSON-deserialized dict (lenient on missing keys)."""
    if not isinstance(data, dict):
        return Ledger()
    raw_days = data.get("days")
    days: dict[str, DayBucket] = {}
    if isinstance(raw_days, dict):
        for date_key, bucket in raw_days.items():
            if not isinstance(bucket, dict):
                continue
            days[str(date_key)] = DayBucket(
                input_tokens=_to_int(bucket.get("input_tokens")),
                output_tokens=_to_int(bucket.get("output_tokens")),
                cached_tokens=_to_int(bucket.get("cached_tokens")),
                usd=float(bucket.get("usd") or 0.0),
            )
    raw_sessions = data.get("sessions")
    sessions: list[SessionEntry] = []
    if isinstance(raw_sessions, list):
        for entry in raw_sessions:
            if not isinstance(entry, dict):
                continue
            try:
                sessions.append(
                    SessionEntry(
                        issue=int(entry.get("issue") or 0),
                        session_id=str(entry.get("session_id") or ""),
                        provider=str(entry.get("provider") or ""),
                        model=str(entry.get("model") or ""),
                        started_at=str(entry.get("started_at") or ""),
                        ended_at=str(entry.get("ended_at") or ""),
                        input_tokens=_to_int(entry.get("input_tokens")),
                        output_tokens=_to_int(entry.get("output_tokens")),
                        cached_tokens=_to_int(entry.get("cached_tokens")),
                        usd=float(entry.get("usd") or 0.0),
                        duration_s=float(entry.get("duration_s") or 0.0),
                        outcome=str(entry.get("outcome") or ""),
                    )
                )
            except (TypeError, ValueError):
                continue
    return Ledger(
        days=MappingProxyType(days),
        lifetime_usd=float(data.get("lifetime_usd") or 0.0),
        sessions=tuple(sessions),
    )


def ledger_to_dict(ledger: Ledger) -> dict[str, Any]:
    """Serialize a ``Ledger`` to the on-disk JSON shape."""
    return {
        "days": {
            date_key: {
                "input_tokens": bucket.input_tokens,
                "output_tokens": bucket.output_tokens,
                "cached_tokens": bucket.cached_tokens,
                "usd": bucket.usd,
            }
            for date_key, bucket in ledger.days.items()
        },
        "lifetime_usd": ledger.lifetime_usd,
        "sessions": [
            {
                "issue": entry.issue,
                "session_id": entry.session_id,
                "provider": entry.provider,
                "model": entry.model,
                "started_at": entry.started_at,
                "ended_at": entry.ended_at,
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "cached_tokens": entry.cached_tokens,
                "usd": entry.usd,
                "duration_s": entry.duration_s,
                "outcome": entry.outcome,
            }
            for entry in ledger.sessions
        ],
    }


def load_ledger(path: Path) -> Ledger:
    """Load the ledger from ``path`` with corrupt-file recovery.

    A missing file is a valid empty ledger. A corrupt / unparsable file is
    moved aside to a timestamped ``.corrupt`` sibling (preserving the original
    for forensics) and then treated as empty — accounting data is never
    silently destroyed.
    """
    if not path.exists():
        return Ledger()
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        # ``_ledger_from_dict`` coerces ``usd`` / ``lifetime_usd`` / ``duration_s``
        # via ``float()``; a syntactically-valid JSON file with a wrong-typed
        # (non-numeric, non-falsy) field raises ``TypeError`` / ``ValueError``
        # here. Wrapping it inside the same guard sends that structural
        # corruption through the quarantine path (forensic log + preserved
        # original) instead of propagating uncaught and wedging every future
        # settlement — consistent with the per-session-entry loop in
        # ``_ledger_from_dict`` which already drops bad entries of this class.
        return _ledger_from_dict(data)
    except (json.JSONDecodeError, LookupError, ValueError, TypeError) as exc:
        _quarantine_ledger(path, exc)
        return Ledger()
    except OSError as exc:
        _quarantine_ledger(path, exc)
        return Ledger()


def save_ledger(path: Path, ledger: Ledger) -> None:
    """Atomically persist ``ledger`` to ``path`` (temp-file + ``replace()``).

    Mirrors ``state.save_state`` / ``claude_code._write_json_atomic``: the
    write is atomic so a concurrent reader never observes a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(ledger_to_dict(ledger), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def settle_session_to_disk(path: Path, entry: SessionEntry) -> bool:
    """Atomically settle ``entry`` into the ledger at ``path`` under the advisory lock.

    This is the locked read-modify-write primitive for the on-disk ledger:
    ``load_ledger`` → ``settle_session`` → ``save_ledger`` happen entirely
    inside ``state.advisory_file_lock(path)``, so concurrent reaps of different
    api sessions cannot lose a settlement (the same lost-update hazard that
    ``state_lock`` closes for ``state.json``). The per-path threading.Lock in
    ``advisory_file_lock`` also serializes concurrent THREADS in this process.

    Fail-as-a-value: if the advisory lock cannot be acquired within its budget
    (``StateLockBusy``), the settlement is SKIPPED — logged as a warning and
    ``False`` returned — never written unlocked. This mirrors the codebase
    invariant that a writer which cannot acquire the lock fails that unit of
    work as a value rather than degrading integrity. Settlement is best-effort
    accounting (it must never break the reap that calls it), so a lock-contention
    skip loses one settlement record but preserves ledger consistency.

    Returns ``True`` if the entry was settled (or was already present —
    idempotent no-op still returns ``True``), ``False`` if skipped due to lock
    contention. Unexpected errors propagate to the caller.
    """
    from .state import StateLockBusy, advisory_file_lock

    try:
        # Ensure the ledger directory exists before acquiring the lock — the
        # advisory lock touches a sibling ``.lock`` file whose parent must
        # exist. state.json's parent always exists by the time state_lock runs,
        # but the ledger may be settled into a fresh state_dir on the very first
        # api reap. mkdir is idempotent and safe outside the lock.
        path.parent.mkdir(parents=True, exist_ok=True)
        with advisory_file_lock(path):
            ledger = load_ledger(path)
            ledger = settle_session(ledger, entry)
            save_ledger(path, ledger)
    except StateLockBusy:
        logger.warning(
            "api budget ledger lock busy at %s; skipping settlement of issue %s "
            "(best-effort accounting — ledger consistency preserved, one record may be lost)",
            path,
            entry.issue,
        )
        return False
    return True


def ledger_path(state_dir: Path | str) -> Path:
    """Return the ledger path for a runtime ``state_dir`` root."""
    return Path(state_dir) / LEDGER_FILENAME


__all__ = [
    "LEDGER_FILENAME",
    "Usage",
    "DayBucket",
    "SessionEntry",
    "Ledger",
    "usage_from_events",
    "cost_usd",
    "budget_status",
    "settle_session",
    "settle_session_to_disk",
    "load_ledger",
    "save_ledger",
    "ledger_to_dict",
    "ledger_path",
]
