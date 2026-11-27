"""``state.json``: the foreman's runtime cache.

ADR-0010 is explicit that this file is a *cache*, not truth — the GitHub
ownership label is the foreman's truth, and this file may be deleted and
rebuilt. Two consequences:

* **Loads are maximally tolerant.** A missing, corrupt, or partially-shaped
  ``state.json`` loads as the default paused state rather than failing a
  foreman that GitHub still says is chartered.
* **Writes are atomic and values immutable.** ``save_state`` goes through
  ``write_json_atomic`` like every other orchestrator JSON write, and updates
  return a new ``ForemanState`` — nothing mutates the cache in place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..atomic_write import write_json_atomic
from . import FOREMAN_STATES, Result
from .paths import state_path

DEFAULT_STATE = "paused"


@dataclass(frozen=True)
class AuthoredIssue:
    """The foreman's record of one issue it authored (label truth lives on GitHub)."""

    number: int
    body_hash: str


@dataclass(frozen=True)
class ForemanState:
    """The cached runtime picture of one foreman (design §7).

    ``event_cursor`` is how far the hub's event stream this foreman has
    consumed; ``lifecycle_fingerprint`` detects a charter/registry change that
    should reset the turn cycle; ``issues`` mirrors the authored-issue label
    state with each issue's body hash so edits are detected.
    """

    state: str = DEFAULT_STATE
    event_cursor: int = 0
    lifecycle_fingerprint: str = ""
    turns_total: int = 0
    turns_today: int = 0
    turns_today_date: str = ""
    consecutive_failures: int = 0
    last_turn_at: str | None = None
    last_scheduled_turn_at: str | None = None
    issues: dict[int, AuthoredIssue] = field(default_factory=dict)
    version: int = 1


def _to_payload(state: ForemanState) -> dict[str, Any]:
    return {
        "version": state.version,
        "state": state.state,
        "event_cursor": state.event_cursor,
        "lifecycle_fingerprint": state.lifecycle_fingerprint,
        "turns_total": state.turns_total,
        "turns_today": state.turns_today,
        "turns_today_date": state.turns_today_date,
        "consecutive_failures": state.consecutive_failures,
        "last_turn_at": state.last_turn_at,
        "last_scheduled_turn_at": state.last_scheduled_turn_at,
        "issues": {
            str(number): {"body_hash": issue.body_hash}
            for number, issue in sorted(state.issues.items())
        },
    }


def _issue_from(number: int, raw: Any) -> AuthoredIssue | None:
    if isinstance(raw, dict):
        body_hash = raw.get("body_hash", "")
    elif isinstance(raw, str):
        body_hash = raw
    else:
        return None
    if not isinstance(body_hash, str):
        return None
    return AuthoredIssue(number=number, body_hash=body_hash)


def _from_payload(raw: Any) -> ForemanState:
    """Best-effort decode; anything unusable falls back to its default.

    This is a cache read, not a contract check — a truncated or older-shaped
    file must never take the foreman down.
    """
    if not isinstance(raw, dict):
        return ForemanState()
    state_value = raw.get("state")
    issues: dict[int, AuthoredIssue] = {}
    raw_issues = raw.get("issues")
    if isinstance(raw_issues, dict):
        for key, entry in raw_issues.items():
            try:
                number = int(key)
            except (TypeError, ValueError):
                continue
            decoded = _issue_from(number, entry)
            if decoded is not None:
                issues[number] = decoded
    return ForemanState(
        state=state_value if state_value in FOREMAN_STATES else DEFAULT_STATE,
        event_cursor=raw.get("event_cursor") if isinstance(raw.get("event_cursor"), int) else 0,
        lifecycle_fingerprint=(
            raw.get("lifecycle_fingerprint")
            if isinstance(raw.get("lifecycle_fingerprint"), str)
            else ""
        ),
        turns_total=raw.get("turns_total") if isinstance(raw.get("turns_total"), int) else 0,
        turns_today=raw.get("turns_today") if isinstance(raw.get("turns_today"), int) else 0,
        turns_today_date=(
            raw.get("turns_today_date") if isinstance(raw.get("turns_today_date"), str) else ""
        ),
        consecutive_failures=(
            raw.get("consecutive_failures")
            if isinstance(raw.get("consecutive_failures"), int)
            else 0
        ),
        last_turn_at=raw.get("last_turn_at") if isinstance(raw.get("last_turn_at"), str) else None,
        last_scheduled_turn_at=(
            raw.get("last_scheduled_turn_at")
            if isinstance(raw.get("last_scheduled_turn_at"), str)
            else None
        ),
        issues=issues,
        version=raw.get("version") if isinstance(raw.get("version"), int) else 1,
    )


def read_state_snapshot(foreman_root: Path) -> ForemanState:
    """Lock-free, non-mutating read of ``state.json`` — the foreman-layer
    analogue of ``charlie_work.state.read_state_snapshot`` (issue #3141).

    Safe unlocked: it never quarantines, renames, or writes anything, so a
    racing atomic replace cannot be misread into mutating a healthy file.
    The failure contract differs from the hub's reader by design: anything
    unreadable returns the default paused state instead of ``None``, because
    this file is a *cache* — it can always be rebuilt from GitHub labels and
    the append-only logs.
    """
    try:
        path = state_path(foreman_root)
    except ValueError:
        return ForemanState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ForemanState()
    return _from_payload(raw)


def save_state(foreman_root: Path, state: ForemanState) -> Result[ForemanState]:
    """Persist ``state`` atomically, rejecting an unknown lifecycle value."""
    if state.state not in FOREMAN_STATES:
        return Result(
            error=f"foreman state {state.state!r}: must be one of {sorted(FOREMAN_STATES)}"
        )
    try:
        path = state_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    try:
        write_json_atomic(path, _to_payload(state))
    except OSError as exc:
        return Result(error=f"state.json: unwritable: {exc}")
    return Result(value=state)


def with_updates(current: ForemanState, **changes: Any) -> ForemanState:
    """Return a new ``ForemanState`` with ``changes`` applied — never mutates."""
    return replace(current, **changes)
