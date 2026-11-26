"""``items.json``: the foreman's typed questions, permissions, and reports.

An *item* (CONTEXT.md) is how a foreman surfaces something that needs a
human: a ``decision`` (pick an option), a ``permission`` (approve an action —
the optional ``command`` field is what would run), or a ``report`` (something
worth knowing, no response needed). Items have a small lifecycle —
``open`` -> ``answered``/``dismissed``, or ``snoozed`` until a timestamp —
and carry a ``notified_at`` stamp so a notify consumer can tell which open
items the operator has not yet been told about.

The whole file is a single JSON document written through
``write_json_atomic``; reads tolerate a missing or corrupt file as empty,
matching the cache semantics of the rest of the store. Mutators are stricter:
a corrupt ``items.json`` is an error value, never an overwrite of whatever
was there.
"""

from __future__ import annotations

import json
from collections.abc import Callable
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..atomic_write import write_json_atomic
from . import Result
from .journal import utc_now
from .paths import items_path

ITEM_DECISION = "decision"
ITEM_PERMISSION = "permission"
ITEM_REPORT = "report"
ITEM_KINDS = frozenset({ITEM_DECISION, ITEM_PERMISSION, ITEM_REPORT})

STATUS_OPEN = "open"
STATUS_ANSWERED = "answered"
STATUS_SNOOZED = "snoozed"
STATUS_DISMISSED = "dismissed"
ITEM_STATUSES = frozenset({STATUS_OPEN, STATUS_ANSWERED, STATUS_SNOOZED, STATUS_DISMISSED})

_ITEM_ID_RE = re.compile(r"it-(\d+)")


@dataclass(frozen=True)
class Item:
    """One decision, permission, or report surfaced to the operator."""

    id: str
    kind: str
    status: str = STATUS_OPEN
    question: str | None = None
    text: str | None = None
    options: tuple[str, ...] = ()
    recommended: int | None = None
    command: str | None = None
    created_at: str = ""
    answered_at: str | None = None
    answer: int | None = None
    answer_note: str | None = None
    snoozed_until: str | None = None
    notified_at: str | None = None


def _to_record(item: Item) -> dict[str, Any]:
    return {
        "id": item.id,
        "kind": item.kind,
        "status": item.status,
        "question": item.question,
        "text": item.text,
        "options": list(item.options),
        "recommended": item.recommended,
        "command": item.command,
        "created_at": item.created_at,
        "answered_at": item.answered_at,
        "answer": item.answer,
        "answer_note": item.answer_note,
        "snoozed_until": item.snoozed_until,
        "notified_at": item.notified_at,
    }


def _from_record(raw: Any) -> Item | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
        return None
    kind = raw.get("kind")
    if kind not in ITEM_KINDS:
        return None
    options_raw = raw.get("options")
    options = (
        tuple(o for o in options_raw if isinstance(o, str))
        if isinstance(options_raw, list)
        else ()
    )
    status = raw.get("status")
    return Item(
        id=raw["id"],
        kind=kind,
        status=status if status in ITEM_STATUSES else STATUS_OPEN,
        question=raw.get("question") if isinstance(raw.get("question"), str) else None,
        text=raw.get("text") if isinstance(raw.get("text"), str) else None,
        options=options,
        recommended=raw.get("recommended") if isinstance(raw.get("recommended"), int) else None,
        command=raw.get("command") if isinstance(raw.get("command"), str) else None,
        created_at=raw.get("created_at") if isinstance(raw.get("created_at"), str) else "",
        answered_at=raw.get("answered_at") if isinstance(raw.get("answered_at"), str) else None,
        answer=raw.get("answer") if isinstance(raw.get("answer"), int) else None,
        answer_note=raw.get("answer_note") if isinstance(raw.get("answer_note"), str) else None,
        snoozed_until=(
            raw.get("snoozed_until") if isinstance(raw.get("snoozed_until"), str) else None
        ),
        notified_at=raw.get("notified_at") if isinstance(raw.get("notified_at"), str) else None,
    )


def _decode_items(raw: Any) -> tuple[Item, ...]:
    records = raw.get("items") if isinstance(raw, dict) else raw if isinstance(raw, list) else ()
    if not isinstance(records, list):
        return ()
    return tuple(item for item in (_from_record(r) for r in records) if item is not None)


def load_items(foreman_root: Path) -> tuple[Item, ...]:
    """Read every item, oldest-first; a missing/corrupt file reads as empty."""
    try:
        path = items_path(foreman_root)
    except ValueError:
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    return _decode_items(raw)


def _load_items_strict(foreman_root: Path) -> Result[tuple[Item, ...]]:
    """``items.json`` decode for mutators: absent reads empty, corrupt is an error.

    ``load_items`` tolerates corruption because readers may treat the store as
    a cache; a mutator may not — writing ``{"items": [...]}`` over a file it
    could not read would silently destroy whatever was there.
    """
    try:
        path = items_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    if not path.exists():
        return Result(value=())
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Result(error=f"items.json: unreadable or corrupt: {exc}")
    return Result(value=_decode_items(raw))


def _parse_utc(value: str) -> datetime | None:
    """Parse an aware ISO-8601 timestamp, normalized to UTC; ``None`` if invalid.

    Naive timestamps are rejected — without an offset a wall-clock string is
    ambiguous, and comparing it against ``Z``-normalized deadlines silently
    shifts the deadline by the host timezone.
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        return None
    return parsed.astimezone(UTC)


def _format_utc(moment: datetime) -> str:
    """The ``Z``-suffix shape ``utc_now`` writes — the canonical stored form."""
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _save_items(path: Path, items: tuple[Item, ...]) -> Result[None]:
    try:
        write_json_atomic(path, {"version": 1, "items": [_to_record(i) for i in items]})
    except OSError as exc:
        return Result(error=f"items.json: unwritable: {exc}")
    return Result()


def _next_item_id(items: tuple[Item, ...]) -> str:
    highest = 0
    for item in items:
        match = _ITEM_ID_RE.fullmatch(item.id)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"it-{highest + 1}"


def open_items(items: tuple[Item, ...], *, now: str | None = None) -> tuple[Item, ...]:
    """Items still awaiting the operator: ``open``, or ``snoozed`` past ``until``.

    Deadlines are compared as parsed datetimes, not strings — a lexicographic
    compare of unvalidated strings orders ``+02:00`` forms wrong. A stored
    ``snoozed_until`` that fails to parse is treated as expired (the item
    resurfaces: fail-visible rather than silently hidden).
    """
    moment = _parse_utc(now if now is not None else utc_now())
    open_list: list[Item] = []
    for item in items:
        if item.status == STATUS_OPEN:
            open_list.append(item)
            continue
        if item.status != STATUS_SNOOZED:
            continue
        until = _parse_utc(item.snoozed_until) if item.snoozed_until else None
        if until is None or (moment is not None and until <= moment):
            open_list.append(item)
    return tuple(open_list)


def create_item(
    foreman_root: Path,
    *,
    kind: str,
    question: str | None = None,
    text: str | None = None,
    options: tuple[str, ...] | list[str] = (),
    recommended: int | None = None,
    command: str | None = None,
    item_id: str | None = None,
    created_at: str | None = None,
) -> Result[Item]:
    """Create one item and persist it; returns the stored ``Item``.

    Validation is by kind: a ``decision`` or ``permission`` needs a question
    and at least two options; a ``report`` needs text and takes no options.
    """
    if kind not in ITEM_KINDS:
        return Result(error=f"item.kind {kind!r}: must be one of {sorted(ITEM_KINDS)}")
    options = tuple(options)
    if kind == ITEM_REPORT:
        if not text or not text.strip():
            return Result(error="item.text: a non-empty report body")
        if options:
            return Result(error="item.options: a report item takes no options")
    else:
        if not question or not question.strip():
            return Result(error=f"item.question: a non-empty question for a {kind} item")
        if len(options) < 2:
            return Result(
                error=f"item.options: at least 2 options for a {kind} item, got {len(options)}"
            )
        if recommended is not None and not 0 <= recommended < len(options):
            return Result(
                error=f"item.recommended {recommended}: index into {len(options)} options"
            )
    try:
        path = items_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    loaded = _load_items_strict(foreman_root)
    if not loaded.ok:
        return Result(error=loaded.error)
    items = loaded.value
    resolved_id = item_id or _next_item_id(items)
    if any(i.id == resolved_id for i in items):
        return Result(error=f"item.id {resolved_id!r}: already exists")
    item = Item(
        id=resolved_id,
        kind=kind,
        question=question,
        text=text,
        options=options,
        recommended=recommended if kind != ITEM_REPORT else None,
        command=command,
        created_at=created_at or utc_now(),
    )
    saved = _save_items(path, (*items, item))
    if not saved.ok:
        return Result(error=saved.error)
    return Result(value=item)


def _update(
    foreman_root: Path, item_id: str, update: Callable[[Item], Item], *, verb: str
) -> Result[Item]:
    """Find ``item_id``, apply ``update(item) -> Item``, and persist."""
    try:
        path = items_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    loaded = _load_items_strict(foreman_root)
    if not loaded.ok:
        return Result(error=loaded.error)
    items = loaded.value
    index = next((i for i, item in enumerate(items) if item.id == item_id), None)
    if index is None:
        return Result(error=f"item {item_id!r}: not found (cannot {verb})")
    updated = update(items[index])
    new_items = (*items[:index], updated, *items[index + 1 :])
    saved = _save_items(path, new_items)
    if not saved.ok:
        return Result(error=saved.error)
    return Result(value=updated)


def answer_item(
    foreman_root: Path,
    item_id: str,
    *,
    choice: int | str,
    note: str | None = None,
    answered_at: str | None = None,
) -> Result[Item]:
    """Record the operator's answer: an option index or the option's exact text."""
    loaded = _load_items_strict(foreman_root)
    if not loaded.ok:
        return Result(error=loaded.error)
    target = next((i for i in loaded.value if i.id == item_id), None)
    if target is None:
        return Result(error=f"item {item_id!r}: not found (cannot answer)")
    if target.status != STATUS_OPEN and target.status != STATUS_SNOOZED:
        return Result(error=f"item {item_id!r}: already {target.status}")
    if isinstance(choice, str):
        if choice not in target.options:
            return Result(error=f"item.answer {choice!r}: not one of {list(target.options)}")
        index = target.options.index(choice)
    else:
        index = choice
        if not 0 <= index < len(target.options):
            return Result(error=f"item.answer {index}: index into {len(target.options)} options")
    stamp = answered_at or utc_now()
    return _update(
        foreman_root,
        item_id,
        lambda item: replace(
            item,
            status=STATUS_ANSWERED,
            answer=index,
            answer_note=note,
            answered_at=stamp,
            snoozed_until=None,
        ),
        verb="answer",
    )


def snooze_item(
    foreman_root: Path, item_id: str, *, until: str, snoozed_at: str | None = None
) -> Result[Item]:
    """Hide an item until an aware ISO-8601 ``until`` timestamp.

    ``until`` must carry an offset (``Z`` or ``±HH:MM``); it is normalized to
    UTC ``Z`` before storing so ``open_items`` compares instants, not strings.
    Unparseable or naive input is an error value.
    """
    deadline = _parse_utc(until) if until and until.strip() else None
    if deadline is None:
        return Result(
            error=f"item.snoozed_until {until!r}: an aware ISO-8601 timestamp "
            "(e.g. 2026-11-26T09:00:00Z or ...+02:00)"
        )
    loaded = _load_items_strict(foreman_root)
    if not loaded.ok:
        return Result(error=loaded.error)
    target = next((i for i in loaded.value if i.id == item_id), None)
    if target is None:
        return Result(error=f"item {item_id!r}: not found (cannot snooze)")
    if target.status in (STATUS_ANSWERED, STATUS_DISMISSED):
        return Result(error=f"item {item_id!r}: already {target.status}")
    normalized = _format_utc(deadline)
    return _update(
        foreman_root,
        item_id,
        lambda item: replace(item, status=STATUS_SNOOZED, snoozed_until=normalized),
        verb="snooze",
    )


def dismiss_item(
    foreman_root: Path, item_id: str, *, dismissed_at: str | None = None
) -> Result[Item]:
    """Close an item without an answer."""
    loaded = _load_items_strict(foreman_root)
    if not loaded.ok:
        return Result(error=loaded.error)
    target = next((i for i in loaded.value if i.id == item_id), None)
    if target is None:
        return Result(error=f"item {item_id!r}: not found (cannot dismiss)")
    if target.status in (STATUS_ANSWERED, STATUS_DISMISSED):
        return Result(error=f"item {item_id!r}: already {target.status}")
    stamp = dismissed_at or utc_now()
    return _update(
        foreman_root,
        item_id,
        lambda item: replace(item, status=STATUS_DISMISSED, answered_at=stamp, snoozed_until=None),
        verb="dismiss",
    )


def mark_notified(foreman_root: Path, item_id: str, *, at: str | None = None) -> Result[Item]:
    """Stamp ``notified_at`` so a notify pass does not re-send the same item."""
    stamp = at or utc_now()
    return _update(
        foreman_root,
        item_id,
        lambda item: replace(item, notified_at=stamp),
        verb="mark notified",
    )
