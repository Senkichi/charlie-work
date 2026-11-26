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
from ..file_lock import LOCK_ACQUIRE_RETRIES, acquire_byte_range_lock
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


def _raw_records(raw: Any) -> list[Any] | None:
    """The raw record list — ``{"items": [...]}`` or a bare top-level list."""
    records = raw.get("items") if isinstance(raw, dict) else raw if isinstance(raw, list) else ()
    return records if isinstance(records, list) else None


def _decode_items(raw: Any) -> tuple[Item, ...]:
    records = _raw_records(raw)
    if records is None:
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
    records = _raw_records(raw)
    if records is None:
        return Result(error="items.json: unreadable or corrupt: 'items' is not a list")
    decoded = [_from_record(record) for record in records]
    undecodable = sum(1 for item in decoded if item is None)
    if undecodable:
        return Result(
            error=f"items.json: unreadable or corrupt: {undecodable} record(s) failed to decode"
        )
    return Result(value=tuple(item for item in decoded if item is not None))


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


def _mutate_items(
    foreman_root: Path,
    mutate: Callable[[tuple[Item, ...]], Result[tuple[tuple[Item, ...], Item]]],
) -> Result[Item]:
    """One locked load-mutate-save pass over ``items.json``.

    ``mutate`` receives the strictly-decoded items and returns the replacement
    list plus the item to hand back. The per-file lock on ``items.json.lock``
    covers the strict load, the mutation step, and the atomic write: two
    mutators cannot interleave a read-modify-write (a second creator minting
    the same id, an answer landing on a stale read), and a busy lock is an
    error value rather than a hang.
    """
    try:
        path = items_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    # Both the lock acquire and ``write_json_atomic`` mkdir the parent — an
    # item write to a removed foreman must refuse, not silently re-materialize
    # a charterless ``foremen/<name>/``.
    if not path.parent.is_dir():
        return Result(error=f"{path.name}: parent directory {path.parent} does not exist")
    lock = acquire_byte_range_lock(path.with_name(path.name + ".lock"))
    if lock is None:
        return Result(error=f"{path.name}: lock busy after {LOCK_ACQUIRE_RETRIES} retries")
    try:
        loaded = _load_items_strict(foreman_root)
        if not loaded.ok:
            return Result(error=loaded.error)
        mutated = mutate(loaded.value)
        if not mutated.ok:
            return Result(error=mutated.error)
        new_items, result_item = mutated.value
        saved = _save_items(path, new_items)
        if not saved.ok:
            return Result(error=saved.error)
        return Result(value=result_item)
    finally:
        lock.release()


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

    def step(items: tuple[Item, ...]) -> Result[tuple[tuple[Item, ...], Item]]:
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
        return Result(value=((*items, item), item))

    return _mutate_items(foreman_root, step)


def _update(
    foreman_root: Path, item_id: str, update: Callable[[Item], Result[Item]], *, verb: str
) -> Result[Item]:
    """Find ``item_id``, run ``update(item) -> Result[Item]``, and persist.

    ``update`` carries the verb's validation so the whole read-modify-write —
    strict load, validate, replace, save — runs inside ``_mutate_items``'
    lock instead of loading twice across an unprotected gap.
    """

    def step(items: tuple[Item, ...]) -> Result[tuple[tuple[Item, ...], Item]]:
        index = next((i for i, item in enumerate(items) if item.id == item_id), None)
        if index is None:
            return Result(error=f"item {item_id!r}: not found (cannot {verb})")
        updated = update(items[index])
        if not updated.ok:
            return Result(error=updated.error)
        return Result(value=((*items[:index], updated.value, *items[index + 1 :]), updated.value))

    return _mutate_items(foreman_root, step)


def answer_item(
    foreman_root: Path,
    item_id: str,
    *,
    choice: int | str,
    note: str | None = None,
    answered_at: str | None = None,
) -> Result[Item]:
    """Record the operator's answer: an option index or the option's exact text."""

    def step(item: Item) -> Result[Item]:
        if item.status != STATUS_OPEN and item.status != STATUS_SNOOZED:
            return Result(error=f"item {item_id!r}: already {item.status}")
        if isinstance(choice, str):
            if choice not in item.options:
                return Result(error=f"item.answer {choice!r}: not one of {list(item.options)}")
            index = item.options.index(choice)
        else:
            index = choice
            if not 0 <= index < len(item.options):
                return Result(error=f"item.answer {index}: index into {len(item.options)} options")
        return Result(
            value=replace(
                item,
                status=STATUS_ANSWERED,
                answer=index,
                answer_note=note,
                answered_at=answered_at or utc_now(),
                snoozed_until=None,
            )
        )

    return _update(foreman_root, item_id, step, verb="answer")


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
    normalized = _format_utc(deadline)

    def step(item: Item) -> Result[Item]:
        if item.status in (STATUS_ANSWERED, STATUS_DISMISSED):
            return Result(error=f"item {item_id!r}: already {item.status}")
        return Result(value=replace(item, status=STATUS_SNOOZED, snoozed_until=normalized))

    return _update(foreman_root, item_id, step, verb="snooze")


def dismiss_item(
    foreman_root: Path, item_id: str, *, dismissed_at: str | None = None
) -> Result[Item]:
    """Close an item without an answer."""

    def step(item: Item) -> Result[Item]:
        if item.status in (STATUS_ANSWERED, STATUS_DISMISSED):
            return Result(error=f"item {item_id!r}: already {item.status}")
        return Result(
            value=replace(
                item,
                status=STATUS_DISMISSED,
                answered_at=dismissed_at or utc_now(),
                snoozed_until=None,
            )
        )

    return _update(foreman_root, item_id, step, verb="dismiss")


def mark_notified(foreman_root: Path, item_id: str, *, at: str | None = None) -> Result[Item]:
    """Stamp ``notified_at`` so a notify pass does not re-send the same item."""
    return _update(
        foreman_root,
        item_id,
        lambda item: Result(value=replace(item, notified_at=at or utc_now())),
        verb="mark notified",
    )
