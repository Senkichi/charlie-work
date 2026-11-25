"""The append-only logs: ``journal.jsonl``, ``inbox.jsonl``, ``signals.jsonl``.

Three line-delimited JSON streams with different authors (the foreman, the
operator, and external signal producers) but one contract: each entry carries
a monotonically increasing integer ``id`` and a UTC timestamp. A reader asks
for entries "unread since id N" — the foreman's ``event_cursor`` consumption
model — and tolerates a torn final line, which is the only corruption an
append-only file can exhibit.

These files append rather than rewrite: a JSONL stream exists so a writer can
add one line without reading or replacing what is already there, and a torn
write can only damage the line being written — which ``_iter_entries`` skips.
The atomic-rewrite invariant covers whole-document JSON state files
(``state.json``, ``items.json``), which this module does not write.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import Result
from .paths import inbox_path, journal_path, signals_path


def utc_now() -> str:
    """The timestamp shape every log entry carries (ISO-8601, ``Z`` suffix)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class JournalEntry:
    """One dated line recording a Turn or a noteworthy store event."""

    id: int
    ts: str
    text: str
    turn_id: str | None = None


@dataclass(frozen=True)
class InboxMessage:
    """One operator message waiting for (or already read by) the foreman."""

    id: int
    ts: str
    text: str


@dataclass(frozen=True)
class SignalEntry:
    """One external signal: a source label plus a free-form payload."""

    id: int
    ts: str
    source: str
    data: dict[str, Any] = field(default_factory=dict)


def _append(path: Path, record: dict[str, Any]) -> Result[None]:
    try:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        return Result(error=f"{path.name}: unwritable: {exc}")
    return Result()


def _iter_entries(path: Path) -> Iterator[dict[str, Any]]:
    """Yield decoded entries, skipping blank lines and a torn final write."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def _next_id(path: Path) -> int:
    last = 0
    for record in _iter_entries(path):
        entry_id = record.get("id")
        if isinstance(entry_id, int) and entry_id > last:
            last = entry_id
    return last + 1


def append_journal(
    foreman_root: Path, text: str, *, turn_id: str | None = None, ts: str | None = None
) -> Result[JournalEntry]:
    """Append one journal line; returns the entry as written."""
    try:
        path = journal_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    entry = JournalEntry(id=_next_id(path), ts=ts or utc_now(), text=text, turn_id=turn_id)
    record: dict[str, Any] = {"id": entry.id, "ts": entry.ts, "text": entry.text}
    if turn_id is not None:
        record["turn_id"] = turn_id
    written = _append(path, record)
    if not written.ok:
        return Result(error=written.error)
    return Result(value=entry)


def read_journal(foreman_root: Path, *, limit: int | None = None) -> tuple[JournalEntry, ...]:
    """Read journal entries oldest-first; ``limit`` keeps the newest N."""
    try:
        path = journal_path(foreman_root)
    except ValueError:
        return ()
    entries = [
        JournalEntry(
            id=record.get("id", 0),
            ts=record.get("ts", ""),
            text=record.get("text", ""),
            turn_id=record.get("turn_id"),
        )
        for record in _iter_entries(path)
    ]
    if limit is not None and limit >= 0:
        entries = entries[-limit:] if limit else []
    return tuple(entries)


def append_inbox(foreman_root: Path, text: str, *, ts: str | None = None) -> Result[InboxMessage]:
    """Append one operator message to ``inbox.jsonl``."""
    try:
        path = inbox_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    entry = InboxMessage(id=_next_id(path), ts=ts or utc_now(), text=text)
    written = _append(path, {"id": entry.id, "ts": entry.ts, "text": entry.text})
    if not written.ok:
        return Result(error=written.error)
    return Result(value=entry)


def read_inbox(foreman_root: Path, *, after_id: int = 0) -> tuple[InboxMessage, ...]:
    """Read inbox messages with ``id > after_id`` — the "unread since" read."""
    try:
        path = inbox_path(foreman_root)
    except ValueError:
        return ()
    return tuple(
        InboxMessage(id=record.get("id", 0), ts=record.get("ts", ""), text=record.get("text", ""))
        for record in _iter_entries(path)
        if isinstance(record.get("id"), int) and record["id"] > after_id
    )


def append_signal(
    foreman_root: Path,
    source: str,
    data: dict[str, Any] | None = None,
    *,
    ts: str | None = None,
) -> Result[SignalEntry]:
    """Append one external signal to ``signals.jsonl``."""
    try:
        path = signals_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    entry = SignalEntry(id=_next_id(path), ts=ts or utc_now(), source=source, data=data or {})
    written = _append(
        path, {"id": entry.id, "ts": entry.ts, "source": entry.source, "data": entry.data}
    )
    if not written.ok:
        return Result(error=written.error)
    return Result(value=entry)


def read_signals(foreman_root: Path, *, after_id: int = 0) -> tuple[SignalEntry, ...]:
    """Read signals with ``id > after_id`` — the "unread since" read."""
    try:
        path = signals_path(foreman_root)
    except ValueError:
        return ()
    return tuple(
        SignalEntry(
            id=record.get("id", 0),
            ts=record.get("ts", ""),
            source=record.get("source", ""),
            data=record.get("data") if isinstance(record.get("data"), dict) else {},
        )
        for record in _iter_entries(path)
        if isinstance(record.get("id"), int) and record["id"] > after_id
    )
