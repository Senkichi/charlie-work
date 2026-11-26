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
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..file_lock import try_acquire_byte_range_lock
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
    line = json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
    try:
        # ``a+b``: reads may seek, writes always land at EOF. A torn final line
        # (no trailing newline) gets a leading ``\n`` first — otherwise the new
        # record would glue onto the torn bytes and the pair would decode as
        # one invalid line, silently swallowing the new entry.
        with path.open("a+b") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            if size:
                handle.seek(size - 1)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write(line)
    except OSError as exc:
        return Result(error=f"{path.name}: unwritable: {exc}")
    return Result()


_LOCK_RETRIES = 50
_LOCK_RETRY_DELAY_SECONDS = 0.02


def _append_entry(path: Path, record: dict[str, Any]) -> Result[int]:
    """Allocate ``record["id"]`` and append it under the per-file lock.

    ``_next_id`` + ``_append`` as one critical section: two concurrent
    appenders (the foreman and, for inbox/signals, external writers) cannot
    interleave the read-modify-write and mint a duplicate id. The lock is a
    byte-range lock on ``<name>.lock`` beside the log — a bounded retry so a
    wedged writer surfaces as an error value rather than hanging a Turn.
    """
    lock_path = path.with_name(path.name + ".lock")
    lock = None
    for _ in range(_LOCK_RETRIES):
        lock = try_acquire_byte_range_lock(lock_path)
        if lock is not None:
            break
        time.sleep(_LOCK_RETRY_DELAY_SECONDS)
    if lock is None:
        return Result(error=f"{path.name}: append lock busy after {_LOCK_RETRIES} retries")
    try:
        record["id"] = _next_id(path)
        written = _append(path, record)
        if not written.ok:
            return Result(error=written.error)
        return Result(value=record["id"])
    finally:
        lock.release()


def _iter_entries(path: Path) -> Iterator[dict[str, Any]]:
    """Yield decoded entries, skipping blank lines and a torn final write."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
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
    stamp = ts or utc_now()
    record: dict[str, Any] = {"ts": stamp, "text": text}
    if turn_id is not None:
        record["turn_id"] = turn_id
    written = _append_entry(path, record)
    if not written.ok:
        return Result(error=written.error)
    return Result(value=JournalEntry(id=written.value, ts=stamp, text=text, turn_id=turn_id))


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
    stamp = ts or utc_now()
    written = _append_entry(path, {"ts": stamp, "text": text})
    if not written.ok:
        return Result(error=written.error)
    return Result(value=InboxMessage(id=written.value, ts=stamp, text=text))


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
    stamp = ts or utc_now()
    payload = data or {}
    written = _append_entry(path, {"ts": stamp, "source": source, "data": payload})
    if not written.ok:
        return Result(error=written.error)
    return Result(value=SignalEntry(id=written.value, ts=stamp, source=source, data=payload))


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
