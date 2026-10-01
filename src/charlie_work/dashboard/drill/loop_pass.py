"""Loop-pass drill-down: the ordered event sequence of one correlation id, from one events.db.

``instrumentation.events_by_correlation_id`` opens the database through ``_get_db`` (the
read-write writer path, which creates the file), so it is not used here. This runs the same
``WHERE correlation_id = ? ORDER BY id`` query over a ``?mode=ro`` connection that
``sources.open_events_ro`` opens. The path is taken from the fleet registry entry for the
slug, never built from the request.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from datetime import tzinfo
from typing import Any

from .. import sources as src
from .types import (
    DrillError,
    PassDrill,
    PassEvent,
    check_slug,
    local_text,
    valid_correlation_id,
)

MAX_PASS_EVENTS = 2000
PREVIEW_CHARS = 160
_VALUE_CHARS = 48
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")


def _scalar(value: Any) -> str:
    if isinstance(value, dict):
        return f"{{{len(value)} keys}}"
    if isinstance(value, list):
        return f"[{len(value)} items]"
    text = _CONTROL.sub(" ", json.dumps(value) if not isinstance(value, str) else value)
    return text if len(text) <= _VALUE_CHARS else text[: _VALUE_CHARS - 1] + "…"


def payload_preview(raw: Any) -> str:
    """Compact ``key=value`` summary of an event payload, never the whole blob.

    Control characters are removed and the result is cut at ``PREVIEW_CHARS``; nested values
    show only their size. The text is plain: the renderer escapes it for HTML.
    """
    try:
        payload = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except ValueError:
        return "(unreadable payload)"
    if not isinstance(payload, dict):
        return "" if payload is None else _scalar(payload)
    parts: list[str] = []
    used = 0
    for key, value in payload.items():
        part = f"{_CONTROL.sub(' ', str(key))[:24]}={_scalar(value)}"
        if used + len(part) > PREVIEW_CHARS:
            parts.append("…")
            break
        parts.append(part)
        used += len(part) + 1
    return " ".join(parts)


def _summary(
    conn: sqlite3.Connection, cid: str
) -> tuple[str | None, str | None, bool | None, float | None]:
    try:
        row = conn.execute(
            "SELECT started_at, completed_at, ok, elapsed_seconds FROM loop_passes"
            " WHERE correlation_id = ?",
            (cid,),
        ).fetchone()
    except sqlite3.Error:
        return None, None, None, None  # an older events.db without loop_passes: events alone
    if row is None:
        return None, None, None, None
    return row[0], row[1], None if row[2] is None else bool(row[2]), row[3]


def _events(rows: Sequence[sqlite3.Row], tz: tzinfo | None) -> tuple[PassEvent, ...]:
    return tuple(
        PassEvent(
            id=r["id"],
            ts=r["ts"],
            ts_local=local_text(r["ts"], tz),
            level=r["level"] or "info",
            kind=r["kind"],
            issue=r["issue_number"],
            pr=r["pr_number"],
            preview=payload_preview(r["payload"]),
        )
        for r in rows
    )


def pass_drill(
    slug: str,
    correlation_id: str,
    repos: Sequence[src.RepoSource],
    *,
    tz: tzinfo | None = None,
) -> PassDrill | DrillError:
    """Every event of one loop pass of ``slug`` in insertion order (read-only)."""
    if (bad := check_slug(slug)) is not None:
        return bad
    if not valid_correlation_id(correlation_id):
        return DrillError("invalid", "not a correlation id")
    repo = next((r for r in repos if r.key == slug), None)
    if repo is None:
        return DrillError("not_found", "no such repo in the fleet registry")
    conn, err = src.open_events_ro(repo.events_db)
    if conn is None:
        return DrillError("unavailable", f"events unavailable: {err}")
    try:
        rows = conn.execute(
            "SELECT id, ts, kind, payload, level, issue_number, pr_number FROM events"
            " WHERE correlation_id = ? ORDER BY id ASC LIMIT ?",
            (correlation_id, MAX_PASS_EVENTS + 1),
        ).fetchall()
        started, completed, ok, elapsed = _summary(conn, correlation_id)
    except sqlite3.Error as exc:
        return DrillError("unavailable", f"events unreadable: {exc}")
    finally:
        conn.close()
    if not rows and started is None:
        return DrillError("not_found", "no events for that correlation id")
    events = _events(rows[:MAX_PASS_EVENTS], tz)
    levels = Counter(e.level for e in events)
    return PassDrill(
        repo=slug,
        correlation_id=correlation_id,
        events=events,
        truncated=len(rows) > MAX_PASS_EVENTS,
        level_counts=tuple(sorted(levels.items())),
        started_at=started,
        completed_at=completed,
        ok=ok,
        elapsed_seconds=elapsed,
    )
