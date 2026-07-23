"""SQLite-backed structured event log and correlation-ID infrastructure.

This module provides the architecturally robust instrumentation layer for
charlie-work. It complements ``state.json``'s 200-entry ``events`` array
(which serves as a convenience cache for recent activity) with an unlimited,
append-only SQLite database (``events.db``) that preserves the complete audit
history for root-cause analysis.

Key design decisions:

1. **SQLite, not JSONL**: The database lives alongside ``state.json`` as
   ``events.db``. SQLite provides indexed lookups, aggregation queries,
   and concurrent reads (WAL mode) while remaining zero-dependency (stdlib
   ``sqlite3``). The previous JSONL file is migrated automatically on first
   access.

2. **Indexed query columns**: High-value fields (``kind``, ``ts``,
   ``correlation_id``, ``pr_number``, ``issue_number``, ``repo``) are
   extracted from the payload into typed, indexed columns for O(log n)
   filtering. The full payload is preserved as a JSON blob for flexibility.

3. **Correlation IDs**: A thread-local correlation ID links all events
   from a single ``loop()`` pass (or any other top-level operation),
   making it trivial to reconstruct a complete timeline of what happened
   in a single orchestration cycle.

4. **Best-effort, never fatal**: Event logging failures are swallowed
   and logged via standard Python logging. Instrumentation must never
   break the orchestrator's core workflow.

Schema::

    CREATE TABLE events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts              TEXT    NOT NULL,
        kind            TEXT    NOT NULL,
        payload         TEXT    NOT NULL,      -- JSON blob
        repo            TEXT,
        correlation_id  TEXT,
        pr_number       INTEGER,
        issue_number    INTEGER,
        level           TEXT DEFAULT 'info'
    );

    CREATE INDEX idx_events_correlation_id ON events(correlation_id);
    CREATE INDEX idx_events_kind           ON events(kind);
    CREATE INDEX idx_events_ts             ON events(ts);
    CREATE INDEX idx_events_pr             ON events(pr_number);
    CREATE INDEX idx_events_issue          ON events(issue_number);

    CREATE TABLE loop_passes (
        correlation_id  TEXT PRIMARY KEY,
        started_at      TEXT    NOT NULL,
        completed_at    TEXT,
        ok              INTEGER,
        elapsed_seconds REAL,
        error_count     INTEGER DEFAULT 0,
        merge_count     INTEGER DEFAULT 0,
        review_count    INTEGER DEFAULT 0
    );
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Thread-local storage for the current correlation ID.
_correlation_local = threading.local()

# Per-path connection cache with thread-safe initialization.
# We keep one connection per state_path (database file) to amortize
# open/PRAGMA overhead. Connections use check_same_thread=False with
# a threading.Lock for write serialization.
_db_locks: dict[str, threading.Lock] = {}
_db_connections: dict[str, sqlite3.Connection] = {}
_db_init_lock = threading.Lock()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    kind            TEXT    NOT NULL,
    payload         TEXT    NOT NULL,
    repo            TEXT,
    correlation_id  TEXT,
    pr_number       INTEGER,
    issue_number    INTEGER,
    level           TEXT DEFAULT 'info'
);

CREATE INDEX IF NOT EXISTS idx_events_correlation_id ON events(correlation_id);
CREATE INDEX IF NOT EXISTS idx_events_kind           ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_ts             ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_pr             ON events(pr_number);
CREATE INDEX IF NOT EXISTS idx_events_issue          ON events(issue_number);

CREATE TABLE IF NOT EXISTS loop_passes (
    correlation_id  TEXT PRIMARY KEY,
    started_at      TEXT    NOT NULL,
    completed_at    TEXT,
    ok              INTEGER,
    elapsed_seconds REAL,
    error_count     INTEGER DEFAULT 0,
    merge_count     INTEGER DEFAULT 0,
    review_count    INTEGER DEFAULT 0
);
"""

# Event kinds that are considered errors or warnings for the ``level`` column.
_ERROR_KINDS = frozenset(
    {
        "github_error",
        "github_not_found_error",
        "intake_failed",
        "session_stalled",
        "session_failed_escalated",
        "session_failed_relabeled",
        "session_salvaged",
        "review_dispatch_stalled",
        "review_checkout_removal_failed",
        "dispatch_failed",
        "orphan_processes_killed",
        "orphaned_worker_routed_to_review",
        "pre_review_rework_routed",
        "rework_requeued",
        "merge_blocked",
        "merge_failed",
        "spec_review_failed",
        "operator_claim_failed",
    }
)
_WARNING_KINDS = frozenset(
    {
        "dispatch_skip_blocked",
        "dispatch_skip_operator_claimed",
        "dispatch_merged_pr_references_closed",
        "dispatch_merged_pr_mention_flagged",
        "review_dispatch_lifecycle_reaped",
        "session_rate_limit_deferred",
    }
)


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with 'Z' suffix."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_correlation_id() -> str | None:
    """Return the current thread-local correlation ID, or None if not set."""
    return getattr(_correlation_local, "correlation_id", None)


def _set_correlation_id(cid: str | None) -> None:
    _correlation_local.correlation_id = cid


@contextmanager
def correlation_context(correlation_id: str | None = None) -> Generator[str, None, None]:
    """Set a correlation ID for the current thread for the duration of the block.

    If ``correlation_id`` is None, a new UUID4 hex string is generated.
    The previous value is restored on exit (supporting nesting).

    Yields the active correlation ID so callers can log it or pass it along.
    """
    cid = correlation_id or uuid.uuid4().hex[:12]
    prev = getattr(_correlation_local, "correlation_id", None)
    _set_correlation_id(cid)
    try:
        yield cid
    finally:
        _set_correlation_id(prev)


def _db_path(state_path: Path) -> Path:
    """Derive the ``events.db`` SQLite path from a ``state.json`` path."""
    return state_path.parent / "events.db"


def _jsonl_path(state_path: Path) -> Path:
    """Derive the legacy ``events.jsonl`` path from a ``state.json`` path."""
    return state_path.parent / "events.jsonl"


def _classify_level(kind: str) -> str:
    """Classify an event kind into a log level for the ``level`` column."""
    if kind in _ERROR_KINDS:
        return "error"
    if kind in _WARNING_KINDS:
        return "warning"
    return "info"


def _extract_payload_refs(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    """Extract pr_number and issue_number from a payload dict for indexed columns.

    These are the most common query dimensions for root-cause analysis.
    Returns ``(pr_number, issue_number)`` with None for absent values.
    """
    pr_number = payload.get("pr_number")
    if pr_number is None:
        pr_number = payload.get("pr")
    issue_number = payload.get("issue_number")
    if issue_number is None:
        issue_number = payload.get("issue")
    return (
        int(pr_number) if isinstance(pr_number, (int, float)) and pr_number == pr_number else None,
        int(issue_number)
        if isinstance(issue_number, (int, float)) and issue_number == issue_number
        else None,
    )


def _migrate_jsonl(db_conn: sqlite3.Connection, jsonl: Path) -> int:
    """Migrate existing events.jsonl entries into the SQLite database.

    Returns the number of migrated rows. Each line is parsed and inserted
    individually so a malformed line doesn't abort the whole migration.
    """
    if not jsonl.exists():
        return 0
    migrated = 0
    try:
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = record.get("payload", {})
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        payload = {}
                if not isinstance(payload, dict):
                    payload = {}
                pr_num, issue_num = _extract_payload_refs(payload)
                db_conn.execute(
                    """INSERT OR IGNORE INTO events
                       (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.get("ts", _now_iso()),
                        record.get("kind", "unknown"),
                        json.dumps(payload, sort_keys=True, default=str),
                        record.get("repo"),
                        record.get("correlation_id"),
                        pr_num,
                        issue_num,
                        _classify_level(record.get("kind", "unknown")),
                    ),
                )
                migrated += 1
        db_conn.commit()
    except OSError as exc:
        logger.warning("Failed to migrate events.jsonl at %s: %s", jsonl, exc)
    if migrated:
        logger.info("Migrated %d events from events.jsonl to events.db", migrated)
    return migrated


def _get_db(state_path: Path) -> sqlite3.Connection | None:
    """Get or create a SQLite connection for the given state_path.

    Returns None if the database cannot be opened (best-effort semantics).
    The connection is cached per database path and reused across calls.
    Thread safety is ensured via a per-path lock.
    """
    db_path = _db_path(state_path)
    key = str(db_path.resolve())

    with _db_init_lock:
        if key not in _db_locks:
            _db_locks[key] = threading.Lock()
        if key in _db_connections:
            return _db_connections[key]

    lock = _db_locks[key]
    with lock:
        # Double-check after acquiring lock
        if key in _db_connections:
            return _db_connections[key]
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
                isolation_level=None,  # autocommit mode; we manage transactions explicitly
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA_SQL)

            # Migrate legacy events.jsonl if it exists
            jsonl = _jsonl_path(state_path)
            if jsonl.exists():
                _migrate_jsonl(conn, jsonl)

            with _db_init_lock:
                _db_connections[key] = conn
            return conn
        except sqlite3.Error as exc:
            logger.warning("Failed to open event database at %s: %s", db_path, exc)
            return None


def log_event(
    state_path: Path,
    kind: str,
    payload: dict[str, Any],
    *,
    repo: str | None = None,
    correlation_id: str | None = None,
) -> None:
    """Append a single structured event to the SQLite event log.

    This is the low-level write primitive. It is best-effort: any I/O error
    is caught and logged via standard Python logging so that instrumentation
    never breaks the orchestrator's core workflow.

    Args:
        state_path: Path to ``state.json`` — the event database is written
            alongside it as ``events.db``.
        kind: Event type string (e.g. ``"dispatch"``, ``"loop_started"``).
        payload: Event-specific data dict.
        repo: Optional repo name for cross-repo fleet correlation.
        correlation_id: Optional correlation ID. If not provided, the
            current thread-local correlation ID is used (may be None).
    """
    cid = correlation_id or current_correlation_id()
    ts = _now_iso()
    payload_json = json.dumps(payload, sort_keys=True, default=str)
    pr_num, issue_num = _extract_payload_refs(payload)
    level = _classify_level(kind)

    conn = _get_db(state_path)
    if conn is None:
        return

    key = str(_db_path(state_path).resolve())
    lock = _db_locks.get(key)
    if lock is None:
        return
    try:
        with lock:
            conn.execute(
                """INSERT INTO events
                   (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, kind, payload_json, repo, cid, pr_num, issue_num, level),
            )
    except sqlite3.Error as exc:
        logger.warning("Failed to write event to %s: %s", _db_path(state_path), exc)


def record_loop_pass(
    state_path: Path,
    correlation_id: str,
    started_at: str,
    completed_at: str | None = None,
    *,
    ok: bool | None = None,
    elapsed_seconds: float | None = None,
    error_count: int = 0,
    merge_count: int = 0,
    review_count: int = 0,
) -> None:
    """Record or update a loop pass summary in the ``loop_passes`` table.

    On first call (with ``completed_at=None``) an INSERT is issued.
    On second call (with ``completed_at`` set) an UPDATE is issued.
    """
    conn = _get_db(state_path)
    if conn is None:
        return
    key = str(_db_path(state_path).resolve())
    lock = _db_locks.get(key)
    if lock is None:
        return
    try:
        with lock:
            if completed_at is None:
                conn.execute(
                    """INSERT OR IGNORE INTO loop_passes
                       (correlation_id, started_at, completed_at, ok,
                        elapsed_seconds, error_count, merge_count, review_count)
                       VALUES (?, ?, NULL, NULL, NULL, 0, 0, 0)""",
                    (correlation_id, started_at),
                )
            else:
                conn.execute(
                    """UPDATE loop_passes
                       SET completed_at = ?, ok = ?, elapsed_seconds = ?,
                           error_count = ?, merge_count = ?, review_count = ?
                       WHERE correlation_id = ?""",
                    (
                        completed_at,
                        1 if ok else 0,
                        elapsed_seconds,
                        error_count,
                        merge_count,
                        review_count,
                        correlation_id,
                    ),
                )
    except sqlite3.Error as exc:
        logger.warning("Failed to record loop pass: %s", exc)


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a database row to an event dict matching the old JSONL format."""
    return {
        "ts": row["ts"],
        "kind": row["kind"],
        "payload": json.loads(row["payload"]),
        "repo": row["repo"],
        "correlation_id": row["correlation_id"],
        "pr_number": row["pr_number"],
        "issue_number": row["issue_number"],
        "level": row["level"],
    }


def read_event_log(state_path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Read events from the SQLite event database.

    Args:
        state_path: Path to ``state.json``.
        limit: If provided, return only the last N events (by insertion order).

    Returns:
        A list of event dicts, oldest first (or the last N if limited).
    """
    conn = _get_db(state_path)
    if conn is None:
        return []
    try:
        if limit is not None:
            cursor = conn.execute(
                """SELECT * FROM (
                       SELECT * FROM events ORDER BY id DESC LIMIT ?
                   ) ORDER BY id ASC""",
                (limit,),
            )
        else:
            cursor = conn.execute("SELECT * FROM events ORDER BY id ASC")
        return [_row_to_event(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        logger.warning("Failed to read event log: %s", exc)
        return []


def events_by_correlation_id(state_path: Path, correlation_id: str) -> list[dict[str, Any]]:
    """Return all events sharing a correlation ID, in chronological order.

    This is the primary investigation tool: given a loop pass correlation ID
    (e.g. from a notification or error report), reconstruct the complete
    timeline of everything that happened in that pass.
    """
    conn = _get_db(state_path)
    if conn is None:
        return []
    try:
        cursor = conn.execute(
            "SELECT * FROM events WHERE correlation_id = ? ORDER BY id ASC",
            (correlation_id,),
        )
        return [_row_to_event(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        logger.warning("Failed to query events by correlation ID: %s", exc)
        return []


def query_events(
    state_path: Path,
    *,
    kind: str | None = None,
    correlation_id: str | None = None,
    pr_number: int | None = None,
    issue_number: int | None = None,
    repo: str | None = None,
    level: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Query events with structured filters against indexed columns.

    All filter parameters are optional; only provided filters are applied.
    Results are ordered chronologically (by insertion id).

    Args:
        state_path: Path to ``state.json``.
        kind: Filter by event kind (exact match).
        correlation_id: Filter by correlation ID.
        pr_number: Filter by PR number.
        issue_number: Filter by issue number.
        repo: Filter by repo name.
        level: Filter by log level ('info', 'warning', 'error').
        since: ISO-8601 timestamp; only events at or after this time.
        until: ISO-8601 timestamp; only events at or before this time.
        limit: Maximum number of events to return (most recent N).

    Returns:
        A list of event dicts, oldest first.
    """
    conn = _get_db(state_path)
    if conn is None:
        return []
    conditions: list[str] = []
    params: list[Any] = []
    if kind is not None:
        conditions.append("kind = ?")
        params.append(kind)
    if correlation_id is not None:
        conditions.append("correlation_id = ?")
        params.append(correlation_id)
    if pr_number is not None:
        conditions.append("pr_number = ?")
        params.append(pr_number)
    if issue_number is not None:
        conditions.append("issue_number = ?")
        params.append(issue_number)
    if repo is not None:
        conditions.append("repo = ?")
        params.append(repo)
    if level is not None:
        conditions.append("level = ?")
        params.append(level)
    if since is not None:
        conditions.append("ts >= ?")
        params.append(since)
    if until is not None:
        conditions.append("ts <= ?")
        params.append(until)

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    sql = f"SELECT * FROM events WHERE {where_clause} ORDER BY id ASC"
    if limit is not None:
        sql = f"SELECT * FROM (SELECT * FROM events WHERE {where_clause} ORDER BY id DESC LIMIT ?) ORDER BY id ASC"
        params.append(limit)

    try:
        cursor = conn.execute(sql, params)
        return [_row_to_event(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        logger.warning("Failed to query events: %s", exc)
        return []


def event_counts_by_kind(state_path: Path, *, since: str | None = None) -> dict[str, int]:
    """Return a summary of event counts grouped by kind.

    Useful for quick dashboards: "what kinds of things happened?"
    """
    conn = _get_db(state_path)
    if conn is None:
        return {}
    try:
        if since is not None:
            cursor = conn.execute(
                "SELECT kind, COUNT(*) FROM events WHERE ts >= ? GROUP BY kind ORDER BY COUNT(*) DESC",
                (since,),
            )
        else:
            cursor = conn.execute(
                "SELECT kind, COUNT(*) FROM events GROUP BY kind ORDER BY COUNT(*) DESC"
            )
        return {row[0]: row[1] for row in cursor.fetchall()}
    except sqlite3.Error as exc:
        logger.warning("Failed to get event counts: %s", exc)
        return {}


def close_db(state_path: Path) -> None:
    """Close the database connection for the given state_path.

    Primarily useful for tests that need to ensure clean teardown.
    """
    db_path = _db_path(state_path)
    key = str(db_path.resolve())
    with _db_init_lock:
        lock = _db_locks.get(key)
        conn = _db_connections.pop(key, None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    if lock is not None:
        with _db_init_lock:
            _db_locks.pop(key, None)
