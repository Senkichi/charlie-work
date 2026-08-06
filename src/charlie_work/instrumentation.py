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
from types import MappingProxyType
from typing import Any, Generator, Mapping

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

# Tracks unknown event kinds we have already warned about once. A one-time
# warning preserves the best-effort contract (log_event never raises) while
# still making unregistered kinds visible in the logs.
_unknown_kind_warned: set[str] = set()

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

# Event kind to ``level`` column mapping. This is the single source of truth
# for event-level classification; the old ``_ERROR_KINDS`` / ``_WARNING_KINDS``
# allow-lists are derived from it below for compatibility.
#
# A kind not present in this registry is classified as ``"info"`` with a
# warning, so the instrumentation layer stays best-effort and never breaks a
# caller. New kinds are caught instead by the static test that requires every
# literal kind passed to ``log_event`` / ``append_event`` / ``_record_event`` in
# this package to be registered.
_LEVEL_BY_KIND: Mapping[str, str] = MappingProxyType(
    {
        # -----------------------------------------------------------------
        # error-level kinds: conditions that ended a lane or lost work
        # -----------------------------------------------------------------
        "cross_family_verdict_abandoned": "error",
        # The head-SHA guard could not adjudicate, so the verdict is never
        # recorded and nothing regenerates the report: the PR stalls in
        # "reviewing" until a human intervenes. Ended a lane -> error.
        "cross_family_verdict_head_indeterminate": "error",
        "dispatch_blocked_chain_dead": "error",
        "dispatch_failed": "error",
        "fleet_pass_config_error": "error",
        "github_error": "error",
        "github_not_found_error": "error",
        "infra_rerun_escalated": "error",
        "intake_failed": "error",
        "janitor_rework_cycle_failed": "error",
        "janitor_rework_escalated": "error",
        "merge_blocked": "error",
        "merge_deferred_stale_base_alarm": "error",
        "merge_failed": "error",
        "merge_failed_attempt_alarm": "error",
        "operator_claim_failed": "error",
        "orphan_processes_killed": "error",
        "orphaned_worker_routed_to_review": "error",
        "pre_review_rework_routed": "error",
        "reconcile_pass_failed": "error",
        "rescue_review_escalated": "error",
        "review_checkout_removal_failed": "error",
        "review_dispatch_escalated": "error",
        "review_dispatch_stalled": "error",
        "review_verdict_missed": "error",
        "rework_requeued": "error",
        "self_deploy_alarm": "error",
        "self_deploy_failed": "error",
        "session_failed_escalated": "error",
        "session_failed_relabeled": "error",
        "session_salvaged": "error",
        "session_stalled": "error",
        "spec_review_failed": "error",
        "supervisor_zero_pass_alarm": "error",
        "unauthorized_merge_detected": "error",
        # -----------------------------------------------------------------
        # warning-level kinds: handled-but-notable conditions
        # -----------------------------------------------------------------
        "cross_family_verdict_unparseable": "warning",
        "deescalation_cap_exhausted": "warning",
        "dispatch_merged_pr_mention_flagged": "warning",
        "dispatch_merged_pr_references_closed": "warning",
        "dispatch_skip_blocked": "warning",
        "dispatch_skip_operator_claimed": "warning",
        "dispatch_stale": "warning",
        "draft_pr_blocked": "warning",
        "draft_pr_ready_failed": "warning",
        "draft_pr_ready_held": "warning",
        "flake_rerun_failed": "warning",
        "graphql_rate_limit_deferred": "warning",
        "infra_rerun_failed": "warning",
        "janitor_rework_stalled": "warning",
        "main_ci_reclaim_failed": "warning",
        "quota_probe_failed": "warning",
        "required_changes_vacuous": "warning",
        "review_dispatch_lifecycle_reaped": "warning",
        "review_packet_template_stale": "warning",
        "review_quota_exhausted": "warning",
        "rework_issue_fetch_skipped": "warning",
        "runner_allocation_refused": "warning",
        "runner_allocation_skipped": "warning",
        "runner_capacity_starved": "warning",
        "session_budget_exceeded": "warning",
        "session_exited": "warning",
        "session_rate_limit_deferred": "warning",
        "supervise_relaunch_cap_reached": "warning",
        "unauthorized_merge_check_skipped": "warning",
        "worktree_foreign_writer": "warning",
        # -----------------------------------------------------------------
        # info-level kinds: routine bookkeeping, success, recovery, and
        # other ordinary lifecycle events
        # -----------------------------------------------------------------
        "check_failure_rework_requested": "info",
        "closed_unmerged_pr_state_converged": "info",
        "containment_check": "info",
        "cross_pr_revert_rework_requested": "info",
        "deescalation_cleared": "info",
        "deescalation_pass_completed": "info",
        "deescalation_reason_class_backfilled": "info",
        "dispatch": "info",
        "dispatch_closed_unmerged_ready_stripped": "info",
        "dispatch_rework": "info",
        "draft_pr_ready_triggered": "info",
        "escalated_label_repaired": "info",
        "finalize_externally_merged": "info",
        "flake_rerun_triggered": "info",
        "fleet_canary": "info",
        "fleet_job_observations": "info",
        "head_moved": "info",
        "infra_rerun_triggered": "info",
        "intake": "info",
        "intake_prose_only_deps": "info",
        "janitor_gate": "info",
        "live_worker_redispatch_averted": "info",
        "loop_completed": "info",
        "loop_started": "info",
        "main_ci_reclaim_cancelled": "info",
        "merge_conflict_rework_requested": "info",
        "merge_deferred_stale_base": "info",
        "merge_ready": "info",
        "no_op_rework_repair_requested": "info",
        "operator_claim": "info",
        "operator_claim_released": "info",
        "orphaned_worker_drift": "info",
        "orphaned_worker_opened_pr": "info",
        "orphaned_worker_recovered": "info",
        "quota_probe_succeeded": "info",
        "readiness_no_ci_rework_requested": "info",
        "reconcile": "info",
        "reconcile_pass_completed": "info",
        "reconcile_pass_deferred": "info",
        "reconcile_pass_skipped": "info",
        "record_review": "info",
        "rescue_dispatched": "info",
        "review_dispatch": "info",
        "review_dispatch_claim": "info",
        "review_packet": "info",
        "rework_already_pushed": "info",
        "rework_brief_regenerated": "info",
        "runner_allocation": "info",
        "runner_capacity_recovered": "info",
        "self_deploy_skipped": "info",
        "self_deploy_succeeded": "info",
        "spec_review": "info",
        "stranded_request_changes_rework_requested": "info",
        "supervisor_exited": "info",
        "supervisor_started": "info",
        "unauthorized_merge_acknowledged": "info",
        "unauthorized_merge_baseline_armed": "info",
        "unescalate": "info",
        "verdict_carried_forward_clean_rebase": "info",
        "verdict_carried_forward_line_content": "info",
        "verdict_carried_forward_verified_sync": "info",
        "worktrees_reclaimed": "info",
    }
)

# Compatibility shims derived from the registry. Existing code and comments
# that refer to ``_ERROR_KINDS`` / ``_WARNING_KINDS`` continue to work.
_ERROR_KINDS = frozenset({k for k, v in _LEVEL_BY_KIND.items() if v == "error"})
_WARNING_KINDS = frozenset({k for k, v in _LEVEL_BY_KIND.items() if v == "warning"})


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
    """Classify an event kind into a log level for the ``level`` column.

    The registry is the source of truth. Kinds produced by the sweep
    aggregator (``{base}_sweep``) inherit the level of the base kind. Any
    still-unknown kind defaults to ``"info"`` so the instrumentation layer
    never breaks a caller; the test suite's
    ``test_event_kind_registry_exhaustive`` is the enforcement point that
    requires new kinds to be registered.
    """
    if kind in _LEVEL_BY_KIND:
        return _LEVEL_BY_KIND[kind]
    if kind.endswith("_sweep"):
        base = kind[: -len("_sweep")]
        if base in _LEVEL_BY_KIND:
            return _LEVEL_BY_KIND[base]
    return "info"


# Plural payload keys that carry lists of PR or issue numbers, consulted when
# the singular keys (``pr_number``/``pr``, ``issue_number``/``issue``) are
# absent. Ordered by preference: the first non-empty numeric list wins. Only
# the first numeric element is used to backfill the single-valued indexed
# column — the events table has one ``pr_number``/``issue_number`` slot per
# row, so a multi-ref event is indexed by its most representative ref (the
# first launched PR, the first dispatched issue). Non-numeric elements
# (dicts, strings) are skipped so list-of-summary shapes (e.g. ``issues`` as
# a list of dicts in CommandResult data) never produce a false ref.
_PR_PLURAL_KEYS: tuple[str, ...] = ("pr_numbers", "prs", "launched", "failed")
_ISSUE_PLURAL_KEYS: tuple[str, ...] = ("issue_numbers", "issues")


def _first_number_from_list(value: Any) -> int | None:
    """Return the first int/float element of a list, or None.

    Bools are excluded because ``isinstance(True, int)`` is True in Python but
    a boolean is never a valid PR/issue reference. Non-list values return None.
    """
    if not isinstance(value, list):
        return None
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)) and item == item:
            return int(item)
    return None


def _extract_payload_refs(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    """Extract pr_number and issue_number from a payload dict for indexed columns.

    These are the most common query dimensions for root-cause analysis.
    Returns ``(pr_number, issue_number)`` with None for absent values.

    Singular keys are tried first (``pr_number``/``pr``,
    ``issue_number``/``issue``). When absent, common plural keys
    (``issue_numbers``, ``pr_numbers``, ``launched``, ``failed``, …) are
    unwrapped and the first numeric element backfills the indexed column.
    This makes list-valued events (``dispatch``, ``review_dispatch``,
    ``dispatch_rework``, ``review_dispatch_claim``) visible to
    ``query_events``/``events_by_correlation_id`` PR/issue filtering instead
    of landing with NULL refs (issue #553).
    """
    pr_number = payload.get("pr_number")
    if pr_number is None:
        pr_number = payload.get("pr")
    if pr_number is None:
        for key in _PR_PLURAL_KEYS:
            candidate = _first_number_from_list(payload.get(key))
            if candidate is not None:
                pr_number = candidate
                break
    issue_number = payload.get("issue_number")
    if issue_number is None:
        issue_number = payload.get("issue")
    if issue_number is None:
        for key in _ISSUE_PLURAL_KEYS:
            candidate = _first_number_from_list(payload.get(key))
            if candidate is not None:
                issue_number = candidate
                break
    return (
        int(pr_number) if isinstance(pr_number, (int, float)) and pr_number == pr_number else None,
        int(issue_number)
        if isinstance(issue_number, (int, float)) and issue_number == issue_number
        else None,
    )


def _migrate_jsonl(db_conn: sqlite3.Connection, jsonl: Path) -> int:
    """Migrate existing events.jsonl entries into the SQLite database.

    Returns the number of newly inserted rows. Each line is parsed and
    inserted individually so a malformed line doesn't abort the whole
    migration.

    The migration is idempotent: a row is only inserted if no existing
    event shares its full ``(ts, kind, payload, repo, correlation_id,
    pr_number, issue_number)`` tuple. Using the complete meaningful row
    (not just ``(ts, kind, payload)``) ensures that distinct events which
    happen to share a timestamp/kind/payload but differ in ``repo``,
    ``correlation_id``, or PR/issue references are all preserved. This
    protects against a crash between the commit and the post-migration
    rename re-inserting the same legacy rows on the next process start.

    After a successful commit the legacy file is atomically renamed to
    ``events.jsonl.migrated`` (kept for audit) so subsequent processes do
    not re-run the migration. The rename uses ``Path.replace`` for the same
    atomic-rename discipline as other state writes.
    """
    if not jsonl.exists():
        return 0
    inserted = 0
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
                ts = record.get("ts", _now_iso())
                kind = record.get("kind", "unknown")
                payload_json = json.dumps(payload, sort_keys=True, default=str)
                repo_val = record.get("repo")
                cid_val = record.get("correlation_id")
                cursor = db_conn.execute(
                    """INSERT INTO events
                       (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
                       SELECT ?, ?, ?, ?, ?, ?, ?, ?
                       WHERE NOT EXISTS (
                           SELECT 1 FROM events
                           WHERE ts = ? AND kind = ? AND payload = ?
                             AND repo IS ?
                             AND correlation_id IS ?
                             AND pr_number IS ?
                             AND issue_number IS ?
                       )""",
                    (
                        ts,
                        kind,
                        payload_json,
                        repo_val,
                        cid_val,
                        pr_num,
                        issue_num,
                        _classify_level(kind),
                        ts,
                        kind,
                        payload_json,
                        repo_val,
                        cid_val,
                        pr_num,
                        issue_num,
                    ),
                )
                if cursor.rowcount > 0:
                    inserted += 1
        db_conn.commit()
    except OSError as exc:
        logger.warning("Failed to migrate events.jsonl at %s: %s", jsonl, exc)
        return inserted
    # Atomically rename the legacy file so the migration is one-shot.
    # The file is retained (as .migrated) for audit; it is never deleted.
    migrated_path = jsonl.with_suffix(jsonl.suffix + ".migrated")
    try:
        jsonl.replace(migrated_path)
    except OSError as exc:
        logger.warning("Failed to rename events.jsonl to %s: %s", migrated_path, exc)
    if inserted:
        logger.info("Migrated %d events from events.jsonl to events.db", inserted)
    return inserted


def _dedupe_events(db_conn: sqlite3.Connection) -> int:
    """Remove duplicate event rows, keeping the earliest inserted copy.

    Duplicates are identified by the full meaningful row tuple
    ``(ts, kind, payload, repo, correlation_id, pr_number, issue_number)``
    — every indexed column except the autoincrement ``id``. The row with
    the smallest ``id`` is retained. Returns the number of rows deleted.

    Using the *complete* row (including ``repo``) as the deduplication key
    is critical: ``_now_iso()`` truncates timestamps to 1-second precision,
    so distinct events from different repos (or different correlation
    contexts) can legitimately share ``(ts, kind, payload)`` within the
    same second. A narrower key would silently and irreversibly delete
    those distinct events from what this module calls its "complete audit
    history" store — inconsistent with the rename-not-delete treatment of
    ``events.jsonl``. Only rows that are identical across *all* meaningful
    columns are collapsed, which is true deduplication, not data loss.

    This is a one-time cleanup for databases polluted by the pre-fix
    migration that re-inserted legacy ``events.jsonl`` rows on every
    process start. The pollution produced true duplicates (the same JSONL
    record re-inserted with identical values across every column), so they
    are still caught by the full-row key. It is guarded by
    ``PRAGMA user_version`` so it runs exactly once per database file.
    """
    cursor = db_conn.execute(
        """DELETE FROM events
           WHERE id NOT IN (
               SELECT MIN(id) FROM events
               GROUP BY ts, kind, payload, repo, correlation_id, pr_number, issue_number
           )"""
    )
    deleted = cursor.rowcount
    db_conn.commit()
    if deleted:
        logger.info("Deduplicated %d duplicate event rows from events.db", deleted)
    return deleted


def _run_db_migrations(db_conn: sqlite3.Connection) -> None:
    """Run one-time database migrations guarded by ``PRAGMA user_version``.

    Each migration step bumps the version so it never re-runs on the same
    database file. This is the single enforcement point for historical
    cleanup of pollution caused by the pre-fix ``events.jsonl`` migration.
    """
    cursor = db_conn.execute("PRAGMA user_version")
    version = cursor.fetchone()[0]
    if version < 1:
        # Migration v1: dedupe rows polluted by the re-migrating jsonl
        # importer (issue #557). Runs once per database file.
        _dedupe_events(db_conn)
        db_conn.execute("PRAGMA user_version = 1")


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

            # Run one-time migrations (e.g. historical duplicate cleanup).
            _run_db_migrations(conn)

            # Migrate legacy events.jsonl if it exists
            jsonl = _jsonl_path(state_path)
            if jsonl.exists():
                _migrate_jsonl(conn, jsonl)

            with _db_init_lock:
                _db_connections[key] = conn
            return conn
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Failed to open event database at %s: %s", db_path, exc)
            return None


def log_event(
    state_path: Path,
    kind: str,
    payload: dict[str, Any],
    *,
    repo: str | None = None,
    correlation_id: str | None = None,
    level: str | None = None,
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
        level: Optional explicit level (``"info"``, ``"warning"``,
            ``"error"``). When omitted, the level is looked up in
            ``_LEVEL_BY_KIND``. This lets new call sites declare their level
            at the emission point without editing the registry.
    """
    cid = correlation_id or current_correlation_id()
    ts = _now_iso()
    payload_json = json.dumps(payload, sort_keys=True, default=str)
    pr_num, issue_num = _extract_payload_refs(payload)
    if level is None:
        level = _classify_level(kind)
        if kind not in _LEVEL_BY_KIND and not (
            kind.endswith("_sweep") and kind[: -len("_sweep")] in _LEVEL_BY_KIND
        ):
            if kind not in _unknown_kind_warned:
                _unknown_kind_warned.add(kind)
                logger.warning(
                    "Unknown event kind %r: defaulting to 'info'. "
                    "Register it in _LEVEL_BY_KIND or pass level= explicitly.",
                    kind,
                )
    elif level not in ("info", "warning", "error"):
        # Invalid explicit level is a programming mistake; fall back to the
        # registry rather than write a garbage level.
        logger.warning("Invalid level %r for kind %r; using registry/default", level, kind)
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


# --- ci_fleet seams ---------------------------------------------------------
# ci_fleet must never import charlie_work -- that would make it un-importable
# without charlie-work installed, which is the independence this extraction
# exists to create. So the *provider* registers itself, at module scope, after
# both functions above are defined.
#
# Both seams are required, and the reader is the one that looks redundant.
# Capacity signalling (#799) is edge-triggered, so "have I already signalled?"
# can only be answered by reading the store back. With no reader installed,
# query_events() returns None, the pass correctly declines to guess, and
# runner_capacity_starved never fires -- indistinguishable from a host that was
# never starved.
#
# This comment used to justify that by claiming the fleet pass is "a fresh
# process every cycle". It is not, and the correction strengthens the argument
# rather than weakening it. `fleet_dispatch.run_fleet_supervise` loads config
# once (fleet_dispatch.py:1729) and runs the pass loop in-process for the
# lifetime of the supervisor, so passes share a process across many cycles.
#
# That is exactly why the state must live in the store rather than a module
# global. Under the old false premise a global would fail immediately and
# obviously -- re-firing every pass, visible the first time anyone looked.
# Under the truth it survives within one process lifetime and is dropped only
# when the process is replaced (self-deploy restart, or the scheduled tick
# after supervise_loop's relaunch cap). It would pass every test and misfire
# rarely and non-deterministically, across respawns only.
#
# So: do not "optimise" this back into an in-memory global on discovering the
# fresh-process claim was false. The false premise was load-bearing for the
# wrong reason; the true one is a stronger argument for the same design.
# ci_fleet carried the identical claim on its half of this seam
# (runner_allocation_pass.py, observability.py) and corrected it in b20f3a4.
#
# The provenance anchor is the third seam and is installed here for the same
# reason as the other two: ci_fleet cannot fetch it (the boundary is one-way),
# so the provider has to hand it over. It is the one seam whose absence is
# *reported* rather than silent -- ci_fleet accumulates a `no_anchor` streak and
# escalates -- but only in ci_fleet's own logs and events, which nobody reads
# until something else has already gone wrong. See ci_fleet_anchor for why the
# declaration is read from pyproject.toml rather than from the install
# artifacts it is supposed to be checking.
from ci_fleet.observability import set_event_query, set_event_sink  # noqa: E402
from ci_fleet.provenance import set_provenance_anchor  # noqa: E402

from charlie_work.ci_fleet_anchor import declared_ci_fleet_root  # noqa: E402

set_event_sink(log_event)
set_event_query(query_events)
set_provenance_anchor(declared_ci_fleet_root)
