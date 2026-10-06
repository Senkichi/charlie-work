"""Derived rollup: fleet events.db files -> ``dashboard.db`` fact tables (ADR-0008).

``dashboard.db`` is a disposable cache. Source databases are only ever opened
read-only (``sources.open_events_ro``); the sole writer connection here is the one on
``dashboard.db``. Errors come back as values (``RollupResult``), never raise.

Incremental ingest is keyed by a ``(source, max ingested id)`` watermark. Because
``instrumentation._dedupe_events`` can delete rows, each run also re-derives a trailing
window (``WINDOW``): a source's fact rows stamped inside the window are deleted
and rebuilt, so a deduped event disappears from the facts instead of lingering. A source
whose max id fell below its watermark was replaced, and is rebuilt from scratch.

Repo attribution is the database an event came from, never the ``repo`` column (recon
section 2). ``GLOBAL_ONLY_KINDS`` are written to both the global DB and per-repo DBs; the
global DB is authoritative, but the per-repo copies are the only record of the period
before the global DB started writing them. A per-repo copy is therefore ingested only for
timestamps strictly before the global DB's first row of that kind (``_kind_filter``), so
the two never overlap and nothing is counted twice.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .. import layout
from . import sources as src
from .rollup_common import _int
from .rollup_derive import GLOBAL_ONLY_KINDS, HANDLERS, derive_event, is_classified
from .rollup_schema import (
    FLEET_SOURCE,
    LOOP_PASS_COLUMNS,
    SCHEMA_VERSION,
    SOURCE_SCOPED_TABLES,
    WINDOWED_TABLES,
    schema_statements,
)

WINDOW = timedelta(hours=6)
_IGNORE_TABLES = frozenset({"job_observations"})
# reconcile sub-kind repeated every pass for the same issue (recon section 4): noise.
_STALE_RECONCILE = "terminal_state_stale"
log = logging.getLogger("charlie_work.dashboard")


@dataclass(frozen=True)
class RollupSources:
    """Everything a rollup reads and writes; built from the registry, never globbed."""

    db_path: Path
    fleet_events_db: Path
    repos: tuple[tuple[str, Path], ...]  # (repo key, that repo's events.db)
    registry_error: str | None = None  # fleet.json exists but is unusable


@dataclass(frozen=True)
class SourceResult:
    source: str
    ingested: int  # handled events newer than the watermark
    rederived: int  # handled events re-derived inside the trailing window
    rebuilt: bool  # source wiped and re-ingested from id 1
    error: str | None = None
    # kind -> row count, for every kind in this source's events table the rollup
    # neither handles nor declares in KNOWN_IGNORED (issue #2269).
    unclassified: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RollupResult:
    sources: tuple[SourceResult, ...]
    db_rebuilt: bool  # dashboard.db was dropped (missing schema version / mismatch)
    error: str | None = None  # dashboard.db itself unusable
    registry_error: str | None = None  # fleet.json corrupt: repos silently missing

    @property
    def ingested(self) -> int:
        return sum(s.ingested for s in self.sources)

    @property
    def errors(self) -> tuple[str, ...]:
        found = [f"{s.source}: {s.error}" for s in self.sources if s.error]
        head = [e for e in (self.error, self.registry_error) if e]
        return tuple(head + found)


def rollup_sources(fleet_dir_override: str | None = None) -> RollupSources:
    """Resolve rollup inputs from ``fleet.json`` plus the global events DB."""
    repos, registry_error = src.load_repos(fleet_dir_override)
    return RollupSources(
        db_path=layout.dashboard_db_path(override=fleet_dir_override),
        fleet_events_db=src.fleet_sources(fleet_dir_override).events_db,
        repos=tuple((r.key, r.events_db) for r in repos),
        registry_error=registry_error,
    )


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return [r[0] for r in rows]


def _open_dashboard_db(path: Path) -> tuple[sqlite3.Connection | None, bool, str | None]:
    """Open dashboard.db, creating it or rebuilding it in place on a version mismatch.

    The file is never unlinked: another process (collector, CLI) may be creating or
    rebuilding it at the same moment. Instead the check, drop and re-create run under one
    ``BEGIN IMMEDIATE``, which serialises concurrent openers on SQLite's own write lock.
    A directory that does not exist is an error (a mistyped fleet dir must not grow a
    ghost dashboard.db). A file that is not a database at all is an error value too;
    deleting it is the operator's call.
    """
    if not path.parent.is_dir():
        return None, False, f"cannot open {path}: fleet directory does not exist"
    conn: sqlite3.Connection | None = None
    try:
        # Generous busy timeout: a concurrent cold rollup holds the write lock for its
        # whole first ingest, and waiting beats failing the second runner.
        conn = sqlite3.connect(path, isolation_level=None, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN IMMEDIATE")
        try:
            tables = _user_tables(conn)
            row = (
                conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
                if "meta" in tables
                else None
            )
            rebuilt = False
            if row is None or row[0] != str(SCHEMA_VERSION):
                rebuilt = bool(tables)  # a cache: drop stale tables, rebuild from sources
                for table in tables:
                    conn.execute(f'DROP TABLE "{table}"')
                for statement in schema_statements():
                    conn.execute(statement)
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return conn, rebuilt, None
    except (sqlite3.Error, OSError) as exc:
        if conn is not None:
            conn.close()
        return None, False, f"cannot open {path}: {exc}"


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    ev = dict(row)
    try:
        payload = json.loads(ev["payload"]) if ev["payload"] else {}
    except ValueError:
        payload = {}
    ev["payload"] = payload if isinstance(payload, dict) else {}
    return ev


def _insert(dst: sqlite3.Connection, table: str, cols: dict[str, Any]) -> None:
    verb = "INSERT OR IGNORE" if table in _IGNORE_TABLES else "INSERT OR REPLACE"
    names = ", ".join(cols)
    marks = ", ".join("?" for _ in cols)
    dst.execute(f"{verb} INTO {table} ({names}) VALUES ({marks})", tuple(cols.values()))


def _copy_loop_passes(
    srcdb: sqlite3.Connection, dst: sqlite3.Connection, source: str, cutoff: str | None
) -> None:
    where, args = ("", ())
    if cutoff is not None:  # unfinished passes keep changing, so always re-copy them
        where, args = " WHERE started_at >= ? OR completed_at IS NULL", (cutoff,)
    cols = ", ".join(LOOP_PASS_COLUMNS)
    try:
        rows = srcdb.execute(f"SELECT {cols} FROM loop_passes{where}", args).fetchall()
    except sqlite3.OperationalError:
        return  # source predates loop_passes
    marks = ", ".join("?" for _ in range(len(LOOP_PASS_COLUMNS) + 1))
    for row in rows:
        dst.execute(
            f"INSERT OR REPLACE INTO loop_passes (source, {cols}) VALUES ({marks})",
            (source, *_typed_pass(tuple(row))),
        )


_PASS_INT_COLUMNS = frozenset(LOOP_PASS_COLUMNS) - {
    "correlation_id", "started_at", "completed_at", "elapsed_seconds"
}  # fmt: skip


def _typed_pass(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Coerce a copied pass row to the column types: SQLite INT affinity keeps a
    non-numeric TEXT count as TEXT, and dashboard.db must never hold one."""
    out = []
    for name, value in zip(LOOP_PASS_COLUMNS, row, strict=True):
        if name in _PASS_INT_COLUMNS:
            value = _int(value)
        elif name == "elapsed_seconds" and not isinstance(value, (int, float)):
            value = None
        out.append(value)
    return tuple(out)


def _write_coverage(
    srcdb: sqlite3.Connection, dst: sqlite3.Connection, source: str
) -> list[tuple]:
    """Refresh the source's coverage rows; return the per-kind rows they came from.

    The caller reuses the grouped counts for ``SourceResult.unclassified`` (issue
    #2269) so the classification never needs a second scan of the events table.
    """
    dst.execute("DELETE FROM coverage WHERE source = ?", (source,))
    rows = srcdb.execute(
        "SELECT kind, MIN(ts), MAX(ts), COUNT(*) FROM events GROUP BY kind"
    ).fetchall()
    total_n = sum(r[3] for r in rows)
    if not rows:
        return rows
    first, last = min(r[1] for r in rows), max(r[2] for r in rows)
    dst.execute(
        "INSERT INTO coverage (source, kind, first_ts, last_ts, n) VALUES (?, '*', ?, ?, ?)",
        (source, first, last, total_n),
    )
    dst.executemany(
        "INSERT INTO coverage (source, kind, first_ts, last_ts, n) VALUES (?, ?, ?, ?, ?)",
        [(source, str(k), lo, hi, n) for k, lo, hi, n in rows],
    )
    return rows


def _write_pulse(srcdb: sqlite3.Connection, dst: sqlite3.Connection, source: str) -> None:
    """Hours (``YYYY-MM-DDTHH``) in which the source wrote an event / a dispatch / a loop pass.

    Lets a metric tell a quiet hour from a blackout: coverage alone only has first/last ts.
    """
    dst.execute("DELETE FROM pulse WHERE source = ?", (source,))
    for hour, is_dispatch in srcdb.execute(
        "SELECT DISTINCT substr(ts, 1, 13), kind = 'dispatch' FROM events"
    ):
        dst.execute("INSERT OR IGNORE INTO pulse VALUES (?, 'events', ?)", (source, hour))
        if is_dispatch:
            dst.execute("INSERT OR IGNORE INTO pulse VALUES (?, 'dispatch', ?)", (source, hour))
    dst.execute(
        "INSERT OR IGNORE INTO pulse SELECT source, 'loop_pass', substr(completed_at, 1, 13)"
        " FROM loop_passes WHERE source = ? AND completed_at IS NOT NULL",
        (source,),
    )


def _wipe_source(dst: sqlite3.Connection, source: str) -> None:
    for table in SOURCE_SCOPED_TABLES:
        dst.execute(f"DELETE FROM {table} WHERE source = ?", (source,))


_NO_GLOBAL_ROWS = "9999-12-31T23:59:59Z"


def _kind_filter(dst: sqlite3.Connection, source: str) -> tuple[str, tuple[str, ...]]:
    """SQL predicate over ``kind`` (plus its args) for what ``source`` may contribute.

    The global source takes every handled kind. A per-repo source takes the ordinary kinds
    plus, for each ``GLOBAL_ONLY_KINDS`` kind, only rows older than the global DB's first row
    of that kind (taken from the global source's ``coverage``, written earlier in the same
    pass because the global source is ingested first). No global row yet means every copy
    predates it, so all are admitted.
    """
    if source == FLEET_SOURCE:
        kinds = sorted(HANDLERS)
        return f"kind IN ({', '.join('?' for _ in kinds)})", tuple(kinds)
    ordinary = sorted(k for k in HANDLERS if k not in GLOBAL_ONLY_KINDS)
    first = dict(
        dst.execute(
            "SELECT kind, first_ts FROM coverage WHERE source = ? AND first_ts IS NOT NULL",
            (FLEET_SOURCE,),
        ).fetchall()
    )
    clauses = [f"kind IN ({', '.join('?' for _ in ordinary)})"]
    args: list[str] = list(ordinary)
    for kind in sorted(GLOBAL_ONLY_KINDS & HANDLERS.keys()):
        clauses.append("(kind = ? AND ts < ?)")
        args += [kind, first.get(kind, _NO_GLOBAL_ROWS)]
    return " OR ".join(clauses), tuple(args)


def _ingest_source(dst: sqlite3.Connection, source: str, db: Path, cutoff: str) -> SourceResult:
    srcdb, err = src.open_events_ro(db)
    if srcdb is None:
        return SourceResult(source, 0, 0, False, err)
    try:
        max_id = srcdb.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0]
        # Take the write lock BEFORE reading the watermark: a concurrent pass that wins
        # the race commits its watermark first, so this one starts from it instead of
        # re-ingesting (and re-reporting) the same rows.
        dst.execute("BEGIN IMMEDIATE")
        try:
            row = dst.execute(
                "SELECT max_id FROM watermarks WHERE source = ?", (source,)
            ).fetchone()
            wm = row[0] if row else 0
            rebuilt = wm > max_id  # source DB was replaced under us
            if rebuilt:
                wm = 0
            kind_sql, kind_args = _kind_filter(dst, source)
            events = srcdb.execute(
                "SELECT id, ts, kind, payload, pr_number, issue_number FROM events"
                f" WHERE (id > ? OR ts >= ?) AND id <= ? AND ({kind_sql})"
                " AND NOT (kind = 'reconcile' AND CASE WHEN json_valid(payload)"
                f" THEN json_extract(payload, '$.kind') END IS '{_STALE_RECONCILE}') ORDER BY id",
                (wm, cutoff, max_id, *kind_args),
            )
            if wm == 0:
                _wipe_source(dst, source)
            else:
                for table in WINDOWED_TABLES:
                    dst.execute(
                        f"DELETE FROM {table} WHERE source = ? AND (ts >= ? OR src_id > ?)",
                        (source, cutoff, wm),
                    )
            ingested = rederived = 0
            for raw in events:
                ev = _decode(raw)
                if ev["id"] > wm:
                    ingested += 1
                else:
                    rederived += 1
                for table, cols in derive_event(source, ev):
                    _insert(dst, table, cols)
            if source != FLEET_SOURCE:
                _copy_loop_passes(srcdb, dst, source, cutoff if wm else None)
            coverage_rows = _write_coverage(srcdb, dst, source)
            _write_pulse(srcdb, dst, source)
            dst.execute(
                "INSERT OR REPLACE INTO watermarks (source, max_id) VALUES (?, ?)",
                (source, max_id),
            )
            dst.execute("COMMIT")
        except BaseException:
            dst.execute("ROLLBACK")
            raise
        unclassified = {
            str(kind): int(n)
            for kind, _first, _last, n in coverage_rows
            if not is_classified(str(kind))
        }
        return SourceResult(source, ingested, rederived, rebuilt, unclassified=unclassified)
    except sqlite3.Error as exc:
        return SourceResult(source, 0, 0, False, f"{type(exc).__name__}: {exc}")
    finally:
        srcdb.close()


def _record_unclassified(dst: sqlite3.Connection, results: tuple[SourceResult, ...]) -> None:
    """Persist the unclassified-kind counts and warn when the set changes (issue #2269).

    The JSON ``{source: {kind: count}}`` lands on ``meta`` under
    ``unclassified_kinds`` so read-only consumers (the History page) can name the
    kinds the rollup does not interpret. A warning fires only when the set of
    unclassified kinds differs from what the previous pass recorded -- a new kind
    appearing, one going away, or the first pass that sees any. A source whose
    ``error`` is set never read its events table this pass, so its empty
    ``unclassified`` is not "no unclassified kinds": the last recorded entry for
    that source is carried forward instead -- a transient read error must neither
    blank the kinds from the History page note nor fire a pair of false
    set-changed warnings (one on the error pass, one on recovery).
    """
    row = dst.execute("SELECT value FROM meta WHERE key = 'unclassified_kinds'").fetchone()
    try:
        stored = json.loads(row[0]) if row else {}
        previous = {kind for counts in stored.values() for kind in counts}
    except (ValueError, AttributeError, TypeError):
        stored, previous = {}, set()
    unclassified: dict[str, dict[str, int]] = {}
    for s in results:
        carried = stored.get(s.source) if s.error else None
        if isinstance(carried, dict) and carried:
            unclassified[s.source] = carried
        elif s.unclassified:
            unclassified[s.source] = s.unclassified
    current = sorted({kind for counts in unclassified.values() for kind in counts})
    if set(current) != previous:
        log.warning(
            "dashboard rollup: %d event kind(s) not interpreted: %s",
            len(current),
            ", ".join(current),
        )
    dst.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('unclassified_kinds', ?)",
        (json.dumps(unclassified, sort_keys=True),),
    )


def run_rollup(sources: RollupSources, now: datetime) -> RollupResult:
    """Ingest every registered events DB into ``dashboard.db``; never raises.

    ``now`` is injected (it only positions the trailing re-derive window). One source
    failing leaves the others ingested and the failure on that ``SourceResult``.
    """
    dst, db_rebuilt, err = _open_dashboard_db(sources.db_path)
    if dst is None:
        return RollupResult((), db_rebuilt, err, sources.registry_error)
    cutoff = _iso(now - WINDOW)
    try:
        plan = [(FLEET_SOURCE, sources.fleet_events_db), *sources.repos]
        results = tuple(_ingest_source(dst, s, db, cutoff) for s, db in plan)
        dst.execute("BEGIN IMMEDIATE")
        try:
            _record_unclassified(dst, results)
            dst.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('rolled_up_at', ?)",
                (_iso(now),),
            )
            dst.execute("COMMIT")
        except BaseException:
            dst.execute("ROLLBACK")
            raise
        return RollupResult(results, db_rebuilt, None, sources.registry_error)
    except sqlite3.Error as exc:
        return RollupResult((), db_rebuilt, f"{type(exc).__name__}: {exc}", sources.registry_error)
    finally:
        dst.close()
