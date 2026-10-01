"""dashboard.db schema (ADR-0008): a disposable, rebuildable cache of derived facts.

Bump ``SCHEMA_VERSION`` whenever a table or its derivation changes; ``rollup`` drops
and rebuilds the whole file on a mismatch, so no migration code exists or is needed.

Every per-event fact table is keyed ``(source, src_id, seq)``: ``source`` is the events
DB the row came from (a ``fleet.json`` repo key, or ``FLEET_SOURCE`` for the global DB),
``src_id`` the source event id, ``seq`` the row's index within that event. The key makes
re-deriving an event an idempotent ``INSERT OR REPLACE``.
"""

from __future__ import annotations

SCHEMA_VERSION = 1
FLEET_SOURCE = "fleet"

_KEY = (
    "source TEXT NOT NULL, src_id INTEGER NOT NULL, seq INTEGER NOT NULL,"
    " ts TEXT NOT NULL, repo TEXT NOT NULL"
)
_PK = "PRIMARY KEY (source, src_id, seq)"

# table -> column DDL (the shared key columns are prepended).
_FACT_TABLES: dict[str, str] = {
    "pass_samples": (
        "live_sessions INT, fleet_live_sessions INT, concurrency_limit INT,"
        " fleet_concurrency_limit INT, available_slots INT, dispatch_limit INT, clamped INT,"
        " deferred_by_concurrency INT, launched INT, open_total INT, dispatchable INT,"
        " active_label INT, missing_ready INT, terminal_label INT,"
        " blocked_by_open_dependency INT, operator_claimed INT"
    ),
    "review_samples": (
        "available_slots INT, live_reviews INT, review_limit INT, launched INT, failed INT,"
        " quota_hit INT"
    ),
    "issue_milestones": (
        "issue INT, pr INT, milestone TEXT NOT NULL, event_kind TEXT NOT NULL, approx INT NOT NULL"
    ),
    "worker_exits": "issue INT, failure_kind TEXT, worker_health TEXT",
    "escalations": "issue INT, pr INT, event_kind TEXT NOT NULL, reason TEXT",
    "verdict_missed": (
        "issue INT, pr INT, reason TEXT, reason_group TEXT, exit_code INT, turn_count INT,"
        " tool_call_count INT"
    ),
    "runner_samples": (
        "target_repo TEXT NOT NULL, capacity INT, demand INT, running INT, target INT,"
        " budget INT, oldest_queued_seconds REAL"
    ),
    "deploys": "ok INT NOT NULL, changed INT, from_sha TEXT, to_sha TEXT, error TEXT",
    "throttles": "event_kind TEXT NOT NULL, until TEXT, detail TEXT",
    "capped_demand": "event_kind TEXT NOT NULL, requested INT, granted INT, reason TEXT",
}

# Keyed by the CI job, not the event: a job is observed repeatedly while it runs, and only
# its completed observation is kept (first wins), so re-observation never double counts.
JOB_TABLE = "job_observations"
_JOB_DDL = (
    "job_id TEXT PRIMARY KEY, source TEXT NOT NULL, src_id INTEGER NOT NULL, ts TEXT NOT NULL,"
    " name TEXT, status TEXT, queue_wait_seconds REAL, execution_seconds REAL, wall_seconds REAL"
)

# Tables re-derived by deleting a source's rows at/after the window's first event id.
WINDOWED_TABLES = tuple(_FACT_TABLES)
# Tables wiped when one source is rebuilt from scratch.
SOURCE_SCOPED_TABLES = (*WINDOWED_TABLES, JOB_TABLE, "loop_passes", "coverage")

_OTHER_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS watermarks (source TEXT PRIMARY KEY, max_id INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS loop_passes (
    source TEXT NOT NULL, correlation_id TEXT NOT NULL, started_at TEXT, completed_at TEXT,
    ok INT, elapsed_seconds REAL, error_count INT, merge_count INT, review_count INT,
    sink_population INT, sink_arrivals INT, sink_clears INT,
    PRIMARY KEY (source, correlation_id));
CREATE TABLE IF NOT EXISTS coverage (
    source TEXT NOT NULL, kind TEXT NOT NULL, first_ts TEXT, last_ts TEXT, n INT NOT NULL,
    PRIMARY KEY (source, kind));
"""

LOOP_PASS_COLUMNS = (
    "correlation_id",
    "started_at",
    "completed_at",
    "ok",
    "elapsed_seconds",
    "error_count",
    "merge_count",
    "review_count",
    "sink_population",
    "sink_arrivals",
    "sink_clears",
)


def schema_sql() -> str:
    """Full DDL for a fresh dashboard.db."""
    parts = [_OTHER_DDL]
    for name, cols in _FACT_TABLES.items():
        parts.append(f"CREATE TABLE IF NOT EXISTS {name} ({_KEY}, {cols}, {_PK});")
    parts.append(f"CREATE TABLE IF NOT EXISTS {JOB_TABLE} ({_JOB_DDL});")
    return "\n".join(parts)
