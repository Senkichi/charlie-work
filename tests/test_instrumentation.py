"""Tests for the instrumentation module: SQLite event log, correlation IDs, dual-write."""

from __future__ import annotations

import ast
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from charlie_work.instrumentation import (
    EXPECTED_OPERATIONAL_KINDS,
    _LEVEL_BY_KIND,
    close_db,
    correlation_context,
    current_correlation_id,
    event_counts_by_kind,
    events_by_correlation_id,
    log_event,
    query_events,
    read_event_log,
    record_loop_pass,
)
from charlie_work.state import append_event, empty_state


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path) -> None:
    """Ensure DB connections are closed between tests to avoid cross-test contamination."""
    yield
    # Close any connections that were opened during this test
    close_db(tmp_path / "state.json")
    # Also try the variant paths used in some tests
    close_db(tmp_path / "subdir" / "state.json")
    close_db(tmp_path / "nonexistent_dir" / "state.json")


def test_log_event_writes_sqlite(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    log_event(state_path, "test_event", {"key": "value"}, repo="test-repo")

    events = read_event_log(state_path)
    assert len(events) == 1
    assert events[0]["kind"] == "test_event"
    assert events[0]["payload"] == {"key": "value"}
    assert events[0]["repo"] == "test-repo"
    assert "ts" in events[0]


def test_log_event_with_correlation_id(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    with correlation_context() as cid:
        log_event(state_path, "event_a", {"a": 1})
        log_event(state_path, "event_b", {"b": 2})

    events = read_event_log(state_path)
    assert len(events) == 2
    assert all(e["correlation_id"] == cid for e in events)


def test_correlation_context_nesting(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    with correlation_context("outer"):
        assert current_correlation_id() == "outer"
        with correlation_context("inner"):
            assert current_correlation_id() == "inner"
            log_event(state_path, "inner_event", {})
        assert current_correlation_id() == "outer"
        log_event(state_path, "outer_event", {})

    events = read_event_log(state_path)
    inner = [e for e in events if e["kind"] == "inner_event"]
    outer = [e for e in events if e["kind"] == "outer_event"]
    assert len(inner) == 1 and inner[0]["correlation_id"] == "inner"
    assert len(outer) == 1 and outer[0]["correlation_id"] == "outer"


def test_correlation_context_restores_previous() -> None:
    with correlation_context("first"):
        assert current_correlation_id() == "first"
        with correlation_context("second"):
            assert current_correlation_id() == "second"
        assert current_correlation_id() == "first"
    assert current_correlation_id() is None


def test_events_by_correlation_id(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    with correlation_context("pass-1"):
        log_event(state_path, "loop_started", {})
        log_event(state_path, "dispatch", {"issue": 42})
        log_event(state_path, "loop_completed", {})
    with correlation_context("pass-2"):
        log_event(state_path, "loop_started", {})
        log_event(state_path, "dispatch", {"issue": 43})

    pass1 = events_by_correlation_id(state_path, "pass-1")
    pass2 = events_by_correlation_id(state_path, "pass-2")
    assert len(pass1) == 3
    assert len(pass2) == 2
    assert all(e["correlation_id"] == "pass-1" for e in pass1)


def test_read_event_log_empty(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    assert read_event_log(state_path) == []


def test_read_event_log_limit(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    for i in range(10):
        log_event(state_path, f"event_{i}", {})
    events = read_event_log(state_path, limit=3)
    assert len(events) == 3
    assert events[0]["kind"] == "event_7"
    assert events[2]["kind"] == "event_9"


def test_append_event_dual_write(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = empty_state()

    state = append_event(
        state,
        "test_kind",
        {"data": 123},
        state_path=state_path,
        repo="my-repo",
    )

    # state.json events array should have the event
    assert len(state["events"]) == 1
    assert state["events"][0]["kind"] == "test_kind"

    # events.db should also have the event
    db_events = read_event_log(state_path)
    assert len(db_events) == 1
    assert db_events[0]["kind"] == "test_kind"
    assert db_events[0]["repo"] == "my-repo"


def test_append_event_without_state_path_no_db(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = empty_state()

    state = append_event(state, "test_kind", {"data": 123})

    assert len(state["events"]) == 1
    # No events.db should be created
    assert not (state_path.parent / "events.db").exists()


def test_append_event_200_cap_preserved(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = empty_state()

    for i in range(250):
        state = append_event(state, f"event_{i}", {}, max_size=200, state_path=state_path)

    # state.json events array capped at 200
    assert len(state["events"]) == 200

    # events.db has all 250
    db_events = read_event_log(state_path)
    assert len(db_events) == 250


def test_log_event_best_effort_no_crash(tmp_path: Path) -> None:
    state_path = tmp_path / "nonexistent_dir" / "state.json"
    # Should not raise even if directory creation fails
    log_event(state_path, "test", {})


def test_log_event_swallows_mkdir_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#746: _get_db's mkdir() failure must be caught like sqlite errors.

    If Path.mkdir raises an OSError (permissions, race, virtualized path),
    log_event must remain best-effort and return without escaping.
    """
    state_path = tmp_path / "blocked_dir" / "state.json"

    def _raising_mkdir(self, *args, **kwargs):
        raise PermissionError("simulated directory-creation failure")

    monkeypatch.setattr("pathlib.Path.mkdir", _raising_mkdir)

    # Must not raise; best-effort logging should swallow the OSError.
    log_event(state_path, "test_mkdir_blocked", {})


def test_pr_number_extraction(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    log_event(state_path, "dispatch", {"pr_number": 42, "issue_number": 7})
    log_event(state_path, "review", {"pr": 99})
    log_event(state_path, "intake", {"issue": 5})

    events = read_event_log(state_path)
    assert events[0]["pr_number"] == 42
    assert events[0]["issue_number"] == 7
    assert events[1]["pr_number"] == 99
    assert events[1]["issue_number"] is None
    assert events[2]["pr_number"] is None
    assert events[2]["issue_number"] == 5


def test_plural_payload_key_extraction(tmp_path: Path) -> None:
    """Issue #553: list-valued payload keys must populate pr_number/issue_number.

    dispatch/dispatch_rework carry ``issue_numbers`` (list), review_dispatch_claim
    carries ``pr_numbers`` (list), and review_dispatch carries ``launched``/``failed``
    (lists of PR ints). Without unwrapping, ~13% of events land with NULL indexed
    columns and are invisible to query_events/events_by_correlation_id filtering.
    The single-valued column is backfilled from the first numeric element.
    """
    state_path = tmp_path / "state.json"
    # dispatch: issue_numbers list
    log_event(state_path, "dispatch", {"issue_numbers": [101, 102, 103]})
    # dispatch_rework: issue_numbers list
    log_event(state_path, "dispatch_rework", {"issue_numbers": [200], "failed_issue_numbers": []})
    # review_dispatch_claim: pr_numbers list
    log_event(state_path, "review_dispatch_claim", {"pr_numbers": [55, 56], "count": 2})
    # review_dispatch: launched list (preferred over failed)
    log_event(
        state_path,
        "review_dispatch",
        {"launched": [77, 78], "failed": [79], "quota_hit": False},
    )
    # review_dispatch with only failures
    log_event(
        state_path,
        "review_dispatch",
        {"launched": [], "failed": [88], "quota_hit": False},
    )

    events = read_event_log(state_path)
    assert events[0]["issue_number"] == 101
    assert events[0]["pr_number"] is None
    assert events[1]["issue_number"] == 200
    assert events[1]["pr_number"] is None
    assert events[2]["pr_number"] == 55
    assert events[2]["issue_number"] is None
    assert events[3]["pr_number"] == 77
    assert events[3]["issue_number"] is None
    assert events[4]["pr_number"] == 88
    assert events[4]["issue_number"] is None


def test_plural_extraction_skips_non_numeric_lists(tmp_path: Path) -> None:
    """Plural keys whose elements are dicts/objects must not produce false refs.

    Some payloads use ``issues``/``prs``/``failed`` as lists of summary dicts
    (e.g. CommandResult data). Only numeric elements are indexed; dict-shaped
    lists leave the column NULL rather than guessing.
    """
    state_path = tmp_path / "state.json"
    log_event(
        state_path,
        "intake",
        {"issues": [{"number": 1, "title": "x"}], "prs": [{"number": 2}]},
    )
    log_event(state_path, "review_dispatch", {"failed": [{"pr": 9, "error": "boom"}]})

    events = read_event_log(state_path)
    assert events[0]["pr_number"] is None
    assert events[0]["issue_number"] is None
    assert events[1]["pr_number"] is None
    assert events[1]["issue_number"] is None


def test_singular_key_preferred_over_plural(tmp_path: Path) -> None:
    """An explicit singular ref must win over a plural list."""
    state_path = tmp_path / "state.json"
    log_event(
        state_path,
        "dispatch",
        {"pr_number": 42, "issue_numbers": [101, 102]},
    )

    events = read_event_log(state_path)
    assert events[0]["pr_number"] == 42
    assert events[0]["issue_number"] == 101


def test_dispatch_rework_payload_pr_number_indexes(tmp_path: Path) -> None:
    """Issue #770: a ``pr_number`` key in a ``dispatch_rework`` payload must populate the indexed column.

    This is the schema-level guard: the caller supplies ``pr_number`` and the
    instrumentation layer copies it to the ``pr_number`` SQLite column. Without
    this, ``query_events(pr_number=...)`` silently returns empty for the kind.
    """
    state_path = tmp_path / "state.json"
    log_event(
        state_path,
        "dispatch_rework",
        {
            "pr_number": 456,
            "issue_numbers": [123],
            "failed_issue_numbers": [],
        },
    )

    events = read_event_log(state_path)
    assert events[0]["pr_number"] == 456
    assert events[0]["payload"]["pr_number"] == 456


def test_plural_extraction_query_events_filter(tmp_path: Path) -> None:
    """Issue #553 core symptom: events with plural refs must be findable by PR/issue."""
    state_path = tmp_path / "state.json"
    log_event(state_path, "dispatch", {"issue_numbers": [101, 102]})

    results = query_events(state_path, issue_number=101)
    assert len(results) == 1
    assert results[0]["kind"] == "dispatch"


def test_level_classification(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    log_event(state_path, "dispatch", {})
    log_event(state_path, "github_error", {})
    log_event(state_path, "dispatch_skip_blocked", {})
    log_event(state_path, "loop_started", {})

    events = read_event_log(state_path)
    levels = {e["kind"]: e["level"] for e in events}
    assert levels["dispatch"] == "info"
    assert levels["github_error"] == "error"
    assert levels["dispatch_skip_blocked"] == "warning"
    assert levels["loop_started"] == "info"


def test_fleet_pass_config_error_classified_and_queryable_by_level(tmp_path: Path) -> None:
    """#6-G: a lane-startup failure is classified as an error and reachable
    through query_events(level="error") without any new query infrastructure
    (the plan's explicit constraint for the events.db side of the fix)."""
    state_path = tmp_path / "state.json"
    log_event(
        state_path,
        "fleet_pass_config_error",
        {
            "repo_key": "owner/repo",
            "error": "ConfigError: unknown key(s) in config section 'cross_family': auto_verdict",
        },
        repo="owner/repo",
    )

    events = read_event_log(state_path)
    assert events[0]["level"] == "error"

    by_level = query_events(state_path, level="error")
    assert len(by_level) == 1
    assert by_level[0]["kind"] == "fleet_pass_config_error"
    assert by_level[0]["payload"]["repo_key"] == "owner/repo"
    assert by_level[0]["repo"] == "owner/repo"


def test_dispatch_blocked_chain_dead_classified_and_queryable_by_level(
    tmp_path: Path,
) -> None:
    """#829: a permanently dead blocker chain must not sit at info level.

    The event is emitted from ``Orchestrator.dispatch()`` when every open
    blocker of a blocked issue is itself dead (escalated, or its tracked PR
    is escalated/janitor_blocked). It makes no GitHub label change, so the
    ``level`` column is its only consumer surface.
    """
    state_path = tmp_path / "state.json"
    log_event(
        state_path,
        "dispatch_blocked_chain_dead",
        {"issue": 829, "chain_root": [123, 456]},
        repo="owner/repo",
    )

    events = read_event_log(state_path)
    assert events[0]["level"] == "error"

    by_level = query_events(state_path, level="error")
    assert len(by_level) == 1
    assert by_level[0]["kind"] == "dispatch_blocked_chain_dead"
    assert by_level[0]["payload"]["issue"] == 829
    assert by_level[0]["repo"] == "owner/repo"


def test_query_events_by_kind(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    log_event(state_path, "dispatch", {"issue": 1})
    log_event(state_path, "review", {"pr_number": 2})
    log_event(state_path, "dispatch", {"issue": 3})

    results = query_events(state_path, kind="dispatch")
    assert len(results) == 2
    assert all(e["kind"] == "dispatch" for e in results)


def test_query_events_by_pr_number(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    log_event(state_path, "review", {"pr_number": 42})
    log_event(state_path, "merge_ready", {"pr_number": 42})
    log_event(state_path, "dispatch", {"pr_number": 99})

    results = query_events(state_path, pr_number=42)
    assert len(results) == 2
    assert all(e["pr_number"] == 42 for e in results)


def test_query_events_by_level(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    log_event(state_path, "dispatch", {})
    log_event(state_path, "github_error", {})
    log_event(state_path, "session_stalled", {})
    log_event(state_path, "loop_started", {})

    errors = query_events(state_path, level="error")
    assert len(errors) == 2
    assert all(e["level"] == "error" for e in errors)


def test_session_exited_is_warning_while_session_stalled_stays_error(tmp_path: Path) -> None:
    """Issue #873: the two worker-reap outcomes must not share a level.

    ``session_stalled`` (WorkerHealth.STALLED — a live process that stopped
    making progress) is a genuine fault and stays error-level.
    ``session_exited`` (WorkerHealth.DEAD — the process is already gone) is
    also the normal terminal state of every worker that finished and exited,
    so it must not land in the error stream that #864/#866 consume.

    Warning, not info, is deliberate: liveness alone does not distinguish a
    clean exit from a crash, so the reap stays surfaced — it just stops being
    reported as a fault.
    """
    state_path = tmp_path / "state.json"
    log_event(state_path, "session_stalled", {"worker_health": "STALLED"})
    log_event(state_path, "session_exited", {"worker_health": "DEAD"})

    errors = query_events(state_path, level="error")
    assert [e["kind"] for e in errors] == ["session_stalled"]

    warnings = query_events(state_path, level="warning")
    assert [e["kind"] for e in warnings] == ["session_exited"]


def test_query_events_by_correlation_id(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    with correlation_context("abc123"):
        log_event(state_path, "loop_started", {})
        log_event(state_path, "dispatch", {})
    with correlation_context("def456"):
        log_event(state_path, "loop_started", {})

    results = query_events(state_path, correlation_id="abc123")
    assert len(results) == 2
    assert all(e["correlation_id"] == "abc123" for e in results)


def test_query_events_with_limit(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    for i in range(10):
        log_event(state_path, f"event_{i}", {})

    results = query_events(state_path, limit=3)
    assert len(results) == 3
    assert results[0]["kind"] == "event_7"
    assert results[2]["kind"] == "event_9"


def test_query_events_multiple_filters(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    with correlation_context("pass-1"):
        log_event(state_path, "github_error", {"pr_number": 42})
        log_event(state_path, "dispatch", {"pr_number": 42})
    with correlation_context("pass-2"):
        log_event(state_path, "github_error", {"pr_number": 42})

    results = query_events(state_path, kind="github_error", correlation_id="pass-1", pr_number=42)
    assert len(results) == 1
    assert results[0]["kind"] == "github_error"
    assert results[0]["correlation_id"] == "pass-1"
    assert results[0]["pr_number"] == 42


def test_query_events_since_until(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    # Insert an event with a known past timestamp
    from charlie_work.instrumentation import _get_db

    conn = _get_db(state_path)
    assert conn is not None
    conn.execute(
        """INSERT INTO events (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
           VALUES (?, ?, ?, NULL, NULL, NULL, NULL, 'info')""",
        ("2025-06-01T12:00:00Z", "mid_event", json.dumps({"x": 1})),
    )
    # Insert an event with a known future timestamp
    conn.execute(
        """INSERT INTO events (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
           VALUES (?, ?, ?, NULL, NULL, NULL, NULL, 'info')""",
        ("2099-12-31T23:59:59Z", "future_event", json.dumps({"x": 2})),
    )
    log_event(state_path, "current_event", {})

    # since filter: should include mid and current but not future (current is ~2026)
    results = query_events(state_path, since="2025-01-01T00:00:00Z", until="2099-01-01T00:00:00Z")
    kinds = [e["kind"] for e in results]
    assert "mid_event" in kinds
    assert "current_event" in kinds
    assert "future_event" not in kinds

    # until filter: should include only mid_event
    results_until = query_events(state_path, until="2025-06-01T23:59:59Z")
    kinds_until = [e["kind"] for e in results_until]
    assert "mid_event" in kinds_until
    assert "current_event" not in kinds_until
    assert "future_event" not in kinds_until


def test_event_counts_by_kind(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    log_event(state_path, "dispatch", {})
    log_event(state_path, "dispatch", {})
    log_event(state_path, "review", {})
    log_event(state_path, "merge_ready", {})

    counts = event_counts_by_kind(state_path)
    assert counts["dispatch"] == 2
    assert counts["review"] == 1
    assert counts["merge_ready"] == 1


def test_record_loop_pass(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    record_loop_pass(state_path, "cid-1", "2025-01-01T00:00:00Z")
    record_loop_pass(
        state_path,
        "cid-1",
        "2025-01-01T00:00:00Z",
        completed_at="2025-01-01T00:05:00Z",
        ok=True,
        elapsed_seconds=300.0,
        error_count=0,
        merge_count=2,
        review_count=3,
        sink_population=23,
        sink_arrivals=5,
        sink_clears=2,
    )

    from charlie_work.instrumentation import _get_db

    conn = _get_db(state_path)
    assert conn is not None
    cursor = conn.execute("SELECT * FROM loop_passes WHERE correlation_id = ?", ("cid-1",))
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "cid-1"
    assert row[1] == "2025-01-01T00:00:00Z"
    assert row[2] == "2025-01-01T00:05:00Z"
    assert row[3] == 1  # ok
    assert row[4] == 300.0
    assert row[5] == 0  # error_count
    assert row[6] == 2  # merge_count
    assert row[7] == 3  # review_count
    # Issue #1083: sink-metric columns appended at the end.
    assert row[8] == 23  # sink_population
    assert row[9] == 5  # sink_arrivals
    assert row[10] == 2  # sink_clears


def test_loop_passes_sink_columns_migrated_on_old_db(tmp_path: Path) -> None:
    """Issue #1083: a pre-existing events.db gains the sink-metric columns.

    A database created before #1083 has the 8-column ``loop_passes`` schema
    and ``user_version = 1``. Opening it through ``_get_db`` must ALTER it
    forward to the 11-column schema and bump ``user_version`` to 2, without
    losing the pre-existing row. Re-opening must not re-run the migration.
    """
    import sqlite3

    from charlie_work.instrumentation import _get_db, close_db

    state_path = tmp_path / "state.json"
    db_path = state_path.parent / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a pre-#1083 database: old loop_passes schema, user_version=1,
    # with one already-recorded pass.
    pre = sqlite3.connect(str(db_path))
    pre.executescript(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
            repo TEXT, correlation_id TEXT, pr_number INTEGER,
            issue_number INTEGER, level TEXT DEFAULT 'info'
        );
        CREATE TABLE loop_passes (
            correlation_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL, completed_at TEXT,
            ok INTEGER, elapsed_seconds REAL,
            error_count INTEGER DEFAULT 0,
            merge_count INTEGER DEFAULT 0,
            review_count INTEGER DEFAULT 0
        );
        """
    )
    pre.execute(
        """INSERT INTO loop_passes
           (correlation_id, started_at, completed_at, ok, elapsed_seconds,
            error_count, merge_count, review_count)
           VALUES (?, ?, ?, 1, 12.0, 0, 1, 0)""",
        ("cid-old", "2025-01-01T00:00:00Z", "2025-01-01T00:00:12Z"),
    )
    pre.execute("PRAGMA user_version = 1")
    pre.commit()
    pre.close()

    # First access triggers the v2 migration.
    conn = _get_db(state_path)
    assert conn is not None
    cols = {row[1] for row in conn.execute("PRAGMA table_info(loop_passes)")}
    assert "sink_population" in cols
    assert "sink_arrivals" in cols
    assert "sink_clears" in cols
    # The pre-existing row is preserved and the new columns default to 0.
    row = conn.execute(
        "SELECT * FROM loop_passes WHERE correlation_id = ?", ("cid-old",)
    ).fetchone()
    assert row is not None
    assert row["merge_count"] == 1
    assert row["sink_population"] == 0
    assert row["sink_arrivals"] == 0
    assert row["sink_clears"] == 0
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == 2

    # Re-opening must not re-run the migration (user_version guard).
    close_db(state_path)
    conn2 = _get_db(state_path)
    assert conn2 is not None
    assert conn2.execute("PRAGMA user_version").fetchone()[0] == 2


def test_jsonl_migration(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    jsonl_path = state_path.parent / "events.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Write some legacy JSONL entries
    records = [
        {
            "ts": "2025-01-01T00:00:00Z",
            "kind": "dispatch",
            "payload": {"issue": 1},
            "repo": "test",
        },
        {
            "ts": "2025-01-01T00:01:00Z",
            "kind": "review",
            "payload": {"pr_number": 42},
            "correlation_id": "abc",
        },
        {"ts": "2025-01-01T00:02:00Z", "kind": "loop_completed", "payload": {"ok": True}},
    ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # First access triggers migration
    events = read_event_log(state_path)
    assert len(events) == 3
    assert events[0]["kind"] == "dispatch"
    assert events[0]["repo"] == "test"
    assert events[1]["kind"] == "review"
    assert events[1]["correlation_id"] == "abc"
    assert events[1]["pr_number"] == 42
    assert events[2]["kind"] == "loop_completed"

    # New events should be appended to the DB
    log_event(state_path, "new_event", {})
    events = read_event_log(state_path)
    assert len(events) == 4
    assert events[3]["kind"] == "new_event"


def test_jsonl_migration_skips_malformed(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    jsonl_path = state_path.parent / "events.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"ts": "2025-01-01T00:00:00Z", "kind": "good", "payload": {}}) + "\n")
        f.write("not valid json\n")
        f.write(
            json.dumps({"ts": "2025-01-01T00:01:00Z", "kind": "also_good", "payload": {}}) + "\n"
        )

    events = read_event_log(state_path)
    assert len(events) == 2
    assert events[0]["kind"] == "good"
    assert events[1]["kind"] == "also_good"


def test_sqlite_db_file_created(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    log_event(state_path, "test", {})
    assert (state_path.parent / "events.db").exists()


def test_wal_mode_enabled(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    log_event(state_path, "test", {})

    from charlie_work.instrumentation import _get_db

    conn = _get_db(state_path)
    assert conn is not None
    cursor = conn.execute("PRAGMA journal_mode")
    mode = cursor.fetchone()[0]
    assert mode == "wal"


# ---------------------------------------------------------------------------
# Regression tests for issue #557: events.jsonl re-migrates on every start
# ---------------------------------------------------------------------------


def _write_jsonl(jsonl_path: Path, records: list[dict]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_jsonl_migration_one_shot_across_processes(tmp_path: Path) -> None:
    """Two fresh connections (simulating two processes) must not duplicate rows."""
    state_path = tmp_path / "state.json"
    jsonl_path = state_path.parent / "events.jsonl"
    migrated_path = state_path.parent / "events.jsonl.migrated"
    records = [
        {"ts": "2025-01-01T00:00:00Z", "kind": "dispatch", "payload": {"issue": 1}},
        {"ts": "2025-01-01T00:01:00Z", "kind": "review", "payload": {"pr_number": 42}},
        {"ts": "2025-01-01T00:02:00Z", "kind": "loop_completed", "payload": {"ok": True}},
    ]
    _write_jsonl(jsonl_path, records)

    # First "process" — triggers migration and renames the file.
    events = read_event_log(state_path)
    assert len(events) == 3
    assert not jsonl_path.exists()
    assert migrated_path.exists()

    # Second "process" — close the cached connection to simulate a new process.
    close_db(state_path)
    events2 = read_event_log(state_path)
    assert len(events2) == 3

    # Third "process" — still no duplicates.
    close_db(state_path)
    events3 = read_event_log(state_path)
    assert len(events3) == 3


def test_jsonl_migration_idempotent_if_rerun(tmp_path: Path) -> None:
    """If migration runs again (crash before rename), no rows are duplicated."""
    state_path = tmp_path / "state.json"
    jsonl_path = state_path.parent / "events.jsonl"
    migrated_path = state_path.parent / "events.jsonl.migrated"
    records = [
        {"ts": "2025-01-01T00:00:00Z", "kind": "dispatch", "payload": {"issue": 1}},
        {"ts": "2025-01-01T00:01:00Z", "kind": "review", "payload": {"pr": 42}},
    ]
    _write_jsonl(jsonl_path, records)

    # First migration.
    events = read_event_log(state_path)
    assert len(events) == 2
    assert migrated_path.exists()

    # Simulate a crash-before-rename by restoring the legacy file.
    close_db(state_path)
    migrated_path.replace(jsonl_path)
    assert jsonl_path.exists()

    # Second migration must not duplicate any rows.
    events2 = read_event_log(state_path)
    assert len(events2) == 2
    # And the file is renamed again.
    assert not jsonl_path.exists()
    assert migrated_path.exists()


def test_dedupe_existing_duplicates_on_first_access(tmp_path: Path) -> None:
    """One-time cleanup removes duplicate rows from prior pollution."""
    state_path = tmp_path / "state.json"
    db_path = state_path.parent / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a polluted database directly: 4 identical rows + 1 unique row.
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
            repo TEXT, correlation_id TEXT, pr_number INTEGER,
            issue_number INTEGER, level TEXT DEFAULT 'info'
        );
        """
    )
    payload_json = json.dumps({"issue": 1}, sort_keys=True)
    for _ in range(4):
        conn.execute(
            """INSERT INTO events
               (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
               VALUES (?, ?, ?, NULL, NULL, NULL, NULL, 'info')""",
            ("2025-01-01T00:00:00Z", "dispatch", payload_json),
        )
    conn.execute(
        """INSERT INTO events
           (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
           VALUES (?, ?, ?, NULL, NULL, NULL, NULL, 'info')""",
        ("2025-01-01T00:01:00Z", "review", json.dumps({"pr": 2}, sort_keys=True)),
    )
    conn.commit()
    conn.close()

    # First access triggers the user_version=1 dedupe migration.
    events = read_event_log(state_path)
    assert len(events) == 2
    kinds = [e["kind"] for e in events]
    assert kinds == ["dispatch", "review"]

    # Re-opening must not re-run the dedupe (user_version guard).
    close_db(state_path)
    events2 = read_event_log(state_path)
    assert len(events2) == 2


def test_dedupe_preserves_distinct_events_different_repo(tmp_path: Path) -> None:
    """Distinct events that share (ts, kind, payload) but differ in repo must survive.

    Regression for the review finding that the original dedupe key omitted
    ``repo``.  ``_now_iso()`` truncates to 1-second precision, so two repos
    logging the same kind in the same second is a realistic collision —
    collapsing them would silently delete audit history.
    """
    state_path = tmp_path / "state.json"
    db_path = state_path.parent / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
            repo TEXT, correlation_id TEXT, pr_number INTEGER,
            issue_number INTEGER, level TEXT DEFAULT 'info'
        );
        """
    )
    payload_json = json.dumps({}, sort_keys=True)
    # Two rows identical except for ``repo`` — must both survive dedupe.
    conn.execute(
        """INSERT INTO events
           (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
           VALUES (?, ?, ?, ?, NULL, NULL, NULL, 'info')""",
        ("2025-01-01T00:00:00Z", "loop_started", payload_json, "repo-a"),
    )
    conn.execute(
        """INSERT INTO events
           (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
           VALUES (?, ?, ?, ?, NULL, NULL, NULL, 'info')""",
        ("2025-01-01T00:00:00Z", "loop_started", payload_json, "repo-b"),
    )
    # Plus a true duplicate of the repo-a row (simulating old pollution).
    conn.execute(
        """INSERT INTO events
           (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
           VALUES (?, ?, ?, ?, NULL, NULL, NULL, 'info')""",
        ("2025-01-01T00:00:00Z", "loop_started", payload_json, "repo-a"),
    )
    conn.commit()
    conn.close()

    events = read_event_log(state_path)
    # 3 inserted, 1 true duplicate removed → 2 survive (repo-a, repo-b).
    assert len(events) == 2
    repos = sorted(e["repo"] for e in events)
    assert repos == ["repo-a", "repo-b"]


def test_dedupe_preserves_distinct_events_different_correlation_id(
    tmp_path: Path,
) -> None:
    """Distinct events that share (ts, kind, payload, repo) but differ in
    correlation_id must survive — two loop passes starting in the same second
    for the same repo is realistic given 1-second timestamp truncation."""
    state_path = tmp_path / "state.json"
    db_path = state_path.parent / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
            repo TEXT, correlation_id TEXT, pr_number INTEGER,
            issue_number INTEGER, level TEXT DEFAULT 'info'
        );
        """
    )
    payload_json = json.dumps({}, sort_keys=True)
    for cid in ("pass-aaa", "pass-bbb"):
        conn.execute(
            """INSERT INTO events
               (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
               VALUES (?, ?, ?, ?, ?, NULL, NULL, 'info')""",
            ("2025-01-01T00:00:00Z", "loop_started", payload_json, "same-repo", cid),
        )
    conn.commit()
    conn.close()

    events = read_event_log(state_path)
    assert len(events) == 2
    cids = sorted(e["correlation_id"] for e in events)
    assert cids == ["pass-aaa", "pass-bbb"]


def test_jsonl_migration_preserves_distinct_events_different_repo(
    tmp_path: Path,
) -> None:
    """Two JSONL records sharing (ts, kind, payload) but differing in repo
    must both be migrated — the old ``(ts, kind, payload)`` idempotency key
    would have silently dropped the second."""
    state_path = tmp_path / "state.json"
    jsonl_path = state_path.parent / "events.jsonl"
    records = [
        {
            "ts": "2025-01-01T00:00:00Z",
            "kind": "loop_started",
            "payload": {},
            "repo": "repo-a",
        },
        {
            "ts": "2025-01-01T00:00:00Z",
            "kind": "loop_started",
            "payload": {},
            "repo": "repo-b",
        },
    ]
    _write_jsonl(jsonl_path, records)

    events = read_event_log(state_path)
    assert len(events) == 2
    repos = sorted(e["repo"] for e in events)
    assert repos == ["repo-a", "repo-b"]


def test_jsonl_migration_preserves_distinct_events_different_correlation_id(
    tmp_path: Path,
) -> None:
    """Two JSONL records sharing (ts, kind, payload, repo) but differing in
    correlation_id must both be migrated."""
    state_path = tmp_path / "state.json"
    jsonl_path = state_path.parent / "events.jsonl"
    records = [
        {
            "ts": "2025-01-01T00:00:00Z",
            "kind": "loop_started",
            "payload": {},
            "repo": "same-repo",
            "correlation_id": "pass-aaa",
        },
        {
            "ts": "2025-01-01T00:00:00Z",
            "kind": "loop_started",
            "payload": {},
            "repo": "same-repo",
            "correlation_id": "pass-bbb",
        },
    ]
    _write_jsonl(jsonl_path, records)

    events = read_event_log(state_path)
    assert len(events) == 2
    cids = sorted(e["correlation_id"] for e in events)
    assert cids == ["pass-aaa", "pass-bbb"]


def test_jsonl_migration_malformed_tolerance_preserved(tmp_path: Path) -> None:
    """Malformed-line tolerance must be preserved alongside the new idempotency."""
    state_path = tmp_path / "state.json"
    jsonl_path = state_path.parent / "events.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"ts": "2025-01-01T00:00:00Z", "kind": "good", "payload": {}}) + "\n")
        f.write("not valid json\n")
        f.write(
            json.dumps({"ts": "2025-01-01T00:01:00Z", "kind": "also_good", "payload": {}}) + "\n"
        )

    events = read_event_log(state_path)
    assert len(events) == 2
    assert events[0]["kind"] == "good"
    assert events[1]["kind"] == "also_good"
    # File renamed despite the malformed line.
    assert not jsonl_path.exists()
    assert (state_path.parent / "events.jsonl.migrated").exists()


# ---------------------------------------------------------------------------
# Issue #910 / #995: event-level registry must cover all in-repo emit sites
#
# #910 added ``_LEVEL_BY_KIND`` and this guard as its enforcement point: an
# emit site's ``kind`` should either be a registry member, or be provably one
# of a small number of literal values that are.
#
# #995: the original scanner (now superseded) understood exactly two shapes
# -- a bare string constant and a ternary between two string constants -- and
# treated everything else as contributing *nothing*. That makes an emit site
# whose kind is a bare variable, an f-string, or a function call indistin-
# guishable from a site that legitimately had no kind to check: the guard
# matches the shapes it recognises and fails *open* on every other shape.
#
# The replacement below inverts that. ``_resolve_literal`` still reduces an
# expression to a finite set of literal strings where it can (constants,
# ternaries, f-strings built entirely from resolved parts, module-level
# constants, and local variable assignments -- including multi-branch
# if/elif chains, which generalises the old single-case ``event_kind``
# special-case) -- but anything it cannot reduce is recorded as *unresolved*
# instead of silently dropped. ``test_event_kind_registry_exhaustive`` fails
# the build on any unresolved site absent from ``_ALLOWED_UNRESOLVED_KIND_SITES``
# (a small, named, reasoned allow-list), and, symmetrically, on any allow-list
# entry that no longer matches a real unresolved site -- so a site that later
# becomes resolvable (or is rewritten) can't leave stale cover behind.
# ---------------------------------------------------------------------------

_EMIT_FUNCS = {"log_event", "append_event", "_record_event", "record_event"}
_WRAPPER_FUNCS = {"_route_to_rework"}
_VALID_LEVELS = {"info", "warning", "error"}


def _is_scope_boundary(node: ast.AST) -> bool:
    """True for nodes that start a new variable-assignment scope."""
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))


def _collect_local_assignments(scope: ast.AST) -> dict[str, list[ast.expr]]:
    """Every ``name = <expr>`` assigned directly within ``scope``.

    Descends through control flow (if/for/while/try/with) but stops at any
    nested function, lambda, or class body -- those are separate scopes with
    their own assignment map, not this one.
    """
    assigns: dict[str, list[ast.expr]] = {}

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if _is_scope_boundary(child):
                continue
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        assigns.setdefault(target.id, []).append(child.value)
            walk(child)

    walk(scope)
    return assigns


def _collect_param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    """Every name bound as a parameter of ``node`` (positional, keyword-only, *args, **kwargs)."""
    args = node.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return frozenset(names)


def _resolve_literal(
    node: ast.expr,
    local_assigns: dict[str, list[ast.expr]],
    module_constants: dict[str, set[str]],
    local_params: frozenset[str] = frozenset(),
) -> set[str] | None:
    """Best-effort resolution of ``node`` to the finite set of strings it can be.

    Returns ``None`` when ``node`` cannot be proven to reduce to a literal set
    of strings -- including when it provably reduces to something that is
    *not* a string (e.g. a bare ``None`` constant), or when any branch of a
    multi-branch expression (an ``IfExp``, or multiple assignments to the same
    local name) is itself unresolvable. Callers MUST check ``is None``, never
    falsiness: ``None`` and ``set()`` are different signals, and conflating
    "unresolvable" with "resolved to nothing" is the #995/#1029 bug shape --
    #995 for ``kind``, #1029 for ``level`` (a ``level="x" if cond else None``
    site unioned the ``None`` branch away and was wrongly treated as
    self-classifying). Every exit point is also guaranteed to never return a
    resolved-but-empty set: ``set().issubset(_VALID_LEVELS)`` is ``True``, so
    an empty set handed to an ``is not None`` consumer would silently
    re-open the same fail-open bug one level up. This is enforced here (see
    the ``or None`` coercions below) rather than left as an invariant callers
    must trust -- a future branch added to this function only needs to avoid
    returning ``set()`` on its own exit, not reason about every consumer.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return {node.value}
        # Any other constant (None, an int, a bool, ...) is provably not a
        # string literal -- not merely unrecognised syntax. This is what lets
        # `level="warning" if cond else None` fail closed: the `None` branch
        # resolves to `None` (unresolvable) instead of silently contributing
        # the empty set that a union would then discard.
        return None
    if isinstance(node, ast.IfExp):
        body = _resolve_literal(node.body, local_assigns, module_constants, local_params)
        orelse = _resolve_literal(node.orelse, local_assigns, module_constants, local_params)
        if body is None or orelse is None:
            return None
        return body | orelse
    if isinstance(node, ast.JoinedStr):
        # An f-string resolves only if every interpolated part resolves (no
        # format spec, no conversion beyond the str-identity ``!s``); the
        # result is the cross product of the static and resolved parts, e.g.
        # ``f"{kind}_sweep"`` with kind in {"a", "b"} resolves to
        # {"a_sweep", "b_sweep"}. This subsumes the old bespoke ``_sweep``
        # special-case in ``_known_level`` below with a general mechanism.
        parts: list[set[str]] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append({value.value})
                continue
            if (
                isinstance(value, ast.FormattedValue)
                and value.format_spec is None
                and value.conversion in (-1, ord("s"))
            ):
                resolved = _resolve_literal(
                    value.value, local_assigns, module_constants, local_params
                )
                if resolved is not None:
                    parts.append(resolved)
                    continue
            return None
        combined = {""}
        for part in parts:
            combined = {prefix + suffix for prefix in combined for suffix in part}
        # Every `parts` entry is proven non-empty above (Constant str is a
        # singleton; a FormattedValue only appends when `resolved is not
        # None`, and _resolve_literal never itself returns an empty set --
        # see the `or None` guards below), so `combined` cannot legitimately
        # come out empty. Coerce defensively anyway: `_resolve_literal` must
        # never hand a resolved-but-empty set to a caller that branches on
        # `is not None`, since `set().issubset(_VALID_LEVELS)` is True and an
        # empty set would silently re-open the #1029 fail-open one level up.
        return combined or None
    if isinstance(node, ast.Name):
        if node.id in local_params:
            # A function parameter is caller-controlled. A local reassignment
            # of the same name inside the body doesn't prove every path
            # reaches the emit call *after* that reassignment -- the
            # un-reassigned (or not-yet-reassigned) parameter value could
            # still be the one that flows through on some path. Treat
            # conservatively as unresolved rather than trusting a partial
            # local reassignment over the parameter's own (unknown) value.
            return None
        if node.id in local_assigns:
            values: set[str] = set()
            for value_node in local_assigns[node.id]:
                resolved = _resolve_literal(
                    value_node, local_assigns, module_constants, local_params
                )
                if resolved is None:
                    # One reassignment on this name is unresolvable -- some
                    # path through the function could carry that value to the
                    # emit call, so the name as a whole is unresolvable too,
                    # same as an IfExp branch that doesn't resolve.
                    return None
                values |= resolved
            # Same empty-set guard as the JoinedStr exit above: `values`
            # should be non-empty by construction (every resolved branch is
            # itself non-empty, and `local_assigns[node.id]` is never an
            # empty list -- `_collect_local_assignments` only creates the key
            # when appending a value), but the guarantee must live at this
            # producer boundary, not be assumed by every consumer.
            return values or None
        if node.id in module_constants:
            # Third empty-set guard, for the same reason as the two above.
            # `_scan_tree` only inserts a key when `parts` is non-empty and
            # every part came back non-empty, so this cannot be empty today --
            # but that reasoning lives in a *different* function, and the
            # docstring above promises the invariant is enforced at every exit
            # of *this* one. Enforce it here rather than leaving the promise
            # true only by inspection of a caller.
            return module_constants[node.id] or None
        return None
    return None


@dataclass(frozen=True)
class _UnresolvedKindSite:
    """An emit-site ``kind`` expression the scanner could not reduce to literals."""

    path: str  # POSIX-relative to the scanned root
    scope: str  # enclosing function/method name, or "<module>"
    source: str  # ast.unparse() of the expression -- stable across line drift
    lineno: int = 0  # human-readable only; deliberately excluded from ``key``
    reason: str = ""  # human-readable only; deliberately excluded from ``key``

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.path, self.scope, self.source)


def _emit_func_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name) and node.func.id in _EMIT_FUNCS | _WRAPPER_FUNCS:
        return node.func.id
    if isinstance(node.func, ast.Attribute) and node.func.attr in _EMIT_FUNCS | _WRAPPER_FUNCS:
        return node.func.attr
    return None


def _has_explicit_level(
    node: ast.Call,
    local_assigns: dict[str, list[ast.expr]],
    module_constants: dict[str, set[str]],
    local_params: frozenset[str],
) -> bool:
    """True if ``node`` passes a ``level=`` keyword that resolves to a valid level.

    ``log_event`` lets a call site declare its level explicitly instead of
    relying on the registry (``instrumentation.log_event``'s ``level``
    parameter). Such a site is legitimately self-classifying, so its
    (possibly unresolvable) ``kind`` need not be registered or allow-listed.
    """
    for kw in node.keywords:
        if kw.arg != "level":
            continue
        resolved = _resolve_literal(kw.value, local_assigns, module_constants, local_params)
        if resolved is not None and resolved.issubset(_VALID_LEVELS):
            return True
    return False


def _locate_arg(node: ast.Call, position: int, keyword: str) -> ast.expr | None:
    """Find an argument by position or by keyword name, whichever the call used.

    A call that supplies the kind (or ``event_kind``) as a keyword argument
    -- ``log_event(state_path, kind=k, payload=p)`` -- must not silently fall
    through just because it doesn't match the common positional shape. That
    is the same fail-open pattern #995 was filed against, one layer down: a
    scanner that only recognizes the arity/shape it expects and drops
    everything else. No call site in this package currently uses ``*args``
    unpacking into these functions, so positional lookup by index is sound;
    if that ever changes, the returned ``None`` correctly routes the site to
    "unresolved" rather than silently skipping it.
    """
    if len(node.args) > position:
        return node.args[position]
    for kw in node.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _record_kind_site(
    kind_node: ast.expr | None,
    call_node: ast.Call,
    module_constants: dict[str, set[str]],
    local_assigns: dict[str, list[ast.expr]],
    local_params: frozenset[str],
    scope_name: str,
    rel_path: str,
    used: set[str],
    unresolved: list[_UnresolvedKindSite],
) -> None:
    if kind_node is None:
        # The call matched an emit function name but no kind/event_kind
        # argument could be located by position or keyword. Rather than
        # assume this isn't really one of our functions (the #995 failure
        # mode), record it as unresolved so it surfaces for review.
        unresolved.append(
            _UnresolvedKindSite(
                path=rel_path,
                scope=scope_name,
                source="<no kind argument located>",
                lineno=call_node.lineno,
            )
        )
        return
    resolved = _resolve_literal(kind_node, local_assigns, module_constants, local_params)
    if resolved is not None:
        used.update(resolved)
        return
    unresolved.append(
        _UnresolvedKindSite(
            path=rel_path,
            scope=scope_name,
            source=ast.unparse(kind_node),
            lineno=call_node.lineno,
        )
    )


def _scan_node(
    node: ast.AST,
    module_constants: dict[str, set[str]],
    local_assigns: dict[str, list[ast.expr]],
    local_params: frozenset[str],
    scope_name: str,
    rel_path: str,
    used: set[str],
    unresolved: list[_UnresolvedKindSite],
) -> None:
    if isinstance(node, ast.Call):
        func_name = _emit_func_name(node)
        if func_name in _EMIT_FUNCS:
            if not _has_explicit_level(node, local_assigns, module_constants, local_params):
                kind_node = _locate_arg(node, 1, "kind")
                _record_kind_site(
                    kind_node,
                    node,
                    module_constants,
                    local_assigns,
                    local_params,
                    scope_name,
                    rel_path,
                    used,
                    unresolved,
                )
        elif func_name == "_route_to_rework":
            kind_node = _locate_arg(node, 4, "event_kind")
            _record_kind_site(
                kind_node,
                node,
                module_constants,
                local_assigns,
                local_params,
                scope_name,
                rel_path,
                used,
                unresolved,
            )

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Entering a new function scope: its local assignments and parameters
        # shadow the enclosing ones rather than extending them.
        scoped_assigns = _collect_local_assignments(node)
        scoped_params = _collect_param_names(node)
        for child in ast.iter_child_nodes(node):
            _scan_node(
                child,
                module_constants,
                scoped_assigns,
                scoped_params,
                node.name,
                rel_path,
                used,
                unresolved,
            )
        return

    for child in ast.iter_child_nodes(node):
        _scan_node(
            child,
            module_constants,
            local_assigns,
            local_params,
            scope_name,
            rel_path,
            used,
            unresolved,
        )


def _scan_tree(tree: ast.Module, rel_path: str) -> tuple[set[str], list[_UnresolvedKindSite]]:
    """Scan one already-parsed module for emit-site kinds.

    Exposed separately from ``_scan_event_kinds`` so tests can feed it a
    synthetic module without writing a file to disk.
    """
    used: set[str] = set()
    unresolved: list[_UnresolvedKindSite] = []
    module_local = _collect_local_assignments(tree)
    module_constants: dict[str, set[str]] = {}
    for name, value_nodes in module_local.items():
        # A module-level name assigned more than once (e.g. reassigned under
        # an `if`) is only a "constant" -- resolvable at every reference site
        # -- if *every* assignment resolves. One unresolvable assignment (a
        # function call, an unknown name, ...) means some path could carry an
        # unproven value, so `parts` is reset and abandoned rather than
        # unioning in only the assignments that happened to resolve: that
        # would silently drop the unresolvable branch the same way `None`
        # was dropped from a `level=` union (#1029).
        parts: list[set[str]] = []
        for value_node in value_nodes:
            part = _resolve_literal(value_node, module_local, {}, frozenset())
            if part is None:
                parts = []
                break
            parts.append(part)
        if parts:
            module_constants[name] = set().union(*parts)
    _scan_node(
        tree, module_constants, module_local, frozenset(), "<module>", rel_path, used, unresolved
    )
    return used, unresolved


def _scan_event_kinds(root: Path) -> tuple[set[str], list[_UnresolvedKindSite]]:
    """Walk every Python file under ``root``; return (used kinds, unresolved sites)."""
    used: set[str] = set()
    unresolved: list[_UnresolvedKindSite] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        file_used, file_unresolved = _scan_tree(tree, path.relative_to(root).as_posix())
        used |= file_used
        unresolved.extend(file_unresolved)
    return used, unresolved


def _known_level(kind: str) -> bool:
    """Return True if ``kind`` is in the registry or is a registered sweep."""
    if kind in _LEVEL_BY_KIND:
        return True
    if kind.endswith("_sweep") and kind[: -len("_sweep")] in _LEVEL_BY_KIND:
        return True
    return False


# Sites where the scanner cannot statically resolve the ``kind`` argument to a
# finite literal set, and why that's fine. Every entry must be independently
# justified: either the real literal is chosen at a call site the scanner
# already covers elsewhere (a forwarding wrapper -- the literal at the
# *caller* is what's checked), or it's covered by a dedicated test below.
# ``test_event_kind_registry_exhaustive`` enforces both directions: an
# unresolved site missing from this list fails the test, and so does an
# entry here that no longer matches a real unresolved site.
_ALLOWED_UNRESOLVED_KIND_SITES: tuple[_UnresolvedKindSite, ...] = (
    _UnresolvedKindSite(
        path="state.py",
        scope="append_event",
        source="kind",
        reason=(
            "append_event forwards its own `kind` parameter to log_event. "
            "The literal is chosen at each call site, and append_event is "
            "itself in _EMIT_FUNCS, so every call site is already scanned -- "
            "this is the same site observed from inside the callee."
        ),
    ),
    _UnresolvedKindSite(
        path="workflow.py",
        scope="_record_event",
        source="kind",
        reason=(
            "OrchestratorApp._record_event forwards its own `kind` parameter "
            "to append_event. Same pass-through as append_event/log_event; "
            "every self._record_event(...) call site is scanned."
        ),
    ),
    _UnresolvedKindSite(
        path="orchestration/state_rework_routing.py",
        scope="_route_to_rework",
        source="event_kind",
        reason=(
            "_route_to_rework forwards its own `event_kind` parameter to "
            "self._record_event. Every self._route_to_rework(...) call site "
            "is scanned (it is in _WRAPPER_FUNCS, 5th positional argument). "
            "Moved out of workflow.py to charlie_work.orchestration."
            "state_rework_routing by leaf L01 b1 (#1632); the allow-list key "
            "follows the member to its new module -- scope/source/reason "
            "unchanged."
        ),
    ),
    _UnresolvedKindSite(
        path="stalled_review_reap.py",
        scope="_append_sweep_events",
        source="kind",
        reason=(
            "kind is the loop variable over sweep_events, a list built by "
            'many `sweep_events.append(("literal_kind", payload))` call '
            "sites elsewhere in this file. Those literals are independently "
            "verified by test_sweep_event_append_kinds_are_registered."
        ),
    ),
    _UnresolvedKindSite(
        path="stalled_review_reap.py",
        scope="_append_sweep_events",
        source="f'{kind}_sweep'",
        reason="Same `kind` loop variable as above, with the `_sweep` suffix appended.",
    ),
    _UnresolvedKindSite(
        path="supervise.py",
        scope="_log_self_deploy_outcome",
        source="_self_deploy_event_kind(result)",
        reason=(
            "_self_deploy_event_kind(result) returns one of "
            "self_deploy_{failed,succeeded,skipped}; every branch is "
            "verified by test_self_deploy_event_kind_only_returns_registered_kinds."
        ),
    ),
    _UnresolvedKindSite(
        path="write_gate.py",
        scope="append_event",
        source="kind",
        reason=(
            "WriteGate.append_event forwards its own `kind` parameter to "
            "state.append_event. Same pass-through as the existing "
            "state.py/append_event entry above; the literal is chosen at "
            "each call site, and every self.write_gate.append_event(...) "
            "call site is scanned there."
        ),
    ),
    _UnresolvedKindSite(
        path="write_gate.py",
        scope="record_event",
        source="kind",
        reason=(
            "WriteGate.record_event forwards its own `kind` parameter to "
            "state.append_event, mirroring OrchestratorApp._record_event's "
            "own forwarding shape (see the workflow.py/_record_event entry "
            "above). Every self.write_gate.record_event(...) call site is "
            "scanned there; `record_event` is itself in _EMIT_FUNCS so no "
            "coverage gap opens once a call site migrates onto it."
        ),
    ),
    _UnresolvedKindSite(
        path="write_gate.py",
        scope="log_event",
        source="kind",
        reason=(
            "WriteGate.log_event forwards its own `kind` parameter to "
            "instrumentation.log_event. The literal is chosen at each call "
            "site, and every self.write_gate.log_event(...) call site is "
            "scanned there."
        ),
    ),
    _UnresolvedKindSite(
        path="dead_worker_reap.py",
        scope="_attempt_salvage",
        source="salvage_skip_event_kind(skip_reason)",
        reason=(
            "Issue #1241: salvage_skip_event_kind maps skip_reason to one of "
            "two registered literals (salvage_skipped_already_landed for the "
            "#1221 reasons, salvage_skipped_superseded for the new "
            "commits_reachable reason), both in _LEVEL_BY_KIND. The mapping "
            "is verified by test_salvage_skip_event_kind_only_returns_registered_kinds "
            "in tests/test_salvage_superseded_1241.py."
        ),
    ),
    _UnresolvedKindSite(
        path="reconcile.py",
        scope="apply_fixes",
        source="salvage_skip_event_kind(skip_reason)",
        reason=(
            "Issue #1241: same salvage_skip_event_kind mapping as the "
            "dead_worker_reap.py/_attempt_salvage entry above -- the reconcile "
            "salvage lane and the workflow salvage lane share the single "
            "enforcement point in salvage_superseded.py. Both target literals "
            "are in _LEVEL_BY_KIND and verified by "
            "test_salvage_skip_event_kind_only_returns_registered_kinds."
        ),
    ),
)


def test_event_kind_registry_exhaustive() -> None:
    """#910/#995: every emit-site kind in this package is registered or accounted for.

    Every kind resolvable to a literal set must be a member of
    ``_LEVEL_BY_KIND`` (or a registered ``_sweep`` variant). Every site the
    scanner cannot resolve must be in ``_ALLOWED_UNRESOLVED_KIND_SITES`` with
    a reason -- an unresolved, unlisted site fails the build instead of
    silently contributing nothing (#995), and a listed entry that no longer
    matches a real unresolved site fails too (a stale allow-list is a lie).
    """
    src_root = Path(__file__).parents[1] / "src" / "charlie_work"
    used, unresolved = _scan_event_kinds(src_root)

    # ci_fleet is a separate package (a sibling repo, not owned by this PR)
    # that logs through this package's sink. Its literal kinds still belong
    # in the registry check below -- an unregistered kind is a real bug
    # regardless of which repo introduced it. But its *unresolved* sites
    # deliberately do NOT feed the fail-closed assertions further down: this
    # test can only allow-list (and can only fix) unresolved sites in the
    # charlie_work tree it ships. Enforcing ci_fleet's unresolved sites here
    # would make charlie_work's CI fail on a file no charlie_work change
    # touched, and would require an allow-list entry this repo can't attach
    # a supporting test to. If ci_fleet needs the same guard, it belongs in
    # ci_fleet's own test suite, scanning its own source.
    spec = importlib.util.find_spec("ci_fleet")
    if spec is not None and spec.origin:
        ci_root = Path(spec.origin).parent
        ci_used, _ci_unresolved_not_enforced_here = _scan_event_kinds(ci_root)
        used |= ci_used

    unregistered = {k for k in used if not _known_level(k)}
    assert not unregistered, f"unregistered event kinds: {sorted(unregistered)}"

    allowed_keys = {site.key for site in _ALLOWED_UNRESOLVED_KIND_SITES}
    found_keys = {site.key for site in unresolved}

    unaccounted = [site for site in unresolved if site.key not in allowed_keys]
    assert not unaccounted, (
        "unresolved event-kind expression(s) not in _ALLOWED_UNRESOLVED_KIND_SITES "
        "(either make the expression statically resolvable, pass an explicit "
        "level= at the call site, or add a reasoned allow-list entry): "
        + "; ".join(
            f"{site.path}:{site.lineno} in {site.scope}(): `{site.source}`" for site in unaccounted
        )
    )

    stale = [site for site in _ALLOWED_UNRESOLVED_KIND_SITES if site.key not in found_keys]
    assert not stale, (
        "_ALLOWED_UNRESOLVED_KIND_SITES entry no longer matches any unresolved "
        "site -- remove it or update it to match the current source: "
        + "; ".join(f"{site.path} in {site.scope}(): `{site.source}`" for site in stale)
    )


def test_review_dispatch_skipped_ci_red_kind_registered_matches_family() -> None:
    """Issue #1258: the janitor's CI-red short-circuit (sole-failure and the
    new co-occurring-failure branch alike) must have a dedicated, registered
    provenance kind -- previously it only produced whatever generic
    ``record_review`` itself logs, with nothing naming the deterministic
    gate as the decision's source.

    Pinned to the ``review_dispatch_*`` family per the issue's binding
    comment (which corrects the plan body's originally-proposed
    ``review_skipped_ci_red`` naming) so it groups with
    ``review_dispatch_claim``/``review_dispatch`` for ``event_counts_by_kind``
    roll-ups, and pinned to level ``info``: this is the deterministic gate
    doing its routine job (routing to rework without ever starting a paid
    reviewer session), not a condition that ended a lane or lost work.
    """
    assert "review_dispatch_skipped_ci_red" in _LEVEL_BY_KIND
    assert _LEVEL_BY_KIND["review_dispatch_skipped_ci_red"] == "info"
    assert "review_dispatch_skipped_ci_red".startswith("review_dispatch_")

    # Deferral (d): the stale/absent-checks auto-retrigger is W17's, landing
    # after this item in the lane -- no retrigger emitter exists yet, so no
    # retrigger-family kind may be registered here. A registered-but-unused
    # kind would be exactly as misleading as an emitted-but-unregistered one:
    # it would claim a mechanism exists that this diff never builds.
    retrigger_kinds = {
        kind
        for kind in _LEVEL_BY_KIND
        if kind.startswith("review_dispatch_") and "retrigger" in kind
    }
    assert not retrigger_kinds, (
        f"no retrigger-family kind may be registered by this item (W17's job): {retrigger_kinds}"
    )


def test_expected_operational_kinds_are_all_registered_warnings() -> None:
    """#1271: bucketing only makes sense for warnings.

    Every member of ``EXPECTED_OPERATIONAL_KINDS`` must be registered in
    ``_LEVEL_BY_KIND`` at ``"warning"`` -- an info or error kind (or an
    unregistered one) accidentally added to the set would silently vanish
    from ``check_error_events``'s coverage or from the info stream, since
    ``check_warning_events`` only ever queries ``level = 'warning'`` rows.
    """
    assert EXPECTED_OPERATIONAL_KINDS, "the set must not be empty"
    for kind in EXPECTED_OPERATIONAL_KINDS:
        assert kind in _LEVEL_BY_KIND, f"{kind} is not registered in _LEVEL_BY_KIND"
        assert _LEVEL_BY_KIND[kind] == "warning", (
            f"{kind} is registered at level {_LEVEL_BY_KIND[kind]!r}, not 'warning' -- "
            "bucketing only makes sense for warning-level kinds"
        )


def test_self_deploy_event_kind_only_returns_registered_kinds() -> None:
    """#995: independently verify the claim behind the supervise.py allow-list entry.

    ``_self_deploy_event_kind`` is call-based, so the static scanner cannot
    resolve it and it is allow-listed above on the strength of this test.
    This enumerates every branch of the function (by constructing a
    ``SelfDeployResult`` for each) and checks each returned kind against the
    registry, so the allow-list's claim is enforced on every run rather than
    trusted from a comment.
    """
    from charlie_work.supervise import SelfDeployResult, _self_deploy_event_kind

    failed = SelfDeployResult(ok=False, pulled=False, changed=False, synced=False)
    succeeded_changed = SelfDeployResult(ok=True, pulled=True, changed=True, synced=True)
    succeeded_venv_repaired = SelfDeployResult(
        ok=True, pulled=True, changed=False, synced=False, venv_repaired=True
    )
    skipped = SelfDeployResult(ok=True, pulled=True, changed=False, synced=False)

    for result in (failed, succeeded_changed, succeeded_venv_repaired, skipped):
        kind = _self_deploy_event_kind(result)
        assert _known_level(kind), f"{kind} (from {result}) is not registered"

    assert _self_deploy_event_kind(failed) == "self_deploy_failed"
    assert _self_deploy_event_kind(succeeded_changed) == "self_deploy_succeeded"
    assert _self_deploy_event_kind(succeeded_venv_repaired) == "self_deploy_succeeded"
    assert _self_deploy_event_kind(skipped) == "self_deploy_skipped"


def _scan_sweep_append_kinds(root: Path) -> tuple[set[str], list[str]]:
    """Scan every ``*.py`` under ``root`` for ``sweep_events.append((kind, payload))``
    call sites; return (resolved literal kinds, unresolvable-element descriptions).

    Factored out of ``test_sweep_event_append_kinds_are_registered`` so the
    fail-closed behavior on an unresolvable kind element (#1029) can be
    exercised directly against a synthetic tree, the same way
    ``_scan_event_kinds`` is factored out for the main scanner.
    """
    found: set[str] = set()
    unresolved: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sweep_events"
            ):
                continue
            if not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Tuple) and arg.elts:
                resolved = _resolve_literal(arg.elts[0], {}, {})
                if resolved is None:
                    # #1029: this scan's whole premise is that every literal
                    # at a `sweep_events.append((kind, payload))` site is
                    # registered -- an element the resolver can't prove a
                    # literal for undermines that premise and must surface,
                    # not silently contribute nothing to `found` (the same
                    # union-with-empty-set fail-open the `level=` bug had).
                    unresolved.append(
                        f"{path.relative_to(root).as_posix()}:{node.lineno}: "
                        f"{ast.unparse(arg.elts[0])}"
                    )
                    continue
                found |= resolved
    return found, unresolved


def test_sweep_event_append_kinds_are_registered() -> None:
    """#995: independently verify the claim behind the two `_append_sweep_events` entries.

    `_append_sweep_events`'s `kind` loop variable is allow-listed above
    because the real literal is chosen at each `sweep_events.append((kind,
    payload))` call site, not in the loop. This scans for exactly those call
    sites directly and checks every literal they contribute against the
    registry, so that claim is enforced rather than trusted.
    """
    src_root = Path(__file__).parents[1] / "src" / "charlie_work"
    found, unresolved = _scan_sweep_append_kinds(src_root)

    assert not unresolved, (
        "sweep_events.append((kind, payload)) site(s) with an unresolvable "
        "kind element -- make it a literal or trace it manually: " + "; ".join(unresolved)
    )
    assert found, "expected at least one sweep_events.append((kind, payload)) call site"
    unregistered = {k for k in found if not _known_level(k)}
    assert not unregistered, f"unregistered sweep_events kinds: {sorted(unregistered)}"


def test_sweep_append_kind_scan_flags_unresolvable_element_not_silently_dropped(
    tmp_path: Path,
) -> None:
    """#1029 (sweep-kind direction): an unresolvable ``sweep_events.append((kind,
    payload))`` element must surface as unresolved, not vanish from ``found``
    via ``found |= _resolve_literal(...)`` unioning in an empty result.

    Asserts on the observable consequence -- the site is flagged -- rather
    than on ``found`` being empty, since an empty ``found`` is also what a
    correctly-behaving scan produces when nothing resolves; only the
    unresolved list distinguishes "silently dropped" from "surfaced".
    """
    (tmp_path / "fixture.py").write_text(
        "def _append(kind_choice):\n    sweep_events.append((kind_choice(), {'x': 1}))\n",
        encoding="utf-8",
    )
    found, unresolved = _scan_sweep_append_kinds(tmp_path)
    assert not found
    assert unresolved, "unresolvable sweep_events.append kind element was silently dropped"
    assert "kind_choice()" in unresolved[0]


def test_scanner_module_constant_with_unresolvable_value_not_silently_dropped() -> None:
    """#1029 (module-constant direction): a module-level name reassigned on
    two branches -- one a string literal, one unresolvable (a function call)
    -- must not be treated as resolved to just the literal branch.

    A single-assignment unresolvable constant doesn't discriminate this bug:
    an empty accumulator unioned with one unresolvable ``None``/empty result
    stays empty either way. The bug only shows up with a *mix*, exactly like
    ``level="warning" if cond else None`` -- the old code's
    ``resolved |= _resolve_literal(...) or set()`` would union in the
    resolvable branch (``{"kind_a"}``) and silently drop the unresolvable one,
    ending up "resolved" to ``{"kind_a"}`` when the true value could be
    anything ``_compute_kind()`` returns. Asserts the emit site is flagged
    *unresolved* -- not merely that ``"kind_a"`` is absent from ``used``,
    since that alone doesn't distinguish "silently dropped" (buggy: `used` is
    `{"kind_a"}`, non-empty) from "correctly surfaced" (fixed: `used` is
    empty because `unresolved` is used instead).
    """
    source = (
        "if _flag():\n"
        "    KIND = 'kind_a'\n"
        "else:\n"
        "    KIND = _compute_kind()\n\n\n"
        "def emit():\n    log_event(state_path, KIND, {})\n"
    )
    tree = ast.parse(source)
    used, unresolved = _scan_tree(tree, "fixture.py")
    assert not used, f"expected no resolvable kinds (one branch is unresolvable), got {used}"
    assert unresolved, (
        "module constant with a mixed resolvable/unresolvable value was dropped (#1029)"
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "def emit(kind):\n    log_event(state_path, kind, {})\n",
            id="bare-variable",
        ),
        pytest.param(
            "def emit(suffix):\n    log_event(state_path, f'job_{suffix}', {})\n",
            id="f-string-with-unresolvable-interpolation",
        ),
        pytest.param(
            "def emit(a, b):\n    log_event(state_path, a if _flag() else b, {})\n",
            id="conditional-expression-with-nonliteral-branches",
        ),
        pytest.param(
            "def emit(k):\n    log_event(state_path, kind=k, payload={})\n",
            id="keyword-passed-kind-argument",
        ),
        pytest.param(
            "def emit(kind, flag):\n    if flag:\n        kind = 'kind_a'\n    log_event(state_path, kind, {})\n",
            id="parameter-reassigned-on-one-branch-only",
        ),
    ],
)
def test_scanner_flags_unresolved_nonliteral_kind_forms(source: str) -> None:
    """#995 regression control: each non-literal form must be surfaced as
    *unresolved*, never silently dropped.

    This reproduces the exact #995 bug shape: the pre-fix scanner recognised
    only a bare string constant and a ternary between two literals, and
    treated every other shape -- a bare variable, an f-string, a ternary
    between two non-literal branches -- as contributing no kinds, which is
    indistinguishable from a legitimately kind-less call. A test that only
    exercises the literal case cannot detect a regression back to that
    behavior; this asserts the *unresolved* branch is reached instead.

    The keyword-passed-kind case guards the scanner's own `_locate_arg`
    fallback: a call site that passes ``kind=`` by keyword must be located
    the same as a positional one, not silently skipped because
    ``len(node.args) < 2``. The reassigned-on-one-branch case guards
    ``local_params``: a function parameter that gets a literal value on only
    one conditional path must still be treated as unresolved overall, since
    the parameter's own (unknown) value can reach the call site on the path
    that never executes the reassignment. Without the `local_params` check
    this would incorrectly resolve to ``{"kind_a"}``.
    """
    tree = ast.parse(source)
    used, unresolved = _scan_tree(tree, "fixture.py")
    assert not used, f"expected no resolvable kinds for a non-literal form, got {used}"
    assert unresolved, "non-literal kind form was silently dropped instead of flagged (#995)"


@pytest.mark.parametrize(
    "source,expected",
    [
        pytest.param(
            "def emit():\n    log_event(state_path, 'plain_kind', {})\n",
            {"plain_kind"},
            id="literal",
        ),
        pytest.param(
            "def emit(flag):\n    log_event(state_path, 'kind_a' if flag else 'kind_b', {})\n",
            {"kind_a", "kind_b"},
            id="ternary-of-literals",
        ),
        pytest.param(
            "def emit(flag):\n"
            "    if flag:\n"
            "        kind = 'kind_a'\n"
            "    else:\n"
            "        kind = 'kind_b'\n"
            "    log_event(state_path, kind, {})\n",
            {"kind_a", "kind_b"},
            id="multi-branch-local-assignment",
        ),
        pytest.param(
            "KIND = 'module_kind'\n\n\ndef emit():\n    log_event(state_path, KIND, {})\n",
            {"module_kind"},
            id="module-level-constant",
        ),
        pytest.param(
            "def emit(flag):\n"
            "    kind = 'kind_a' if flag else 'kind_b'\n"
            "    log_event(state_path, f'{kind}_sweep', {})\n",
            {"kind_a_sweep", "kind_b_sweep"},
            id="f-string-with-resolvable-interpolation",
        ),
    ],
)
def test_scanner_resolves_provably_finite_nonliteral_kind_forms(
    source: str, expected: set[str]
) -> None:
    """Contrast case for the regression control above: forms that ARE
    statically provable to a finite literal set must still resolve, not be
    over-eagerly flagged. Covers the generalised local-variable (including
    multi-branch if/elif), module-constant, and f-string tracing this fix
    adds -- each was a real emit site in this package before this fix.
    """
    tree = ast.parse(source)
    used, unresolved = _scan_tree(tree, "fixture.py")
    assert used == expected
    assert not unresolved


def test_scanner_accepts_explicit_level_without_requiring_kind_resolution() -> None:
    """A call site with a literal ``level=`` is self-classifying (mirrors
    ``log_event``'s runtime behavior) and does not need its kind registered
    or allow-listed, even if the kind itself is unresolvable."""
    source = (
        "def emit(dynamic_kind):\n    log_event(state_path, dynamic_kind, {}, level='warning')\n"
    )
    tree = ast.parse(source)
    used, unresolved = _scan_tree(tree, "fixture.py")
    assert not used
    assert not unresolved


def _call_node_for(source: str) -> tuple[ast.Call, dict[str, list[ast.expr]], frozenset[str]]:
    """Parse ``source``, return the single ``log_event`` ``Call`` node plus its
    enclosing function's local assignments and parameter names.

    Test helper for exercising ``_has_explicit_level`` directly, without going
    through the full ``_scan_tree`` walk.
    """
    tree = ast.parse(source)
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    call = next(
        n for n in ast.walk(func) if isinstance(n, ast.Call) and _emit_func_name(n) == "log_event"
    )
    return call, _collect_local_assignments(func), _collect_param_names(func)


@pytest.mark.parametrize(
    "source,expected",
    [
        pytest.param(
            "def emit():\n    log_event(state_path, 'k', {}, level='warning')\n",
            True,
            id="literal-str-only",
        ),
        pytest.param(
            "def emit(cond):\n"
            "    log_event(state_path, 'k', {}, level='warning' if cond else None)\n",
            False,
            id="str-or-none-conditional",
        ),
        pytest.param(
            "def emit():\n    log_event(state_path, 'k', {}, level=None)\n",
            False,
            id="plain-none",
        ),
        pytest.param(
            "def emit():\n    log_event(state_path, 'k', {}, level=_compute_level())\n",
            False,
            id="call-node",
        ),
        pytest.param(
            "def emit(lvl):\n    log_event(state_path, 'k', {}, level=lvl)\n",
            False,
            id="name-variable",
        ),
    ],
)
def test_has_explicit_level_fails_closed_on_none_admitting_expressions(
    source: str, expected: bool
) -> None:
    """#1029: a ``level=`` expression that can be ``None`` on some branch must
    not exempt the site from kind-registry verification.

    ``None`` is the documented "fall back to ``_LEVEL_BY_KIND``" signal (see
    ``instrumentation.log_event``), so a site whose level is conditionally
    ``None`` -- e.g. ``level="warning" if cond else None`` -- is still
    registry-dependent on the ``None`` path. Before this fix, ``_resolve_literal``
    unioned away the ``None`` branch (it only collects string constants) and
    ``_has_explicit_level`` saw only ``{"warning"}``, wrongly exempting the site.

    ``literal-str-only`` is the positive control: it must stay ``True`` so this
    test cannot pass merely because the helper started returning ``False``
    unconditionally. ``call-node`` and ``name-variable`` guard the same
    fail-closed behavior for expression shapes the resolver cannot statically
    reduce at all.
    """
    call, local_assigns, local_params = _call_node_for(source)
    assert _has_explicit_level(call, local_assigns, {}, local_params) is expected


def test_issue_910_active_kinds_are_error_or_warning(tmp_path: Path) -> None:
    """#910: the 11 production-missed active kinds are now classified.

    The table from the issue body; two rows (review_dispatch_escalated and
    review_verdict_missed) were also discussed in the co-occurrence comment,
    which did not change their enrollment in the error stream. Their levels are
    the same as the issue's proposed table.
    """
    expected = {
        "review_verdict_missed": "error",
        "review_dispatch_escalated": "error",
        "merge_failed_attempt_alarm": "error",
        "dispatch_blocked_chain_dead": "error",
        "flake_rerun_failed": "warning",
        "quota_probe_failed": "warning",
        "janitor_rework_escalated": "error",
        "merge_deferred_stale_base_alarm": "error",
        "janitor_rework_stalled": "warning",
        "supervise_relaunch_cap_reached": "warning",
    }
    for kind, level in expected.items():
        assert _LEVEL_BY_KIND[kind] == level, f"{kind} should be {level!r}"


def test_issue_910_latent_kinds_are_classified(tmp_path: Path) -> None:
    """#910: the additional unclassified but zero-event kinds are enrolled."""
    expected = {
        "infra_rerun_failed": "warning",
        "infra_rerun_escalated": "error",
        "reconcile_pass_failed": "error",
        "session_budget_exceeded": "warning",
        "deescalation_cap_exhausted": "warning",
        "required_changes_vacuous": "warning",
        "rescue_review_escalated": "error",
        "janitor_rework_cycle_failed": "error",
        "worktree_foreign_writer": "warning",
    }
    for kind, level in expected.items():
        assert _LEVEL_BY_KIND[kind] == level, f"{kind} should be {level!r}"


def test_sweep_inherits_base_level(tmp_path: Path) -> None:
    """Sweep-aggregated kinds (``{base}_sweep``) inherit the base kind's level."""
    state_path = tmp_path / "state.json"
    log_event(state_path, "review_dispatch_stalled_sweep", {"count": 3})
    log_event(state_path, "orphaned_worker_drift_sweep", {"count": 5})
    log_event(state_path, "unknown_kind_sweep", {"count": 1})

    events = read_event_log(state_path)
    levels = {e["kind"]: e["level"] for e in events}
    assert levels["review_dispatch_stalled_sweep"] == "error"
    assert levels["orphaned_worker_drift_sweep"] == "info"
    assert levels["unknown_kind_sweep"] == "info"
