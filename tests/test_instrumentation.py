"""Tests for the instrumentation module: SQLite event log, correlation IDs, dual-write."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work.instrumentation import (
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
