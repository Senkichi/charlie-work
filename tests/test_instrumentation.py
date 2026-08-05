"""Tests for the instrumentation module: SQLite event log, correlation IDs, dual-write."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from charlie_work.instrumentation import (
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
# Issue #910: event-level registry must cover all in-repo emit sites
# ---------------------------------------------------------------------------


def _literal_strings(node: ast.expr) -> set[str]:
    """Extract every string-constant branch from an expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _literal_strings(node.body) | _literal_strings(node.orelse)
    return set()


def _event_kind_usage_from_ast(tree: ast.Module) -> set[str]:
    """Return all literal event-kind strings used in emit or wrapper calls."""
    kinds: set[str] = set()
    _EMIT_FUNCS = {"log_event", "append_event", "_record_event"}
    _WRAPPER_FUNCS = {"_route_to_rework"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func_name: str | None = None
        if isinstance(node.func, ast.Name) and node.func.id in _EMIT_FUNCS | _WRAPPER_FUNCS:
            func_name = node.func.id
        elif (
            isinstance(node.func, ast.Attribute) and node.func.attr in _EMIT_FUNCS | _WRAPPER_FUNCS
        ):
            func_name = node.func.attr

        if func_name in _EMIT_FUNCS and len(node.args) >= 2:
            kinds.update(_literal_strings(node.args[1]))

        if func_name == "_route_to_rework" and len(node.args) >= 5:
            kinds.update(_literal_strings(node.args[4]))

    # Also collect strings assigned to a local named ``event_kind`` that is
    # later passed to an emit function as a variable (e.g. session_exited).
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "event_kind" for target in node.targets
        ):
            kinds.update(_literal_strings(node.value))

    return kinds


def _scan_event_kinds(root: Path) -> set[str]:
    """Walk every Python file under ``root`` and collect literal event kinds."""
    kinds: set[str] = set()
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        kinds.update(_event_kind_usage_from_ast(tree))
    return kinds


def _known_level(kind: str) -> bool:
    """Return True if ``kind`` is in the registry or is a registered sweep."""
    if kind in _LEVEL_BY_KIND:
        return True
    if kind.endswith("_sweep") and kind[: -len("_sweep")] in _LEVEL_BY_KIND:
        return True
    return False


def test_event_kind_registry_exhaustive() -> None:
    """#910: every literal event kind in this package must be registered.

    The registry is the single source of truth for ``level`` classification.
    A static test is the enforcement point: new call sites cannot introduce an
    unregistered kind without this test failing. Unknown kinds still default to
    ``"info"`` at runtime to preserve the best-effort instrumentation contract.
    """
    src_root = Path(__file__).parents[1] / "src" / "charlie_work"
    used = _scan_event_kinds(src_root)

    # ci_fleet is a separate package, but it logs through this package's sink.
    # If its source is available, include its literal kinds as well.
    spec = importlib.util.find_spec("ci_fleet")
    if spec is not None and spec.origin:
        ci_root = Path(spec.origin).parent
        used |= _scan_event_kinds(ci_root)

    unregistered = {k for k in used if not _known_level(k)}
    assert not unregistered, f"unregistered event kinds: {sorted(unregistered)}"


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
        "cross_family_verdict_abandoned": "error",
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
