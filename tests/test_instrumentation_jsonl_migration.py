"""JSONL-migration and dedupe regression tests for ``charlie_work.instrumentation``.

Split out of ``tests/test_instrumentation.py`` (issue #1569, Track-1):
the ``events.jsonl`` -> ``events.db`` one-shot migration, its crash-rerun
idempotency and malformed-line tolerance, and the first-access dedupe
regressions (#557), plus the ``_write_jsonl`` helper they share.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work.instrumentation import close_db, log_event, read_event_log


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path) -> None:
    """Ensure DB connections are closed between tests to avoid cross-test contamination."""
    yield
    # Close any connections that were opened during this test
    close_db(tmp_path / "state.json")
    # Also try the variant paths used in some tests
    close_db(tmp_path / "subdir" / "state.json")
    close_db(tmp_path / "nonexistent_dir" / "state.json")


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
