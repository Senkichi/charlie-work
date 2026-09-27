"""Payload-tolerance regression tests for the events.db readers (issue #1883).

Split out of ``test_instrumentation.py`` to keep that module inside its
frozen attachment budget: these tests exercise the degrade-not-crash
boundary for event rows whose ``payload`` (or schema) no in-tree writer
would produce — NULL, non-JSON text, non-dict JSON, and missing columns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work.instrumentation import (
    close_db,
    events_by_correlation_id,
    query_events,
    read_event_log,
)


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path) -> None:
    """Close the cached reader connection between tests (see test_instrumentation.py)."""
    yield
    close_db(tmp_path / "state.json")


def _make_legacy_events_db(state_path: Path, rows: list[tuple[str, str, object]]) -> None:
    """Create an events.db whose ``events`` table allows NULL payloads.

    ``CREATE TABLE IF NOT EXISTS`` in ``_SCHEMA_SQL`` leaves a pre-existing
    ``events`` table untouched, so a database file created by a different
    (older/foreign) schema — or corrupted — can hold ``payload`` values no
    in-tree writer would produce. This builds exactly such a table (the
    current column set minus the ``NOT NULL`` on ``payload``) and inserts
    ``(ts, kind, payload)`` rows verbatim.

    Regression fixture for issue #1883: ``json.loads(row["payload"])`` on a
    NULL payload raised ``TypeError: the JSON object must be str, bytes or
    bytearray, not NoneType``, which escaped the readers' ``sqlite3.Error``
    catch and crashed the calling fleet lane.
    """
    import sqlite3

    db_path = state_path.parent / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, kind TEXT, payload TEXT,
            repo TEXT, correlation_id TEXT, pr_number INTEGER,
            issue_number INTEGER, level TEXT DEFAULT 'info'
        );
        """
    )
    for ts, kind, payload in rows:
        conn.execute(
            "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
            (ts, kind, payload),
        )
    conn.commit()
    conn.close()


def test_read_event_log_tolerates_null_payload(tmp_path: Path) -> None:
    """A NULL payload degrades to ``{}`` instead of raising TypeError (#1883)."""
    state_path = tmp_path / "state.json"
    _make_legacy_events_db(
        state_path,
        [
            ("2025-01-01T00:00:00Z", "loop_started", None),
            ("2025-01-01T00:01:00Z", "dispatch", json.dumps({"issue": 7})),
        ],
    )

    events = read_event_log(state_path)
    assert [e["kind"] for e in events] == ["loop_started", "dispatch"]
    assert events[0]["payload"] == {}
    assert events[0]["ts"] == "2025-01-01T00:00:00Z"
    assert events[1]["payload"] == {"issue": 7}


def test_query_events_tolerates_null_payload(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    _make_legacy_events_db(state_path, [("2025-01-01T00:00:00Z", "loop_started", None)])

    events = query_events(state_path, kind="loop_started")
    assert len(events) == 1
    assert events[0]["payload"] == {}


def test_events_by_correlation_id_tolerates_null_payload(tmp_path: Path) -> None:
    import sqlite3

    state_path = tmp_path / "state.json"
    _make_legacy_events_db(state_path, [("2025-01-01T00:00:00Z", "loop_started", None)])
    conn = sqlite3.connect(str(state_path.parent / "events.db"))
    conn.execute("UPDATE events SET correlation_id = 'pass-1'")
    conn.commit()
    conn.close()

    events = events_by_correlation_id(state_path, "pass-1")
    assert len(events) == 1
    assert events[0]["payload"] == {}


def test_query_events_tolerates_malformed_payload_text(tmp_path: Path) -> None:
    """A non-JSON payload degrades to ``{}`` rather than raising JSONDecodeError."""
    state_path = tmp_path / "state.json"
    _make_legacy_events_db(state_path, [("2025-01-01T00:00:00Z", "dispatch", "not json {")])

    events = query_events(state_path)
    assert len(events) == 1
    assert events[0]["payload"] == {}


def test_query_events_tolerates_nondict_payload(tmp_path: Path) -> None:
    """A payload that parses to a non-dict coerces to ``{}`` — consumers
    call ``payload.get(...)`` unconditionally, so passing ``42`` or a list
    through would trade a TypeError for an AttributeError (#1883)."""
    state_path = tmp_path / "state.json"
    _make_legacy_events_db(
        state_path,
        [
            ("2025-01-01T00:00:00Z", "dispatch", "42"),
            ("2025-01-01T00:01:00Z", "dispatch", "[1, 2]"),
        ],
    )

    events = query_events(state_path, kind="dispatch")
    assert [e["payload"] for e in events] == [{}, {}]


def test_query_events_tolerates_missing_columns(tmp_path: Path) -> None:
    """An ``events`` table missing a reader column degrades to ``[]``.

    Same boundary as the NULL payload: ``SELECT *`` succeeds on a
    foreign-schema table (here: every column but ``level``, which only newer
    schema versions carry), then ``row["level"]`` raises ``IndexError`` —
    also outside the old ``sqlite3.Error`` catch (#1883). ``user_version``
    is set so the dedupe migration (which would fail first on a
    narrower table) is skipped and the row-level read is what runs.
    """
    import sqlite3

    state_path = tmp_path / "state.json"
    db_path = state_path.parent / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE events (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               ts TEXT, kind TEXT, payload TEXT,
               repo TEXT, correlation_id TEXT, pr_number INTEGER,
               issue_number INTEGER
           )"""
    )
    conn.execute(
        "INSERT INTO events (ts, kind, payload) VALUES ('2025-01-01T00:00:00Z', 'dispatch', '{}')"
    )
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()

    assert query_events(state_path) == []
    assert read_event_log(state_path) == []
    assert events_by_correlation_id(state_path, "pass-1") == []
