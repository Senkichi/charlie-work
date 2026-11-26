"""Tests for the exact-id event cursor: ``query_events(after_id=...)`` and ``max_event_id``.

Issue #3114. ``query_events`` could already filter by ``since`` (a
timestamp), but a timestamp is not an exact cursor: rows sharing one
``ts`` can be repeated or dropped across pages. The ``events`` table's
autoincrement ``id`` gives a consumer an exact "every event I have not
seen" boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work.instrumentation import (
    _get_db,
    close_db,
    events_by_correlation_id,
    log_event,
    max_event_id,
    query_events,
    read_event_log,
)


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path):
    """Ensure DB connections are closed between tests to avoid cross-test contamination."""
    yield
    close_db(tmp_path / "state.json")


def _insert_event(state_path: Path, ts: str, kind: str, payload: dict) -> None:
    """Insert one event row with an explicit timestamp, bypassing ``_now_iso``."""
    conn = _get_db(state_path)
    assert conn is not None
    conn.execute(
        """INSERT INTO events (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
           VALUES (?, ?, ?, NULL, NULL, NULL, NULL, 'info')""",
        (ts, kind, json.dumps(payload)),
    )


def test_event_dicts_carry_id(tmp_path: Path) -> None:
    """Every read helper's event dict carries its row ``id``."""
    state_path = tmp_path / "state.json"
    log_event(state_path, "evt_a", {"n": 1}, correlation_id="cid-1")
    log_event(state_path, "evt_b", {"n": 2}, correlation_id="cid-1")

    for events in (
        read_event_log(state_path),
        events_by_correlation_id(state_path, "cid-1"),
        query_events(state_path),
    ):
        assert len(events) == 2
        ids = [e["id"] for e in events]
        assert len(ids) == len(events)
        assert ids == sorted(ids)
        assert all(isinstance(i, int) for i in ids)


def test_after_id_pages_visit_every_row_once(tmp_path: Path) -> None:
    """Paging with ``after_id`` + ``limit`` visits each row exactly once."""
    state_path = tmp_path / "state.json"
    for i in range(10):
        log_event(state_path, f"evt_{i}", {"seq": i})

    all_ids = {e["id"] for e in query_events(state_path)}
    seen: list[int] = []
    cursor = 0
    while True:
        page = query_events(state_path, after_id=cursor, limit=3)
        if not page:
            break
        assert len(page) <= 3
        assert all(e["id"] > cursor for e in page)
        seen.extend(e["id"] for e in page)
        cursor = page[-1]["id"]

    assert seen == sorted(seen)
    assert set(seen) == all_ids
    assert len(seen) == len(all_ids)


def test_after_id_limit_returns_oldest_rows_after_cursor(tmp_path: Path) -> None:
    """With a cursor, ``limit`` is the oldest N after it — not the most recent N."""
    state_path = tmp_path / "state.json"
    for i in range(10):
        log_event(state_path, f"evt_{i}", {})

    boundary = query_events(state_path, limit=10)[3]["id"]
    page = query_events(state_path, after_id=boundary, limit=3)
    kinds = [e["kind"] for e in page]
    assert kinds == ["evt_4", "evt_5", "evt_6"]
    # The boundary row itself is excluded (strict ``id > after_id``).
    assert boundary not in {e["id"] for e in page}


def test_after_id_same_timestamp_rows_neither_repeated_nor_dropped(tmp_path: Path) -> None:
    """Rows sharing one ``ts`` page exactly once — the defect ``after_id`` exists for."""
    state_path = tmp_path / "state.json"
    shared_ts = "2026-11-25T12:00:00Z"
    for i in range(6):
        _insert_event(state_path, shared_ts, f"same_ts_{i}", {"seq": i})

    seen: list[int] = []
    cursor = 0
    while True:
        page = query_events(state_path, after_id=cursor, limit=2)
        if not page:
            break
        seen.extend(e["id"] for e in page)
        cursor = page[-1]["id"]

    assert len(seen) == 6
    assert len(set(seen)) == 6
    assert seen == sorted(seen)


def test_after_id_composes_with_filters(tmp_path: Path) -> None:
    """``after_id`` composes with ``kind``, ``issue_number``, ``pr_number`` and ``repo``."""
    state_path = tmp_path / "state.json"
    # A matching row BEFORE the cursor must stay excluded.
    log_event(state_path, "hit", {"pr_number": 7, "issue_number": 42}, repo="r1")
    cursor = query_events(state_path)[-1]["id"]
    # Rows after the cursor: only the fully-matching one may come back.
    log_event(state_path, "hit", {"pr_number": 7, "issue_number": 42}, repo="r1")
    log_event(state_path, "hit", {"pr_number": 7, "issue_number": 43}, repo="r1")
    log_event(state_path, "hit", {"pr_number": 8, "issue_number": 42}, repo="r1")
    log_event(state_path, "miss", {"pr_number": 7, "issue_number": 42}, repo="r1")
    log_event(state_path, "hit", {"pr_number": 7, "issue_number": 42}, repo="r2")

    results = query_events(
        state_path,
        after_id=cursor,
        kind="hit",
        pr_number=7,
        issue_number=42,
        repo="r1",
    )
    assert len(results) == 1
    assert results[0]["id"] > cursor


def test_query_events_without_after_id_unchanged(tmp_path: Path) -> None:
    """Calls that do not pass ``after_id`` return exactly what they returned before."""
    state_path = tmp_path / "state.json"
    for i in range(10):
        log_event(state_path, f"evt_{i}", {})

    results = query_events(state_path, limit=3)
    assert [e["kind"] for e in results] == ["evt_7", "evt_8", "evt_9"]

    unlimited = query_events(state_path)
    assert [e["kind"] for e in unlimited] == [f"evt_{i}" for i in range(10)]


def test_max_event_id_empty_and_populated(tmp_path: Path) -> None:
    """``max_event_id`` returns 0 for an empty/absent db and the top id otherwise."""
    state_path = tmp_path / "state.json"
    assert max_event_id(state_path) == 0
    # An empty-but-created database also reports 0.
    assert _get_db(state_path) is not None
    assert max_event_id(state_path) == 0

    log_event(state_path, "evt", {})
    first_id = query_events(state_path)[0]["id"]
    assert max_event_id(state_path) == first_id

    log_event(state_path, "evt", {})
    top = query_events(state_path)[-1]["id"]
    assert max_event_id(state_path) == top

    # The "start my cursor at now" flow: nothing new yet.
    assert query_events(state_path, after_id=max_event_id(state_path)) == []
    log_event(state_path, "evt", {})
    assert len(query_events(state_path, after_id=top)) == 1


def test_cursor_missing_or_unreadable_db_never_raises(tmp_path: Path) -> None:
    """A missing or unreadable database returns empty/0, never raises."""
    # Missing entirely.
    state_path = tmp_path / "nonexistent_dir" / "state.json"
    assert query_events(state_path, after_id=0) == []

    # Present but not a database.
    state_path = tmp_path / "state.json"
    db_path = state_path.parent / "events.db"
    db_path.write_bytes(b"this is not a sqlite database")
    assert query_events(state_path, after_id=0) == []
    assert max_event_id(state_path) == 0
