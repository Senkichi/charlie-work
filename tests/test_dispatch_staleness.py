"""Tests for issue #946: staleness detector on dispatch cadence.

The detector reads events.db for the most recent ``dispatch`` event whose
payload ``issue_numbers`` is non-empty. When that event is older than a
configurable threshold and the unfiltered backlog is observed to be non-empty,
it returns a stale diagnostic that the dispatch path records as a
``dispatch_stale`` warning event.

``classify_backlog_reachability`` already answers the second half: the
``observed: False`` case must not be treated as "backlog empty".
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.config import DispatchConfig
from charlie_work.instrumentation import log_event, query_events
from charlie_work.workflow import check_dispatch_staleness


def _iso_now(now: datetime) -> str:
    return now.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_dispatch_event(state_path: Path, ts: str, issue_numbers: list[int]) -> None:
    """Write one dispatch event with a caller-chosen timestamp.

    ``log_event`` stamps real wall-clock time, so we insert normally and then
    backdate the row. This matches the pattern in other event-db tests.
    """
    log_event(state_path, "dispatch", {"issue_numbers": issue_numbers})
    db_path = state_path.parent / "events.db"
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute("SELECT MAX(id) FROM events WHERE kind = 'dispatch'")
        row = cursor.fetchone()
        if row and row[0]:
            conn.execute("UPDATE events SET ts = ? WHERE id = ?", (ts, row[0]))
            conn.commit()
    finally:
        conn.close()


def _backlog(*, nonempty: bool, observed: bool = True) -> dict[str, object]:
    if not observed:
        return {"observed": False, "open_total": 0, "dispatchable": 0}
    return {"observed": True, "open_total": 1 if nonempty else 0, "dispatchable": 0}


def test_stale_when_last_nonempty_dispatch_exceeds_threshold(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    now = datetime.now(UTC).replace(microsecond=0)
    threshold_minutes = 60
    config = DispatchConfig(dispatch_staleness_minutes=threshold_minutes)
    old = _iso_now(now - timedelta(minutes=threshold_minutes + 10))
    _write_dispatch_event(state_path, old, [1, 2])

    result = check_dispatch_staleness(state_path, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is True
    assert result["age_seconds"] == (threshold_minutes + 10) * 60
    assert result["threshold_seconds"] == threshold_minutes * 60
    assert result["last_dispatch_at"] == old
    assert result["last_dispatch_issue_numbers"] == [1, 2]
    assert result["backlog_observed"] is True
    assert result["backlog_open_total"] == 1


def test_not_stale_when_last_nonempty_dispatch_within_threshold(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    now = datetime.now(UTC).replace(microsecond=0)
    threshold_minutes = 60
    config = DispatchConfig(dispatch_staleness_minutes=threshold_minutes)
    recent = _iso_now(now - timedelta(minutes=threshold_minutes - 10))
    _write_dispatch_event(state_path, recent, [3])

    result = check_dispatch_staleness(state_path, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is False
    assert result["age_seconds"] == (threshold_minutes - 10) * 60
    assert result["last_dispatch_at"] == recent


def test_not_stale_when_no_baseline(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)

    result = check_dispatch_staleness(state_path, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is False
    assert result["last_dispatch_at"] is None
    assert result["reason"] == "no_baseline"


def test_empty_issue_numbers_dispatch_events_are_ignored(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)
    old = _iso_now(now - timedelta(minutes=70))
    _write_dispatch_event(state_path, old, [1])
    empty = _iso_now(now - timedelta(minutes=10))
    _write_dispatch_event(state_path, empty, [])

    result = check_dispatch_staleness(state_path, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is True
    assert result["age_seconds"] == 70 * 60
    assert result["last_dispatch_at"] == old


def test_most_recent_nonempty_dispatch_wins_past_empty_events(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)
    old = _iso_now(now - timedelta(minutes=90))
    _write_dispatch_event(state_path, old, [1])
    recent = _iso_now(now - timedelta(minutes=30))
    _write_dispatch_event(state_path, recent, [2])
    newer_empty = _iso_now(now - timedelta(minutes=5))
    _write_dispatch_event(state_path, newer_empty, [])

    result = check_dispatch_staleness(state_path, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is False
    assert result["age_seconds"] == 30 * 60
    assert result["last_dispatch_at"] == recent


def test_not_stale_when_backlog_not_observed(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)
    old = _iso_now(now - timedelta(minutes=90))
    _write_dispatch_event(state_path, old, [1])

    result = check_dispatch_staleness(
        state_path,
        config,
        _backlog(nonempty=True, observed=False),
        now=now,
    )

    assert result["stale"] is False
    assert result["reason"] == "backlog_not_observed"


def test_not_stale_when_backlog_genuinely_empty(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)
    old = _iso_now(now - timedelta(minutes=90))
    _write_dispatch_event(state_path, old, [1])

    result = check_dispatch_staleness(
        state_path,
        config,
        _backlog(nonempty=False),
        now=now,
    )

    assert result["stale"] is False
    assert result["backlog_open_total"] == 0
    assert result["reason"] == "empty_backlog"


def test_zero_threshold_disables(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=0)
    old = _iso_now(now - timedelta(minutes=90))
    _write_dispatch_event(state_path, old, [1])

    result = check_dispatch_staleness(state_path, config, _backlog(nonempty=True), now=now)

    assert result["stale"] is False
    assert result["threshold_seconds"] == 0


def test_current_pass_dispatched_short_circuits(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    now = datetime.now(UTC).replace(microsecond=0)
    config = DispatchConfig(dispatch_staleness_minutes=60)
    old = _iso_now(now - timedelta(minutes=90))
    _write_dispatch_event(state_path, old, [1])

    result = check_dispatch_staleness(
        state_path,
        config,
        _backlog(nonempty=True),
        now=now,
        recent_issue_numbers=[2],
    )

    assert result["stale"] is False
    assert result["age_seconds"] == 0
    assert result["last_dispatch_at"] == _iso_now(now)


def test_dispatch_stale_event_classified_as_warning(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    log_event(state_path, "dispatch_stale", {"age_seconds": 123})

    events = query_events(state_path, kind="dispatch_stale")

    assert len(events) == 1
    assert events[0]["level"] == "warning"
