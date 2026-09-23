"""Supervisor and loop-pass freshness tests for ``scripts/heartbeat_check.py``.

Split out of ``tests/test_heartbeat_check.py`` (issue #1556, Track-1):
``check_log_freshness``, ``check_supervisor_heartbeat`` (issue #627),
and ``check_loop_pass_freshness`` events.db recency checks.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import (
    _iso,
    _load_heartbeat_check,
    _make_repo,
    _write_events_db,
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


def test_check_log_freshness_ok_when_fresh(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    (repo.state_dir / "run.log").write_text("hi", encoding="utf-8")
    report = hb.Report()
    hb.check_log_freshness(report, repo)
    assert not report.anomaly
    assert "run.log" in report.lines[0]


def test_check_log_freshness_anomaly_when_no_files(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    # state_dir exists but contains no log/state/checkpoint files
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    report = hb.Report()
    hb.check_log_freshness(report, repo)
    assert report.anomaly
    assert "no log/state/checkpoint files" in report.lines[0]


def test_check_log_freshness_anomaly_when_stale(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    stale_path = repo.state_dir / "run.log"
    stale_path.write_text("hi", encoding="utf-8")
    old_time = (datetime(2020, 1, 1, tzinfo=timezone.utc)).timestamp()
    os.utime(stale_path, (old_time, old_time))
    report = hb.Report()
    hb.check_log_freshness(report, repo)
    assert report.anomaly
    assert "older than threshold" in report.lines[0]


# ---------------------------------------------------------------------------
# Supervisor heartbeat freshness (issue #627)
# ---------------------------------------------------------------------------


def _set_fleet_dir(hb: ModuleType, monkeypatch: Any, tmp_path: Path) -> Path:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path))
    return tmp_path


def _write_heartbeat(
    hb: ModuleType,
    fleet_dir: Path,
    *,
    last_beat_at: str,
    exited_at: str | None = None,
    pid: int = 12345,
    full_pass_interval_seconds: int = 300,
    max_pass_runtime_seconds: int = 0,
) -> None:
    payload = {
        "pid": pid,
        # Use the beat time as the start time; check_supervisor_heartbeat only
        # reads last_beat_at/exited_at, so this keeps the fixture date-free.
        "started_at": last_beat_at,
        "last_beat_at": last_beat_at,
        "pass_number": 5,
        "full_pass_interval_seconds": full_pass_interval_seconds,
        "max_pass_runtime_seconds": max_pass_runtime_seconds,
        "exited_at": exited_at,
        "exit_code": 0 if exited_at else None,
    }
    (fleet_dir / hb.SUPERVISOR_HEARTBEAT_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_check_supervisor_heartbeat_anomaly_when_absent(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    _set_fleet_dir(hb, monkeypatch, tmp_path)
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert report.anomaly
    assert "no supervisor-heartbeat.json found" in report.lines[0]


def test_check_supervisor_heartbeat_ok_when_fresh(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    last_beat = (now - timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
    _write_heartbeat(hb, fleet_dir, last_beat_at=last_beat, full_pass_interval_seconds=300)
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert not report.anomaly
    assert "supervisor-heartbeat" in report.lines[0]


def test_check_supervisor_heartbeat_anomaly_when_stale_no_exit(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A stale heartbeat with no exited_at means the supervisor was likely killed."""
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    # 30 minutes old, threshold is 2*300s = 10 minutes.
    now = datetime.now(timezone.utc)
    last_beat = (now - timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    _write_heartbeat(
        hb, fleet_dir, last_beat_at=last_beat, exited_at=None, full_pass_interval_seconds=300
    )
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert report.anomaly
    assert "likely killed or hung" in report.lines[0]


def test_check_supervisor_heartbeat_anomaly_when_stale_with_clean_exit(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A stale heartbeat with exited_at set means the watchdog did not restart it."""
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    last_beat = (now - timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    _write_heartbeat(
        hb,
        fleet_dir,
        last_beat_at=last_beat,
        exited_at=last_beat,
        full_pass_interval_seconds=300,
    )
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert report.anomaly
    assert "watchdog may be disabled" in report.lines[0]


def test_check_supervisor_heartbeat_anomaly_on_corrupt_file(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    (fleet_dir / hb.SUPERVISOR_HEARTBEAT_FILENAME).write_text("{not json", encoding="utf-8")
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert report.anomaly
    assert "unreadable" in report.lines[0]


def test_check_supervisor_heartbeat_threshold_derives_from_max_pass_runtime(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A longer max_pass_runtime_seconds raises the stale threshold."""
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    # 20 minutes old. With max_pass_runtime=600s, threshold = 2*600/60 = 20 min, so OK.
    last_beat = (now - timedelta(minutes=19)).isoformat().replace("+00:00", "Z")
    _write_heartbeat(
        hb,
        fleet_dir,
        last_beat_at=last_beat,
        full_pass_interval_seconds=300,
        max_pass_runtime_seconds=600,
    )
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert not report.anomaly


def test_check_supervisor_heartbeat_uses_max_pass_runtime_not_full_pass_interval(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """The stale threshold is keyed on max_pass_runtime_seconds, not full_pass_interval_seconds.

    This is the M1 fix: a short full_pass_interval_seconds (the fallback pass
    cadence) does not bound a pass's wall-clock runtime, so a long-running but
    live supervisor must not be flagged as killed.
    """
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    # 15 minutes old. full_pass=300s would give a 10-minute threshold and flag
    # this as stale, but max_pass_runtime=1800s gives a 60-minute threshold.
    last_beat = (now - timedelta(minutes=15)).isoformat().replace("+00:00", "Z")
    _write_heartbeat(
        hb,
        fleet_dir,
        last_beat_at=last_beat,
        full_pass_interval_seconds=300,
        max_pass_runtime_seconds=1800,
    )
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert not report.anomaly


def test_check_loop_pass_freshness_anomaly_when_stale_but_log_fresh(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Regression for the log-fresh-but-loop-stale failure shape (#851/#854).

    The #851/#854 outage made the supervisor exit immediately every ~5min
    for a "watchdog restart" while doing ZERO repo passes; state.json kept
    getting touched every beat even though no real loop pass ran, so
    `check_log_freshness` read healthy the entire time. That outage was
    only ~45 minutes -- shorter than charlie-work's own measured healthy
    worst-case gap between loop passes (53.9m), so `LOOP_PASS_STALE_MINUTES`
    is deliberately set to 90 and this check no longer catches that specific
    45-minute magnitude (PR #865 / issue #855 does, via consecutive
    zero-repo-pass cycles instead of elapsed time). What this test still
    pins down is the mechanism: a gap comfortably past the 90m threshold,
    with a fresh log, must still trip this check -- proving
    `check_loop_pass_freshness` is not itself fooled by the fresh-log
    artifact that fooled `check_log_freshness` during the real incident.
    """
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    (repo.state_dir / "state.json").write_text("{}", encoding="utf-8")
    _write_events_db(repo.state_dir, [(_iso(120), "loop_started")])

    log_report = hb.Report()
    hb.check_log_freshness(log_report, repo)
    assert not log_report.anomaly, "log freshness must read healthy, matching the real outage"

    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert report.anomaly
    assert "loop_started" in report.lines[0]


def test_check_loop_pass_freshness_ok_when_recent(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(5), "loop_started")])
    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert not report.anomaly
    assert "newest_loop_started=" in report.lines[0]


def test_check_loop_pass_freshness_ok_at_measured_healthy_worst_case(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Pins the false-alarm fix: 54m must NOT trip the 90m threshold.

    Measured production `loop_started` gaps (charlie-work, 39 intervals):
    max=53.9m. At the original LOOP_PASS_STALE_MINUTES=30 this magnitude of
    gap fired on a healthy fleet (~3-4 false alarms/day); at 90 it must not.
    If this threshold is ever lowered back toward 30-45 without addressing
    the repo-ordering cause described on LOOP_PASS_STALE_MINUTES, this test
    goes red before the false alarms return to production.
    """
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(54), "loop_started")])
    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert not report.anomaly, report.lines


def test_check_loop_pass_freshness_ok_when_db_missing(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert not report.anomaly
    assert "no events.db" in report.lines[0]


def test_check_loop_pass_freshness_ok_when_table_missing(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    db_path = repo.state_dir / "events.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
        conn.commit()
    finally:
        conn.close()

    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert not report.anomaly
    assert "no events table" in report.lines[0]


def test_check_loop_pass_freshness_ok_when_zero_loop_started_rows(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    # events table exists and has rows, but none of kind loop_started --
    # must be distinguishable from both "no table" and "stale".
    _write_events_db(repo.state_dir, [(_iso(1), "dispatch")])
    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert not report.anomaly
    assert "no loop_started rows" in report.lines[0]


def test_check_loop_pass_freshness_recent_iso_row_not_misjudged_stale(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Positive control for the ISO-vs-SQLite string-comparison trap.

    `ts` values are `...THH:MM:SSZ`. SQLite's `datetime('now','-90
    minutes')` returns a space-separated, non-`Z` string like
    `2026-07-31 22:25:04`. A predicate such as
    `WHERE ts < datetime('now','-90 minutes')` string-compares these, and
    `'T'` (0x54) sorting after `' '` (0x20) makes the comparison
    unreliable in either direction. This row is genuinely 2 minutes old and
    must read as fresh; if the SQL-based comparison is ever reintroduced in
    place of the Python-side `parse_iso` + timedelta comparison, this test
    must go red.
    """
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(2), "loop_started")])
    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)


# ---------------------------------------------------------------------------
# Wedge-kill loop detection (issue #1832)
# ---------------------------------------------------------------------------


def test_check_wedge_kill_loop_ok_when_no_fleet_events_db(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """No fleet events.db yet (fresh install) is OK, not an anomaly."""
    _set_fleet_dir(hb, monkeypatch, tmp_path)
    report = hb.Report()
    hb.check_wedge_kill_loop(report)
    assert not report.anomaly


def test_check_wedge_kill_loop_ok_when_no_wedge_loop_event(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A fleet events.db with unrelated events (no supervisor_wedge_loop) is OK."""
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    _write_events_db(
        fleet_dir,
        [
            (_iso(5), "supervisor_started"),
            (_iso(4), "fleet_pass_completed"),
            (_iso(3), "supervisor_wedged_killed", "error"),
        ],
    )
    report = hb.Report()
    hb.check_wedge_kill_loop(report)
    assert not report.anomaly


def test_check_wedge_kill_loop_anomaly_when_recent_event(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A recent supervisor_wedge_loop event is its own distinct ANOMALY."""
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    _write_events_db(
        fleet_dir,
        [
            (_iso(120), "supervisor_wedged_killed", "error"),
            (_iso(90), "supervisor_wedged_killed", "error"),
            (_iso(60), "supervisor_wedged_killed", "error"),
            (_iso(30), "supervisor_wedge_loop", "error"),
        ],
    )
    report = hb.Report()
    hb.check_wedge_kill_loop(report)
    assert report.anomaly
    assert "supervisor_wedge_loop" in report.lines[0]
    assert "wedge-kill backstop is looping" in report.lines[0]


def test_check_wedge_kill_loop_ignores_event_outside_lookback_window(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A supervisor_wedge_loop event older than the lookback window does not alarm forever."""
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    old_ts = _iso(60 * (hb.SUPERVISOR_WEDGE_LOOP_LOOKBACK_HOURS + 1))
    _write_events_db(fleet_dir, [(old_ts, "supervisor_wedge_loop", "error")])
    report = hb.Report()
    hb.check_wedge_kill_loop(report)
    assert not report.anomaly


def test_check_wedge_kill_loop_anomaly_on_unreadable_db(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    fleet_dir.mkdir(parents=True, exist_ok=True)
    (fleet_dir / "events.db").write_text("not a sqlite file", encoding="utf-8")
    report = hb.Report()
    hb.check_wedge_kill_loop(report)
    assert report.anomaly
    assert "unreadable" in report.lines[0]
