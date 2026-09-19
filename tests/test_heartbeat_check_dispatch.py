"""Dispatch-throttle/coverage/failures tests for ``scripts/heartbeat_check.py``.

Split out of ``tests/test_heartbeat_check.py`` (issue #1556, Track-1):
the ``state_file`` env-override smoke tests, ``save_state``/``load_state``
round-trip, ``check_dispatch_throttle``, ``check_dispatch_coverage``
(drain-at-cap and governor-free-slots paths included), and
``check_dispatch_failures``.
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
    _gh_dispatch,
    _iso,
    _load_heartbeat_check,
    _make_repo,
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


# ---------------------------------------------------------------------------
# Smoke tests for the remaining checks (review-liveness is covered above).
# Each test exercises one check's OK and/or anomaly path with stubbed I/O.
# ---------------------------------------------------------------------------


def test_state_file_respects_env_override(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """CHARLIE_WORK_HEARTBEAT_STATE overrides the derived fleet_dir path."""
    custom = tmp_path / "custom-state.json"
    monkeypatch.setenv("CHARLIE_WORK_HEARTBEAT_STATE", str(custom))
    assert hb.state_file() == custom


def test_state_file_derives_from_fleet_dir(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Without an explicit override, state_file() follows CHARLIE_WORK_FLEET_DIR."""
    monkeypatch.delenv("CHARLIE_WORK_HEARTBEAT_STATE", raising=False)
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path))
    assert hb.state_file() == tmp_path / "heartbeat-state.json"


def test_save_and_load_state_round_trip(hb: ModuleType, monkeypatch: Any, tmp_path: Path) -> None:
    """save_state writes atomically and load_state reads it back."""
    monkeypatch.setenv("CHARLIE_WORK_HEARTBEAT_STATE", str(tmp_path / "hb.json"))
    payload = {"last_beat_at": "2026-07-22T00:00:00Z", "repos": {}}
    hb.save_state(payload)
    assert hb.load_state() == payload


def test_check_dispatch_throttle_ok_when_no_state(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    report = hb.Report()
    hb.check_dispatch_throttle(report, repo)
    assert not report.anomaly
    assert "none (no state.json)" in report.lines[0]


def test_check_dispatch_throttle_ok_when_not_throttled(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    (repo.state_dir / "state.json").write_text(
        json.dumps({"throttled_until": None}), encoding="utf-8"
    )
    report = hb.Report()
    hb.check_dispatch_throttle(report, repo)
    assert not report.anomaly
    assert "none" in report.lines[0]


def test_check_dispatch_throttle_ok_within_threshold(hb: ModuleType, tmp_path: Path) -> None:
    """Regression for issue #828: the fixture write and the check call are back-to-back
    statements, but a substring assertion against text that flips at the 10-minute
    remaining-time boundary is still exposed on a sufficiently long stall. Freeze `now`
    so `until` and `resolved_now` never drift apart regardless of scheduling delay.
    """
    frozen_now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    # _iso(-10, base=frozen_now) = 10 minutes ahead of frozen_now (throttle still active)
    (repo.state_dir / "state.json").write_text(
        json.dumps({"throttled_until": _iso(-10, base=frozen_now)}), encoding="utf-8"
    )
    report = hb.Report()
    hb.check_dispatch_throttle(report, repo, now=frozen_now)
    assert not report.anomaly
    assert "throttled until" in report.lines[0]


def test_check_dispatch_throttle_anomaly_when_exceeds_threshold(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Regression for issue #828: same class as the "within threshold" test above,
    frozen against the 30-minute anomaly boundary instead of the 10-minute one.
    """
    frozen_now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    # _iso(-60, base=frozen_now) = 60 minutes ahead of frozen_now, beyond the 30-min threshold
    (repo.state_dir / "state.json").write_text(
        json.dumps({"throttled_until": _iso(-60, base=frozen_now)}), encoding="utf-8"
    )
    report = hb.Report()
    hb.check_dispatch_throttle(report, repo, now=frozen_now)
    assert report.anomaly
    assert "cooldown exceeds threshold" in report.lines[0]


def test_check_dispatch_coverage_ok_when_no_dispatchable(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (True, [], ""),
    )
    report = hb.Report()
    hb.check_dispatch_coverage(
        report, repo, {}, {}, skip_delta=False, blocked_numbers=None, blocked_err=""
    )
    assert not report.anomaly
    assert "dispatch-coverage" in report.lines[0]


def test_check_dispatch_coverage_anomaly_when_persisting(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    issues = [
        {"number": 42, "labels": [], "updatedAt": _iso(1)},
    ]
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (True, issues, ""),
    )
    prev = {"dispatchable_issues": [42]}
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_dispatch_coverage(
        report, repo, prev, new, skip_delta=False, blocked_numbers=None, blocked_err=""
    )
    assert report.anomaly
    assert "dispatchable across 2 consecutive beats" in report.lines[0]


def test_check_dispatch_coverage_ok_when_degraded_but_empty(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Degraded blocked lookup with no dispatchable issues is a sound OK.

    The blocked set is unavailable, but that can only *inflate* dispatchable,
    so an empty dispatchable set cannot be a false negative. The OK line must
    still note the degraded lookup so a reader does not infer an empty fleet.
    """
    repo = _make_repo(hb, tmp_path)
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (True, [], ""),
    )
    report = hb.Report()
    hb.check_dispatch_coverage(
        report,
        repo,
        {},
        {},
        skip_delta=False,
        blocked_numbers=None,
        blocked_err="charlie fleet status --json timed out",
    )
    assert not report.anomaly
    assert "OK dispatch-coverage" in report.lines[0]
    assert "result is sound" in report.lines[0]
    assert "charlie fleet status --json timed out" in report.lines[0]


def test_check_dispatch_coverage_anomaly_possibly_spurious_when_degraded(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Degraded blocked lookup makes a dispatchable-persisting anomaly suspect.

    A dispatchable issue may actually be blocked; the anomaly must carry a
    caveat rather than read as a confirmed dispatch failure.
    """
    repo = _make_repo(hb, tmp_path)
    issues = [
        {"number": 42, "labels": [], "updatedAt": _iso(1)},
    ]
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (True, issues, ""),
    )
    prev = {"dispatchable_issues": [42]}
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_dispatch_coverage(
        report,
        repo,
        prev,
        new,
        skip_delta=False,
        blocked_numbers=None,
        blocked_err="blocked-issue lookup failed",
    )
    assert report.anomaly
    assert "possibly-spurious" in report.lines[0]
    assert "blocked-issue lookup failed" in report.lines[0]
    assert "dispatchable across 2 consecutive beats" in report.lines[0]


# ---------------------------------------------------------------------------
# check_dispatch_coverage drain-at-cap (issue #1424)
# ---------------------------------------------------------------------------


def _write_dispatch_event(
    state_dir: Path,
    *,
    concurrency_governor: dict[str, Any],
    deferred_by_concurrency_count: int = 0,
    ts: str | None = None,
) -> Path:
    """Seed a ``dispatch`` event row in ``state_dir/events.db``.

    Mirrors the payload shape ``workflow.py``'s dispatch path writes: a
    top-level ``concurrency_governor`` sub-dict (with ``dispatch_limit``,
    ``fleet_concurrency_limit`` / ``fleet_live_session_count`` when the fleet
    governor is enabled) and a sibling ``deferred_by_concurrency_count``.
    Uses the production ``events`` schema by hand rather than importing
    ``charlie_work.instrumentation``, matching ``_write_events_db``.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "events.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT    NOT NULL,
                kind            TEXT    NOT NULL,
                payload         TEXT    NOT NULL,
                repo            TEXT,
                correlation_id  TEXT,
                pr_number       INTEGER,
                issue_number    INTEGER,
                level           TEXT DEFAULT 'info'
            )
            """
        )
        payload = json.dumps(
            {
                "concurrency_governor": concurrency_governor,
                "deferred_by_concurrency_count": deferred_by_concurrency_count,
            }
        )
        conn.execute(
            "INSERT INTO events (ts, kind, payload, level) VALUES (?, 'dispatch', ?, 'info')",
            (ts or _iso(1), payload),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_check_dispatch_coverage_drain_note_when_fleet_at_cap(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #1424: fleet-wide cap saturated -> drain note, not anomaly.

    The repo is at 2/5 in-progress (below its per-repo cap of 5), but the
    fleet governor shows fleet_live=5/fleet_cap=5 -- the sibling repo holds
    the other slots. The old per-repo denominator read this as a dispatch
    failure; the governor-based condition recognises it as designed drain.
    """
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_text("dispatch:\n  max_concurrent_sessions: 5\n", encoding="utf-8")
    issues = [
        {"number": 1401, "labels": [], "updatedAt": _iso(1)},
        {"number": 1402, "labels": [], "updatedAt": _iso(1)},
    ]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    _write_dispatch_event(
        repo.state_dir,
        concurrency_governor={
            "clamped": True,
            "dispatch_limit": 0,
            "concurrency_limit": 5,
            "live_session_count": 2,
            "available_slots": 3,
            "fleet_concurrency_limit": 5,
            "fleet_live_session_count": 5,
        },
        deferred_by_concurrency_count=4,
    )
    prev = {"dispatchable_issues": [1401, 1402]}
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_dispatch_coverage(
        report, repo, prev, new, skip_delta=False, blocked_numbers=None, blocked_err=""
    )
    assert not report.anomaly
    line = report.lines[0]
    assert "draining at cap" in line
    assert "fleet_live=5/fleet_cap=5" in line
    assert "dispatchable across 2 consecutive beats" not in line


def test_check_dispatch_coverage_anomaly_when_governor_has_free_slots(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #1424: governor shows free slots and issues persist -> anomaly.

    The fleet has room (fleet_live=3/fleet_cap=5) and the last dispatch had
    dispatch_limit=5 with zero deferrals -- there was capacity but the issues
    were not picked up. This is the #1398 head-of-line case, which is real.
    """
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_text("dispatch:\n  max_concurrent_sessions: 5\n", encoding="utf-8")
    issues = [
        {"number": 1398, "labels": [], "updatedAt": _iso(1)},
    ]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    _write_dispatch_event(
        repo.state_dir,
        concurrency_governor={
            "clamped": False,
            "dispatch_limit": 5,
            "concurrency_limit": 5,
            "live_session_count": 2,
            "available_slots": 3,
            "fleet_concurrency_limit": 5,
            "fleet_live_session_count": 3,
        },
        deferred_by_concurrency_count=0,
    )
    prev = {"dispatchable_issues": [1398]}
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_dispatch_coverage(
        report, repo, prev, new, skip_delta=False, blocked_numbers=None, blocked_err=""
    )
    assert report.anomaly
    assert "dispatchable across 2 consecutive beats" in report.lines[0]
    assert "draining at cap" not in report.lines[0]


def test_check_dispatch_coverage_drain_note_when_dispatch_limit_at_deferred(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #1424: dispatch_limit <= deferred_by_concurrency_count -> drain note.

    The fleet governor is not enabled (no fleet_concurrency_limit in the
    payload), but the effective dispatch_limit was 1 with 4 deferred issues
    -- the backlog exceeds the per-pass dispatch rate, so the backlog is
    draining at the allowed rate.
    """
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_text("dispatch:\n  max_concurrent_sessions: 5\n", encoding="utf-8")
    issues = [
        {"number": 1401, "labels": [], "updatedAt": _iso(1)},
        {"number": 1402, "labels": [], "updatedAt": _iso(1)},
    ]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    _write_dispatch_event(
        repo.state_dir,
        concurrency_governor={
            "clamped": True,
            "dispatch_limit": 1,
            "concurrency_limit": 5,
            "live_session_count": 4,
            "available_slots": 1,
        },
        deferred_by_concurrency_count=4,
    )
    prev = {"dispatchable_issues": [1401, 1402]}
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_dispatch_coverage(
        report, repo, prev, new, skip_delta=False, blocked_numbers=None, blocked_err=""
    )
    assert not report.anomaly
    line = report.lines[0]
    assert "draining at cap" in line
    assert "dispatch_limit=1" in line
    assert "dispatchable across 2 consecutive beats" not in line


def test_check_dispatch_coverage_fallback_per_repo_cap_when_no_events_db(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #1424: no events.db -> fall back to the per-repo cap check.

    A fresh install with no dispatch events yet cannot provide governor data.
    The existing per-repo cap drain condition is preserved as a conservative
    fallback so the check does not regress on repos without instrumentation.
    """
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_text("dispatch:\n  max_concurrent_sessions: 3\n", encoding="utf-8")
    # 3 in-progress issues fill the per-repo cap; 2 dispatchable persist.
    issues = [
        {"number": 10, "labels": [{"name": "agent:in-progress"}], "updatedAt": _iso(1)},
        {"number": 11, "labels": [{"name": "agent:in-progress"}], "updatedAt": _iso(1)},
        {"number": 12, "labels": [{"name": "agent:in-progress"}], "updatedAt": _iso(1)},
        {"number": 20, "labels": [], "updatedAt": _iso(1)},
        {"number": 21, "labels": [], "updatedAt": _iso(1)},
    ]
    _gh_dispatch(monkeypatch, hb, lambda args, cwd: (True, issues, ""))
    # No events.db written -- _last_dispatch_drain_signal returns None.
    prev = {"dispatchable_issues": [20, 21]}
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_dispatch_coverage(
        report, repo, prev, new, skip_delta=False, blocked_numbers=None, blocked_err=""
    )
    assert not report.anomaly
    line = report.lines[0]
    assert "draining at cap" in line
    assert "in_progress=3/cap=3" in line


def test_check_dispatch_failures_ok_when_no_dir(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    report = hb.Report()
    hb.check_dispatch_failures(report, repo, datetime.now(timezone.utc))
    assert not report.anomaly
    assert "scanned=0" in report.lines[0]


def test_check_dispatch_failures_anomaly_for_new_failure(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    dispatches = repo.state_dir / "dispatches"
    dispatches.mkdir(parents=True, exist_ok=True)
    (dispatches / "bad.json").write_text(json.dumps({"error": "boom"}), encoding="utf-8")
    baseline = datetime.now(timezone.utc) - timedelta(hours=1)
    report = hb.Report()
    hb.check_dispatch_failures(report, repo, baseline)
    assert report.anomaly
    assert "bad.json" in report.lines[0]


def test_check_dispatch_failures_ok_when_failure_before_baseline(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    dispatches = repo.state_dir / "dispatches"
    dispatches.mkdir(parents=True, exist_ok=True)
    old_path = dispatches / "old.json"
    old_path.write_text(json.dumps({"error": "boom"}), encoding="utf-8")
    old_time = (datetime(2020, 1, 1, tzinfo=timezone.utc)).timestamp()
    os.utime(old_path, (old_time, old_time))
    baseline = datetime.now(timezone.utc)
    report = hb.Report()
    hb.check_dispatch_failures(report, repo, baseline)
    assert not report.anomaly
