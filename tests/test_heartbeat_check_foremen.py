"""Foreman checks for ``scripts/heartbeat_check.py`` (issue #3123, design §10).

``check_foremen`` decodes the foreman store by file shape — ADR-0010 forbids
importing ``charlie_work.foreman`` from a script — so these tests seed that
shape directly (``foremen/<name>/state.json``, ``charter.yaml``, the
foreman's own ``events.db``, and the fleet-level ``events.db`` holding
``foreman_pass_completed``) and assert the five shared
``heartbeat_alarms_foreman`` verdicts land on the report.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

from _heartbeat_check_fixtures import _iso, _load_heartbeat_check, _write_events_db


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


def _fleet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path))
    return tmp_path


def _foreman(
    fleet: Path,
    name: str = "alpha",
    *,
    state: dict | None = None,
    charter: str | None = None,
) -> Path:
    """Create ``foremen/<name>/`` with state.json (and optionally charter.yaml)."""
    fdir = fleet / "foremen" / name
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / "state.json").write_text(
        json.dumps(state if state is not None else {"state": "idle"}),
        encoding="utf-8",
    )
    if charter is not None:
        (fdir / "charter.yaml").write_text(charter, encoding="utf-8")
    return fdir


def _turn_rows(fdir: Path, tokens: list[int]) -> None:
    """Seed ``foreman_turn_completed`` rows whose payloads carry token totals."""
    conn = sqlite3.connect(str(fdir / "events.db"))
    try:
        conn.execute(
            """
            CREATE TABLE events (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      TEXT NOT NULL,
                kind    TEXT NOT NULL,
                payload TEXT NOT NULL,
                level   TEXT DEFAULT 'info'
            )
            """
        )
        for index, total in enumerate(tokens):
            payload = json.dumps({"input_tokens": total, "output_tokens": 0})
            conn.execute(
                "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
                (_iso(index - len(tokens)), "foreman_turn_completed", payload),
            )
        conn.commit()
    finally:
        conn.close()


def _completed_pass(fleet: Path, minutes_ago: float = 1.0) -> None:
    _write_events_db(fleet, [(_iso(minutes_ago), "foreman_pass_completed")])


def test_check_foremen_ok_when_no_foremen(hb: ModuleType, monkeypatch, tmp_path) -> None:
    _fleet(monkeypatch, tmp_path)
    report = hb.Report()
    hb.check_foremen(report)
    assert not report.anomaly
    assert report.lines == ["OK foremen: no foremen under foremen/ — foreman checks skipped"]


def test_check_foremen_ignores_non_foreman_dirs(hb: ModuleType, monkeypatch, tmp_path) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    for stray in ("_shared", "UPPER", ".hidden"):
        (fleet / "foremen" / stray).mkdir(parents=True)
    report = hb.Report()
    hb.check_foremen(report)
    assert not report.anomaly
    assert report.lines == ["OK foremen: no foremen under foremen/ — foreman checks skipped"]


def test_check_foremen_healthy(hb: ModuleType, monkeypatch, tmp_path) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    _foreman(fleet)
    _completed_pass(fleet)
    report = hb.Report()
    hb.check_foremen(report)
    assert not report.anomaly
    assert not any(line.startswith("WARN") for line in report.lines)
    assert any(line.startswith("OK foreman-pass-freshness") for line in report.lines)


def test_check_foremen_faulted_is_anomaly(hb: ModuleType, monkeypatch, tmp_path) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    _foreman(fleet, state={"state": "faulted"})
    _completed_pass(fleet)
    report = hb.Report()
    hb.check_foremen(report)
    assert report.anomaly
    assert any(line.startswith("ANOMALY foreman-state alpha") for line in report.lines)


def test_check_foremen_breaker_is_anomaly(hb: ModuleType, monkeypatch, tmp_path) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    _foreman(
        fleet,
        state={"state": "degraded", "degraded_reasons": ["breaker"]},
    )
    _completed_pass(fleet)
    report = hb.Report()
    hb.check_foremen(report)
    assert report.anomaly
    assert any(line.startswith("ANOMALY foreman-breaker alpha") for line in report.lines)
    # breaker-degraded is the breaker's own alarm — the state check stays OK.
    assert any(line.startswith("OK foreman-state alpha") for line in report.lines)


def test_check_foremen_non_breaker_degraded_is_warn(hb: ModuleType, monkeypatch, tmp_path) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    _foreman(
        fleet,
        state={"state": "degraded", "degraded_reasons": ["unattended"]},
    )
    _completed_pass(fleet)
    report = hb.Report()
    hb.check_foremen(report)
    # A warn never flips the anomaly flag — the limit working is not a fault.
    assert not report.anomaly
    assert any(line.startswith("WARN foreman-state alpha") for line in report.lines)


def test_check_foremen_turn_token_spike(hb: ModuleType, monkeypatch, tmp_path) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    fdir = _foreman(fleet)
    _turn_rows(fdir, [100] * 21 + [400])
    _completed_pass(fleet)
    report = hb.Report()
    hb.check_foremen(report)
    assert report.anomaly
    assert any(line.startswith("ANOMALY foreman-turn-tokens alpha") for line in report.lines)


def test_check_foremen_zero_median_stays_quiet(hb: ModuleType, monkeypatch, tmp_path) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    fdir = _foreman(fleet)
    _turn_rows(fdir, [0, 0, 500])
    _completed_pass(fleet)
    report = hb.Report()
    hb.check_foremen(report)
    assert not any(line.startswith("ANOMALY foreman-turn-tokens") for line in report.lines)


def test_check_foremen_turns_per_day_reached(hb: ModuleType, monkeypatch, tmp_path) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    _foreman(
        fleet,
        state={"state": "idle", "turns_today": 5, "turns_today_date": today},
        charter="bounds:\n  max_turns_per_day: 5\n",
    )
    _completed_pass(fleet)
    report = hb.Report()
    hb.check_foremen(report, now=now)
    assert report.anomaly
    assert any(line.startswith("ANOMALY foreman-turns-per-day alpha") for line in report.lines)


def test_check_foremen_turns_per_day_not_today(hb: ModuleType, monkeypatch, tmp_path) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    _foreman(
        fleet,
        state={"state": "idle", "turns_today": 99, "turns_today_date": "2000-01-01"},
        charter="bounds:\n  max_turns_per_day: 5\n",
    )
    _completed_pass(fleet)
    report = hb.Report()
    hb.check_foremen(report)
    assert not any(line.startswith("ANOMALY foreman-turns-per-day") for line in report.lines)


def test_check_foremen_stale_pass_is_anomaly(hb: ModuleType, monkeypatch, tmp_path) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    _foreman(fleet)
    _completed_pass(fleet, minutes_ago=120)  # default interval 5m -> threshold 10m
    report = hb.Report()
    hb.check_foremen(report)
    assert report.anomaly
    assert any(line.startswith("ANOMALY foreman-pass-freshness") for line in report.lines)


def test_check_foremen_never_completed_pass_is_anomaly(
    hb: ModuleType, monkeypatch, tmp_path
) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    _foreman(fleet)
    report = hb.Report()
    hb.check_foremen(report)
    assert report.anomaly
    assert any(
        line.startswith("ANOMALY foreman-pass-freshness") and "never run" in line
        for line in report.lines
    )


def test_check_foremen_disabled_skips_pass_freshness(
    hb: ModuleType, monkeypatch, tmp_path
) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    _foreman(fleet)
    (fleet / "config.yaml").write_text("foreman:\n  enabled: false\n", encoding="utf-8")
    report = hb.Report()
    hb.check_foremen(report)
    assert not report.anomaly
    assert any(
        line.startswith("OK foreman-pass-freshness") and "enabled is false" in line
        for line in report.lines
    )


def test_check_foremen_fleet_paused_skips_pass_freshness(
    hb: ModuleType, monkeypatch, tmp_path
) -> None:
    fleet = _fleet(monkeypatch, tmp_path)
    _foreman(fleet)
    (fleet / "fleet-pause.json").write_text("{}", encoding="utf-8")
    report = hb.Report()
    hb.check_foremen(report)
    assert not report.anomaly
    assert any(
        line.startswith("OK foreman-pass-freshness") and "fleet paused" in line
        for line in report.lines
    )
