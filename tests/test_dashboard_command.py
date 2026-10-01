# ruff: noqa: F811  (the imported ``fleet`` fixture is re-bound as a test parameter)
"""``charlie dashboard {rollup,now}`` through the real ``cli.main`` (tmp fleet dir)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest
from _dashboard_rollup_fixtures import ALPHA, BETA, fleet  # noqa: F401  (pytest fixture)

from charlie_work import cli
from charlie_work.dashboard.timeutil import local_iso


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    rc = cli.main(["--json", *argv])
    captured = capsys.readouterr()
    assert captured.out, captured.err
    return rc, json.loads(captured.out)


def test_rollup_prints_result_and_writes_only_dashboard_db(fleet, capsys) -> None:
    rc, out = _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", "rollup")
    assert rc == 0 and out["ok"] is True
    data = out["data"]
    assert data["ingested"] == 24  # 18 alpha + 2 beta + 4 fleet handled events
    assert data["errors"] == [] and data["error"] is None and data["db_rebuilt"] is False
    assert {s["source"]: s["ingested"] for s in data["sources"]} == {
        ALPHA: 18,
        BETA: 2,
        "fleet": 4,
    }
    conn = sqlite3.connect(fleet.sources().db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM pass_samples").fetchone() == (2,)
    finally:
        conn.close()


def test_now_builds_model_from_local_sources_without_findings(fleet, capsys) -> None:
    rc, out = _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", "now")
    assert rc == 0 and out["ok"] is True
    data = out["data"]
    assert len(data["freshness"]) == 2  # both registered repos, snapshots unreadable
    assert all(item.get("kind") != "alarm" for item in data["needs_me"])
    assert data["generated_at"].endswith("+00:00")


def test_disabled_kill_switch_refuses_both_commands(fleet, capsys) -> None:
    (fleet.dir / "config.yaml").write_text("dashboard:\n  enabled: false\n", encoding="utf-8")
    for sub in ("rollup", "now"):
        rc, out = _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", sub)
        assert rc == 1 and out["ok"] is False
        assert out["message"] == (
            "dashboard is disabled (dashboard.enabled: false in the fleet config)"
        )
    assert not fleet.sources().db_path.exists()  # refusal happens before any write


def test_invalid_dashboard_config_is_an_error_value(fleet, capsys) -> None:
    (fleet.dir / "config.yaml").write_text("dashboard:\n  port: 0\n", encoding="utf-8")
    rc, out = _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", "now")
    assert rc == 1 and out["ok"] is False
    assert out["message"].startswith("dashboard config error: dashboard.port: ")


def test_history_without_a_rollup_is_an_error_value(fleet, capsys) -> None:
    rc, out = _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", "history")
    assert rc == 1 and out["ok"] is False
    assert out["message"].startswith("dashboard.db unavailable (missing: ")


def test_history_after_rollup_returns_every_tab_with_takeaways(fleet, capsys) -> None:
    assert _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", "rollup")[0] == 0
    rc, out = _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", "history", "--days", "30")
    assert rc == 0 and out["ok"] is True
    tabs = out["data"]["tabs"]
    assert sorted(tabs) == ["Capacity", "Flow", "Quality", "Reliability"]
    merges = tabs["Flow"]["merges_per_day"][0]
    assert merges["name"] == "merges_per_day" and isinstance(merges["takeaway"], str)
    assert merges["window_end"] > merges["window_start"]


def test_history_rejects_nonpositive_window(fleet, capsys) -> None:
    rc, out = _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", "history", "--days", "0")
    assert rc == 1 and out["message"] == "--days and --bucket-hours must be at least 1"


def test_local_iso_renders_the_same_instant_in_the_given_zone() -> None:
    pdt = timezone(timedelta(hours=-7))
    assert local_iso("2026-10-01T03:30:00Z", pdt) == "2026-09-30T20:30:00-07:00"
    assert local_iso("2026-10-01T03:30:00Z", UTC) == "2026-10-01T03:30:00+00:00"


def test_history_carries_local_bucket_times_alongside_utc(fleet, capsys) -> None:
    assert _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", "rollup")[0] == 0
    _, out = _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", "history", "--days", "30")
    checked = 0
    for metrics in out["data"]["tabs"].values():
        for series_list in metrics.values():
            for s in series_list:
                assert [p[1] for p in s["points_local"]] == [p[1] for p in s["points"]]
                for (utc, _), (local, _) in zip(s["points"], s["points_local"], strict=True):
                    expected = datetime.fromisoformat(utc.replace("Z", "+00:00")).astimezone()
                    assert local == expected.isoformat()
                    checked += 1
    assert checked > 0  # the comparison above ran against real points, not an empty catalogue
