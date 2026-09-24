"""Notify-digest freshness consumer tests for ``scripts/heartbeat_check.py``.

Issue #1859: the notify digest shipped as a signal without a consumer -- the
live daemon's file-sink writer was dead for three weeks while nothing read
the file. ``check_notify_digest_freshness`` is that consumer: it reads the
fleet ``events.db`` for the supervisor's own ``notify_resolution`` event --
what the daemon actually resolved, including the absolute digest path --
plus its ``notify_digest_stale`` tripwire, then probes the published path
read-only. It deliberately does NOT re-derive the notify: config from the
script's own checkout: on this host the script's checkout and the daemon's
config root are different trees (round-1 review, verified on host).
"""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import _iso, _load_heartbeat_check


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


@pytest.fixture()
def fleet_dir(tmp_path: Path, monkeypatch: Any) -> Path:
    """Point the script's fleet_dir() at a hermetic directory."""
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    return tmp_path / "fleet"


def _write_fleet_events(fleet_dir: Path, rows: list[tuple[str, str, str]]) -> Path:
    """Create a fleet events.db with (ts, kind, payload_json) rows.

    Mirrors the production ``events`` schema (same shape as
    ``_heartbeat_check_fixtures._write_events_db``, plus the payload column
    this check actually reads).
    """
    fleet_dir.mkdir(parents=True, exist_ok=True)
    db_path = fleet_dir / "events.db"
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
        for ts, kind, payload in rows:
            conn.execute(
                "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
                (ts, kind, payload),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _resolution_payload(**overrides: Any) -> str:
    """A ``notify_resolution`` payload as ``report_notify_resolution`` writes it."""
    payload: dict[str, Any] = {
        "enabled": True,
        "sink": "file",
        "file_path": ".var/charlie-work/notify/digest.jsonl",
        "resolved_file_path": None,
        "file_path_empty": False,
        "global_config_path": "C:/fleet/config.yaml",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _write_digest(path: Path, generated_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"generated_at": generated_at, "repo": "fleet", "transitions": []}
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")


def _resolution_row(
    hb: ModuleType, minutes_ago: float = 1.0, **overrides: Any
) -> tuple[str, str, str]:
    return (_iso(minutes_ago), hb.NOTIFY_RESOLUTION_EVENT_KIND, _resolution_payload(**overrides))


def test_notify_digest_warn_when_no_fleet_events_db(hb: ModuleType, fleet_dir: Path) -> None:
    """No events.db at all -> WARN (cannot tell what the daemon resolved),
    not OK and not an anomaly-flipping verdict."""
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert not report.anomaly
    assert report.lines[0].startswith("WARN notify-digest")
    assert "no fleet events.db" in report.lines[0]


def test_notify_digest_warn_when_no_resolution_event(hb: ModuleType, fleet_dir: Path) -> None:
    """An events.db without a notify_resolution row means a supervisor that
    predates this instrumentation or died before its startup report."""
    _write_fleet_events(fleet_dir, [(_iso(1), "loop_started", "{}")])
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert not report.anomaly
    assert report.lines[0].startswith("WARN notify-digest")
    assert "no notify_resolution event" in report.lines[0]


def test_notify_digest_warn_when_resolved_disabled(hb: ModuleType, fleet_dir: Path) -> None:
    """enabled=false in the daemon's own resolution: a fleet that never
    opted in must not flip the exit code, but a fleet that DID opt in and
    lost its notify: block shows up here -- the 2026-08-31 invisibility
    shape."""
    _write_fleet_events(fleet_dir, [_resolution_row(hb, enabled=False)])
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert not report.anomaly
    assert report.lines[0].startswith("WARN notify-digest")
    assert "enabled=false" in report.lines[0]


def test_notify_digest_anomaly_when_file_path_empty(hb: ModuleType, fleet_dir: Path) -> None:
    """enabled + sink=file + empty file_path is the incoherent combination
    every emit fails on -- now surfaced through the consumed resolution
    event, not only a supervisor-start log line."""
    _write_fleet_events(
        fleet_dir,
        [_resolution_row(hb, file_path="", resolved_file_path=None, file_path_empty=True)],
    )
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert report.anomaly
    assert "file_path is unset" in report.lines[0]


def test_notify_digest_anomaly_when_enabled_but_file_missing(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    """The 2026-08-31 shape: enabled file sink, no digest on disk."""
    digest = tmp_path / "daemon" / "notify" / "digest.jsonl"
    _write_fleet_events(fleet_dir, [_resolution_row(hb, resolved_file_path=str(digest))])
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert report.anomaly
    assert "does not exist" in report.lines[0]
    assert str(digest) in report.lines[0]


def test_notify_digest_anomaly_when_last_entry_stale(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    digest = tmp_path / "digest.jsonl"
    _write_fleet_events(fleet_dir, [_resolution_row(hb, resolved_file_path=str(digest))])
    old_ts = _iso(60 * (hb.NOTIFY_DIGEST_STALE_HOURS + 1))
    _write_digest(digest, old_ts)
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert report.anomaly
    assert "writer looks dead" in report.lines[0]


def test_notify_digest_ok_when_last_entry_fresh(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    digest = tmp_path / "digest.jsonl"
    _write_fleet_events(fleet_dir, [_resolution_row(hb, resolved_file_path=str(digest))])
    _write_digest(digest, _iso(5))
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert not report.anomaly
    assert report.lines[0].startswith("OK notify-digest")


def test_notify_digest_mtime_fallback_when_tail_unparseable(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    """A digest file with no parseable generated_at still gets a verdict
    from its mtime -- a file merely touched is not mistaken for live
    output."""
    digest = tmp_path / "digest.jsonl"
    _write_fleet_events(fleet_dir, [_resolution_row(hb, resolved_file_path=str(digest))])
    digest.write_text("{broken json\n", encoding="utf-8")
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert not report.anomaly  # mtime is fresh (just written)
    assert "mtime" in report.lines[0]


def test_notify_digest_non_string_generated_at_does_not_crash(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    """A non-string generated_at (round-1 review: unguarded parse_iso call)
    must degrade to the mtime fallback, not raise."""
    digest = tmp_path / "digest.jsonl"
    _write_fleet_events(fleet_dir, [_resolution_row(hb, resolved_file_path=str(digest))])
    digest.write_text(
        json.dumps({"generated_at": 1759000000, "transitions": []}) + "\n",
        encoding="utf-8",
    )
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert not report.anomaly
    assert "mtime" in report.lines[0]


def test_notify_digest_ok_for_non_file_sink(hb: ModuleType, fleet_dir: Path) -> None:
    _write_fleet_events(fleet_dir, [_resolution_row(hb, sink="webhook")])
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert not report.anomaly
    assert "no digest file" in report.lines[0]


def test_notify_digest_follows_supervisor_resolution_not_script_checkout(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    """The review's core complaint: the verdict must observe what the
    supervisor resolved, not what the script's own checkout would derive.

    Plant a STALE digest at the daemon-resolved path and a FRESH decoy at
    the same relative location under a directory standing in for the
    script's checkout: the old code probed the script's tree and would have
    reported OK; the consumed resolution points at the daemon's file, so
    the verdict must be ANOMALY on the daemon path.
    """
    daemon_dir = tmp_path / "daemon-checkout"
    script_dir = tmp_path / "script-checkout"
    rel = Path(".var") / "charlie-work" / "notify" / "digest.jsonl"
    daemon_digest = daemon_dir / rel
    script_digest = script_dir / rel
    _write_digest(script_digest, _iso(1))  # fresh decoy under the script's tree
    _write_digest(
        daemon_digest, _iso(60 * (hb.NOTIFY_DIGEST_STALE_HOURS + 1))
    )  # stale under the daemon's
    _write_fleet_events(fleet_dir, [_resolution_row(hb, resolved_file_path=str(daemon_digest))])
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert report.anomaly
    assert str(daemon_digest) in report.lines[0]
    assert str(script_digest) not in report.lines[0]


def test_notify_digest_stale_events_surfaced_in_facts(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    """notify_digest_stale is a genuinely consumed kind: a recent writer-side
    tripwire shows up in the heartbeat's facts line even when the file
    probe itself is fresh (recovered-writer shape)."""
    digest = tmp_path / "digest.jsonl"
    _write_digest(digest, _iso(1))
    _write_fleet_events(
        fleet_dir,
        [
            _resolution_row(hb, 120.0, resolved_file_path=str(digest)),
            (_iso(30), hb.NOTIFY_DIGEST_STALE_EVENT_KIND, "{}"),
        ],
    )
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert not report.anomaly
    assert report.lines[0].startswith("OK notify-digest")
    assert f"stale_events_{hb.NOTIFY_DIGEST_STALE_HOURS}h=1" in report.lines[0]


def test_notify_digest_warns_when_package_unimportable(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Guarded-leaf contract: a broken package install degrades the check to
    a WARN line instead of crashing the script (scripts/README invariant)."""
    monkeypatch.setattr(hb, "_ndc", None)
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert not report.anomaly
    assert report.lines[0].startswith("WARN notify-digest")
    assert "not importable" in report.lines[0]


def test_notify_digest_never_raises_on_internal_error(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    """An unexpected failure inside the check (round-1 review: exists()
    outside try) must degrade to a WARN line, never propagate -- the check
    runs before save_state and report output in main()."""
    digest = tmp_path / "digest.jsonl"
    _write_fleet_events(fleet_dir, [_resolution_row(hb, resolved_file_path=str(digest))])

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated probe failure")

    monkeypatch.setattr(hb._ndc, "probe_digest_file", _boom)
    report = hb.Report()
    hb.check_notify_digest_freshness(report)  # must not raise
    assert not report.anomaly
    assert report.lines[0].startswith("WARN notify-digest")
    assert "check failed unexpectedly" in report.lines[0]


def test_notify_digest_warns_on_unparseable_resolution_payload(
    hb: ModuleType, fleet_dir: Path
) -> None:
    _write_fleet_events(fleet_dir, [(_iso(1), hb.NOTIFY_RESOLUTION_EVENT_KIND, "{not json")])
    report = hb.Report()
    hb.check_notify_digest_freshness(report)
    assert not report.anomaly
    assert report.lines[0].startswith("WARN notify-digest")
    assert "not a JSON object" in report.lines[0]


def test_main_emits_notify_digest_line(hb: ModuleType, monkeypatch: Any, tmp_path: Path) -> None:
    """main() wiring: the notify-digest line must appear in a real beat's
    output, so removing the call from main() fails this test."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True)
    (fleet_dir / "fleet.json").write_text(json.dumps({"repos": {}}), encoding="utf-8")
    digest = tmp_path / "digest.jsonl"
    _write_digest(digest, _iso(1))
    _write_fleet_events(fleet_dir, [_resolution_row(hb, resolved_file_path=str(digest))])
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))
    monkeypatch.setenv("CHARLIE_WORK_HEARTBEAT_STATE", str(tmp_path / "hb-state.json"))
    monkeypatch.setenv(
        "CHARLIE_WORK_HEARTBEAT_SUPPRESSIONS", str(tmp_path / "no-suppressions.yaml")
    )

    class _FakeProc:
        returncode = 1
        stdout = ""
        stderr = "fake subprocess disabled in test"

    monkeypatch.setattr(hb.subprocess, "run", lambda *a, **k: _FakeProc())
    captured = io.StringIO()
    monkeypatch.setattr(hb.sys, "stdout", captured)

    hb.main()
    lines = captured.getvalue().splitlines()
    assert any("notify-digest" in line for line in lines), (
        f"main() emitted no notify-digest line:\n{captured.getvalue()}"
    )
