# ruff: noqa: F811  (the imported ``fleet`` fixture is re-bound as a test parameter)
"""Safety review fixes for the dashboard data layer (kill switch, registry, db open)."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from _dashboard_rollup_fixtures import ALPHA, NOW, fleet  # noqa: F401  (pytest fixture)

from charlie_work import cli
from charlie_work.dashboard import rollup, sources
from charlie_work.dashboard.rollup_schema import SCHEMA_VERSION


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    rc = cli.main(["--json", *argv])
    return rc, json.loads(capsys.readouterr().out)


# F1: a malformed `dashboard:` section must fail closed, never mean "enabled".
@pytest.mark.parametrize("value", ["false", "[]", "'off'", "0"])
def test_non_mapping_dashboard_section_fails_closed(fleet, capsys, value) -> None:
    (fleet.dir / "config.yaml").write_text(f"dashboard: {value}\n", encoding="utf-8")
    for sub in ("rollup", "now", "history"):
        rc, out = _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", sub)
        assert rc == 1 and out["ok"] is False
        assert out["message"].startswith("dashboard config error: dashboard: expected a mapping")
    assert not fleet.sources().db_path.exists()


# F2: a corrupt registry is an error, distinct from an absent one.
def test_read_registry_distinguishes_corrupt_from_absent(tmp_path: Path) -> None:
    assert sources.read_registry(tmp_path / "fleet.json") == ({"repos": {}}, None)  # control
    bad = tmp_path / "fleet.json"
    bad.write_text('{ "repos": {', encoding="utf-8")
    data, error = sources.read_registry(bad)
    assert data == {"repos": {}} and error is not None and "corrupt" in error
    bad.write_text("[1]", encoding="utf-8")
    assert sources.read_registry(bad)[1] is not None
    bad.write_text('{"repos": []}', encoding="utf-8")
    assert sources.read_registry(bad)[1] is not None


def test_corrupt_registry_makes_rollup_and_now_fail(fleet, capsys) -> None:
    fleet.close()
    (fleet.dir / "fleet.json").write_text('{ "repos": {', encoding="utf-8")
    for sub in ("rollup", "now"):
        rc, out = _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", sub)
        assert rc == 1 and out["ok"] is False
        assert "fleet registry corrupt" in out["message"]


def test_empty_registry_is_still_ok(fleet, capsys) -> None:
    fleet.close()
    (fleet.dir / "fleet.json").write_text('{"repos": {}}', encoding="utf-8")
    assert _run(capsys, "--fleet-dir", str(fleet.dir), "dashboard", "rollup")[1]["ok"] is True


# F3: opening never unlinks a file another process may hold.
def test_open_tolerates_a_peer_that_created_the_file_but_not_the_schema(tmp_path: Path) -> None:
    path = tmp_path / "dashboard.db"
    peer = sqlite3.connect(path)
    peer.execute("PRAGMA journal_mode=WAL")
    ino = os.stat(path).st_ino
    try:
        conn, rebuilt, err = rollup._open_dashboard_db(path)
        assert err is None and conn is not None and rebuilt is False
        assert os.stat(path).st_ino == ino  # same file, not unlinked and recreated
        conn.close()
    finally:
        peer.close()


def test_version_rebuild_happens_in_place_while_a_peer_is_connected(tmp_path: Path) -> None:
    path = tmp_path / "dashboard.db"
    conn, _, _ = rollup._open_dashboard_db(path)
    conn.execute(
        "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION + 1),)
    )
    conn.execute("INSERT INTO watermarks (source, max_id) VALUES ('ghost', 1)")
    peer = sqlite3.connect(path)
    peer.execute("SELECT 1 FROM meta").fetchall()
    ino = os.stat(path).st_ino
    conn.close()
    try:
        conn, rebuilt, err = rollup._open_dashboard_db(path)
        assert err is None and conn is not None and rebuilt is True
        assert os.stat(path).st_ino == ino
        assert conn.execute("SELECT COUNT(*) FROM watermarks").fetchone() == (0,)
        conn.close()
    finally:
        peer.close()


# F5: a fleet dir that does not exist must not grow a ghost dashboard.db.
def test_missing_fleet_dir_is_an_error_and_creates_nothing(tmp_path: Path) -> None:
    db_path = tmp_path / "typo" / "nested" / "dashboard.db"
    result = rollup.run_rollup(
        rollup.RollupSources(db_path, tmp_path / "typo" / "events.db", ()), NOW
    )
    assert result.error is not None and "does not exist" in result.error
    assert not (tmp_path / "typo").exists()


# F4: a pass that loses the race must not re-ingest or re-report the winner's rows.
class _PeerAfterWatermarkRead:
    """Proxy for the dashboard connection: a peer pass commits right after the watermark read
    if (and only if) this pass has not yet taken the write lock."""

    def __init__(self, real: sqlite3.Connection, peer_run) -> None:
        self._real, self._peer_run, self.peer_result = real, peer_run, None

    def execute(self, sql: str, *args):
        out = self._real.execute(sql, *args)
        if "FROM watermarks WHERE" in sql and not self._real.in_transaction:
            # old code: lock not yet held, so a peer can commit between read and BEGIN
            self.peer_result = self._peer_run()
        return out

    def __getattr__(self, name: str):
        return getattr(self._real, name)


def test_concurrent_pass_does_not_double_report_ingested(fleet) -> None:
    db_path = fleet.sources().db_path
    main, _, _ = rollup._open_dashboard_db(db_path)
    peer, _, _ = rollup._open_dashboard_db(db_path)
    alpha_db = dict(fleet.sources().repos)[ALPHA]
    cutoff = rollup._iso(NOW - rollup.WINDOW)
    proxy = _PeerAfterWatermarkRead(
        main, lambda: rollup._ingest_source(peer, ALPHA, alpha_db, cutoff)
    )
    try:
        mine = rollup._ingest_source(proxy, ALPHA, alpha_db, cutoff)
    finally:
        main.close()
        peer.close()
    peer_ingested = proxy.peer_result.ingested if proxy.peer_result else 0
    assert mine.error is None
    # 19 handled events plus alpha's 2 global-kind copies (no global rows in this dst: admitted)
    assert mine.ingested + peer_ingested == 21  # each counted once
