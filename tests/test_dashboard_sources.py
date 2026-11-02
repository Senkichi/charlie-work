"""Tests for the dashboard's read-only source layer (``dashboard/sources.py``)."""

from __future__ import annotations

import ast
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _src_ast import parsed

from charlie_work import instrumentation
from charlie_work.dashboard import sources
from charlie_work.fleet_registry import touch_repo
from charlie_work.github import GitHub
from charlie_work.paths import runtime_paths
from charlie_work.supervisor_lifecycle import write_supervisor_heartbeat

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)

# ci_fleet modules that start/park/terminate runners or observe GitHub (runners recon
# sections 1 and 5): the dashboard must not be able to reach them.
FORBIDDEN_CI_FLEET = frozenset(
    {"runner_slots", "runners", "runner_allocation_pass", "runner_allocation"}
)


def _register(fleet_dir: Path, repo_root: Path, name: str) -> Path:
    """Register ``repo_root`` through the real registry writer; return its state dir."""

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return name

    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")
    touch_repo(str(fleet_dir), repo_root, paths, FakeGitHub(repo_root=repo_root))
    return paths.root


def _write_snapshot(state_dir: Path, written_at: str, data: dict) -> None:
    """Write the envelope exactly as status_snapshot.write_status_snapshot does.

    The real writer needs a live OrchestratorApp (GitHub-backed ``status()``); this
    mirrors its ``{"snapshot_written_at", "data"}`` envelope and atomic replace.
    """
    path = state_dir / "status-snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"snapshot_written_at": written_at, "data": data}), encoding="utf-8")
    tmp.replace(path)


@pytest.fixture
def fleet(tmp_path: Path):
    fleet_dir = tmp_path / "fleet"
    state_a = _register(fleet_dir, tmp_path / "alpha", "owner/alpha")
    state_b = _register(fleet_dir, tmp_path / "beta", "local/beta")
    yield fleet_dir, state_a, state_b
    for state in (state_a, state_b, fleet_dir):
        instrumentation.close_db(state / "state.json")
    instrumentation.close_db(fleet_dir / sources.HEARTBEAT_FILENAME)


def test_enumerate_repos_from_registry_only(fleet, tmp_path: Path) -> None:
    fleet_dir, state_a, state_b = fleet
    # A stray events.db that is NOT in the registry must never be enumerated.
    stray = tmp_path / "stray" / ".var" / "charlie-work"
    instrumentation.log_event(stray / "state.json", "x", {})
    instrumentation.close_db(stray / "state.json")

    repos = sources.enumerate_repos(str(fleet_dir))

    assert [r.key for r in repos] == ["local/beta", "owner/alpha"]
    alpha = repos[1]
    assert alpha.repo_root == tmp_path / "alpha"
    assert alpha.state_dir == state_a
    assert alpha.snapshot_path == state_a / "status-snapshot.json"
    assert alpha.events_db == state_a / "events.db"
    assert repos[0].state_dir == state_b


def test_enumerate_repos_missing_registry_is_empty(tmp_path: Path) -> None:
    assert sources.enumerate_repos(str(tmp_path / "nowhere")) == ()


def test_fleet_sources_paths(tmp_path: Path) -> None:
    fs = sources.fleet_sources(str(tmp_path))
    assert fs.fleet_dir == tmp_path
    assert fs.registry == tmp_path / "fleet.json"
    assert fs.events_db == tmp_path / "events.db"
    assert fs.supervisor_heartbeat == tmp_path / "supervisor-heartbeat.json"
    assert fs.runner_allocation == tmp_path / "runner-allocation.json"
    assert fs.capacity_starvation_state == tmp_path / "capacity_starvation_state.json"
    assert fs.fleet_pause == tmp_path / "fleet-pause.json"


def test_fleet_sources_default_follows_env(fleet) -> None:
    fleet_dir, _, _ = fleet
    assert sources.fleet_sources().fleet_dir == fleet_dir  # conftest sets CHARLIE_WORK_FLEET_DIR


def test_read_snapshot_age_from_injected_now(fleet) -> None:
    fleet_dir, state_a, _ = fleet
    _write_snapshot(state_a, "2026-10-01T11:57:30Z", {"ready_issue_count": 18, "workers": []})
    repo = sources.enumerate_repos(str(fleet_dir))[1]

    read = sources.read_snapshot(repo, NOW)

    assert read.error is None
    assert read.written_at == datetime(2026, 10, 1, 11, 57, 30, tzinfo=UTC)
    assert read.age_seconds == 150.0
    assert read.data == {"ready_issue_count": 18, "workers": []}


def test_read_snapshot_errors_are_values(fleet) -> None:
    fleet_dir, state_a, state_b = fleet
    alpha, beta = (
        sources.enumerate_repos(str(fleet_dir))[1],
        sources.enumerate_repos(str(fleet_dir))[0],
    )

    missing = sources.read_snapshot(alpha, NOW)
    assert (missing.data, missing.written_at, missing.age_seconds) == (None, None, None)
    assert missing.error == f"missing: {alpha.snapshot_path}"

    state_a.mkdir(parents=True, exist_ok=True)
    (state_a / "status-snapshot.json").write_text("{not json", encoding="utf-8")
    assert sources.read_snapshot(alpha, NOW).error.startswith("unreadable: ")

    _write_snapshot(state_b, "not-a-time", {"a": 1})
    assert (
        sources.read_snapshot(beta, NOW).error
        == f"no valid snapshot_written_at: {beta.snapshot_path}"
    )

    (state_b / "status-snapshot.json").write_text(
        json.dumps({"snapshot_written_at": "2026-10-01T11:00:00Z", "data": []}), encoding="utf-8"
    )
    bad = sources.read_snapshot(beta, NOW)
    assert bad.error == f"snapshot has no data object: {beta.snapshot_path}"
    assert bad.written_at == datetime(2026, 10, 1, 11, 0, 0, tzinfo=UTC)


def test_read_json_file_non_object(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text("[1]", encoding="utf-8")
    assert sources.read_json_file(path) == sources.JsonRead(None, f"not a JSON object: {path}")
    path.write_text('{"pid": 7}', encoding="utf-8")
    assert sources.read_json_file(path) == sources.JsonRead({"pid": 7}, None)


def test_open_events_ro_and_latest_event(fleet) -> None:
    fleet_dir, state_a, _ = fleet
    log = state_a / "state.json"
    instrumentation.log_event(log, "dispatch", {"launched": 1})
    instrumentation.log_event(log, "dispatch", {"launched": 2}, level="warning")
    instrumentation.log_event(log, "other", {"n": 9})
    repo = sources.enumerate_repos(str(fleet_dir))[1]

    conn, error = sources.open_events_ro(repo.events_db)
    assert error is None and conn is not None
    try:
        newest = sources.latest_event(conn, "dispatch")
        assert newest is not None
        assert newest["kind"] == "dispatch"
        assert newest["payload"] == {"launched": 2}
        assert newest["level"] == "warning"
        assert sources.latest_event(conn, "never_logged") is None
        # Read-only: a write through this connection must fail.
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM events")
    finally:
        conn.close()


def test_open_events_ro_missing_file_is_value_and_not_created(tmp_path: Path) -> None:
    path = tmp_path / "absent" / "events.db"
    conn, error = sources.open_events_ro(path)
    assert conn is None
    assert error == f"missing: {path}"
    assert not path.exists() and not path.parent.exists()


def test_open_events_ro_non_database_is_value(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    path.write_bytes(b"this is not a sqlite database" * 20)
    conn, error = sources.open_events_ro(path)
    assert conn is None
    assert error is not None and error.startswith(f"cannot open {path}")


def test_global_events_db_readable(fleet) -> None:
    fleet_dir, _, _ = fleet
    fs = sources.fleet_sources(str(fleet_dir))
    instrumentation.log_event(fs.supervisor_heartbeat, "supervisor_started", {"pid": 42})
    conn, error = sources.open_events_ro(fs.events_db)
    assert error is None and conn is not None
    try:
        event = sources.latest_event(conn, "supervisor_started")
        assert event is not None and event["payload"] == {"pid": 42}
    finally:
        conn.close()


def test_orchestrator_events_db_follows_heartbeat_stamp(fleet, tmp_path: Path) -> None:
    """Issue #2475: the supervisor's checkout DB resolves from its heartbeat.

    ``self_deploy`` logs beside the checkout's own ``state.json``
    (``supervise._self_deploy_state_path``); the heartbeat's
    ``orchestrator_root`` is the durable record the dashboard derives the
    same path from. The returned path must be the DB the writer actually
    writes -- asserted by writing a real event through ``log_event``.
    """
    fleet_dir, _, _ = fleet
    checkout = tmp_path / "daemon-checkout"
    log = checkout / ".var" / "charlie-work" / "state.json"
    instrumentation.log_event(log, "self_deploy_succeeded", {"ok": True})
    instrumentation.close_db(log)
    write_supervisor_heartbeat(
        fleet_dir / sources.HEARTBEAT_FILENAME, {"orchestrator_root": str(checkout)}
    )

    resolved = sources.orchestrator_events_db(str(fleet_dir))

    assert resolved == checkout / ".var" / "charlie-work" / "events.db"
    assert resolved.exists()  # the DB the self-deploy writer just wrote


def test_orchestrator_events_db_missing_or_malformed_is_none(tmp_path: Path) -> None:
    """Absent, predating, unreadable, or wrongly-typed stamps all read as None."""
    fleet_dir = tmp_path / "fleet"
    heartbeat = fleet_dir / sources.HEARTBEAT_FILENAME

    assert sources.orchestrator_events_db(str(fleet_dir)) is None  # no heartbeat

    write_supervisor_heartbeat(heartbeat, {"pid": 1})
    assert sources.orchestrator_events_db(str(fleet_dir)) is None  # predates the stamp

    heartbeat.write_text("{not json", encoding="utf-8")
    assert sources.orchestrator_events_db(str(fleet_dir)) is None  # unreadable

    write_supervisor_heartbeat(heartbeat, {"orchestrator_root": 42})
    assert sources.orchestrator_events_db(str(fleet_dir)) is None  # non-string field

    write_supervisor_heartbeat(heartbeat, {"orchestrator_root": ""})
    assert sources.orchestrator_events_db(str(fleet_dir)) is None  # empty string


def test_sources_does_not_import_actuating_ci_fleet_modules() -> None:
    tree = parsed(Path(sources.__file__))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.update(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            imported.update((node.module or "").split("."))
            imported.update(alias.name for alias in node.names)
    assert imported.isdisjoint(FORBIDDEN_CI_FLEET), imported & FORBIDDEN_CI_FLEET
    # Positive control: the walker does see imports, and detects a forbidden one.
    assert "sqlite3" in imported
    probe = ast.parse("from ci_fleet.runner_slots import park_runner_slot")
    probe_names = {
        p
        for n in ast.walk(probe)
        if isinstance(n, ast.ImportFrom)
        for p in (n.module or "").split(".")
    }
    assert not probe_names.isdisjoint(FORBIDDEN_CI_FLEET)
