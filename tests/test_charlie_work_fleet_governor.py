"""Fleet concurrency governor, cross-repo dispatch lock, and fleet dir.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the fleet-governor half of the ``test_fleet_*`` seam -- fleet-wide session
caps, the cross-repo dispatch lock, and fleet dir resolution. Sibling seam:
``test_charlie_work_fleet_status.py`` (fleet status/queue aggregation).
Shared fakes and helpers in ``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from charlie_work.config import (
    DevinConfig,
    DispatchConfig,
    FleetConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.devin_shell import SessionRecord
from charlie_work.paths import runtime_paths
from charlie_work.state import save_state
from charlie_work.workflow import (
    CommandResult,
    ConcurrencyGovernorResult,
    OrchestratorApp,
)
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_fleet_concurrency_governor_unlimited_when_unset(tmp_path: Path, monkeypatch) -> None:
    """When fleet.global_max_concurrent_sessions is 0 (default), dispatch should behave as before (unlimited)."""

    # Mock count_fleet_live_sessions to return 0 fleet live sessions
    def mock_count_fleet_live(fleet_dir_override):
        return 0, []

    monkeypatch.setattr("charlie_work.workflow.count_fleet_live_sessions", mock_count_fleet_live)

    config = OrchestratorConfig(
        fleet=FleetConfig(global_max_concurrent_sessions=0),
        dispatch=DispatchConfig(max_concurrent_sessions=0),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch()

    # Should dispatch normally without fleet concurrency clamping
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert "fleet_concurrency_limit" not in result.data
    assert "fleet_live_session_count" not in result.data


def test_fleet_concurrency_governor_clamps_when_fleet_live_at_cap(
    tmp_path: Path, monkeypatch
) -> None:
    """When fleet.global_max_concurrent_sessions is set and fleet live count meets cap, dispatch should be clamped."""

    # Mock count_fleet_live_sessions to return 3 fleet live sessions (at cap)
    def mock_count_fleet_live(fleet_dir_override):
        return 3, []

    monkeypatch.setattr("charlie_work.workflow.count_fleet_live_sessions", mock_count_fleet_live)

    config = OrchestratorConfig(
        fleet=FleetConfig(global_max_concurrent_sessions=3),
        dispatch=DispatchConfig(max_concurrent_sessions=5, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(
        tmp_path, paths, config, fake_gh, fleet_dir_override=str(tmp_path / "fleet")
    )

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch()

    # Should clamp to 0 since fleet cap is 3 and fleet live is 3
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["fleet_concurrency_limit"] == 3
    assert result.data["fleet_live_session_count"] == 3


def test_fleet_concurrency_governor_tighter_cap_wins(tmp_path: Path, monkeypatch) -> None:
    """When both per-repo and fleet caps are set, the tighter constraint wins."""

    # Mock count_fleet_live_sessions to return 1 fleet live session
    def mock_count_fleet_live(fleet_dir_override):
        return 1, []

    # Mock _count_live_sessions to return 1 local live session
    def mock_count_live(sessions_dir, state_file=None):
        return 1

    monkeypatch.setattr("charlie_work.workflow.count_fleet_live_sessions", mock_count_fleet_live)
    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        fleet=FleetConfig(global_max_concurrent_sessions=1),
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(
        tmp_path, paths, config, fake_gh, fleet_dir_override=str(tmp_path / "fleet")
    )

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch()

    # Fleet cap (1) is tighter than per-repo cap (2), so should clamp to 0
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["concurrency_limit"] == 2  # per-repo cap
    assert result.data["live_session_count"] == 1  # local live
    assert result.data["fleet_concurrency_limit"] == 1  # fleet cap
    assert result.data["fleet_live_session_count"] == 1  # fleet live


def test_fleet_concurrency_governor_per_repo_cap_tighter(tmp_path: Path, monkeypatch) -> None:
    """When per-repo cap is tighter than fleet cap, per-repo wins."""

    # Mock count_fleet_live_sessions to return 1 fleet live session
    def mock_count_fleet_live(fleet_dir_override):
        return 1, []

    # Mock _count_live_sessions to return 1 local live session
    def mock_count_live(sessions_dir, state_file=None):
        return 1

    monkeypatch.setattr("charlie_work.workflow.count_fleet_live_sessions", mock_count_fleet_live)
    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        fleet=FleetConfig(global_max_concurrent_sessions=5),
        dispatch=DispatchConfig(max_concurrent_sessions=1, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(
        tmp_path, paths, config, fake_gh, fleet_dir_override=str(tmp_path / "fleet")
    )

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch()

    # Per-repo cap (1) is tighter than fleet cap (5), so should clamp to 0
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["concurrency_limit"] == 1  # per-repo cap
    assert result.data["live_session_count"] == 1  # local live
    assert result.data["fleet_concurrency_limit"] == 5  # fleet cap
    assert result.data["fleet_live_session_count"] == 1  # fleet live


def test_fleet_concurrency_governor_result_fleet_enabled_property() -> None:
    """ConcurrencyGovernorResult.fleet_enabled property correctly reflects fleet governor enabled state."""
    result = ConcurrencyGovernorResult(
        clamped=True,
        max_concurrent=5,
        live_count=2,
        available_slots=3,
        dispatch_limit=3,
        fleet_live_count=1,
        fleet_max=3,
    )

    assert result.fleet_enabled is True  # fleet_max > 0 means enabled

    result_unlimited = ConcurrencyGovernorResult(
        clamped=False,
        max_concurrent=0,
        live_count=0,
        available_slots=5,
        dispatch_limit=5,
        fleet_live_count=0,
        fleet_max=0,
    )

    assert result_unlimited.fleet_enabled is False  # fleet_max=0 means disabled


def test_fleet_concurrency_governor_result_report_fields_includes_fleet() -> None:
    """ConcurrencyGovernorResult.report_fields includes fleet fields when fleet_enabled."""
    result = ConcurrencyGovernorResult(
        clamped=True,
        max_concurrent=5,
        live_count=2,
        available_slots=3,
        dispatch_limit=3,
        fleet_live_count=1,
        fleet_max=3,
    )

    fields = result.report_fields()
    assert fields == {
        "concurrency_limit": 5,
        "live_session_count": 2,
        "available_slots": 3,
        "fleet_concurrency_limit": 3,
        "fleet_live_session_count": 1,
    }

    # When fleet disabled, fleet fields should not be present
    result_unlimited = ConcurrencyGovernorResult(
        clamped=False,
        max_concurrent=0,
        live_count=0,
        available_slots=5,
        dispatch_limit=5,
        fleet_live_count=0,
        fleet_max=0,
    )

    fields_unlimited = result_unlimited.report_fields()
    assert fields_unlimited == {
        "concurrency_limit": 0,
        "live_session_count": 0,
        "available_slots": 5,
    }
    assert "fleet_concurrency_limit" not in fields_unlimited
    assert "fleet_live_session_count" not in fields_unlimited


def test_fleet_lock_serializes_cross_repo_dispatch(tmp_path: Path, monkeypatch) -> None:
    """Independent dispatch() calls across two repos sharing a fleet cap cannot
    over-dispatch. Without the fleet lock, both repos could read a stale live
    count of 0 and dispatch up to the cap each, oversubscribing the fleet.
    """
    from charlie_work.adapters import SessionDispatchResult
    from charlie_work.devin_shell import _sidecar_path

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    shared_fleet_dir = str(fleet_dir)

    # Fake a non-blocking devin-shell launch that writes a sidecar so the
    # fleet-wide live count is visible to subsequent dispatchers.
    def fake_run_devin_shell(
        repo_root: Path,
        request: Any,
        sessions_dir: Path,
        settings: Any,
    ) -> SessionDispatchResult:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        record = SessionRecord(
            issue_number=request.issue_number,
            branch=request.branch_name,
            worktree_path=str(repo_root),
            prompt_path=str(request.prompt_path),
            command=("devin",),
            pid=9999,
            started_at="2024-01-01T00:00:00Z",
            log_path=str(sessions_dir / f"issue-{request.issue_number}.log"),
            error=None,
            process_start_time=1.0,
        )
        _sidecar_path(sessions_dir, request.issue_number).write_text(
            json.dumps(record.to_dict()), encoding="utf-8"
        )
        return SessionDispatchResult(
            issue_number=request.issue_number,
            issue_title=request.issue_title,
            prompt_path=str(request.prompt_path),
            branch_name=request.branch_name,
            adapter="devin-shell",
            ok=True,
            command=list(record.command),
            pid=record.pid,
            process_start_time=record.process_start_time,
        )

    monkeypatch.setattr("charlie_work.adapters._run_devin_shell_adapter", fake_run_devin_shell)
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda _record: True)

    # Build two independent repos, each sharing the same fleet directory.
    apps: list[OrchestratorApp] = []
    repo_entries: dict[str, dict[str, str]] = {}
    for repo_name in ("owner/repo-a", "owner/repo-b"):
        repo_root = tmp_path / repo_name.replace("/", "--")
        repo_root.mkdir()
        (repo_root / ".git").mkdir()
        paths = runtime_paths(repo_root, ".var/charlie-work")
        repo_entries[repo_name] = {
            "repo_root": str(repo_root),
            "state_dir": str(paths.root),
        }

        config = OrchestratorConfig(
            devin=DevinConfig(),
            worker=WorkerRoleConfig(harness="devin-shell"),
            dispatch=DispatchConfig(
                default_limit=3,
                launch_stagger_seconds=0,
            ),
            fleet=FleetConfig(global_max_concurrent_sessions=2),
        )
        fake_gh = FakeGitHub()
        fake_gh.prs = []
        fake_gh.issues = [
            {
                "number": 100 + i,
                "title": f"Issue {i}",
                "url": f"https://example.test/issues/{100 + i}",
                "body": "",
                "labels": [{"name": "automated-ready"}],
                "state": "OPEN",
            }
            for i in range(3)
        ]
        app = OrchestratorApp(
            repo_root,
            paths,
            config,
            fake_gh,
            fleet_dir_override=shared_fleet_dir,
        )
        apps.append(app)

    # Seed the fleet registry so both repos are visible to count_fleet_live_sessions.
    save_state(fleet_dir / "fleet.json", {"repos": repo_entries})

    # Launch both dispatch() calls concurrently from the same barrier.
    barrier = threading.Barrier(2)
    results: list[CommandResult] = []

    def worker(app: OrchestratorApp) -> None:
        barrier.wait(timeout=5)
        results.append(app.dispatch())

    threads = [threading.Thread(target=worker, args=(app,)) for app in apps]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    total_dispatched = sum(r.data.get("selected_count", 0) for r in results)
    # With a fleet cap of 2 and 3 ready issues in each repo, the combined
    # dispatch across both repos must never exceed the fleet-wide cap.
    assert total_dispatched <= 2, (
        f"fleet-wide dispatch over-subscribed: {total_dispatched} workers launched"
    )
    # Exactly one path should have succeeded; the other either clamped to 0 or
    # deferred because the fleet lock was held.
    assert any(r.data.get("fleet_live_session_count") is not None for r in results), (
        "fleet live count should be reported in dispatch results"
    )


# Fleet registry and global config tests


def test_fleet_dir_override() -> None:
    """Test that fleet_dir respects the override parameter."""
    from charlie_work.fleet_paths import fleet_dir

    result = fleet_dir(override="/custom/path")
    assert result == Path("/custom/path")


def test_fleet_dir_env_var() -> None:
    """Test that fleet_dir respects CHARLIE_WORK_FLEET_DIR env var."""
    from charlie_work.fleet_paths import fleet_dir

    original = os.environ.get("CHARLIE_WORK_FLEET_DIR")
    try:
        os.environ["CHARLIE_WORK_FLEET_DIR"] = "/env/path"
        result = fleet_dir()
        assert result == Path("/env/path")
    finally:
        if original is None:
            os.environ.pop("CHARLIE_WORK_FLEET_DIR", None)
        else:
            os.environ["CHARLIE_WORK_FLEET_DIR"] = original


def test_fleet_dir_platform_defaults() -> None:
    """Test that fleet_dir uses platform-specific defaults."""
    from charlie_work.fleet_paths import fleet_dir

    # Clear env var to test platform defaults
    original = os.environ.get("CHARLIE_WORK_FLEET_DIR")
    try:
        os.environ.pop("CHARLIE_WORK_FLEET_DIR", None)
        result = fleet_dir()

        if sys.platform == "win32":
            expected_base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        else:
            expected_base = Path(
                os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
            )

        assert result == expected_base / "charlie-work"
    finally:
        if original is not None:
            os.environ["CHARLIE_WORK_FLEET_DIR"] = original
