"""First-class repo retirement tests (issue #2754).

Covers the per-repo ``fleet.retired`` key, the derived registry field, the
wind-down/skip behaviour in ``fleet_loop``, the fleet-wide enumeration
contract, and the ``charlie fleet retire`` / ``unretire`` commands.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

from _fleet_dispatch_fixtures import (  # noqa: F401  (autouse hermeticity)
    _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests,
    _per_repo_runtime_paths,
)
from charlie_work import cli
from charlie_work.config import FleetConfig, LocalIssuesConfig, OrchestratorConfig
from charlie_work.fleet_dispatch import (
    _fleet_has_configured_repos,
    _take_fleet_snapshot,
    fleet_loop,
)
from charlie_work.fleet_registry import _select_repos, managed_repo_names, touch_repo
from charlie_work.paths import runtime_paths
from charlie_work.repo_retirement import (
    LiveWork,
    entry_is_retired,
    repo_is_retired,
    repo_live_work,
)
from charlie_work.command_result import CommandResult


def _write_config(repo: Path, *, retired: bool) -> Path:
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        f"fleet:\n  retired: {str(retired).lower()}\n"
        "local_issues:\n  enabled: true\n"
        "dispatch:\n  default_limit: 2\n",
        encoding="utf-8",
    )
    return config


def _repo_entry(repo: Path, config: Path, *, state_dir: Path | None = None) -> dict[str, object]:
    state = state_dir or repo / ".var" / "charlie-work"
    state.mkdir(parents=True, exist_ok=True)
    return {
        "repo_root": str(repo),
        "name_with_owner": "local/localrepo",
        "config_path": str(config),
        "state_dir": str(state),
        "first_seen": "2026-01-01T00:00:00Z",
        "last_seen": "2026-01-01T00:00:00Z",
    }


class _FakeGitHub:
    def __init__(self, key: str = "local/localrepo") -> None:
        self._key = key

    def name_with_owner(self) -> str:
        return self._key


def _write_worker_sidecar(repo: Path) -> Path:
    sessions = repo / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    sidecar = sessions / "issue-1.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 1,
                "branch": "agent/issue-1",
                "worktree_path": str(repo / "worktrees" / "issue-1"),
                "prompt_path": str(repo / "prompt.md"),
                "command": ["devin"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(repo / "log.txt"),
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def test_config_retired_defaults_false_and_global_layer_cannot_set_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fleet-layer ``fleet.retired`` is stripped; the repo layer still wins."""
    from charlie_work.global_config import load_layered_config

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "config.yaml").write_text("fleet:\n  retired: true\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()

    config = load_layered_config(repo, None, fleet_dir_override=str(fleet_dir))
    assert config.fleet.retired is False

    _write_config(repo, retired=True)
    config = load_layered_config(repo, None, fleet_dir_override=str(fleet_dir))
    assert config.fleet.retired is True


def test_repo_is_retired_reads_only_the_repo_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert repo_is_retired(repo) is False
    _write_config(repo, retired=True)
    assert repo_is_retired(repo) is True


def test_touch_repo_keeps_retirement_and_bumps_last_seen(tmp_path: Path) -> None:
    """``touch_repo`` never clears retirement; it refreshes it from config."""
    repo = tmp_path / "repo"
    repo.mkdir()
    config = _write_config(repo, retired=True)
    paths = runtime_paths(repo, ".var/charlie-work")
    fleet_dir = tmp_path / "fleet"

    first = touch_repo(str(fleet_dir), repo, paths, _FakeGitHub())  # type: ignore[arg-type]
    entry = first["repos"]["local/localrepo"]
    assert entry["retired"] is True
    first_last_seen = entry["last_seen"]
    assert entry["config_path"] == str(config)

    second = touch_repo(str(fleet_dir), repo, paths, _FakeGitHub())  # type: ignore[arg-type]
    entry = second["repos"]["local/localrepo"]
    assert entry["retired"] is True
    assert entry["last_seen"] >= first_last_seen


def test_registry_contract_pins_existing_keys(tmp_path: Path) -> None:
    """The additive ``retired`` field never removes an existing registry key."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_config(repo, retired=False)
    paths = runtime_paths(repo, ".var/charlie-work")
    entry = touch_repo(str(tmp_path / "fleet"), repo, paths, _FakeGitHub())["repos"][  # type: ignore[arg-type]
        "local/localrepo"
    ]
    assert set(entry) == {
        "repo_root",
        "name_with_owner",
        "config_path",
        "state_dir",
        "first_seen",
        "last_seen",
        "retired",
    }


@patch("charlie_work.fleet_dispatch.compute_api_worker_fleet_report")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_retired_quiescent_repo_launches_nothing(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_report: MagicMock,
    tmp_path: Path,
) -> None:
    """A quiescent retired repo is skipped entirely: no app, no lane, no event."""
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_config(repo, retired=True)
    entry = _repo_entry(repo, config_path)

    mock_load_registry.return_value = {"repos": {"local/localrepo": entry}}
    mock_load_config.return_value = OrchestratorConfig(
        fleet=FleetConfig(retired=True),
        local_issues=LocalIssuesConfig(enabled=True),
    )
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()
    mock_report.return_value = None

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        limit=2,
        dry_run=False,
        work_only=False,
    )

    assert result.data["retired_quiescent"] == ["local/localrepo"]
    assert "local/localrepo" not in result.data["repos"]
    mock_app_class.assert_not_called()


@patch("charlie_work.fleet_lanes.run_wind_down")
@patch("charlie_work.fleet_dispatch.compute_api_worker_fleet_report")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_retired_live_worker_winds_down_then_becomes_quiescent(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_report: MagicMock,
    mock_wind_down: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live worker is reaped in wind-down; the next quiescent pass skips."""
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_config(repo, retired=True)
    sidecar = _write_worker_sidecar(repo)
    entry = _repo_entry(repo, config_path)

    mock_load_registry.return_value = {"repos": {"local/localrepo": entry}}
    mock_load_config.return_value = OrchestratorConfig(fleet=FleetConfig(retired=True))
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()
    mock_report.return_value = None
    mock_wind_down.return_value = {"retired": True, "wind_down": True}
    apps = [MagicMock(), MagicMock()]
    mock_app_class.side_effect = apps
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: True)

    first = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        limit=2,
        dry_run=False,
        work_only=False,
    )

    # Wind-down ran, and no normal loop/dispatch lane did.
    first_app = apps[0]
    assert first.data["repos"]["local/localrepo"]["wind_down"] is True
    first_app.loop.assert_not_called()
    first_app.dispatch.assert_not_called()
    first_app.dispatch_reviews.assert_not_called()
    mock_wind_down.assert_called_once()

    # Once the worker is gone the repo is skipped entirely.
    sidecar.unlink()
    second = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        limit=2,
        dry_run=False,
        work_only=False,
    )
    assert second.data["retired_quiescent"] == ["local/localrepo"]
    assert mock_app_class.call_count == 1


@patch("charlie_work.fleet_dispatch.compute_api_worker_fleet_report")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_flipping_the_key_takes_effect_on_the_next_pass(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_report: MagicMock,
    tmp_path: Path,
) -> None:
    """No supervisor restart: the next pass reads the new config value."""
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_config(repo, retired=False)
    entry = _repo_entry(repo, config_path)
    mock_load_registry.return_value = {"repos": {"local/localrepo": entry}}
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()
    mock_report.return_value = None
    app = MagicMock()
    app.loop.return_value = CommandResult(True, "loop complete", {})
    mock_app_class.return_value = app
    mock_load_config.return_value = OrchestratorConfig(fleet=FleetConfig(retired=False))

    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        limit=2,
        dry_run=False,
        work_only=False,
    )
    app.loop.assert_called_once_with(2, merge=None, deadline_exceeded=ANY)

    _write_config(repo, retired=True)
    mock_load_config.return_value = OrchestratorConfig(fleet=FleetConfig(retired=True))
    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        limit=2,
        dry_run=False,
        work_only=False,
    )
    assert result.data["retired_quiescent"] == ["local/localrepo"]
    assert app.loop.call_count == 1


def test_repo_live_work_counts_workers_and_gate_suites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_worker_sidecar(repo)
    state_dir = repo / ".var" / "charlie-work"
    config = OrchestratorConfig()
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: True)

    live = repo_live_work(repo, config, state_dir)
    assert live.workers == 1
    assert live.quiescent is False

    gate = state_dir / "dispatches" / "local-merge-gate" / "pr-7"
    gate.mkdir(parents=True)
    (gate / "suite-runner.json").write_text(
        json.dumps({"pid": 999, "process_start_time": 1.0}), encoding="utf-8"
    )
    fake_probe = type("_Probe", (), {"is_alive": lambda self, *_a: True})()
    fake_clock = type("_Clock", (), {"now": lambda self: datetime.now(UTC)})()
    monkeypatch.setattr(
        "charlie_work.repo_retirement._host.current",
        lambda: type("_Host", (), {"probe": fake_probe, "clock": fake_clock})(),
    )
    live = repo_live_work(repo, config, state_dir)
    assert live.gate_suites == 1


def test_fleet_enumerations_ignore_retired_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_config(repo, retired=True)
    entry = _repo_entry(repo, config_path)
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": {"local/localrepo": entry}}), encoding="utf-8"
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    assert managed_repo_names(None) == frozenset()
    assert _fleet_has_configured_repos(None, None) is False
    assert _take_fleet_snapshot(fleet_dir_override=None).repo_snapshots == frozenset()


def test_fleet_status_reports_retired_not_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_config(repo, retired=True)
    entry = _repo_entry(repo, config_path)
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": {"local/localrepo": entry}}), encoding="utf-8"
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    result = cli.run_fleet_status(cli.build_parser().parse_args(["fleet", "status"]))

    assert result.ok is True
    assert result.data["repos"] == {}
    assert result.data["stale"] == []
    assert result.data["errors"] == []
    assert [row["repo_key"] for row in result.data["retired"]] == ["local/localrepo"]


def test_cli_retire_and_unretire_edit_only_the_repo_config(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    retire_args = cli.build_parser().parse_args(["--repo", str(repo), "fleet", "retire"])
    result = cli.run_fleet_retire(retire_args)
    assert result.ok is True
    assert repo_is_retired(repo) is True

    dry_args = cli.build_parser().parse_args(
        ["--repo", str(repo), "fleet", "unretire", "--dry-run"]
    )
    result = cli.run_fleet_unretire(dry_args)
    assert result.ok is True
    assert result.data["changed"] is True
    assert repo_is_retired(repo) is True  # dry-run wrote nothing

    unretire_args = cli.build_parser().parse_args(["--repo", str(repo), "fleet", "unretire"])
    result = cli.run_fleet_unretire(unretire_args)
    assert result.ok is True
    assert repo_is_retired(repo) is False


def test_status_and_doctor_calls_leave_a_retired_repo_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The auto-registration on every CLI call never clears retirement."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    _write_config(repo, retired=True)
    fleet_dir = tmp_path / "fleet"

    monkeypatch.setattr("charlie_work.cli.github_client_for", lambda *_a, **_k: _FakeGitHub())
    monkeypatch.setattr("charlie_work.cli.OrchestratorApp", lambda *_a, **_k: MagicMock())

    status_args = cli.build_parser().parse_args(
        ["--repo", str(repo), "--fleet-dir", str(fleet_dir), "roll-call"]
    )
    cli.build_app(status_args)
    registry = json.loads((fleet_dir / "fleet.json").read_text(encoding="utf-8"))
    assert registry["repos"]["local/localrepo"]["retired"] is True

    monkeypatch.setattr("charlie_work.cli.run_doctor", lambda *_a, **_k: (True, []))
    doctor_args = cli.build_parser().parse_args(
        ["--repo", str(repo), "--fleet-dir", str(fleet_dir), "doctor"]
    )
    cli.run_doctor_command(doctor_args)
    registry = json.loads((fleet_dir / "fleet.json").read_text(encoding="utf-8"))
    assert registry["repos"]["local/localrepo"]["retired"] is True


def test_select_repos_exposes_the_derived_retired_field(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_config(repo, retired=True)
    entry = _repo_entry(repo, config_path)
    registry = {"repos": {"local/localrepo": entry}}

    selected = _select_repos(registry, None)

    assert selected[0][1]["retired"] is True
    assert entry_is_retired(entry) is True


def test_repo_live_work_quiescent_literal() -> None:
    assert LiveWork().quiescent is True
    assert LiveWork(workers=1).quiescent is False


def test_run_wind_down_reaps_and_ingests_verdicts_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Wind-down calls reap/verdict helpers and no launch/dispatch lane."""
    from charlie_work import repo_retirement

    app = MagicMock()
    app.paths.state_file = tmp_path / "state.json"
    app.fleet_dir_override = None
    app._layout.sessions_dir = tmp_path / "sessions"
    app._run_review_reap_sweeps.return_value = {
        "stalled": [],
        "verdict_result": {"recorded": [1]},
    }
    stalled = MagicMock(return_value=[])
    reaped = MagicMock(return_value=[])
    orphaned = MagicMock(return_value=[])
    monkeypatch.setattr("charlie_work.workflow._detect_and_handle_stalled_sessions", stalled)
    monkeypatch.setattr(
        "charlie_work.workflow._classify_dead_sessions_and_update_throttle_state", reaped
    )
    monkeypatch.setattr("charlie_work.workflow._detect_and_handle_orphaned_workers", orphaned)

    summary = repo_retirement.run_wind_down(app)

    stalled.assert_called_once()
    reaped.assert_called_once()
    orphaned.assert_called_once()
    app._run_review_reap_sweeps.assert_called_once()
    app.loop.assert_not_called()
    app.dispatch.assert_not_called()
    app.dispatch_reviews.assert_not_called()
    app._local_lane.assert_not_called()
    assert summary["review_sweep"]["recorded"] == 1
