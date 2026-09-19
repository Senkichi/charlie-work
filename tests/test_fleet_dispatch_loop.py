"""Per-repo lane mechanics for ``fleet_loop``: dispatch, label ensure, lock handling.

Split out of ``tests/test_fleet_dispatch.py`` (issue #1557, Track 1) --
bodies are verbatim relocations; shared helpers and the autouse hermeticity
fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from _fleet_dispatch_fixtures import (
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work import layout
from charlie_work.config import OrchestratorConfig
from charlie_work.fleet_dispatch import (
    ApiWorkerFleetReport,
    fleet_loop,
)
from charlie_work.instrumentation import query_events
from charlie_work.workflow import CommandResult


@patch("charlie_work.fleet_dispatch.compute_api_worker_fleet_report")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_api_worker_report_none_when_unconfigured(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_compute_report: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop sets api_worker_report to None when no repo configures the section."""
    mock_load_registry.return_value = {"repos": {}}
    mock_compute_report.return_value = None

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=1,
        merge=None,
        dry_run=True,
        work_only=True,
    )

    assert result.ok is True
    assert result.data["api_worker_report"] is None


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_calls_loop_per_repo(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop calls app.loop() exactly once per repo with that repo's config."""
    # Setup registry
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    # Create temp repo dirs
    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    # Mock config and paths
    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    # Mock OrchestratorApp instances
    mock_app1 = MagicMock()
    mock_app2 = MagicMock()
    mock_app1.loop.return_value = CommandResult(True, "repo1 loop complete", {})
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.side_effect = [mock_app1, mock_app2]

    # Mock GitHub
    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    # Run fleet_loop
    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # Verify loop() was called exactly once per repo
    assert mock_app1.loop.call_count == 1
    assert mock_app2.loop.call_count == 1

    # Verify loop() was called with correct args
    mock_app1.loop.assert_called_once_with(3, merge=True)
    mock_app2.loop.assert_called_once_with(3, merge=True)

    # Verify result includes both repos
    assert "repos" in result.data
    assert "owner/repo1" in result.data["repos"]
    assert "owner/repo2" in result.data["repos"]


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_ensure_labels_calls_ensure_per_repo(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #1339: fleet_loop(ensure_labels=True) calls app.ensure_labels()
    once per repo before its lane, so a new LabelConfig field converges to its
    label on the supervisor's first pass.
    """
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry
    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app1 = MagicMock()
    mock_app2 = MagicMock()
    mock_app1.loop.return_value = CommandResult(True, "repo1 loop complete", {})
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.side_effect = [mock_app1, mock_app2]
    mock_gh_class.return_value = MagicMock()

    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
        ensure_labels=True,
    )

    mock_app1.ensure_labels.assert_called_once()
    mock_app2.ensure_labels.assert_called_once()
    # The lane still ran after the ensure.
    mock_app1.loop.assert_called_once_with(3, merge=True)


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_ensure_labels_failure_does_not_block_lane(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """Issue #1339 AC #2: an ensure_labels exception must not block the lane."""
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            }
        }
    }
    mock_load_registry.return_value = registry
    (tmp_path / "repo1").mkdir()

    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(True, "repo1 loop complete", {})
    mock_app.ensure_labels.side_effect = RuntimeError("boom")
    mock_app_class.side_effect = [mock_app]
    mock_gh_class.return_value = MagicMock()

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
        ensure_labels=True,
    )

    # The lane still ran despite the ensure raising.
    mock_app.loop.assert_called_once_with(3, merge=True)
    assert "owner/repo1" in result.data["repos"]


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_missing_repo_root_records_stale_event_to_daemon(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """#1372: a repo whose repo_root no longer exists is STALE, not a failing
    lane. The ``fleet_registry_stale_entry`` warning is emitted into the
    DAEMON's own events.db (fleet_state_path), never into the dead entry's
    recorded state_dir — which would resurrect a zombie directory under %TEMP%
    via log_event's auto-mkdir (#746). The pass completes with ok=True."""
    repo1_state_dir = tmp_path / "repo1-state"
    repo1_state_dir.mkdir(parents=True)
    repo2 = tmp_path / "repo2"
    repo2.mkdir()

    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "nonexistent"),
                "state_dir": str(repo1_state_dir),
            },
            "owner/repo2": {
                "repo_root": str(repo2),
            },
        }
    }
    mock_load_registry.return_value = registry

    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app2 = MagicMock()
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.return_value = mock_app2
    mock_gh_class.return_value = MagicMock()

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # repo1 is stale (ok=True), repo2 still runs.
    assert result.data["repos"]["owner/repo1"]["ok"] is True
    assert result.data["repos"]["owner/repo1"].get("stale") is True
    assert "stale entry skipped" in result.data["repos"]["owner/repo1"]["message"]
    assert result.data["repos"]["owner/repo2"]["ok"] is True
    assert mock_app_class.call_count == 1
    assert mock_app2.loop.call_count == 1
    assert mock_load_layered_config.call_count == 1

    # The stale warning is recorded to the DAEMON's events.db, not repo1's.
    fleet_state_path = layout.state_file_path(tmp_path / "fleet")
    daemon_recorded = query_events(fleet_state_path, kind="fleet_registry_stale_entry")
    assert len(daemon_recorded) == 1
    assert daemon_recorded[0]["level"] == "warning"
    assert daemon_recorded[0]["payload"]["repo_key"] == "owner/repo1"
    assert daemon_recorded[0]["payload"]["reason"] == "repo_root_missing"

    # NOTHING is written to the dead entry's state_dir — no zombie directory
    # is resurrected under the recorded state_dir path.
    repo1_state_path = layout.state_file_path(repo1_state_dir)
    repo1_recorded = query_events(repo1_state_path, kind="fleet_pass_config_error")
    assert len(repo1_recorded) == 0

    # The fleet digest does NOT carry an ERROR entry for the stale repo.
    digest_events = result.data["digest"]["events"]
    stale_digest_events = [e for e in digest_events if e.get("repo_key") == "owner/repo1"]
    assert len(stale_digest_events) == 0


@patch("charlie_work.fleet_dispatch._load_registry")
def test_fleet_loop_missing_repo_root_skipped(
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop skips repos with missing repo_root as stale (issue #1372).

    A stale entry is not a failing lane: ok=True, pass_skipped=True, and the
    repo is listed under the result's ``stale`` key for prune-after-grace.
    """
    # Setup registry with one missing repo
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "nonexistent"),
                "config_path": "orchestrator.config.yaml",
            }
        }
    }
    mock_load_registry.return_value = registry

    # Run fleet_loop
    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # Verify result includes the stale repo (ok=True, not a failure)
    assert "repos" in result.data
    assert "owner/repo1" in result.data["repos"]
    assert result.data["repos"]["owner/repo1"]["ok"] is True
    assert result.data["repos"]["owner/repo1"].get("stale") is True
    assert result.data["repos"]["owner/repo1"].get("pass_skipped") is True
    assert "stale entry skipped" in result.data["repos"]["owner/repo1"]["message"]
    # The stale key is collected for prune-after-grace
    assert "owner/repo1" in result.data.get("stale", [])


@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch._load_registry")
def test_fleet_loop_releases_lock_after_each_repo(
    mock_load_registry: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_gh_class: MagicMock,
    mock_app_class: MagicMock,
    mock_try_acquire: MagicMock,
    tmp_path: Path,
) -> None:
    """The per-repo supervisor lock is released after each repo, and the next repo can acquire it."""
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_runtime_paths.return_value = mock_paths

    mock_app1 = MagicMock()
    mock_app2 = MagicMock()
    mock_app1.loop.return_value = CommandResult(True, "repo1 loop complete", {})
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.side_effect = [mock_app1, mock_app2]

    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    lock1 = MagicMock()
    lock2 = MagicMock()
    mock_try_acquire.side_effect = [lock1, lock2]

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    assert result.ok is True
    assert mock_app1.loop.call_count == 1
    assert mock_app2.loop.call_count == 1
    assert lock1.release.called
    assert lock2.release.called


@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch._load_registry")
def test_fleet_loop_skips_repo_when_supervisor_lock_held(
    mock_load_registry: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_gh_class: MagicMock,
    mock_app_class: MagicMock,
    mock_try_acquire: MagicMock,
    tmp_path: Path,
) -> None:
    """If the supervisor lock is held, the repo is skipped and others are still processed."""
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_runtime_paths.return_value = mock_paths

    mock_app2 = MagicMock()
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.return_value = mock_app2

    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    lock = MagicMock()
    # repo1 is held (external supervised loop), repo2 is free
    mock_try_acquire.side_effect = [None, lock]

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # repo1 was skipped because the lock was held
    assert "owner/repo1" in result.data["repos"]
    repo1_data = result.data["repos"]["owner/repo1"]
    assert repo1_data["ok"] is True
    assert repo1_data["pass_skipped"] is True
    assert repo1_data["reason"] == "supervisor_lock_held"

    # repo2 still ran
    assert "owner/repo2" in result.data["repos"]
    assert result.data["repos"]["owner/repo2"]["ok"] is True
    assert mock_app2.loop.call_count == 1

    # The lock taken for repo2 was released
    lock.release.assert_called_once()

    # The fleet digest surfaces the skipped repo
    skipped_events = [e for e in result.data["digest"]["events"] if e["type"] == "skipped"]
    assert len(skipped_events) == 1
    assert skipped_events[0]["repo_key"] == "owner/repo1"
    assert skipped_events[0]["reason"] == "supervisor_lock_held"


@patch("charlie_work.fleet_dispatch.compute_api_worker_fleet_report")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_threads_api_worker_report_into_data(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_compute_report: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop places compute_api_worker_fleet_report's dict into CommandResult.data.

    The standalone compute function is covered by the tests above; this
    verifies the fleet_loop wiring (the api_worker_report key in the returned
    CommandResult.data) so a silent breakage in the key-lookup path can't ship
    undetected. An empty registry means no per-repo work runs.
    """
    mock_load_registry.return_value = {"repos": {}}
    report = ApiWorkerFleetReport(
        provider="kimi-k3",
        today_usd=1.50,
        lifetime_usd=7.25,
        cap_usd=15.00,
        live=2,
        enabled_k=1,
        enabled_m=4,
    )
    mock_compute_report.return_value = report

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=1,
        merge=None,
        dry_run=True,
        work_only=True,
    )

    assert result.ok is True
    assert result.data["api_worker_report"] == report.to_dict()
    assert result.data["api_worker_report"]["provider"] == "kimi-k3"
    assert "line" in result.data["api_worker_report"]
    # The compute function is called with the fleet_dir override and the
    # configs fleet_loop already loaded this pass (no redundant reload).
    mock_compute_report.assert_called_once_with(
        fleet_dir_override=str(tmp_path / "fleet"),
        preloaded_configs={},
    )


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_work_only_calls_dispatch(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop with work_only=True calls app.dispatch() instead of loop()."""
    # Setup registry
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            }
        }
    }
    mock_load_registry.return_value = registry

    # Create temp repo dir
    (tmp_path / "repo1").mkdir()

    # Mock config and paths
    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    # Mock OrchestratorApp
    mock_app = MagicMock()
    mock_app.dispatch.return_value = CommandResult(True, "repo1 dispatch complete", {})
    mock_app_class.return_value = mock_app

    # Mock GitHub
    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    # Run fleet_loop with work_only=True
    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=None,
        dry_run=False,
        work_only=True,
    )

    # Verify dispatch() was called instead of loop()
    assert mock_app.dispatch.call_count == 1
    assert mock_app.loop.call_count == 0

    # Verify dispatch() was called with correct args
    mock_app.dispatch.assert_called_once_with(3)


@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch._load_registry")
def test_fleet_loop_work_only_skips_locked_repo(
    mock_load_registry: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_gh_class: MagicMock,
    mock_app_class: MagicMock,
    mock_try_acquire: MagicMock,
    tmp_path: Path,
) -> None:
    """The dispatch-only (work_only) path also respects the supervisor lock."""
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            }
        }
    }
    mock_load_registry.return_value = registry

    (tmp_path / "repo1").mkdir()

    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_runtime_paths.return_value = mock_paths

    mock_app = MagicMock()
    mock_app_class.return_value = mock_app

    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    mock_try_acquire.return_value = None

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=None,
        dry_run=False,
        work_only=True,
    )

    assert mock_app.dispatch.call_count == 0
    repo_data = result.data["repos"]["owner/repo1"]
    assert repo_data["ok"] is True
    assert repo_data["pass_skipped"] is True
    assert repo_data["reason"] == "supervisor_lock_held"
