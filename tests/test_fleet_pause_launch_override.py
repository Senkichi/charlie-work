"""``--ignore-pause`` reaches the worker launch gate (issue #2766).

The launch gate (``worker_launch_gate.issue_worker_launch_permit``) defers every
worker launch while the fleet is paused, so the one-shot override from #2660
only still means "run" if it travels from the CLI flag onto each per-repo app.
These tests pin the three hops: ``cli.main`` -> ``fleet_loop`` / ``build_app`` ->
``OrchestratorApp``. The gate's own behaviour lives in
``tests/test_worker_launch_gate.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from _fleet_dispatch_fixtures import (
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
    _per_repo_runtime_paths,
)
from charlie_work import cli
from charlie_work.command_result import CommandResult
from charlie_work.config import OrchestratorConfig
from charlie_work.fleet_dispatch import fleet_loop
from charlie_work.fleet_pause import ignore_pause_requested, write_fleet_pause
from charlie_work.supervise import SelfDeployResult


def test_ignore_pause_requested_reads_only_the_flag() -> None:
    assert ignore_pause_requested(argparse.Namespace(ignore_pause=True)) is True
    assert ignore_pause_requested(argparse.Namespace(ignore_pause=False)) is False
    # A command that does not carry the flag (hand-built Namespaces, unmarked parsers).
    assert ignore_pause_requested(argparse.Namespace()) is False


@pytest.mark.parametrize("subcommand", ["work", "bash-rats"])
@pytest.mark.parametrize("flag", [True, False])
def test_fleet_entrypoints_hand_the_override_to_fleet_loop(
    subcommand: str, flag: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fleet = tmp_path / "paused-fleet"
    fleet.mkdir()
    write_fleet_pause(str(fleet), reason="maintenance window")
    loop = MagicMock(return_value=CommandResult(True, "ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", loop)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: None)
    deploy = SelfDeployResult(
        ok=True, pulled=True, changed=False, synced=False, message="up to date"
    )
    monkeypatch.setattr(cli, "self_deploy", MagicMock(return_value=deploy))
    argv = ["--fleet-dir", str(fleet), "fleet", subcommand, *(["--ignore-pause"] if flag else [])]

    rc = cli.main(argv)

    # Paused and not overridden: refused before fleet_loop (#2660); overridden: runs.
    assert rc == (0 if flag else 5)
    if flag:
        assert loop.call_args.kwargs["ignore_pause"] is True
    else:
        loop.assert_not_called()


def test_build_app_carries_the_override_onto_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = MagicMock(name="ctx")
    monkeypatch.setattr(cli, "bootstrap_command", lambda _args: ctx)
    monkeypatch.setattr(cli, "touch_repo", lambda *a, **k: None)
    app_class = MagicMock(name="OrchestratorApp")
    monkeypatch.setattr(cli, "OrchestratorApp", app_class)

    def _args(**extra: Any) -> argparse.Namespace:
        return argparse.Namespace(
            command="work", repo="explicit", fleet_dir=None, dry_run=False, **extra
        )

    cli.build_app(_args(ignore_pause=True))
    assert app_class.call_args.kwargs["ignore_pause"] is True
    cli.build_app(_args(ignore_pause=False))
    assert app_class.call_args.kwargs["ignore_pause"] is False
    # An unmarked command has no such attribute at all: the gate stays armed.
    cli.build_app(_args())
    assert app_class.call_args.kwargs["ignore_pause"] is False


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
@pytest.mark.parametrize("override", [None, True, False])
def test_fleet_loop_builds_each_lane_app_with_the_override(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    override: bool | None,
    tmp_path: Path,
) -> None:
    """``None`` is the supervisor's call (it never passes the flag): the gate stays armed."""
    names = ("repo1", "repo2")
    mock_load_registry.return_value = {
        "repos": {
            f"owner/{name}": {
                "repo_root": str(tmp_path / name),
                "config_path": "orchestrator.config.yaml",
            }
            for name in names
        }
    }
    for name in names:
        (tmp_path / name).mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_app_class.return_value.loop.return_value = CommandResult(True, "loop complete", {})
    mock_gh_class.return_value = MagicMock()
    extra: dict[str, Any] = {} if override is None else {"ignore_pause": override}

    fleet_loop(fleet_dir_override=str(tmp_path / "fleet"), merge=True, **extra)

    assert mock_app_class.call_count == len(names)
    assert [c.kwargs["ignore_pause"] for c in mock_app_class.call_args_list] == [
        bool(override)
    ] * len(names)
