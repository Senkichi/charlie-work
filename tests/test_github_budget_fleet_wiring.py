"""``fleet_loop`` hands each lane the fleet-level state path (#2439).

The lane emits ``github_budget_pass`` into whatever ``fleet_state_path`` it is
given, so a call site that dropped the argument would silently disable the
accounting. ``fleet_state_path`` is a required parameter of the lane (a missing
one is a ``TypeError``); this drives the real call site end to end and checks
the event lands in the fleet-level ``events.db``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from _fake_transport import FakeAdapter, make_github, ok
from _fleet_dispatch_fixtures import (
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
    _per_repo_runtime_paths,
)
from charlie_work import layout
from charlie_work.command_result import CommandResult
from charlie_work.config import OrchestratorConfig
from charlie_work.fleet_dispatch import fleet_loop
from charlie_work.fleet_paths import fleet_dir
from charlie_work.github_transport.guarded import UsedCursor
from charlie_work.instrumentation import close_db, query_events


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_threads_the_fleet_state_path_into_the_lane(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    (tmp_path / "repo1").mkdir()
    mock_load_registry.return_value = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            }
        }
    }
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths

    gh, _, _ = make_github(
        tmp_path,
        http=FakeAdapter(
            "http",
            [
                ok([], headers={"X-RateLimit-Used": "7", "X-RateLimit-Reset": "9000"}),
                ok([], headers={"X-RateLimit-Used": "9", "X-RateLimit-Reset": "9000"}),
            ],
        ),
    )
    gh._transport_v2.budget._cursor = UsedCursor()
    mock_gh_class.return_value = gh

    def loop(*args, **kwargs) -> CommandResult:
        gh.run(["api", "repos/{owner}/{repo}/issues"])
        gh.run(["api", "repos/{owner}/{repo}/issues"])
        return CommandResult(True, "done", {})

    app = MagicMock()
    app.gh = gh
    app.loop.side_effect = loop
    mock_app_class.return_value = app

    fleet_root = tmp_path / "fleet"
    fleet_state = layout.state_file_path(fleet_dir(override=str(fleet_root)))
    result = fleet_loop(
        fleet_dir_override=str(fleet_root),
        global_config=None,
        repos=None,
        limit=1,
        merge=True,
        dry_run=False,
        work_only=False,
    )
    try:
        assert result.ok is True
        (event,) = query_events(fleet_state, kind="github_budget_pass")
    finally:
        close_db(fleet_state)
    payload = event["payload"]
    assert (payload["pass_kind"], payload["repo_key"]) == ("lane", "owner/repo1")
    assert (payload["requests"], payload["points"]) == (2, 2)
