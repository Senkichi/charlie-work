"""Issue #3089: ``dispatch`` / ``dispatch_rework`` Events report how long the
fleet launch lock was held (Host clock), and omit the field when none was.

Helpers are reinlined per file -- the zero cross-test-import guard forbids
``from test_* import``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import (
    DispatchConfig,
    FleetConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.host.fakes import FakeClock, FakeWorkerLauncher
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

FRESH = "fresh"
REWORK = "rework"
LANES = pytest.mark.parametrize("lane", [FRESH, REWORK])
_KIND = {FRESH: "dispatch", REWORK: "dispatch_rework"}
FIELD = "launch_lock_held_seconds"
LAUNCH_SECONDS = 2.5


def _app(tmp_path: Path, lane: str, *, fleet_cap: int, dry_run: bool = False) -> OrchestratorApp:
    config = OrchestratorConfig(
        worker=WorkerRoleConfig(harness="claude-code"),
        dispatch=DispatchConfig(default_limit=5),
        fleet=FleetConfig(global_max_concurrent_sessions=fleet_cap),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    fake_gh = FakeGitHub()
    if lane == REWORK:
        with state_lock(paths.state_file):
            state = load_state(paths.state_file)
            state["issues"]["123"] = {"number": 123, "status": "rework_requested"}
            state["prs"]["456"] = {"number": 456, "issue_number": 123}
            save_state(paths.state_file, state)
    else:
        fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(
        tmp_path,
        paths,
        config,
        fake_gh,
        dry_run=dry_run,
        fleet_dir_override=str(tmp_path / "fleet"),
    )
    if lane == REWORK:
        pr_dir = paths.prs / "pr-456"
        pr_dir.mkdir(parents=True, exist_ok=True)
        (pr_dir / "rework-prompt.md").write_text("rework prompt", encoding="utf-8")
    return app


def _install_hosts(fake_host: Any) -> FakeClock:
    clock = FakeClock(start=datetime(2026, 1, 1, tzinfo=UTC))

    def _launch(_repo_root, _manifest, _results, settings, requests):
        clock.advance(LAUNCH_SECONDS)  # time passes inside the locked window
        return [
            SessionDispatchResult(
                issue_number=r.issue_number,
                issue_title=r.issue_title,
                prompt_path=str(r.prompt_path),
                branch_name=r.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=4242,
                process_start_time=1.0,
            )
            for r in requests
        ]

    fake_host(clock=clock, worker_launch=FakeWorkerLauncher([_launch]))
    return clock


def _events(app: OrchestratorApp, lane: str) -> list[dict[str, Any]]:
    state = load_state(app.paths.state_file)
    return [e for e in state["events"] if e.get("kind") == _KIND[lane]]


def _run(app: OrchestratorApp, lane: str) -> Any:
    return app.dispatch() if lane == FRESH else app.dispatch_rework()


@LANES
def test_pass_holding_the_lock_reports_held_seconds(tmp_path: Path, lane: str, fake_host) -> None:
    _install_hosts(fake_host)
    app = _app(tmp_path, lane, fleet_cap=4)

    result = _run(app, lane)

    assert result.data["selected_count"] == 1, result.message
    (event,) = _events(app, lane)
    assert event["payload"][FIELD] == pytest.approx(LAUNCH_SECONDS)


@LANES
def test_pass_without_a_fleet_cap_omits_the_field(tmp_path: Path, lane: str, fake_host) -> None:
    _install_hosts(fake_host)
    app = _app(tmp_path, lane, fleet_cap=0)

    result = _run(app, lane)

    assert result.data["selected_count"] == 1, result.message
    (event,) = _events(app, lane)
    assert FIELD not in event["payload"]
