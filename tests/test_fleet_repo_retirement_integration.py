"""End-to-end retirement tests (issue #2754): real wind-down and a full pass.

The unit tests pin each skip independently; these drive a real
``OrchestratorApp`` over a real (no-remote) local-issues repo so the
wind-down contract is proven by the reaper's own side effects -- the dead
worker's sidecar disappears because it was reaped, not because a test
unlinked it -- and a retired but worker-eligible repo cannot launch
anything through ``fleet_loop``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from _fleet_dispatch_fixtures import (  # noqa: F401  (autouse hermeticity)
    _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests,
)
from _local_lane_fixtures import _init_repo
from charlie_work.config import build_config_from_data
from charlie_work.fleet_dispatch import fleet_loop
from charlie_work.instrumentation import query_events
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import runtime_paths
from charlie_work.repo_retirement import (
    repo_live_work,
    retired_pass_disposition,
    run_wind_down,
)
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _retirement_data(*, retired: bool = True) -> dict[str, Any]:
    return {
        "fleet": {"retired": retired},
        "local_issues": {"enabled": True, "issues_dir": "docs/issues"},
        "dispatch": {"test_command": 'python -c "pass"'},
        # A bare test environment has no real sessions.db, so the probe is
        # inconclusive; 0 reaps on the first pass instead of deferring.
        "watchdog": {"max_inconclusive_probe_deferrals": 0},
        "post_mortem": {"enabled": False},
    }


def _write_repo_config(repo: Path, *, retired: bool = True) -> Path:
    path = repo / "orchestrator.config.yaml"
    path.write_text(
        yaml.safe_dump(_retirement_data(retired=retired), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_local_issue(
    issues_dir: Path,
    number: int,
    *,
    labels: tuple[str, ...],
    state: str = "open",
) -> Path:
    now = datetime.now(UTC)
    issues_dir.mkdir(parents=True, exist_ok=True)
    path = issues_dir / f"{number:03d}_{now:%Y-%m-%d}_retirement.md"
    path.write_text(
        "---\n"
        f'title: "Retirement issue {number}"\n'
        f"state: {state}\n"
        f"labels: [{', '.join(labels)}]\n"
        f'created: "{now:%Y-%m-%d}"\n'
        'author: "test"\n'
        "---\n"
        "Body text.\n",
        encoding="utf-8",
    )
    return path


def _app(repo: Path, issues_dir: Path, config: Any) -> OrchestratorApp:
    gh = LocalFileGitHub(repo_root=repo, issues_dir=issues_dir)
    paths = runtime_paths(repo, config.runtime.state_dir)
    return OrchestratorApp(repo, paths, config, gh)


def _seed_dispatched_issue(app: OrchestratorApp, number: int, branch: str) -> None:
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state.setdefault("issues", {})[str(number)] = {
            "number": number,
            "title": f"Retirement issue {number}",
            "status": "dispatched",
            "branch_name": branch,
            "worker_pid": 99999,
            "worker_process_start_time": 1.0,
        }
        save_state(app.paths.state_file, state)


def _write_dead_session(app: OrchestratorApp, repo: Path, number: int, branch: str) -> Path:
    sessions = app._layout.sessions_dir
    sessions.mkdir(parents=True, exist_ok=True)
    log_path = sessions / f"issue-{number}.log"
    log_path.write_text("Session log\n", encoding="utf-8")
    sidecar = sessions / f"issue-{number}.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": number,
                "branch": branch,
                "worktree_path": str(repo / "worktrees" / f"issue-{number}"),
                "prompt_path": str(repo / "prompt.md"),
                "command": ["devin", "--prompt-file", str(repo / "prompt.md")],
                "pid": 99999,
                "started_at": _now_iso(),
                "log_path": str(log_path),
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def test_real_wind_down_reaps_a_dead_worker_and_leaves_no_live_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real reaper, real app: the sidecar is gone because it was reaped."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    issues_dir = repo / "docs" / "issues"
    _write_repo_config(repo, retired=True)
    config = build_config_from_data(_retirement_data(retired=True))
    app = _app(repo, issues_dir, config)
    number = 1
    branch = f"agent/issue-{number}"
    _write_local_issue(issues_dir, number, labels=(config.labels.in_progress,))
    _seed_dispatched_issue(app, number, branch)
    sidecar = _write_dead_session(app, repo, number, branch)
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: False)

    summary = run_wind_down(app)

    assert summary["retired"] is True
    assert summary["wind_down"] is True
    assert summary["reaped"] == 1
    assert not sidecar.exists(), "the wind-down reaper must remove the dead sidecar"
    live = repo_live_work(repo, app.config, app.paths.root)
    assert live.quiescent is True
    assert retired_pass_disposition("owner/retired", repo, app.config, app.paths.root, {}) is True


def test_retired_worker_eligible_repo_launches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full pass: wind-down only, then the quiescent retired repo is skipped."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    issues_dir = repo / "docs" / "issues"
    config_path = _write_repo_config(repo, retired=True)
    config = build_config_from_data(_retirement_data(retired=True))
    number = 1
    branch = f"agent/issue-{number}"
    _write_local_issue(issues_dir, number, labels=(config.labels.ready,))
    # A dead worker makes the first pass run the real wind-down lane while a
    # ready issue sits eligible for a fresh dispatch.
    app = _app(repo, issues_dir, config)
    _seed_dispatched_issue(app, number, branch)
    sidecar = _write_dead_session(app, repo, number, branch)

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    state_dir = repo / ".var" / "charlie-work"
    state_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "repo_root": str(repo),
        "name_with_owner": "owner/retired",
        "config_path": str(config_path),
        "state_dir": str(state_dir),
        "first_seen": _now_iso(),
        "last_seen": _now_iso(),
    }
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": {"owner/retired": entry}}),
        encoding="utf-8",
    )

    worker_launches: list[object] = []
    reviewer_launches: list[object] = []
    suite_launches: list[object] = []
    tracker_commits: list[object] = []

    monkeypatch.setattr(
        "charlie_work.adapters.dispatch_sessions",
        lambda *args, **kwargs: worker_launches.append(args) or [],
    )
    for name in ("launch_claude_worker", "launch_devin_session", "launch_api_worker"):
        monkeypatch.setattr(
            f"charlie_work.workflow.{name}",
            lambda *args, _name=name, **kwargs: reviewer_launches.append(_name),
        )
    monkeypatch.setattr(
        "charlie_work.local_suite_runner.launch_suite_gate",
        lambda *args, **kwargs: suite_launches.append(args),
    )
    monkeypatch.setattr(
        "charlie_work.orchestration.local_tracker_flush.flush_tracker_writes",
        lambda *args, **kwargs: tracker_commits.append(args),
    )
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._run_fleet_allocation_prologue",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._run_fleet_autoscale_prologue",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch._run_fleet_config_retirement_sweep",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "charlie_work.fleet_dispatch.compute_api_worker_fleet_report",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: False)

    first = fleet_loop(
        fleet_dir_override=str(fleet_dir),
        global_config=None,
        limit=2,
        dry_run=False,
        work_only=False,
    )

    assert first.data["retired"] == ["owner/retired"]
    assert first.data["repos"]["owner/retired"]["wind_down"] is True
    assert not sidecar.exists()
    assert worker_launches == []
    assert reviewer_launches == []
    assert suite_launches == []
    assert tracker_commits == []
    assert query_events(app.paths.state_file, kind="local_lane_kill_switch_stalled") == []

    second = fleet_loop(
        fleet_dir_override=str(fleet_dir),
        global_config=None,
        limit=2,
        dry_run=False,
        work_only=False,
    )

    assert second.data["retired_quiescent"] == ["owner/retired"]
    assert "owner/retired" not in second.data["repos"]
    assert worker_launches == []
    assert reviewer_launches == []
    assert suite_launches == []
    assert tracker_commits == []
    assert query_events(app.paths.state_file, kind="local_lane_kill_switch_stalled") == []
