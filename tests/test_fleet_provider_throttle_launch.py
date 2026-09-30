"""App-level staggered-resume wiring for both worker lanes (issue #1993).

The gate's pure decisions are covered in ``test_fleet_provider_throttle.py``;
these tests drive the real ``dispatch`` / ``dispatch_rework`` passes so the
``admit_one`` launch cap and the probe stamp cannot be deleted silently.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _fakes_github import FakeGitHub
from charlie_work import fleet_provider_throttle as fpt
from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import DevinConfig, DispatchConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, set_throttled_until, state_lock
from charlie_work.workflow import OrchestratorApp

ADAPTER = "command"
ISSUES = (123, 124, 125)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _config() -> OrchestratorConfig:
    return OrchestratorConfig(
        dispatch=DispatchConfig(default_limit=3),
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness=ADAPTER),
    )


class _MultiIssueGitHub(FakeGitHub):
    """Three eligible issues; open PRs (rework) or closed PRs (fresh dispatch)."""

    def __init__(self, rework_label: str | None = None) -> None:
        super().__init__()
        label = rework_label or self.issues[0]["labels"][0]["name"]
        self.issues = [
            {**self.issues[0], "number": n, "title": f"Issue {n}", "labels": [{"name": label}]}
            for n in ISSUES
        ]
        self.prs = [
            {
                **self.prs[0],
                "number": n + 333,
                "title": f"Fix #{n}",
                "headRefName": f"agent/issue-{n}-fix",
                "headRefOid": f"sha-{n}",
                "body": f"Closes #{n}\n\nTests: covered.",
                "state": "CLOSED" if rework_label is None else "OPEN",
            }
            for n in ISSUES
        ]


def _app(tmp_path: Path, name: str, fleet: Path, *, rework: bool = False, dry_run: bool = False):
    root = tmp_path / name
    root.mkdir()
    config = _config()
    paths = runtime_paths(root, config.runtime.state_dir)
    gh = _MultiIssueGitHub(config.labels.needs_rework if rework else None)
    if rework:
        paths.root.mkdir(parents=True, exist_ok=True)
        with state_lock(paths.state_file):
            state = load_state(paths.state_file)
            for n in ISSUES:
                state["issues"][str(n)] = {
                    "number": n,
                    "title": f"Issue {n}",
                    "url": f"https://example.test/issues/{n}",
                    "status": "rework_requested",
                    "branch_name": f"agent/issue-{n}-fix",
                }
                state["prs"][str(n + 333)] = {
                    "number": n + 333,
                    "issue_number": n,
                    "decision": "request_changes",
                    "reviewed_head_sha": f"sha-{n}",
                }
            save_state(paths.state_file, state)
        for n in ISSUES:
            pr_dir = root / ".var" / "charlie-work" / "prs" / f"pr-{n + 333}"
            pr_dir.mkdir(parents=True)
            (pr_dir / "rework-prompt.md").write_text("Fix it", encoding="utf-8")
    app = OrchestratorApp(root, paths, config, gh, dry_run=dry_run, fleet_dir_override=str(fleet))
    return app, paths


def _register(fleet: Path, *paths_list) -> None:
    fleet.mkdir(parents=True, exist_ok=True)
    repos = {f"o/r{i}": {"state_dir": str(p.state_file.parent)} for i, p in enumerate(paths_list)}
    (fleet / "fleet.json").write_text(json.dumps({"repos": repos}), encoding="utf-8")


def _expire_window(paths, *, minutes_ago: int = 1) -> None:
    until = _iso(datetime.now(UTC) - timedelta(minutes=minutes_ago))
    save_state(
        paths.state_file,
        set_throttled_until(
            load_state(paths.state_file),
            until,
            reason="rate_limited",
            adapter_kind=ADAPTER,
            source="test",
        ),
    )


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Capture launched issue numbers; every launch succeeds with a live pid."""
    launched: list[int] = []

    def fake_dispatch_sessions(_root, _manifest, _results, _settings, requests):
        launched.extend(r.issue_number for r in requests)
        return [
            SessionDispatchResult(
                issue_number=r.issue_number,
                issue_title=r.issue_title,
                prompt_path=str(r.prompt_path),
                branch_name=r.branch_name,
                adapter=ADAPTER,
                ok=True,
                pid=os.getpid(),
            )
            for r in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)
    return launched


def _stamp(fleet: Path) -> dict:
    return json.loads(fpt.resume_probe_path(str(fleet)).read_text(encoding="utf-8"))


def _lane(app, lane: str, **kw):
    return app.dispatch(**kw) if lane == "dispatch" else app.dispatch_rework(**kw)


@pytest.mark.parametrize("lane", ["dispatch", "rework"])
def test_expired_window_admits_exactly_one_launch_and_stamps_probe(
    tmp_path: Path, launches: list[int], lane: str
) -> None:
    fleet = tmp_path / "fleet"
    app, paths = _app(tmp_path, "a", fleet, rework=lane == "rework")
    _register(fleet, paths)
    _expire_window(paths)

    _lane(app, lane, limit=3)

    assert len(launches) == 1
    probe = _stamp(fleet)[ADAPTER]
    assert [p["pid"] for p in probe["probes"]] == [os.getpid()]


@pytest.mark.parametrize("lane", ["dispatch", "rework"])
def test_second_repo_in_same_pass_defers_while_probe_lives(
    tmp_path: Path, launches: list[int], lane: str
) -> None:
    fleet = tmp_path / "fleet"
    app_a, paths_a = _app(tmp_path, "a", fleet, rework=lane == "rework")
    app_b, paths_b = _app(tmp_path, "b", fleet, rework=lane == "rework")
    _register(fleet, paths_a, paths_b)
    _expire_window(paths_a)

    _lane(app_a, lane, limit=3)
    assert len(launches) == 1
    result = _lane(app_b, lane, limit=3)

    assert len(launches) == 1
    assert result.ok is False
    assert result.data["deferred_reason"] == "provider_resume_staggered"


@pytest.mark.parametrize("lane", ["dispatch", "rework"])
def test_dry_run_neither_launches_nor_stamps(
    tmp_path: Path, launches: list[int], lane: str
) -> None:
    fleet = tmp_path / "fleet"
    app, paths = _app(tmp_path, "a", fleet, rework=lane == "rework", dry_run=True)
    _register(fleet, paths)
    _expire_window(paths)

    _lane(app, lane, limit=3)

    assert launches == []
    assert not fpt.resume_probe_path(str(fleet)).exists()


def test_probe_that_died_without_a_window_does_not_open_the_fleet(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    app, paths = _app(tmp_path, "a", fleet)
    _register(fleet, paths)
    _expire_window(paths, minutes_ago=120)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    fpt.note_probe_launch(
        ADAPTER,
        [(child.pid, None)],
        fleet_dir_override=str(fleet),
        now=datetime.now(UTC) - timedelta(hours=1),
    )

    decision = fpt.decide_for_app(app)

    # An hour past launch with no newer window: the old logic reopened here.
    assert decision.action == "admit_one"
    assert not decision.probe_survived


def test_live_probe_past_survival_period_opens_and_survival_is_persisted(
    tmp_path: Path,
) -> None:
    fleet = tmp_path / "fleet"
    app, paths = _app(tmp_path, "a", fleet)
    _register(fleet, paths)
    _expire_window(paths, minutes_ago=30)
    fpt.note_probe_launch(
        ADAPTER,
        [(os.getpid(), None)],
        fleet_dir_override=str(fleet),
        now=datetime.now(UTC) - timedelta(seconds=fpt.RESUME_SURVIVAL_SECONDS + 5),
    )

    assert fpt.decide_for_app(app).action == "open"
    assert "survived_at" in _stamp(fleet)[ADAPTER]
    # Once survival is persisted, the probe process exiting must not re-arm probe mode.
    stamp = _stamp(fleet)
    stamp[ADAPTER]["probes"] = [{"pid": 0, "process_start_time": None}]
    fpt.resume_probe_path(str(fleet)).write_text(json.dumps(stamp), encoding="utf-8")
    assert fpt.decide_for_app(app).action == "open"
