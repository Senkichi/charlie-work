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
from charlie_work import role_quota_ledger
from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import DevinConfig, DispatchConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.paths import runtime_paths
from charlie_work.role_chain import RoleEntry
from charlie_work.state import load_state, save_state, set_throttled_until, state_lock
from charlie_work.workflow import OrchestratorApp

ADAPTER = "command"
ISSUES = (123, 124, 125)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _config(worker: WorkerRoleConfig | None = None) -> OrchestratorConfig:
    return OrchestratorConfig(
        dispatch=DispatchConfig(default_limit=3),
        devin=DevinConfig(),
        worker=worker or WorkerRoleConfig(harness=ADAPTER),
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


def _app(
    tmp_path: Path,
    name: str,
    fleet: Path,
    *,
    rework: bool = False,
    dry_run: bool = False,
    worker: WorkerRoleConfig | None = None,
):
    root = tmp_path / name
    root.mkdir()
    config = _config(worker)
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


def _expire_window(
    paths, *, minutes_ago: int = 1, adapter_kind: str = ADAPTER, minutes_ahead: int = 0
) -> None:
    """Record a fleet-scoped window; expired by default, active when ``minutes_ahead``."""
    until = _iso(
        datetime.now(UTC) + timedelta(minutes=minutes_ahead - minutes_ago * (not minutes_ahead))
    )
    save_state(
        paths.state_file,
        set_throttled_until(
            load_state(paths.state_file),
            until,
            reason="rate_limited",
            adapter_kind=adapter_kind,
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

    monkeypatch.setattr("charlie_work.adapters.dispatch_sessions", fake_dispatch_sessions)
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


# --- worker.fallbacks: the gate follows the SELECTED chain entry (issue #1993 review) ---

DEVIN_PRIMARY = RoleEntry("devin-shell", "swe-2")
CLAUDE_FALLBACK = RoleEntry("claude-code", "claude-sonnet-5-5")
CHAINED = WorkerRoleConfig(
    harness=DEVIN_PRIMARY.harness, model=DEVIN_PRIMARY.model, fallbacks=(CLAUDE_FALLBACK,)
)


@pytest.fixture
def chained_launches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture the adapter each launch actually ran on; every launch succeeds."""
    adapters: list[str] = []

    def fake_dispatch_sessions(_root, _manifest, _results, settings, requests):
        adapters.extend(settings.adapter for _ in requests)
        return [
            SessionDispatchResult(
                issue_number=r.issue_number,
                issue_title=r.issue_title,
                prompt_path=str(r.prompt_path),
                branch_name=r.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=os.getpid(),
            )
            for r in requests
        ]

    monkeypatch.setattr("charlie_work.adapters.dispatch_sessions", fake_dispatch_sessions)
    return adapters


def _restrict_primary() -> None:
    """Fleet quota ledger restricts the primary, so selection lands on the fallback."""
    assert role_quota_ledger.record_restriction(
        DEVIN_PRIMARY.harness,
        DEVIN_PRIMARY.model,
        datetime.now(UTC) + timedelta(hours=1),
        reason="quota_exhausted",
        source="test",
    )


def _chained_pair(tmp_path: Path, fleet: Path, lane: str):
    """Launching repo plus a peer repo that carries the fleet window."""
    launching, launching_paths = _app(
        tmp_path, "a", fleet, rework=lane == "rework", worker=CHAINED
    )
    _, peer_paths = _app(tmp_path, "b", fleet, rework=lane == "rework", worker=CHAINED)
    _register(fleet, launching_paths, peer_paths)
    _restrict_primary()
    return launching, peer_paths


@pytest.mark.parametrize("lane", ["dispatch", "rework"])
def test_fleet_window_on_primary_adapter_does_not_defer_fallback_launch(
    tmp_path: Path, chained_launches: list[str], lane: str
) -> None:
    fleet = tmp_path / "fleet"
    app, peer_paths = _chained_pair(tmp_path, fleet, lane)
    _expire_window(peer_paths, adapter_kind="devin", minutes_ahead=30)

    _lane(app, lane, limit=3)

    assert chained_launches
    assert set(chained_launches) == {"claude-code"}


@pytest.mark.parametrize("lane", ["dispatch", "rework"])
def test_fleet_window_on_fallback_adapter_defers_fallback_launch(
    tmp_path: Path, chained_launches: list[str], lane: str
) -> None:
    fleet = tmp_path / "fleet"
    app, peer_paths = _chained_pair(tmp_path, fleet, lane)
    _expire_window(peer_paths, adapter_kind="claude-code", minutes_ahead=30)

    result = _lane(app, lane, limit=3)

    assert chained_launches == []
    assert result.ok is False
    assert result.data["deferred_reason"] == "provider_throttled_fleet"


@pytest.mark.parametrize("lane", ["dispatch", "rework"])
def test_probe_stamp_records_the_launched_adapter_not_the_primary(
    tmp_path: Path, chained_launches: list[str], lane: str
) -> None:
    fleet = tmp_path / "fleet"
    app, peer_paths = _chained_pair(tmp_path, fleet, lane)
    _expire_window(peer_paths, adapter_kind="claude-code")

    _lane(app, lane, limit=3)

    assert chained_launches == ["claude-code"]
    stamp = _stamp(fleet)
    assert "claude-code" in stamp
    assert "devin" not in stamp
    assert [p["pid"] for p in stamp["claude-code"]["probes"]] == [os.getpid()]
