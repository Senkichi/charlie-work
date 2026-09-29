"""Fleet-scoped provider throttle + staggered resume (issue #1993)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work import fleet_provider_throttle as fpt
from charlie_work.config import (
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, set_throttled_until
from charlie_work.workflow import OrchestratorApp


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo(tmp_path: Path, name: str, harness: str):
    root = tmp_path / name
    root.mkdir()
    config = OrchestratorConfig(
        dispatch=DispatchConfig(default_limit=3),
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness=harness),
    )
    return root, runtime_paths(root, config.runtime.state_dir), config


def _throttle(paths, *, until: datetime, reason="rate_limited", adapter="devin") -> None:
    save_state(
        paths.state_file,
        set_throttled_until(
            load_state(paths.state_file), _iso(until), reason=reason, adapter_kind=adapter
        ),
    )


def _register(fleet: Path, *paths_list) -> None:
    fleet.mkdir(parents=True, exist_ok=True)
    repos = {f"o/r{i}": {"state_dir": str(p.state_file.parent)} for i, p in enumerate(paths_list)}
    (fleet / "fleet.json").write_text(json.dumps({"repos": repos}), encoding="utf-8")


def _rework_app(tmp_path: Path, name: str, harness: str, fleet: Path):
    root, paths, config = _repo(tmp_path, name, harness)
    app = OrchestratorApp(root, paths, config, FakeGitHub(), fleet_dir_override=str(fleet))
    return app, paths


def test_rate_limit_in_repo_a_defers_rework_in_repo_b(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    _, paths_a, _ = _repo(tmp_path, "a", "devin-shell")
    app_b, paths_b = _rework_app(tmp_path, "b", "devin-shell", fleet)
    _register(fleet, paths_a, paths_b)
    _throttle(paths_a, until=datetime.now(UTC) + timedelta(hours=1))

    result = app_b.dispatch_rework()

    assert result.ok is False
    assert result.data["deferred_reason"] == "provider_throttled_fleet"


def test_rate_limit_in_repo_a_defers_dispatch_in_repo_b(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    _, paths_a, _ = _repo(tmp_path, "a", "devin-shell")
    app_b, paths_b = _rework_app(tmp_path, "b", "devin-shell", fleet)
    _register(fleet, paths_a, paths_b)
    _throttle(paths_a, until=datetime.now(UTC) + timedelta(hours=1))

    result = app_b.dispatch()

    assert result.ok is False
    assert result.data["deferred_reason"] == "provider_throttled_fleet"
    assert result.data["attempted_count"] == 0


def test_devin_throttle_does_not_defer_claude_code_worker(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    _, paths_a, _ = _repo(tmp_path, "a", "devin-shell")
    app_b, paths_b = _rework_app(tmp_path, "b", "claude-code", fleet)
    _register(fleet, paths_a, paths_b)
    _throttle(paths_a, until=datetime.now(UTC) + timedelta(hours=1))

    result = app_b.dispatch_rework()

    assert result.data.get("deferred_reason") not in (
        "provider_throttled_fleet",
        "provider_resume_staggered",
    )


def test_provider_auth_window_is_not_fleet_scoped(tmp_path: Path) -> None:
    _, paths_a, _ = _repo(tmp_path, "a", "devin-shell")
    _throttle(paths_a, until=datetime.now(UTC) + timedelta(hours=1), reason="provider_auth")
    assert fpt.latest_fleet_throttle([paths_a.state_file], "devin") is None


def test_deadline_is_max_across_repos_and_never_shortened(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    _, paths_a, _ = _repo(tmp_path, "a", "devin-shell")
    _, paths_b, _ = _repo(tmp_path, "b", "devin-shell")
    _throttle(paths_a, until=now + timedelta(hours=3))
    _throttle(paths_b, until=now + timedelta(hours=1))
    latest = fpt.latest_fleet_throttle([paths_b.state_file, paths_a.state_file], "devin")
    assert latest is not None and latest.replace(microsecond=0) == (
        now + timedelta(hours=3)
    ).replace(microsecond=0)


def test_expired_window_admits_one_then_staggers_until_probe_survives(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    now = datetime.now(UTC)
    _, paths_a, _ = _repo(tmp_path, "a", "devin-shell")
    expired = now - timedelta(minutes=1)
    _throttle(paths_a, until=expired)
    kw = {"fleet_dir_override": str(fleet)}

    first = fpt.decide_launch([paths_a.state_file], "devin", now=now, **kw)
    assert first.action == "admit_one"

    fpt.note_probe_launch("devin", [(os.getpid(), None)], now=now, **kw)
    soon = fpt.decide_launch([paths_a.state_file], "devin", now=now + timedelta(minutes=2), **kw)
    assert soon.action == "defer_probe"

    later = fpt.decide_launch(
        [paths_a.state_file],
        "devin",
        now=now + timedelta(seconds=fpt.RESUME_SURVIVAL_SECONDS + 1),
        **kw,
    )
    assert later.action == "open"

    # The probe dies of the limit: a newer window re-arms probe mode.
    _throttle(paths_a, until=now + timedelta(minutes=10))
    rearmed = fpt.decide_launch(
        [paths_a.state_file], "devin", now=now + timedelta(minutes=11), **kw
    )
    assert rearmed.action == "admit_one"


def test_probe_in_flight_defers_rework_in_other_repo(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    _, paths_a, _ = _repo(tmp_path, "a", "devin-shell")
    app_b, paths_b = _rework_app(tmp_path, "b", "devin-shell", fleet)
    _register(fleet, paths_a, paths_b)
    _throttle(paths_a, until=datetime.now(UTC) - timedelta(minutes=1))
    fpt.note_probe_launch("devin", [(os.getpid(), None)], fleet_dir_override=str(fleet))

    result = app_b.dispatch_rework()

    assert result.ok is False
    assert result.data["deferred_reason"] == "provider_resume_staggered"
