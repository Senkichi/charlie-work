"""Issue #2041: every worker lane launches through one permit-gated point.

App-level lane tests (fresh ``dispatch`` and remote ``dispatch_rework``): an
active provider throttle, a held fleet lock, or a governor clamped to zero each
mean zero launches, and an ungated control launches. The local rework lane's
equivalents live in ``tests/test_local_rework_launch_gates.py`` (issue #2039),
which now exercises the same shared gate. The unit tests at the bottom pin
``_launch_workers``' refusals: a forged / absent / released permit, and a batch
over the permit's (shared) budget.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from charlie_work.adapters import AdapterSettings, SessionDispatchResult, SessionRequest
from charlie_work.config import (
    DispatchConfig,
    FleetConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.fleet_registry import try_acquire_fleet_lock
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, set_throttled_until, state_lock
from charlie_work.worker_launch_gate import (
    WorkerLaunchDeferral,
    WorkerLaunchPermit,
    _launch_workers,
    acquire_fleet_launch_lock,
    issue_worker_launch_permit,
)
from charlie_work.workflow import OrchestratorApp

FRESH = "fresh"
REWORK = "rework"


def _config(*, fleet_cap: int = 0, max_concurrent: int = 0) -> OrchestratorConfig:
    return OrchestratorConfig(
        worker=WorkerRoleConfig(harness="claude-code"),
        dispatch=DispatchConfig(default_limit=5, max_concurrent_sessions=max_concurrent),
        fleet=FleetConfig(global_max_concurrent_sessions=fleet_cap),
    )


def _app(tmp_path: Path, lane: str, **config_kw: int) -> OrchestratorApp:
    config = _config(**config_kw)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    fake_gh = FakeGitHub()
    if lane == REWORK:
        # FakeGitHub's open PR 456 links issue 123; seed it rework_requested.
        with state_lock(paths.state_file):
            state = load_state(paths.state_file)
            state["issues"]["123"] = {"number": 123, "status": "rework_requested"}
            state["prs"]["456"] = {"number": 456, "issue_number": 123}
            save_state(paths.state_file, state)
    else:
        # No open PR covers the ready issue, so fresh dispatch selects it.
        fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(
        tmp_path, paths, config, fake_gh, fleet_dir_override=str(tmp_path / "fleet")
    )
    if lane == REWORK:
        pr_dir = paths.prs / "pr-456"
        pr_dir.mkdir(parents=True, exist_ok=True)
        (pr_dir / "rework-prompt.md").write_text("rework prompt", encoding="utf-8")
    return app


def _spy_dispatch_sessions(monkeypatch: pytest.MonkeyPatch) -> list[SessionRequest]:
    calls: list[SessionRequest] = []

    def _fake(_repo_root, _manifest, _results, settings, requests):
        calls.extend(requests)
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

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", _fake)
    return calls


def _run(app: OrchestratorApp, lane: str) -> Any:
    return app.dispatch() if lane == FRESH else app.dispatch_rework()


def _throttle(app: OrchestratorApp) -> None:
    until = (datetime.now(UTC) + timedelta(hours=1)).replace(microsecond=0)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state = set_throttled_until(state, until.isoformat().replace("+00:00", "Z"), source="test")
        save_state(app.paths.state_file, state)


LANES = pytest.mark.parametrize("lane", [FRESH, REWORK])


@LANES
def test_ungated_lane_launches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str) -> None:
    """Control: with no gate active the lane launches -- otherwise every
    zero-launch assertion below proves nothing."""
    app = _app(tmp_path, lane)
    calls = _spy_dispatch_sessions(monkeypatch)

    result = _run(app, lane)

    assert [r.issue_number for r in calls] == [123], result.message
    assert "deferred_reason" not in result.data


@LANES
def test_provider_throttle_blocks_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    app = _app(tmp_path, lane)
    calls = _spy_dispatch_sessions(monkeypatch)
    _throttle(app)

    result = _run(app, lane)

    assert calls == []
    assert result.data["deferred_reason"] == "provider_throttled"
    assert result.data["throttled_until"] is not None


@LANES
def test_held_fleet_lock_blocks_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """The real fleet lock, held by another dispatcher, not a patched stub."""
    app = _app(tmp_path, lane, fleet_cap=4)
    calls = _spy_dispatch_sessions(monkeypatch)
    held = try_acquire_fleet_lock(app.fleet_dir_override)
    assert held is not None
    try:
        result = _run(app, lane)
    finally:
        held.release()

    assert calls == []
    assert result.data["deferred_reason"] == "fleet_lock_held"


@LANES
def test_governor_at_cap_blocks_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", lambda *_a, **_k: 1)
    app = _app(tmp_path, lane, max_concurrent=1)
    calls = _spy_dispatch_sessions(monkeypatch)

    result = _run(app, lane)

    assert calls == []
    assert result.data["available_slots"] == 0
    assert result.data["selected_count"] == 0


@LANES
def test_lane_releases_fleet_lock_after_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """The lock is held across the launch and released afterwards -- a leaked
    lock would starve every other repo's dispatcher."""
    app = _app(tmp_path, lane, fleet_cap=4)
    calls = _spy_dispatch_sessions(monkeypatch)

    _run(app, lane)

    assert [r.issue_number for r in calls] == [123]
    lock = try_acquire_fleet_lock(app.fleet_dir_override)
    assert lock is not None
    lock.release()


# --- _launch_workers refusals ------------------------------------------------


def _request(issue_number: int) -> SessionRequest:
    return SessionRequest(
        issue_number=issue_number,
        issue_title=f"issue {issue_number}",
        prompt_path=Path(f"prompt-{issue_number}.md"),
        branch_name=f"agent/issue-{issue_number}",
    )


def _settings() -> AdapterSettings:
    return AdapterSettings(adapter="claude-code")


def _permit(app: OrchestratorApp, requested: int) -> WorkerLaunchPermit:
    permit = issue_worker_launch_permit(app, requested)
    assert isinstance(permit, WorkerLaunchPermit), permit
    return permit


def _assert_refused(results: list[SessionDispatchResult], calls: list, n: int) -> None:
    assert calls == []
    assert len(results) == n
    assert all(not r.ok and (r.error or "").startswith("worker launch refused") for r in results)


def test_issued_permit_launches_within_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for the refusals below: a gate-issued permit launches."""
    app = _app(tmp_path, FRESH)
    calls = _spy_dispatch_sessions(monkeypatch)
    with _permit(app, 2) as permit:
        results = _launch_workers(app, permit, _settings(), [_request(1), _request(2)])

    assert [r.issue_number for r in calls] == [1, 2]
    assert all(r.ok for r in results)


def test_forged_permit_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path, FRESH)
    calls = _spy_dispatch_sessions(monkeypatch)
    lock = acquire_fleet_launch_lock(app)
    forged = WorkerLaunchPermit(requested=5, max_launches=5, governor=None, _launch_lock=lock)

    results = _launch_workers(app, forged, _settings(), [_request(1)])

    _assert_refused(results, calls, 1)


def test_absent_permit_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path, FRESH)
    calls = _spy_dispatch_sessions(monkeypatch)

    results = _launch_workers(app, None, _settings(), [_request(1)])  # type: ignore[arg-type]

    _assert_refused(results, calls, 1)


def test_over_budget_batch_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", lambda *_a, **_k: 2)
    app = _app(tmp_path, FRESH, max_concurrent=3)
    calls = _spy_dispatch_sessions(monkeypatch)
    with _permit(app, 5) as permit:
        assert permit.max_launches == 1
        results = _launch_workers(app, permit, _settings(), [_request(1), _request(2)])

    _assert_refused(results, calls, 2)


def test_successive_launches_share_one_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remote rework lane's normal + rescue tiers launch in two calls on
    one permit; together they may not exceed it."""
    app = _app(tmp_path, FRESH)
    calls = _spy_dispatch_sessions(monkeypatch)
    with _permit(app, 2) as permit:
        first = _launch_workers(app, permit, _settings(), [_request(1)])
        over = _launch_workers(app, permit, _settings(), [_request(2), _request(3)])
        last = _launch_workers(app, permit, _settings(), [_request(4)])

    assert [r.ok for r in first] == [True]
    assert [r.ok for r in over] == [False, False]
    assert [r.ok for r in last] == [True]
    assert [r.issue_number for r in calls] == [1, 4]


def test_permit_is_refused_once_its_fleet_lock_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, FRESH, fleet_cap=4)
    calls = _spy_dispatch_sessions(monkeypatch)
    with _permit(app, 2) as permit:
        pass

    results = _launch_workers(app, permit, _settings(), [_request(1)])

    _assert_refused(results, calls, 1)


def test_throttle_deferral_releases_the_permit_owned_lock(tmp_path: Path) -> None:
    app = _app(tmp_path, FRESH, fleet_cap=4)
    _throttle(app)

    decision = issue_worker_launch_permit(app, 3)

    assert isinstance(decision, WorkerLaunchDeferral)
    assert decision.reason == "provider_throttled"
    lock = try_acquire_fleet_lock(app.fleet_dir_override)
    assert lock is not None, "a deferring permit leaked the fleet lock"
    lock.release()
