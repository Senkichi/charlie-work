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
from charlie_work.host.fakes import FakeWorkerLauncher
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

import charlie_work.workflow as wf

FRESH = "fresh"
REWORK = "rework"


def _config(
    *, fleet_cap: int = 0, max_concurrent: int = 0, launch_lock_wait: float = 10.0
) -> OrchestratorConfig:
    return OrchestratorConfig(
        worker=WorkerRoleConfig(harness="claude-code"),
        dispatch=DispatchConfig(default_limit=5, max_concurrent_sessions=max_concurrent),
        fleet=FleetConfig(
            global_max_concurrent_sessions=fleet_cap,
            launch_lock_wait_seconds=launch_lock_wait,
        ),
    )


def _app(tmp_path: Path, lane: str, **config_kw: Any) -> OrchestratorApp:
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


def _spy_dispatch_sessions(fake_host) -> list[SessionRequest]:
    """Install a launch-port fake recording each request; returns the log."""
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

    fake_host(worker_launch=FakeWorkerLauncher([_fake]))
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
def test_ungated_lane_launches(tmp_path: Path, fake_host, lane: str) -> None:
    """Control: with no gate active the lane launches -- otherwise every
    zero-launch assertion below proves nothing."""
    app = _app(tmp_path, lane)
    calls = _spy_dispatch_sessions(fake_host)

    result = _run(app, lane)

    assert [r.issue_number for r in calls] == [123], result.message
    assert "deferred_reason" not in result.data


@LANES
def test_provider_throttle_blocks_launch(tmp_path: Path, fake_host, lane: str) -> None:
    app = _app(tmp_path, lane)
    calls = _spy_dispatch_sessions(fake_host)
    _throttle(app)

    result = _run(app, lane)

    assert calls == []
    assert result.data["deferred_reason"] == "provider_throttled"
    assert result.data["throttled_until"] is not None


@LANES
def test_held_fleet_lock_blocks_launch(tmp_path: Path, fake_host, lane: str) -> None:
    """The real fleet lock, held by another dispatcher, not a patched stub.
    Issue #2055: a short bounded wait is retried before the deferral."""
    app = _app(tmp_path, lane, fleet_cap=4, launch_lock_wait=0.2)
    calls = _spy_dispatch_sessions(fake_host)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host, lane: str
) -> None:
    monkeypatch.setattr("charlie_work.live_session_count.count_live_sessions", lambda *_a, **_k: 1)
    app = _app(tmp_path, lane, max_concurrent=1)
    calls = _spy_dispatch_sessions(fake_host)

    result = _run(app, lane)

    assert calls == []
    assert result.data["available_slots"] == 0
    assert result.data["selected_count"] == 0


@LANES
def test_lane_releases_fleet_lock_after_launch(tmp_path: Path, fake_host, lane: str) -> None:
    """The lock is held across the launch and released afterwards -- a leaked
    lock would starve every other repo's dispatcher."""
    app = _app(tmp_path, lane, fleet_cap=4)
    calls = _spy_dispatch_sessions(fake_host)

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


def test_issued_permit_launches_within_budget(tmp_path: Path, fake_host) -> None:
    """Control for the refusals below: a gate-issued permit launches."""
    app = _app(tmp_path, FRESH)
    calls = _spy_dispatch_sessions(fake_host)
    with _permit(app, 2) as permit:
        results = _launch_workers(app, permit, _settings(), [_request(1), _request(2)])

    assert [r.issue_number for r in calls] == [1, 2]
    assert all(r.ok for r in results)


def test_forged_permit_is_refused(tmp_path: Path, fake_host) -> None:
    app = _app(tmp_path, FRESH)
    calls = _spy_dispatch_sessions(fake_host)
    lock = acquire_fleet_launch_lock(app)
    forged = WorkerLaunchPermit(requested=5, max_launches=5, governor=None, _launch_lock=lock)

    results = _launch_workers(app, forged, _settings(), [_request(1)])

    _assert_refused(results, calls, 1)


def test_absent_permit_is_refused(tmp_path: Path, fake_host) -> None:
    app = _app(tmp_path, FRESH)
    calls = _spy_dispatch_sessions(fake_host)

    results = _launch_workers(app, None, _settings(), [_request(1)])  # type: ignore[arg-type]

    _assert_refused(results, calls, 1)


def test_over_budget_batch_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    monkeypatch.setattr("charlie_work.live_session_count.count_live_sessions", lambda *_a, **_k: 2)
    app = _app(tmp_path, FRESH, max_concurrent=3)
    calls = _spy_dispatch_sessions(fake_host)
    with _permit(app, 5) as permit:
        assert permit.max_launches == 1
        results = _launch_workers(app, permit, _settings(), [_request(1), _request(2)])

    _assert_refused(results, calls, 2)


def test_successive_launches_share_one_budget(tmp_path: Path, fake_host) -> None:
    """The remote rework lane's normal + rescue tiers launch in two calls on
    one permit; together they may not exceed it."""
    app = _app(tmp_path, FRESH)
    calls = _spy_dispatch_sessions(fake_host)
    with _permit(app, 2) as permit:
        first = _launch_workers(app, permit, _settings(), [_request(1)])
        over = _launch_workers(app, permit, _settings(), [_request(2), _request(3)])
        last = _launch_workers(app, permit, _settings(), [_request(4)])

    assert [r.ok for r in first] == [True]
    assert [r.ok for r in over] == [False, False]
    assert [r.ok for r in last] == [True]
    assert [r.issue_number for r in calls] == [1, 4]


def test_permit_is_refused_once_its_fleet_lock_is_released(tmp_path: Path, fake_host) -> None:
    app = _app(tmp_path, FRESH, fleet_cap=4)
    calls = _spy_dispatch_sessions(fake_host)
    with _permit(app, 2) as permit:
        pass

    results = _launch_workers(app, permit, _settings(), [_request(1)])

    _assert_refused(results, calls, 1)


def test_launcher_exception_is_a_recorded_failure_and_releases_the_permit(
    tmp_path: Path, fake_host
) -> None:
    """Issue #2229: a raise out of the worker launch port comes back as
    per-request failure values -- the errors-as-values boundary at the launch
    seam converts it, one ``launch_failed`` event lands per request, the
    permit ledger still debits the batch, and the pass completes so the
    permit-owned fleet lock is released."""
    from charlie_work.instrumentation import query_events

    app = _app(tmp_path, FRESH, fleet_cap=4)

    def _boom(*_a: Any, **_k: Any) -> list[Any]:
        raise RuntimeError("dispatch exploded")

    fake_host(worker_launch=FakeWorkerLauncher([_boom]))
    with _permit(app, 2) as permit:
        results = _launch_workers(app, permit, _settings(), [_request(1), _request(2)])
        # The ledger debited the failed batch: the permit's budget is spent.
        over = _launch_workers(app, permit, _settings(), [_request(3)])

    assert [r.ok for r in results] == [False, False]
    assert all("dispatch exploded" in (r.error or "") for r in results)
    assert (over[0].error or "").startswith("worker launch refused")
    events = query_events(app.paths.state_file, kind="launch_failed")
    assert [e["payload"]["issue_number"] for e in events] == [1, 2]
    assert all(e["payload"]["role"] == "worker" for e in events)
    assert all(e["payload"]["error_class"] == "internal" for e in events)
    lock = try_acquire_fleet_lock(app.fleet_dir_override)
    assert lock is not None, "a raising launch leaked the permit's fleet lock"
    lock.release()


def test_throttle_deferral_releases_the_permit_owned_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The AUTHORITATIVE (post-governor) throttle deferral -- not the lock-free
    pre-check -- must release the permit-owned fleet lock. The throttle is on
    disk before the call; the patched first ``load_state`` (the pre-check
    read) sees it unthrottled so the gate proceeds to the fleet lock, the
    governor, and the state-locked re-read that finds it. The lock object is
    wrapped so the release itself is observed -- probing the OS lock after
    the fact cannot see a leaked handle (it is dropped with the deferral and
    GC releases it)."""
    app = _app(tmp_path, FRESH, fleet_cap=4)
    _throttle(app)

    orig_load_state = wf.load_state
    pre_check_saw_unthrottled = {"done": False}

    def _load_state_hiding_throttle_once(path: Path) -> dict[str, Any]:
        state = orig_load_state(path)
        if not pre_check_saw_unthrottled["done"]:
            pre_check_saw_unthrottled["done"] = True
            state = {**state, "throttled_until": None}
        return state

    monkeypatch.setattr("charlie_work.workflow.load_state", _load_state_hiding_throttle_once)

    class _RecordingLock:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.released = False

        def release(self) -> None:
            self.released = True
            self._inner.release()

    acquired: list[_RecordingLock] = []

    def _recording_acquire(override: Any) -> Any:
        real = try_acquire_fleet_lock(override)
        if real is None:
            return None
        wrapped = _RecordingLock(real)
        acquired.append(wrapped)
        return wrapped

    decision = issue_worker_launch_permit(app, 3, acquire=_recording_acquire)

    assert pre_check_saw_unthrottled["done"], "the lock-free pre-check never ran"
    assert isinstance(decision, WorkerLaunchDeferral)
    assert decision.reason == "provider_throttled"
    # Proof the POST-governor path deferred: a pre-check deferral carries no
    # governor (it ran before the governor); this one does, and its report
    # fields reach the lane payload.
    assert decision.governor is not None
    fields = decision.report_fields()
    assert fields["throttled_until"] is not None
    assert fields["fleet_concurrency_limit"] == 4
    # The deferral released the lock it owned.
    assert len(acquired) == 1
    assert acquired[0].released, "a deferring permit leaked the fleet lock"
    lock = try_acquire_fleet_lock(app.fleet_dir_override)
    assert lock is not None, "a deferring permit leaked the fleet lock"
    lock.release()
