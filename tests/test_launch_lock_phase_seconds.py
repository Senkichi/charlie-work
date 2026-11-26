"""Issue #3139: a dispatch lane whose lock-free scan finds nothing to launch
must never realize the fleet launch lock -- the pending handle reports an
exact ``0`` hold and an empty phase map -- and the aggregate
``dispatch``/``dispatch_rework`` events carry ``launch_lock_phase_seconds``,
the per-phase split (``governor``/``selection``/``launch``/``post_launch``)
of the hold.

Helpers are reinlined per file -- the zero cross-test-import guard forbids
``from test_* import``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
from charlie_work.fleet_registry import try_acquire_fleet_lock
from charlie_work.host.fakes import FakeClock, FakeWorkerLauncher
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, set_throttled_until, state_lock
from charlie_work.workflow import OrchestratorApp

FRESH = "fresh"
REWORK = "rework"
LANES = pytest.mark.parametrize("lane", [FRESH, REWORK])
_KIND = {FRESH: "dispatch", REWORK: "dispatch_rework"}
_LANE_MOD = {
    FRESH: "charlie_work.orchestration.reap_dispatch",
    REWORK: "charlie_work.orchestration.misc_worker_dispatch",
}
HELD = "launch_lock_held_seconds"
PHASES = "launch_lock_phase_seconds"
PHASE_NAMES = {"governor", "selection", "launch", "post_launch"}
LAUNCH_SECONDS = 2.5


def _app(
    tmp_path: Path,
    lane: str,
    *,
    fleet_cap: int = 4,
    dry_run: bool = False,
    candidate: bool = True,
) -> OrchestratorApp:
    config = OrchestratorConfig(
        worker=WorkerRoleConfig(harness="claude-code"),
        dispatch=DispatchConfig(default_limit=5),
        fleet=FleetConfig(global_max_concurrent_sessions=fleet_cap),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    fake_gh = FakeGitHub()
    if lane == REWORK:
        if candidate:
            # FakeGitHub's open PR 456 links issue 123; seed it rework_requested.
            with state_lock(paths.state_file):
                state = load_state(paths.state_file)
                state["issues"]["123"] = {"number": 123, "status": "rework_requested"}
                state["prs"]["456"] = {"number": 456, "issue_number": 123}
                save_state(paths.state_file, state)
    else:
        # No open PR covers the ready issue, so fresh dispatch selects it.
        fake_gh.prs[0]["state"] = "CLOSED"
        if not candidate:
            fake_gh.issues = []
    app = OrchestratorApp(
        tmp_path,
        paths,
        config,
        fake_gh,
        dry_run=dry_run,
        fleet_dir_override=str(tmp_path / "fleet"),
    )
    if lane == REWORK and candidate:
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


def _counting_acquire(monkeypatch: pytest.MonkeyPatch, lane: str) -> list[int]:
    calls: list[int] = []

    def _counting(*a: Any, **kw: Any) -> Any:
        calls.append(1)
        return try_acquire_fleet_lock(*a, **kw)

    monkeypatch.setattr(f"{_LANE_MOD[lane]}.try_acquire_fleet_lock", _counting)
    return calls


def _throttle(app: OrchestratorApp) -> None:
    until = (datetime.now(UTC) + timedelta(hours=1)).replace(microsecond=0)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state = set_throttled_until(state, until.isoformat().replace("+00:00", "Z"), source="test")
        save_state(app.paths.state_file, state)


# --- empty pass: never realized --------------------------------------------


@LANES
def test_empty_pass_never_realizes_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str, fake_host
) -> None:
    """The 183s-hold starvation bug: an empty pass must not ask the acquirer
    for the OS lock at all."""
    _install_hosts(fake_host)
    app = _app(tmp_path, lane, candidate=False)
    acquire_calls = _counting_acquire(monkeypatch, lane)

    result = _run(app, lane)

    assert result.ok, result.message
    assert result.data["selected_count"] == 0
    assert acquire_calls == []


def test_empty_fresh_pass_records_a_zero_hold(tmp_path: Path, fake_host) -> None:
    """The fresh lane's ``dispatch`` event is unconditional; on an empty
    pass it records an exact ``0`` hold and an empty phase map, with the
    governor-only fields at their pre-governor-deferral values."""
    _install_hosts(fake_host)
    app = _app(tmp_path, FRESH, candidate=False)

    result = app.dispatch()

    assert result.message == "dispatch complete", result.message
    assert result.data["attempted_count"] == 0
    assert result.data["merged_prs"] == []
    (event,) = _events(app, FRESH)
    assert event["payload"][HELD] == 0.0
    assert event["payload"][PHASES] == {}
    # `clamped` false and `dispatch_limit` unclamped -- none of the
    # report_fields() keys a governor that never ran could produce.
    assert event["payload"]["concurrency_governor"] == {
        "clamped": False,
        "dispatch_limit": 5,
    }


def test_empty_rework_pass_mirrors_the_no_candidates_tail(tmp_path: Path, fake_host) -> None:
    """Mirrors the lane's ``not selected`` tail: "no rework candidates
    found" and no ``dispatch_rework`` event (that path emits none either)."""
    _install_hosts(fake_host)
    app = _app(tmp_path, REWORK, candidate=False)

    result = app.dispatch_rework()

    assert result.message == "no rework candidates found", result.message
    assert _events(app, REWORK) == []


@LANES
def test_throttled_empty_pass_still_defers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str, fake_host
) -> None:
    """The empty-pass early return runs the same lock-free gates the permit
    would have: a throttled repo reports provider_throttled, not a clean
    nothing-to-dispatch -- and still never contends for the lock."""
    _install_hosts(fake_host)
    app = _app(tmp_path, lane, candidate=False)
    acquire_calls = _counting_acquire(monkeypatch, lane)
    _throttle(app)

    result = _run(app, lane)

    assert result.data["deferred_reason"] == "provider_throttled"
    assert result.data["throttled_until"] is not None
    assert acquire_calls == []


# --- empty pass that is NOT empty: closed issue owes bookkeeping ------------


def test_closed_ready_issue_bound_to_merged_pr_still_realizes_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """A pass holding only a CLOSED ready issue is not automatically empty:
    the merged-PR finalization bookkeeping (issue #427) is real
    under-the-lock work, so a bound issue still takes the permit."""
    _install_hosts(fake_host)
    app = _app(tmp_path, FRESH)
    # Aviator merged the PR; GitHub auto-closed the issue.
    app.gh.prs[0]["state"] = "MERGED"
    app.gh.issues[0]["state"] = "CLOSED"
    acquire_calls = _counting_acquire(monkeypatch, FRESH)

    result = app.dispatch()

    assert result.ok, result.message
    assert acquire_calls == [1]
    assert result.data["merged_pr_closed_issue_numbers"] == [123]
    (event,) = _events(app, FRESH)
    assert event["payload"][HELD] >= 0.0
    assert event["payload"][PHASES]


# --- phase map on a pass that holds the lock --------------------------------


@LANES
def test_phase_map_partitions_the_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str, fake_host
) -> None:
    """A pass that launches reports all four phases, non-negative, summing
    to ``launch_lock_held_seconds`` -- each phase runs to the next phase's
    boundary."""
    _install_hosts(fake_host)
    app = _app(tmp_path, lane)
    acquire_calls = _counting_acquire(monkeypatch, lane)

    result = _run(app, lane)

    assert result.data["selected_count"] == 1, result.message
    assert acquire_calls == [1]
    (event,) = _events(app, lane)
    held = event["payload"][HELD]
    phases = event["payload"][PHASES]
    assert set(phases) == PHASE_NAMES
    assert all(value >= 0.0 for value in phases.values())
    assert phases["launch"] == pytest.approx(LAUNCH_SECONDS)
    assert sum(phases.values()) == pytest.approx(held, abs=0.1)


@LANES
def test_post_governor_deferral_reports_only_the_governor_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str, fake_host
) -> None:
    """The authoritative throttle read runs after the governor under the
    lock; a deferral there never reaches selection, so its whole hold is
    the single ``governor`` phase."""
    _install_hosts(fake_host)
    app = _app(tmp_path, lane)
    # Hide the throttle from the lock-free pre-check (which reads state via
    # worker_launch_gates' own binding) so the lane realizes the lock and
    # runs the governor before the locked read sees it.
    monkeypatch.setattr(
        "charlie_work.worker_launch_gates.read_state_snapshot", lambda *a, **kw: {}
    )
    _throttle(app)

    result = _run(app, lane)

    assert result.data["deferred_reason"] == "provider_throttled"
    phases = result.data[PHASES]
    assert set(phases) == {"governor"}
    assert phases["governor"] == pytest.approx(result.data[HELD])


# --- dry run: unchanged ------------------------------------------------------


@LANES
def test_dry_run_empty_pass_still_realizes_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str, fake_host
) -> None:
    """Dry-run is unchanged: the early return is skipped, the permit still
    realizes the lock, and the planning scan releases it as before."""
    _install_hosts(fake_host)
    app = _app(tmp_path, lane, dry_run=True, candidate=False)
    acquire_calls = _counting_acquire(monkeypatch, lane)

    result = _run(app, lane)

    assert result.ok, result.message
    assert acquire_calls == [1]
    assert _events(app, lane) == []
