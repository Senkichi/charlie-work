"""Issue #2055: the fleet launch lock covers only governor -> claim -> launch.

Before the fix both remote lanes took the lock at dispatch entry -- before
the ``issue_list``/``pr_list``/``issue_view``/stall-sweep scan -- and held it
across the whole pass, and took it even when the provider was throttled; a
slow repo starved fast ones (``fleet_lock_held`` -> ``dispatch_starved``).
These tests pin the new scope: the lane mints a pending handle at entry, the
OS lock is realized inside ``issue_worker_launch_permit`` with a bounded
jittered wait (``fleet.launch_lock_wait_seconds``), a throttled repo never
contends for it, ``live_count`` is computed under it, and the holder sidecar
(``fleet.lock.holder``) exists only while the lock is held.

App/fixture helpers duplicated from test_worker_launch_gate.py -- the zero
cross-test-import guard forbids ``from test_* import`` so helpers are
reinlined per file.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from charlie_work import layout
from charlie_work.adapters import SessionDispatchResult, SessionRequest
from charlie_work.config import (
    ConfigError,
    DispatchConfig,
    FleetConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
    build_config_from_data,
)
from charlie_work.fleet_registry import try_acquire_fleet_lock
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, set_throttled_until, state_lock
from charlie_work.worker_launch_gate import acquire_fleet_launch_lock
from charlie_work.workflow import OrchestratorApp

import charlie_work.workflow as wf

FRESH = "fresh"
REWORK = "rework"
LANES = pytest.mark.parametrize("lane", [FRESH, REWORK])

_SCAN_MOD = {
    FRESH: "charlie_work.orchestration.dispatch_state",
    REWORK: "charlie_work.orchestration.state_dispatch_rework",
}
_LANE_MOD = {
    FRESH: "charlie_work.orchestration.reap_dispatch",
    REWORK: "charlie_work.orchestration.misc_worker_dispatch",
}


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


def _app(tmp_path: Path, lane: str, *, dry_run: bool = False, **config_kw: Any) -> OrchestratorApp:
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


def _lock_is_free(app: OrchestratorApp) -> bool:
    """Probe the real fleet lock: True when nothing holds it right now."""
    probe = try_acquire_fleet_lock(app.fleet_dir_override)
    if probe is None:
        return False
    probe.release()
    return True


# --- scan scope -------------------------------------------------------------


@LANES
def test_read_only_scan_runs_without_the_fleet_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """Every scan call issued before issue_worker_launch_permit must observe a
    free fleet lock -- the lock is realized inside the permit, after the scan."""
    app = _app(tmp_path, lane, fleet_cap=4, launch_lock_wait=0.5)
    calls = _spy_dispatch_sessions(monkeypatch)
    order = itertools.count()
    probes: list[tuple[int, str, bool]] = []
    boundary: list[int] = []

    def _wrap(fn: Any, name: str) -> Any:
        def _inner(*a: Any, **kw: Any) -> Any:
            probes.append((next(order), name, _lock_is_free(app)))
            return fn(*a, **kw)

        return _inner

    for name in ("issue_list", "pr_list", "issue_view"):
        monkeypatch.setattr(app.gh, name, _wrap(getattr(app.gh, name), name))
    monkeypatch.setattr(
        "charlie_work.workflow._detect_and_handle_stalled_sessions",
        _wrap(wf._detect_and_handle_stalled_sessions, "stall_sweep"),
    )

    orig_permit = getattr(sys.modules[_SCAN_MOD[lane]], "issue_worker_launch_permit")

    def _permit_boundary(*a: Any, **kw: Any) -> Any:
        boundary.append(next(order))
        return orig_permit(*a, **kw)

    monkeypatch.setattr(f"{_SCAN_MOD[lane]}.issue_worker_launch_permit", _permit_boundary)

    result = _run(app, lane)

    assert [r.issue_number for r in calls] == [123], result.message
    assert boundary, "issue_worker_launch_permit never ran"
    scan_probes = [(name, free) for seq, name, free in probes if seq < boundary[0]]
    assert scan_probes, "no scan probes recorded before the permit"
    assert all(free for _name, free in scan_probes), scan_probes
    seen_names = {name for name, _ in scan_probes}
    assert "stall_sweep" in seen_names
    assert ({"issue_list"} if lane == FRESH else {"pr_list", "issue_view"}) <= seen_names


@LANES
def test_throttled_repo_never_contends_for_the_fleet_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """A throttled repo defers on the lock-free pre-check -- the acquirer is
    never invoked, so a throttled fleet cannot saturate the shared lock."""
    app = _app(tmp_path, lane, fleet_cap=4)
    calls = _spy_dispatch_sessions(monkeypatch)
    acquire_calls: list[int] = []

    def _counting_acquire(*a: Any, **kw: Any) -> Any:
        acquire_calls.append(1)
        return try_acquire_fleet_lock(*a, **kw)

    monkeypatch.setattr(f"{_LANE_MOD[lane]}.try_acquire_fleet_lock", _counting_acquire)
    _throttle(app)

    result = _run(app, lane)

    assert acquire_calls == []
    assert calls == []
    assert result.data["deferred_reason"] == "provider_throttled"
    assert result.data["throttled_until"] is not None


# --- bounded wait -----------------------------------------------------------


@LANES
def test_briefly_held_lock_is_acquired_within_the_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """A lock released inside the wait budget does not cost a whole pass --
    the bounded jittered retry lands it and the lane launches."""
    app = _app(tmp_path, lane, fleet_cap=4, launch_lock_wait=2.0)
    calls = _spy_dispatch_sessions(monkeypatch)
    held = try_acquire_fleet_lock(app.fleet_dir_override)
    assert held is not None
    timer = threading.Timer(0.25, held.release)
    timer.start()
    try:
        result = _run(app, lane)
    finally:
        timer.cancel()
        held.release()  # idempotent: safe if the timer already ran

    assert [r.issue_number for r in calls] == [123], result.message
    assert "deferred_reason" not in result.data


def test_wait_zero_is_a_single_attempt(tmp_path: Path) -> None:
    """``launch_lock_wait_seconds: 0`` restores the old non-blocking try."""
    app = _app(tmp_path, FRESH, fleet_cap=4)
    acquire_calls: list[int] = []

    def _failing(_override: Any) -> None:
        acquire_calls.append(1)
        return None

    handle = acquire_fleet_launch_lock(app, acquire=_failing)
    assert handle.ensure_acquired(0) is None
    assert len(acquire_calls) == 1
    assert not handle.valid
    handle.release()  # dead handle: nothing to release


def test_retry_acquires_a_lock_released_mid_wait(tmp_path: Path) -> None:
    app = _app(tmp_path, FRESH, fleet_cap=4)
    acquire_calls: list[int] = []

    class _FakeLock:
        released = False

        def release(self) -> None:
            self.released = True

    fake = _FakeLock()

    def _eventually(_override: Any) -> Any:
        acquire_calls.append(1)
        return fake if len(acquire_calls) >= 3 else None

    handle = acquire_fleet_launch_lock(app, acquire=_eventually)
    waited = handle.ensure_acquired(1.0)
    assert waited is not None and waited >= 0
    assert handle.valid
    handle.release()
    assert fake.released


# --- live_count under the lock ----------------------------------------------


@LANES
def test_governor_receives_a_live_count_computed_under_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """The governor's ``live_count`` kwarg is the int the permit computed while
    holding the fleet lock, and the governor itself runs under the lock."""
    app = _app(tmp_path, lane, fleet_cap=4)
    calls = _spy_dispatch_sessions(monkeypatch)

    count_held: list[bool] = []
    orig_count = wf._count_live_sessions

    def _count_probe(*a: Any, **kw: Any) -> Any:
        count_held.append(not _lock_is_free(app))
        return orig_count(*a, **kw)

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", _count_probe)

    gov_calls: list[tuple[Any, bool]] = []
    orig_gov = app._apply_concurrency_governor

    def _gov_probe(*a: Any, **kw: Any) -> Any:
        gov_calls.append((kw.get("live_count"), not _lock_is_free(app)))
        return orig_gov(*a, **kw)

    monkeypatch.setattr(app, "_apply_concurrency_governor", _gov_probe)

    result = _run(app, lane)

    assert [r.issue_number for r in calls] == [123], result.message
    assert gov_calls, "governor never ran"
    assert all(isinstance(live_count, int) and held for live_count, held in gov_calls), gov_calls
    assert count_held and count_held[-1] is True


# --- early release: the lock does not outlive the launch window -------------


@LANES
def test_fleet_lock_is_released_before_post_launch_bookkeeping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """Every ``state_lock`` entry after ``_launch_workers`` returns -- the
    dispatch_pending -> dispatched upgrade and the rest of the pass's
    bookkeeping -- must observe a FREE fleet lock: the governor -> claim ->
    launch window the lock exists to serialize closed when the session
    sidecars landed. Probes are discriminated by whether dispatch_sessions
    has already run, so the claim-phase entries (held, by design) do not
    count."""
    app = _app(tmp_path, lane, fleet_cap=4)
    calls = _spy_dispatch_sessions(monkeypatch)
    probes: list[tuple[bool, int]] = []
    orig_state_lock = wf.state_lock

    def _state_lock_probe(*a: Any, **kw: Any) -> Any:
        probes.append((_lock_is_free(app), len(calls)))
        return orig_state_lock(*a, **kw)

    monkeypatch.setattr("charlie_work.workflow.state_lock", _state_lock_probe)

    result = _run(app, lane)

    assert [r.issue_number for r in calls] == [123], result.message
    post_launch = [free for free, launched in probes if launched]
    assert post_launch, "no state_lock entry observed after launch"
    assert all(post_launch), probes


# Probe target inside each lane's dry-run planning branch (post-release):
# fresh dispatch plans from pr_list; remote rework calls issue_view for each
# selected candidate. Both run strictly after the dry-run release, and any
# same-named call in the pre-permit scan is filtered out by the boundary.
_DRY_RUN_PROBE = {FRESH: "pr_list", REWORK: "issue_view"}


@LANES
def test_fleet_lock_is_released_before_dry_run_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """The dry-run branch releases the realized lock before its read-only
    planning pass: every planning probe after issue_worker_launch_permit ran
    must observe a free fleet lock -- without the branch's early release the
    lock stayed held to the lane's finally, i.e. across the whole planning
    pass."""
    app = _app(tmp_path, lane, fleet_cap=4, dry_run=True)
    calls = _spy_dispatch_sessions(monkeypatch)
    order = itertools.count()
    probes: list[tuple[int, bool]] = []
    boundary: list[int] = []

    probe_name = _DRY_RUN_PROBE[lane]
    orig_probe = getattr(app.gh, probe_name)

    def _probe(*a: Any, **kw: Any) -> Any:
        probes.append((next(order), _lock_is_free(app)))
        return orig_probe(*a, **kw)

    monkeypatch.setattr(app.gh, probe_name, _probe)

    orig_permit = getattr(sys.modules[_SCAN_MOD[lane]], "issue_worker_launch_permit")

    def _permit_boundary(*a: Any, **kw: Any) -> Any:
        boundary.append(next(order))
        return orig_permit(*a, **kw)

    monkeypatch.setattr(f"{_SCAN_MOD[lane]}.issue_worker_launch_permit", _permit_boundary)

    result = _run(app, lane)

    assert result.ok, result.message
    assert calls == [], "a dry run launched workers"
    assert boundary, "issue_worker_launch_permit never ran"
    planning_probes = [free for seq, free in probes if seq > boundary[0]]
    assert planning_probes, f"no {probe_name} probe observed during dry-run planning"
    assert all(planning_probes), probes


# --- holder metadata ---------------------------------------------------------


@LANES
def test_holder_sidecar_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """``fleet.lock.holder`` names the holder while the lock is held and is
    removed on release -- never a stale sidecar."""
    app = _app(tmp_path, lane, fleet_cap=4)
    calls = _spy_dispatch_sessions(monkeypatch)
    sidecar = layout.fleet_lock_holder_path(override=app.fleet_dir_override)
    seen: list[dict[str, Any]] = []
    orig_count = wf._count_live_sessions

    def _count_probe(*a: Any, **kw: Any) -> Any:
        if sidecar.exists():
            seen.append(json.loads(sidecar.read_text(encoding="utf-8")))
        return orig_count(*a, **kw)

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", _count_probe)

    result = _run(app, lane)

    assert [r.issue_number for r in calls] == [123], result.message
    assert seen, "holder sidecar never observed while the lock was held"
    holder = seen[-1]
    assert holder["repo"] == app.repo_root.name
    assert holder["pid"] == os.getpid()
    assert holder.get("acquired_at")
    assert not sidecar.exists(), "holder sidecar left stale after release"


@LANES
def test_fleet_lock_held_deferral_reports_holder_and_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """A ``fleet_lock_held`` deferral (ok=True so dispatch_deferral counts it
    toward starvation) carries the wait budget and the recorded holder."""
    app = _app(tmp_path, lane, fleet_cap=4, launch_lock_wait=0.2)
    calls = _spy_dispatch_sessions(monkeypatch)
    held = try_acquire_fleet_lock(app.fleet_dir_override)
    assert held is not None
    sidecar = layout.fleet_lock_holder_path(override=app.fleet_dir_override)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    acquired_at = (
        (datetime.now(UTC) - timedelta(seconds=3))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"repo": "other-repo", "pid": 4242, "acquired_at": acquired_at}),
        encoding="utf-8",
    )
    tmp.replace(sidecar)
    try:
        result = _run(app, lane)
    finally:
        held.release()

    assert calls == []
    assert result.ok is True
    assert result.data["deferred_reason"] == "fleet_lock_held"
    assert result.data["lock_wait_seconds"] == 0.2
    assert result.data["lock_holder_repo"] == "other-repo"
    assert result.data["lock_holder_pid"] == 4242
    assert result.data["lock_held_seconds"] >= 0


# --- entry handle is pending only -------------------------------------------


def test_entry_handle_does_not_touch_the_os_lock(tmp_path: Path) -> None:
    """acquire_fleet_launch_lock mints the lane handle without acquiring --
    the OS lock stays free until issue_worker_launch_permit realizes it."""
    app = _app(tmp_path, FRESH, fleet_cap=4)
    handle = acquire_fleet_launch_lock(app)
    try:
        assert handle.valid
        probe = try_acquire_fleet_lock(app.fleet_dir_override)
        assert probe is not None, "minting the lane handle took the OS lock"
        probe.release()
    finally:
        handle.release()


# --- config knob -------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, -0.5, True, "10"])
def test_launch_lock_wait_seconds_rejects_bad_values(bad: Any) -> None:
    with pytest.raises(ConfigError, match="launch_lock_wait_seconds"):
        build_config_from_data({"fleet": {"launch_lock_wait_seconds": bad}})


@pytest.mark.parametrize(("good", "expected"), [(0, 0), (7, 7), (2.5, 2.5)])
def test_launch_lock_wait_seconds_accepts_non_negative_numbers(good: Any, expected: float) -> None:
    config = build_config_from_data({"fleet": {"launch_lock_wait_seconds": good}})
    assert config.fleet.launch_lock_wait_seconds == expected


def test_launch_lock_wait_seconds_defaults_to_bounded_wait() -> None:
    config = build_config_from_data({})
    assert config.fleet.launch_lock_wait_seconds == 10.0
