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
from charlie_work import layout
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


PRE, HELD, POST = 7.0, 3.0, 11.0


class _Phased:
    """Fake clock plus a spy that advances it on every GitHub call.

    The step depends on where the pass is relative to the fleet launch lock
    (told apart by the holder sidecar, which exists only while the OS lock is
    held): ``PRE`` before the lock is realized (the handle already exists),
    ``HELD`` while it is held, ``POST`` once it has been released.  A correct
    ``launch_lock_held_seconds`` therefore counts only the HELD steps (plus
    the launch itself) -- stamping the start at handle mint or the end at
    Event emission would add PRE / POST steps and fail.
    """

    def __init__(self, app: OrchestratorApp, clock: FakeClock) -> None:
        self.clock = clock
        self.holder = layout.fleet_lock_holder_path(override=app.fleet_dir_override)
        self.seen_held = False
        self.calls = {"pre": 0, "held": 0, "post": 0}
        for name in dir(app.gh):
            fn = getattr(app.gh, name)
            if not name.startswith("_") and callable(fn):
                setattr(app.gh, name, self._wrap(fn))

    def phase(self) -> str:
        if self.holder.exists():
            self.seen_held = True
            return "held"
        return "post" if self.seen_held else "pre"

    def _wrap(self, fn: Any) -> Any:
        def _inner(*a: Any, **kw: Any) -> Any:
            phase = self.phase()
            self.calls[phase] += 1
            self.clock.advance({"pre": PRE, "held": HELD, "post": POST}[phase])
            return fn(*a, **kw)

        return _inner

    @property
    def expected(self) -> float:
        return HELD * self.calls["held"] + self.launched

    launched = 0.0


def _install_hosts(fake_host: Any, spy: list[_Phased] | None = None) -> FakeClock:
    clock = FakeClock(start=datetime(2026, 1, 1, tzinfo=UTC))

    def _launch(_repo_root, _manifest, _results, settings, requests):
        clock.advance(LAUNCH_SECONDS)  # time passes inside the locked window
        if spy:
            spy[0].launched += LAUNCH_SECONDS
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
def test_value_is_exactly_acquire_to_release(tmp_path: Path, lane: str, fake_host) -> None:
    """The clock also moves before the lock is realized (after the handle is
    minted) and after it is released but before the aggregate Event is built;
    neither may leak into the reported value."""
    spy: list[_Phased] = []
    clock = _install_hosts(fake_host, spy)
    app = _app(tmp_path, lane, fleet_cap=4)
    spy.append(_Phased(app, clock))

    result = _run(app, lane)

    assert result.data["selected_count"] == 1, result.message
    assert spy[0].calls["pre"] > 0 and spy[0].calls["held"] > 0 and spy[0].calls["post"] > 0
    (event,) = _events(app, lane)
    assert event["payload"][FIELD] == pytest.approx(spy[0].expected)


@LANES
def test_pass_without_a_fleet_cap_omits_the_field(tmp_path: Path, lane: str, fake_host) -> None:
    _install_hosts(fake_host)
    app = _app(tmp_path, lane, fleet_cap=0)

    result = _run(app, lane)

    assert result.data["selected_count"] == 1, result.message
    (event,) = _events(app, lane)
    assert FIELD not in event["payload"]


@LANES
def test_dry_run_with_a_fleet_cap_omits_the_field(tmp_path: Path, lane: str, fake_host) -> None:
    """A dry run releases the lock right after the governor and emits no
    aggregate ``dispatch`` / ``dispatch_rework`` Event at all, so no Event may
    carry the field."""
    _install_hosts(fake_host)
    app = _app(tmp_path, lane, fleet_cap=4, dry_run=True)

    result = _run(app, lane)

    assert result.ok, result.message
    assert [e for e in _events(app, lane) if FIELD in e["payload"]] == []


def test_rework_without_valid_prompts_reports_elapsed_so_far(tmp_path: Path, fake_host) -> None:
    """The no-valid-rework-prompts ``dispatch_rework`` Event is built while the
    lock is still held (the lane's ``finally`` releases it afterwards), so the
    field there is the Host-clock time elapsed up to the emission."""
    spy: list[_Phased] = []
    clock = _install_hosts(fake_host, spy)
    app = _app(tmp_path, REWORK, fleet_cap=4)
    (app.paths.prs / "pr-456" / "rework-prompt.md").unlink()
    spy.append(_Phased(app, clock))

    result = app.dispatch_rework()

    assert result.message == "no valid rework prompts found", result.message
    assert spy[0].calls["held"] > 0 and spy[0].calls["pre"] > 0
    (event,) = _events(app, REWORK)
    assert event["payload"][FIELD] == pytest.approx(HELD * spy[0].calls["held"])
