"""Silent dispatch deferrals emit events and trip a starvation alarm (issue #1986).

Before the fix a deferred ``dispatch`` / ``dispatch_rework`` pass returned
``ok=True`` with no event and no log line, so a starved repo looked exactly
like a repo with nothing to do.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from charlie_work.config import FleetConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.dispatch_deferral import DEFAULT_STARVATION_THRESHOLD
from charlie_work.github import GitHubError, GraphQLBudgetError
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import StateLockBusy
from charlie_work.workflow import CommandResult, OrchestratorApp

_DISPATCH_MOD = "charlie_work.orchestration.reap_dispatch"
_REWORK_MOD = "charlie_work.orchestration.misc_worker_dispatch"


def _app(tmp_path: Path) -> OrchestratorApp:
    # launch_lock_wait_seconds=0: these tests pin the deferral's visibility,
    # not the issue-#2055 retry -- a blocked acquirer should not stall the
    # suite for the production wait budget on every call. A non-manual worker
    # harness is required so dispatch_rework reaches the launch gate (the
    # manual adapter early-returns before it).
    config = OrchestratorConfig(
        worker=WorkerRoleConfig(harness="claude-code"),
        fleet=FleetConfig(global_max_concurrent_sessions=2, launch_lock_wait_seconds=0),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub())


def _events(app: OrchestratorApp, kind: str) -> list[dict[str, Any]]:
    return query_events(app.paths.state_file, kind=kind)


def _raise(exc: Exception):
    def _inner(*_a: Any, **_kw: Any) -> Any:
        raise exc

    return _inner


def _defer_fleet_lock(monkeypatch: pytest.MonkeyPatch, app: OrchestratorApp, lane: str) -> None:
    mod = _DISPATCH_MOD if lane == "dispatch" else _REWORK_MOD
    monkeypatch.setattr(f"{mod}.try_acquire_fleet_lock", lambda *_a, **_k: None)


def _defer_state_lock(monkeypatch: pytest.MonkeyPatch, app: OrchestratorApp, lane: str) -> None:
    target = (
        "_finalize_externally_merged_issues" if lane == "dispatch" else "_dispatch_rework_impl"
    )
    monkeypatch.setattr(app, target, _raise(StateLockBusy("busy")))


def _defer_graphql(monkeypatch: pytest.MonkeyPatch, app: OrchestratorApp, lane: str) -> None:
    monkeypatch.setattr(app, "_dispatch_impl", _raise(GraphQLBudgetError(3, 1_900_000_000, 100)))


def _defer_github_error(monkeypatch: pytest.MonkeyPatch, app: OrchestratorApp, lane: str) -> None:
    monkeypatch.setattr(app, "_dispatch_impl", _raise(GitHubError("gh exploded")))


def _call(app: OrchestratorApp, lane: str) -> CommandResult:
    return app.dispatch() if lane == "dispatch" else app.dispatch_rework()


@pytest.mark.parametrize(
    ("lane", "install", "reason"),
    [
        ("dispatch", _defer_fleet_lock, "fleet_lock_held"),
        ("dispatch", _defer_state_lock, "state_lock_busy"),
        ("dispatch", _defer_graphql, "graphql_rate_limit"),
        ("dispatch", _defer_github_error, "github_error"),
        ("dispatch_rework", _defer_fleet_lock, "fleet_lock_held"),
        ("dispatch_rework", _defer_state_lock, "state_lock_busy"),
    ],
)
def test_each_deferral_path_emits_dispatch_deferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    lane: str,
    install: Any,
    reason: str,
) -> None:
    app = _app(tmp_path)
    install(monkeypatch, app, lane)

    with caplog.at_level(logging.WARNING, logger="charlie_work.dispatch_deferral"):
        result = _call(app, lane)

    assert result.data["deferred_reason"] == reason
    (event,) = _events(app, "dispatch_deferred")
    assert event["level"] == "warning"
    assert event["payload"]["deferred_reason"] == reason
    assert event["payload"]["lane"] == lane
    assert event["payload"]["consecutive_deferrals"] == 1
    assert any(reason in r.getMessage() for r in caplog.records)


def test_consecutive_deferrals_trip_starvation_once_and_success_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path)
    with monkeypatch.context() as m:
        _defer_fleet_lock(m, app, "dispatch")
        for _ in range(DEFAULT_STARVATION_THRESHOLD - 1):
            app.dispatch()
        assert _events(app, "dispatch_starved") == []

        app.dispatch()
        (starved,) = _events(app, "dispatch_starved")
        assert starved["level"] == "error"
        assert starved["payload"]["consecutive_deferrals"] == DEFAULT_STARVATION_THRESHOLD

        app.dispatch()  # past the threshold: still one alarm per episode
        assert len(_events(app, "dispatch_starved")) == 1

    # A pass that is not deferred (empty backlog, real dispatch result) resets.
    result = app.dispatch()
    assert "deferred_reason" not in result.data
    with monkeypatch.context() as m:
        _defer_fleet_lock(m, app, "dispatch")
        app.dispatch()
    # 1,2,3,4 before the reset, then the streak restarts at 1 (order-agnostic).
    counts = sorted(
        e["payload"]["consecutive_deferrals"] for e in _events(app, "dispatch_deferred")
    )
    assert counts == [1, 1, 2, 3, 4]


def test_streaks_are_tracked_per_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path)
    _defer_fleet_lock(monkeypatch, app, "dispatch")
    _defer_fleet_lock(monkeypatch, app, "dispatch_rework")
    for _ in range(DEFAULT_STARVATION_THRESHOLD - 1):
        app.dispatch()
    app.dispatch_rework()  # rework's first deferral must not inherit fresh dispatch's streak

    assert _events(app, "dispatch_starved") == []


def test_dry_run_records_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path)
    monkeypatch.setattr(app, "dry_run", True, raising=False)
    _defer_fleet_lock(monkeypatch, app, "dispatch")

    app.dispatch()

    assert _events(app, "dispatch_deferred") == []
