"""Issue #2039: local rework dispatch applies the remote rework lane's launch gates.

Split from ``test_local_lane.py`` (file-size ratchet); reuses its fixtures.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from _local_lane_fixtures import new_repo_root, rework_pending_app, spy_dispatch_sessions
from charlie_work import fleet_provider_throttle as fpt
from charlie_work.adapters import SessionDispatchResult
from charlie_work.state import (
    load_state,
    load_state_locked,
    save_state,
    set_throttled_until,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp


@pytest.fixture
def repo() -> Path:
    return new_repo_root()


class TestLocalReworkLaunchGates:
    """Issue #2039: local rework dispatch applies the remote rework lane's three
    launch gates -- provider throttle, fleet lock, concurrency governor. Before
    the fix a no-remote repo launched a Devin rework worker through an active
    operator hold, uncounted against the fleet-wide cap."""

    _rework_pending = staticmethod(rework_pending_app)
    _spy = staticmethod(spy_dispatch_sessions)

    @staticmethod
    def _assert_deferred(app: OrchestratorApp, result: dict, reason: str) -> None:
        assert result["dispatched"] == []
        assert result["skipped"] == [{"issue": 7, "reason": reason}]
        state = load_state_locked(app.paths.state_file)
        assert state["issues"]["7"]["status"] == "rework_requested"
        deferrals = [
            e
            for e in state.get("events", [])
            if e.get("kind") == "dispatch_deferred"
            and e.get("payload", e).get("lane") == "local_dispatch_rework"
        ]
        assert deferrals, "gated local rework deferral was not recorded"
        assert deferrals[-1].get("payload", deferrals[-1])["deferred_reason"] == reason

    def test_provider_throttle_defers_local_rework(self, repo: Path, fake_host) -> None:
        app = self._rework_pending(repo)
        calls = self._spy(fake_host)
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["throttled_until"] = "2999-01-01T00:00:00Z"
            state["throttle_reason"] = "operator_hold"
            state["throttle_adapter_kind"] = "devin"
            save_state(app.paths.state_file, state)

        result = app._local_dispatch_rework()

        assert calls == []
        self._assert_deferred(app, result, "provider_throttled")

    def test_fleet_lock_held_defers_local_rework(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch, fake_host
    ) -> None:
        import dataclasses

        app = self._rework_pending(repo)
        calls = self._spy(fake_host)
        app.config = dataclasses.replace(
            app.config,
            fleet=dataclasses.replace(
                app.config.fleet,
                global_max_concurrent_sessions=6,
                launch_lock_wait_seconds=0,
            ),
        )
        monkeypatch.setattr(
            "charlie_work.orchestration.local_lanes.try_acquire_fleet_lock", lambda _o: None
        )

        result = app._local_dispatch_rework()

        assert calls == []
        self._assert_deferred(app, result, "fleet_lock_held")

    def test_concurrency_governor_clamps_local_rework(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch, fake_host
    ) -> None:
        from types import SimpleNamespace

        app = self._rework_pending(repo)
        calls = self._spy(fake_host)
        requested: list[int] = []

        def _governor(limit: int, **_kw: object) -> SimpleNamespace:
            requested.append(limit)
            return SimpleNamespace(dispatch_limit=0, report_fields=lambda: {"fleet_max": 6})

        monkeypatch.setattr(app, "_apply_concurrency_governor", _governor)

        result = app._local_dispatch_rework()

        assert requested == [1]
        assert calls == []
        self._assert_deferred(app, result, "concurrency_cap")

    def test_ungated_local_rework_still_launches(self, repo: Path, fake_host) -> None:
        """Control: same state, no active gate -> the worker launches."""
        app = self._rework_pending(repo)
        calls = self._spy(fake_host)

        result = app._local_dispatch_rework()

        assert [r.issue_number for r in calls] == [7]
        assert result["dispatched"] == [7]


class TestLocalReworkFleetThrottle:
    """Issue #1993: the local rework lane honours the fleet-wide provider window
    and the staggered resume, keyed on the selected worker adapter."""

    @staticmethod
    def _app_with_window(repo: Path, fleet: Path, *, minutes_ahead: int):
        """Local rework app whose SIBLING repo (registered in the fleet) holds the window.

        The window lives in another repo's state.json -- the cross-repo case the
        per-repo permit gate cannot see, which only the fleet gate covers.
        """
        app = rework_pending_app(repo)
        app.fleet_dir_override = str(fleet)
        adapter = fpt.worker_adapter_kind(app.config.worker.harness)
        sibling_state = fleet / "sibling" / "state.json"
        sibling_state.parent.mkdir(parents=True)
        until = datetime.now(UTC) + timedelta(minutes=minutes_ahead)
        save_state(
            sibling_state,
            set_throttled_until(
                load_state(sibling_state),
                until.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                reason="rate_limited",
                adapter_kind=adapter,
                source="test",
            ),
        )
        (fleet / "fleet.json").write_text(
            json.dumps({"repos": {"o/sibling": {"state_dir": str(sibling_state.parent)}}}),
            encoding="utf-8",
        )
        return app, adapter

    def test_active_fleet_window_defers_local_rework(
        self, repo: Path, tmp_path: Path, fake_host
    ) -> None:
        app, _adapter = self._app_with_window(repo, tmp_path / "fleet", minutes_ahead=30)
        calls = spy_dispatch_sessions(fake_host)

        result = app._local_dispatch_rework()

        assert calls == []
        TestLocalReworkLaunchGates._assert_deferred(app, result, "provider_throttled_fleet")

    def test_expired_window_admits_one_launch_and_stamps_probe(
        self, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fleet = tmp_path / "fleet"
        app, adapter = self._app_with_window(repo, fleet, minutes_ahead=-5)
        launched: list[int] = []

        def _fake(_root, _manifest, _results, _settings, requests):
            launched.extend(r.issue_number for r in requests)
            return [
                SessionDispatchResult(
                    issue_number=r.issue_number,
                    issue_title=r.issue_title,
                    prompt_path=str(r.prompt_path),
                    branch_name=r.branch_name,
                    adapter=adapter,
                    ok=True,
                    pid=os.getpid(),
                )
                for r in requests
            ]

        monkeypatch.setattr("charlie_work.adapters.dispatch_sessions", _fake)

        result = app._local_dispatch_rework()

        assert launched == [7]
        assert result["dispatched"] == [7]
        stamp = json.loads(fpt.resume_probe_path(str(fleet)).read_text(encoding="utf-8"))
        assert [p["pid"] for p in stamp[adapter]["probes"]] == [os.getpid()]


class TestLocalReworkCarriesIssueLabels:
    """TIS-CW-6: the local rework lane hands the issue's labels to the launch
    gate, so a ``model:<tier>`` issue reworks on its tier as the remote lane
    does. Before the fix every local rework request carried no labels."""

    def test_live_issue_labels_reach_the_launch(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch, fake_host
    ) -> None:
        app = rework_pending_app(repo)
        calls = spy_dispatch_sessions(fake_host)
        monkeypatch.setattr(
            type(app.gh),
            "issue_view",
            lambda _self, _n: {"number": 7, "labels": [{"name": "model:opus"}, {"name": "x"}]},
        )

        result = app._local_dispatch_rework()

        assert result["dispatched"] == [7]
        assert [r.labels for r in calls] == [("model:opus", "x")]

    def test_unreadable_issue_falls_back_to_the_state_snapshot(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch, fake_host
    ) -> None:
        from charlie_work.github import GitHubError

        app = rework_pending_app(repo)
        calls = spy_dispatch_sessions(fake_host)
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["issues"]["7"]["labels"] = ["model:opus"]
            save_state(app.paths.state_file, state)

        def _boom(_self: object, _n: int) -> dict:
            raise GitHubError("unreadable")

        monkeypatch.setattr(type(app.gh), "issue_view", _boom)

        result = app._local_dispatch_rework()

        assert result["dispatched"] == [7]
        assert [r.labels for r in calls] == [("model:opus",)]
