"""Issue #2039: local rework dispatch applies the remote rework lane's launch gates.

Split from ``test_local_lane.py`` (file-size ratchet); reuses its fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _local_lane_fixtures import new_repo_root, rework_pending_app, spy_dispatch_sessions
from charlie_work.state import load_state, load_state_locked, save_state, state_lock
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

    def test_provider_throttle_defers_local_rework(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = self._rework_pending(repo)
        calls = self._spy(monkeypatch)
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
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dataclasses

        app = self._rework_pending(repo)
        calls = self._spy(monkeypatch)
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
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        app = self._rework_pending(repo)
        calls = self._spy(monkeypatch)
        requested: list[int] = []

        def _governor(limit: int, **_kw: object) -> SimpleNamespace:
            requested.append(limit)
            return SimpleNamespace(dispatch_limit=0, report_fields=lambda: {"fleet_max": 6})

        monkeypatch.setattr(app, "_apply_concurrency_governor", _governor)

        result = app._local_dispatch_rework()

        assert requested == [1]
        assert calls == []
        self._assert_deferred(app, result, "concurrency_cap")

    def test_ungated_local_rework_still_launches(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: same state, no active gate -> the worker launches."""
        app = self._rework_pending(repo)
        calls = self._spy(monkeypatch)

        result = app._local_dispatch_rework()

        assert [r.issue_number for r in calls] == [7]
        assert result["dispatched"] == [7]
