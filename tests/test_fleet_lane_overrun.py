"""A slow repo lane must not block other lanes or the pass (issue #2142).

Two fake lanes: one blocks on an Event, one is fast. The pass must return
within its budget with the fast lane's results, emit ``fleet_lane_overrun``
for the blocked lane, and the next pass must skip the blocked repo while its
supervisor lock is still held. Shared hermeticity fixtures live in
``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from _fleet_dispatch_fixtures import (
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
    _per_repo_runtime_paths,
)
from charlie_work import fleet_dispatch, layout
from charlie_work.config import FleetSupervisorConfig, OrchestratorConfig
from charlie_work.fleet_dispatch import fleet_loop
from charlie_work.instrumentation import query_events
from charlie_work.workflow import CommandResult

# Cap on how long a fake wedged lane blocks, and the wall-clock backstop for
# the bounded-return assertion: a pass that ever waited on the wedged lane
# takes at least this long, while a correct pass is bounded by its
# deadline-derived wait budgets (~deadline + drain grace, twice over).
_RELEASE_TIMEOUT = 30.0


def _app(
    key: str,
    release: threading.Event | None,
    finished: threading.Event | None = None,
) -> MagicMock:
    app = MagicMock()

    def _loop(limit: int | None, merge: bool | None = None, **_kwargs: Any) -> CommandResult:
        if release is not None:
            release.wait(timeout=_RELEASE_TIMEOUT)
        if finished is not None:
            finished.set()
        return CommandResult(True, f"{key} loop complete", {})

    app.loop.side_effect = _loop
    return app


def _tick_clock() -> Callable[[], float]:
    """A deterministic ``pass_clock``: one millisecond per call, no wall time.

    Issue #2251: ``fleet_loop`` measures ``deadline_seconds`` from pass
    entry, but the allocation/autoscale prologues and per-repo prep run
    before the first lane is submitted -- on a loaded runner the whole 1s
    budget can be spent before any lane holds a pool slot, deferring every
    repo. A call-counted clock keeps every deadline-derived bound (the
    full-pool wait budget, the final drain) intact while making the lane
    submission order independent of wall-clock timing: the deadline cannot
    trip during submission, so which repo defers is decided by the pool
    throttle alone, and the wedged lane's ``release`` event -- not elapsed
    time -- decides when its lane ends. The clock is strictly increasing,
    so recorded ``elapsed_seconds`` values stay positive.
    """
    tick = 0.0

    def _clock() -> float:
        nonlocal tick
        tick += 0.001
        return tick

    return _clock


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_blocked_lane_overruns_without_blocking_pass(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(fleet_dispatch, "_LANE_DRAIN_GRACE_SECONDS", 0.2, raising=False)
    names = ("slow", "fast")
    mock_load_registry.return_value = {
        "repos": {
            f"owner/{n}": {"repo_root": str(tmp_path / n), "config_path": "c.yaml"} for n in names
        }
    }
    for n in names:
        (tmp_path / n).mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()

    release = threading.Event()
    slow_finished = threading.Event()
    apps = {"slow": _app("slow", release, finished=slow_finished), "fast": _app("fast", None)}
    mock_app_class.side_effect = lambda root, *a, **k: apps[Path(root).name]
    fleet_dir = str(tmp_path / "fleet")

    def _run_pass() -> Any:
        return fleet_loop(
            fleet_dir_override=fleet_dir,
            global_config=None,
            repos=None,
            limit=1,
            merge=True,
            dry_run=False,
            work_only=False,
            deadline_seconds=1,
            pass_clock=_tick_clock(),
        )

    try:
        started = time.monotonic()
        result = _run_pass()
        elapsed = time.monotonic() - started

        # Bounded return: the pass came back while the wedged lane was still
        # blocked on ``release`` (the deterministic check), long before the
        # lane's own self-unblock cap (the wall-clock backstop).
        assert not slow_finished.is_set()
        assert elapsed < _RELEASE_TIMEOUT
        assert result.data["repos"]["owner/fast"]["ok"] is True
        assert "owner/slow" not in result.data["repos"]
        assert "owner/slow" in result.data["deadline_partial_repo_keys"]
        fleet_state_path = layout.state_file_path(fleet_dispatch.fleet_dir(override=fleet_dir))
        overruns = query_events(fleet_state_path, kind="fleet_lane_overrun")
        assert len(overruns) == 1
        assert overruns[0]["payload"]["repo_key"] == "owner/slow"
        assert overruns[0]["level"] == "warning"
        assert overruns[0]["payload"]["elapsed_seconds"] > 0

        # The still-running lane holds its supervisor lock: next pass skips it.
        second = _run_pass()
        slow = second.data["repos"]["owner/slow"]
        assert slow["message"] == "supervisor lock held, skipped"
    finally:
        release.set()


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_wedged_lane_on_full_pool_defers_waiting_repo_and_returns_bounded(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Pool width 1 across 2 repos: the first lane wedges past the wait budget,
    so the submission throttle times out and the waiting repo is deferred
    (never prepped, no supervisor lock taken) while the pass returns bounded."""
    monkeypatch.setattr(fleet_dispatch, "_LANE_DRAIN_GRACE_SECONDS", 0.2, raising=False)
    names = ("a", "b")
    mock_load_registry.return_value = {
        "repos": {
            f"owner/{n}": {"repo_root": str(tmp_path / n), "config_path": "c.yaml"} for n in names
        }
    }
    for n in names:
        (tmp_path / n).mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()

    release = threading.Event()
    lane_finished = threading.Event()
    apps = {n: _app(n, release, finished=lane_finished) for n in names}
    mock_app_class.side_effect = lambda root, *a, **k: apps[Path(root).name]
    fleet_dir = str(tmp_path / "fleet")

    try:
        started = time.monotonic()
        result = fleet_loop(
            fleet_dir_override=fleet_dir,
            global_config=OrchestratorConfig(
                fleet_supervisor=FleetSupervisorConfig(fleet_lane_concurrency=1)
            ),
            repos=None,
            limit=1,
            merge=True,
            dry_run=False,
            work_only=False,
            deadline_seconds=1,
            pass_clock=_tick_clock(),
        )
        elapsed = time.monotonic() - started

        # Bounded return: the pass came back while the wedged lane was still
        # blocked on ``release`` (the deterministic check), long before the
        # lane's own self-unblock cap (the wall-clock backstop).
        assert not lane_finished.is_set()
        assert elapsed < _RELEASE_TIMEOUT
        # Exactly one repo got the single pool slot (wedged); the other waited.
        assert len(result.data["deferred"]) == 1
        assert len(result.data["deadline_partial_repo_keys"]) == 1
        wedged = result.data["deadline_partial_repo_keys"][0]
        waiting = result.data["deferred"][0]
        assert wedged != waiting
        assert result.data["repos"] == {}
        # The deferred repo's lane body and prep never ran.
        assert apps[waiting.split("/")[1]].loop.call_count == 0
        assert mock_app_class.call_count == 1
        fleet_state_path = layout.state_file_path(fleet_dispatch.fleet_dir(override=fleet_dir))
        deferred = query_events(fleet_state_path, kind="fleet_pass_deadline_deferred")
        assert len(deferred) == 1
        assert deferred[0]["payload"]["deferred_repo_keys"] == [waiting]
    finally:
        release.set()
