"""Concurrent per-repo lanes and the out-of-band reap scheduler (issue #1934).

``fleet_loop`` used to run each repo's whole lane inline in one ``for``
loop, so a fleet-wide pass cost the SUM of every repo's lane (~35-42 min
observed across 6 repos against a configured 5-minute cadence) and every
lane-embedded sweep -- including the dead-review-claim reap -- starved
behind sibling lanes. These tests pin the replacement semantics:

* lane bodies run on a bounded ``ThreadPoolExecutor`` while repo prep
  (deadline check, config load, lock acquisition, app construction) stays
  deterministic on the submitting thread;
* result ordering, error isolation, and lock hold/release are unchanged;
* the fleet supervisor runs ``_run_fleet_reap_sweeps`` once per repo on its
  own interval from a dedicated thread, independent of -- and concurrent
  with -- any in-flight pass, and never takes the supervisor lock.

Shared helpers and the autouse hermeticity fixtures live in
``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from _fleet_dispatch_fixtures import (
    _StepClock,
    _drained_fleet_result,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
    _per_repo_runtime_paths,
)
from charlie_work import fleet_dispatch, layout
from charlie_work.config import OrchestratorConfig, SupervisorConfig
from charlie_work.fleet_dispatch import (
    _fleet_reap_sweep_loop,
    _resolve_fleet_lane_concurrency,
    _run_fleet_reap_sweep,
    fleet_loop,
    run_fleet_supervise,
)
from charlie_work.instrumentation import query_events
from charlie_work.supervise import try_acquire_supervisor_lock
from charlie_work.workflow import CommandResult

_LANE_GATE_TIMEOUT = 15.0


def _registry(*names: str, root: Path) -> dict[str, Any]:
    return {
        "repos": {
            f"owner/{name}": {
                "repo_root": str(root / name),
                "config_path": "orchestrator.config.yaml",
            }
            for name in names
        }
    }


def _gated_app(
    key: str,
    live: dict[str, int],
    peak: dict[str, int],
    gate: threading.Event,
    guard: threading.Lock,
    trip_at: int,
) -> MagicMock:
    """An OrchestratorApp mock whose ``loop()`` blocks until ``trip_at``
    lanes are concurrently in flight. Under the pre-#1934 serial loop the
    gate can never trip, so these tests spend the timeout once and fail."""
    app = MagicMock()

    def _loop(limit: int | None, merge: bool | None = None) -> CommandResult:
        with guard:
            live["n"] += 1
            peak["n"] = max(peak["n"], live["n"])
            if live["n"] >= trip_at:
                gate.set()
        gate.wait(timeout=_LANE_GATE_TIMEOUT)
        with guard:
            live["n"] -= 1
        return CommandResult(True, f"{key} loop complete", {})

    app.loop.side_effect = _loop
    return app


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_runs_repo_lanes_concurrently(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """Three lanes must be simultaneously in flight inside one pass.

    Each mocked ``loop()`` blocks until all three lanes have entered it; a
    serial pass would sit at the gate until the timeout and report peak==1.
    """
    names = ("repo1", "repo2", "repo3")
    mock_load_registry.return_value = _registry(*names, root=tmp_path)
    for name in names:
        (tmp_path / name).mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()

    live: dict[str, int] = {"n": 0}
    peak: dict[str, int] = {"n": 0}
    gate = threading.Event()
    guard = threading.Lock()
    mock_app_class.side_effect = [
        _gated_app(name, live, peak, gate, guard, trip_at=3) for name in names
    ]

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    assert result.ok is True
    assert peak["n"] == 3
    for name in names:
        assert result.data["repos"][f"owner/{name}"]["ok"] is True


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_lane_concurrency_is_bounded(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """``supervisor.fleet_lane_concurrency`` caps live lanes; a cap of 2
    across 4 repos peaks at exactly 2 live lanes."""
    names = ("repo1", "repo2", "repo3", "repo4")
    mock_load_registry.return_value = _registry(*names, root=tmp_path)
    for name in names:
        (tmp_path / name).mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()

    live: dict[str, int] = {"n": 0}
    peak: dict[str, int] = {"n": 0}
    gate = threading.Event()
    guard = threading.Lock()
    # trip_at=2: the gate releases once the cap is reached, so the remaining
    # lanes can be admitted as pool slots free up.
    mock_app_class.side_effect = [
        _gated_app(name, live, peak, gate, guard, trip_at=2) for name in names
    ]

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=OrchestratorConfig(supervisor=SupervisorConfig(fleet_lane_concurrency=2)),
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    assert result.ok is True
    assert peak["n"] == 2
    assert len(result.data["repos"]) == 4
    for name in names:
        assert result.data["repos"][f"owner/{name}"]["ok"] is True


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_deadline_defers_lanes_waiting_on_a_full_pool(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """#1832 under #1934: a lane whose turn arrives past the deadline is
    deferred before its prep runs -- it is never queued on a full pool.

    ``fleet_lane_concurrency=2`` across 4 selected repos. The step clock
    keeps repo1/repo2/repo3's top-of-loop deadline checks under budget, then
    jumps past it for the re-check repo3 makes after the throttle collects
    repo1's finished lane -- waiting for a pool slot consumed the rest of
    the budget. repo3 and repo4 must land in ``deferred`` with no app built
    for either (a third OrchestratorApp construction would mean a lane was
    submitted), while the two in-flight lanes finish cooperatively. On the
    unthrottled submit-all-upfront shape this replaces, every repo's prep
    ran before the first deadline check could trip, so all four lanes
    executed to completion past ``max_pass_runtime_seconds``.
    """
    names = ("repo1", "repo2", "repo3", "repo4")
    mock_load_registry.return_value = _registry(*names, root=tmp_path)
    for name in names:
        (tmp_path / name).mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()

    apps = []
    for name in ("repo1", "repo2"):
        app = MagicMock()
        app.dispatch.return_value = CommandResult(True, f"{name} dispatch complete", {})
        apps.append(app)
    mock_app_class.side_effect = apps

    # pass_clock reads (work_only skips the prologue deadline checks):
    #   1. pass_started_at                         -> 0
    #   2. repo1 deadline check                    -> 0   (under the 100s budget)
    #   3. repo1 repo_lane_start                   -> 0
    #   4. repo2 deadline check                    -> 0
    #   5. repo2 repo_lane_start                   -> 0
    #   6. repo3 deadline check                    -> 50  (still under -- proceed)
    #   7. throttle collects repo1's lane; its elapsed log reads -> 60
    #   8. repo3's post-wait re-check              -> 10000 (blown -> defer)
    #   9. repo4 deadline check                    -> 10000 (defer)
    #   10+. drain elapsed / deferred-event reads  -> 10000
    clock = _StepClock(steps=[0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 60.0], after=10000.0)

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=OrchestratorConfig(supervisor=SupervisorConfig(fleet_lane_concurrency=2)),
        repos=("owner/repo1", "owner/repo2", "owner/repo3", "owner/repo4"),
        limit=3,
        work_only=True,
        deadline_seconds=100,
        pass_clock=clock,
    )

    assert result.ok is True
    assert set(result.data["repos"]) == {"owner/repo1", "owner/repo2"}
    assert result.data["deferred"] == ["owner/repo3", "owner/repo4"]
    assert mock_app_class.call_count == 2
    for app in apps:
        app.dispatch.assert_called_once_with(3)

    from charlie_work.fleet_paths import fleet_dir

    fleet_state_path = layout.state_file_path(fleet_dir(override=str(tmp_path / "fleet")))
    deferred_events = query_events(fleet_state_path, kind="fleet_pass_deadline_deferred")
    assert len(deferred_events) == 1
    assert deferred_events[0]["payload"]["deferred_repo_keys"] == [
        "owner/repo3",
        "owner/repo4",
    ]


@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_results_follow_selection_order_not_completion_order(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_try_acquire: MagicMock,
    tmp_path: Path,
) -> None:
    """repo1 finishes last but stays first in ``result.data["repos"]``.

    repo2's lock is held externally, so its skip result is recorded during
    submission -- before either sibling lane finishes. Selection order, not
    record order, must win.
    """
    names = ("repo1", "repo2", "repo3")
    mock_load_registry.return_value = _registry(*names, root=tmp_path)
    for name in names:
        (tmp_path / name).mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()

    other_lane_done = threading.Event()
    finished: list[str] = []
    guard = threading.Lock()

    def _app_for(name: str) -> MagicMock:
        app = MagicMock()

        def _loop(limit: int | None, merge: bool | None = None) -> CommandResult:
            if name == "repo1":
                # Deliberately the last lane to finish.
                other_lane_done.wait(timeout=_LANE_GATE_TIMEOUT)
            with guard:
                finished.append(name)
                if "repo3" in finished:
                    other_lane_done.set()
            return CommandResult(True, f"{name} loop complete", {})

        app.loop.side_effect = _loop
        return app

    mock_app_class.side_effect = [_app_for("repo1"), _app_for("repo3")]

    # repo1 -> lock granted, repo2 -> held externally (skipped), repo3 ->
    # lock granted.
    mock_try_acquire.side_effect = [MagicMock(), None, MagicMock()]

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=("owner/repo1", "owner/repo2", "owner/repo3"),
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    assert finished == ["repo3", "repo1"]  # repo1 really did finish last
    assert result.data["repos"]["owner/repo2"]["reason"] == "supervisor_lock_held"
    assert list(result.data["repos"]) == [
        "owner/repo1",
        "owner/repo2",
        "owner/repo3",
    ]


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_lane_exception_isolated_and_lock_released(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """A lane body raising mid-pool produces the same per-repo error result
    the serial loop produced, does not poison sibling lanes, and still
    releases the supervisor lock its prep acquired."""
    names = ("repo1", "repo2")
    mock_load_registry.return_value = _registry(*names, root=tmp_path)
    for name in names:
        (tmp_path / name).mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()

    app1 = MagicMock()
    app1.loop.side_effect = RuntimeError("lane exploded")
    app2 = MagicMock()
    app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.side_effect = [app1, app2]

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    repo1 = result.data["repos"]["owner/repo1"]
    assert repo1["ok"] is False
    assert repo1["errored"] is True
    assert "fleet pass error" in repo1["message"]
    assert "RuntimeError" in repo1["message"]
    assert result.data["repos"]["owner/repo2"]["ok"] is True

    # The lane's supervisor lock was released: a fresh acquire on repo1's
    # lock file succeeds after the pass.
    lock_path = layout.supervisor_lock_path(tmp_path / "repo1" / ".var" / "charlie-work")
    freed = try_acquire_supervisor_lock(lock_path)
    try:
        assert freed is not None
    finally:
        if freed is not None:
            freed.release()


def test_resolve_fleet_lane_concurrency_defaults_and_overrides() -> None:
    """The lane cap reads ``supervisor.fleet_lane_concurrency`` when a global
    config supplies one; otherwise (or when misconfigured) the built-in
    default applies."""
    assert _resolve_fleet_lane_concurrency(None) == 8
    assert _resolve_fleet_lane_concurrency(OrchestratorConfig()) == 8
    assert (
        _resolve_fleet_lane_concurrency(
            OrchestratorConfig(supervisor=SupervisorConfig(fleet_lane_concurrency=2))
        )
        == 2
    )
    # <=0 and non-int spellings fall back to the built-in default at the
    # call site rather than failing the pass.
    assert (
        _resolve_fleet_lane_concurrency(
            OrchestratorConfig(supervisor=SupervisorConfig(fleet_lane_concurrency=0))
        )
        == 8
    )


# ---------------------------------------------------------------------------
# Out-of-band fleet reap sweep
# ---------------------------------------------------------------------------


def _sweep_summary() -> dict[str, Any]:
    """The shape ``OrchestratorApp._run_review_reap_sweeps`` returns."""
    return {
        "verdict_result": {"recorded": [1], "missed": []},
        "reconciled_verdicts": [],
        "stalled": [{"pr": 7}],
        "reaped_checkouts": [],
        "orphaned_checkouts": [],
    }


@patch("charlie_work.fleet_lanes._load_registry")
@patch("charlie_work.fleet_lanes.load_layered_config")
@patch("charlie_work.fleet_lanes.runtime_paths")
@patch("charlie_work.fleet_lanes.GitHub")
@patch("charlie_work.fleet_lanes.OrchestratorApp")
def test_fleet_reap_sweep_runs_once_per_repo_and_records_events(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """The sweep runs ``_run_review_reap_sweeps`` once per selected repo --
    the identical block ``dispatch_reviews`` runs -- and records one
    ``fleet_reap_sweep`` event per repo into the fleet-level events.db."""
    names = ("repo1", "repo2")
    mock_load_registry.return_value = _registry(*names, root=tmp_path)
    for name in names:
        (tmp_path / name).mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()

    apps = []
    for name in names:
        app = MagicMock()
        app._run_review_reap_sweeps.return_value = _sweep_summary()
        apps.append(app)
    mock_app_class.side_effect = apps

    fleet_dir = tmp_path / "fleet"
    results = _run_fleet_reap_sweep(fleet_dir_override=str(fleet_dir))

    assert set(results) == {"owner/repo1", "owner/repo2"}
    for app in apps:
        app._run_review_reap_sweeps.assert_called_once()
    for name in names:
        assert results[f"owner/{name}"]["stalled_reaped"] == 1
        assert results[f"owner/{name}"]["verdicts_recorded"] == 1

    events = query_events(layout.state_file_path(fleet_dir), kind="fleet_reap_sweep")
    assert len(events) == 2
    assert {e["repo"] for e in events} == {"owner/repo1", "owner/repo2"}
    assert all(e["level"] == "info" for e in events)


@patch("charlie_work.fleet_lanes._load_registry")
@patch("charlie_work.fleet_lanes.load_layered_config")
@patch("charlie_work.fleet_lanes.runtime_paths")
@patch("charlie_work.fleet_lanes.GitHub")
@patch("charlie_work.fleet_lanes.OrchestratorApp")
def test_fleet_reap_sweep_repo_failure_isolated(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """One repo's sweep raising must not starve the rest of the round; the
    failure is recorded as a warning-level fleet_reap_sweep event."""
    names = ("repo1", "repo2")
    mock_load_registry.return_value = _registry(*names, root=tmp_path)
    for name in names:
        (tmp_path / name).mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()

    app1 = MagicMock()
    app1._run_review_reap_sweeps.side_effect = RuntimeError("boom")
    app2 = MagicMock()
    app2._run_review_reap_sweeps.return_value = _sweep_summary()
    mock_app_class.side_effect = [app1, app2]

    fleet_dir = tmp_path / "fleet"
    results = _run_fleet_reap_sweep(fleet_dir_override=str(fleet_dir))

    assert "error" in results["owner/repo1"]
    assert results["owner/repo2"]["stalled_reaped"] == 1
    app2._run_review_reap_sweeps.assert_called_once()

    events = query_events(layout.state_file_path(fleet_dir), kind="fleet_reap_sweep")
    by_repo = {e["repo"]: e for e in events}
    assert by_repo["owner/repo1"]["level"] == "warning"
    assert "RuntimeError" in by_repo["owner/repo1"]["payload"]["error"]
    assert by_repo["owner/repo2"]["level"] == "info"


@patch("charlie_work.fleet_lanes._load_registry")
@patch("charlie_work.fleet_lanes.load_layered_config")
@patch("charlie_work.fleet_lanes.runtime_paths")
@patch("charlie_work.fleet_lanes.GitHub")
@patch("charlie_work.fleet_lanes.OrchestratorApp")
def test_fleet_reap_sweep_never_takes_supervisor_lock(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """AC: the sweep frees dead claims even while the repo's lane holds its
    supervisor lock.

    Hold repo1's real supervisor lock, and make the fleet_dispatch
    ``try_acquire_supervisor_lock`` seam blow up if the sweep ever probes
    it: the sweep must still run to completion. The sweep set is
    state_lock-serialized / merge-on-write safe against a concurrent lane
    (issue #1874), so it needs -- and takes -- no supervisor lock.
    """
    mock_load_registry.return_value = _registry("repo1", root=tmp_path)
    (tmp_path / "repo1").mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()

    app = MagicMock()
    app._run_review_reap_sweeps.return_value = _sweep_summary()
    mock_app_class.return_value = app

    held = try_acquire_supervisor_lock(
        layout.supervisor_lock_path(tmp_path / "repo1" / ".var" / "charlie-work")
    )
    assert held is not None
    try:
        with patch(
            "charlie_work.fleet_dispatch.try_acquire_supervisor_lock",
            MagicMock(side_effect=AssertionError("sweep must not probe the lock")),
        ):
            results = _run_fleet_reap_sweep(fleet_dir_override=str(tmp_path / "fleet"))
    finally:
        held.release()

    app._run_review_reap_sweeps.assert_called_once()
    assert results["owner/repo1"]["stalled_reaped"] == 1


@patch("charlie_work.fleet_lanes._load_registry")
@patch("charlie_work.fleet_lanes.load_layered_config")
@patch("charlie_work.fleet_lanes.runtime_paths")
@patch("charlie_work.fleet_lanes.GitHub")
@patch("charlie_work.fleet_lanes.OrchestratorApp")
def test_fleet_reap_sweep_skips_stale_registry_entries(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """A registry entry whose repo_root no longer exists is skipped without
    failing the round -- same stance as the lane loop's stale handling."""
    mock_load_registry.return_value = {
        "repos": {
            "owner/dead": {"repo_root": str(tmp_path / "nonexistent")},
            "owner/repo1": {"repo_root": str(tmp_path / "repo1")},
        }
    }
    (tmp_path / "repo1").mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()

    app = MagicMock()
    app._run_review_reap_sweeps.return_value = _sweep_summary()
    mock_app_class.return_value = app

    results = _run_fleet_reap_sweep(fleet_dir_override=str(tmp_path / "fleet"))

    assert set(results) == {"owner/repo1"}
    app._run_review_reap_sweeps.assert_called_once()


def test_fleet_reap_sweep_loop_interval_and_stop() -> None:
    """The scheduler loop fires on the interval and exits promptly when the
    stop event is set mid-wait."""
    calls: list[int] = []
    stop = threading.Event()

    def _sweep(**_kwargs: Any) -> None:
        calls.append(1)
        if len(calls) >= 2:
            stop.set()

    with patch("charlie_work.fleet_lanes._run_fleet_reap_sweep", side_effect=_sweep):
        _fleet_reap_sweep_loop(
            stop,
            0.01,
            fleet_dir_override=None,
            repos=None,
            dry_run=False,
        )

    assert len(calls) >= 2


def test_fleet_reap_sweep_loop_survives_a_failing_round() -> None:
    """A raising round is logged and the scheduler keeps running."""
    stop = threading.Event()
    calls: list[int] = []

    def _sweep(**_kwargs: Any) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient registry error")
        stop.set()

    with patch("charlie_work.fleet_lanes._run_fleet_reap_sweep", side_effect=_sweep):
        _fleet_reap_sweep_loop(
            stop,
            0.01,
            fleet_dir_override=None,
            repos=None,
            dry_run=False,
        )

    assert len(calls) == 2


@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_starts_and_stops_reap_scheduler(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """run_fleet_supervise starts the reap scheduler with the configured
    interval and stops it (thread joined, stop event set) on exit."""
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            reap_sweep_interval_seconds=300,
        )
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    captured: dict[str, Any] = {}
    real_start = fleet_dispatch._start_fleet_reap_scheduler

    def _spy(**kwargs: Any) -> Any:
        captured["kwargs"] = kwargs
        thread, stop_event = real_start(**kwargs)
        captured["thread"] = thread
        captured["stop_event"] = stop_event
        return thread, stop_event

    monkeypatch.setattr("charlie_work.fleet_dispatch._start_fleet_reap_scheduler", _spy)

    result = run_fleet_supervise(
        fleet_dir_override=str(tmp_path / "fleet"),
        max_passes=1,
        clock=lambda: 0.0,
        sleep=lambda _s: None,
    )

    assert result.ok is True
    assert captured["kwargs"]["interval_seconds"] == 300
    assert captured["kwargs"]["fleet_dir_override"] == str(tmp_path / "fleet")
    assert captured["stop_event"].is_set()
    captured["thread"].join(timeout=5)
    assert not captured["thread"].is_alive()


@patch("charlie_work.fleet_dispatch._start_fleet_reap_scheduler")
@patch("charlie_work.fleet_dispatch.fleet_loop")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.try_acquire_supervisor_lock")
def test_run_fleet_supervise_reap_scheduler_disabled(
    mock_lock: MagicMock,
    mock_load_config: MagicMock,
    mock_fleet_loop: MagicMock,
    mock_start_scheduler: MagicMock,
    tmp_path: Path,
) -> None:
    """reap_sweep_interval_seconds <= 0 keeps the scheduler off (kill switch)."""
    mock_load_config.return_value = OrchestratorConfig(
        supervisor=SupervisorConfig(
            poll_interval_seconds=5,
            full_pass_interval_seconds=1,
            reap_sweep_interval_seconds=0,
        )
    )
    mock_fleet_loop.return_value = _drained_fleet_result()

    result = run_fleet_supervise(
        fleet_dir_override=str(tmp_path / "fleet"),
        max_passes=1,
        clock=lambda: 0.0,
        sleep=lambda _s: None,
    )

    assert result.ok is True
    mock_start_scheduler.assert_not_called()
