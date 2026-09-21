"""Wiring tests for the CI-headroom clamp in ``_apply_concurrency_governor``
(issue #1770, step 2 of 2).

``ci_headroom_available`` itself (the pure computation over the fleet's
``runner_allocation`` event) is covered by ``tests/test_ci_headroom.py``. This
file covers the governor wiring: the clamp only engages for fresh-issue
dispatch, fails open when the reading is unavailable, and leaves repos with
no self-hosted ``runner_allocation`` entry (e.g. hosted-runner repos)
unaffected -- derived from the fleet's own data, never a repo-name list.

``tests/conftest.py``'s autouse ``_isolate_fleet_registry`` fixture points
``CHARLIE_WORK_FLEET_DIR`` at a per-test ``tmp_path`` for every test in the
suite, so ``fleet_dir()`` with no override already resolves safely here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from charlie_work.config import DevinConfig, DispatchConfig, OrchestratorConfig
from charlie_work.fleet_paths import fleet_dir
from charlie_work.instrumentation import log_event, query_events
from charlie_work.orchestration.ci_headroom import ALLOCATION_EVENT_KIND
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from charlie_work import layout
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def _target(
    repo: str,
    *,
    capacity: int,
    demand: int,
    pinned: bool = False,
) -> dict[str, Any]:
    return {
        "repo": repo,
        "capacity": capacity,
        "demand": demand,
        "running": min(capacity, demand),
        "target": min(capacity, demand),
        "pinned": pinned,
    }


def _log_allocation(targets: list[dict[str, Any]], *, budget: int = 8) -> None:
    """Write a ``runner_allocation`` event to this test's isolated fleet state,
    shaped like ``ci_fleet.runner_allocation.plan_summary``'s output."""
    fleet_state_path = layout.state_file_path(fleet_dir())
    log_event(
        fleet_state_path,
        ALLOCATION_EVENT_KIND,
        {
            "budget": budget,
            "budget_reason": "test",
            "targets": targets,
            "changes": [],
            "notes": [],
        },
    )


def _build_app(tmp_path: Path, ratio: float) -> OrchestratorApp:
    config = OrchestratorConfig(
        dispatch=DispatchConfig(ci_capacity_headroom_ratio=ratio, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    return OrchestratorApp(tmp_path, paths, config, fake_gh)


# ---------------------------------------------------------------------------
# Clamp engages for fresh-issue dispatch
# ---------------------------------------------------------------------------


def test_ci_headroom_clamps_fresh_dispatch(tmp_path: Path) -> None:
    """Demand already at/above capacity clamps fresh dispatch down."""
    # FakeGitHub.name_with_owner() -> "test-owner/test-repo".
    _log_allocation([_target("test-owner/test-repo", capacity=5, demand=9)])
    app = _build_app(tmp_path, ratio=1.0)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    # max(0, floor(5 * 1.0) - 9) = 0
    assert result.ci_headroom_enabled is True
    assert result.ci_headroom == 0
    assert result.dispatch_limit == 0
    assert result.clamped is True
    assert result.clamped_by == "ci_headroom"


def test_ci_headroom_clamps_to_partial_slots(tmp_path: Path) -> None:
    """A repo with some spare capacity is clamped to exactly that many slots,
    not to zero -- proves this is a real headroom computation, not a hard gate."""
    _log_allocation([_target("test-owner/test-repo", capacity=5, demand=3)])
    app = _build_app(tmp_path, ratio=1.0)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    # max(0, floor(5 * 1.0) - 3) = 2
    assert result.ci_headroom == 2
    assert result.dispatch_limit == 2
    assert result.clamped is True


def test_ci_headroom_does_not_clamp_when_ample(tmp_path: Path) -> None:
    """Headroom above the requested limit does not tighten it."""
    _log_allocation([_target("test-owner/test-repo", capacity=10, demand=0)])
    app = _build_app(tmp_path, ratio=1.5)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    # max(0, floor(10 * 1.5) - 0) = 15, well above the requested limit of 5.
    assert result.ci_headroom == 15
    assert result.dispatch_limit == 5
    assert result.clamped is False


def test_ci_headroom_event_records_clamp(tmp_path: Path) -> None:
    """The dispatch_backpressure event names ci_headroom as the clamp cause
    (design doc Section 4 step 3) so a zero-dispatch pass is explainable."""
    _log_allocation([_target("test-owner/test-repo", capacity=5, demand=9)])
    app = _build_app(tmp_path, ratio=1.0)

    app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    events = query_events(app.paths.state_file, kind="dispatch_backpressure")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["clamped_by"] == "ci_headroom"
    assert payload["ci_headroom"] == 0
    assert payload["ci_headroom_ratio"] == 1.0
    assert payload["requested_limit"] == 5
    assert payload["clamped_limit"] == 0


# ---------------------------------------------------------------------------
# Fail-open: no trustworthy reading means no clamp
# ---------------------------------------------------------------------------


def test_ci_headroom_fails_open_with_no_allocation_data(tmp_path: Path) -> None:
    """No runner_allocation event ever recorded -> ci_headroom is None,
    dispatch is not clamped. Without the fail-open discipline this would
    read as zero capacity and hard-stop every repo the first time the
    fleet's allocation pass has not run yet."""
    app = _build_app(tmp_path, ratio=1.5)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    assert result.ci_headroom_enabled is True
    assert result.ci_headroom is None
    assert result.dispatch_limit == 5
    assert result.clamped is False
    assert result.clamped_by is None


# ---------------------------------------------------------------------------
# Rework lane exemption: reworks reduce WIP, so they are never gated
# ---------------------------------------------------------------------------


def test_ci_headroom_exempts_rework_lane(tmp_path: Path) -> None:
    """Rework/recovery/loop callers pass apply_open_pr_backpressure=False (the
    default) and must stay unaffected even when the repo is severely over its
    CI capacity -- reworks of already-open PRs reduce WIP rather than adding
    to it, mirroring the existing open_pr_max exemption for the same lanes."""
    _log_allocation([_target("test-owner/test-repo", capacity=5, demand=18)])
    app = _build_app(tmp_path, ratio=1.0)

    # Default apply_open_pr_backpressure=False -- the rework/loop path.
    result = app._apply_concurrency_governor(5)

    assert result.ci_headroom_enabled is False
    assert result.ci_headroom is None
    assert result.dispatch_limit == 5
    assert result.clamped is False


# ---------------------------------------------------------------------------
# Hosted-runner repos (no self-hosted runner_allocation entry) are unaffected
# ---------------------------------------------------------------------------


def test_ci_headroom_unaffected_for_repo_absent_from_allocation_plan(tmp_path: Path) -> None:
    """A repo with no entry in the fleet's runner_allocation targets (the
    shape a hosted-runner-only repo takes -- ci_fleet's plan only ever lists
    repos discovered under managed_root, so a repo with no self-hosted
    registration simply never appears) is derived as unmeasured, never as
    zero capacity, so fresh dispatch for it is not clamped."""
    # Other repos are present and even severely starved -- this repo just
    # never shows up, exactly like a repo with no self-hosted runners at all.
    _log_allocation(
        [
            _target("Senkichi/job-cannon", capacity=5, demand=18),
            _target("Senkichi/swole", capacity=2, demand=6),
        ]
    )
    app = _build_app(tmp_path, ratio=1.0)

    # FakeGitHub.name_with_owner() -> "test-owner/test-repo", which has no
    # entry above -- standing in for a hosted-runner-only repo.
    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    assert result.ci_headroom_enabled is True
    assert result.ci_headroom is None
    assert result.dispatch_limit == 5
    assert result.clamped is False


def test_ci_headroom_off_by_default(tmp_path: Path) -> None:
    """ci_capacity_headroom_ratio=0 (the default) preserves current behavior
    even when the fleet data would otherwise clamp hard."""
    _log_allocation([_target("test-owner/test-repo", capacity=5, demand=18)])
    app = _build_app(tmp_path, ratio=0.0)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    assert result.ci_headroom_enabled is False
    assert result.ci_headroom is None
    assert result.dispatch_limit == 5
    assert result.clamped is False
