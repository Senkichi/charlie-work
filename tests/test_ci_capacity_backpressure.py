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
from charlie_work.ci_headroom import ALLOCATION_EVENT_KIND
from charlie_work.config import DevinConfig, DispatchConfig, OrchestratorConfig
from charlie_work.fleet_paths import fleet_dir
from charlie_work.instrumentation import log_event, query_events
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

    # max(0, floor(10 * 1.5) - max(0, min_in_flight_demand)) where
    # min_in_flight_demand = live_count(0) + open_pr_count(1) -- FakeGitHub's
    # one default PR (issue #1770 review finding 3's floor) = 14, still well
    # above the requested limit of 5.
    assert result.ci_headroom == 14
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


# ---------------------------------------------------------------------------
# Issue #1770 review finding 1: ci_headroom as the ONLY enabled term must
# still surface in a real end-to-end dispatch() call, not just in a direct
# _apply_concurrency_governor() call.
# ---------------------------------------------------------------------------


def test_ci_headroom_only_term_enabled_surfaces_in_dispatch_result(tmp_path: Path) -> None:
    """A repo that opts into ONLY the CI-headroom clamp -- every other
    governor knob (max_concurrent_sessions/global_max_concurrent_sessions/
    max_open_agent_prs) left at its 0 default, exactly the "opt into just
    the new clamp" rollout the config comment advertises -- must still see
    ci_headroom/clamped_by in a real ``dispatch()`` call's ``CommandResult
    .data``. Before the fix, every governor-gated call site's ``gov.enabled
    or gov.fleet_enabled or gov.open_pr_enabled`` check omitted
    ci_headroom_enabled, so this exact configuration produced a bare
    ``selected_count: 0`` with no explanation in ``result.data`` -- the
    precise "zero dispatch is a capacity question first" regression this
    test pins at the ``dispatch()`` boundary rather than the lower-level
    governor call the other tests in this file use.
    """
    _log_allocation([_target("test-owner/test-repo", capacity=5, demand=9)])
    app = _build_app(tmp_path, ratio=1.0)

    result = app.dispatch()

    assert result.ok is True
    # max(0, floor(5 * 1.0) - 9) = 0 -- FakeGitHub's one default issue (#123,
    # "automated-ready") is deferred by the clamp, never dispatched.
    assert result.data["selected_count"] == 0
    assert result.data["ci_headroom"] == 0
    assert result.data["ci_headroom_ratio"] == 1.0
    assert result.data["clamped_by"] == "ci_headroom"


# ---------------------------------------------------------------------------
# Issue #1770 review finding 3: min_in_flight_demand floors demand between
# allocation-pass refreshes.
# ---------------------------------------------------------------------------


def test_ci_headroom_floors_demand_with_live_sessions_and_open_prs(tmp_path: Path) -> None:
    """A stale-relative-to-reality allocation reading (demand=0, because
    ci_fleet's last pass predates this repo's in-flight work) must not let
    the clamp re-grant full headroom: the caller's own live_count/open_pr_
    count floor demand at what is already known to be in flight, even
    though the allocation event itself has not caught up yet."""
    _log_allocation([_target("test-owner/test-repo", capacity=5, demand=0)])
    app = _build_app(tmp_path, ratio=1.0)
    # One open agent PR already in flight -> open_pr_count=1, which becomes
    # min_in_flight_demand's floor (live_count=0, no session files).
    fake_gh = app.gh
    assert isinstance(fake_gh, FakeGitHub)
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "state": "OPEN",
        }
    ]

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    # Without the floor: max(0, floor(5*1.0) - 0) = 5, no clamp at all.
    # With the floor: max(0, floor(5*1.0) - max(0, 1)) = 4.
    assert result.ci_headroom == 4
    assert result.dispatch_limit == 4
    assert result.clamped is True
    assert result.clamped_by == "ci_headroom"


# ---------------------------------------------------------------------------
# Issue #1770 review finding 8: the ci_headroom_unavailable diagnostic lands
# in the per-repo events.db, not the fleet-wide one.
# ---------------------------------------------------------------------------


def test_ci_headroom_unavailable_diagnostic_lands_in_per_repo_store(tmp_path: Path) -> None:
    """No runner_allocation event exists at all, so ci_headroom_available
    takes its "no_data" fail-open path -- the resulting ci_headroom_
    unavailable diagnostic must land in THIS repo's own events.db (self.
    paths.state_file), under repo=self.repo_root.name, not in the fleet-wide
    store the runner_allocation read itself uses. Before finding 8's fix,
    both writes shared fleet_state_path/repo, so a per-repo events.db read
    (the shape check_draft_pr_blocked_events-style heartbeat checks use)
    would never find this diagnostic."""
    app = _build_app(tmp_path, ratio=1.5)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    assert result.ci_headroom is None
    # Per-repo store: app.paths.state_file, keyed by the repo directory name.
    per_repo_events = query_events(app.paths.state_file, kind="ci_headroom_unavailable")
    assert len(per_repo_events) == 1
    assert per_repo_events[0]["repo"] == app.repo_root.name
    assert per_repo_events[0]["payload"]["reason"] == "no_data"
    # The fleet-wide store must NOT also carry a copy under the old key.
    fleet_state_path = layout.state_file_path(fleet_dir())
    fleet_events = query_events(fleet_state_path, kind="ci_headroom_unavailable")
    assert fleet_events == []


# ---------------------------------------------------------------------------
# Issue #1770 review finding 9: dry_run suppresses the event write but not
# the clamp itself.
# ---------------------------------------------------------------------------


def test_ci_headroom_dry_run_clamps_but_suppresses_event(tmp_path: Path) -> None:
    """dry_run=True must still apply the clamp (a preview must match what a
    live pass would select) but must not write the durable dispatch_
    backpressure event -- the same write-suppression discipline the
    open_pr_max block already documents."""
    _log_allocation([_target("test-owner/test-repo", capacity=5, demand=9)])
    config = OrchestratorConfig(
        dispatch=DispatchConfig(ci_capacity_headroom_ratio=1.0, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    assert result.ci_headroom == 0
    assert result.dispatch_limit == 0
    assert result.clamped is True
    assert result.clamped_by == "ci_headroom"
    events = query_events(app.paths.state_file, kind="dispatch_backpressure")
    assert events == []


# ---------------------------------------------------------------------------
# Issue #1770 review finding 9: governor-level fail-open coverage (not just
# the pure-function level test_ci_headroom.py already has).
# ---------------------------------------------------------------------------


def test_ci_headroom_governor_fails_open_when_demand_is_pinned(tmp_path: Path) -> None:
    """A pinned target's demand=0 bookkeeping placeholder must fail open at
    the governor level too, not only in ci_headroom_available's own unit
    tests -- proves the governor passes pinned data through rather than
    treating None specially in some way that only the direct-call tests
    would catch."""
    _log_allocation([_target("test-owner/test-repo", capacity=5, demand=0, pinned=True)])
    app = _build_app(tmp_path, ratio=1.5)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    assert result.ci_headroom_enabled is True
    assert result.ci_headroom is None
    assert result.dispatch_limit == 5
    assert result.clamped is False
    assert result.clamped_by is None


# ---------------------------------------------------------------------------
# Issue #1770 review findings 9/10: two governor terms clamping in the same
# pass leave clamped_by naming the LAST (tightest) term, and both events
# report the SAME original requested_limit.
# ---------------------------------------------------------------------------


def test_ci_headroom_clamped_by_and_requested_limit_survive_two_clamps(tmp_path: Path) -> None:
    """open_pr_max clamps first (5 -> 3), then ci_headroom clamps tighter
    still (3 -> 1). clamped_by must end at "ci_headroom" (the term that
    actually bound), and BOTH dispatch_backpressure events must report
    requested_limit=5 -- the caller's original ask -- not 3, the
    intermediate value open_pr_max left behind (finding 10)."""
    _log_allocation([_target("test-owner/test-repo", capacity=5, demand=4)])
    config = OrchestratorConfig(
        dispatch=DispatchConfig(
            ci_capacity_headroom_ratio=1.0,
            max_open_agent_prs=5,
            default_limit=5,
        ),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "state": "OPEN",
        },
        {
            "number": 457,
            "title": "Fix #124: parse",
            "headRefName": "agent/issue-124-fix-parse",
            "baseRefName": "main",
            "state": "OPEN",
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # open_pr_available = max(0, 5 - 2) = 3 -> clamps 5 down to 3.
    # min_in_flight_demand = live_count(0) + open_pr_count(2) = 2;
    # effective_demand = max(4, 2) = 4; ci_headroom = max(0, 5 - 4) = 1,
    # which is < 3 -> clamps 3 down to 1.
    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    assert result.dispatch_limit == 1
    assert result.clamped is True
    assert result.clamped_by == "ci_headroom"

    events = query_events(app.paths.state_file, kind="dispatch_backpressure")
    assert len(events) == 2
    open_pr_event, ci_headroom_event = events
    assert open_pr_event["payload"]["requested_limit"] == 5
    assert open_pr_event["payload"]["clamped_limit"] == 3
    assert ci_headroom_event["payload"]["clamped_by"] == "ci_headroom"
    assert ci_headroom_event["payload"]["requested_limit"] == 5
    assert ci_headroom_event["payload"]["clamped_limit"] == 1
