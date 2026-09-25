"""Concurrency governor helper and open-agent-PR backpressure.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the ``test_apply_concurrency_governor*`` seam -- the governor helper's slot
clamping and the open-PR backpressure clamp (issue #1129). Shared fakes and
helpers in ``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work.config import (
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
)
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_apply_concurrency_governor_helper_unlimited(tmp_path: Path) -> None:
    """_apply_concurrency_governor returns unclamped result when max_concurrent is 0."""
    config = OrchestratorConfig(
        # Issue #1903: the tree-headroom term ships ON (default cpu_count//2)
        # and would clamp a 5-wide request on small hosts -- pin both
        # host-load knobs to 0 so this test exercises max_concurrent alone.
        dispatch=DispatchConfig(
            max_concurrent_sessions=0,
            host_load_max_pytest_processes=0,
            host_load_max_pytest_trees=0,
        ),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app._apply_concurrency_governor(5)

    assert result.clamped is False
    assert result.max_concurrent == 0
    assert result.live_count == 0
    assert result.available_slots == 5
    assert result.dispatch_limit == 5


def test_apply_concurrency_governor_helper_clamped(tmp_path: Path, monkeypatch) -> None:
    """_apply_concurrency_governor returns clamped result when sessions are alive."""

    def mock_count_live(sessions_dir, state_file=None):
        return 2

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app._apply_concurrency_governor(5)

    assert result.clamped is True
    assert result.max_concurrent == 2
    assert result.live_count == 2
    assert result.available_slots == 0
    assert result.dispatch_limit == 0


def test_apply_concurrency_governor_helper_partial_slots(tmp_path: Path, monkeypatch) -> None:
    """_apply_concurrency_governor returns partial clamped result when some slots available."""

    def mock_count_live(sessions_dir, state_file=None):
        return 1

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app._apply_concurrency_governor(5)

    assert result.clamped is True
    assert result.max_concurrent == 2
    assert result.live_count == 1
    assert result.available_slots == 1
    assert result.dispatch_limit == 1


# ---------------------------------------------------------------------------
# Issue #1129: open-PR backpressure in the dispatch governor
# ---------------------------------------------------------------------------


def test_apply_concurrency_governor_open_pr_backpressure_clamps(tmp_path: Path) -> None:
    """Issue #1129: fresh dispatch clamped by open agent PR count."""

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_open_agent_prs=2, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Default FakeGitHub has 1 open PR with headRefName "agent/issue-123-fix-search".
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    assert result.open_pr_enabled is True
    assert result.open_pr_max == 2
    assert result.open_pr_count == 1
    # max(0, 2 - 1) = 1 slot available for fresh dispatch
    assert result.dispatch_limit == 1
    assert result.clamped is True


def test_apply_concurrency_governor_open_pr_backpressure_zero_when_full(tmp_path: Path) -> None:
    """Issue #1129: fresh dispatch clamped to 0 when open PRs meet the cap."""

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_open_agent_prs=1, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    assert result.open_pr_count == 1
    assert result.open_pr_max == 1
    assert result.dispatch_limit == 0
    assert result.clamped is True


def test_apply_concurrency_governor_open_pr_backpressure_off_when_disabled(tmp_path: Path) -> None:
    """Issue #1129: max_open_agent_prs=0 preserves current behavior."""

    config = OrchestratorConfig(
        # Issue #1903: pin both default-on host-load knobs off so the only
        # term under test is open_pr_max.
        dispatch=DispatchConfig(
            max_open_agent_prs=0,
            default_limit=5,
            host_load_max_pytest_processes=0,
            host_load_max_pytest_trees=0,
        ),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    assert result.open_pr_enabled is False
    assert result.open_pr_max == 0
    assert result.open_pr_count == 0
    assert result.dispatch_limit == 5
    assert result.clamped is False


def test_apply_concurrency_governor_open_pr_backpressure_exempt_by_default(tmp_path: Path) -> None:
    """Issue #1129: rework/loop paths (apply_open_pr_backpressure=False) are exempt."""

    config = OrchestratorConfig(
        # Issue #1903: pin both default-on host-load knobs off so the only
        # term under test is the open_pr_max exemption.
        dispatch=DispatchConfig(
            max_open_agent_prs=1,
            default_limit=5,
            host_load_max_pytest_processes=0,
            host_load_max_pytest_trees=0,
        ),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Default apply_open_pr_backpressure=False — the rework/loop path.
    result = app._apply_concurrency_governor(5)

    assert result.open_pr_enabled is False
    assert result.open_pr_max == 0
    assert result.open_pr_count == 0
    assert result.dispatch_limit == 5
    assert result.clamped is False


def test_apply_concurrency_governor_open_pr_backpressure_records_event(tmp_path: Path) -> None:
    """Issue #1129: dispatch_backpressure event recorded when clamp reduces limit."""

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_open_agent_prs=1, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    events = query_events(paths.state_file, kind="dispatch_backpressure")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["open_pr_count"] == 1
    assert payload["max_open_agent_prs"] == 1
    assert payload["requested_limit"] == 5
    assert payload["clamped_limit"] == 0


def test_apply_concurrency_governor_open_pr_backpressure_no_event_when_not_clamping(
    tmp_path: Path,
) -> None:
    """Issue #1129: no event when open PRs are below the cap (no clamping)."""

    config = OrchestratorConfig(
        # Issue #1903: pin both default-on host-load knobs off -- an armed
        # tree cap would clamp and write a dispatch_backpressure event of
        # its own, breaking the empty-events assertion.
        dispatch=DispatchConfig(
            max_open_agent_prs=10,
            default_limit=5,
            host_load_max_pytest_processes=0,
            host_load_max_pytest_trees=0,
        ),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # 1 open PR, cap 10 → 9 available, dispatch_limit 5 → no clamp.
    app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    events = query_events(paths.state_file, kind="dispatch_backpressure")
    assert events == []


def test_apply_concurrency_governor_open_pr_backpressure_no_event_under_dry_run(
    tmp_path: Path,
) -> None:
    """Issue #1129 rework: dry-run must not record the dispatch_backpressure event.

    The clamp behavior (dispatch_limit reduction, clamped flag, open_pr_count/
    open_pr_max on the result) must still engage so a dry-run preview reports
    the same clamped selected_count a live pass would -- matching the
    provider_throttled deferral precedent in
    _dispatch_impl, where the deferral is NOT dry-run-gated.
    Only the log_event write to events.db is suppressed under dry_run.
    """

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_open_agent_prs=1, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Same scenario as test_apply_concurrency_governor_open_pr_backpressure_records_event:
    # 1 open PR, cap 1 → 0 available, dispatch_limit 5 → clamp to 0.
    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    # The clamp still engages under dry-run (preview must match a live pass).
    assert result.clamped is True
    assert result.open_pr_count == 1
    assert result.open_pr_max == 1
    assert result.dispatch_limit == 0

    # But no durable event is written to events.db.
    events = query_events(paths.state_file, kind="dispatch_backpressure")
    assert events == []


def test_apply_concurrency_governor_open_pr_backpressure_combined_with_sessions(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #1129: open-PR clamp combines with session cap (min of all clamps)."""

    def mock_count_live(sessions_dir, state_file=None):
        return 1

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=3, max_open_agent_prs=2, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    # Session cap: max(0, 3 - 1) = 2
    # Open-PR cap: max(0, 2 - 1) = 1
    # Effective: min(2, 1) = 1
    assert result.dispatch_limit == 1
    assert result.clamped is True
    assert result.open_pr_count == 1
