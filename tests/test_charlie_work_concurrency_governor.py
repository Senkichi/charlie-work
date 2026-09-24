"""Concurrency governor: per-pass dispatch clamps and result reporting.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the ``test_concurrency_governor_*`` seam -- live-session counting, per-pass
cap clamps, partial dispatch, and governor result fields. Sibling seams:
``test_charlie_work_concurrency_governor_zero.py`` (clamped-to-zero
self-explaining reporting) and ``test_charlie_work_concurrency_backpressure.py``
(open-PR backpressure). Shared fakes and helpers in ``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work.config import (
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    state_lock,
)
from charlie_work.workflow import (
    ConcurrencyGovernorResult,
    OrchestratorApp,
)
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_concurrency_governor_unlimited_when_unset(tmp_path: Path) -> None:
    """When max_concurrent_sessions is 0 (default), dispatch should behave as before (unlimited)."""
    config = OrchestratorConfig(
        # Issue #1843: host-load backpressure is ON by default, which would
        # splat its report_fields (including concurrency_limit) into
        # result.data -- this test pins the all-terms-off shape, so the new
        # knob is disabled here explicitly.
        dispatch=DispatchConfig(max_concurrent_sessions=0, host_load_max_pytest_processes=0),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch()

    # Should dispatch normally without concurrency clamping
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert "concurrency_limit" not in result.data


def test_concurrency_governor_clamps_dispatch_when_sessions_alive(
    tmp_path: Path, monkeypatch
) -> None:
    """When max_concurrent_sessions is set and there are live sessions, dispatch should be clamped."""

    # Mock _count_live_sessions to return 2 live sessions
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

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch()

    # Should clamp to 0 since 2 sessions are alive and cap is 2
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["concurrency_limit"] == 2
    assert result.data["live_session_count"] == 2
    assert result.data["available_slots"] == 0


def test_concurrency_governor_clamps_rework_dispatch(tmp_path: Path, monkeypatch) -> None:
    """Concurrency governor should also clamp rework dispatch."""

    # Mock _count_live_sessions to return 2 live sessions (at the cap)
    def mock_count_live(sessions_dir, state_file=None):
        return 2

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Add needs-rework label to the issue
            self.issues[0]["labels"] = [{"name": "agent:needs-rework"}]

        def issue_list(self, labels=None, state=None):
            # Support both old and new signature
            if isinstance(labels, str):
                ready_label = labels
                if ready_label == "agent:needs-rework":
                    return self.issues
                return []
            elif labels and "agent:needs-rework" in labels:
                return self.issues
            return []

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a rework prompt
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    # Should clamp to 0 since 2 sessions are alive and cap is 2
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["concurrency_limit"] == 2
    assert result.data["live_session_count"] == 2
    assert result.data["available_slots"] == 0


def test_concurrency_governor_allows_partial_dispatch(tmp_path: Path, monkeypatch) -> None:
    """When some slots are available, dispatch should launch up to that limit."""

    # Mock _count_live_sessions to return 1 live session
    def mock_count_live(sessions_dir, state_file=None):
        return 1

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Add a second ready issue to test truncation
    fake_gh.issues.append(
        {
            "number": 124,
            "title": "Another fix",
            "url": "https://example.test/issues/124",
            "body": "Another issue",
            "labels": [{"name": "automated-ready"}],
        }
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch()

    # Should allow only 1 launch since 1 session is alive and cap is 2
    # (2 candidates available, but only 1 slot)
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["concurrency_limit"] == 2
    assert result.data["live_session_count"] == 1
    assert result.data["available_slots"] == 1


def test_concurrency_governor_result_dataclass() -> None:
    """ConcurrencyGovernorResult is a frozen dataclass with all fields bound together."""
    result = ConcurrencyGovernorResult(
        clamped=True,
        max_concurrent=2,
        live_count=1,
        available_slots=1,
        dispatch_limit=1,
    )

    assert result.clamped is True
    assert result.max_concurrent == 2
    assert result.live_count == 1
    assert result.available_slots == 1
    assert result.dispatch_limit == 1

    # Test report_fields method
    fields = result.report_fields()
    assert fields == {
        "concurrency_limit": 2,
        "live_session_count": 1,
        "available_slots": 1,
    }

    # Test immutability (frozen dataclass)
    try:
        result.clamped = False
        assert False, "Should not be able to modify frozen dataclass"
    except dataclasses.FrozenInstanceError:
        pass


def test_concurrency_governor_clamps_only_issues_dispatch(tmp_path: Path, monkeypatch) -> None:
    """Issue #105: when --issues names more issues than available slots, excess should be deferred by concurrency."""

    # Mock _count_live_sessions to return 0 live sessions
    def mock_count_live(sessions_dir, state_file=None):
        return 0

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with 3 ready issues
    class FakeGitHubWithMultipleIssues(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 101,
                    "title": "First fix",
                    "url": "https://example.test/issues/101",
                    "body": "First issue",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
                {
                    "number": 102,
                    "title": "Second fix",
                    "url": "https://example.test/issues/102",
                    "body": "Second issue",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
                {
                    "number": 103,
                    "title": "Third fix",
                    "url": "https://example.test/issues/103",
                    "body": "Third issue",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
            ]

    fake_gh = FakeGitHubWithMultipleIssues()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Request dispatch of all 3 issues, but cap is 2
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(only_issues="101,102,103")

    # Should dispatch exactly 2, defer the third
    assert result.ok is True
    assert result.data["selected_count"] == 2
    assert result.data["concurrency_limit"] == 2
    assert result.data["live_session_count"] == 0
    assert result.data["available_slots"] == 2
    assert result.data["deferred_by_concurrency"] == [103]
    assert result.data["skipped_issue_numbers"] == []

    # Verify deferred issue was NOT marked as dispatched (no label/state mutation)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
    # Only the dispatched issues should be in state
    assert set(state["issues"].keys()) == {"101", "102"}
    assert "103" not in state["issues"]


def test_concurrency_governor_clamps_only_issues_dispatch_with_live_sessions(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #105: when --issues names more issues than available slots (with live sessions), excess should be deferred."""

    # Mock _count_live_sessions to return 1 live session
    def mock_count_live(sessions_dir, state_file=None):
        return 1

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with 3 ready issues
    class FakeGitHubWithMultipleIssues(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 101,
                    "title": "First fix",
                    "url": "https://example.test/issues/101",
                    "body": "First issue",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
                {
                    "number": 102,
                    "title": "Second fix",
                    "url": "https://example.test/issues/102",
                    "body": "Second issue",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
                {
                    "number": 103,
                    "title": "Third fix",
                    "url": "https://example.test/issues/103",
                    "body": "Third issue",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
            ]

    fake_gh = FakeGitHubWithMultipleIssues()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Request dispatch of all 3 issues, but only 1 slot available (2 cap - 1 live)
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(only_issues="101,102,103")

    # Should dispatch exactly 1, defer the other 2
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["concurrency_limit"] == 2
    assert result.data["live_session_count"] == 1
    assert result.data["available_slots"] == 1
    assert set(result.data["deferred_by_concurrency"]) == {102, 103}
    assert result.data["skipped_issue_numbers"] == []

    # Verify deferred issues were NOT marked as dispatched
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
    # Only the dispatched issue should be in state
    assert set(state["issues"].keys()) == {"101"}
    assert "102" not in state["issues"]
    assert "103" not in state["issues"]


def test_concurrency_governor_clamps_only_issues_dry_run(tmp_path: Path, monkeypatch) -> None:
    """Issue #105: dry-run with --issues should also respect concurrency governor."""

    # Mock _count_live_sessions to return 0 live sessions
    def mock_count_live(sessions_dir, state_file=None):
        return 0

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with 3 ready issues
    class FakeGitHubWithMultipleIssues(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 101,
                    "title": "First fix",
                    "url": "https://example.test/issues/101",
                    "body": "First issue",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
                {
                    "number": 102,
                    "title": "Second fix",
                    "url": "https://example.test/issues/102",
                    "body": "Second issue",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
                {
                    "number": 103,
                    "title": "Third fix",
                    "url": "https://example.test/issues/103",
                    "body": "Third issue",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
            ]

    fake_gh = FakeGitHubWithMultipleIssues()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Request dry-run dispatch of all 3 issues, but cap is 2
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(only_issues="101,102,103")

    # Should report exactly 2 would be dispatched, third deferred
    assert result.ok is True
    assert result.data["selected_count"] == 2
    assert result.data["concurrency_limit"] == 2
    assert result.data["live_session_count"] == 0
    assert result.data["available_slots"] == 2
    assert result.data["deferred_by_concurrency"] == [103]
    assert result.data["skipped_issue_numbers"] == []

    # Verify state is unchanged in dry-run
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
    assert state["issues"] == {}
    assert state["events"] == []


def test_concurrency_governor_clamps_only_issues_rework_dispatch(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #105: dispatch_rework with --issues should also respect concurrency governor."""

    # Mock _count_live_sessions to return 0 live sessions
    def mock_count_live(sessions_dir, state_file=None):
        return 0

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with 3 issues needing rework
    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 101,
                    "title": "First rework",
                    "url": "https://example.test/issues/101",
                    "body": "First issue",
                    "labels": [{"name": "agent:needs-rework"}],
                    "state": "OPEN",
                },
                {
                    "number": 102,
                    "title": "Second rework",
                    "url": "https://example.test/issues/102",
                    "body": "Second issue",
                    "labels": [{"name": "agent:needs-rework"}],
                    "state": "OPEN",
                },
                {
                    "number": 103,
                    "title": "Third rework",
                    "url": "https://example.test/issues/103",
                    "body": "Third issue",
                    "labels": [{"name": "agent:needs-rework"}],
                    "state": "OPEN",
                },
            ]
            # Add corresponding PRs (matching FakeGitHub's default PR 456 pattern)
            self.prs = [
                {
                    "number": 456,
                    "title": "Fix #101",
                    "url": "https://example.test/pull/456",
                    "headRefName": "agent/issue-101",
                    "headRefOid": "sha-abc101",
                    "body": "Closes #101",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                },
                {
                    "number": 457,
                    "title": "Fix #102",
                    "url": "https://example.test/pull/457",
                    "headRefName": "agent/issue-102",
                    "headRefOid": "sha-abc102",
                    "body": "Closes #102",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                },
                {
                    "number": 458,
                    "title": "Fix #103",
                    "url": "https://example.test/pull/458",
                    "headRefName": "agent/issue-103",
                    "headRefOid": "sha-abc103",
                    "body": "Closes #103",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                },
            ]

        def issue_list(self, labels=None, state=None):
            # Support both old and new signature
            if isinstance(labels, str):
                ready_label = labels
                if ready_label == "agent:needs-rework":
                    return self.issues
                return []
            elif labels and "agent:needs-rework" in labels:
                return self.issues
            return []

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Initialize state with rework_requested status for all 3 issues
    from charlie_work.state import save_state

    initial_state = {
        "issues": {
            "101": {"status": "rework_requested", "branch": "agent/issue-101"},
            "102": {"status": "rework_requested", "branch": "agent/issue-102"},
            "103": {"status": "rework_requested", "branch": "agent/issue-103"},
        },
        "prs": {},
        "events": [],
        "generated_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, initial_state)

    # Create rework prompts for all 3 PRs
    for pr_num in [456, 457, 458]:
        pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{pr_num}"
        pr_dir.mkdir(parents=True)
        rework_prompt = pr_dir / "rework-prompt.md"
        rework_prompt.write_text("Fix the issues", encoding="utf-8")

    # Request rework dispatch of all 3 issues, but cap is 2
    result = app.dispatch_rework(only_issues="101,102,103")

    # Should dispatch exactly 2, defer the third
    assert result.ok is True
    assert result.data["selected_count"] == 2
    assert result.data["concurrency_limit"] == 2
    assert result.data["live_session_count"] == 0
    assert result.data["available_slots"] == 2
    assert result.data["deferred_by_concurrency"] == [103]


def test_concurrency_governor_result_unclamped() -> None:
    """ConcurrencyGovernorResult correctly represents unclamped state."""
    result = ConcurrencyGovernorResult(
        clamped=False,
        max_concurrent=0,
        live_count=0,
        available_slots=5,
        dispatch_limit=5,
    )

    assert result.clamped is False
    assert result.max_concurrent == 0
    assert result.live_count == 0
    assert result.available_slots == 5
    assert result.dispatch_limit == 5

    # report_fields should still work even when unclamped
    fields = result.report_fields()
    assert fields == {
        "concurrency_limit": 0,
        "live_session_count": 0,
        "available_slots": 5,
    }

    # Test enabled property
    assert result.enabled is False  # max_concurrent=0 means disabled


def test_concurrency_governor_result_enabled_property() -> None:
    """ConcurrencyGovernorResult.enabled property correctly reflects governor enabled state."""
    # Disabled (max_concurrent=0)
    disabled = ConcurrencyGovernorResult(
        clamped=False,
        max_concurrent=0,
        live_count=0,
        available_slots=5,
        dispatch_limit=5,
    )
    assert disabled.enabled is False

    # Enabled but not clamped (max_concurrent > 0, available_slots >= dispatch_limit)
    enabled_unclamped = ConcurrencyGovernorResult(
        clamped=False,
        max_concurrent=5,
        live_count=0,
        available_slots=5,
        dispatch_limit=5,
    )
    assert enabled_unclamped.enabled is True

    # Enabled and clamped (max_concurrent > 0, available_slots < dispatch_limit)
    enabled_clamped = ConcurrencyGovernorResult(
        clamped=True,
        max_concurrent=5,
        live_count=4,
        available_slots=1,
        dispatch_limit=5,
    )
    assert enabled_clamped.enabled is True


def test_concurrency_governor_result_open_pr_report_fields() -> None:
    """Issue #1129: report_fields includes open-PR fields when enabled."""

    result = ConcurrencyGovernorResult(
        clamped=True,
        max_concurrent=0,
        live_count=0,
        available_slots=5,
        dispatch_limit=1,
        open_pr_count=3,
        open_pr_max=4,
    )
    fields = result.report_fields()
    assert fields["open_pr_count"] == 3
    assert fields["open_pr_max"] == 4
    assert result.open_pr_enabled is True


def test_concurrency_governor_result_open_pr_report_fields_absent_when_disabled() -> None:
    """Issue #1129: report_fields omits open-PR fields when open_pr_max=0."""

    result = ConcurrencyGovernorResult(
        clamped=False,
        max_concurrent=0,
        live_count=0,
        available_slots=5,
        dispatch_limit=5,
    )
    fields = result.report_fields()
    assert "open_pr_count" not in fields
    assert "open_pr_max" not in fields
    assert result.open_pr_enabled is False
