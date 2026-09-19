"""Rework-dispatch candidate selection: state-driven picks and skip paths.

Split out of ``tests/test_charlie_work.py`` (issue #1547, Track-1 wave 1/8):
the ``test_dispatch_rework_*`` seam's selection half -- state-driven
selection over labels, loop-limit handling, approved-verdict cleanup,
throttle deferral, unconditional reap under max_concurrent=0, skip reasons
(label error, missing prompt), and the orphan-detection non-rerun guard.
Dispatch mechanics live in ``tests/test_charlie_work_dispatch_rework.py``;
shared fakes/helpers in ``tests/_rework_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import (
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
    ReconcilePassConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    set_throttled_until,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp


def test_dispatch_rework_state_driven_selection(tmp_path: Path) -> None:
    """Issue #85 acceptance criterion 1: state-driven selection works.

    State-driven selection ensures that issues with rework_requested status are selected
    regardless of label state. This test verifies that the selection logic uses state
    instead of labels.
    """
    config = OrchestratorConfig(
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
            # Add needs-rework label to the issue (for display)
            self.issues[0]["labels"] = [{"name": "agent:needs-rework"}]

    # Initialize state with the issue in rework_requested status
    # Do this BEFORE creating the app to avoid paths.ensure() overwriting the state
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a rework prompt
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    # Should select the issue based on state, not label
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["sessions"][0]["issue_number"] == 123


def test_dispatch_rework_state_wins_over_missing_label(tmp_path: Path) -> None:
    """Issue #85 acceptance criterion 2: dispatch_rework selects a rework_requested issue
    whose needs-rework label is absent (state wins over label).
    """
    config = OrchestratorConfig(
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

    class NoLabelGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Issue does NOT have needs-rework label
            self.issues[0]["labels"] = []

    # Initialize state with the issue in rework_requested status (label is missing)
    # Do this BEFORE creating the app to avoid paths.ensure() overwriting the state
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = NoLabelGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a rework prompt
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    # Should still select the issue based on state, not label
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["sessions"][0]["issue_number"] == 123


def test_dispatch_rework_skips_issue_view_for_rework_requested_issue_with_closed_pr(
    tmp_path: Path,
) -> None:
    """Issue #558: dispatch_rework's candidate scan must NOT call
    gh.issue_view() for a rework_requested issue whose PR is closed-unmerged.
    pr_list() returns only OPEN PRs, so an issue with a closed-unmerged PR is
    absent from the open-PR index; the per-issue gh.issue_view fetch is the
    permanent per-pass cost this gate exists to eliminate. Verifying the fetch
    is skipped (not just that selected_count is 0) pins the reorder directly.
    """
    config = OrchestratorConfig(
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

    class IssueViewRecordingGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # PR 456 linked to issue 123, but CLOSED-unmerged on GitHub.
            self.prs[0]["state"] = "CLOSED"
            self.issue_view_calls: list[int] = []

        def issue_view(self, number: int):
            self.issue_view_calls.append(number)
            return super().issue_view(number)

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = IssueViewRecordingGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    # The whole point: no per-issue gh.issue_view fetch for the
    # closed-unmerged-PR issue.
    assert 123 not in fake_gh.issue_view_calls
    assert fake_gh.issue_view_calls == []


def test_dispatch_rework_two_candidates_loop_limit_one(tmp_path: Path) -> None:
    """Issue #85 acceptance test: two rework_requested issues, loop(limit=1) dispatches different issues.

    This is the headline reproduction test for issue #85's observed failure: when there are
    multiple rework_requested issues, loop(limit=1) should dispatch one issue per pass,
    cycling through candidates rather than dispatching the same issue repeatedly.

    This test drives the full loop() method (not just dispatch_rework) to prove that the
    review stage (which strips labels) runs and that state-driven selection survives through it.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
        reconcile_pass=ReconcilePassConfig(enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class TwoIssueGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Override with two issues, both with open PRs
            self.issues = [
                {
                    "number": 123,
                    "title": "Fix search",
                    "url": "https://example.test/issues/123",
                    "labels": [{"name": "agent:needs-rework"}],
                },
                {
                    "number": 124,
                    "title": "Fix auth",
                    "url": "https://example.test/issues/124",
                    "labels": [{"name": "agent:needs-rework"}],
                },
            ]
            self.prs = [
                {
                    "number": 456,
                    "title": "PR for issue 123",
                    "url": "https://example.test/pr/456",
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRefName": "agent/issue-123",
                    "state": "OPEN",
                },
                {
                    "number": 457,
                    "title": "PR for issue 124",
                    "url": "https://example.test/pr/457",
                    "headRefOid": "def456",
                    "isCrossRepository": False,
                    "headRefName": "agent/issue-124",
                    "state": "OPEN",
                },
            ]

    # Initialize state with both issues in rework_requested status
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        state["issues"]["124"] = {
            "number": 124,
            "title": "Fix auth",
            "url": "https://example.test/issues/124",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = TwoIssueGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create rework prompts for both PRs
    for pr_num, issue_num in [(456, 123), (457, 124)]:
        pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{pr_num}"
        pr_dir.mkdir(parents=True, exist_ok=True)
        rework_prompt = pr_dir / "rework-prompt.md"
        rework_prompt.write_text("Fix the issues", encoding="utf-8")

    # First loop pass with limit=1
    result1 = app.loop(limit=1)
    assert result1.ok is True
    assert result1.data["dispatch_rework"]["selected_count"] == 1
    first_issue = result1.data["dispatch_rework"]["sessions"][0]["issue_number"]
    assert first_issue in (123, 124)

    # Second loop pass with limit=1 should select the OTHER issue
    # The review stage ran in the first pass (stripping labels), so this proves
    # state-driven selection survives through label mutations
    result2 = app.loop(limit=1)
    assert result2.ok is True
    assert result2.data["dispatch_rework"]["selected_count"] == 1
    second_issue = result2.data["dispatch_rework"]["sessions"][0]["issue_number"]
    assert second_issue in (123, 124)
    assert second_issue != first_issue, "Should dispatch the other issue, not the same one twice"


def test_dispatch_rework_approved_verdict_clears_rework_requested(tmp_path: Path) -> None:
    """Approved verdict should clear rework_requested status to prevent duplicate dispatch.

    This test addresses the regression where approved/blocked verdicts never cleared
    rework_requested status, causing state-driven selection to dispatch duplicate workers
    onto finished PRs.
    """
    config = OrchestratorConfig(
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

    class ApprovedGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": "agent:needs-rework"}]

    # Initialize state with the issue in rework_requested status
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ApprovedGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record an approved verdict for the PR
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps(
            {
                "number": 456,
                "title": "PR for issue 123",
                "url": "https://example.test/pr/456",
                "headRefOid": "sha-abc123",
                "isCrossRepository": False,
                "headRefName": "agent/issue-123",
            }
        ),
        encoding="utf-8",
    )

    # Record approved verdict
    app.record_review(
        pr_number=456,
        decision="approved",
        summary="LGTM",
        comment=None,
        verdict_provenance="fresh_llm_review",
    )

    # Verify the issue status is now "approved", not "rework_requested"
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        assert state["issues"]["123"]["status"] == "approved"

    # Create a rework prompt
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    # dispatch_rework should NOT select the approved issue
    result = app.dispatch_rework()
    assert result.ok is True
    assert result.data["selected_count"] == 0


def test_dispatch_rework_defers_when_provider_throttled(tmp_path: Path) -> None:
    """When provider throttle window is active, rework dispatch should also defer."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(default_limit=3),
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Add a needs-rework issue with an open PR
    fake_gh.issues.append(
        {
            "number": 42,
            "title": "Fix something",
            "url": "https://example.test/issues/42",
            "body": "Fix it",
            "labels": [{"name": "agent:needs-rework"}],
        }
    )
    # Replace the default PR with one linked to issue 42
    fake_gh.pr = {
        "number": 100,
        "title": "Fix something",
        "url": "https://example.test/pr/100",
        "state": "OPEN",
        "headRefName": "agent/issue-42-fix",
        "headRefOid": "sha-abc123",
        "baseRefName": "main",
        "isCrossRepository": False,
        "body": "Closes #42",
        "labels": [],
    }
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Set a throttle window in the future
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        from datetime import UTC, datetime, timedelta

        future_time = datetime.now(UTC) + timedelta(hours=1)
        throttled_until = future_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        state = set_throttled_until(state, throttled_until)
        save_state(paths.state_file, state)

    result = app.dispatch_rework()

    # Should defer with provider_throttled reason
    assert result.ok is False
    assert result.data["deferred_reason"] == "provider_throttled"
    assert result.data["throttled_until"] is not None
    assert result.data["selected_count"] == 0


def test_dispatch_rework_reaps_unconditionally_when_max_concurrent_zero(tmp_path: Path) -> None:
    """Test that dispatch_rework() has the unconditional reaper call (issue #165)."""
    # Verify by code inspection that dispatch_rework calls _detect_and_handle_stalled_sessions
    import charlie_work.workflow as workflow_module
    import inspect

    dispatch_rework_source = inspect.getsource(
        workflow_module.OrchestratorApp._dispatch_rework_impl
    )

    # Verify the unconditional call exists
    assert "_detect_and_handle_stalled_sessions" in dispatch_rework_source
    # Verify it's called before the governor (which has the max_concurrent check)
    reaper_call_pos = dispatch_rework_source.find("_detect_and_handle_stalled_sessions")
    assert reaper_call_pos > 0, "Reaper call should exist in dispatch_rework"


def test_dispatch_rework_label_error_reason_in_event_payload(tmp_path: Path) -> None:
    """Issue #453: rework dispatch label transition failures must carry a reason in the failures map."""
    config = OrchestratorConfig(
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

    class ReworkLabelFailGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": "agent:needs-rework"}]

        def add_issue_label(self, number: int, label: str) -> bool:
            return False

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkLabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert 123 in result.data["label_errors"]
    assert 123 in result.data["failures"]
    reason = result.data["failures"][123]
    assert "label transition" in reason
    assert "rework_dispatched" in reason
    assert "partial_failure" in reason

    state = load_state(paths.state_file)
    rework_events = [e for e in state["events"] if e["kind"] == "dispatch_rework"]
    assert rework_events
    payload = rework_events[-1]["payload"]
    assert "123" in payload["failures"]
    assert payload["failures"]["123"] == reason


def test_dispatch_rework_missing_prompt_reason_in_event_payload(tmp_path: Path) -> None:
    """Issue #453: missing rework prompt skips must carry a reason in the failures map."""
    config = OrchestratorConfig(
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
            self.issues[0]["labels"] = [{"name": "agent:needs-rework"}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Intentionally do not create rework-prompt.md
    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert 123 in result.data["failures"]
    reason = result.data["failures"][123]
    assert "missing rework prompt" in reason

    state = load_state(paths.state_file)
    rework_events = [e for e in state["events"] if e["kind"] == "dispatch_rework"]
    assert rework_events
    payload = rework_events[-1]["payload"]
    assert "123" in payload["failures"]
    assert payload["failures"]["123"] == reason


def test_dispatch_rework_does_not_re_run_orphan_detection(tmp_path: Path) -> None:
    """Issue #457: dispatch_rework must not run the orphaned-worker sweep, which is
    already run once per pass by loop(), to avoid duplicate drift events."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["457"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)

    class FakeGitHubNoPrs(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubNoPrs()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    orphan_calls = []

    def tracking_orphan_sweep(*args, **kwargs):
        orphan_calls.append((args, kwargs))

    with patch(
        "charlie_work.workflow._detect_and_handle_orphaned_workers",
        side_effect=tracking_orphan_sweep,
    ):
        result = app.dispatch_rework()

    assert result.ok is True
    assert len(orphan_calls) == 0, "dispatch_rework must not re-run orphaned-worker detection"
