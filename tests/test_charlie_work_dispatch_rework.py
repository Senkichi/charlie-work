"""Rework-dispatch mechanics: transitions, claims, and event payloads.

Split out of ``tests/test_charlie_work.py`` (issue #1547, Track-1 wave 1/8):
the ``test_dispatch_rework_*`` seam's dispatch half -- candidate discovery,
rework_dispatched transitions, claim release/restore on skip/failure, and
failure/label-error event payloads. Sibling seams:
``tests/test_charlie_work_dispatch_rework_selection.py`` (state-driven
selection and skip conditions),
``tests/test_charlie_work_dispatch_rework_routing.py`` (head-moved review
routing), and the safety-net files
``tests/test_charlie_work_dispatch_rework_{worktree,caps,briefs}.py``;
shared fakes/helpers in ``tests/_rework_dispatch_fixtures.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp


def test_dispatch_rework_skips_manual_adapter(tmp_path: Path) -> None:
    """Rework dispatch must skip manual adapters to preserve human-paste path."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["adapter"] == "manual"
    assert result.data["selected_count"] == 0


def test_dispatch_rework_finds_needs_rework_issues_with_open_prs(tmp_path: Path) -> None:
    """Rework dispatch must find issues with rework_requested status and open PRs."""
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

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert str(result.data["sessions"][0]["prompt_path"]).endswith("rework-prompt.md")
    assert result.data["sessions"][0]["branch_name"] == "agent/issue-123-fix-search"


def test_dispatch_rework_transitions_to_rework_dispatched(tmp_path: Path) -> None:
    """Rework dispatch must transition to rework_dispatched label on success."""
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

    assert result.ok is True
    assert (123, "agent:in-progress") in fake_gh.labels_added
    assert (123, "agent:needs-rework") in fake_gh.labels_removed


def test_dispatch_rework_transition_failure_recorded(tmp_path: Path) -> None:
    """Issue #135: PARTIAL_FAILURE during rework_dispatched transition must be recorded."""
    from charlie_work.labels import TransitionOutcome

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
            # Return False to simulate add failure (error-as-value)
            return False

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

    fake_gh = ReworkLabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a rework prompt
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert 123 in result.data["label_errors"]
    state = load_state(paths.state_file)
    label_error = state["issues"]["123"]["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "rework_dispatched"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value


def test_dispatch_rework_releases_claims_when_all_skipped(tmp_path: Path) -> None:
    """When all candidates lack rework-prompt.md, dispatch_pending claims must be released.

    Issue #116: Missing rework-prompt.md may be transient (review agent hasn't written it yet),
    so restore to rework_requested for retry instead of dispatch_failed.
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

    # Do NOT create a rework prompt - this should trigger the all-skipped path
    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    # Verify the claim was released: status should be rework_requested (not dispatch_pending)
    state = load_state(paths.state_file)
    issue_state = state["issues"].get("123")
    assert issue_state is not None
    # Issue #116: restore to rework_requested for retry (missing prompt may be transient)
    assert issue_state.get("status") == "rework_requested"
    assert issue_state.get("dispatch_pending_at") is None


def test_dispatch_rework_restores_rework_requested_on_dispatch_failure(tmp_path: Path) -> None:
    """Issue #116: Failed rework dispatch must restore status to rework_requested for retry.

    When a rework dispatch attempt fails (e.g., git worktree add error), the issue's
    status must be restored to rework_requested so it can be retried in the next pass.
    The bug was that failed dispatches left status as dispatch_failed, permanently
    excluding the issue from rework selection (state-driven selection).
    """
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
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

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a rework prompt
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    # First dispatch attempt fails
    result = app.dispatch_rework()

    # result.ok is False when there are dispatch failures
    assert result.ok is False
    assert result.data["selected_count"] == 0
    assert result.data["failed_count"] == 1

    # Verify status is restored to rework_requested (not dispatch_failed)
    state = load_state(paths.state_file)
    issue_state = state["issues"].get("123")
    assert issue_state is not None
    assert issue_state.get("status") == "rework_requested"

    # Second dispatch attempt should select the issue again
    # Fix the command to succeed
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
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["failed_count"] == 0

    # Verify the issue is now dispatched
    state = load_state(paths.state_file)
    issue_state = state["issues"].get("123")
    assert issue_state is not None
    assert issue_state.get("status") == "dispatched"
    assert issue_state.get("dispatch_pending_at") is None


def test_dispatch_rework_failure_reason_in_event_payload(tmp_path: Path) -> None:
    """Issue #448: failed rework dispatch must record the per-issue reason in the event payload."""
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
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

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is False
    assert result.data["failed_count"] == 1
    assert result.message.startswith("rework dispatch failures:")
    assert "#123" in result.message

    state = load_state(paths.state_file)
    rework_events = [e for e in state["events"] if e["kind"] == "dispatch_rework"]
    assert rework_events, "dispatch_rework event must be emitted"
    payload = rework_events[-1]["payload"]
    assert payload["failed_issue_numbers"] == [123]
    assert "123" in payload["failures"]
    assert "command exited 1" in payload["failures"]["123"]
    assert result.data["failures"][123] == payload["failures"]["123"]


def test_dispatch_rework_event_indexes_pr_number(tmp_path: Path) -> None:
    """Issue #770: dispatch_rework events must populate the indexed pr_number column.

    The payload must carry an explicit ``pr_number`` (mirroring ``rework_already_pushed``)
    so the SQLite-backed event log indexes it; without it, ``query_events(pr_number=...)``
    silently returns empty for rework dispatches.
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

    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 1

    events = query_events(paths.state_file, kind="dispatch_rework")
    assert len(events) == 1
    assert events[0]["pr_number"] == 456
    assert events[0]["payload"]["pr_number"] == 456


def test_dispatch_rework_clears_startup_death_flag_on_new_dispatch(
    tmp_path: Path,
) -> None:
    """Issue #1106: ``_dispatch_rework_impl``'s new-dispatch-supersedes-clearing
    block must clear ``last_rework_was_startup_death`` /
    ``last_rework_failure_kind`` when a fresh rework session is successfully
    dispatched after a prior startup-death flag was set.

    The stale flag belongs to the *dead* session; the new session is the one
    whose outcome the next janitor pass will attribute.  If the flag survives
    the dispatch, a subsequent janitor pass would misattribute the new
    session's outcome to the old death and skip the cap counter.

    Mutation gate: removing the clearing block at the ``if ok:`` branch in
    ``_dispatch_rework_impl`` (the ``last_rework_failure_kind`` /
    ``last_rework_was_startup_death`` reset on the PR state) makes this test
    fail (the flags stay True/``"launch_failed"`` instead of being cleared).
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
    paths.root.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    # Seed state: issue in rework_requested, PR carrying a stale startup-death
    # flag from a prior dead session.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        state["prs"]["456"] = {
            **state.get("prs", {}).get("456", {}),
            "number": 456,
            "issue_number": 123,
            "last_rework_failure_kind": "launch_failed",
            "last_rework_was_startup_death": True,
        }
        save_state(paths.state_file, state)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a rework prompt so dispatch can proceed.
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()
    assert result.ok is True
    assert result.data["selected_count"] == 1

    state = load_state(paths.state_file)
    pr_state = state["prs"]["456"]
    # The new dispatch must have cleared the stale startup-death flags.
    assert pr_state["last_rework_was_startup_death"] is False
    assert pr_state["last_rework_failure_kind"] is None
