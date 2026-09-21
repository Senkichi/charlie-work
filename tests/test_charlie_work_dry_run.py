"""Dry-run mode: worker launch, dispatch, dependency gate, prompt files, and intake all stay side-effect free.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

from pathlib import Path
from _dispatch_fixtures import (
    _cross_repo_issue_body,
    _write_fleet_registry,
    _write_sibling_repo_file,
)
from _fakes_github import FakeGitHub
from charlie_work.config import (
    ClaudeCodeConfig,
    DevinConfig,
    DispatchConfig,
    LabelConfig,
    OrchestratorConfig,
    RuntimeConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_dry_run_skips_worker_launch(monkeypatch, tmp_path: Path) -> None:
    """Test that --dry-run prevents worker process launch and worktree creation."""
    from charlie_work.adapters import AdapterSettings, SessionRequest, dispatch_sessions

    subprocess_calls: list[list[str]] = []

    def fake_subprocess(*args, **kwargs):
        subprocess_calls.append(args[0])
        raise AssertionError("subprocess should not be called in dry-run mode")

    monkeypatch.setattr("charlie_work.claude_code.subprocess.Popen", fake_subprocess)
    monkeypatch.setattr("charlie_work.devin_shell.subprocess.Popen", fake_subprocess)
    monkeypatch.setattr("charlie_work.subprocess_runner.subprocess.run", fake_subprocess)

    manifest_path = tmp_path / "manifest.json"
    results_path = tmp_path / "results.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("test prompt", encoding="utf-8")
    settings = AdapterSettings(adapter="claude-code", dry_run=True)
    request = SessionRequest(
        issue_number=1,
        issue_title="Test issue",
        prompt_path=prompt_path,
        branch_name="agent/issue-1-test",
    )

    results = dispatch_sessions(tmp_path, manifest_path, results_path, settings, [request])

    assert len(results) == 1
    assert results[0].ok is True
    assert (
        results[0].error is None
    )  # error=None for dry-run (informational note is in workflow layer)
    assert len(subprocess_calls) == 0  # No subprocess should be invoked


def test_dry_run_dispatch_leaves_state_unchanged(tmp_path: Path) -> None:
    """Test that --dry-run dispatch does not modify state.json or labels."""
    # Setup: create a minimal state file
    config = OrchestratorConfig(
        labels=LabelConfig(),
        dispatch=DispatchConfig(),
        devin=DevinConfig(),
        claude_code=ClaudeCodeConfig(),
        runtime=RuntimeConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    initial_state = {
        "issues": {},
        "prs": {},
        "events": [],
        "generated_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, initial_state)

    # Use FakeGitHub which returns a ready issue by default
    fake_gh = FakeGitHub()
    app = OrchestratorApp(
        repo_root=tmp_path,
        paths=paths,
        config=config,
        gh=fake_gh,
        dry_run=True,
    )

    # Run dry-run dispatch
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch()

    # Verify the result indicates dry-run
    assert result.ok is True
    assert "dry-run" in result.message.lower()
    assert result.data["selected_count"] == 1

    # Verify state.json is unchanged (load_state adds metadata, so check key fields)
    with state_lock(paths.state_file):
        final_state = load_state(paths.state_file)

    assert final_state["issues"] == {}, "No issues should be marked as dispatched in state"
    assert final_state["prs"] == {}, "No PRs should be recorded"
    assert final_state["events"] == [], "No dispatch events should be recorded"


def test_dry_run_dispatch_dependency_gate_filter(tmp_path: Path) -> None:
    """Issue #127: dry-run dispatch dependency-gate filter must exclude blocked issues.

    When a blocked issue is ordered ahead of an eligible candidate with
    dispatch_limit=1, the dry-run report should list the eligible issue as
    dispatchable and the blocked issue should be excluded from sessions.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with blocked issue first, then eligible issue
    class FakeGitHubWithDryRunDependencyGate(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Override with test issues: blocked first, then eligible
            self.issues = [
                {
                    "number": 100,
                    "title": "Blocked issue (first in order)",
                    "url": "https://example.test/issues/100",
                    "body": "Blocked by #200",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
                {
                    "number": 101,
                    "title": "Eligible issue (second in order)",
                    "url": "https://example.test/issues/101",
                    "body": "No blockers",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
                {
                    "number": 200,
                    "title": "Blocker issue",
                    "url": "https://example.test/issues/200",
                    "body": "Foundation work",
                    "labels": [],
                    "state": "OPEN",  # Still open, blocks #100
                },
            ]

        def issue_list(self, labels=None, state=None):
            # Support both old and new signature
            if isinstance(labels, str):
                ready_label = labels
                return [
                    issue
                    for issue in self.issues
                    if ready_label in [label["name"] for label in issue.get("labels", [])]
                ]
            elif labels:
                return [
                    issue
                    for issue in self.issues
                    if any(
                        label in [label_obj["name"] for label_obj in issue.get("labels", [])]
                        for label in labels
                    )
                ]
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return {200}

    fake_gh = FakeGitHubWithDryRunDependencyGate()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    # Only the eligible issue should be selected (blocked issue doesn't consume slot)
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["attempted_count"] == 1

    # Verify the selected issue is exactly 101 (the eligible one), not 100 (blocked)
    assert len(result.data["sessions"]) == 1
    assert result.data["sessions"][0]["issue_number"] == 101

    # Verify issue 100 is absent from sessions
    dispatched_issue_numbers = {session["issue_number"] for session in result.data["sessions"]}
    assert 100 not in dispatched_issue_numbers

    # Verify the blocked section contains issue 100 with its declared blockers
    assert "blocked" in result.data
    blocked_entries = {entry["issue"]: entry["blockers"] for entry in result.data["blocked"]}
    assert 100 in blocked_entries
    assert blocked_entries[100] == [200]


def test_dry_run_dispatch_does_not_write_worker_prompt(tmp_path: Path) -> None:
    """Issue #618-A: dry-run dispatch must not write worker-prompt.md or create
    issue directories. The dry-run block promises "skip all state writes, label
    transitions, and file mutations" — ``_write_worker_prompt`` used to
    ``mkdir`` + ``write_text`` unconditionally inside it.
    """
    config = OrchestratorConfig(
        labels=LabelConfig(),
        dispatch=DispatchConfig(),
        devin=DevinConfig(),
        claude_code=ClaudeCodeConfig(),
        runtime=RuntimeConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    initial_state = {
        "issues": {},
        "prs": {},
        "events": [],
        "generated_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, initial_state)

    fake_gh = FakeGitHub()
    app = OrchestratorApp(
        repo_root=tmp_path,
        paths=paths,
        config=config,
        gh=fake_gh,
        dry_run=True,
    )

    # Close the default PR so the issue is dispatchable
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch()

    assert result.ok is True
    assert result.data["selected_count"] == 1

    # The issue directory and worker-prompt.md must NOT exist after dry-run
    issue_dir = paths.issues / "issue-123"
    assert not issue_dir.exists(), "dry-run dispatch must not create issue directories"
    assert not (issue_dir / "worker-prompt.md").exists(), (
        "dry-run dispatch must not write worker-prompt.md"
    )


def test_dry_run_dispatch_preserves_existing_worker_prompt(tmp_path: Path) -> None:
    """Issue #618-A: for a dead-worker recovery candidate (previous status
    ``dispatched``, same branch), dry-run dispatch must not overwrite the
    prompt a crashed worker was launched with — that is the forensic record
    the preview was meant to inspect.
    """
    config = OrchestratorConfig(
        labels=LabelConfig(),
        dispatch=DispatchConfig(),
        devin=DevinConfig(),
        claude_code=ClaudeCodeConfig(),
        runtime=RuntimeConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Seed state with a "dispatched" issue on the same branch the dry-run
    # would use — this makes the candidate a dead-worker recovery target.
    branch_name = f"{config.dispatch.branch_prefix}-123-fix-search"
    initial_state = {
        "issues": {
            "123": {
                "number": 123,
                "status": "dispatched",
                "branch_name": branch_name,
            }
        },
        "prs": {},
        "events": [],
        "generated_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, initial_state)

    # Plant the forensic prompt from the crashed worker
    issue_dir = paths.issues / "issue-123"
    issue_dir.mkdir(parents=True, exist_ok=True)
    forensic_prompt = issue_dir / "worker-prompt.md"
    original_content = "# ORIGINAL CRASHED WORKER PROMPT\nDo not overwrite me."
    forensic_prompt.write_text(original_content, encoding="utf-8")

    fake_gh = FakeGitHub()
    app = OrchestratorApp(
        repo_root=tmp_path,
        paths=paths,
        config=config,
        gh=fake_gh,
        dry_run=True,
    )

    # Close the default PR so the issue is dispatchable (dead worker, no open PR)
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch()

    assert result.ok is True
    assert result.data["selected_count"] == 1
    # The recovery flag should be set for this candidate
    assert result.data["sessions"][0]["recovery"] is not None

    # The forensic prompt must be untouched
    assert forensic_prompt.read_text(encoding="utf-8") == original_content


def test_dry_run_intake_does_not_write_files_or_state(tmp_path: Path) -> None:
    """Issue #618-C: ``intake()`` in dry-run must not create issue dirs, write
    issue.json/worker-prompt.md, add labels, or merge state.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    initial_state = {
        "issues": {},
        "prs": {},
        "events": [],
        "generated_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, initial_state)

    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    result = app.intake()

    assert result.ok is True
    assert "dry-run" in result.message.lower()
    assert len(result.data["issues"]) == 1
    assert result.data["issues"][0]["issue"] == 123

    # No issue directory, issue.json, or worker-prompt.md
    issue_dir = paths.issues / "issue-123"
    assert not issue_dir.exists()
    assert not (issue_dir / "issue.json").exists()
    assert not (issue_dir / "worker-prompt.md").exists()

    # No labels added
    assert fake_gh.labels_added == []

    # State unchanged
    with state_lock(paths.state_file):
        final_state = load_state(paths.state_file)
    assert final_state["issues"] == {}
    assert final_state["events"] == []


def test_dry_run_dispatch_cross_repo_gate_reports_without_mutating(tmp_path: Path) -> None:
    """Issue #1010 wiring (dry-run): ``dispatch`` with ``dry_run=True`` reports
    which issues the cross-repo gate would escalate, without mutating state,
    labels, or events.

    Drives the dry-run branch of ``_dispatch_impl`` (the path that populates
    ``cross_repo_escalated_issue_numbers`` in the planning payload), not
    ``cross_repo_gate`` in isolation.

    Issues #1756-#1758 (positive-evidence redesign): also the call-site
    threading regression test for the dry-run branch's
    ``managed_repo_roots``/``dispatching_repo_name`` -- without a registered
    sibling repo that actually contains the missing path, the gate abstains
    instead of escalating and ``selected_count`` would go non-zero.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    fake_gh.issues[0]["body"] = _cross_repo_issue_body()
    sibling_root = tmp_path / "sibling-ci-runners"
    _write_sibling_repo_file(sibling_root, "src/ci_fleet/suite_coverage.py")
    _write_fleet_registry(
        tmp_path / "fleet",
        {"owner/ci-runners": {"repo_root": str(sibling_root)}},
    )
    app = OrchestratorApp(
        repo_root=tmp_path,
        paths=paths,
        config=config,
        gh=fake_gh,
        dry_run=True,
    )

    result = app.dispatch()

    # The issue is reported as cross-repo escalated and excluded from sessions.
    assert result.ok is True
    assert 123 in result.data["cross_repo_escalated_issue_numbers"]
    assert result.data["selected_count"] == 0
    session_issue_numbers = {session["issue_number"] for session in result.data["sessions"]}
    assert 123 not in session_issue_numbers

    # Dry-run must not mutate labels, state, or events.
    assert (123, "agent:human-needed") not in fake_gh.labels_added
    state = load_state(paths.state_file)
    assert state["issues"] == {}
    assert state["events"] == []
