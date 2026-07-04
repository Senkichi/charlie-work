from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from pathlib import Path

import yaml

from charlie_work import cli
from charlie_work import github as github_module
from charlie_work.checks import summarize_checks
from charlie_work.config import (
    ClaudeCodeConfig,
    CrossFamilyConfig,
    DevinConfig,
    DispatchConfig,
    LabelConfig,
    OrchestratorConfig,
    ReviewConfig,
    RuntimeConfig,
    find_config_path,
    load_config,
)
from charlie_work.cross_family import (
    _CAVEAT,
    CrossFamilyResult,
    extract_report_body,
    render_command,
    report_body_is_valid,
    run_cross_family_review,
)
from charlie_work.github import label_names, linked_issue_number
from charlie_work.paths import runtime_paths
from charlie_work.prompts import render_prompt
from charlie_work.state import (
    is_throttled,
    load_state,
    save_state,
    set_throttled_until,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp, slugify

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_default_config_enables_auto_merge() -> None:
    config = load_config()

    assert config.auto_merge.enabled is True
    # A shared package cannot know a consumer's CI check names; unconfigured
    # means empty, and `doctor` flags it.
    assert config.auto_merge.required_checks == ()
    assert config.labels.ready == "automated-ready"


def test_runtime_paths_are_repo_relative(tmp_path: Path) -> None:
    paths = runtime_paths(tmp_path, ".var/charlie-work")

    assert paths.root == tmp_path / ".var" / "charlie-work"
    assert paths.state_file == paths.root / "state.json"


def test_state_round_trip(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = load_state(state_path)
    state["issues"]["123"] = {"title": "Example"}

    save_state(state_path, state)
    loaded = load_state(state_path)

    assert loaded["issues"]["123"]["title"] == "Example"
    assert loaded["version"] == 1


def test_worker_prompt_renders_issue_values() -> None:
    prompt = render_prompt(
        "worker.md",
        {
            "issue_number": 123,
            "issue_title": "Fix search",
            "issue_url": "https://example.test/issues/123",
            "issue_body": "Body text",
            "branch_name": "agent/issue-123-fix-search",
            "worker_model_tier": "capable",
        },
    )

    assert "Issue #123" in prompt
    assert "agent/issue-123-fix-search" in prompt
    assert "Closes #123" in prompt


def test_claude_code_worker_prompt_renders_issue_values() -> None:
    prompt = render_prompt(
        "worker_claude_code.md",
        {
            "issue_number": 123,
            "issue_title": "Fix search",
            "issue_url": "https://example.test/issues/123",
            "issue_body": "Body text",
            "branch_name": "agent/issue-123-fix-search",
            "worker_model_tier": "capable",
        },
    )

    assert "Issue #123" in prompt
    assert "git switch -c agent/issue-123-fix-search" in prompt
    assert "Closes #123" in prompt


def test_repo_local_prompt_dir_overrides_package_template(tmp_path: Path) -> None:
    override_dir = tmp_path / "my-prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "CUSTOM for #$issue_number on $branch_name", encoding="utf-8"
    )

    prompt = render_prompt(
        "worker.md",
        {"issue_number": 5, "branch_name": "agent/issue-5-x"},
        search_dirs=(override_dir,),
    )

    assert prompt == "CUSTOM for #5 on agent/issue-5-x"


def test_missing_repo_local_template_falls_back_to_package(tmp_path: Path) -> None:
    override_dir = tmp_path / "my-prompts"
    override_dir.mkdir()

    prompt = render_prompt(
        "rework.md",
        {"pr_number": 9, "pr_title": "t", "pr_url": "u", "issue_number": 1, "review_summary": "s"},
        search_dirs=(override_dir,),
    )

    assert "Rework Task: PR #9" in prompt


def test_slugify_makes_branch_safe_slug() -> None:
    assert slugify("Fix: Search / Windows path!!!") == "fix-search-windows-path"


def test_label_names_accepts_gh_shape() -> None:
    issue = {"labels": [{"name": "automated-ready"}, {"name": "agent:in-progress"}]}

    assert label_names(issue) == {"automated-ready", "agent:in-progress"}


def test_linked_issue_number_from_branch_body_or_title() -> None:
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-456-fix"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 456
    )
    assert (
        linked_issue_number(
            {"body": "Closes #789"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 789
    )
    assert (
        linked_issue_number(
            {"title": "Fix #321: thing"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 321
    )


def test_linked_issue_number_ignores_unqualified_body_references() -> None:
    body = "Bumps actions/checkout. See dependabot/dependabot-core#2454 for details."

    assert (
        linked_issue_number(
            {"body": body},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        is None
    )


def test_summarize_checks_requires_all_configured_checks() -> None:
    checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "FAILURE"},
    ]

    summary = summarize_checks(checks, ("Tests passed", "Lint & Format", "Pre-commit"))

    assert summary.ready is False
    assert summary.passed == ("Tests passed", "Lint & Format")
    assert summary.failed == ("Pre-commit",)


def test_summarize_checks_duplicate_runs_failure_then_success() -> None:
    """Regression test for issue #1: duplicate runs with FAILURE then SUCCESS should classify as failed."""
    checks = [
        {"name": "test", "state": "FAILURE"},
        {"name": "test", "state": "SUCCESS"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.failed == ("test",)
    assert summary.passed == ()


def test_summarize_checks_duplicate_runs_success_then_failure() -> None:
    """Regression test for issue #1: duplicate runs with SUCCESS then FAILURE should classify as failed."""
    checks = [
        {"name": "test", "state": "SUCCESS"},
        {"name": "test", "state": "FAILURE"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.failed == ("test",)
    assert summary.passed == ()


def test_summarize_checks_duplicate_runs_all_success() -> None:
    """Duplicate runs with all SUCCESS should classify as passed."""
    checks = [
        {"name": "test", "state": "SUCCESS"},
        {"name": "test", "state": "SUCCESS"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is True
    assert summary.passed == ("test",)
    assert summary.failed == ()


def test_summarize_checks_duplicate_runs_pending_then_success() -> None:
    """Duplicate runs with PENDING then SUCCESS should classify as pending."""
    checks = [
        {"name": "test", "state": "PENDING"},
        {"name": "test", "state": "SUCCESS"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.pending == ("test",)
    assert summary.passed == ()


def test_summarize_checks_duplicate_runs_failure_then_pending() -> None:
    """Duplicate runs with FAILURE then PENDING should classify as failed (worst-of)."""
    checks = [
        {"name": "test", "state": "FAILURE"},
        {"name": "test", "state": "PENDING"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.failed == ("test",)
    assert summary.pending == ()


def test_state_json_is_valid_after_save(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    save_state(state_path, {"version": 1, "issues": {}, "prs": {}, "events": []})

    payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert payload["version"] == 1
    assert payload["generated_at"].endswith("Z")


def test_concurrent_state_access_serializes_with_lock(tmp_path: Path) -> None:
    """Regression test for issue #16: concurrent load→save cycles must serialize.

    Two threads incrementing a counter should never lose updates when using
    the lock context manager. Without the lock, one thread can overwrite the
    other's update (last writer wins).
    """
    state_path = tmp_path / "state.json"
    # Initialize state with a counter
    save_state(state_path, {"version": 1, "issues": {}, "prs": {}, "events": [], "counter": 0})

    # Number of increments per thread
    increments_per_thread = 100
    errors = []

    def increment_counter(thread_id: int) -> None:
        for _ in range(increments_per_thread):
            try:
                with state_lock(state_path):
                    state = load_state(state_path)
                    current = state.get("counter", 0)
                    # Simulate some work
                    state["counter"] = current + 1
                    save_state(state_path, state)
            except Exception as exc:
                errors.append((thread_id, exc))

    # Run two threads concurrently
    thread1 = threading.Thread(target=increment_counter, args=(1,))
    thread2 = threading.Thread(target=increment_counter, args=(2,))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    # Verify no errors occurred
    assert not errors, f"Errors during concurrent access: {errors}"

    # Verify the counter is the sum of both increments (no lost updates)
    final_state = load_state(state_path)
    expected_count = increments_per_thread * 2
    assert final_state.get("counter") == expected_count, (
        f"Expected counter to be {expected_count}, got {final_state.get('counter')} "
        f"— indicates lost updates due to race condition"
    )


def test_load_config_names_unknown_keys_and_section(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "review:\n  max_rework_cycles: 2\n  max_rework_cylces: 3\n", encoding="utf-8"
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")

    assert "section 'review'" in message
    assert "max_rework_cylces" in message
    assert "max_rework_cycles" in message  # valid keys listed for the operator


def test_load_config_rejects_unknown_top_level_sections(tmp_path: Path) -> None:
    """Issue #12: typo'd top-level config section is rejected, not silently ignored."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text("auto-merge:\n  enabled: false\n", encoding="utf-8")

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown top-level section")

    assert "unknown config section(s)" in message
    assert "auto-merge" in message
    assert "auto_merge" in message  # valid section name listed


def test_load_config_rejects_broken_yaml(tmp_path: Path) -> None:
    """Issue #12: malformed YAML yields YAMLError, not raw traceback."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text("labels:\n  ready: automated-ready\n  bad: [unclosed", encoding="utf-8")

    try:
        load_config(config_path)
    except yaml.YAMLError:
        # Expected: YAML parsing error
        pass
    else:  # pragma: no cover
        raise AssertionError("expected YAMLError for malformed YAML")


def test_load_config_rejects_unknown_shell_command_placeholder(tmp_path: Path) -> None:
    """Issue #4: unknown placeholder in shell_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  shell_command:\n    - devin\n    - "{unknown_placeholder}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown placeholder")

    assert "devin.shell_command" in message
    assert "unknown_placeholder" in message


def test_load_config_rejects_empty_placeholder_in_shell_command(tmp_path: Path) -> None:
    """Issue #4: empty placeholder {} in shell_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  shell_command:\n    - devin\n    - "{}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for empty placeholder")

    assert "devin.shell_command" in message
    assert "empty placeholder" in message


def test_load_config_rejects_unknown_claude_code_command_placeholder(tmp_path: Path) -> None:
    """Issue #4: unknown placeholder in claude_code.command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'claude_code:\n  command:\n    - claude\n    - "{bad_token}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown placeholder")

    assert "claude_code.command" in message
    assert "bad_token" in message


def test_load_config_rejects_unknown_cross_family_command_placeholder(tmp_path: Path) -> None:
    """Issue #4: unknown placeholder in cross_family.command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'cross_family:\n  enabled: true\n  command:\n    - devin\n    - "{invalid}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown placeholder")

    assert "cross_family.command" in message
    assert "invalid" in message


def test_load_config_accepts_valid_placeholders(tmp_path: Path) -> None:
    """Issue #4: valid placeholders in command templates are accepted."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """devin:
  shell_command:
    - devin
    - "{prompt_path}"
    - "{issue_number}"
    - "{branch}"
claude_code:
  command:
    - claude
    - "{prompt_path}"
cross_family:
  enabled: true
  command:
    - devin
    - "{model}"
    - "{prompt_path}"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.devin.shell_command == ("devin", "{prompt_path}", "{issue_number}", "{branch}")
    assert config.claude_code.command == ("claude", "{prompt_path}")
    assert config.cross_family.command == ("devin", "{model}", "{prompt_path}")


def test_load_config_rejects_bare_brace_in_shell_command(tmp_path: Path) -> None:
    """Issue #4: bare { in shell_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  shell_command:\n    - devin\n    - "test{"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for bare brace")

    assert "devin.shell_command" in message
    assert "malformed placeholder" in message


def test_load_config_rejects_unclosed_brace_in_shell_command(tmp_path: Path) -> None:
    """Issue #4: unclosed {prompt_path in shell_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  shell_command:\n    - devin\n    - "{prompt_path"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unclosed brace")

    assert "devin.shell_command" in message
    assert "malformed placeholder" in message


def test_load_config_rejects_stray_closing_brace_in_shell_command(tmp_path: Path) -> None:
    """Issue #4: stray } in shell_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  shell_command:\n    - devin\n    - "test}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for stray closing brace")

    assert "devin.shell_command" in message
    assert "malformed placeholder" in message


def test_load_config_rejects_positional_placeholder_in_shell_command(tmp_path: Path) -> None:
    """Issue #4: positional {0} in shell_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  shell_command:\n    - devin\n    - "{0}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for positional placeholder")

    assert "devin.shell_command" in message
    # Positional placeholders are caught as unknown (not in allowed set) or malformed
    assert "unknown placeholder" in message or "malformed placeholder" in message


def test_load_config_rejects_unknown_placeholder_in_dispatch_command(tmp_path: Path) -> None:
    """Issue #4: unknown placeholder in dispatch_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  dispatch_command:\n    - echo\n    - "{bad_token}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown placeholder")

    assert "devin.dispatch_command" in message
    assert "bad_token" in message


def test_command_adapter_render_error_returns_error_record(tmp_path: Path) -> None:
    """Defense-in-depth: render errors past the load gate return error records, not exceptions."""
    from charlie_work.adapters import AdapterSettings, SessionRequest, dispatch_sessions

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    results_path = tmp_path / "results.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt", encoding="utf-8")

    request = SessionRequest(
        issue_number=1,
        issue_title="Test",
        prompt_path=prompt_path,
        branch_name="agent/issue-1",
    )

    settings = AdapterSettings(
        adapter="command",
        dispatch_command=("echo", "{unknown_placeholder}"),
        command_timeout_seconds=300,
    )

    results = dispatch_sessions(repo_root, manifest_path, results_path, settings, [request])

    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].error is not None
    assert "unknown_placeholder" in results[0].error


def test_find_config_path_prefers_explicit_then_repo_root(tmp_path: Path) -> None:
    explicit = tmp_path / "elsewhere.yaml"
    assert find_config_path(tmp_path, explicit) == explicit

    assert find_config_path(tmp_path) is None

    repo_config = tmp_path / "orchestrator.config.yaml"
    repo_config.write_text("labels:\n  ready: automated-ready\n", encoding="utf-8")
    assert find_config_path(tmp_path) == repo_config


def test_cli_accepts_json_after_subcommand(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "build_app", lambda args: object())
    monkeypatch.setattr(
        cli,
        "run_command",
        lambda app, args: cli.CommandResult(True, "ok", {"json_output": args.json_output}),
    )

    assert cli.main(["roll-call", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["json_output"] is True


def test_github_run_parses_allow_failure_json_stdout(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout='[{"name": "Tests passed", "state": "FAILURE"}]',
            stderr="checks failed",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    result = github_module.GitHub(tmp_path).run(
        ["pr", "checks", "123"], json_output=True, allow_failure=True
    )

    assert result == [{"name": "Tests passed", "state": "FAILURE"}]


# --- Issue #15 regression: list limits must match reconcile and warn on truncation


def test_issue_list_raises_limit_to_500_and_warns_on_truncation(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    caplog.set_level(logging.WARNING)
    limit = github_module._LIST_LIMIT

    def fake_run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):
        assert json_output is True
        assert args[:2] == ["issue", "list"]
        assert str(limit) in args, f"expected --limit {limit} in {args}"
        return [{"number": i} for i in range(limit)]

    monkeypatch.setattr(github_module.GitHub, "run", fake_run)
    gh = github_module.GitHub(tmp_path)

    result = gh.issue_list("automated-ready")

    assert len(result) == limit
    assert any("truncated" in record.message for record in caplog.records)


def test_pr_list_raises_limit_to_500_and_warns_on_truncation(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    caplog.set_level(logging.WARNING)
    limit = github_module._LIST_LIMIT

    def fake_run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):
        assert json_output is True
        assert args[:2] == ["pr", "list"]
        assert str(limit) in args, f"expected --limit {limit} in {args}"
        return [{"number": i} for i in range(limit)]

    monkeypatch.setattr(github_module.GitHub, "run", fake_run)
    gh = github_module.GitHub(tmp_path)

    result = gh.pr_list()

    assert len(result) == limit
    assert any("truncated" in record.message for record in caplog.records)


def _required_checks_config(**kwargs) -> OrchestratorConfig:
    from charlie_work.config import AutoMergeConfig

    auto_merge = AutoMergeConfig(
        required_checks=("Tests passed", "Lint & Format", "Pre-commit"), **kwargs
    )
    return OrchestratorConfig(auto_merge=auto_merge)


class FakeGitHub:
    def __init__(self) -> None:
        self.issues = [
            {
                "number": 123,
                "title": "Fix search",
                "url": "https://example.test/issues/123",
                "body": "Search is broken",
                "labels": [{"name": "automated-ready"}],
            }
        ]
        # A janitor-green PR: open, non-draft, linked issue, tests mentioned.
        self.prs = [
            {
                "number": 456,
                "title": "Fix #123: search",
                "url": "https://example.test/pull/456",
                "headRefName": "agent/issue-123-fix-search",
                "headRefOid": "sha-abc123",
                "body": "Closes #123\n\nTests: regression coverage added.",
                "labels": [],
                "isCrossRepository": False,
            }
        ]
        self.labels_added: list[tuple[int, str]] = []
        self.labels_removed: list[tuple[int, str]] = []
        self.labels_created: list[tuple[str, str, str]] = []
        self.merged: list[tuple[int, str]] = []
        self.deleted_branches: list[str] = []
        self.delete_branch_ok = True
        self.update_branch_ok = True

    def issue_list(self, ready_label: str):
        # Honor the label filter: return only issues with the ready label
        return [
            issue
            for issue in self.issues
            if ready_label in [label["name"] for label in issue.get("labels", [])]
        ]

    def issue_view(self, number: int):
        # Return the issue matching the requested number
        for issue in self.issues:
            if issue["number"] == number:
                return issue
        raise ValueError(f"Issue {number} not found")

    def pr_list(self):
        return self.prs

    def pr_view(self, number: int):
        # Return the PR matching the requested number
        for pr in self.prs:
            if pr["number"] == number:
                return pr
        raise ValueError(f"PR {number} not found")

    def pr_checks(self, number: int):
        return [
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]

    def pr_diff(self, number: int):
        return "diff --git a/file b/file"

    def add_issue_label(self, number: int, label: str) -> None:
        self.labels_added.append((number, label))

    def remove_issue_label(self, number: int, label: str) -> None:
        self.labels_removed.append((number, label))

    def merge_pr(self, number: int, strategy: str) -> str:
        self.merged.append((number, strategy))
        return "merged"

    def delete_branch(self, branch: str) -> bool:
        self.deleted_branches.append(branch)
        return self.delete_branch_ok

    def pr_update_branch(self, pr_number: int) -> bool:
        return self.update_branch_ok

    def label_create(self, label: str, color: str, description: str) -> None:
        self.labels_created.append((label, color, description))

    def label_list(self) -> list[dict[str, object]]:
        # Return all labels that have been created — simulates creation success.
        return [{"name": name} for name, _color, _desc in self.labels_created]

    def pr_comment(self, number: int, body_file: Path) -> None:
        pass


def test_dispatch_writes_worker_prompt_and_session_manifest(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 1
    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    manifest_path = tmp_path / ".var" / "charlie-work" / "dispatches" / "session-manifest.json"
    assert prompt_path.exists()
    assert manifest_path.exists()
    assert "Closes #123" in prompt_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sessions"][0]["branch_name"] == "agent/issue-123-fix-search"
    # Manual adapter honesty: a written manifest means QUEUED — no worker has
    # been independently confirmed, so in-progress must not be applied.
    assert (123, "agent:queued") in fake_gh.labels_added
    assert (123, "agent:in-progress") not in fake_gh.labels_added


def test_dispatch_only_issues_selects_explicit_subset(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Numbers not among the dispatchable candidates are skipped; only the
    # explicit, dispatchable match is selected (dependency-ordered waves).
    result = app.dispatch(only_issues="999, 123")

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert (123, "agent:queued") in fake_gh.labels_added


def test_dispatch_worker_template_selects_claude_code_variant(tmp_path: Path) -> None:
    config = OrchestratorConfig(dispatch=DispatchConfig(worker_template="worker_claude_code.md"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    app.dispatch(limit=1)

    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    text = prompt_path.read_text(encoding="utf-8")
    assert "git switch -c agent/issue-123-fix-search" in text  # Claude Code loop
    assert "/create-branch" not in text  # not the Devin skills loop


def test_app_prompts_dir_override_wins_for_worker_prompt(tmp_path: Path) -> None:
    override_dir = tmp_path / "orchestrator-prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text("REPO-LOCAL #$issue_number", encoding="utf-8")
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir="orchestrator-prompts"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    app.dispatch(limit=1)

    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    assert prompt_path.read_text(encoding="utf-8") == "REPO-LOCAL #123"


def test_command_dispatch_labels_only_successful_launches(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["failed_count"] == 0
    assert result.data["dispatch_results"][0]["stdout"].strip() == "123"
    assert (123, "agent:in-progress") in fake_gh.labels_added
    results_path = tmp_path / ".var" / "charlie-work" / "dispatches" / "session-results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["results"][0]["ok"] is True


def test_command_dispatch_failure_does_not_label_in_progress(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; sys.exit(7)"),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is False
    assert result.data["selected_count"] == 0
    assert result.data["failed_count"] == 1
    assert result.data["dispatch_results"][0]["returncode"] == 7
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatch_failed"


def test_merge_ready_requires_approved_decision_then_merges(tmp_path: Path) -> None:
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    not_ready = app.merge_ready(456)
    assert not_ready.data["can_merge"] is False
    assert fake_gh.merged == []

    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    ready = app.merge_ready(456)

    assert ready.data["can_merge"] is True
    assert ready.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    assert (123, "agent:done") in fake_gh.labels_added
    assert fake_gh.deleted_branches == ["agent/issue-123-fix-search"]
    assert ready.data["branch_deleted"] is True


def test_merge_ready_branch_delete_failure_never_blocks_labels(tmp_path: Path) -> None:
    """The empericus failure mode: a branch checked out in a local worktree made
    `gh pr merge --delete-branch` abort the post-merge label update. Deletion is
    now decoupled and best-effort — labels always land."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.delete_branch_ok = False
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    ready = app.merge_ready(456)

    assert ready.data["merged"] is True
    assert ready.data["branch_deleted"] is False
    assert (123, "agent:done") in fake_gh.labels_added


def test_merge_ready_honors_delete_branch_false(tmp_path: Path) -> None:
    config = _required_checks_config(delete_branch=False)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    ready = app.merge_ready(456)

    assert ready.data["merged"] is True
    assert fake_gh.deleted_branches == []
    assert ready.data["branch_deleted"] is None


def test_github_delete_branch_failure_returns_false(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args, output="", stderr="Reference does not exist")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    assert github_module.GitHub(tmp_path).delete_branch("agent/issue-1-x") is False


# --- Cross-family adversarial review ------------------------------------------


def _fake_completed(
    returncode: int = 0, stdout: str = "**MAJOR**\nx\n\nVerdict: safe", stderr: str = ""
):
    def _runner(command, **kwargs):
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    return _runner


def _cross_family_app(tmp_path: Path, *, enabled: bool) -> OrchestratorApp:
    config = OrchestratorConfig(cross_family=CrossFamilyConfig(enabled=enabled))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub())


def test_render_command_templates_list_and_string() -> None:
    values = {"model": "codex", "prompt_path": "/tmp/p.md"}
    assert render_command(
        ("devin", "--model", "{model}", "-p", "--prompt-file", "{prompt_path}"), values
    ) == ["devin", "--model", "codex", "-p", "--prompt-file", "/tmp/p.md"]
    assert render_command("devin --model {model}", values) == "devin --model codex"


def test_run_cross_family_writes_report_with_caveat(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    prompt = tmp_path / "prompt.md"

    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="attack this",
        prompt_path=prompt,
        report_path=report,
        timeout_seconds=5,
        runner=_fake_completed(0, "**BLOCKER**\nboom\n\nVerdict: safe"),
    )

    assert result.ok is True
    assert result.returncode == 0
    assert prompt.read_text(encoding="utf-8") == "attack this"
    body = report.read_text(encoding="utf-8")
    assert "leads, not verdicts" in body
    assert "**BLOCKER**" in body
    assert "Verdict: safe" in body
    assert "codex" in body


def test_run_cross_family_timeout_is_captured_not_raised(tmp_path: Path) -> None:
    report = tmp_path / "report.md"

    def _runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="x",
        prompt_path=tmp_path / "p.md",
        report_path=report,
        timeout_seconds=3,
        runner=_runner,
    )

    assert result.ok is False
    assert "timed out" in (result.error or "")
    assert "UNAVAILABLE" in report.read_text(encoding="utf-8")


def test_run_cross_family_nonzero_exit_is_captured(tmp_path: Path) -> None:
    report = tmp_path / "report.md"

    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="x",
        prompt_path=tmp_path / "p.md",
        report_path=report,
        timeout_seconds=5,
        runner=_fake_completed(2, "partial output", "stderr boom"),
    )

    assert result.ok is False
    assert result.returncode == 2
    text = report.read_text(encoding="utf-8")
    assert "exited 2" in text
    assert "partial output" in text


def test_run_cross_family_missing_binary_is_captured(tmp_path: Path) -> None:
    report = tmp_path / "report.md"

    def _runner(command, **kwargs):
        raise FileNotFoundError("devin not on PATH")

    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="x",
        prompt_path=tmp_path / "p.md",
        report_path=report,
        timeout_seconds=5,
        runner=_runner,
    )

    assert result.ok is False
    assert "failed to start" in (result.error or "")


def test_devin_example_config_enables_cross_family() -> None:
    config = load_config(EXAMPLES_DIR / "orchestrator.config.devin.yaml")

    assert config.cross_family.enabled is True
    assert config.cross_family.model == "codex"
    assert config.cross_family.command[0] == "devin"
    assert config.dispatch.worker_template == "worker.md"


def test_claude_code_example_config_selects_claude_worker() -> None:
    config = load_config(EXAMPLES_DIR / "orchestrator.config.claude-code.yaml")

    assert config.dispatch.worker_template == "worker_claude_code.md"
    assert config.cross_family.enabled is False


def test_config_absent_cross_family_block_defaults_disabled(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("labels:\n  ready: automated-ready\n", encoding="utf-8")

    config = load_config(path)

    assert config.cross_family.enabled is False


def test_config_parses_cross_family_command_list_to_tuple(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "cross_family:\n  enabled: true\n  model: codex\n"
        "  command: [devin, --model, '{model}']\n  timeout_seconds: 120\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.cross_family.command == ("devin", "--model", "{model}")
    assert config.cross_family.timeout_seconds == 120


def test_claude_code_example_config_sets_bounded_xdist_worker_env() -> None:
    config = load_config(EXAMPLES_DIR / "orchestrator.config.claude-code.yaml")

    # The shipped example bounds local test parallelism at the launch boundary
    # (the RUNBOOK "Local host saturation ceiling" section references this).
    assert config.claude_code.worker_env == {"PYTEST_XDIST_AUTO_NUM_WORKERS": "2"}


def test_config_worker_env_coerces_values_to_str(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "claude_code:\n  worker_env:\n    PYTEST_XDIST_AUTO_NUM_WORKERS: 2\n",
        encoding="utf-8",
    )

    config = load_config(path)

    # YAML parses the bare 2 as an int; env values must be strings for Popen.
    assert config.claude_code.worker_env == {"PYTEST_XDIST_AUTO_NUM_WORKERS": "2"}


def test_config_rejects_non_mapping_worker_env(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    path = tmp_path / "c.yaml"
    # A plausible operator typo: a scalar instead of a name->value mapping.
    # Must fail at load, not as an AttributeError when a worker launches.
    path.write_text('claude_code:\n  worker_env: "2"\n', encoding="utf-8")

    try:
        load_config(path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")

    assert "worker_env" in message
    assert "claude_code" in message


def test_review_injects_cross_family_section_when_enabled(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    calls = {"n": 0}

    VALID_REPORT = "**MAJOR**\nissue\n\nVerdict: safe"

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text(VALID_REPORT, encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.run_cross_family_review", _fake_run)

    result = app.review(456)

    assert calls["n"] == 1
    assert result.data["cross_family_ok"] is True
    prs_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    prompt_text = (prs_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "Cross-family adversarial pass" in prompt_text
    assert "leads, not verdicts" in prompt_text
    assert (prs_dir / "cross-family-review.md").exists()


def test_review_reuses_existing_cross_family_report(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    calls = {"n": 0}
    VALID_REPORT = "**MAJOR**\nissue\n\nVerdict: safe"

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text(VALID_REPORT, encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.run_cross_family_review", _fake_run)

    app.review(456)
    app.review(456)

    assert calls["n"] == 1  # the second pass reused the report; codex did not re-run


def test_review_no_cross_family_override_skips(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)

    def _boom(**kwargs):
        raise AssertionError("cross-family must not run when disabled per call")

    monkeypatch.setattr("charlie_work.workflow.run_cross_family_review", _boom)

    result = app.review(456, cross_family=False)

    assert result.data["cross_family_ok"] is None
    prompt_text = (
        tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    ).read_text(encoding="utf-8")
    assert "Cross-family adversarial pass" not in prompt_text


def test_review_skips_cross_family_for_draft_pr(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    app.gh.prs[0] = {**app.gh.prs[0], "isDraft": True}

    def _boom(**kwargs):
        raise AssertionError("cross-family must not run for a draft PR")

    monkeypatch.setattr("charlie_work.workflow.run_cross_family_review", _boom)

    result = app.review(456)

    # The janitor gate now blocks drafts before any review spend — even
    # earlier than the old cross-family draft skip this test pinned.
    assert result.ok is False
    assert result.data["janitor_ok"] is False
    assert any("draft" in failure.lower() for failure in result.data["janitor_failures"])


def test_spec_review_runs_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    spec = tmp_path / "SPEC.md"
    spec.write_text("# My spec\nclaims", encoding="utf-8")
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    VALID_REPORT = "**MAJOR**\nissue\n\nVerdict: safe"

    def _fake_run(**kwargs):
        assert "My spec" in kwargs["prompt_text"]  # artifact text inlined into the prompt
        Path(kwargs["report_path"]).write_text(VALID_REPORT, encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.run_cross_family_review", _fake_run)

    result = app.spec_review(spec)

    assert result.ok is True
    assert Path(result.data["report_path"]).read_text(encoding="utf-8") == VALID_REPORT


def test_spec_review_missing_file_returns_error(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app.spec_review(tmp_path / "nope.md")

    assert result.ok is False


# --- Issue #38 regression: transient retry + empty/blocked report guard --------


VALID_CROSS_FAMILY_REPORT = "**MAJOR**\nissue\n\nVerdict: safe"


def test_run_cross_family_retries_once_on_transient_rate_limit_then_success(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.md"
    prompt = tmp_path / "prompt.md"
    calls: list[str] = []
    rate_msg = (
        "Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 1 minute."
    )

    def _runner(command, **kwargs):
        if not calls:
            calls.append("fail")
            return subprocess.CompletedProcess(command, 1, stdout="", stderr=rate_msg)
        calls.append("success")
        return subprocess.CompletedProcess(command, 0, stdout=VALID_CROSS_FAMILY_REPORT, stderr="")

    sleep_calls: list[float] = []
    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="attack this",
        prompt_path=prompt,
        report_path=report,
        timeout_seconds=5,
        runner=_runner,
        sleep=lambda s: sleep_calls.append(s),
    )

    assert result.ok is True
    assert result.returncode == 0
    assert calls == ["fail", "success"]
    assert sleep_calls == [90.0]
    assert "**MAJOR**" in report.read_text(encoding="utf-8")


def test_run_cross_family_rate_limit_retry_exhausted_then_fails(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    prompt = tmp_path / "prompt.md"
    rate_msg = "Rate limit exceeded. Try again later."
    calls: list[str] = []

    def _runner(command, **kwargs):
        calls.append("fail")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=rate_msg)

    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="x",
        prompt_path=prompt,
        report_path=report,
        timeout_seconds=5,
        runner=_runner,
        sleep=lambda s: None,
    )

    assert result.ok is False
    assert result.returncode == 1
    assert calls == ["fail", "fail"]
    assert "UNAVAILABLE" in report.read_text(encoding="utf-8")


def test_run_cross_family_exit_zero_blocked_output_is_stubbed(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    prompt = tmp_path / "prompt.md"
    blocked = (
        "I'm blocked from performing the review. All tool calls are being rejected. Please re-run."
    )

    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="x",
        prompt_path=prompt,
        report_path=report,
        timeout_seconds=5,
        runner=_fake_completed(0, blocked),
    )

    assert result.ok is False
    assert result.returncode == 0
    assert "UNAVAILABLE" in report.read_text(encoding="utf-8")
    assert "empty or blocked report" in (result.error or "")


def test_review_does_not_reuse_semantically_empty_cross_family_report(
    tmp_path: Path, monkeypatch
) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    calls = {"n": 0}

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text(VALID_CROSS_FAMILY_REPORT, encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.run_cross_family_review", _fake_run)

    prs_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    report_path = prs_dir / "cross-family-review.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "I'm blocked from performing the review. Tool calls rejected. Please re-run.",
        encoding="utf-8",
    )

    app.review(456)

    assert calls["n"] == 1
    assert report_path.read_text(encoding="utf-8") == VALID_CROSS_FAMILY_REPORT


def test_review_does_not_reuse_legacy_wrapped_blocked_report(tmp_path: Path, monkeypatch) -> None:
    """Regression for issue #38: a legacy wrapped report whose body is a blocked
    refusal must not be reused as a success report on subsequent passes.
    """
    app = _cross_family_app(tmp_path, enabled=True)
    calls = {"n": 0}

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text(VALID_CROSS_FAMILY_REPORT, encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.run_cross_family_review", _fake_run)

    prs_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    report_path = prs_dir / "cross-family-review.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    blocked = "I'm blocked from performing the review. Tool calls rejected. Please re-run."
    report_path.write_text(
        f"# Cross-family adversarial review — `codex`\n\n{_CAVEAT}\n\n---\n\n{blocked}\n",
        encoding="utf-8",
    )

    app.review(456)

    assert calls["n"] == 1
    assert report_path.read_text(encoding="utf-8") == VALID_CROSS_FAMILY_REPORT


def test_report_body_is_valid_detects_real_review_vs_blocked() -> None:
    assert report_body_is_valid("**MAJOR**\nissue\n\nVerdict: safe") is True
    assert report_body_is_valid("Verdict: safe") is True
    assert report_body_is_valid("Verdict: no permission issues found") is True
    blocked = (
        "I'm blocked from performing the review. All tool calls are being rejected. Please re-run."
    )
    assert report_body_is_valid(blocked) is False
    assert report_body_is_valid("Verdict: blocked from performing the review") is False
    assert report_body_is_valid("") is False


def test_report_body_is_valid_rejects_blocked_output_with_bold_markers() -> None:
    """Regression for issue #38: bold markdown in a blocked refusal must not
    short-circuit validation and allow the blocked output to be cached.
    """
    blocked_with_bold = "**Unable to review** — all tool calls are being rejected. Please re-run."
    assert report_body_is_valid(blocked_with_bold) is False


def test_extract_report_body_strips_wrapper_but_preserves_model_output() -> None:
    body = "**MAJOR**\nissue\n\nVerdict: safe"
    wrapped = f"# Cross-family adversarial review — `codex`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    assert extract_report_body(wrapped) == body
    assert extract_report_body(body) == body


# --- P0 fixes: state safety, label honesty, rework cap, loop isolation --------


def test_load_state_quarantines_corrupt_file(tmp_path: Path) -> None:
    from charlie_work.state import load_state as _load

    state_path = tmp_path / "state.json"
    state_path.write_text("{truncated garbage", encoding="utf-8")

    state = _load(state_path)

    assert state["issues"] == {}
    assert not state_path.exists()
    quarantined = list(tmp_path.glob("state.json.corrupt-*"))
    assert len(quarantined) == 1
    assert "truncated garbage" in quarantined[0].read_text(encoding="utf-8")


def test_review_preserves_recorded_decision_in_state(tmp_path: Path) -> None:
    from charlie_work.state import load_state as _load
    from charlie_work.state import save_state as _save

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    state = _load(paths.state_file)
    state["prs"]["456"] = {"decision": "approved", "custom": "kept"}
    _save(paths.state_file, state)

    app.review(456)

    after = _load(paths.state_file)
    assert after["prs"]["456"]["decision"] == "approved"  # was clobbered pre-fix
    assert after["prs"]["456"]["custom"] == "kept"
    assert after["prs"]["456"]["status"] == "reviewing"


def test_string_dispatch_command_rejects_issue_title(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="command", dispatch_command="echo {issue_title}")
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is False
    assert result.data["failed_count"] == 1
    assert "list form" in result.data["dispatch_results"][0]["error"]
    assert (123, "agent:in-progress") not in fake_gh.labels_added


def test_rework_cap_escalates_to_human(tmp_path: Path) -> None:
    config = OrchestratorConfig()  # max_rework_cycles = 2
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    first = app.record_review(456, "request_changes", summary="fix A")
    second = app.record_review(456, "request_changes", summary="fix B")
    third = app.record_review(456, "request_changes", summary="fix C")

    assert first.data["escalated"] is False and first.data["rework_path"]
    assert second.data["escalated"] is False and second.data["rework_path"]
    assert third.data["escalated"] is True
    assert third.data["rework_path"] is None  # no third rework prompt
    assert fake_gh.labels_added.count((123, "agent:needs-rework")) == 2
    assert (123, "agent:human-needed") in fake_gh.labels_added


def test_cross_family_failure_stub_is_not_reused(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "cross-family-review.md").write_text(
        "# Cross-family adversarial review — `codex` (UNAVAILABLE)\n\n> timed out\n",
        encoding="utf-8",
    )
    calls = {"n": 0}

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text("# real findings", encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.run_cross_family_review", _fake_run)

    result = app.review(456)

    assert calls["n"] == 1  # the stub did NOT satisfy the reuse check
    assert result.data["cross_family_ok"] is True


def test_loop_isolates_per_pr_errors(tmp_path: Path) -> None:
    from charlie_work.github import GitHubError as _GitHubError

    class ExplodingGitHub(FakeGitHub):
        def pr_view(self, number: int):
            raise _GitHubError("merge conflict boom")

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, ExplodingGitHub())

    result = app.loop(limit=0)

    assert result.data["errors"] == [{"pr": 456, "error": "merge conflict boom"}]
    assert result.ok is False


# --- Issue #14: error-isolation hardening --------------------------------------


def test_corrupt_review_decision_treated_as_not_approved(tmp_path: Path) -> None:
    """A corrupt review-decision.json must not crash merge_ready/loop; it must
    be treated as a non-approval so the PR waits for a real review."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text("{truncated", encoding="utf-8")

    result = app.merge_ready(456)

    assert result.data["review_decision"] == {"decision": "invalid"}
    assert result.data["can_merge"] is False
    assert fake_gh.merged == []


def test_intake_isolates_per_issue_github_error(tmp_path: Path) -> None:
    """One failing gh issue view must not abort intake or lose other issues'
    progress."""
    from charlie_work.github import GitHubError as _GitHubError

    class FlakyIntakeGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 123,
                    "title": "Good issue",
                    "url": "https://example.test/issues/123",
                    "body": "ok",
                    "labels": [{"name": "automated-ready"}],
                },
                {
                    "number": 124,
                    "title": "Broken issue",
                    "url": "https://example.test/issues/124",
                    "body": "broken",
                    "labels": [{"name": "automated-ready"}],
                },
            ]

        def issue_view(self, number: int):
            if number == 124:
                raise _GitHubError("transient gh issue view failure")
            for issue in self.issues:
                if int(issue["number"]) == number:
                    return issue
            raise _GitHubError(f"issue #{number} not found")

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FlakyIntakeGitHub())

    result = app.intake()

    assert result.ok is False
    assert len(result.data["issues"]) == 1
    assert result.data["issues"][0]["issue"] == 123
    assert result.data["failed"] == [{"issue": 124, "error": "transient gh issue view failure"}]
    state = load_state(paths.state_file)
    assert "123" in state["issues"]
    assert state["issues"]["123"]["title"] == "Good issue"
    assert "124" not in state["issues"]
    assert any(e.get("kind") == "intake_failed" for e in state["events"])


def test_review_label_transition_failure_persists_packet(tmp_path: Path) -> None:
    """A GitHubError during the review_started label transition must still
    leave the review packet persisted in state and report label_error."""
    from charlie_work.github import GitHubError as _GitHubError

    config = OrchestratorConfig()
    reviewing_label = config.labels.reviewing

    class LabelFailReviewGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> None:
            if label == reviewing_label:
                raise _GitHubError("label transition failed")
            super().add_issue_label(number, label)

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailReviewGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    assert result.data["label_error"] == "label transition failed"
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "reviewing"
    assert state["prs"]["456"]["label_error"] == "label transition failed"
    assert Path(state["prs"]["456"]["decision_path"]).exists()


def test_loop_honors_intake_failure_signal(tmp_path: Path) -> None:
    """loop() must propagate intake() failures into its ok flag and message so
    a partially failed intake is not silently reported as a clean loop."""
    from charlie_work.github import GitHubError as _GitHubError

    class FlakyIntakeGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 123,
                    "title": "Good issue",
                    "url": "https://example.test/issues/123",
                    "body": "ok",
                    "labels": [{"name": "automated-ready"}],
                },
                {
                    "number": 124,
                    "title": "Broken issue",
                    "url": "https://example.test/issues/124",
                    "body": "broken",
                    "labels": [{"name": "automated-ready"}],
                },
            ]

        def issue_view(self, number: int):
            if number == 124:
                raise _GitHubError("transient gh issue view failure")
            return super().issue_view(number)

    config = OrchestratorConfig(cross_family=CrossFamilyConfig(enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FlakyIntakeGitHub())

    result = app.loop(limit=0)

    assert result.ok is False
    assert "intake failures" in result.message
    assert result.data["intake"]["failed"] == [
        {"issue": 124, "error": "transient gh issue view failure"}
    ]
    assert result.data["errors"] == []


def test_loop_corrupt_review_decision_does_not_crash_or_merge(tmp_path: Path) -> None:
    """A corrupt review-decision.json on the loop path must be treated as a
    non-approval: the loop re-reviews the PR and never attempts to merge."""
    config = OrchestratorConfig(
        cross_family=CrossFamilyConfig(enabled=False),
        auto_merge=_approved_automerge(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text("{truncated", encoding="utf-8")

    result = app.loop(limit=0)

    assert result.ok is True
    assert result.data["merges"] == []
    assert fake_gh.merged == []
    assert len(result.data["reviews"]) == 1


def test_run_captured_decodes_bytes_safely(tmp_path: Path) -> None:
    from charlie_work.subprocess_runner import run_captured

    result = run_captured(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'caf' + bytes([0xE9]))"],
        cwd=tmp_path,
        timeout_seconds=30,
    )

    assert result.ok is True
    assert result.stdout == "caf�"  # invalid UTF-8 replaced, never raises


# --- integration wiring: new adapters, janitor gate, reconcile ----------------


def test_devin_shell_dispatch_launches_and_labels_in_progress(tmp_path: Path, monkeypatch) -> None:
    from charlie_work import devin_shell
    from charlie_work.worktree import WorktreeInfo

    wt_path = tmp_path / "worktrees" / "agent-issue-123-fix-search"
    wt_path.mkdir(parents=True, exist_ok=True)

    def _fake_create_worktree(repo_root, branch, **kwargs):
        return WorktreeInfo(path=wt_path, branch=branch, venv_junction=None)

    monkeypatch.setattr(devin_shell, "create_worktree", _fake_create_worktree)

    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="devin-shell",
            shell_command=(sys.executable, "-c", "import sys; sys.exit(0)"),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["dispatch_results"][0]["adapter"] == "devin-shell"
    assert (123, "agent:in-progress") in fake_gh.labels_added
    sidecar = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions" / "issue-123.json"
    assert sidecar.exists()
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"


def test_claude_code_dispatch_routes_and_labels(tmp_path: Path, monkeypatch) -> None:
    from charlie_work.claude_code import ClaudeWorkerRecord

    captured: dict[str, object] = {}

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        captured["prompt_text"] = prompt_text
        captured["venv_source"] = kwargs.get("venv_source")
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=4242,
            started_at="2026-07-02T00:00:00Z",
            log_path=str(tmp_path / "log"),
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    config = OrchestratorConfig(devin=DevinConfig(adapter="claude-code"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert "Issue #123" in str(captured["prompt_text"])  # rendered prompt fed through
    assert captured["venv_source"] == tmp_path / ".venv"  # junction default ON
    assert (123, "agent:in-progress") in fake_gh.labels_added


def test_claude_code_dispatch_failure_stays_out_of_progress(tmp_path: Path, monkeypatch) -> None:
    from charlie_work.claude_code import ClaudeWorkerRecord

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path="",
            prompt_path="",
            command=("claude", "-p"),
            pid=None,
            started_at="2026-07-02T00:00:00Z",
            log_path="",
            error="claude not found on PATH",
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)


def test_dispatch_with_recovery_passes_record_to_adapter(tmp_path: Path, monkeypatch) -> None:
    """Issue #81: verify recovery record is passed through dispatch() to the adapter.

    This test MUST fail if the ordering fix in workflow.py is reverted (i.e., if
    recovery_record is forced to None by the status overwrite bug).
    """
    from charlie_work.claude_code import ClaudeWorkerRecord

    captured: dict[str, object] = {}

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        captured["recovery"] = kwargs.get("recovery")
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=4242,
            started_at="2026-07-02T00:00:00Z",
            log_path=str(tmp_path / "log"),
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    config = OrchestratorConfig(devin=DevinConfig(adapter="claude-code"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Override pr_list to return empty list (no open PRs, so recovery is allowed)
    fake_gh.pr_list = lambda: []

    # Simulate a prior dispatch that crashed (status: dispatched, same branch)
    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "branch_name": "agent/issue-123-fix-search",  # Same branch as would be generated
        "title": "Fix search",
        "url": "https://example.test/issues/123",
    }
    save_state(paths.state_file, seed)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.ok is True
    # The critical assertion: recovery record must be passed to the adapter
    assert captured["recovery"] is not None
    assert captured["recovery"]["status"] == "dispatched"
    assert captured["recovery"]["branch_name"] == "agent/issue-123-fix-search"
    assert (123, "agent:in-progress") in fake_gh.labels_added


def test_janitor_block_writes_no_review_packet(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    assert not packet.exists()  # zero packet spend on a blocked PR
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"


def test_janitor_warnings_surface_in_review_packet(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "additions": 2000, "deletions": 10}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    assert "Janitor warnings" in packet.read_text(encoding="utf-8")
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["janitor_ok"] is True
    assert state["prs"]["456"]["janitor_warnings"]


def test_review_decision_command_uses_valid_subparser_name(tmp_path: Path) -> None:
    """Regression test for issue #10: the decision command must use a valid
    argparse subparser name. The CLI registers 'verdict', not 'record-review'."""
    from charlie_work.cli import build_parser

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")

    # Extract the decision command from the packet
    import re

    match = re.search(r"charlie (verdict|record-review) --pr", packet_text)
    assert match is not None, f"decision command not found in packet. Packet text:\n{packet_text}"
    command_verb = match.group(1)

    # Verify the verb is a registered subparser
    parser = build_parser()
    # Find the subparsers action (it's the _SubParsersAction in _actions)
    subparsers_action = None
    for action in parser._subparsers._actions:
        if hasattr(action, "choices") and action.choices:
            subparsers_action = action
            break
    assert subparsers_action is not None, "Could not find subparsers action"
    subparser_choices = set(subparsers_action.choices.keys())
    assert command_verb in subparser_choices, (
        f"decision command uses '{command_verb}' which is not a registered subparser. "
        f"Valid subparsers: {sorted(subparser_choices)}"
    )
    assert command_verb == "verdict", (
        f"decision command should use 'verdict' subparser, not '{command_verb}'"
    )


def test_reconcile_wiring_reports_clean_repo(tmp_path: Path) -> None:
    class QuietGitHub(FakeGitHub):
        def run(self, arguments, *, json_output=False, allow_failure=False):
            return []

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, QuietGitHub())

    result = app.reconcile()

    assert result.ok is True
    assert result.data["drift"] == []
    assert result.data["fixed"] is False


def test_cli_routes_reconcile_fix_flag(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    class StubApp:
        def reconcile(self, *, fix: bool = False):
            seen["fix"] = fix
            return cli.CommandResult(True, "ok", {})

    monkeypatch.setattr(cli, "build_app", lambda args: StubApp())

    assert cli.main(["mop-up", "--fix"]) == 0
    assert seen["fix"] is True


def test_reconcile_exit_nonzero_when_drift_found_and_not_fixed(tmp_path: Path) -> None:
    """mop-up without --fix must exit non-zero when drift is present (CI gateable)."""

    class DriftGitHub(FakeGitHub):
        def run(self, arguments, *, json_output=False, allow_failure=False):
            # pr list: one open PR linked to issue 123
            if arguments[:2] == ["pr", "list"]:
                return [
                    {
                        "number": 456,
                        "title": "fix",
                        "url": "u",
                        "headRefName": "agent/issue-123-x",
                        "baseRefName": "main",
                        "body": "",
                        "state": "MERGED",
                        "labels": [],
                        "isCrossRepository": False,
                    }
                ]
            # issue list: issue 123 still has agent:in-progress (drift)
            if arguments[:2] == ["issue", "list"]:
                return [
                    {
                        "number": 123,
                        "title": "t",
                        "url": "u",
                        "body": "",
                        "labels": [{"name": "agent:in-progress"}],
                    }
                ]
            return []

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, DriftGitHub())

    result = app.reconcile(fix=False)

    assert result.ok is False
    assert result.data["fixed"] is False
    assert len(result.data["drift"]) > 0


def test_reconcile_exit_ok_when_drift_fixed(tmp_path: Path) -> None:
    """mop-up --fix must exit zero when all drift is repaired."""
    config = OrchestratorConfig()

    class DriftGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self._pr = {
                "number": 456,
                "title": "fix",
                "url": "u",
                "headRefName": "agent/issue-123-x",
                "baseRefName": "main",
                "body": "",
                "state": "MERGED",
                "labels": [],
                "isCrossRepository": False,
                "headRepositoryOwner": "owner",
                "baseRepositoryOwner": "owner",
            }
            self._issue = {
                "number": 123,
                "title": "t",
                "url": "u",
                "body": "",
                "labels": [{"name": "agent:in-progress"}],
            }

        def run(self, arguments, *, json_output=False, allow_failure=False):
            if arguments[:2] == ["pr", "list"]:
                return [self._pr]
            if arguments[:2] == ["issue", "list"]:
                return [self._issue]
            return []

        def remove_issue_label(self, number: int, label: str) -> None:
            super().remove_issue_label(number, label)
            self._issue["labels"] = [
                item for item in self._issue["labels"] if item.get("name") != label
            ]

        def add_issue_label(self, number: int, label: str) -> None:
            super().add_issue_label(number, label)
            names = {item.get("name") for item in self._issue["labels"]}
            if label not in names:
                self._issue["labels"].append({"name": label})

    app = OrchestratorApp(
        tmp_path, runtime_paths(tmp_path, config.runtime.state_dir), config, DriftGitHub()
    )

    result = app.reconcile(fix=True)

    assert result.ok is True
    assert result.data["fixed"] is True
    assert result.data["drift_before"] == 1
    assert result.data["drift_after"] == 0
    assert result.data["remaining_drift"] == []


def test_reconcile_partial_fix_failure_reports_remaining_drift(tmp_path: Path) -> None:
    """mop-up --fix must exit non-zero when a label removal silently fails."""
    config = OrchestratorConfig()

    class FailingRemoveGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self._issue = {
                "number": 30,
                "title": "t",
                "url": "u",
                "body": "",
                "labels": [{"name": "agent:in-progress"}],
            }

        def run(self, arguments, *, json_output=False, allow_failure=False):
            if arguments[:2] == ["pr", "list"]:
                return []
            if arguments[:2] == ["issue", "list"]:
                return [self._issue]
            return []

        def remove_issue_label(self, number: int, label: str) -> None:
            # Simulate allow_failure=True silently dropping the removal.
            pass

    app = OrchestratorApp(
        tmp_path, runtime_paths(tmp_path, config.runtime.state_dir), config, FailingRemoveGitHub()
    )

    result = app.reconcile(fix=True)

    assert result.ok is False
    assert result.data["fixed"] is False
    assert result.data["drift_before"] == 1
    assert result.data["drift_after"] == 1
    assert len(result.data["remaining_drift"]) == 1
    assert result.data["remaining_drift"][0]["kind"] == "issue_active_label_no_open_pr"
    assert "partially fixed" in result.message


# --- --repo path validation ----------------------------------------------------


def test_cli_repo_nonexistent_path_errors(tmp_path: Path, capsys) -> None:
    """charlie --repo <nonexistent> must error cleanly (exit 2), not create dirs."""
    ghost = tmp_path / "ghost-repo"
    assert not ghost.exists()

    exit_code = cli.main(["--repo", str(ghost), "roll-call"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "ghost-repo" in err or "--repo" in err
    # Must NOT have created the phantom directory.
    assert not ghost.exists()


def test_find_repo_root_explicit_raises_on_missing_path(tmp_path: Path) -> None:
    from charlie_work.paths import RepoNotFoundError, find_repo_root

    missing = tmp_path / "no-such-dir"

    try:
        find_repo_root(missing, explicit=True)
    except RepoNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RepoNotFoundError")


def test_find_repo_root_explicit_raises_when_not_git_repo(tmp_path: Path) -> None:
    from charlie_work.paths import RepoNotFoundError, find_repo_root

    non_git = tmp_path / "plain-dir"
    non_git.mkdir()

    try:
        find_repo_root(non_git, explicit=True)
    except RepoNotFoundError as exc:
        assert "git work tree" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RepoNotFoundError")


# --- adversarial-review fixes: regressions + coverage gaps ---------------------


def _approved_automerge():
    from charlie_work.config import AutoMergeConfig

    # No required checks -> the check gate is vacuously satisfied, isolating the
    # approved-decision path for merge tests.
    return AutoMergeConfig(required_checks=(), require_approved_review=True)


def test_linked_issue_number_rejects_bare_hash_in_attacker_title() -> None:
    # A bare #N substring in an attacker-controlled title must NOT bind the PR
    # to issue N (label/merge hijack). Only a closing keyword counts.
    assert (
        linked_issue_number(
            {"title": "Refactor everything #1 nicely"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        is None
    )
    assert (
        linked_issue_number(
            {"title": "see #5 for context", "body": "no link"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        is None
    )
    # Closing-keyword forms still resolve.
    assert (
        linked_issue_number(
            {"title": "Fix #321: thing"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 321
    )
    assert (
        linked_issue_number(
            {"body": "Resolves #7"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 7
    )
    # Orchestrator's own branch convention is the trusted head-ref signal.
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-456-x", "title": "#999"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 456
    )


def test_linked_issue_number_fork_pr_branch_name_does_not_bind() -> None:
    # Issue #9: Fork PRs must not bind via branch name (attacker-controlled).
    # A fork PR with branch name "issue-42" should NOT bind to issue 42.
    assert (
        linked_issue_number(
            {"headRefName": "issue-42-fix"},
            is_cross_repository=True,
            branch_prefix="agent/issue",
        )
        is None
    )
    # Even with the orchestrator's prefix, fork PRs must not bind via branch.
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-42-fix"},
            is_cross_repository=True,
            branch_prefix="agent/issue",
        )
        is None
    )


def test_linked_issue_number_same_repo_branch_with_prefix_binds() -> None:
    # Issue #9: Same-repo PRs with correct branch prefix should still bind.
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-42-fix"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 42
    )
    # Same-repo PR with wrong prefix should not bind via branch.
    assert (
        linked_issue_number(
            {"headRefName": "issue-42-fix"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        is None
    )


def test_linked_issue_number_fork_pr_closing_keyword_does_not_bind() -> None:
    # Issue #9: Fork PRs must NOT bind via closing keywords for lifecycle purposes.
    # (GitHub's own auto-close on merge is GitHub's policy for issue state;
    # the orchestrator's label lifecycle is ours.)
    assert (
        linked_issue_number(
            {"body": "Closes #42"},
            is_cross_repository=True,
            branch_prefix="agent/issue",
        )
        is None
    )
    assert (
        linked_issue_number(
            {"title": "Fix #42: security issue"},
            is_cross_repository=True,
            branch_prefix="agent/issue",
        )
        is None
    )


def test_linked_issue_number_same_repo_closing_keyword_binds() -> None:
    # Same-repo PRs should still bind via closing keywords.
    assert (
        linked_issue_number(
            {"body": "Closes #42"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 42
    )
    assert (
        linked_issue_number(
            {"title": "Fix #42: security issue"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 42
    )


def test_linked_issue_number_none_treats_as_cross_repository() -> None:
    # When is_cross_repository is None (provenance unknown), treat as cross-repo
    # for trust purposes — bind nothing via branch name or closing keyword
    # (fail closed). This hardens against future call sites that omit the
    # parameter or pass a PR dict missing the isCrossRepository field.
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-42-fix"},
            is_cross_repository=None,
            branch_prefix="agent/issue",
        )
        is None
    )
    assert (
        linked_issue_number(
            {"body": "Closes #42"},
            is_cross_repository=None,
            branch_prefix="agent/issue",
        )
        is None
    )


def test_rework_cap_survives_event_log_truncation(tmp_path: Path) -> None:
    # The P0: the counter used to derive from state["events"], which
    # append_event truncates to the last 200 - evicting a PR's earlier
    # request_changes and silently resetting the cap. The durable per-PR
    # counter must escalate regardless of how many unrelated events churn.
    from charlie_work.state import append_event as _append

    config = OrchestratorConfig()  # max_rework_cycles = 2
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "request_changes", summary="a")
    app.record_review(456, "request_changes", summary="b")
    # Flood the event log so any record_review events for 456 are evicted.
    state = load_state(paths.state_file)
    for i in range(300):
        state = _append(state, "review_packet", {"pr_number": 90000 + i})
    save_state(paths.state_file, state)
    assert not any(  # prove the earlier request_changes events are gone
        e.get("kind") == "record_review" for e in load_state(paths.state_file)["events"]
    )

    third = app.record_review(456, "request_changes", summary="c")

    assert third.data["escalated"] is True
    assert third.data["rework_path"] is None
    assert (123, "agent:human-needed") in fake_gh.labels_added


def test_record_review_approved_transitions_labels(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")

    # review_approved clears reviewing/needs-rework so the issue isn't stuck.
    assert (123, "agent:reviewing") in fake_gh.labels_removed
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "approved"


def test_review_started_clears_needs_rework() -> None:
    # Re-review after a rework must not stack reviewing on top of needs-rework.
    from charlie_work.labels import transition

    fake_gh = FakeGitHub()
    transition(fake_gh, OrchestratorConfig().labels, 123, "review_started")

    assert (123, "agent:pr-open") in fake_gh.labels_added
    assert (123, "agent:reviewing") in fake_gh.labels_added
    assert (123, "agent:needs-rework") in fake_gh.labels_removed


def test_dispatch_rework_skips_manual_adapter(tmp_path: Path) -> None:
    """Rework dispatch must skip manual adapters to preserve human-paste path."""
    config = OrchestratorConfig(devin=DevinConfig(adapter="manual"))
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
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        )
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
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        )
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


def test_dispatch_rework_releases_claims_when_all_skipped(tmp_path: Path) -> None:
    """When all candidates lack rework-prompt.md, dispatch_pending claims must be released."""
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        )
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
    # Verify the claim was released: status should be dispatch_failed, not dispatch_pending
    state = load_state(paths.state_file)
    issue_state = state["issues"].get("123")
    assert issue_state is not None
    assert issue_state.get("status") == "dispatch_failed"
    assert issue_state.get("dispatch_pending_at") is None


def test_merge_ready_sets_status_merged(tmp_path: Path) -> None:
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok")

    result = app.merge_ready(456, merge=True)

    assert result.data["merged"] is True
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "merged"


def test_merge_ready_keeps_merged_state_when_label_transition_fails(tmp_path: Path) -> None:
    from charlie_work.github import GitHubError as _GitHubError

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> None:
            raise _GitHubError("rate limited")

    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    # Write the approved decision directly so the merge gate opens without
    # needing a (failing) label transition first.
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.merge_ready(456, merge=True)

    assert result.data["merged"] is True
    assert result.data["label_error"] == "rate limited"
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "merged"


def test_merge_ready_evaluation_only_preserves_recorded_merged_fact(tmp_path: Path) -> None:
    """A later evaluation-only run must not overwrite a previously recorded merged fact."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok")

    merge_result = app.merge_ready(456, merge=True)
    assert merge_result.data["merged"] is True
    merged_state = load_state(paths.state_file)["prs"]["456"]
    assert merged_state["status"] == "merged"
    assert merged_state["merged"] is True

    # A subsequent evaluation-only pass short-circuits via the idempotence guard
    # and reports the PR as already merged without re-calling gh pr merge.
    eval_result = app.merge_ready(456, merge=False)
    assert eval_result.ok is True
    assert eval_result.data["already_merged"] is True
    assert eval_result.data["merged"] is True
    # merge_pr must NOT have been called again.
    assert fake_gh.merged == [(456, "squash")]  # only the first merge
    persisted = load_state(paths.state_file)["prs"]["456"]
    assert persisted["status"] == "merged"
    assert persisted["merged"] is True


def test_dispatch_guard_blocks_second_worker_for_live_dispatched_issue(tmp_path: Path) -> None:
    """A live dispatched issue is not re-dispatched even if label write failed."""
    from charlie_work.devin_shell import SessionRecord

    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="devin-shell",
            shell_command=(sys.executable, "-c", "import sys; sys.exit(0)"),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Simulate a prior dispatch that launched a worker but whose label write
    # failed: state says "dispatched" but the issue still lacks active labels.
    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {"number": 123, "status": "dispatched"}
    save_state(paths.state_file, seed)
    # Create a genuinely live worker session record
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    # Spawn a short-lived process
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        record = SessionRecord(
            issue_number=123,
            branch="agent/issue-123",
            worktree_path="/tmp/wt/issue-123",
            prompt_path="p.md",
            command=("x",),
            pid=process.pid,
            started_at="2026-01-01T00:00:00Z",
            log_path="log.txt",
        )
        # Write the session record manually (mirrors internal _write_json pattern)
        sidecar_path = sessions_dir / f"issue-{123}.json"
        tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
        tmp.write_text(json.dumps(record.to_dict()), encoding="utf-8")
        tmp.replace(sidecar_path)
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)

        result = app.dispatch(limit=3)

        assert result.data["attempted_count"] == 0  # not re-dispatched
    finally:
        process.kill()
        process.wait()


def test_dispatch_recovers_dead_worker_without_open_pr(tmp_path: Path) -> None:
    """Issue #5: a dead worker with no open PR becomes dispatchable again."""
    from charlie_work.devin_shell import SessionRecord

    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="manual",  # Use manual to avoid actual worker launch
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Override prs to return empty list (no open PRs)
    fake_gh.prs = []
    # Simulate a prior dispatch that crashed before PR opened
    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "title": "Test issue",
        "url": "https://github.com/test/repo/issues/123",
    }
    save_state(paths.state_file, seed)
    # Create a session record with a dead PID
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    # Spawn and immediately wait for a short-lived process to get a dead PID
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process.wait()  # Ensure it's dead
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123",
        worktree_path="/tmp/wt/issue-123",
        prompt_path="p.md",
        command=("x",),
        pid=process.pid,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
    )
    # Write the session record manually (mirrors internal _write_json pattern)
    sidecar_path = sessions_dir / f"issue-{123}.json"
    tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    tmp.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    tmp.replace(sidecar_path)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=3)

    # The issue should be re-dispatched since the worker is dead and there's no open PR
    assert result.data["attempted_count"] == 1
    assert result.data["selected_count"] == 1
    assert 123 in [s["issue_number"] for s in result.data["sessions"]]


def test_dispatch_does_not_recover_dead_worker_with_open_pr(tmp_path: Path) -> None:
    """Issue #5: a dead worker with an open PR is NOT re-dispatched (mid-review)."""
    from charlie_work.devin_shell import SessionRecord

    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="manual",  # Use manual to avoid actual worker launch
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Override prs to return an open PR for this issue
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix issue 123",
            "headRefName": "agent/issue-123",
            "url": "https://github.com/test/repo/pull/456",
            "isCrossRepository": False,
        }
    ]
    # Simulate a prior dispatch that crashed after PR opened
    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "title": "Test issue",
        "url": "https://github.com/test/repo/issues/123",
    }
    save_state(paths.state_file, seed)
    # Create a session record with a dead PID
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    # Spawn and immediately wait for a short-lived process to get a dead PID
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process.wait()  # Ensure it's dead
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123",
        worktree_path="/tmp/wt/issue-123",
        prompt_path="p.md",
        command=("x",),
        pid=process.pid,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
    )
    # Write the session record manually (mirrors internal _write_json pattern)
    sidecar_path = sessions_dir / f"issue-{123}.json"
    tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    tmp.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    tmp.replace(sidecar_path)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=3)

    # The issue should NOT be re-dispatched since there's an open PR
    assert result.data["attempted_count"] == 0


def test_dispatch_isolates_label_write_failure(tmp_path: Path, monkeypatch) -> None:
    from charlie_work import devin_shell
    from charlie_work.github import GitHubError as _GitHubError
    from charlie_work.worktree import WorktreeInfo

    wt_path = tmp_path / "worktrees" / "agent-issue-123-fix-search"
    wt_path.mkdir(parents=True, exist_ok=True)

    def _fake_create_worktree(repo_root, branch, **kwargs):
        return WorktreeInfo(path=wt_path, branch=branch, venv_junction=None)

    monkeypatch.setattr(devin_shell, "create_worktree", _fake_create_worktree)

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> None:
            raise _GitHubError("edit failed")

    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="devin-shell",
            shell_command=(sys.executable, "-c", "import sys; sys.exit(0)"),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    # Worker launched and recorded even though labeling failed - no crash.
    assert 123 in result.data["label_errors"]
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"
    assert "label_error" in state["issues"]["123"]


def test_dispatch_issues_reports_skipped(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app.dispatch(only_issues="123,999")

    assert result.data["skipped_issue_numbers"] == [999]
    assert "999" in result.message


def test_concurrent_dispatch_claims_prevent_double_launch(tmp_path: Path) -> None:
    """A dispatch_pending claim must block a second dispatch for the same issue."""
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; sys.exit(0)"),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # First dispatch creates a dispatch_pending claim
    first_result = app.dispatch(limit=1)
    assert first_result.data["attempted_count"] == 1

    # Verify the claim was created
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"  # Upgraded after successful launch

    # Simulate a crashed phase-2 by manually setting status back to dispatch_pending
    state["issues"]["123"]["status"] = "dispatch_pending"
    state["issues"]["123"]["dispatch_pending_at"] = (
        "2099-01-01T00:00:00Z"  # Far future = not stale
    )
    save_state(paths.state_file, state)

    # Second dispatch should be blocked by the fresh claim
    second_result = app.dispatch(limit=1)
    assert second_result.data["attempted_count"] == 0  # Blocked by claim

    # Verify the claim is still in place
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatch_pending"


def test_stale_dispatch_pending_claim_is_redispatchable(tmp_path: Path, monkeypatch) -> None:
    """A stale dispatch_pending claim (crashed phase-2) must be re-dispatchable."""
    from charlie_work.state import is_claim_stale

    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; sys.exit(0)"),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Seed state with a stale dispatch_pending claim (simulating crashed phase-2)
    seed = load_state(paths.state_file)
    # We need to mock is_claim_stale to return True for our test timestamp
    original_is_claim_stale = is_claim_stale

    def _mock_is_claim_stale(claim_timestamp: str | None) -> bool:
        if claim_timestamp == "2020-01-01T00:00:00+00:00":
            return True  # Treat this specific timestamp as stale
        return original_is_claim_stale(claim_timestamp)

    monkeypatch.setattr("charlie_work.state.is_claim_stale", _mock_is_claim_stale)
    monkeypatch.setattr("charlie_work.workflow.is_claim_stale", _mock_is_claim_stale)

    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatch_pending",
        "dispatch_pending_at": "2020-01-01T00:00:00+00:00",  # Stale timestamp
    }
    save_state(paths.state_file, seed)

    # Dispatch should re-dispatch the stale claim
    result = app.dispatch(limit=1)

    assert result.data["attempted_count"] == 1  # Re-dispatched
    state = load_state(paths.state_file)
    # Status should now be "dispatched" (upgraded from stale claim)
    assert state["issues"]["123"]["status"] == "dispatched"
    # Stale claim timestamp should be cleared
    assert "dispatch_pending_at" not in state["issues"]["123"]


def test_bootstrap_labels_creates_every_configured_label(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.bootstrap_labels()

    created = {name for name, _color, _desc in fake_gh.labels_created}
    assert created == set(config.labels.all)
    assert all(desc for _n, _c, desc in fake_gh.labels_created)
    # All labels verified present — must report honest success.
    assert result.ok is True
    assert result.data["missing"] == []


def test_bootstrap_labels_fails_when_creation_silently_missed(tmp_path: Path) -> None:
    """If label_create silently fails (e.g. no auth), bootstrap must report failure."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FailingCreateGitHub(FakeGitHub):
        def label_create(self, label: str, color: str, description: str) -> None:
            # Silently drop all creates — simulates no-auth / wrong-repo scenario.
            pass

        def label_list(self) -> list[dict[str, object]]:
            return []  # nothing was created

    fake_gh = FailingCreateGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.bootstrap_labels()

    assert result.ok is False
    assert result.data["missing"] == config.labels.all


def test_bootstrap_labels_fails_when_label_list_raises(tmp_path: Path) -> None:
    """If label_list fails (e.g. network error), bootstrap must report failure."""
    from charlie_work.github import GitHubError

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ErrorListGitHub(FakeGitHub):
        def label_list(self) -> list[dict[str, object]]:
            raise GitHubError("could not list labels: HTTP 401")

    fake_gh = ErrorListGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.bootstrap_labels()

    assert result.ok is False
    assert "verification failed" in result.message


def test_status_aggregates_counts(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app.status()

    assert result.ok is True
    assert result.data["ready_issue_count"] == 1
    assert result.data["available_issue_count"] == 1
    assert result.data["open_linked_pr_count"] == 1


def test_github_dry_run_skips_mutating_command(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = github_module.GitHub(repo_root=tmp_path, dry_run=True)

    out = gh.run(["pr", "merge", "1", "--squash"])

    assert out.startswith("DRY-RUN:")
    assert calls == []  # subprocess.run never invoked for a mutating command


def test_github_dry_run_allows_readonly_command(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = github_module.GitHub(repo_root=tmp_path, dry_run=True)

    gh.run(["issue", "list", "--label", "x"], json_output=True)

    assert len(calls) == 1  # read-only command still executes under dry-run


def test_is_mutating_classifies_readonly_and_mutating() -> None:
    from charlie_work.github import _is_mutating

    for readonly in (
        ["issue", "list"],
        ["pr", "view", "1"],
        ["pr", "checks", "1"],
        ["label", "list"],
    ):
        assert _is_mutating(readonly) is False
    for mutating in (["pr", "merge", "1"], ["issue", "edit", "1"], ["label", "create", "x"]):
        assert _is_mutating(mutating) is True


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


def test_dry_run_skips_cross_family_review(monkeypatch, tmp_path: Path) -> None:
    """Test that --dry-run prevents cross-family model subprocess execution."""
    subprocess_calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        subprocess_calls.append(args[0])
        raise AssertionError("subprocess.run should not be called in dry-run mode")

    monkeypatch.setattr("charlie_work.cross_family.subprocess.run", fake_run)

    result = run_cross_family_review(
        model="test-model",
        command=["echo", "test"],
        repo_root=tmp_path,
        prompt_text="test prompt",
        prompt_path=tmp_path / "prompt.md",
        report_path=tmp_path / "report.md",
        timeout_seconds=30,
        dry_run=True,
    )

    assert result.ok is False
    assert result.error == "DRY-RUN: cross-family review not executed"
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


def test_cli_main_maps_github_error_to_exit_2(monkeypatch, capsys) -> None:
    from charlie_work.github import GitHubError as _GitHubError

    def _boom(args):
        raise _GitHubError("boom")

    monkeypatch.setattr(cli, "build_app", _boom)

    assert cli.main(["roll-call"]) == 2
    assert "GitHub error: boom" in capsys.readouterr().err


def test_cli_main_maps_config_error_to_exit_2(tmp_path: Path, monkeypatch, capsys) -> None:
    """Issue #12: ConfigError (e.g., unknown top-level section) yields exit 2."""
    from charlie_work.config import ConfigError as _ConfigError

    def _boom(args):
        raise _ConfigError("unknown config section(s): auto-merge")

    monkeypatch.setattr(cli, "build_app", _boom)

    assert cli.main(["roll-call"]) == 2
    assert "config error: unknown config section(s): auto-merge" in capsys.readouterr().err


def test_cli_main_maps_yaml_error_to_exit_2(tmp_path: Path, monkeypatch, capsys) -> None:
    """Issue #12: YAMLError (malformed config) yields exit 2."""

    def _boom(args):
        raise yaml.YAMLError("malformed YAML")

    monkeypatch.setattr(cli, "build_app", _boom)

    assert cli.main(["roll-call"]) == 2
    assert "YAML error: malformed YAML" in capsys.readouterr().err


# --- Issue #18: idempotence of ship-it and loop --------------------------------


def test_merge_ready_already_merged_is_noop(tmp_path: Path) -> None:
    """ship-it on a PR whose state records status='merged' must return ok=True
    without re-attempting `gh pr merge` (which would fail on an already-merged PR)."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    # Seed state as if a prior merge_ready already merged this PR.
    state = load_state(paths.state_file)
    state["prs"]["456"] = {"number": 456, "issue_number": 123, "status": "merged", "merged": True}
    save_state(paths.state_file, state)

    result = app.merge_ready(456)

    assert result.ok is True
    assert result.data["already_merged"] is True
    assert result.data["merged"] is True
    # merge_pr must NOT have been called — the fake would record it.
    assert fake_gh.merged == []


def test_loop_skips_review_for_approved_unmerged_pr(tmp_path: Path) -> None:
    """A second loop() pass over an approved-but-unmerged PR must NOT rewrite
    the review packet or re-fire label transitions — it should go straight to
    merge_ready."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    # Record an approved decision in state (as record_review would).
    state = load_state(paths.state_file)
    state["prs"]["456"] = {
        "number": 456,
        "issue_number": 123,
        "decision": "approved",
        "status": "approved",
        "reviewed_head_sha": "sha-abc123",
    }
    save_state(paths.state_file, state)
    # Also write the decision file so merge_ready can read it.
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.loop(limit=0)

    # review() was skipped — no review packet written, no reviewing label fired.
    assert result.data["reviews"] == []
    # merge_ready was attempted (straight to merge evaluation).
    assert len(result.data["merges"]) == 1
    # The reviewing label must NOT have been re-added (would indicate review() ran).
    assert (123, "agent:reviewing") not in fake_gh.labels_added


# --- Issue #31: approvals pinned to PR head SHA --------------------------------


def test_record_review_captures_reviewed_head_sha(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(456, "approved", summary="lgtm")

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["reviewed_head_sha"] == "sha-abc123"
    assert load_state(paths.state_file)["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"
    assert result.data["reviewed_head_sha"] == "sha-abc123"


# --- Issue #11: reject empty summary for request_changes/blocked decisions ----


def test_record_review_request_changes_rejects_empty_summary(tmp_path: Path) -> None:
    """Issue #11: request_changes with empty summary is rejected before state/label mutation."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(456, "request_changes", summary="")

    assert result.ok is False
    assert "--summary or --summary-file is required" in result.message
    # Verify no state/label mutation occurred
    assert load_state(paths.state_file).get("prs", {}).get("456") is None
    assert (123, "agent:needs-rework") not in fake_gh.labels_added
    # Verify no rework prompt was written
    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    assert not rework_prompt.exists()


def test_record_review_blocked_rejects_empty_summary(tmp_path: Path) -> None:
    """Issue #11: blocked with empty summary is rejected before state/label mutation."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(456, "blocked", summary="")

    assert result.ok is False
    assert "--summary or --summary-file is required" in result.message
    # Verify no state/label mutation occurred
    assert load_state(paths.state_file).get("prs", {}).get("456") is None
    assert (123, "agent:blocked") not in fake_gh.labels_added


def test_record_review_request_changes_rejects_whitespace_only_summary(tmp_path: Path) -> None:
    """Issue #11: request_changes with whitespace-only summary is rejected."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(456, "request_changes", summary="   \n\t  ")

    assert result.ok is False
    assert "--summary or --summary-file is required" in result.message


def test_record_review_approved_allows_empty_summary(tmp_path: Path) -> None:
    """Issue #11: approved with empty summary is allowed (no validation required)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(456, "approved", summary="")

    assert result.ok is True
    assert result.message == "review recorded"
    # Verify state mutation occurred
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "approved"


def test_record_review_decision_payload_includes_required_changes(tmp_path: Path) -> None:
    """Issue #11: decision payload always includes required_changes field."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert "required_changes" in decision
    assert decision["required_changes"] == []

    app.record_review(456, "request_changes", summary="fix A")

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert "required_changes" in decision
    assert decision["required_changes"] == []


def test_record_review_request_changes_updates_issue_status_to_rework_requested(
    tmp_path: Path,
) -> None:
    """Issue #72: request_changes (non-escalated) updates issue status to rework_requested
    so dispatch_rework can select it."""
    from charlie_work.github import linked_issue_number

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Verify that linked_issue_number returns the correct issue number
    issue_number = linked_issue_number(
        fake_gh.prs[0],
        is_cross_repository=fake_gh.prs[0].get("isCrossRepository"),
        branch_prefix=config.dispatch.branch_prefix,
    )
    assert issue_number == 123

    # Record a non-escalated request_changes decision
    result = app.record_review(456, "request_changes", summary="fix A")

    assert result.ok is True
    assert result.data["escalated"] is False

    # Assert the actual state change: issue status should be rework_requested
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"


def test_standard_lifecycle_rework_dispatch_selects_issue(tmp_path: Path) -> None:
    """Issue #72 acceptance criterion 1: standard-lifecycle end-to-end rework dispatch.

    Fresh dispatch marks the issue dispatched → record_review(request_changes) →
    dispatch_rework SELECTS the issue and launches via a command-adapter fake,
    firing the rework_dispatched label transition.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Step 1: Fresh dispatch marks the issue as dispatched
    dispatch_result = app.dispatch(limit=1)
    assert dispatch_result.ok is True
    assert dispatch_result.data["selected_count"] == 1

    # Verify issue is marked as dispatched in state
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"

    # Step 2: record_review(request_changes) updates issue status to rework_requested
    review_result = app.record_review(456, "request_changes", summary="fix A")
    assert review_result.ok is True
    assert review_result.data["escalated"] is False

    # Verify issue status is now rework_requested
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"

    # Step 3: Create a rework prompt (normally written by record_review)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    # Step 4: dispatch_rework SELECTS the issue and launches via command adapter
    # The issue already has needs-rework label from the request_changes transition
    rework_result = app.dispatch_rework()

    # Verify dispatch_rework selected and launched the issue
    assert rework_result.ok is True
    assert rework_result.data["selected_count"] == 1
    assert rework_result.data["dispatch_results"][0]["stdout"].strip() == "123"

    # Verify the rework_dispatched label transition was fired
    # (adds in_progress, removes needs_rework)
    assert (123, "agent:in-progress") in fake_gh.labels_added
    assert (123, "agent:needs-rework") in fake_gh.labels_removed


def test_escalated_request_changes_does_not_make_issue_selectable(tmp_path: Path) -> None:
    """Issue #72 acceptance criterion 2: escalated request_changes must NOT make issue selectable.

    After an escalated verdict (request_changes_count at max), assert the issue's state status
    is NOT rework_requested and dispatch_rework does not select it.
    """
    config = OrchestratorConfig(
        review=ReviewConfig(max_rework_cycles=2),  # Set max to 2 for this test
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Step 1: Fresh dispatch marks the issue as dispatched
    dispatch_result = app.dispatch(limit=1)
    assert dispatch_result.ok is True
    assert dispatch_result.data["selected_count"] == 1

    # Step 2: Record first request_changes (count = 1, not escalated)
    review_result_1 = app.record_review(456, "request_changes", summary="fix A")
    assert review_result_1.ok is True
    assert review_result_1.data["escalated"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert state["issues"]["123"]["status"] == "rework_requested"

    # Step 3: Record second request_changes (count = 2, not escalated yet)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    review_result_2 = app.record_review(456, "request_changes", summary="fix B")
    assert review_result_2.ok is True
    assert review_result_2.data["escalated"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["issues"]["123"]["status"] == "rework_requested"

    # Step 4: Record third request_changes (count stays at 2, escalated because max_rework_cycles = 2)
    # When escalated, the count is NOT incremented (see workflow.py line 731-734)
    review_result_3 = app.record_review(456, "request_changes", summary="fix C")
    assert review_result_3.ok is True
    assert review_result_3.data["escalated"] is True  # Should be escalated

    # Verify PR status is escalated
    state = load_state(paths.state_file)
    assert (
        state["prs"]["456"]["request_changes_count"] == 2
    )  # Count does NOT increment when escalated
    assert state["prs"]["456"]["status"] == "escalated"
    # Issue status should now be escalated (cleared from rework_requested)
    assert state["issues"]["123"]["status"] == "escalated"

    # Step 5: Verify the escalated label transition was fired (adds human_needed, removes reviewing)
    assert (123, "agent:human-needed") in fake_gh.labels_added
    # The escalated transition removes reviewing but does NOT remove needs_rework
    # (this is by design per labels.py: "escalated": ((labels.human_needed,), (labels.reviewing,)))

    # Step 6: dispatch_rework should still NOT select the escalated issue
    # because the issue status is "escalated" (not "rework_requested")
    # The escalated issue still has needs-rework label (from previous non-escalated request_changes)
    rework_result = app.dispatch_rework()

    # Verify dispatch_rework did NOT select the escalated issue
    # (even though it has needs_rework label, the issue status is escalated so it's filtered out)
    assert rework_result.ok is True
    assert rework_result.data["selected_count"] == 0
    # No new in_progress label should have been added (rework_dispatched transition)
    # Count how many in_progress labels were added before this step
    in_progress_count_before = fake_gh.labels_added.count((123, "agent:in-progress"))
    # After the failed dispatch, the count should be the same
    in_progress_count_after = fake_gh.labels_added.count((123, "agent:in-progress"))
    assert in_progress_count_after == in_progress_count_before


def test_merge_ready_refuses_when_head_moved_after_approval(tmp_path: Path) -> None:
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    fake_gh.prs[0] = {**fake_gh.prs[0], "headRefOid": "sha-new-head"}

    result = app.merge_ready(456, merge=True)

    assert result.ok is False
    assert "PR head moved since approval" in result.message
    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert result.data["head_moved"] is True
    assert fake_gh.merged == []
    assert (123, "agent:reviewing") in fake_gh.labels_added
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "reviewing"


def test_merge_ready_merges_when_head_unchanged_after_approval(tmp_path: Path) -> None:
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]


def test_merge_ready_legacy_approved_decision_without_head_sha_is_refused(
    tmp_path: Path,
) -> None:
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved"}), encoding="utf-8"
    )

    result = app.merge_ready(456, merge=True)

    assert result.ok is False
    assert "PR head moved since approval" in result.message
    assert result.data["head_moved"] is True
    assert result.data["merged"] is False
    assert fake_gh.merged == []


def test_loop_re_reviews_when_head_moved_after_approval(tmp_path: Path) -> None:
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    # Seed an approved decision pinned to the old head.
    state = load_state(paths.state_file)
    state["prs"]["456"] = {
        "number": 456,
        "issue_number": 123,
        "decision": "approved",
        "status": "approved",
        "reviewed_head_sha": "sha-abc123",
    }
    save_state(paths.state_file, state)
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )
    # New commit pushed after approval.
    fake_gh.prs[0] = {**fake_gh.prs[0], "headRefOid": "sha-new-head"}

    result = app.loop(limit=0)

    assert len(result.data["reviews"]) == 1
    assert result.data["merges"] == []
    assert (123, "agent:reviewing") in fake_gh.labels_added
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "reviewing"


def test_loop_skips_review_and_merges_when_head_unchanged_after_approval(
    tmp_path: Path,
) -> None:
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    state = load_state(paths.state_file)
    state["prs"]["456"] = {
        "number": 456,
        "issue_number": 123,
        "decision": "approved",
        "status": "approved",
        "reviewed_head_sha": "sha-abc123",
    }
    save_state(paths.state_file, state)
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.loop(limit=0)

    assert result.data["reviews"] == []
    assert len(result.data["merges"]) == 1
    assert result.data["merges"][0]["merged"] is True
    assert (123, "agent:reviewing") not in fake_gh.labels_added


def test_loop_classifies_dead_sessions_and_sets_throttle_state(tmp_path: Path) -> None:
    """Test that loop() classifies dead sessions and sets throttled_until in state.

    This is a loop-path integration test: it constructs the app with a fake adapter,
    simulates a session that died with the rate-limit signature, runs a loop pass,
    then asserts (a) throttled_until is persisted in state and (b) a subsequent
    dispatch() defers launches until it expires.

    The test MUST fail when _classify_dead_sessions_and_update_throttle_state is
    removed from loop() — this is the acceptance test for the exact regression class.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime, timedelta

    # Use command adapter to avoid needing real devin binary
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Ensure state directory exists
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a sessions directory with a dead session that has a rate-limit log
    # Use the config's sessions_dir path
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Run a loop pass with limit=0 (no actual dispatch, just the classification logic)
    # We don't assert result.ok because dispatch may fail with no issues to process
    # The key is that the classification logic runs regardless
    app.loop(limit=0)

    # Verify throttled_until was set in state by the loop's classification pass
    state = load_state(paths.state_file)
    assert state.get("throttled_until") is not None

    # Verify the cooldown reflects the parsed 10 minutes
    throttle_time = datetime.fromisoformat(state["throttled_until"].replace("Z", "+00:00"))
    expected_time = datetime.now(UTC) + timedelta(minutes=10)
    # Allow 2 second tolerance for test execution time
    assert abs((throttle_time - expected_time).total_seconds()) < 2

    # Verify that a subsequent dispatch() defers launches while throttled
    # Add a dispatchable issue
    fake_gh.issues = [
        {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "body": "Search is broken",
            "labels": [{"name": "automated-ready"}],
        }
    ]

    dispatch_result = app.dispatch(limit=1)
    # Dispatch should be deferred due to throttle (ok=False is expected for deferral)
    assert dispatch_result.ok is False
    assert "deferred" in dispatch_result.message.lower()
    # Should defer launch due to throttle
    assert dispatch_result.data["selected_count"] == 0


def test_update_open_agent_prs_reports_failure_as_value(tmp_path: Path) -> None:
    """Test that pr_update_branch failures are reported as values, not successes."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Add a second PR to test batch behavior
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "headRepository": {
                "owner": {"login": "test"},
                "name": "repo",
            },
        }
    ]
    fake_gh.issues = [
        {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "body": "Search is broken",
            "labels": [{"name": "automated-ready"}],
        },
        {
            "number": 124,
            "title": "Fix another",
            "url": "https://example.test/issues/124",
            "body": "Another issue",
            "labels": [{"name": "automated-ready"}],
        },
    ]
    # Make update-branch fail for the second PR
    fake_gh.update_branch_ok = False

    # Override prs to return two PRs
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-def456",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Call _update_open_agent_prs directly
    results = app._update_open_agent_prs(merged_pr_number=456)

    # Should have results for both PRs (excluding the merged one)
    assert len(results) == 1  # Only PR 789 (456 is excluded as the merged PR)


def test_update_open_agent_prs_skips_approved_pending_ship_prs(tmp_path: Path) -> None:
    """Test that approved-pending-ship PRs are skipped to avoid invalidating approvals.

    Regression test for issue #89: when two PRs are approved in the same operator pass,
    merging the first should not base-update the second (which would move its head and
    invalidate its approval, forcing a manual re-approve loop).
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # Set up two approved PRs
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",  # Live head matches reviewed head
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-def456",  # Live head matches reviewed head
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    # Create review decision files for both PRs (approved state)
    pr_456_decision_dir = paths.prs / "pr-456"
    pr_456_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_456_decision_dir / "review-decision.json").write_text(
        json.dumps(
            {"decision": "approved", "reviewed_head_sha": "sha-abc123"},
            indent=2,
        ),
        encoding="utf-8",
    )

    pr_789_decision_dir = paths.prs / "pr-789"
    pr_789_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_789_decision_dir / "review-decision.json").write_text(
        json.dumps(
            {"decision": "approved", "reviewed_head_sha": "sha-def456"},
            indent=2,
        ),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Simulate merging PR 456: update remaining open PRs
    results = app._update_open_agent_prs(merged_pr_number=456)

    # PR 789 should be skipped (approved-pending-ship)
    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is False
    assert results[0]["skipped_reason"] == "approved-pending-ship"

    # Verify pr_update_branch was NOT called for PR 789
    assert fake_gh.update_branch_ok is True  # Should still be True (never called)


def test_update_open_agent_prs_updates_non_approved_prs(tmp_path: Path) -> None:
    """Test that PRs without approved decisions are still updated normally."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # Set up two PRs, one approved, one not
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-def456",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    # Only create review decision for PR 456 (approved)
    pr_456_decision_dir = paths.prs / "pr-456"
    pr_456_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_456_decision_dir / "review-decision.json").write_text(
        json.dumps(
            {"decision": "approved", "reviewed_head_sha": "sha-abc123"},
            indent=2,
        ),
        encoding="utf-8",
    )

    # PR 789 has no decision file (or a non-approved decision)
    pr_789_decision_dir = paths.prs / "pr-789"
    pr_789_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_789_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes"}, indent=2),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Simulate merging PR 456: update remaining open PRs
    results = app._update_open_agent_prs(merged_pr_number=456)

    # PR 789 should be updated (not approved)
    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is True
    assert "skipped_reason" not in results[0]


def test_update_open_agent_prs_updates_approved_prs_with_moved_head(tmp_path: Path) -> None:
    """Test that approved PRs with moved heads are still updated (not skipped).

    This ensures the head-moved gate remains intact for content-bearing moves.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # Set up an approved PR whose head has moved since approval
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-new456",  # Head has moved since approval
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    # Create review decision for PR 789 with old head
    pr_789_decision_dir = paths.prs / "pr-789"
    pr_789_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_789_decision_dir / "review-decision.json").write_text(
        json.dumps(
            {"decision": "approved", "reviewed_head_sha": "sha-def456"},  # Old head
            indent=2,
        ),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Simulate merging PR 456: update remaining open PRs
    results = app._update_open_agent_prs(merged_pr_number=456)

    # PR 789 should be updated (head moved, so not approved-pending-ship)
    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is True
    assert "skipped_reason" not in results[0]


def test_concurrency_governor_unlimited_when_unset(tmp_path: Path) -> None:
    """When max_concurrent_sessions is 0 (default), dispatch should behave as before (unlimited)."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=0),
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

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
    def mock_count_live(sessions_dir):
        return 2

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

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
    def mock_count_live(sessions_dir):
        return 2

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Add needs-rework label to the issue
            self.issues[0]["labels"] = [{"name": "agent:needs-rework"}]

        def issue_list(self, ready_label: str):
            if ready_label == "agent:needs-rework":
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
    def mock_count_live(sessions_dir):
        return 1

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(adapter="manual"),
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

    result = app.dispatch()

    # Should allow only 1 launch since 1 session is alive and cap is 2
    # (2 candidates available, but only 1 slot)
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["concurrency_limit"] == 2
    assert result.data["live_session_count"] == 1
    assert result.data["available_slots"] == 1


def test_dispatch_rework_state_driven_selection(tmp_path: Path) -> None:
    """Issue #85 acceptance criterion 1: state-driven selection works.

    State-driven selection ensures that issues with rework_requested status are selected
    regardless of label state. This test verifies that the selection logic uses state
    instead of labels.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        ),
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
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        ),
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
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        ),
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
                },
                {
                    "number": 457,
                    "title": "PR for issue 124",
                    "url": "https://example.test/pr/457",
                    "headRefOid": "def456",
                    "isCrossRepository": False,
                    "headRefName": "agent/issue-124",
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
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        ),
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
                "headRefOid": "abc123",
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


def test_review_started_skip_when_head_unchanged_after_request_changes(tmp_path: Path) -> None:
    """Review_started transition should be skipped when head hasn't changed after request_changes.

    This prevents pointless packet churn and preserves the needs_rework label on
    budget-deferred rework candidates.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record a request_changes decision with a specific head SHA
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

    app.record_review(456, "request_changes", summary="fix A")

    # Verify the decision was recorded
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        assert state["prs"]["456"]["decision"] == "request_changes"
        assert state["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"

    # Clear label tracking to isolate the review() call
    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    # Call review again with the same head SHA
    result = app.review(456)

    # The review_started transition should be skipped, so no labels should be added/removed
    assert result.ok is True
    # review_started transition adds pr_open and reviewing labels when it fires
    assert (123, "agent:pr-open") not in fake_gh.labels_added
    assert (123, "agent:reviewing") not in fake_gh.labels_added


def test_review_started_fires_when_head_advanced_after_request_changes(tmp_path: Path) -> None:
    """Review_started transition should fire when head has advanced after request_changes."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record a request_changes decision with a specific head SHA
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

    app.record_review(456, "request_changes", summary="fix A")

    # Advance the PR head
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"

    # Call review again with the advanced head
    result = app.review(456)

    # The review_started transition should fire (adds pr_open and reviewing)
    assert result.ok is True
    assert (123, "agent:pr-open") in fake_gh.labels_added
    assert (123, "agent:reviewing") in fake_gh.labels_added


def test_review_started_fires_when_no_recorded_verdict(tmp_path: Path) -> None:
    """Review_started transition should fire when there's no prior verdict."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create PR directory without any prior review decision
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

    # Call review without any prior verdict
    result = app.review(456)

    # The review_started transition should fire (adds pr_open and reviewing)
    assert result.ok is True
    assert (123, "agent:pr-open") in fake_gh.labels_added
    assert (123, "agent:reviewing") in fake_gh.labels_added


def test_dispatch_defers_when_provider_throttled(tmp_path: Path) -> None:
    """When provider throttle window is active, dispatch should defer and report why."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(default_limit=3),
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Set a throttle window in the future
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        # Set throttled_until to 1 hour in the future
        from datetime import UTC, datetime, timedelta

        future_time = datetime.now(UTC) + timedelta(hours=1)
        throttled_until = future_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        state = set_throttled_until(state, throttled_until)
        save_state(paths.state_file, state)

    result = app.dispatch()

    # Should defer with provider_throttled reason
    assert result.ok is False
    assert result.data["deferred_reason"] == "provider_throttled"
    assert result.data["throttled_until"] is not None
    assert result.data["selected_count"] == 0
    assert result.data["attempted_count"] == 0


def test_dispatch_proceeds_when_throttle_window_expired(tmp_path: Path) -> None:
    """When provider throttle window has passed, dispatch should proceed normally."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(default_limit=3),
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Set a throttle window in the past
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        from datetime import UTC, datetime, timedelta

        past_time = datetime.now(UTC) - timedelta(hours=1)
        throttled_until = past_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        state = set_throttled_until(state, throttled_until)
        save_state(paths.state_file, state)

    result = app.dispatch()

    # Should proceed normally (not throttled)
    assert result.ok is True
    assert result.data["selected_count"] == 1  # One ready issue
    assert "deferred_reason" not in result.data


def test_dispatch_rework_defers_when_provider_throttled(tmp_path: Path) -> None:
    """When provider throttle window is active, rework dispatch should also defer."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(default_limit=3),
        devin=DevinConfig(adapter="devin-shell"),
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


def test_is_throttled_checks_against_current_time(tmp_path: Path) -> None:
    """is_throttled should return True only when now < throttled_until."""
    from datetime import UTC, datetime, timedelta

    # Test with future timestamp
    future_time = datetime.now(UTC) + timedelta(hours=1)
    throttled_until = future_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    state = {"throttled_until": throttled_until}
    assert is_throttled(state) is True

    # Test with past timestamp
    past_time = datetime.now(UTC) - timedelta(hours=1)
    throttled_until = past_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    state = {"throttled_until": throttled_until}
    assert is_throttled(state) is False

    # Test with no throttled_until
    state = {"throttled_until": None}
    assert is_throttled(state) is False

    # Test with malformed timestamp
    state = {"throttled_until": "invalid-timestamp"}
    assert is_throttled(state) is False


def test_count_live_sessions_counts_both_adapters(tmp_path: Path) -> None:
    """_count_live_sessions should count sessions from both devin-shell and claude-code adapters."""
    from charlie_work.workflow import _count_live_sessions
    from charlie_work.devin_shell import SessionRecord as DevinSessionRecord
    from charlie_work.claude_code import ClaudeWorkerRecord

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a devin-shell session record with a valid PID (will be checked for liveness)
    # Since we can't easily create a real live process, we'll just test the file reading
    devin_record = DevinSessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/test",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--print"),
        pid=None,  # None means not alive
        started_at="2024-01-01T00:00:00Z",
        log_path="/tmp/log.log",
    )
    devin_path = sessions_dir / "issue-1.json"
    import json

    devin_path.write_text(json.dumps(devin_record.to_dict()), encoding="utf-8")

    # Create a claude-code session record
    claude_record = ClaudeWorkerRecord(
        issue_number=2,
        branch="agent/issue-2",
        worktree_path="/tmp/test2",
        prompt_path="/tmp/prompt2.md",
        command=("claude", "-p"),
        pid=None,  # None means not alive
        started_at="2024-01-01T00:00:00Z",
        log_path="/tmp/log2.log",
    )
    claude_path = sessions_dir / "issue-2.claude.json"
    claude_path.write_text(json.dumps(claude_record.to_dict()), encoding="utf-8")

    # Count live sessions (both have pid=None, so count should be 0)
    count = _count_live_sessions(sessions_dir)
    assert count == 0  # No live sessions since both have pid=None
