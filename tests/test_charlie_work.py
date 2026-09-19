from __future__ import annotations

import contextlib
import dataclasses
import errno
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from _dead_session_fixtures import _git
from _dispatch_fixtures import (
    _cross_repo_issue_body,
    _fail_if_launched,
)
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import (
    FakeGitHub,
    FakeGitHubWithChecks,
    FakeGitHubWithMissingRequired,
)
from _helpers import (
    EXAMPLES_DIR,
    _init_git_repo,
)
from _merge_tripwire_fixtures import (
    _ack_unauthorized_merge,
    _arm_unauthorized_merge_tripwire,
    _merged_worker_pr,
)
from _review_fixtures import (
    _approved_automerge,
    _dispatch_reviews_app,
    _fake_claude_worker_record,
    _make_dead_review_sidecar,
    _make_loop_app,
    _required_checks_config,
    _review_queue_carry_forward_app,
    _set_review_dispatched_state,
    _write_review_packet,
)
from _rework_dispatch_fixtures import (
    _init_repo_with_remote_inline,
    _wg,
)
from _worktree_fixtures import (
    _init_bare_remote_and_clone,
    _setup_completed_worktree,
)
from charlie_work import cli
from charlie_work import github as github_module
from charlie_work.config import (
    AutoMergeConfig,
    ClaudeCodeConfig,
    ConfigError,
    DevinConfig,
    DispatchConfig,
    LabelConfig,
    NotifyConfig,
    OrchestratorConfig,
    PostMortemConfig,
    ReviewConfig,
    ReviewDispatchConfig,
    RuntimeConfig,
    SignatureRule,
    TestAdequacyConfig,
    WatchdogConfig,
    WorkerRoleConfig,
    find_config_path,
    load_config,
)
from charlie_work.rescue_review import (
    LEGACY_VACUOUS_SUMMARY,
    _CAVEAT,
    extract_report_body,
    render_command,
    report_body_is_valid,
)
from charlie_work.github import (
    issue_numbers_mentioned_by_pr,
    label_names,
)
from charlie_work.instrumentation import query_events
from charlie_work.markdown_fence import fenced_block
from charlie_work.paths import (
    resolved_layout,
    runtime_paths,
)
from charlie_work.prompt_test_command import prompt_test_command_values
from charlie_work.prompts import render_prompt
from charlie_work.review_decision import ReviewDecision
from charlie_work.state import (
    PASSIVE_OPEN_STATUS,
    empty_state,
    is_throttled,
    load_state,
    save_state,
    state_lock,
)
import charlie_work.state as state_module
from charlie_work.verdict_parsing import REVIEW_SESSION_SUMMARY_HEADING
from charlie_work.workflow import (
    ORCHESTRATOR_COMMENT_MARKER,
    CommandResult,
    OrchestratorApp,
    _detect_and_handle_stalled_reviews,
    _parse_review_verdict_from_log,
    _render_required_changes_section,
    _summary_is_vacuous,
    sink_census,
    slugify,
)
from charlie_work.worktree import create_worktree
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.devin_shell import SessionRecord


def future_timestamp(*, days: int = 3650) -> str:
    """An ISO-8601 ``Z`` timestamp guaranteed to be in the future.

    Several state predicates decide behaviour by comparing a stored timestamp
    against ``datetime.now(UTC)`` -- ``is_reviewer_quota_exhausted`` (true only
    while ``throttled_until`` is future), ``is_reviewer_probe_ready``,
    ``is_throttled``. A test that hardcodes an absolute date to satisfy one of
    those preconditions is a time bomb: it passes until wall-clock time crosses
    the literal, then fails permanently, on every branch at once.

    That is not a theoretical concern -- a hardcoded ``2026-08-01T00:00:00Z``
    did exactly this, turning every open PR and main itself red at that instant
    and blocking the merge lane. Use this helper for any timestamp whose
    *futureness* is load-bearing.

    Timestamps that are merely round-tripped or compared for equality do not
    need this; an absolute literal is clearer there, and stable.
    """
    return (datetime.now(UTC) + timedelta(days=days)).isoformat().replace("+00:00", "Z")


def test_runner_scaling_config_parses_with_enabled_flag(tmp_path: Path) -> None:
    """RunnerScalingConfig parses with enabled=true and custom values."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
runner_scaling:
  enabled: true
  managed_root: "C:\\\\actions-runners"
  runner_dir_prefix: "jc-"
  runner_name_template: "jc-selfhost-{n}"
  package_zip: "C:\\\\packages\\\\runner.zip"
  min_runners: 2
  max_runners: 20
  ram_per_job_gb: 4.0
  min_free_ram_gb: 8.0
  max_host_cpu_pct: 90.0
  idle_scale_down_minutes: 30
  cooldown_minutes: 10
"""
    )
    config = load_config(config_file)
    assert config.runner_scaling.enabled is True
    assert config.runner_scaling.managed_root == "C:\\actions-runners"
    assert config.runner_scaling.runner_dir_prefix == "jc-"
    assert config.runner_scaling.runner_name_template == "jc-selfhost-{n}"
    assert config.runner_scaling.package_zip == "C:\\packages\\runner.zip"
    assert config.runner_scaling.min_runners == 2
    assert config.runner_scaling.max_runners == 20
    assert config.runner_scaling.ram_per_job_gb == 4.0
    assert config.runner_scaling.min_free_ram_gb == 8.0
    assert config.runner_scaling.max_host_cpu_pct == 90.0
    assert config.runner_scaling.idle_scale_down_minutes == 30
    assert config.runner_scaling.cooldown_minutes == 10


def test_runner_scaling_config_rejects_invalid_numeric_types(tmp_path: Path) -> None:
    """RunnerScalingConfig rejects non-numeric values for numeric fields."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
runner_scaling:
  enabled: true
  min_runners: "not-a-number"
"""
    )
    with pytest.raises(ConfigError, match="must be an int"):
        load_config(config_file)


def test_runner_scaling_config_rejects_invalid_string_types(tmp_path: Path) -> None:
    """RunnerScalingConfig rejects non-string values for string fields."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
runner_scaling:
  enabled: true
  managed_root: 123
"""
    )
    with pytest.raises(ConfigError, match="must be a string"):
        load_config(config_file)


def test_runner_scaling_config_rejects_invalid_boolean_type(tmp_path: Path) -> None:
    """RunnerScalingConfig rejects non-boolean values for enabled field."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
runner_scaling:
  enabled: "true"
"""
    )
    with pytest.raises(ConfigError, match="must be a bool"):
        load_config(config_file)


def test_auto_merge_config_rejects_stale_base_deadlock(tmp_path: Path) -> None:
    """Issue #368: require_current_base=True + update_open_prs='off' is a silent
    permanent merge deadlock, so it is rejected at config construction.
    """
    from charlie_work.config import AutoMergeConfig, ConfigError, OrchestratorConfig, load_config

    with pytest.raises(ConfigError, match="permanent merge deadlock"):
        AutoMergeConfig(require_current_base=True, update_open_prs="off")

    with pytest.raises(ConfigError, match="permanent merge deadlock"):
        AutoMergeConfig(require_current_base=True, update_open_prs=False)

    with pytest.raises(ConfigError, match="permanent merge deadlock"):
        OrchestratorConfig(
            auto_merge=AutoMergeConfig(require_current_base=True, update_open_prs="off")
        )

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  update_open_prs: off
"""
    )
    with pytest.raises(ConfigError, match="permanent merge deadlock"):
        load_config(config_file)

    # Coherent combinations load without error.
    assert AutoMergeConfig(require_current_base=False, update_open_prs="off")
    assert AutoMergeConfig(require_current_base=True, update_open_prs="next")
    assert AutoMergeConfig(require_current_base=True, update_open_prs="all")


def test_auto_merge_config_mergequeue_label_defaults_to_none(tmp_path: Path) -> None:
    """Aviator MergeQueue handoff (task #10) is off by default: the default
    AutoMergeConfig() must preserve today's self-merge behavior byte-for-byte."""
    from charlie_work.config import AutoMergeConfig

    assert AutoMergeConfig().mergequeue_label is None


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
            "issue_body_block": fenced_block("Body text", "md"),
            "branch_name": "agent/issue-123-fix-search",
            "issue_comments": "",
            "module_map": "",
            "attachment_budget": "",
            **prompt_test_command_values("", None),
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
            "issue_body_block": fenced_block("Body text", "md"),
            "branch_name": "agent/issue-123-fix-search",
            "issue_comments": "",
            "module_map": "",
            "attachment_budget": "",
            **prompt_test_command_values("", None),
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
        {
            "pr_number": 9,
            "pr_title": "t",
            "pr_url": "u",
            "issue_number": 1,
            "dispatch_note": "s",
            "dispatch_note_block": fenced_block("s", "md"),
            "required_changes_section": "",
            "branch_name": "agent/issue-1-t",
            **prompt_test_command_values("", None),
        },
        search_dirs=(override_dir,),
    )

    assert "Rework Task: PR #9" in prompt


def test_slugify_makes_branch_safe_slug() -> None:
    assert slugify("Fix: Search / Windows path!!!") == "fix-search-windows-path"


def test_label_names_accepts_gh_shape() -> None:
    issue = {"labels": [{"name": "automated-ready"}, {"name": "agent:in-progress"}]}

    assert label_names(issue) == {"automated-ready", "agent:in-progress"}


def test_issue_numbers_mentioned_by_pr_matches_issue_reference() -> None:
    pr = {
        "title": "fix(scope): reap sidecar files on session exit (issue #113)",
        "body": "This PR addresses issue #113. PR #181 is an unrelated refactor.",
    }

    assert issue_numbers_mentioned_by_pr(pr) == {113}


def test_issue_numbers_mentioned_by_pr_ignores_fenced_code_blocks() -> None:
    # A code sample that happens to contain the literal text must not count
    # as a real reference — advisory-only matching per the function's
    # contract, but obviously-wrong matches are worth stripping.
    pr = {
        "title": "docs: add example",
        "body": "Example:\n```\n# see issue #113 for context\n```\nNo real reference here.",
    }

    assert issue_numbers_mentioned_by_pr(pr) == set()


def test_issue_numbers_mentioned_by_pr_ignores_blockquoted_lines() -> None:
    # Quoted reply text (e.g. an email-style blockquote) must not count.
    pr = {
        "title": "chore: reply to review",
        "body": "> unlike issue #113, this one is fine\n\nAddressed the other comments.",
    }

    assert issue_numbers_mentioned_by_pr(pr) == set()


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


def test_coverage_probe_config_is_frozen() -> None:
    from charlie_work.config import CoverageProbeConfig
    from dataclasses import FrozenInstanceError

    config = CoverageProbeConfig()
    try:
        config.enabled = True  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except FrozenInstanceError:
        pass


def test_orchestrator_config_ctor_wires_coverage_probe_field() -> None:
    """OrchestratorConfig() carries a coverage_probe field with defaults."""
    from charlie_work.config import CoverageProbeConfig

    config = OrchestratorConfig()

    assert config.coverage_probe == CoverageProbeConfig()


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


def test_command_adapter_positional_placeholder_returns_error_record(tmp_path: Path) -> None:
    """AC #2: Command adapter positional placeholder {0} returns error record, not raise.

    The command adapter's render try/except catches IndexError for positional {0} templates.
    This test verifies that a config built directly (bypassing load_config) with a positional
    placeholder returns an error record instead of raising.

    Mutation to verify: remove IndexError from the except clause in adapters.py line 281,
    and the test will fail (it will raise IndexError instead of returning an error record).
    """
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

    # Config built directly (bypassing load_config) with a positional placeholder
    settings = AdapterSettings(
        adapter="command",
        dispatch_command=("echo", "{0}"),
        command_timeout_seconds=300,
    )

    results = dispatch_sessions(repo_root, manifest_path, results_path, settings, [request])

    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].error is not None
    # The error should mention the positional placeholder issue
    assert "0" in results[0].error or "positional" in results[0].error.lower()


def test_adapter_settings_launch_stagger_seconds_wired_from_dispatch_config(
    tmp_path: Path,
) -> None:
    """_adapter_settings() must read dispatch.launch_stagger_seconds -- the
    single point of enforcement between config and both dispatch lanes."""
    config = OrchestratorConfig(dispatch=DispatchConfig(launch_stagger_seconds=17))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    assert app._adapter_settings().launch_stagger_seconds == 17


def test_adapter_settings_api_branch_carries_api_worker_config(
    tmp_path: Path,
) -> None:
    """_adapter_settings() with devin.adapter='api' must carry the resolved
    ApiWorkerConfig into AdapterSettings and reuse the claude-code
    venv_source/worker_env. This is the single point that routes the provider
    registry into the api dispatch lane; a regression here silently breaks all
    api dispatches with no downstream error (the adapter raises an error
    result only at dispatch time, by which point the misconfiguration is
    already fleet-wide)."""
    from charlie_work.config import (
        ApiBudgetConfig,
        ApiProviderConfig,
        ApiWorkerConfig,
    )

    api_provider = ApiProviderConfig(
        base_url="https://api.moonshot.ai/anthropic",
        api_key_env="MOONSHOT_API_KEY",
        model="kimi-k3",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cached_input_usd_per_mtok=0.30,
    )
    api_cfg = ApiWorkerConfig(
        enabled=True,
        provider="kimi-k3",
        max_concurrent_sessions=2,
        providers={"kimi-k3": api_provider},
        budget=ApiBudgetConfig(max_usd_per_session=1.5),
    )
    claude_cfg = ClaudeCodeConfig(
        venv_source=".venv",
        worker_env={"PYTEST_XDIST_AUTO_NUM_WORKERS": "2"},
        tee_stream_json=True,
    )
    config = OrchestratorConfig(
        worker=WorkerRoleConfig(harness="api"),
        claude_code=claude_cfg,
        api_worker=api_cfg,
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    settings = app._adapter_settings()
    assert settings.adapter == "api"
    # The api branch must carry the resolved ApiWorkerConfig (not None) so the
    # api dispatch lane can resolve the provider. This is the regression guard:
    # test_dispatch_sessions_api_adapter_end_to_end constructs AdapterSettings
    # directly and bypasses this path, so without this test a broken wiring
    # (e.g. dropping the api_worker_config= line) would go undetected.
    assert settings.api_worker_config is api_cfg
    # The api branch reuses the claude-code venv_source/worker_env resolution.
    assert settings.venv_source == tmp_path / ".venv"
    assert settings.worker_env == {"PYTEST_XDIST_AUTO_NUM_WORKERS": "2"}
    # tee_stream_json is read from claude_code config (launch_api_worker
    # force-enables it internally, but AdapterSettings still carries the
    # configured value for the manifest/results surface).
    assert settings.tee_stream_json is True


def test_adapter_settings_non_api_branches_omit_api_worker_config(
    tmp_path: Path,
) -> None:
    """For devin-shell / claude-code / manual adapters, _adapter_settings()
    must set api_worker_config=None so a stale config block cannot leak into a
    non-api dispatch lane."""
    for adapter in ("devin-shell", "claude-code", "manual"):
        config = OrchestratorConfig(worker=WorkerRoleConfig(harness=adapter))
        paths = runtime_paths(tmp_path, config.runtime.state_dir)
        app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
        settings = app._adapter_settings()
        assert settings.adapter == adapter
        assert settings.api_worker_config is None, (
            f"adapter={adapter} must not carry api_worker_config"
        )


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

    # allow_failure=True now returns a structured result with an ok flag.
    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.value == [{"name": "Tests passed", "state": "FAILURE"}]


def test_github_run_allow_failure_returns_result_for_success(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"number": 123}',
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    result = github_module.GitHub(tmp_path).run(
        ["pr", "view", "123"], json_output=True, allow_failure=True
    )

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True
    assert result.value == {"number": 123}


def test_github_run_allow_failure_text_value_on_success(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="diff text",
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    result = github_module.GitHub(tmp_path).run(["pr", "diff", "123"], allow_failure=True)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True
    assert result.value == "diff text"


def test_pr_checks_fields_excludes_database_id() -> None:
    """Regression guard: gh pr checks --json does not support "databaseId".

    Adding it to PR_CHECKS_FIELDS (unlike gh run list --json, which does
    support it) makes the installed gh CLI exit non-zero with 'Unknown JSON
    field: "databaseId"'. Because pr_checks() uses allow_failure=True and
    treats a non-list result as "no checks", this silently returns [] from
    EVERY pr_checks() call — summarize_checks() then reports all required
    checks "missing" and merge_ready() computes can_merge=False for every PR,
    killing the entire auto-merge lane. This exact string broke the merge lane
    on 2026-07-10. The job id workflow.py needs is instead derived from "link"
    by pr_checks() via _job_id_from_link().
    """
    fields = github_module.PR_CHECKS_FIELDS.split(",")
    assert "databaseId" not in fields
    assert "link" in fields


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        (
            "https://github.com/OWNER/REPO/actions/runs/123456/job/789012",
            789012,
        ),
        (
            "https://github.com/OWNER/REPO/actions/runs/123456/job/789012/",
            789012,
        ),
        (
            "https://github.com/OWNER/REPO/actions/runs/123456/job/789012?check_suite_focus=true",
            789012,
        ),
        (
            "https://github.com/OWNER/REPO/actions/runs/123456/job/789012#step:3:1",
            789012,
        ),
        ("https://example.com/some/external/status-check", None),
        ("", None),
        (None, None),
    ],
)
def test_job_id_from_link(link, expected) -> None:
    assert github_module._job_id_from_link(link) == expected


def test_pr_checks_injects_database_id_from_link(monkeypatch, tmp_path: Path) -> None:
    """pr_checks() derives databaseId from link for Actions checks, None otherwise."""

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "name": "Tests passed",
                        "state": "SUCCESS",
                        "bucket": "pass",
                        "link": "https://github.com/OWNER/REPO/actions/runs/1/job/42",
                    },
                    {
                        "name": "external-status-check",
                        "state": "SUCCESS",
                        "bucket": "pass",
                        "link": "https://example.com/status",
                    },
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    checks = github_module.GitHub(tmp_path).pr_checks(123)

    assert checks[0]["databaseId"] == 42
    assert checks[1]["databaseId"] is None


def test_pr_checks_returns_empty_list_on_empty_success(monkeypatch, tmp_path: Path) -> None:
    """Empty successful gh pr checks --json response returns [], not None."""

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    checks = github_module.GitHub(tmp_path).pr_checks(123)

    assert checks == []


def test_pr_checks_returns_none_on_gh_command_failure(monkeypatch, tmp_path: Path) -> None:
    """gh pr checks command-level failure (Unknown JSON field) returns None."""

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr='Unknown JSON field: "databaseId"\nAvailable fields:\n  name\n  state\n  bucket\n  link',
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    checks = github_module.GitHub(tmp_path).pr_checks(123)

    assert checks is None


def test_check_run_annotations_returns_parsed_list_on_success(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(
                [{"path": "src/foo.py", "start_line": 42, "message": "line too long"}]
            ),
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    result = github_module.GitHub(tmp_path).check_run_annotations(999)

    assert result == [{"path": "src/foo.py", "start_line": 42, "message": "line too long"}]


def test_check_run_annotations_returns_empty_list_on_api_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #771: the annotations accessor must return a value (empty list),
    never raise, when the gh api call fails -- callers building required_changes
    from it must never crash the review() codepath on a transient GitHub error."""

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="HTTP 404: Not Found",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    result = github_module.GitHub(tmp_path).check_run_annotations(999)

    assert result == []


def test_pr_checks_returns_list_when_checks_fail(monkeypatch, tmp_path: Path) -> None:
    """gh pr checks exits non-zero but with JSON list (failing checks) -> list."""

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=2,
            stdout='[{"name": "Tests", "state": "FAILURE", "bucket": "fail", "link": ""}]',
            stderr="checks failed",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    checks = github_module.GitHub(tmp_path).pr_checks(123)

    assert checks == [
        {
            "name": "Tests",
            "state": "FAILURE",
            "bucket": "fail",
            "link": "",
            "databaseId": None,
            "runId": None,
        }
    ]


def test_validate_field_lists_passes_when_gh_lists_all_fields(monkeypatch, tmp_path: Path) -> None:
    """Startup self-check accepts field lists gh supports."""
    captured: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        captured.append(cmd)
        # Return a generic "all these fields are available" stderr.
        available_fields = [
            "number",
            "title",
            "name",
            "state",
            "bucket",
            "link",
            "url",
            "body",
            "labels",
            "headRefName",
            "baseRefName",
            "isCrossRepository",
            "mergeable",
            "headRefOid",
            "closedAt",
            "databaseId",
            "status",
            "createdAt",
            "headBranch",
            "assignees",
            "author",
            "updatedAt",
            "createdAt",
            "description",
            "color",
            "comments",
            "isDraft",
            "reviewDecision",
            "statusCheckRollup",
            "mergeStateStatus",
            "additions",
            "deletions",
        ]
        stderr = (
            'Unknown JSON field: "nonexistent"\nAvailable fields:\n  '
            + "\n  ".join(available_fields)
            + "\n"
        )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr=stderr,
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    github_module.GitHub(tmp_path).validate_field_lists()

    # Should have probed all 10 field-list constants.
    assert len(captured) == 10
    assert all(c[0] == "gh" for c in captured)


def test_validate_field_lists_fails_on_unsupported_field(monkeypatch, tmp_path: Path) -> None:
    """Startup self-check fails fast if a configured field is not supported by gh."""

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr='Unknown JSON field: "nonexistent"\nAvailable fields:\n  name\n  state\n  bucket',
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    with pytest.raises(ConfigError):
        github_module.GitHub(tmp_path).validate_field_lists()


def test_validate_field_lists_timeout_raises_config_error(monkeypatch, tmp_path: Path) -> None:
    """A hung `gh` probe during startup field-list validation surfaces as
    ConfigError rather than blocking boot forever. This probe runs once and
    is not retried, and it is bound by the configured gh_timeout_seconds."""
    call_count = 0
    captured_timeouts: list[float] = []

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        captured_timeouts.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path, runtime=RuntimeConfig(gh_timeout_seconds=45.0))
    with pytest.raises(ConfigError, match="timed out"):
        gh.validate_field_lists()

    # Fails fast on the first field list probed, not after trying all 10.
    assert call_count == 1
    # The timeout= kwarg passed to subprocess.run came from the configured
    # gh_timeout_seconds, not a hardcoded value.
    assert captured_timeouts == [45.0]


def test_github_merge_pr_argv_with_merge_flags(monkeypatch, tmp_path: Path) -> None:
    """Test that merge_flags are correctly passed to gh pr merge."""
    captured_args = []

    def fake_run(cmd, *args, **kwargs):
        captured_args.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    gh.merge_pr(123, "squash", admin=False, merge_flags=("--auto", "--subject"))

    assert len(captured_args) == 1
    args = captured_args[0]
    # Expected: ["gh", "pr", "merge", "123", "--auto", "--subject", "--squash"]
    assert args[0] == "gh"
    assert args[1:4] == ["pr", "merge", "123"]
    assert "--auto" in args
    assert "--subject" in args
    assert "--squash" in args
    # Verify merge_flags come before strategy flag
    auto_idx = args.index("--auto")
    subject_idx = args.index("--subject")
    squash_idx = args.index("--squash")
    assert auto_idx < squash_idx
    assert subject_idx < squash_idx


def test_github_merge_pr_argv_with_admin_flag(monkeypatch, tmp_path: Path) -> None:
    """Test that legacy admin flag is passed when merge_flags is empty."""
    captured_args = []

    def fake_run(cmd, *args, **kwargs):
        captured_args.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    gh.merge_pr(123, "squash", admin=True, merge_flags=())

    assert len(captured_args) == 1
    args = captured_args[0]
    # Expected: ["gh", "pr", "merge", "123", "--admin", "--squash"]
    assert args[0] == "gh"
    assert args[1:4] == ["pr", "merge", "123"]
    assert "--admin" in args
    assert "--squash" in args


def test_github_merge_pr_argv_merge_flags_precedence(monkeypatch, tmp_path: Path) -> None:
    """Test that merge_flags takes precedence over admin flag.

    Uses a legal non-managed flag (--auto) with admin=True to ensure the
    precedence logic is observable (the argv differs depending on which wins).
    """
    captured_args = []

    def fake_run(cmd, *args, **kwargs):
        captured_args.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    # Both admin=True and merge_flags set; merge_flags should win
    gh.merge_pr(123, "squash", admin=True, merge_flags=("--auto",))

    assert len(captured_args) == 1
    args = captured_args[0]
    # Expected: ["gh", "pr", "merge", "123", "--auto", "--squash"]
    # merge_flags wins, so --auto is present and --admin is NOT present
    assert "--auto" in args
    assert "--admin" not in args
    assert "--squash" in args
    # Verify exact order: merge_flags before strategy flag
    auto_idx = args.index("--auto")
    squash_idx = args.index("--squash")
    assert auto_idx < squash_idx


def test_github_merge_pr_flags_are_orchestrator_managed(monkeypatch, tmp_path: Path) -> None:
    """Invariant: every flag merge_pr appends is in ORCHESTRATOR_MANAGED_MERGE_FLAGS.

    This gate ensures that removing a flag from the constant derivation fails tests
    on BOTH the validation side (config.py) and the argv side (merge_pr), preventing
    the drift issue #107 where merge_pr could add flags without config validation
    rejecting them.
    """
    captured_args = []

    def fake_run(cmd, *args, **kwargs):
        captured_args.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    strategies = ["merge", "squash", "rebase"]

    for strategy in strategies:
        for admin in (False, True):
            captured_args.clear()
            gh.merge_pr(123, strategy, admin=admin, merge_flags=())

            assert len(captured_args) == 1
            args = captured_args[0]

            # Extract flags (skip "gh", "pr", "merge", and the PR number)
            flags = [arg for arg in args if arg.startswith("--")]

            # Every flag merge_pr appends must be in ORCHESTRATOR_MANAGED_MERGE_FLAGS
            for flag in flags:
                assert flag in github_module.ORCHESTRATOR_MANAGED_MERGE_FLAGS, (
                    f"Flag {flag} appended by merge_pr(strategy={strategy}, admin={admin}) "
                    f"is not in ORCHESTRATOR_MANAGED_MERGE_FLAGS"
                )


# --- Issue #361: merged_pr_list() cost (field scope + transient-gateway retry)


def test_github_merged_pr_list_uses_rest_pagination(monkeypatch, tmp_path: Path) -> None:
    """merged_pr_list() now uses the REST pulls endpoint instead of the
    GraphQL-backed `gh pr list --state merged`, avoiding expensive field sets
    such as `statusCheckRollup` (issue #361).
    """
    captured_args: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        captured_args.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    gh.merged_pr_list()

    assert len(captured_args) == 1
    args = captured_args[0]
    assert args[:2] == ["gh", "api"]
    assert "pulls" in args[2]
    assert "state=closed" in args[2]
    assert not any(c[:2] == ["gh", "pr"] and "merged" in c for c in captured_args)


def test_github_merged_pr_list_retries_on_transient_gateway_error(
    monkeypatch, tmp_path: Path
) -> None:
    """A transient 502/503/504 from the REST pulls endpoint retries
    in-pass (bounded) instead of immediately failing the whole fleet pass for
    that repo (issue #361). Succeeds on the 2nd attempt here.
    """
    call_count = 0
    sleeps: list[float] = []

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    gh = github_module.GitHub(tmp_path)
    result = gh.merged_pr_list()

    assert result == []
    assert call_count == 2
    assert len(sleeps) == 1


def test_github_merged_pr_list_gives_up_after_max_retries(monkeypatch, tmp_path: Path) -> None:
    """Persistent 502s must eventually raise GitHubError — never hang or retry
    forever — so the per-repo fleet-pass boundary can still catch it and move
    on to the next repo.
    """
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="HTTP 502: 502 Bad Gateway (https://api.github.com/graphql)",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = github_module.GitHub(tmp_path, runtime=RuntimeConfig(gh_max_retries=2))
    with pytest.raises(github_module.GitHubError):
        gh.merged_pr_list()

    assert call_count == 3


def test_github_merged_pr_list_does_not_retry_non_transient_error(
    monkeypatch, tmp_path: Path
) -> None:
    """A non-gateway error (e.g. bad credentials) must fail immediately rather
    than be swallowed into the transient-gateway retry loop.
    """
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="HTTP 401: Bad credentials"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    with pytest.raises(github_module.GitHubError):
        gh.merged_pr_list()

    assert call_count == 1


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


def test_fake_github_default_pr_head_is_indexed() -> None:
    """Issue #347: the default FakeGitHub fixture must index the PR head in commits.

    The default fixture assigns ``self.prs`` before ``self.commits`` exists, so
    the ``__setattr__`` hook's call to ``_record_pr_heads`` silently no-ops.
    This test ensures the PR head is indexed and that ``compare()`` can derive
    the merge-base from the commit graph.
    """
    gh = FakeGitHub()
    assert "sha-abc123" in gh.commits
    assert gh.commits["sha-abc123"]["parents"] == [{"sha": "base-sha"}]
    assert gh.base_head_sha in gh.commits
    result = gh.compare("main", "sha-abc123")
    assert result is not None
    assert result["merge_base_commit"]["sha"] == gh.base_head_sha


def test_fake_github_merge_base_criss_cross_is_deterministic() -> None:
    """Issue #347: _merge_base must be deterministic across PYTHONHASHSEED.

    In a criss-cross graph, the two best common ancestors are both minimal.
    A correct BFS distance and a deterministic tie-break must produce the same
    merge base regardless of hash randomization.

    Issue #1292: the harness was previously flaky because (a) it imported
    ``FakeGitHub`` via ``test_charlie_work`` -- dragging the entire 50k-line
    test module (pytest, yaml, every helper fixture) into a subprocess whose
    import can fail transiently under full-suite parallel-run contention --
    and (b) it collected results into a ``set``, discarding the seed-to-result
    mapping so a failure was unactionable. The harness now imports
    ``FakeGitHub`` directly from ``_fakes_github`` (the lightweight module that
    defines it) and records the seed that produced each result so any future
    non-determinism is immediately attributable.
    """
    tests_dir = Path(__file__).parent
    repo_root = tests_dir.parent
    code = "\n".join(
        [
            "import os, sys",
            "sys.path.insert(0, sys.argv[1])",
            "from _fakes_github import FakeGitHub",
            "gh = FakeGitHub()",
            "gh.commits = {",
            '    "R": {"parents": []},',
            '    "A1": {"parents": [{"sha": "R"}]},',
            '    "B1": {"parents": [{"sha": "R"}]},',
            '    "A2": {"parents": [{"sha": "A1"}, {"sha": "B1"}]},',
            '    "B2": {"parents": [{"sha": "B1"}, {"sha": "A1"}]},',
            "}",
            'gh.base_head_sha = "A2"',
            'print(gh._merge_base("A2", "B2"))',
        ]
    )
    # Record the seed that produced each result so a failure is actionable.
    results: dict[str, str] = {}
    for seed in range(5):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        proc = subprocess.run(
            [sys.executable, "-c", code, str(tests_dir)],
            env=env,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"subprocess for PYTHONHASHSEED={seed} exited {proc.returncode}.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        results[str(seed)] = proc.stdout.strip()
    distinct = set(results.values())
    assert len(distinct) == 1, f"merge_base varied across hash seeds (seed -> result): {results}"


def test_base_fake_github_merged_prs_for_issue_returns_typed_result() -> None:
    """Issue #882 regression guard.

    Production ``GitHubCLI.merged_prs_for_issue`` always returns a
    ``MergedPRSearchResult`` carrying ``.ok``. The base ``FakeGitHub`` used to
    return a plain ``list`` with no ``.ok`` attribute, so it silently disagreed
    with the real thing -- only inert because the sole consumer reads
    defensively via ``getattr(merged_prs, "ok", True)``. A future caller that
    reads ``.ok`` directly would pass tests against this fake and raise
    ``AttributeError`` in production.

    Pin the typed shape on the base fake so the fake and the real thing agree.
    """
    fake_gh = FakeGitHub()
    # The default PR ships OPEN; flip it to MERGED so it matches.
    fake_gh.prs[0]["state"] = "MERGED"

    result = fake_gh.merged_prs_for_issue(123, "agent/issue-")

    assert isinstance(result, github_module.MergedPRSearchResult)
    assert result.ok is True
    assert [pr["number"] for pr in result] == [456]
    # An empty search must still carry the typed shape (ok=True, not a bare []).
    empty = fake_gh.merged_prs_for_issue(999, "agent/issue-")
    assert isinstance(empty, github_module.MergedPRSearchResult)
    assert empty.ok is True
    assert list(empty) == []


def test_app_prompts_dir_override_wins_for_worker_prompt(tmp_path: Path) -> None:
    override_dir = tmp_path / "orchestrator-prompts"
    override_dir.mkdir()
    # The override must carry the no-merge contract markers (issue #714),
    # the conventional-commit title instruction (issue #715), the
    # execution-contract escalation trigger (issue #717), and the widened
    # containment clause markers (issue #1010):
    # _write_worker_prompt's post-render guards reject a flat override that
    # drops any of these.
    (override_dir / "worker.md").write_text(
        "REPO-LOCAL #$issue_number\n\n"
        "## No-merge contract\n\n"
        "Your deliverable ENDS at pushing the branch and opening the PR.\n\n"
        "## PR requirements\n\n"
        "- Title format: Conventional-Commits format (`type(scope): description`).\n\n"
        "**Execution contract (self-detect from your diff):** the default is "
        "the targeted command. Only if the diff changes any public function "
        "signature/return shape, run the **FULL suite** locally at the final "
        "head before pushing.\n\n"
        "**Containment:** All file edits happen in the assigned worktree; "
        "never modify any path outside the assigned worktree root.\n",
        encoding="utf-8",
    )
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir="orchestrator-prompts"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    app.gh.prs[0]["state"] = "CLOSED"
    app.dispatch(limit=1)

    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    assert prompt_path.read_text(encoding="utf-8") == (
        "REPO-LOCAL #123\n\n"
        "## No-merge contract\n\n"
        "Your deliverable ENDS at pushing the branch and opening the PR.\n\n"
        "## PR requirements\n\n"
        "- Title format: Conventional-Commits format (`type(scope): description`).\n\n"
        "**Execution contract (self-detect from your diff):** the default is "
        "the targeted command. Only if the diff changes any public function "
        "signature/return shape, run the **FULL suite** locally at the final "
        "head before pushing.\n\n"
        "**Containment:** All file edits happen in the assigned worktree; "
        "never modify any path outside the assigned worktree root.\n"
    )


def test_command_dispatch_labels_only_successful_launches(tmp_path: Path) -> None:
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
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
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
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(7)")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    assert result.ok is False
    assert result.data["selected_count"] == 0
    assert result.data["failed_count"] == 1
    assert result.data["dispatch_results"][0]["returncode"] == 7
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatch_failed"


def test_roll_call_json_dependencies_schema(tmp_path: Path) -> None:
    """Test that roll-call --json includes dependencies payload with correct schema per issue #152."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake GitHub with dependency markers
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 100,
            "title": "issue-with-deps",
            "url": "https://github.com/test/repo/issues/100",
            "body": "Blocked by #200, #300",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-01T00:00:00Z",
            "updatedAt": "2026-07-01T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 200,
            "title": "blocker-1",
            "url": "https://github.com/test/repo/issues/200",
            "body": "Blocker issue",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-02T00:00:00Z",
            "updatedAt": "2026-07-02T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 300,
            "title": "blocker-2",
            "url": "https://github.com/test/repo/issues/300",
            "body": "Another blocker",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-03T00:00:00Z",
            "updatedAt": "2026-07-03T00:00:00Z",
            "state": "OPEN",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Run status with JSON output
    result = app.status()

    assert result.ok is True
    roll_call_data = result.data

    # Verify dependencies payload exists and has correct schema
    assert "issues" in roll_call_data
    issues_by_number = {issue["number"]: issue for issue in roll_call_data["issues"]}

    # Check issue 100 has dependencies
    issue_100 = issues_by_number[100]
    assert "dependencies" in issue_100
    deps = issue_100["dependencies"]
    assert "declared" in deps
    assert "open" in deps
    assert isinstance(deps["declared"], list)
    assert isinstance(deps["open"], list)
    # Issue 100 declares blockers 200 and 300
    assert set(deps["declared"]) == {200, 300}
    # Both blockers are open, so open blockers should match declared
    assert set(deps["open"]) == {200, 300}

    # Check blocker issues have empty dependencies
    issue_200 = issues_by_number[200]
    assert "dependencies" in issue_200
    assert issue_200["dependencies"]["declared"] == []
    assert issue_200["dependencies"]["open"] == []


def test_check_carry_forward_tier1_whitespace_collision_bypasses_tier2(
    tmp_path: Path,
) -> None:
    """Issue #1187 (audit #634 rework): ``git patch-id --stable`` strips
    leading whitespace from ``+``/``-`` content lines, so two diffs that
    differ ONLY in indentation depth produce the identical patch-id.  The
    tier-1 fast path in ``_check_carry_forward`` previously matched on that
    hash and returned ``"patch-id"`` immediately — it never consulted the
    tier-2 line-content signature, which DOES preserve whitespace and WOULD
    distinguish the two diffs.

    In Python, an indentation-only change can alter control flow (e.g. moving
    a ``return`` into or out of an ``if`` block).  Carrying forward an
    approved verdict across such a change without review is a review-gate
    bypass.  After the fix, the tier-1 patch-id match is no longer
    sufficient: the tier-2 line-content signature is also validated, and
    when it differs (a whitespace-only change that patch-id collapsed) the
    carry-forward is REFUSED — the verdict is reported stale.
    """
    from charlie_work.janitor import _calculate_patch_id, _diff_content_signature

    # Diff reviewed at approval time: indent ``return True`` to 8 spaces.
    reviewed_diff = (
        "diff --git a/mod.py b/mod.py\n"
        "index 0c54d1a..8e25a7e 100644\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return True\n"
        "+        return True\n"
    )
    # Live diff: same logical change but indented to 12 spaces instead of 8.
    # ``git patch-id --stable`` strips leading whitespace, so both produce
    # the same hash.  The tier-2 signature preserves whitespace verbatim and
    # differs.
    live_diff = (
        "diff --git a/mod.py b/mod.py\n"
        "index 0c54d1a..f80ba40 100644\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return True\n"
        "+            return True\n"
    )

    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    live_patch_id = _calculate_patch_id(live_diff)
    assert reviewed_patch_id == live_patch_id, (
        "two diffs differing only in indentation depth must produce the "
        "same git patch-id --stable (the vulnerability's precondition)"
    )
    assert reviewed_patch_id != "", "sanity: patch-id must be non-empty"

    reviewed_sig = _diff_content_signature(reviewed_diff)
    live_sig = _diff_content_signature(live_diff)
    assert reviewed_sig.changed_lines != live_sig.changed_lines, (
        "tier-2 signatures must differ — tier-2 preserves whitespace and "
        "WOULD catch the indentation change that tier-1 misses"
    )
    assert reviewed_sig.changed_files == live_sig.changed_files

    old_head = "sha-reviewed-head"
    new_head = "sha-reindented-head"
    pr_number = 456
    issue_number = 123

    prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": new_head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _review_queue_carry_forward_app(tmp_path, prs=prs)
    fake_gh = app.gh
    fake_gh.diffs[pr_number] = live_diff

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": reviewed_patch_id,
            "reviewed_changed_lines": list(reviewed_sig.changed_lines),
            "reviewed_changed_files": sorted(reviewed_sig.changed_files),
            "reviewed_has_binary": reviewed_sig.has_binary,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] != [], (
        "the verdict must NOT be carried forward across an "
        "indentation-only change — tier-2 signature differs, so the "
        "verdict is reported stale for re-review"
    )

    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == old_head, (
        "the approved verdict must NOT be carried forward to the "
        "reindented head — the review-gate bypass is closed"
    )
    assert decision.get("carry_forward_tier") != "patch-id", (
        "carry-forward must not be via tier-1 (patch-id) when the tier-2 "
        "line-content signature differs — the whitespace collision is "
        "detected and the verdict is refused"
    )


def test_update_approval_head_records_event_for_every_tier(tmp_path: Path) -> None:
    """Issue #638: ``_update_approval_head`` must record a carry-forward event
    itself, so a new call site cannot forget it. The event kind is
    tier-dependent so the three mechanisms (patch-id, line-content,
    verified-sync) stay separately auditable, and exactly one event is
    emitted per call."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_number = 456
    issue_number = 123
    decision_dir = paths.prs / f"pr-{pr_number}"
    decision_dir.mkdir(parents=True)

    def _run(tier: str, old_head: str, new_head: str) -> dict[str, Any]:
        decision = {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": "pid-xyz",
        }
        (decision_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")
        app._update_approval_head(
            pr_number,
            decision,
            new_head,
            old_head=old_head,
            issue_number=issue_number,
            tier=tier,
        )
        return load_state(paths.state_file)

    expected_kind = {
        "patch-id": "verdict_carried_forward_clean_rebase",
        "line-content": "verdict_carried_forward_line_content",
        "verified-sync": "verdict_carried_forward_verified_sync",
    }

    for tier, kind in expected_kind.items():
        state = _run(tier, f"old-{tier}", f"new-{tier}")
        carry_events = [e for e in state["events"] if e["kind"] == kind]
        assert len(carry_events) == 1, f"tier {tier}: expected 1 {kind} event"
        payload = carry_events[0]["payload"]
        assert payload["pr_number"] == pr_number
        assert payload["issue_number"] == issue_number
        assert payload["old_reviewed_head_sha"] == f"old-{tier}"
        assert payload["new_head_sha"] == f"new-{tier}"
        assert payload["carry_forward_tier"] == tier
        assert payload["carried_forward_from"] == [f"old-{tier}"]


def test_update_open_agent_prs_front_of_train_records_verified_sync_event(
    tmp_path: Path,
) -> None:
    """Issue #638: the front-of-train ``_update_open_agent_prs`` carry-forward
    (a previously-silent ``verified-sync`` call site) must record a
    ``verdict_carried_forward_verified_sync`` event."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "baseRefName": "main",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.record_review(789, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    for idx, pr_number in enumerate((456, 789)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    result = app.merge_ready(456, merge=True)
    assert result.ok is True
    assert result.data["merged"] is True

    update_results = result.data["update_open_prs_results"]
    assert update_results is not None
    assert len(update_results) == 1
    assert update_results[0]["pr_number"] == 789
    assert update_results[0]["updated"] is True

    state = load_state(paths.state_file)
    sync_events = [
        e for e in state["events"] if e["kind"] == "verdict_carried_forward_verified_sync"
    ]
    assert len(sync_events) == 1
    payload = sync_events[0]["payload"]
    assert payload["pr_number"] == 789
    assert payload["issue_number"] == 124
    assert payload["carry_forward_tier"] == "verified-sync"


def test_detect_and_handle_stalled_reviews_removes_review_checkout(tmp_path: Path) -> None:
    """Issue #397: a reaped stale-claim review must tear down that PR's
    isolated review checkout, not just free the state.json claim."""
    from datetime import timedelta

    from charlie_work.workflow import _detect_and_handle_stalled_reviews
    from charlie_work.worktree import create_review_checkout

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    checkout = create_review_checkout(repo_root, 100, head_sha, reviews_dir=reviews_dir)
    assert checkout.path.exists()

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"] = {
            "number": 100,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old_dispatched,
            "reviewer_pid": 999999999,  # not a real live pid
            "reviewer_process_start_time": 1.0,
        }
        save_state(state_file, state)

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert any(entry.get("pr") == 100 for entry in stalled)
    assert not checkout.path.exists()
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(checkout.path) not in result.stdout


def test_detect_and_handle_stalled_reviews_backs_off_on_provider_throttle_in_log(
    tmp_path: Path,
) -> None:
    """A dead reviewer whose own log shows a provider throttle signature
    (e.g. Claude Code CLI's "You've hit your session limit ...") must set the
    global reviewer-quota cooldown and roll back the claim, not mark it
    review_dispatch_failed. Marking it failed lets the next dispatch_reviews
    pass relaunch straight into the same limit -- job-cannon PRs #1342,
    #1343, #1344, #1346 hot-looped for 5.5-20+ hours this way on 2026-07-21
    before this reap path also learned to log-tail classify."""
    from datetime import timedelta

    from charlie_work.state import load_state as _load_state

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text(
        "You've hit your session limit · resets 4:40pm (America/Los_Angeles)\n",
        encoding="utf-8",
    )
    old_started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-10-fix",
        "worktree_path": str(tmp_path / "worktrees" / "issue-100"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,  # not a real live pid
        "started_at": old_started,
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    (reviews_dir / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"] = {
            "number": 100,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old_started,
            "reviewer_pid": 999999999,
            "reviewer_process_start_time": 1.0,
        }
        save_state(state_file, state)

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert any(
        entry.get("pr") == 100 and entry.get("reason") == "provider_throttled" for entry in stalled
    )

    state = _load_state(state_file)
    pr_state = state["prs"]["100"]
    # Rolled back, not failed -- the claim is immediately re-dispatchable
    # once the reviewer-quota gate clears.
    assert pr_state.get("review_dispatch_status") is None
    assert pr_state.get("reviewer_pid") is None

    quota = state.get("reviewer_quota", {})
    assert quota.get("throttled_until")
    assert quota.get("probe_after")
    assert any(
        event.get("kind") == "review_dispatch_stalled"
        and event.get("payload", {}).get("reason") == "provider_throttled"
        for event in state.get("events", [])
    )


def test_detect_and_handle_stalled_reviews_suppresses_backoff_after_probe_recovery(
    tmp_path: Path,
) -> None:
    """Issue #662: a dead reviewer whose log shows a throttle signature must
    NOT re-poison reviewer_quota when a green flat-interval probe already
    cleared the throttle AFTER the reviewer died. The throttle signature in a
    dead session's log tail is frozen at death time; re-applying backoff from
    it would anchor throttled_until/probe_after to "now" rather than the
    original death time, delaying the next dispatch by up to one probe cycle
    even though the quota window is open. The claim is still rolled back and
    the sidecar reaped -- the reviewer is dead regardless, and with the quota
    recovered the PR should be immediately re-dispatchable.
    """
    from datetime import timedelta

    from charlie_work.state import load_state as _load_state

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text(
        "You've hit your session limit · resets 4:40pm (America/Los_Angeles)\n",
        encoding="utf-8",
    )
    # Pin the log mtime to 10 minutes ago -- the reviewer died then, and a
    # green probe cleared the quota 1 minute ago (after death).
    death_dt = datetime.now(UTC) - timedelta(minutes=10)
    cleared_dt = datetime.now(UTC) - timedelta(minutes=1)
    os.utime(log_path, (death_dt.timestamp(), death_dt.timestamp()))

    old_started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-10-fix",
        "worktree_path": str(tmp_path / "worktrees" / "issue-100"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,
        "started_at": old_started,
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    (reviews_dir / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    cleared_iso = cleared_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"] = {
            "number": 100,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old_started,
            "reviewer_pid": 999999999,
            "reviewer_process_start_time": 1.0,
        }
        # Simulate a green probe that cleared the quota after the reviewer died.
        state["reviewer_quota"] = {
            "consecutive_probe_failures": 0,
            "last_probe_cleared_at": cleared_iso,
        }
        save_state(state_file, state)

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert any(
        entry.get("pr") == 100 and entry.get("reason") == "provider_throttled" for entry in stalled
    )

    state = _load_state(state_file)
    pr_state = state["prs"]["100"]
    # Rolled back, not failed -- immediately re-dispatchable (quota is open).
    assert pr_state.get("review_dispatch_status") is None
    assert pr_state.get("reviewer_pid") is None

    quota = state.get("reviewer_quota", {})
    # Backoff was suppressed: no throttled_until / probe_after re-poisoning.
    assert not quota.get("throttled_until")
    assert not quota.get("probe_after")
    # The recovery marker must survive the sweep unchanged.
    assert quota.get("last_probe_cleared_at") == cleared_iso

    # The event must record the suppression for observability.
    assert any(
        event.get("kind") == "review_dispatch_stalled"
        and event.get("payload", {}).get("reason") == "provider_throttled"
        and event.get("payload", {}).get("backoff_suppressed") is True
        for event in state.get("events", [])
    )
    # The sidecar was reaped (one-shot, no re-entry next sweep).
    assert not (reviews_dir / "issue-100.claude.json").exists()


def test_detect_and_handle_stalled_reviews_applies_backoff_when_probe_predates_death(
    tmp_path: Path,
) -> None:
    """Issue #662 control: when the green probe cleared BEFORE the reviewer
    died, the throttle signature is fresh evidence the quota closed again and
    backoff must still be applied. Only a probe recovery that post-dates the
    death suppresses backoff.
    """
    from datetime import timedelta

    from charlie_work.state import load_state as _load_state

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text(
        "You've hit your session limit · resets 4:40pm (America/Los_Angeles)\n",
        encoding="utf-8",
    )
    # The reviewer died 1 minute ago; the probe cleared 10 minutes ago (before
    # death) -- the throttle signature is fresh, backoff must engage.
    death_dt = datetime.now(UTC) - timedelta(minutes=1)
    cleared_dt = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (death_dt.timestamp(), death_dt.timestamp()))

    old_started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-10-fix",
        "worktree_path": str(tmp_path / "worktrees" / "issue-100"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,
        "started_at": old_started,
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    (reviews_dir / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    cleared_iso = cleared_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"] = {
            "number": 100,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old_started,
            "reviewer_pid": 999999999,
            "reviewer_process_start_time": 1.0,
        }
        state["reviewer_quota"] = {
            "consecutive_probe_failures": 0,
            "last_probe_cleared_at": cleared_iso,
        }
        save_state(state_file, state)

    _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    state = _load_state(state_file)
    quota = state.get("reviewer_quota", {})
    # Backoff applied: throttled_until / probe_after are set.
    assert quota.get("throttled_until")
    assert quota.get("probe_after")
    assert any(
        event.get("kind") == "review_dispatch_stalled"
        and event.get("payload", {}).get("reason") == "provider_throttled"
        and event.get("payload", {}).get("backoff_suppressed") is False
        for event in state.get("events", [])
    )


def test_detect_and_handle_stalled_reviews_reaps_unclaimed_reviewing_packet(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #487: a reviewing PR that was never claimed/dispatched is reaped
    and then re-dispatched once its packet is past the stale-claim timeout."""
    from datetime import timedelta

    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")

    # Age the packet so the unclaimed safety net triggers on the next pass.
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-100"
    prompt_path = pr_dir / "review-prompt.md"
    old_mtime = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
    os.utime(prompt_path, (old_mtime, old_mtime))

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "status": "reviewing",
            "prompt_path": str(prompt_path),
            "decision_path": str(pr_dir / "review-decision.json"),
        }
        save_state(app.paths.state_file, state)

    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(100, "agent/issue-10-fix")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["launched_count"] == 1
    assert launched == [100]
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert any(
        event.get("kind") == "review_dispatch_stalled"
        and event.get("payload", {}).get("status") == "unclaimed"
        for event in state.get("events", [])
    )


def test_stale_claim_recovery_skipped_logs_when_prompt_path_missing_from_state(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #708: a reviewing PR whose prompt_path is missing from state must
    emit a review_stale_claim_recovery_skipped event instead of silently moving
    on, so a stuck-PR investigation can distinguish "recovery gave up" from
    "recovery was not needed"."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)

    # reviewing PR with NO review_dispatch_status and NO prompt_path -- the
    # stale-claim recovery path's first skip branch.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "status": "reviewing",
        }
        save_state(app.paths.state_file, state)

    # No launch should happen: recovery gave up before reaching dispatch.
    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(100, "agent/issue-10-fix")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert launched == []
    state = load_state(app.paths.state_file)
    # The PR was not reaped -- it stays in its stuck reviewing state with no
    # dispatch claim, exactly as before #708. The fix is observability, not a
    # behavior change to the recovery decision itself.
    assert state["prs"]["100"].get("review_dispatch_status") is None

    skip_events = query_events(app.paths.state_file, kind="review_stale_claim_recovery_skipped")
    assert len(skip_events) == 1
    payload = skip_events[0]["payload"]
    assert payload["pr_number"] == 100
    assert payload["reason"] == "prompt_path missing from state"
    assert skip_events[0]["level"] == "warning"


def test_stale_claim_recovery_skipped_logs_when_prompt_path_file_gone(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #708: a reviewing PR whose prompt_path points at a file that no
    longer exists on disk must emit a review_stale_claim_recovery_skipped event
    (with the path) instead of silently moving on."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)

    gone_prompt = tmp_path / "deleted-review-prompt.md"

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "status": "reviewing",
            "prompt_path": str(gone_prompt),
        }
        save_state(app.paths.state_file, state)

    assert not gone_prompt.exists()

    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(100, "agent/issue-10-fix")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert launched == []
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"].get("review_dispatch_status") is None

    skip_events = query_events(app.paths.state_file, kind="review_stale_claim_recovery_skipped")
    assert len(skip_events) == 1
    payload = skip_events[0]["payload"]
    assert payload["pr_number"] == 100
    assert payload["reason"] == "prompt_path file does not exist on disk"
    assert payload["prompt_path"] == str(gone_prompt)
    assert skip_events[0]["level"] == "warning"


def test_stale_claim_recovery_skipped_logs_when_decision_already_recorded(
    tmp_path: Path,
) -> None:
    """Issue #734: a reviewing PR whose decision_path already holds a verdict
    (e.g. ``request_changes``) is silently passed over by stale-claim recovery
    on every pass -- the verdict was never acted upon, but without an event
    nobody can tell recovery considered the PR and declined. This is the second
    of the three silent skip paths identified in #734."""
    from datetime import timedelta

    from charlie_work.workflow import _detect_and_handle_stalled_reviews

    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    # Create a valid prompt_path on disk and a decision file with a verdict.
    pr_dir = tmp_path / "prs" / "pr-100"
    pr_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = pr_dir / "review-prompt.md"
    prompt_path.write_text("review prompt", encoding="utf-8")
    decision_path = pr_dir / "review-decision.json"
    decision_path.write_text(json.dumps({"decision": "request_changes"}), encoding="utf-8")

    # Age the packet so the stale-claim timeout is satisfied -- the skip must
    # come from the decision gate, not from the packet-age gate.
    old_mtime = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
    os.utime(prompt_path, (old_mtime, old_mtime))

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "status": "reviewing",
            "prompt_path": str(prompt_path),
            "decision_path": str(decision_path),
        }
        save_state(state_file, state)

    _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    state = load_state(state_file)
    # The PR was not reaped -- recovery declined because a verdict exists.
    assert state["prs"]["100"].get("review_dispatch_status") is None

    skip_events = query_events(state_file, kind="review_stale_claim_recovery_skipped")
    assert len(skip_events) == 1
    payload = skip_events[0]["payload"]
    assert payload["pr_number"] == 100
    assert payload["reason"] == "decision_already_recorded"
    assert payload["decision"] == "request_changes"
    assert skip_events[0]["level"] == "warning"


def test_stale_claim_recovery_skipped_logs_when_packet_not_stale(
    tmp_path: Path,
) -> None:
    """Issue #734: a reviewing PR whose packet is not yet past the stale-claim
    timeout is silently skipped on every pass until it becomes stale. This is
    the third of the three silent skip paths identified in #734. The event is
    info-level (not warning) because this is expected flow control -- the
    packet simply is not old enough yet -- unlike the other two skips which
    indicate a PR recovery cannot help."""
    from charlie_work.workflow import _detect_and_handle_stalled_reviews

    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    # Create a valid prompt_path on disk with NO decision file (decision_value
    # defaults to "missing", which passes the decision gate). The packet is
    # fresh -- not aged -- so the stale-claim timeout is not met.
    pr_dir = tmp_path / "prs" / "pr-100"
    pr_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = pr_dir / "review-prompt.md"
    prompt_path.write_text("review prompt", encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "status": "reviewing",
            "prompt_path": str(prompt_path),
            "decision_path": str(pr_dir / "review-decision.json"),
        }
        save_state(state_file, state)

    _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    state = load_state(state_file)
    assert state["prs"]["100"].get("review_dispatch_status") is None

    skip_events = query_events(state_file, kind="review_stale_claim_recovery_skipped")
    assert len(skip_events) == 1
    payload = skip_events[0]["payload"]
    assert payload["pr_number"] == 100
    assert payload["reason"] == "packet_not_stale"
    assert "packet_age" in payload
    assert skip_events[0]["level"] == "info"


def test_loop_dispatches_reviews_and_evaluates_merge(monkeypatch, tmp_path: Path) -> None:
    """Issue #370: loop() runs dispatch_reviews() and the per-PR merge lane uses the verdict."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        auto_merge=AutoMergeConfig(
            enabled=True,
            strategy="squash",
            delete_branch=True,
            require_approved_review=True,
            required_checks=(),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    prs = [
        {
            "number": 456,
            "title": "Fix #456",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-456-fix",
            "baseRefName": "main",
            "headRefOid": "sha-456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #456",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(tmp_path, 456, "sha-456")

    def fake_launch(
        issue_number: int, branch: str, prompt_text: str, **kwargs: Any
    ) -> ClaudeWorkerRecord:
        # Simulate the reviewer agent writing an approved verdict.
        pr_dir = app.paths.prs / f"pr-{issue_number}"
        decision = {
            "decision": "approved",
            "summary": "lgtm",
            "required_changes": [],
            "reviewed_head_sha": "sha-456",
            "reviewed_patch_id": "",
            "reviewed_at": "2026-07-06T12:00:00Z",
            "pr_number": issue_number,
            "issue_number": 456,
            "escalated": False,
        }
        (pr_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")
        return _fake_claude_worker_record(issue_number, branch)

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.loop(merge=False)

    assert result.ok is True
    assert result.data["dispatch_reviews"]["launched_count"] == 1
    assert len(result.data["merges"]) == 1
    assert result.data["merges"][0]["can_merge"] is True


def test_loop_checks_unavailable_review_lands_in_errors_bucket(tmp_path: Path) -> None:
    """A PR whose review is blocked by checks unavailable must be recorded as an error, not reviewed or merged."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubWithChecksUnavailable(FakeGitHub):
        def pr_checks(self, number: int):
            return None

    fake_gh = FakeGitHubWithChecksUnavailable()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.loop(merge=False)

    assert result.ok is False
    assert result.data["reviews"] == []
    assert result.data["merges"] == []
    assert len(result.data["errors"]) == 1
    assert result.data["errors"][0]["pr"] == 456
    assert "checks unavailable" in result.data["errors"][0]["error"].lower()


def test_loop_zero_checks_pr_does_not_land_in_errors_bucket(tmp_path: Path) -> None:
    """Issue #846: a PR with zero checks (pr_checks() == []) must not be
    counted as a loop-pass error the way checks_unavailable (pr_checks() is
    None) is. This mirrors test_loop_checks_unavailable_review_lands_in_errors_bucket
    above but exercises the other outcome the client boundary must now
    produce: [] means "genuinely no checks yet" (e.g. a conflicted/dirty PR
    CI never ran on), which is a normal review outcome, not an
    infrastructure failure.

    This is a downstream-contract characterization test: it stubs
    GitHub.pr_checks() directly, the same way the sibling test above does, so
    it does not exercise GitHubClient.pr_checks()'s own gh-subprocess
    disambiguation logic -- that regression coverage lives in
    tests/test_github.py (test_pr_checks_zero_checks_returns_empty_list_not_none
    and friends). What this test proves is that once the client returns []
    (as it now correctly does for a zero-check PR), workflow.py's loop() does
    not misclassify it as an error the way it misclassifies None.
    """
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubWithZeroChecks(FakeGitHub):
        def pr_checks(self, number: int):
            return []

    fake_gh = FakeGitHubWithZeroChecks()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.loop(merge=False)

    assert result.data["errors"] == []
    assert result.data["merges"] == []
    assert len(result.data["reviews"]) == 1
    assert result.data["reviews"][0]["pr"] == 456


def test_detect_unauthorized_merges_flags_worker_self_merge(tmp_path: Path) -> None:
    """A merged worker branch without an approved review decision is flagged as a possible self-merge (issue #502)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    class FakeGitHubWithMergedWorkerPR(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.prs = [
                {
                    "number": 501,
                    "title": "fix: worker self-merge",
                    "url": "https://example.test/pull/501",
                    "headRefName": "agent/issue-494-fix",
                    "baseRefName": "main",
                    "headRefOid": "sha-501",
                    "state": "MERGED",
                    "isCrossRepository": False,
                    "body": "Closes #494",
                    "labels": [],
                },
                {
                    "number": 502,
                    "title": "fix: approved merge",
                    "url": "https://example.test/pull/502",
                    "headRefName": "agent/issue-495-fix",
                    "baseRefName": "main",
                    "headRefOid": "sha-502",
                    "state": "MERGED",
                    "isCrossRepository": False,
                    "body": "Closes #495",
                    "labels": [],
                },
            ]

    fake_gh = FakeGitHubWithMergedWorkerPR()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # PR 502 has an approved review decision on the merged head
    pr_dir = paths.prs / "pr-502"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": "sha-502",
            }
        ),
        encoding="utf-8",
    )

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 501
    assert detected[0]["issue"] == 494
    assert detected[0]["decision"] == "missing"


def test_detect_unauthorized_merges_flags_approved_sha_mismatch(tmp_path: Path) -> None:
    """A merged worker branch with an approved decision for a different head SHA is flagged (issue #502 / cw #467)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    class FakeGitHubWithMergedWorkerPR(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.prs = [
                {
                    "number": 503,
                    "title": "fix: approved but then amended",
                    "url": "https://example.test/pull/503",
                    "headRefName": "agent/issue-496-fix",
                    "baseRefName": "main",
                    "headRefOid": "sha-503-final",
                    "state": "MERGED",
                    "isCrossRepository": False,
                    "body": "Closes #496",
                    "labels": [],
                },
            ]

    fake_gh = FakeGitHubWithMergedWorkerPR()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # The approval was recorded for an earlier head; the merged head differs.
    pr_dir = paths.prs / "pr-503"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": "sha-503-reviewed",
            }
        ),
        encoding="utf-8",
    )

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 503
    assert detected[0]["issue"] == 496
    assert detected[0]["decision"] == "approved"
    assert detected[0]["reviewed_head_sha"] == "sha-503-reviewed"
    assert detected[0]["live_head_sha"] == "sha-503-final"


def test_detect_unauthorized_merges_reuses_dispatch_merged_prs(tmp_path: Path) -> None:
    """loop() should reuse the merged PR list from dispatch() instead of calling merged_pr_list() again (issue #502 Finding 3)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    class CountingFakeGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_pr_list_calls = 0

        def merged_pr_list(self):
            self.merged_pr_list_calls += 1
            return super().merged_pr_list()

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)
    fake_gh = CountingFakeGitHub()
    fake_gh.prs = [
        {
            "number": 501,
            "title": "fix: worker self-merge",
            "url": "https://example.test/pull/501",
            "headRefName": "agent/issue-494-fix",
            "baseRefName": "main",
            "headRefOid": "sha-501",
            "state": "MERGED",
            "isCrossRepository": False,
            "body": "Closes #494",
            "labels": [],
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # dispatch() fetches merged PRs as part of its normal pass.
    result = app.dispatch(limit=1)
    assert result.data.get("merged_prs") == fake_gh.prs
    assert fake_gh.merged_pr_list_calls == 1

    # The tripwire, when handed that list, must not make a second API call.
    detected = app._detect_unauthorized_merges(result.data["merged_prs"])
    assert fake_gh.merged_pr_list_calls == 1
    assert len(detected) == 1
    assert detected[0]["pr"] == 501


def test_detect_unauthorized_merges_against_real_rest_merged_pr_list(
    monkeypatch, tmp_path: Path
) -> None:
    """The tripwire must fire against the shape merged_pr_list() ACTUALLY returns.

    Every other tripwire test above builds its merged-PR fixture by hand and
    spells ``headRefOid`` explicitly. But ``merged_pr_list()`` is REST-only by
    construction (issue #361) and the REST payload spells that field
    ``head.sha``, so those fixtures asserted a key the production path did not
    emit: ``live_head_sha`` was always ``None``, ``head_matches`` was always
    False, and the SHA half of this control could never distinguish an
    authorized merge from a bypass. Fixed in #631; this test is the regression
    guard, and it exercises the real producer so the *contract* between
    ``merged_pr_list()`` and the tripwire is what is under test rather than a
    hand-written dict.
    """
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    merged_head_sha = "27a20fbd9c1e4d3a8f5b6c7d8e9f0a1b2c3d4e5f"
    rest_page = [
        {
            "number": 501,
            "title": "fix: a worker branch that got merged",
            "body": "Closes #494",
            "merged_at": "2026-07-20T20:19:07Z",
            "head": {
                "ref": "agent/issue-494-fix",
                "sha": merged_head_sha,
                "repo": {"full_name": "o/r"},
            },
            "base": {"repo": {"full_name": "o/r"}},
        }
    ]
    # merged_pr_list() paginates until it sees an empty page.
    responses = [json.dumps(rest_page), "[]"]

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=responses.pop(0), stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    # The real producer, driven off a real REST payload.
    merged_prs = github_module.GitHub(tmp_path).merged_pr_list()
    assert len(merged_prs) == 1
    assert merged_prs[0]["headRefOid"] == merged_head_sha, (
        "merged_pr_list() must map REST head.sha onto headRefOid (#631) — "
        "without it the tripwire below cannot compare SHAs at all"
    )

    # The real consumer. loop() hands the fetched list in as a parameter on the
    # hot path, so passing it explicitly is the production shape; FakeGitHub is
    # here only to satisfy the app's constructor, not to supply the fixture.
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    # Steady state, not the arming pass: this test is about the producer/consumer
    # SHA contract, so the baseline must not be what silences it.
    _arm_unauthorized_merge_tripwire(paths)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    pr_dir = paths.prs / "pr-501"
    pr_dir.mkdir(parents=True, exist_ok=True)
    decision_path = pr_dir / "review-decision.json"

    # 1. Approved for exactly the merged head -> authorized, tripwire silent.
    #    This is the assertion that fails when the REST normalizer omits
    #    headRefOid: head_matches degrades to False and a properly reviewed
    #    merge gets reported as a bypass.
    decision_path.write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": merged_head_sha}),
        encoding="utf-8",
    )
    assert app._detect_unauthorized_merges(merged_prs) == []

    # 2. Approved for a DIFFERENT head -> flagged, and live_head_sha must carry
    #    the real REST head.sha rather than None.
    decision_path.write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "stale-review-sha"}),
        encoding="utf-8",
    )
    detected = app._detect_unauthorized_merges(merged_prs)

    assert len(detected) == 1
    assert detected[0]["pr"] == 501
    assert detected[0]["issue"] == 494
    assert detected[0]["decision"] == "approved"
    assert detected[0]["reviewed_head_sha"] == "stale-review-sha"
    assert detected[0]["live_head_sha"] == merged_head_sha


def test_auto_merge_config_queue_bot_login_defaults_to_none(tmp_path: Path) -> None:
    """Issue #1194: the default AutoMergeConfig() must disable queue sync-merge
    recognition entirely, preserving today's #502 tripwire behavior."""
    assert AutoMergeConfig().queue_bot_login is None


def test_auto_merge_config_mergequeue_wedge_hours_defaults_to_24() -> None:
    """Issue #1401: default AutoMergeConfig() enables the time-in-mergequeue
    watchdog at 24h -- the live #1751 case ran 28h+ undetected, so the default
    must be on, not opt-in."""
    assert AutoMergeConfig().mergequeue_wedge_hours == 24.0


def test_loop_surfaces_unauthorized_merge_in_errors_bucket(tmp_path: Path) -> None:
    """loop() must wire the post-merge tripwire into the errors bucket even when dispatch() had no ready issues and returned an empty merged_prs list (issue #502)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    class CountingFakeGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_pr_list_calls = 0

        def merged_pr_list(self):
            self.merged_pr_list_calls += 1
            return super().merged_pr_list()

    fake_gh = CountingFakeGitHub()
    # No ready issues and no open PRs — only a merged worker PR the tripwire
    # must catch. dispatch() will return merged_prs=[] (no ready issues), so
    # the tripwire must fall back to fetching its own list to stay armed.
    fake_gh.issues = []
    fake_gh.prs = [
        {
            "number": 501,
            "title": "fix: worker self-merge",
            "url": "https://example.test/pull/501",
            "headRefName": "agent/issue-494-fix",
            "baseRefName": "main",
            "headRefOid": "sha-501",
            "state": "MERGED",
            "isCrossRepository": False,
            "body": "Closes #494",
            "labels": [],
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.loop(merge=False)

    # dispatch() returned an empty merged_prs list because there were no ready
    # issues — the tripwire must NOT treat that as "no merged PRs to check".
    assert result.data["dispatch"].get("merged_prs") == []
    assert len(result.data["errors"]) == 1
    error = result.data["errors"][0]
    assert error["pr"] == 501
    assert error["issue"] == 494
    assert "MERGED" in error["error"]
    assert "possible worker self-merge" in error["error"]
    # The tripwire fetched its own list because the reused list was empty.
    assert fake_gh.merged_pr_list_calls >= 1


def test_loop_tripwire_silent_for_approved_matching_head(tmp_path: Path) -> None:
    """loop() must not flag a merged worker PR whose approved review decision covers the merged head (issue #502)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    # Arm with an empty baseline so the silence asserted below is evidence that
    # the approval matched the merged head — an unarmed pass is silent about
    # everything, which would make this test pass for the wrong reason.
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [
        {
            "number": 502,
            "title": "fix: approved merge",
            "url": "https://example.test/pull/502",
            "headRefName": "agent/issue-495-fix",
            "baseRefName": "main",
            "headRefOid": "sha-502",
            "state": "MERGED",
            "isCrossRepository": False,
            "body": "Closes #495",
            "labels": [],
        },
    ]

    pr_dir = paths.prs / "pr-502"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-502"}),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.loop(merge=False)

    tripwire_errors = [
        e for e in result.data["errors"] if "possible worker self-merge" in e["error"]
    ]
    assert tripwire_errors == [], (
        f"approved matching-head merge must not be flagged, got {tripwire_errors}"
    )


def test_unauthorized_merge_tripwire_arms_instead_of_flagging_history(tmp_path: Path) -> None:
    """The first pass records pre-existing uncovered merges as a baseline and reports nothing.

    Without this bound the tripwire asserts its policy retroactively over the
    whole 500-PR ``merged_pr_list()`` window. Measured against the live repo
    before this landed, that was 48 findings appended to ``loop()``'s ``errors``
    bucket on EVERY pass — there is no dedupe — which pins ok=False forever and
    buries a real self-merge in constant noise. A control that can never go quiet
    is not a control.
    """
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import UNAUTHORIZED_MERGE_BASELINE_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    history = [
        _merged_worker_pr(101, 91, "sha-101"),
        _merged_worker_pr(102, 92, "sha-102"),
    ]

    # No decision files at all: both would be flagged by an unbounded tripwire.
    assert app._detect_unauthorized_merges(history) == [], (
        "the arming pass must report nothing, not the whole backlog"
    )

    state = load_state(paths.state_file)
    baseline = state.get(UNAUTHORIZED_MERGE_BASELINE_KEY)
    assert isinstance(baseline, dict), "arming must persist a baseline to state.json"
    assert baseline["pre_existing_prs"] == [101, 102]
    assert baseline["armed_at"]

    # The backlog must remain auditable rather than being silently dropped: the
    # event carries the full PR list, not just a count.
    armed = [e for e in state["events"] if e["kind"] == "unauthorized_merge_baseline_armed"]
    assert len(armed) == 1
    assert armed[0]["payload"]["pre_existing_count"] == 2
    assert armed[0]["payload"]["pre_existing_prs"] == [101, 102]


def test_unauthorized_merge_tripwire_flags_merges_after_arming(tmp_path: Path) -> None:
    """A merge that lands after arming is still flagged — the baseline suppresses history only."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    history = [_merged_worker_pr(101, 91, "sha-101")]
    assert app._detect_unauthorized_merges(history) == []

    # A NEW uncovered merge appears. Note its number (99) is BELOW the baselined
    # PR's (101): a high-water-mark watermark would wrongly exempt it, which is
    # why the baseline is an explicit set. Three worker PRs were open and
    # below the highest merged PR when this armed on the live repo, so this is
    # the real case, not a contrived one.
    later = [*history, _merged_worker_pr(99, 89, "sha-99")]
    detected = app._detect_unauthorized_merges(later)

    assert [d["pr"] for d in detected] == [99], (
        f"only the post-arming merge may be flagged, got {detected}"
    )
    assert detected[0]["issue"] == 89
    assert detected[0]["live_head_sha"] == "sha-99"


def test_unauthorized_merge_baseline_arms_once_and_does_not_widen(tmp_path: Path) -> None:
    """Re-running the tripwire must not re-arm and swallow merges that landed after the first pass."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import UNAUTHORIZED_MERGE_BASELINE_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    assert app._detect_unauthorized_merges([_merged_worker_pr(101, 91, "sha-101")]) == []
    first = load_state(paths.state_file)[UNAUTHORIZED_MERGE_BASELINE_KEY]

    grown = [_merged_worker_pr(101, 91, "sha-101"), _merged_worker_pr(102, 92, "sha-102")]
    for _ in range(3):
        assert [d["pr"] for d in app._detect_unauthorized_merges(grown)] == [102]

    after = load_state(paths.state_file)[UNAUTHORIZED_MERGE_BASELINE_KEY]
    assert after == first, "the baseline must be written once and never widened"


def test_unauthorized_merge_tripwire_does_not_arm_when_pr_fetch_fails(tmp_path: Path) -> None:
    """A gh failure must not bake an empty baseline that permanently exempts real history.

    This is the sharpest failure mode of the whole mechanism: if the very first
    pass after deployment cannot fetch the merged PR list and arms anyway, the
    baseline records "nothing pre-existed" and the tripwire is then permanently
    blind to every merge it never saw.
    """
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import UNAUTHORIZED_MERGE_BASELINE_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubFailingMergedList(FakeGitHub):
        def merged_pr_list(self):
            raise github_module.GitHubError("gh unavailable")

    app = OrchestratorApp(tmp_path, paths, config, FakeGitHubFailingMergedList())

    assert app._detect_unauthorized_merges() == []
    assert UNAUTHORIZED_MERGE_BASELINE_KEY not in load_state(paths.state_file), (
        "a failed fetch must leave the tripwire unarmed so it can arm from real data later"
    )

    # Proof the unarmed state is recoverable: once gh works, arming sees the real
    # history and the tripwire still reports post-arming merges.
    app.gh = FakeGitHub()  # type: ignore[assignment]
    assert app._detect_unauthorized_merges([_merged_worker_pr(101, 91, "sha-101")]) == []
    assert load_state(paths.state_file)[UNAUTHORIZED_MERGE_BASELINE_KEY]["pre_existing_prs"] == [
        101
    ]


def test_unauthorized_merge_baseline_arming_writes_nothing_in_dry_run(tmp_path: Path) -> None:
    """--dry-run must not persist the baseline (issues #609/#613/#621).

    A preview that arms the tripwire would silently consume the one-time arming
    opportunity, permanently baselining whatever happened to be merged at preview
    time. The preview still reports nothing, which is what an armed pass reports.
    """
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import UNAUTHORIZED_MERGE_BASELINE_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)

    assert app._detect_unauthorized_merges([_merged_worker_pr(101, 91, "sha-101")]) == []
    assert UNAUTHORIZED_MERGE_BASELINE_KEY not in load_state(paths.state_file), (
        "dry-run must leave no baseline behind"
    )


def test_unauthorized_merge_ack_suppresses_acknowledged_finding(tmp_path: Path) -> None:
    """An acknowledged post-arming finding must stop polluting every pass (issue #673).

    The tripwire keeps its bite until a finding is explicitly acknowledged; once
    acked it is filtered the same way the pre-arming baseline filters history, so
    ``ok=False`` / ``errors`` go back to meaning "there is something new to look
    at" instead of "the mechanism cannot ever clear this".
    """
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [
        _merged_worker_pr(1408, 1404, "sha-1408"),
        _merged_worker_pr(1392, 1268, "sha-1392"),
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Pre-ack: both post-arming findings are flagged.
    detected = app._detect_unauthorized_merges(fake_gh.prs)
    assert sorted(d["pr"] for d in detected) == [1392, 1408]

    # Acknowledge both (e.g. root cause fixed in #672, confirmed benign per #634).
    _ack_unauthorized_merge(paths, 1408, "root cause fixed in #672")
    _ack_unauthorized_merge(paths, 1392, "root cause fixed in #672")

    # Post-ack: both are suppressed — the tripwire can go quiet.
    assert app._detect_unauthorized_merges(fake_gh.prs) == [], (
        "an acknowledged finding must not be re-reported on every pass"
    )

    # A NEW post-arming finding is still flagged — ack only suppresses what was
    # explicitly acked, it does not auto-acknowledge anything else.
    new_prs = [*fake_gh.prs, _merged_worker_pr(1500, 1501, "sha-1500")]
    detected_after = app._detect_unauthorized_merges(new_prs)
    assert [d["pr"] for d in detected_after] == [1500]


def test_unauthorized_merge_ack_does_not_suppress_unacked(tmp_path: Path) -> None:
    """The ack set suppresses only the acked PR, never a sibling finding (issue #673)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)
    _ack_unauthorized_merge(paths, 1408, "fixed")

    fake_gh = FakeGitHub()
    prs = [
        _merged_worker_pr(1408, 1404, "sha-1408"),
        _merged_worker_pr(1392, 1268, "sha-1392"),
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    detected = app._detect_unauthorized_merges(prs)
    assert [d["pr"] for d in detected] == [1392], (
        "acking #1408 must not also suppress the unrelated #1392 finding"
    )


def test_ack_unauthorized_merge_records_ack_and_event(tmp_path: Path) -> None:
    """``ack_unauthorized_merge`` persists the ack set and an audit event (issue #673)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.workflow import UNAUTHORIZED_MERGE_ACK_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app.ack_unauthorized_merge(1408, "root cause fixed in #672", by="operator")

    assert result.ok is True
    state = load_state(paths.state_file)
    acks = state.get(UNAUTHORIZED_MERGE_ACK_KEY)
    assert isinstance(acks, dict), "ack must persist an ack set to state.json"
    entry = acks["1408"]
    assert entry["reason"] == "root cause fixed in #672"
    assert entry["acknowledged_at"]
    assert entry["by"] == "operator"

    # The ack must be auditable: an event carries who/why/when.
    acked = [e for e in state["events"] if e["kind"] == "unauthorized_merge_acknowledged"]
    assert len(acked) == 1
    assert acked[0]["payload"]["pr"] == 1408
    assert acked[0]["payload"]["reason"] == "root cause fixed in #672"
    assert acked[0]["payload"]["by"] == "operator"


def test_ack_unauthorized_merge_requires_reason(tmp_path: Path) -> None:
    """An ack without a reason is rejected — a tripwire that can be silenced silently is no control (issue #673)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.workflow import UNAUTHORIZED_MERGE_ACK_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app.ack_unauthorized_merge(1408, "   ")
    assert result.ok is False
    assert "reason" in result.message.lower()
    assert UNAUTHORIZED_MERGE_ACK_KEY not in load_state(paths.state_file), (
        "a rejected ack must not have written anything to state"
    )


def test_ack_unauthorized_merge_updates_existing_ack(tmp_path: Path) -> None:
    """Re-acking a PR updates the record rather than refusing or duplicating (issue #673)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.workflow import UNAUTHORIZED_MERGE_ACK_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    app.ack_unauthorized_merge(1408, "initial triage", by="alice")
    first = load_state(paths.state_file)[UNAUTHORIZED_MERGE_ACK_KEY]["1408"]

    app.ack_unauthorized_merge(1408, "root cause fixed in #672", by="bob")
    second = load_state(paths.state_file)[UNAUTHORIZED_MERGE_ACK_KEY]["1408"]

    assert len(load_state(paths.state_file)[UNAUTHORIZED_MERGE_ACK_KEY]) == 1, (
        "re-acking must not duplicate the entry"
    )
    assert second["reason"] == "root cause fixed in #672"
    assert second["by"] == "bob"
    assert second["acknowledged_at"] >= first["acknowledged_at"]


def test_cli_tripwire_ack_writes_state(monkeypatch, tmp_path: Path) -> None:
    """`charlie tripwire ack <pr> --reason ...` persists the ack through the CLI (issue #673)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.workflow import UNAUTHORIZED_MERGE_ACK_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    monkeypatch.setattr(cli, "build_app", lambda args: app)

    exit_code = cli.main(
        ["tripwire", "ack", "1408", "--reason", "root cause fixed in #672", "--by", "operator"]
    )
    assert exit_code == 0

    acks = load_state(paths.state_file)[UNAUTHORIZED_MERGE_ACK_KEY]
    assert acks["1408"]["reason"] == "root cause fixed in #672"
    assert acks["1408"]["by"] == "operator"


def test_cli_tripwire_ack_requires_reason(monkeypatch, capsys, tmp_path: Path) -> None:
    """`charlie tripwire ack` without --reason exits non-zero and writes nothing (issue #673)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.workflow import UNAUTHORIZED_MERGE_ACK_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    monkeypatch.setattr(cli, "build_app", lambda args: app)

    exit_code = cli.main(["tripwire", "ack", "1408"])
    assert exit_code == 1
    assert UNAUTHORIZED_MERGE_ACK_KEY not in load_state(paths.state_file)


def test_github_delete_branch_failure_returns_false(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="Reference does not exist",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    assert github_module.GitHub(tmp_path).delete_branch("agent/issue-1-x") is False


def test_github_add_issue_label_failure_does_not_raise(monkeypatch, tmp_path: Path) -> None:
    """C5 boundary test: add_issue_label with allow_failure=True returns error value, does not raise."""

    def fake_run(cmd, *args, check=False, **kwargs):
        if check:
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="simulated failure")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="simulated failure")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    # Should not raise despite subprocess failure (allow_failure=True in add_issue_label)
    gh.add_issue_label(123, "agent:in-progress")


def test_github_remove_issue_label_failure_does_not_raise(monkeypatch, tmp_path: Path) -> None:
    """C5 boundary test: remove_issue_label with allow_failure=True returns error value, does not raise."""

    def fake_run(cmd, *args, check=False, **kwargs):
        if check:
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="simulated failure")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="simulated failure")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    # Should not raise despite subprocess failure (allow_failure=True in remove_issue_label)
    gh.remove_issue_label(123, "agent:in-progress")


def test_github_add_issue_label_returns_false_on_failure(monkeypatch, tmp_path: Path) -> None:
    """Boolean-truthfulness test: add_issue_label returns False on subprocess failure (returncode=1)."""

    def fake_run(cmd, *args, check=False, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="simulated failure")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.add_issue_label(123, "agent:in-progress")
    assert result is False, "add_issue_label must return False on failure"


def test_github_add_issue_label_returns_true_on_success(monkeypatch, tmp_path: Path) -> None:
    """Boolean-truthfulness test: add_issue_label returns True on subprocess success (returncode=0)."""

    def fake_run(cmd, *args, check=False, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.add_issue_label(123, "agent:in-progress")
    assert result is True, "add_issue_label must return True on success"


def test_github_remove_issue_label_returns_false_on_failure(monkeypatch, tmp_path: Path) -> None:
    """Boolean-truthfulness test: remove_issue_label returns False on subprocess failure (returncode=1)."""

    def fake_run(cmd, *args, check=False, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="simulated failure")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.remove_issue_label(123, "agent:in-progress")
    assert result is False, "remove_issue_label must return False on failure"


def test_github_remove_issue_label_returns_true_on_success(monkeypatch, tmp_path: Path) -> None:
    """Boolean-truthfulness test: remove_issue_label returns True on subprocess success (returncode=0)."""

    def fake_run(cmd, *args, check=False, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.remove_issue_label(123, "agent:in-progress")
    assert result is True, "remove_issue_label must return True on success"


# --- Cross-family adversarial review ------------------------------------------


def test_render_command_templates_list_and_string() -> None:
    values = {"model": "codex", "prompt_path": "/tmp/p.md"}
    assert render_command(
        ("devin", "--model", "{model}", "-p", "--prompt-file", "{prompt_path}"), values
    ) == ["devin", "--model", "codex", "-p", "--prompt-file", "/tmp/p.md"]
    assert render_command("devin --model {model}", values) == "devin --model codex"


def test_claude_code_example_config_selects_claude_worker() -> None:
    config = load_config(EXAMPLES_DIR / "orchestrator.config.claude-code.yaml")

    assert config.dispatch.worker_template == "worker_claude_code.md"


def test_claude_code_example_config_sets_bounded_xdist_worker_env() -> None:
    config = load_config(EXAMPLES_DIR / "orchestrator.config.claude-code.yaml")

    # The shipped example bounds local test parallelism at the launch boundary
    # (the RUNBOOK "Local host saturation ceiling" section references this).
    assert config.claude_code.worker_env == {"PYTEST_XDIST_AUTO_NUM_WORKERS": "2"}


def test_run_cross_family_sanitizes_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_cross_family_review must sanitize the environment before spawning the subprocess."""
    from charlie_work.env_sanitize import sanitize_env

    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    # Set parent env variables (simulating orchestrator leak)
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    env = sanitize_env(repo_root)

    assert "VIRTUAL_ENV" not in env, "VIRTUAL_ENV must be dropped when repo has no .venv"
    assert "UV_PROJECT_ENVIRONMENT" not in env, (
        "UV_PROJECT_ENVIRONMENT must be dropped when repo has no .venv"
    )


def test_run_cross_family_sanitizes_environment_with_repo_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When repo has a real .venv, VIRTUAL_ENV must be set and UV_PROJECT_ENVIRONMENT dropped."""
    from charlie_work.env_sanitize import sanitize_env

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    repo_venv = repo_root / ".venv"
    repo_venv.mkdir()

    # Set parent env variables (simulating orchestrator leak)
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    env = sanitize_env(repo_root)

    assert env.get("VIRTUAL_ENV") == str(repo_venv), "VIRTUAL_ENV must be set to repo .venv"
    assert "UV_PROJECT_ENVIRONMENT" not in env, (
        "UV_PROJECT_ENVIRONMENT must be dropped; uv's default is the same repo .venv (issue #649)"
    )


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


def test_report_body_is_valid_accepts_heading_style_markers() -> None:
    """Cross-family models (e.g. kimi-k3) emit findings as ``### NIT —`` headings
    and a ``## Verdict`` heading instead of ``**NIT**`` bold / ``Verdict:`` line.
    These are real reviews and must not be falsely rejected as UNAVAILABLE.
    """
    heading_severity = (
        "### NIT — worktree.py:2977 (new): dead weight\n\n"
        "## Verdict\n\n**Approve** — claims verified."
    )
    assert report_body_is_valid(heading_severity) is True
    # Heading verdict alone (no severity heading) is also valid.
    assert report_body_is_valid("## Verdict\n\nApprove") is True
    # Heading severity alone (no verdict heading) is also valid.
    assert report_body_is_valid("### MAJOR — bug.py:10: off-by-one\n\nfix it") is True


def test_extract_report_body_strips_wrapper_but_preserves_model_output() -> None:
    body = "**MAJOR**\nissue\n\nVerdict: safe"
    wrapped = f"# Cross-family adversarial review — `codex`\n\n{_CAVEAT}\n\n---\n\n{body}\n"
    assert extract_report_body(wrapped) == body
    assert extract_report_body(body) == body


# --- P0 fixes: state safety, label honesty, rework cap, loop isolation --------


def test_string_dispatch_command_rejects_issue_title(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command="echo {issue_title}"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    assert result.ok is False
    assert result.data["failed_count"] == 1
    assert "list form" in result.data["dispatch_results"][0]["error"]
    assert (123, "agent:in-progress") not in fake_gh.labels_added


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


def test_loop_parks_foreign_issue_ref_pr(monkeypatch, tmp_path: Path) -> None:
    """A PR whose branch-derived issue number does not exist in this repo
    (e.g. opened against the wrong fleet repo) is parked via
    ``foreign_issue_ref`` instead of failing the pass every 5 minutes
    forever. GitHubNotFoundError from issue_view is caught before the
    general GitHubError handler, so it never lands in result.data["errors"]
    and does not flip result.ok to False.

    Issue #1132: parking now requires ``confirm_passes`` (default 2)
    consecutive not-found passes before the marker is confirmed and the
    one-shot digest is emitted. A transient window (minutes) clears before
    two 5-minute passes complete."""
    from charlie_work.config import NotifyConfig
    from charlie_work.github import GitHubNotFoundError

    class ForeignIssueGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []
            self.prs = [
                {
                    "number": 789,
                    "title": "Fix #4242: foreign",
                    "url": "https://example.test/pull/789",
                    "headRefName": "agent/issue-4242-x",
                    "baseRefName": "main",
                    "headRefOid": "sha-789",
                    "mergeStateStatus": "CLEAN",
                    "body": "Closes #4242",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                }
            ]
            self.issue_view_calls = 0

        def issue_view(self, number: int):
            if number == 4242:
                self.issue_view_calls += 1
                raise GitHubNotFoundError("could not resolve to a Issue with the number 4242.")
            return super().issue_view(number)

    config = OrchestratorConfig(
        notify=NotifyConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = ForeignIssueGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    captured: list[Any] = []
    monkeypatch.setattr(
        "charlie_work.workflow.emit_digest",
        lambda notify_config, digest: captured.append(digest),
    )

    # Pass 1: first not-found — marker written with confirmations=1, but
    # not yet confirmed (1 < 2), so no digest and the PR is still tracked.
    result = app.loop(limit=0)

    assert result.ok is True
    assert result.data["errors"] == []
    assert fake_gh.issue_view_calls == 1
    assert len(captured) == 0  # not yet confirmed

    state = load_state(app.paths.state_file)
    assert state["prs"]["789"]["foreign_issue_ref"]["issue"] == 4242
    assert state["prs"]["789"]["foreign_issue_ref"]["confirmations"] == 1

    # Pass 2: second not-found — confirmations reaches 2, marker confirmed,
    # one-shot digest emitted.
    result2 = app.loop(limit=0)

    assert result2.ok is True
    assert result2.data["errors"] == []
    assert fake_gh.issue_view_calls == 2
    assert len(captured) == 1
    assert captured[0].transitions[0].health == "FOREIGN_ISSUE_REF"
    assert captured[0].transitions[0].issue_number == 789

    state = load_state(app.paths.state_file)
    assert state["prs"]["789"]["foreign_issue_ref"]["confirmations"] == 2

    # Pass 3: the confirmed marker skips all per-PR work with zero GitHub
    # calls and no repeat digest.
    result3 = app.loop(limit=0)

    assert result3.ok is True
    assert result3.data["open_tracked_prs"] == 0
    assert fake_gh.issue_view_calls == 2
    assert len(captured) == 1
    # Issue #1132: parked PRs are now visible in the loop_completed payload.
    assert result3.data["parked_prs"] == [789]


def test_loop_dead_session_notifies_when_watchdog_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #706: when ``watchdog.enabled=False``, ``_detect_stalled_sessions``
    returns ``[]`` (it is gated on watchdog), so the stalled-session notify path
    never fires. But dead workers ARE still reaped by
    ``_classify_dead_sessions_and_update_throttle_state`` (which is NOT
    watchdog-gated). The loop must feed those reaped dead-session transitions
    into the notify digest so an operator monitoring
    ``notify/digest.jsonl`` gets signal instead of a permanently empty file.
    """
    from charlie_work.devin_shell import SessionRecord, _sidecar_path as devin_sidecar_path

    issue_number = 706
    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            enabled=False,
            stall_minutes=20,
            # Pin to 0 so the dead-session lane reaps immediately rather than
            # deferring for max_inconclusive_probe_deferrals passes (a bare
            # test environment has no real sessions.db, so the probe is always
            # inconclusive).
            max_inconclusive_probe_deferrals=0,
        ),
        notify=NotifyConfig(enabled=True, sink="file", file_path=""),
        review_dispatch=ReviewDispatchConfig(enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)

    # Seed state so the dead-session lane has a dispatched issue to reap.
    with state_lock(paths.state_file):
        save_state(
            paths.state_file,
            {
                "version": 1,
                "issues": {
                    str(issue_number): {
                        "status": "dispatched",
                        "branch_name": f"agent/issue-{issue_number}-test",
                        "worker_pid": 99999,
                        "worker_process_start_time": 1234567890.0,
                    }
                },
                "prs": {},
                "events": [],
            },
        )

    # Create a dead session sidecar (non-existent PID).
    sessions_dir = paths.root / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text("Session log\n", encoding="utf-8")
    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    record = SessionRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}-test",
        worktree_path="/tmp/worktree-706",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=99999,  # Non-existent PID → dead
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # FakeGitHub with the issue present (so the dead-session lane can relabel).
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": issue_number,
            "title": "Test issue 706",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "Test",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    captured: list[Any] = []
    monkeypatch.setattr(
        "charlie_work.workflow.emit_digest",
        lambda notify_config, digest: captured.append(digest),
    )

    result = app.loop(limit=0)

    assert result.ok is True
    # The dead session was reaped and a DEAD transition was emitted to notify.
    dead_digests = [d for d in captured if any(t.health == "DEAD" for t in d.transitions)]
    assert len(dead_digests) == 1, (
        f"Expected exactly one DEAD notify digest, got {len(dead_digests)} (captured: {captured})"
    )
    transition = dead_digests[0].transitions[0]
    assert transition.issue_number == issue_number
    assert transition.health == "DEAD"
    # The sidecar was reaped (proving the dead-session lane ran).
    assert not sidecar_path.exists()


def test_clear_reviewer_quota_drops_alerted_at() -> None:
    """clear_reviewer_quota must also pop alerted_at so a later exhaustion
    episode alerts again instead of staying silently suppressed."""
    from charlie_work.state import (
        clear_reviewer_quota,
        mark_reviewer_quota_alerted,
        set_reviewer_quota_exhausted,
    )

    state: dict[str, Any] = {"version": 1, "issues": {}, "prs": {}, "events": []}
    state = set_reviewer_quota_exhausted(
        state, throttled_until="2026-08-01T00:00:00Z", probe_after="2026-08-01T00:00:00Z"
    )
    state = mark_reviewer_quota_alerted(state)
    assert "alerted_at" in state["reviewer_quota"]

    cleared = clear_reviewer_quota(state)

    assert "alerted_at" not in cleared["reviewer_quota"]
    assert "throttled_until" not in cleared["reviewer_quota"]


def test_defer_reviewer_probe_after_bumps_past_probe_after() -> None:
    """A red flat probe must bump reviewer_quota.probe_after forward so
    dispatch_reviews's probe_mode gate defers instead of independently
    launching a real reviewer session into the same still-closed window
    (issue #663)."""
    from charlie_work.state import defer_reviewer_probe_after, set_reviewer_quota_exhausted

    state: dict[str, Any] = {"version": 1, "issues": {}, "prs": {}, "events": []}
    # Quota exhausted, probe_after in the past (ready to probe).
    state = set_reviewer_quota_exhausted(
        state,
        throttled_until="2099-01-01T00:00:00Z",
        probe_after="2020-01-01T00:00:00Z",
    )
    result = defer_reviewer_probe_after(state, "2026-08-01T00:30:00Z")
    assert result["reviewer_quota"]["probe_after"] == "2026-08-01T00:30:00Z"
    # throttled_until is untouched.
    assert result["reviewer_quota"]["throttled_until"] == "2099-01-01T00:00:00Z"


def test_defer_reviewer_probe_after_noop_when_quota_not_exhausted() -> None:
    """Must not write probe_after on a non-exhausted quota -- that would
    leave stale state for no reason (issue #663)."""
    from charlie_work.state import defer_reviewer_probe_after

    state: dict[str, Any] = {"version": 1, "issues": {}, "prs": {}, "events": []}
    result = defer_reviewer_probe_after(state, "2026-08-01T00:30:00Z")
    assert "probe_after" not in result.get("reviewer_quota", {})


def test_defer_reviewer_probe_after_never_moves_earlier() -> None:
    """If the reviewer quota's own exponential backoff already pushed
    probe_after further out than the flat probe's interval, the bump must
    not shorten it -- that would make dispatch_reviews probe more often,
    not less (issue #663)."""
    from charlie_work.state import defer_reviewer_probe_after, set_reviewer_quota_exhausted

    state: dict[str, Any] = {"version": 1, "issues": {}, "prs": {}, "events": []}
    state = set_reviewer_quota_exhausted(
        state,
        throttled_until="2099-01-01T00:00:00Z",
        probe_after="2026-08-01T04:00:00Z",
    )
    result = defer_reviewer_probe_after(state, "2026-08-01T00:30:00Z")
    assert result["reviewer_quota"]["probe_after"] == "2026-08-01T04:00:00Z"


def test_defer_reviewer_probe_after_overwrites_malformed_current() -> None:
    """A malformed current probe_after must not wedge the bump -- overwrite
    with the well-formed new value (issue #663)."""
    from charlie_work.state import defer_reviewer_probe_after, set_reviewer_quota_exhausted

    state: dict[str, Any] = {"version": 1, "issues": {}, "prs": {}, "events": []}
    state = set_reviewer_quota_exhausted(
        state,
        throttled_until="2099-01-01T00:00:00Z",
        probe_after="not-a-timestamp",
    )
    result = defer_reviewer_probe_after(state, "2026-08-01T00:30:00Z")
    assert result["reviewer_quota"]["probe_after"] == "2026-08-01T00:30:00Z"


def test_set_throttled_until_records_reason_and_adapter_kind() -> None:
    from charlie_work.state import empty_state, set_throttled_until

    state = set_throttled_until(
        empty_state(),
        "2026-08-01T00:00:00Z",
        reason="quota_exhausted",
        adapter_kind="claude-code",
    )

    assert state["throttled_until"] == "2026-08-01T00:00:00Z"
    assert state["throttle_reason"] == "quota_exhausted"
    assert state["throttle_adapter_kind"] == "claude-code"


def test_set_throttled_until_defaults_reason_and_adapter_kind_to_none() -> None:
    from charlie_work.state import empty_state, set_throttled_until

    state = set_throttled_until(empty_state(), "2026-08-01T00:00:00Z")

    assert state["throttle_reason"] is None
    assert state["throttle_adapter_kind"] is None


def test_quota_probe_arm_disarm_and_due_lifecycle() -> None:
    from datetime import UTC, datetime, timedelta

    from charlie_work.state import (
        arm_quota_probe,
        disarm_quota_probe,
        empty_state,
        is_quota_probe_armed,
        is_quota_probe_due,
    )

    state = empty_state()
    assert is_quota_probe_armed(state) is False
    assert is_quota_probe_due(state) is False  # unarmed is never "due"

    future = (datetime.now(UTC) + timedelta(minutes=15)).isoformat().replace("+00:00", "Z")
    state = arm_quota_probe(state, future)
    assert is_quota_probe_armed(state) is True
    assert is_quota_probe_due(state) is False

    past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    state = arm_quota_probe(state, past)
    assert is_quota_probe_due(state) is True

    state = disarm_quota_probe(state)
    assert is_quota_probe_armed(state) is False


def test_is_quota_probe_due_treats_malformed_timestamp_as_due() -> None:
    from charlie_work.state import arm_quota_probe, empty_state, is_quota_probe_due

    state = arm_quota_probe(empty_state(), "not-a-timestamp")

    assert is_quota_probe_due(state) is True


def test_any_quota_exhausted_indicator_gate() -> None:
    from datetime import UTC, datetime, timedelta

    from charlie_work.state import (
        any_quota_exhausted_indicator,
        empty_state,
        set_reviewer_quota_exhausted,
        set_throttled_until,
    )

    assert any_quota_exhausted_indicator(empty_state()) is False

    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    root_throttled = set_throttled_until(empty_state(), future, reason="rate_limited")
    assert any_quota_exhausted_indicator(root_throttled) is True

    reviewer_throttled = set_reviewer_quota_exhausted(
        empty_state(), throttled_until=future, probe_after=future
    )
    assert any_quota_exhausted_indicator(reviewer_throttled) is True


def test_clear_quota_throttles_clears_root_throttle_for_claude_code_or_unset_adapter() -> None:
    from charlie_work.state import clear_quota_throttles, empty_state, set_throttled_until

    for adapter_kind in (None, "claude-code"):
        state = set_throttled_until(
            empty_state(),
            "2026-08-01T00:00:00Z",
            reason="rate_limited",
            adapter_kind=adapter_kind,
        )

        cleared = clear_quota_throttles(state)

        assert cleared["throttled_until"] is None
        assert cleared["throttle_reason"] is None
        assert cleared["throttle_adapter_kind"] is None


def test_clear_quota_throttles_preserves_provider_auth_throttle() -> None:
    """A dead key does not self-heal within minutes -- see
    claude_code._classify_session_failure; a green probe must not mask it."""
    from charlie_work.state import clear_quota_throttles, empty_state, set_throttled_until

    state = set_throttled_until(
        empty_state(),
        "2026-08-01T00:00:00Z",
        reason="provider_auth",
        adapter_kind="claude-code",
    )

    cleared = clear_quota_throttles(state)

    assert cleared["throttled_until"] == "2026-08-01T00:00:00Z"
    assert cleared["throttle_reason"] == "provider_auth"


def test_clear_quota_throttles_preserves_non_claude_code_adapter_throttle() -> None:
    """A devin/api-adapter throttle is on a different credential the ambient
    Claude Code CLI probe provides no evidence about."""
    from charlie_work.state import clear_quota_throttles, empty_state, set_throttled_until

    for adapter_kind in ("devin", "api"):
        state = set_throttled_until(
            empty_state(),
            "2026-08-01T00:00:00Z",
            reason="rate_limited",
            adapter_kind=adapter_kind,
        )

        cleared = clear_quota_throttles(state)

        assert cleared["throttled_until"] == "2026-08-01T00:00:00Z"
        assert cleared["throttle_adapter_kind"] == adapter_kind


def test_clear_quota_throttles_always_clears_reviewer_quota_and_resets_probe_failures() -> None:
    from charlie_work.state import (
        clear_quota_throttles,
        empty_state,
        set_reviewer_quota_exhausted,
        set_throttled_until,
    )

    # The reviewer-quota throttle MUST be derived from now, never hardcoded.
    # ``clear_quota_throttles`` only resets ``consecutive_probe_failures`` when
    # ``is_reviewer_quota_exhausted(data) or cleared_root`` holds, and
    # ``is_reviewer_quota_exhausted`` is true only while ``throttled_until`` is
    # still in the future (``datetime.now(UTC) < throttle_time``). This test
    # deliberately uses a *devin* root throttle below so ``cleared_root`` is
    # False, which makes the exhaustion check the sole path to the reset --
    # so the instant a hardcoded date goes stale, the assertion below flips
    # from testing the reset to testing nothing, and fails.
    #
    # This is not hypothetical: this test pinned "2026-08-01T00:00:00Z" and
    # began failing on every PR and on main at exactly that instant, blocking
    # the merge lane until the date was made relative.
    future = future_timestamp(days=3650)
    # Opaque by contrast: the root throttle is only ever round-tripped and
    # compared for equality, never against the clock, so an absolute literal is
    # safe here and keeps the "untouched" assertion easy to read.
    root_throttle = "2026-08-01T00:00:00Z"

    state = set_reviewer_quota_exhausted(empty_state(), throttled_until=future, probe_after=future)
    state = {
        **state,
        "reviewer_quota": {**state["reviewer_quota"], "consecutive_probe_failures": 3},
    }
    # Also carry a devin-adapter root throttle, to confirm reviewer_quota
    # clears independently of what the root-throttle branch decides.
    state = set_throttled_until(state, root_throttle, reason="rate_limited", adapter_kind="devin")

    cleared = clear_quota_throttles(state)

    assert "throttled_until" not in cleared["reviewer_quota"]
    assert cleared["reviewer_quota"]["consecutive_probe_failures"] == 0
    # The devin adapter's root throttle must still be untouched.
    assert cleared["throttled_until"] == root_throttle


def test_clear_quota_throttles_records_last_probe_cleared_at() -> None:
    """Issue #662: ``clear_quota_throttles`` stamps ``last_probe_cleared_at``
    on reviewer_quota so the dead-reviewer reap sweep can tell a recovery
    happened. It is recorded even when reviewer_quota was never exhausted
    (a green probe clearing a root-only throttle still proves the provider
    recovered), and survives ``clear_reviewer_quota`` across episodes.
    """
    from charlie_work.state import (
        clear_quota_throttles,
        clear_reviewer_quota,
        empty_state,
        reviewer_quota_last_probe_cleared_at,
        set_reviewer_quota_exhausted,
        set_throttled_until,
    )

    # Reviewer-quota exhaustion present: marker recorded after clear.
    state = set_reviewer_quota_exhausted(
        empty_state(), throttled_until="2026-08-01T00:00:00Z", probe_after="2026-08-01T00:00:00Z"
    )
    cleared = clear_quota_throttles(state)
    assert reviewer_quota_last_probe_cleared_at(cleared) is not None
    assert "throttled_until" not in cleared["reviewer_quota"]

    # Root-only throttle (reviewer_quota never set): marker still recorded.
    root_only = set_throttled_until(
        empty_state(), "2026-08-01T00:00:00Z", reason="rate_limited", adapter_kind="claude-code"
    )
    cleared_root = clear_quota_throttles(root_only)
    assert reviewer_quota_last_probe_cleared_at(cleared_root) is not None
    assert "throttled_until" not in cleared_root["reviewer_quota"]

    # The marker survives a subsequent clear_reviewer_quota (new episode).
    re_exhausted = set_reviewer_quota_exhausted(
        cleared_root, throttled_until="2026-09-01T00:00:00Z", probe_after="2026-09-01T00:00:00Z"
    )
    re_cleared = clear_reviewer_quota(re_exhausted)
    assert reviewer_quota_last_probe_cleared_at(
        re_cleared
    ) == reviewer_quota_last_probe_cleared_at(cleared_root)


def test_detect_and_handle_stalled_reviews_skips_terminal_pr_reaps_sidecar(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue observed 07-22: a dead reviewer sidecar for a PR already
    lifecycle-reaped to merged/closed (review_dispatch_status None) must be
    silently reaped -- no review_dispatch_stalled event, no rewrite to
    failed. A second, non-terminal PR in the same pass still gets the normal
    reap-to-failed treatment, proving the terminal skip is scoped to that PR
    only."""
    from charlie_work.state import empty_state

    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    state_file = tmp_path / "state.json"
    config = OrchestratorConfig()

    state = empty_state()
    state["prs"]["100"] = {
        "number": 100,
        "status": "merged",
        "review_dispatch_status": None,
    }
    state["prs"]["200"] = {
        "number": 200,
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T00:00:00Z",
        "reviewer_pid": 12345,
        "reviewer_process_start_time": 1.0,
    }
    save_state(state_file, state)

    def _sidecar(pr_number: int, started_at: str) -> dict[str, Any]:
        return {
            "issue_number": pr_number,
            "branch": f"agent/issue-{pr_number}-fix",
            "worktree_path": str(reviews_dir / f"pr-{pr_number}"),
            "prompt_path": str(reviews_dir / f"pr-{pr_number}" / ".orchestrator-prompt.md"),
            "command": ["claude", "-p"],
            "pid": 999999999,
            "started_at": started_at,
            "log_path": str(reviews_dir / f"issue-{pr_number}.claude.log"),
            "error": None,
            "process_start_time": 1.0,
            "adapter_kind": "claude-code",
        }

    old_started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar_100 = reviews_dir / "issue-100.claude.json"
    sidecar_100.write_text(json.dumps(_sidecar(100, old_started)), encoding="utf-8")
    sidecar_200 = reviews_dir / "issue-200.claude.json"
    sidecar_200.write_text(json.dumps(_sidecar(200, old_started)), encoding="utf-8")
    # PR 200 is the non-terminal PR that should get the normal reap-to-failed
    # treatment. Give it a readable log with no throttle marker so it
    # classifies as NOT_THROTTLED (the counted-failure path). Without this the
    # log read would fail and classify as UNDETERMINED (issue #1069), which
    # rolls back instead of failing — not the path this test exercises.
    (reviews_dir / "issue-200.claude.log").write_text("ordinary crash output\n", encoding="utf-8")

    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)
    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", lambda *a, **k: True
    )

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert not sidecar_100.exists()
    assert not sidecar_200.exists()
    assert [entry["pr"] for entry in stalled] == [200]

    state_after = load_state(state_file)
    assert state_after["prs"]["100"]["review_dispatch_status"] is None
    assert state_after["prs"]["100"]["status"] == "merged"
    assert state_after["prs"]["200"]["review_dispatch_status"] == "review_dispatch_failed"

    stalled_events = [
        e for e in state_after.get("events", []) if e.get("kind") == "review_dispatch_stalled"
    ]
    assert len(stalled_events) == 1
    assert stalled_events[0]["payload"]["pr_number"] == 200


def test_detect_and_handle_stalled_reviews_warns_on_checkout_removal_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #526: a genuine worktree-removal failure must not be silently
    discarded; the stalled sweep emits a one-shot warning event and sets a
    per-PR marker so the next pass can retry without flooding the event ring."""
    from datetime import timedelta

    from charlie_work.state import empty_state
    from charlie_work.workflow import _detect_and_handle_stalled_reviews

    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    state_file = tmp_path / "state.json"

    config = OrchestratorConfig()
    state = empty_state()
    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    state["prs"]["100"] = {
        "number": 100,
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": old_dispatched,
        "reviewer_pid": 12345,
        "reviewer_process_start_time": 1.0,
    }
    save_state(state_file, state)

    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-100-fix",
        "worktree_path": str(reviews_dir / "pr-100"),
        "prompt_path": str(reviews_dir / "pr-100" / ".orchestrator-prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,
        "started_at": old_dispatched,
        "log_path": str(reviews_dir / "issue-100.claude.log"),
        "error": None,
        "process_start_time": 1.0,
        "adapter_kind": "claude-code",
    }
    (reviews_dir / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")
    # Give the reviewer a readable log with no throttle marker so it
    # classifies as NOT_THROTTLED (the counted-failure path that calls
    # _remove_review_checkout_with_warning). Without this the log read would
    # fail and classify as UNDETERMINED (issue #1069), which uses a different
    # checkout-removal path — not the warning path this test exercises.
    (reviews_dir / "issue-100.claude.log").write_text("ordinary crash output\n", encoding="utf-8")

    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)
    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", lambda *a, **k: False
    )

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert [entry["pr"] for entry in stalled] == [100]
    state_after = load_state(state_file)
    assert state_after["prs"]["100"]["review_dispatch_status"] == "review_dispatch_failed"
    assert state_after["prs"]["100"]["review_checkout_removal_warned"] is True
    warning_events = [
        e
        for e in state_after.get("events", [])
        if e.get("kind") == "review_checkout_removal_failed"
    ]
    assert len(warning_events) == 1
    assert warning_events[0]["payload"]["pr_number"] == 100


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

    # Issue #1362 Stage 1: a corrupt flat file with no round-archive fallback
    # now resolves to {"decision": "missing"} rather than the old "invalid"
    # sentinel (review_decision.resolve_decision_payload) -- both are
    # equally non-terminal, so the fail-safe outcome below is unchanged.
    assert result.data["review_decision"] == {"decision": "missing"}
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


def test_intake_labels_prose_only_dependencies(tmp_path: Path) -> None:
    """Issue #225: intake should label issues with prose-only dependencies."""

    class ProseOnlyDepsGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 908,
                    "title": "Issue with prose-only deps",
                    "url": "https://example.test/issues/908",
                    "body": "Do not dispatch before P2-T2/P2-T3 have landed.",
                    "labels": [{"name": "automated-ready"}],
                },
                {
                    "number": 909,
                    "title": "Issue with structured blockers",
                    "url": "https://example.test/issues/909",
                    "body": "Blocked by #123",
                    "labels": [{"name": "automated-ready"}],
                },
                {
                    "number": 910,
                    "title": "Normal issue",
                    "url": "https://example.test/issues/910",
                    "body": "Just a normal issue",
                    "labels": [{"name": "automated-ready"}],
                },
            ]

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = ProseOnlyDepsGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.intake()

    assert result.ok is True
    # Issue 908 should be labeled with prose-only-deps
    assert 908 in result.data["prose_only_deps_issues"]
    # Issues 909 and 910 should not be labeled
    assert 909 not in result.data["prose_only_deps_issues"]
    assert 910 not in result.data["prose_only_deps_issues"]

    # Check that the label was added to issue 908
    assert (908, config.labels.prose_only_deps) in fake_gh.labels_added

    # Check state event was logged
    state = load_state(paths.state_file)
    assert any(e.get("kind") == "intake_prose_only_deps" for e in state["events"])
    prose_event = next(e for e in state["events"] if e.get("kind") == "intake_prose_only_deps")
    # The event structure uses "payload" key with "issue_numbers" inside
    assert prose_event.get("payload", {}).get("issue_numbers") == [908]


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

    config = OrchestratorConfig()
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
        devin=DevinConfig(shell_command=(sys.executable, "-c", "import sys; sys.exit(0)")),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
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
    config = OrchestratorConfig(worker=WorkerRoleConfig(harness="claude-code"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert "Issue #123" in str(captured["prompt_text"])  # rendered prompt fed through
    assert captured["venv_source"] is None  # issue #274: no shared venv by default
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


def test_janitor_block_writes_no_review_packet(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Issue #818: draft is no longer alone here -- pair it with an empty body
    # (a second, unrelated janitor failure) so this stays a genuine
    # janitor_blocked park rather than the new pure-draft auto-ready path.
    # Empty body (not mergeable=CONFLICTING) deliberately avoids also
    # tripping the separate merge-conflict rework-routing special case.
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True, "body": ""}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert fake_gh.pr_ready_calls == []  # not "otherwise ready" -- never attempted
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    assert not packet.exists()  # zero packet spend on a blocked PR
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"


def test_janitor_draft_only_block_auto_readies_pr(tmp_path: Path) -> None:
    """Issue #818: a draft PR that is otherwise mergeable is not a terminal
    park. When draft is the ONLY janitor failure, review() calls `gh pr
    ready` and defers the actual review to the next poll pass instead of
    writing status="janitor_blocked" -- pinning the actuation itself, not
    merely that the (pre-fix) failure string was appended.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False  # deferred to the next pass, not approved this pass
    assert result.data["draft_readied"] is True
    assert fake_gh.pr_ready_calls == [456]
    # GitHub's real effect: the PR is no longer a draft on the next fetch.
    assert fake_gh.pr_view(456)["isDraft"] is False
    # No packet spend -- the review itself is deferred, not performed now.
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    assert not packet.exists()
    # Distinguishable event kind (issue #818 AC4): greppable via
    # query_events(kind=...) rather than a manual `gh pr list` sweep.
    recorded = query_events(paths.state_file, kind="draft_pr_ready_triggered")
    assert len(recorded) == 1
    assert recorded[0]["payload"]["pr_number"] == 456
    # Issue #820 regression guard: no merge-hold anywhere means the new hold
    # check must not suppress the pre-existing auto-ready behavior, and must
    # not emit the hold-suppression event.
    assert query_events(paths.state_file, kind="draft_pr_ready_held") == []


def test_janitor_draft_only_block_gh_pr_ready_failure_stays_blocked(tmp_path: Path) -> None:
    """Issue #818 AC3: when `gh pr ready` itself fails, the PR must NOT be
    treated as ready -- the fleet must not proceed to merge on the
    assumption un-draft succeeded. Errors from external processes come back
    as values (GitHubRunResult.ok/.error), never exceptions.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True}
    fake_gh.pr_ready_ok = False
    fake_gh.pr_ready_error = "gh: insufficient permissions to mark PR ready"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("draft_readied") is not True
    assert fake_gh.pr_ready_calls == [456]
    # GitHub-side state is unchanged: still a draft.
    assert fake_gh.pr_view(456)["isDraft"] is True
    # Parked exactly like the pre-fix behavior -- not silently advanced.
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["prs"]["456"]["janitor_ok"] is False
    assert any("draft" in f.lower() for f in state["prs"]["456"]["janitor_failures"])
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    assert not packet.exists()  # never routed toward review/merge
    # Distinguishable, deduped event (issue #818 AC4).
    recorded = query_events(paths.state_file, kind="draft_pr_ready_failed")
    assert len(recorded) == 1
    assert recorded[0]["level"] == "warning"
    assert recorded[0]["payload"]["pr_number"] == 456
    assert "insufficient permissions" in recorded[0]["payload"]["error"]

    # A second pass with the same gh error does not re-fire the event
    # (cost-spirals.md dedup discipline: verdict.failures is byte-identical
    # every pass regardless of the actuator's outcome, so dedup is keyed on
    # the actuator's own error message, not on verdict.failures).
    app.review(456)
    # `gh pr ready` is attempted again -- it's the event that's deduped, not
    # the actuator call itself. Without this, a control-flow change that
    # skips the whole branch on pass 2 (e.g. an early return keyed off the
    # state just written) would still satisfy the event-count assertion
    # below, making it silently vacuous.
    assert fake_gh.pr_ready_calls == [456, 456]
    recorded_again = query_events(paths.state_file, kind="draft_pr_ready_failed")
    assert len(recorded_again) == 1


class _IssueViewCountingGitHub(FakeGitHub):
    """FakeGitHub subclass that records every ``issue_view`` call so tests can
    assert the merge-hold check reuses review()'s own per-pass issue fetch
    instead of issuing a redundant second ``gh issue view`` call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.issue_view_calls: list[int] = []

    def issue_view(self, number: int):
        self.issue_view_calls.append(number)
        return super().issue_view(number)


def test_janitor_draft_only_block_merge_hold_on_pr_suppresses_auto_ready(
    tmp_path: Path,
) -> None:
    """Issue #820: an operator parks a draft PR by applying the configured
    merge-hold label directly to the PR. The #818 auto-ready actuator must
    not un-draft it -- mirrors merge_ready's mergequeue-handoff hold check
    (workflow.py's merge_hold pattern) rather than inventing a variant.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _IssueViewCountingGitHub()
    fake_gh.prs[0] = {
        **fake_gh.prs[0],
        "isDraft": True,
        "labels": [{"name": config.labels.merge_hold}],
    }
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert fake_gh.pr_ready_calls == []  # never attempted -- operator hold wins
    # The PR-label hold is resolved from `pr` alone, with no additional
    # issue_view call beyond review()'s own unconditional per-pass fetch
    # (used for packet building) -- the hold check must not double-fetch.
    assert fake_gh.issue_view_calls == [123]
    assert fake_gh.pr_view(456)["isDraft"] is True  # still parked as a draft
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["prs"]["456"]["draft_ready_held_reason"] == "operator_merge_hold"
    recorded = query_events(paths.state_file, kind="draft_pr_ready_held")
    assert len(recorded) == 1
    assert recorded[0]["payload"]["pr_number"] == 456
    assert recorded[0]["payload"]["reason"] == "operator_merge_hold"
    assert recorded[0]["level"] == "warning"


def test_janitor_draft_only_block_merge_hold_on_issue_suppresses_auto_ready(
    tmp_path: Path,
) -> None:
    """Issue #820: the merge-hold label on the linked issue is an equally
    valid operator park signal (matching merge_ready's issue-side check)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True}
    fake_gh.issues[0]["labels"] = [
        {"name": config.labels.ready},
        {"name": config.labels.merge_hold},
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert fake_gh.pr_ready_calls == []
    assert fake_gh.pr_view(456)["isDraft"] is True
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["prs"]["456"]["draft_ready_held_reason"] == "operator_merge_hold"
    recorded = query_events(paths.state_file, kind="draft_pr_ready_held")
    assert len(recorded) == 1
    assert recorded[0]["payload"]["reason"] == "operator_merge_hold"


@pytest.mark.parametrize("degraded_payload", [{}, {"number": 123}])
def test_janitor_draft_only_block_merge_hold_check_unavailable_fails_safe(
    tmp_path: Path, degraded_payload: dict[str, Any]
) -> None:
    """Issue #820 fail-safe: when the linked issue's payload can't yield a
    usable "labels" set (empty dict, or a dict missing the "labels" key),
    auto-ready must be suppressed exactly like a confirmed hold -- never
    treated as "no hold" just because the check couldn't be evaluated.
    Mirrors merge_ready's merge_hold_check_unavailable degraded-payload arm
    (test_merge_ready_mergequeue_hold_issue_degraded_payload_fails_closed)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class DegradedIssueViewGitHub(FakeGitHub):
        def issue_view(self, number: int):
            if number == 123:
                return degraded_payload
            return super().issue_view(number)

    fake_gh = DegradedIssueViewGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert fake_gh.pr_ready_calls == []  # fail safe: do not act
    assert fake_gh.pr_view(456)["isDraft"] is True
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["prs"]["456"]["draft_ready_held_reason"] == "merge_hold_check_unavailable"
    recorded = query_events(paths.state_file, kind="draft_pr_ready_held")
    assert len(recorded) == 1
    assert recorded[0]["payload"]["reason"] == "merge_hold_check_unavailable"


def test_janitor_draft_only_block_issue_view_raise_never_calls_pr_ready(
    tmp_path: Path,
) -> None:
    """Issue #820 fail-safe, raising arm: review() fetches the linked issue
    unconditionally near the top of the method (for packet building), before
    the janitor verdict is even computed. The merge-hold check added for
    #820 reuses that fetch rather than issuing its own redundant call (see
    the comment at the `is_draft_only_block` branch), so when the fetch
    itself raises, review() aborts before verdict computation and `gh pr
    ready` is never reached -- the pre-existing per-PR GitHubError handling
    in loop() (not a new in-branch code path) is what makes this fail safe.
    """
    from charlie_work.github import GitHubError as _GitHubError

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class RaisingIssueViewGitHub(FakeGitHub):
        def issue_view(self, number: int):
            if number == 123:
                raise _GitHubError("transient gh issue view failure")
            return super().issue_view(number)

    fake_gh = RaisingIssueViewGitHub()
    fake_gh.prs[0] = {**fake_gh.prs[0], "isDraft": True}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    with pytest.raises(_GitHubError):
        app.review(456)

    assert fake_gh.pr_ready_calls == []
    assert fake_gh.pr_view(456)["isDraft"] is True
    # No state was written -- the failure happened before any state_lock
    # section in review() runs for this pass.
    state = load_state(paths.state_file)
    assert "456" not in state.get("prs", {})


def test_janitor_draft_only_block_merge_hold_event_dedupes_and_refires_after_lift(
    tmp_path: Path,
) -> None:
    """Issue #820: the suppression event must dedupe across identical-hold
    passes (cost-spirals.md discipline) but must re-fire if the hold is
    lifted (the PR gets auto-readied) and the operator re-parks it later --
    a stale dedup marker must not silently swallow the second park."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    hold_label = config.labels.merge_hold
    fake_gh.prs[0] = {
        **fake_gh.prs[0],
        "isDraft": True,
        "labels": [{"name": hold_label}],
    }
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    first = app.review(456)
    assert first.ok is False
    assert fake_gh.pr_ready_calls == []
    assert len(query_events(paths.state_file, kind="draft_pr_ready_held")) == 1

    # Same hold, second pass: dedup -- no new event.
    second = app.review(456)
    assert second.ok is False
    assert fake_gh.pr_ready_calls == []
    assert len(query_events(paths.state_file, kind="draft_pr_ready_held")) == 1

    # Operator lifts the hold: the PR is auto-readied as normal.
    fake_gh.prs[0]["labels"] = []
    lifted = app.review(456)
    assert lifted.data.get("draft_readied") is True
    assert fake_gh.pr_ready_calls == [456]
    assert fake_gh.pr_view(456)["isDraft"] is False
    assert len(query_events(paths.state_file, kind="draft_pr_ready_held")) == 1

    # Operator re-parks: re-drafts and re-applies the hold label. This must
    # emit a FRESH suppression event, not stay deduped against the first
    # park's stale reason.
    fake_gh.prs[0]["isDraft"] = True
    fake_gh.prs[0]["labels"] = [{"name": hold_label}]
    reparked = app.review(456)
    assert reparked.ok is False
    assert fake_gh.pr_ready_calls == [456]  # not called again
    held_events = query_events(paths.state_file, kind="draft_pr_ready_held")
    assert len(held_events) == 2
    assert all(e["payload"]["pr_number"] == 456 for e in held_events)

    # A DIFFERENT route out of the held state: the operator bypasses our
    # actuator entirely -- manually un-drafts via the GitHub UI *and* lifts
    # the hold in the same step -- landing on the packet-success path
    # (verdict.ok True) rather than the `gh pr ready` branch exercised
    # above. This path never re-enters `is_draft_only_block`, so it only
    # proves the fix if the reason was reconciled to None unconditionally
    # rather than by a clear buried inside that branch.
    fake_gh.prs[0]["isDraft"] = False
    fake_gh.prs[0]["labels"] = []
    bypassed = app.review(456)
    assert bypassed.ok is True
    state_after_bypass = load_state(paths.state_file)
    assert state_after_bypass["prs"]["456"]["draft_ready_held_reason"] is None

    # Re-park a second time. If the packet-success pass above had left the
    # reason stale, this would wrongly dedupe against the pass-4 reason and
    # no third event would fire.
    fake_gh.prs[0]["isDraft"] = True
    fake_gh.prs[0]["labels"] = [{"name": hold_label}]
    reparked_again = app.review(456)
    assert reparked_again.ok is False
    assert fake_gh.pr_ready_calls == [456]  # still not called again
    held_events_final = query_events(paths.state_file, kind="draft_pr_ready_held")
    assert len(held_events_final) == 3
    assert all(e["payload"]["pr_number"] == 456 for e in held_events_final)


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


def test_janitor_required_check_failure_routes_to_rework(tmp_path: Path) -> None:
    """Issue #376: a definitive required-check failure on a linked issue routes to rework."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["decision"] == "request_changes"
    assert state["prs"]["456"]["status"] == "request_changes"
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert (123, config.labels.needs_rework) in fake_gh.labels_added

    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    assert rework_prompt.exists()
    prompt_text = rework_prompt.read_text(encoding="utf-8")
    assert "CI failed on Tests passed; push a fix" in prompt_text

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["summary"] == "CI failed on Tests passed; push a fix"


def test_janitor_required_check_failure_after_stale_packet_routes_to_rework(
    tmp_path: Path,
) -> None:
    """Issue #467: a stale review packet on disk must not block the automated
    check-failure rework path; review() must record the verdict against the
    live PR head, not the stale packet head."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    passing_checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    failing_checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    diff_a = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    diff_b = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+fix"
    fake_gh = FakeGitHubWithChecks(checks=passing_checks)
    fake_gh.diffs[456] = diff_a
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Round 1: clean review writes a packet at the original head.
    result1 = app.review(456)
    assert result1.ok is True
    packet = paths.prs / "pr-456" / "pr.json"
    assert packet.exists()
    assert json.loads(packet.read_text(encoding="utf-8"))["headRefOid"] == "sha-abc123"

    # Round 2: PR head advances and a required check fails.
    fake_gh.checks = failing_checks
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = diff_b

    result2 = app.review(456)

    assert result2.ok is True, result2.message
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["decision"] == "request_changes"
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-new-head"
    assert (123, config.labels.needs_rework) in fake_gh.labels_added

    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    assert rework_prompt.exists()
    assert "CI failed on Tests passed; push a fix" in rework_prompt.read_text(encoding="utf-8")

    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-new-head"
    assert decision["reviewed_head_source"] == "live"


def test_janitor_required_check_failure_without_linked_issue_stays_blocked(
    tmp_path: Path,
) -> None:
    """Issue #376: a check-failure PR with no linked issue still dead-ends at the janitor gate."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    # Remove every issue reference so linked_issue_number returns None.
    fake_gh.prs[0]["headRefName"] = "misc/fix-search"
    fake_gh.prs[0]["title"] = "fix search"
    fake_gh.prs[0]["body"] = "No issue reference here."
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert "123" not in state.get("issues", {})
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


def test_janitor_required_check_infra_failure_stays_blocked(tmp_path: Path) -> None:
    """Issue #376: an infrastructure check failure (CANCELLED) is never routed
    to code-fix rework -- that part of this test's original intent is
    unchanged. Issue #841: unlike issue #376's era, an infra failure no
    longer sits in janitor_blocked forever with zero remediation -- this
    fixture's check has no parseable Actions run id (no `link` field) at
    all, so it cannot be auto-retried and correctly escalates to a human on
    the very first pass rather than looping/blocking indefinitely (the bug
    issue #841 fixes). A check that DOES carry a run id instead gets one
    auto-rerun first -- see
    test_janitor_infra_failed_first_cancel_triggers_rerun_without_failed_flag.

    Issue #1266: this escalates through the same infra_rerun_cap_exceeded
    site as the cap-exhaustion case, which is a mechanical reason, so it
    lands agent:operator-queue, not agent:human-needed.
    """
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "CANCELLED"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("infra_escalated") is True
    state = load_state(paths.state_file)
    # Still never routed to the code-fix rework path -- issue #376's original
    # assertion, unchanged.
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    # Issue #841: escalated to a human instead of blocking silently forever.
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["issues"]["123"]["status"] == "escalated"
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


# ---------------------------------------------------------------------------
# Issue #1383: infra_blocked routing -- AC1 through AC4
# ---------------------------------------------------------------------------


def test_janitor_required_check_repeated_failure_escalates(tmp_path: Path) -> None:
    """Issue #376: repeated check-failure reworks escalate via the
    request_changes cap. Issue #1266: max_rework_cycles_exceeded is
    mechanical, so this lands agent:operator-queue, not agent:human-needed."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    fake_gh = FakeGitHubWithChecks(checks=checks)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result1 = app.review(456)
    assert result1.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["request_changes_count"] == 1

    fake_gh.pr_head_shas[456] = "sha-2"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+fix1"
    )
    result2 = app.review(456)
    assert result2.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["request_changes_count"] == 2

    fake_gh.pr_head_shas[456] = "sha-3"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+fix2"
    )
    result3 = app.review(456)
    assert result3.ok is True
    assert result3.data["escalated"] is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


def test_janitor_required_check_failure_noop_does_not_reroute(tmp_path: Path) -> None:
    """Issue #376: a check-failure rework that produced no new content is not re-reviewed."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    fake_gh = FakeGitHubWithChecks(checks=checks)
    diff_text = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    fake_gh.diffs[456] = diff_text
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result1 = app.review(456)
    assert result1.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    request_count = state["prs"]["456"]["request_changes_count"]
    needs_rework_count = fake_gh.labels_added.count((123, config.labels.needs_rework))

    result2 = app.review(456)
    assert result2.ok is False
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["prs"]["456"]["request_changes_count"] == request_count
    assert fake_gh.labels_added.count((123, config.labels.needs_rework)) == needs_rework_count
    assert any("unchanged" in f.lower() for f in result2.data["janitor_failures"])


class FakeGitHubWithRerunCapture(FakeGitHubWithChecks):
    """FakeGitHub that captures gh run rerun calls and can simulate failures."""

    def __init__(
        self,
        checks: list[dict[str, Any]] | None = None,
        *,
        rerun_ok: bool = True,
        rerun_error: str = "This workflow run cannot be retried",
    ) -> None:
        super().__init__(checks)
        self.rerun_ok = rerun_ok
        self.rerun_error = rerun_error
        self.rerun_calls: list[list[str]] = []

    def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):  # noqa: ANN202
        if len(args) >= 2 and args[0] == "run" and args[1] == "rerun":
            self.rerun_calls.append(list(args))
            if self.rerun_ok:
                return "DRY-RUN: gh run rerun " + " ".join(args[2:])
            return github_module.GitHubRunResult(
                ok=False,
                returncode=1,
                stdout="",
                stderr=self.rerun_error,
                value=None,
                error=self.rerun_error,
            )
        return super().run(args, json_output=json_output, allow_failure=allow_failure)


def test_janitor_required_check_first_failure_triggers_rerun(tmp_path: Path) -> None:
    """Issue #391: first required-check failure triggers one auto-rerun and defers rework."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("rerun_run_ids") == [12345]
    assert len(fake_gh.rerun_calls) == 1
    assert fake_gh.rerun_calls[0][:3] == ["run", "rerun", "12345"]
    assert "--failed" in fake_gh.rerun_calls[0]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["check_rerun_attempts"] == {"sha-abc123": {"Tests passed": [12345]}}
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    assert "123" not in state.get("issues", {})


def test_janitor_required_check_second_failure_routes_to_rework(tmp_path: Path) -> None:
    """Issue #391: the same check failing again on the same head is definitive and routes to rework.

    Issue #1258 extends this test (rather than adding a parallel one) to also
    cover: (a) AC2 -- the pre-existing sole-failure short-circuit and its
    one-time flake-debounce rerun are unchanged by the new co-occurring-
    failure branch (the rerun fires exactly once, on pass 1, never again on
    pass 2's definitive failure); (b) AC4 -- the new
    ``review_dispatch_skipped_ci_red`` provenance kind is emitted exactly
    once, alongside (additive to) this short-circuit's pre-existing
    ``record_review``-driven routing, tagged ``co_occurring: False`` because
    the required-check failure is this PR's sole janitor failure.
    """
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result1 = app.review(456)
    assert result1.ok is False
    assert result1.data.get("rerun_run_ids") == [12345]
    assert len(fake_gh.rerun_calls) == 1
    # Pass 1 is the one-time flake-debounce rerun itself, not the definitive
    # short-circuit -- no provenance event yet.
    assert query_events(paths.state_file, kind="review_dispatch_skipped_ci_red") == []

    result2 = app.review(456)
    assert result2.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["decision"] == "request_changes"
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert (123, config.labels.needs_rework) in fake_gh.labels_added
    # No additional rerun was triggered on the second pass: AC2's "exactly
    # one rerun, not zero, not twice" across both passes.
    assert len(fake_gh.rerun_calls) == 1

    ci_red_events = query_events(paths.state_file, kind="review_dispatch_skipped_ci_red")
    assert len(ci_red_events) == 1
    payload = ci_red_events[0]["payload"]
    assert payload["pr_number"] == 456
    assert payload["issue_number"] == 123
    assert payload["failed_required_checks"] == ["Tests passed"]
    assert payload["co_occurring"] is False
    assert payload["co_occurring_failures"] == []


def test_janitor_required_check_failure_with_co_occurring_body_failure_routes_to_rework(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1258 (AC3): a required-check failure that is NOT the PR's sole
    janitor failure -- here, co-occurring with an empty PR body
    (``_check_body``) -- used to fall straight through
    ``is_check_failure_block`` (which requires the check failure be the SOLE
    blocker) into the passive ``janitor_blocked`` dead end: neither reviewed
    nor routed to rework, silently re-logging the same failure set forever.
    This must now route to rework via the same ``record_review(request_
    changes)`` machinery the sole-failure short-circuit uses, naming BOTH the
    failing check and the co-occurring janitor failure, and it must never
    launch a reviewer.

    The launch-avoidance assertion drives the real, end-to-end pipeline
    (``review()`` then ``dispatch_reviews()``), not just ``review()`` in
    isolation: launch happens in ``dispatch_reviews`` (workflow.py), which a
    ``review()``-only test never reaches, so a bare ``assert launched == []``
    against a monkeypatch that method can't trigger would pass for ANY
    mutation -- an inert control (see the AC1 sibling test immediately below,
    which drives the identical seam and gets ``launched_count == 1``; that is
    the positive-control proof this guard is live and this zero is real).
    """
    config = dataclasses.replace(
        _required_checks_config(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    # Trip _check_body's "PR body is empty" failure alongside the red check.
    # This co-occurring failure is independent of issue_number binding
    # (unlike a missing-linked-issue fixture, which review() itself requires
    # non-None before reaching ANY routing branch -- see the janitor.py
    # cross-reference in _check_linked_issue) and is not a merge conflict or
    # a no-op-rework, so it exercises exactly the new branch, not the
    # existing sole-failure short-circuit or the merge-conflict/no-op-rework
    # routing block.
    fake_gh.prs[0]["body"] = ""
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    launched = _fail_if_launched(monkeypatch)

    result = app.review(456)

    assert result.ok is True
    assert launched == []

    # Drive the real launch seam: no packet was written (the PR never left
    # the janitor-blocked/rework path), so dispatch_reviews must select and
    # launch nothing. This is what makes ``launched == []`` above meaningful
    # rather than vacuous.
    dispatch_result = app.dispatch_reviews()
    assert dispatch_result.ok is True
    assert dispatch_result.data["launched_count"] == 0
    assert dispatch_result.data["selected_count"] == 0
    assert launched == []

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["decision"] == "request_changes"
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert (123, config.labels.needs_rework) in fake_gh.labels_added

    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    assert rework_prompt.exists()
    prompt_text = rework_prompt.read_text(encoding="utf-8")
    assert "CI failed on Tests passed" in prompt_text
    assert "PR body is empty" in prompt_text

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert "Tests passed" in decision["summary"]
    assert "PR body is empty" in decision["summary"]
    assert decision["required_changes"] == ["PR body is empty"]

    ci_red_events = query_events(paths.state_file, kind="review_dispatch_skipped_ci_red")
    assert len(ci_red_events) == 1
    payload = ci_red_events[0]["payload"]
    assert payload["pr_number"] == 456
    assert payload["issue_number"] == 123
    assert payload["failed_required_checks"] == ["Tests passed"]
    assert payload["co_occurring"] is True
    assert payload["co_occurring_failures"] == ["PR body is empty"]


def test_janitor_required_check_failure_with_co_occurring_infra_failure_stays_blocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1258 (AC3 carve-out): unlike AC3's ``_check_body`` co-occurring
    failure immediately above, a genuine required-check FAILURE co-occurring
    with an INFRA-failed required check (CANCELLED/TIMED_OUT, issue #841/#847)
    must NOT route through the new ``is_co_occurring_check_failure_block``
    branch -- it has its own dedicated infra-rerun/escalation remediation
    that this fix must not shadow or double-dispatch against. This is the
    same combination the pre-existing
    ``test_janitor_mixed_genuine_failure_and_infra_failure_routes_to_rework_not_infra_rerun``
    (issue #847) pins from the infra side; this sibling pins it from #1258's
    side -- through the real ``review()`` + ``dispatch_reviews()`` pipeline,
    with the new provenance kind explicitly asserted ABSENT -- so the
    boundary is owned by this issue's own tests, not inferred from an
    unrelated suite.
    """
    config = dataclasses.replace(
        _required_checks_config(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "state": "CANCELLED"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    launched = _fail_if_launched(monkeypatch)

    result = app.review(456)

    # Falls through to the passive janitor_blocked path, unchanged by this
    # diff -- this combination is an accepted carve-out (infra remediation
    # owns it), not a new rework route.
    assert result.ok is False
    assert launched == []

    dispatch_result = app.dispatch_reviews()
    assert dispatch_result.ok is True
    assert dispatch_result.data["launched_count"] == 0
    assert dispatch_result.data["selected_count"] == 0
    assert launched == []

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert state["prs"]["456"].get("decision") is None
    failures = state["prs"]["456"]["janitor_failures"]
    assert any("Tests passed" in f for f in failures)
    assert any("infrastructure" in f.lower() and "Lint & Format" in f for f in failures)

    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    assert not rework_prompt.exists()

    # The new co-occurring-failure provenance kind must NOT fire for this
    # carve-out -- it is reserved for the code-fixable branch above.
    assert query_events(paths.state_file, kind="review_dispatch_skipped_ci_red") == []


def test_co_occurring_ci_red_branch_stays_inside_janitor_ok_gate() -> None:
    """Issue #1258 (AC8): a mutation-testing gap found in review that AC8's
    literal construction (disable the co-occurring branch's own boolean
    guard, expect a launch) cannot be satisfied against this architecture,
    and explains why, before pinning the mutation that CAN be.

    Why the local mutation is structurally inert: ``janitor.run_janitor``
    (janitor.py) always appends "Required check(s) failed: ..." to
    ``failures`` whenever ``failed_required_checks`` is truthy, and
    ``JanitorVerdict.ok = not failures`` -- so ``verdict.ok`` is NEVER True
    while CI is red, sole or co-occurring. ``review()``'s
    ``if not verdict.ok:`` gate (workflow.py) is therefore always entered on
    red CI, and every branch inside it is an early ``return`` ending in an
    UNCONDITIONAL default that overwrites ``status`` to ``"janitor_blocked"``
    and returns ``CommandResult(False, ...)`` before the packet-write/
    dispatch code textually after (i.e. outside) the gate is ever reached.
    Disabling ``is_co_occurring_check_failure_block``'s own guard therefore
    cannot produce a ``launch_claude_worker`` call -- it can only fall
    through to the pre-existing ``janitor_blocked`` stall, which
    ``test_janitor_required_check_failure_with_co_occurring_body_failure_routes_to_rework``'s
    ``launched == []`` assertion cannot distinguish from the branch actually
    firing.

    The mutation that DOES reach a launch is hoisting a red-CI exclusion out
    to the outer gate itself, e.g. rewriting
    ``if not verdict.ok:`` as
    ``if not verdict.ok and not bool(verdict.failed_required_checks):`` --
    that skips the fail-safe default entirely for every red-CI PR (sole or
    co-occurring) and falls through to the packet-write/dispatch path.
    Applied by hand against this diff and reverted immediately (not part of
    this suite's harness -- AST source-mutation isn't a fixture here), it
    made
    ``test_janitor_required_check_failure_with_co_occurring_body_failure_routes_to_rework``
    fail exactly at the ``_fail_if_launched`` fake's
    ``launch_claude_worker`` call:
    ``AssertionError: launch_claude_worker must not be called on red CI``,
    raised from ``dispatch_reviews`` -- a real, reachable launch-on-red-CI,
    not a hypothetical one.

    This test is the permanent guard against that specific hoist: it
    AST-scans ``review()`` and asserts (a) the outer gate's test is exactly
    ``not verdict.ok`` with no additional ``and``/``or`` operand, and (b) the
    ``is_co_occurring_check_failure_block`` branch stays lexically nested
    inside that outer gate's body rather than becoming a sibling of it (or
    being folded into its condition). Either change is exactly the refactor
    mistake that would reopen this gap; this test fails CI the moment either
    lands, rather than relying on a mutation that this architecture makes
    unreachable at the branch's own guard.

    Both assertions were verified live (positive control), applied by hand
    against this diff and reverted immediately (confirmed byte-identical via
    diff against a pre-mutation backup) -- neither is a mutation this suite
    runs automatically:
    - Rewriting the outer gate as
      ``if not verdict.ok and not bool(verdict.failed_required_checks):``
      makes the AST node a ``BoolOp``, not the ``UnaryOp``-wrapping-
      ``Attribute`` shape ``is_outer_gate`` matches, so ``outer_gates``
      drops to 0 and assertion (a) (``len(outer_gates) == 1``) fires:
      "found 0 -- a rewrite changed the outer janitor-blocked gate's shape".
    - Dedenting the ``is_co_occurring_check_failure_block`` ``if``-statement
      (and its body) by one level so it becomes a sibling statement
      immediately after the outer gate's closing ``)`` -- textually after,
      not inside, ``if not verdict.ok:`` -- still parses (this is valid
      Python) and still yields exactly one ``co_occurring_ifs`` match, so
      assertion (a) and the count check in (b) both stay green; it is
      specifically the ``nested_inside_outer_gate`` assertion that fires:
      "must stay lexically nested inside `if not verdict.ok:` ... or moved
      to be a sibling of it". This is the assertion that actually guards
      the sibling-hoist shape, distinct from the one guarding the
      condition-hoist shape above.
    """
    import ast

    src_path = Path(__file__).parents[1] / "src" / "charlie_work" / "workflow.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_path))

    review_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "review":
            review_fn = node
            break
    assert review_fn is not None, "could not find review() -- a rename invalidated this probe"

    def is_outer_gate(stmt: ast.AST) -> bool:
        if not isinstance(stmt, ast.If):
            return False
        test = stmt.test
        return (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Attribute)
            and test.operand.attr == "ok"
            and isinstance(test.operand.value, ast.Name)
            and test.operand.value.id == "verdict"
        )

    outer_gates = [stmt for stmt in ast.walk(review_fn) if is_outer_gate(stmt)]
    assert len(outer_gates) == 1, (
        "expected exactly one `if not verdict.ok:` gate (no additional and/or "
        f"operand) in review(), found {len(outer_gates)} -- a rewrite changed "
        "the outer janitor-blocked gate's shape"
    )
    outer_gate = outer_gates[0]

    def has_co_occurring_guard(stmt: ast.AST) -> bool:
        if not isinstance(stmt, ast.If):
            return False
        names = {n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)}
        return "is_co_occurring_check_failure_block" in names

    co_occurring_ifs = [stmt for stmt in ast.walk(review_fn) if has_co_occurring_guard(stmt)]
    assert len(co_occurring_ifs) == 1, (
        "expected exactly one `is_co_occurring_check_failure_block` guard in "
        f"review(), found {len(co_occurring_ifs)} -- a rename/duplication invalidated this probe"
    )
    co_occurring_if = co_occurring_ifs[0]

    nested_inside_outer_gate = any(
        stmt is co_occurring_if for stmt in ast.walk(outer_gate) if stmt is not outer_gate
    )
    assert nested_inside_outer_gate, (
        "the co-occurring CI-red branch must stay lexically nested inside "
        "`if not verdict.ok:`, not hoisted into the outer gate's own condition "
        "(e.g. `if not verdict.ok and not is_co_occurring_check_failure_block:`) "
        "or moved to be a sibling of it -- either change skips the fail-safe "
        "janitor_blocked default and is the one refactor mistake that makes "
        "launch_claude_worker reachable on red CI (confirmed by hand-mutation, "
        "see this test's docstring)"
    )


def test_janitor_all_checks_green_dispatches_reviewer_ci_red_kind_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1258 (AC1): the green-checks path is unaffected by the new
    co-occurring-failure branch and the new provenance kind -- both guards
    require ``verdict.failed_required_checks`` to be truthy, which an
    all-green PR never has, so neither new branch's body executes at all.
    Exercises the real, end-to-end pipeline (``review()`` packet-build then
    ``dispatch_reviews()`` launch), not just ``review()`` in isolation, so
    "the reviewer is launched with the same command as pre-diff" is an
    actual launch-call assertion, not merely an absence-of-packet inference.
    """
    config = dataclasses.replace(
        _required_checks_config(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = paths.prs / "pr-456" / "review-prompt.md"
    assert packet.exists()
    # The janitor gate never blocked this PR, so none of its routing kinds
    # (old or new) fired.
    assert query_events(paths.state_file, kind="review_dispatch_skipped_ci_red") == []
    assert not any(
        e["kind"] == "janitor_gate" for e in load_state(paths.state_file).get("events", [])
    )

    launched: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append((args, kwargs))
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    dispatch_result = app.dispatch_reviews()

    assert dispatch_result.ok is True
    assert dispatch_result.data["launched_count"] == 1
    assert len(launched) == 1
    _launch_args, launch_kwargs = launched[0]
    assert launch_kwargs.get("review") is True
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["review_dispatch_status"] == "review_dispatch_dispatched"
    # Pre-existing dispatch-lane kinds still fire (additive, not replaced).
    assert query_events(paths.state_file, kind="review_dispatch_claim") != []
    # The new CI-red kind is scoped to the janitor gate and never fires on
    # the dispatch-claim/launch path itself (also AC6's redundant-gate claim,
    # from the launch side).
    assert query_events(paths.state_file, kind="review_dispatch_skipped_ci_red") == []


def test_scope_fence_no_verdict_source_added() -> None:
    """Issue #1258 (AC7, structural half): this item's diff must add NO
    ``verdict_source`` field/enum anywhere in the new CI-red gate (W8's
    "ci_gate_auto_reject provenance enum" -- W8 lands after W1).

    Originally this test also pinned ``ReviewConfig.stale_checks_grace_minutes``
    / ``max_retriggers`` absent, guarding against W1 (or #1258 itself)
    re-adding W17's fields prematurely. W17 (issue #1274, this same lane) has
    now landed those two fields on ``ReviewConfig`` on purpose -- see
    ``ReviewConfig.stale_checks_grace_minutes``/``stale_checks_max_retriggers``
    and their loader validation block in config.py. That half of this test is
    therefore retired; the ``verdict_source`` guard below is unrelated
    (W8/#1258 concern) and still applies.

    ``verdict_source`` already exists elsewhere in this codebase
    (``_reap_review_verdicts`` provenance, an unrelated pre-existing
    mechanism -- see workflow.py's own field of that name) -- this test does
    not (and must not) assert the string is absent from the whole repo, only
    that ``JanitorVerdict`` (the structure this item actually touches) never
    gained the field.
    """
    from charlie_work.janitor import JanitorVerdict

    janitor_verdict_fields = {f.name for f in dataclasses.fields(JanitorVerdict)}
    assert "verdict_source" not in janitor_verdict_fields


def test_missing_checks_only_pr_falls_through_to_janitor_blocked_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1258 (AC7, behavioral half): a PR whose required checks are
    entirely MISSING (never reported), as opposed to FAILED, is the
    pre-existing "absent checks" code path this item must leave untouched
    -- it is classification-only (``is_missing_checks_only_block``, issue
    #1133) with no retrigger, and W17's not-yet-built
    stale-checks-grace/retrigger policy is what would eventually act on it.

    Both of this item's new/extended gates require
    ``verdict.failed_required_checks`` to be truthy
    (``is_check_failure_block`` and the new co-occurring branch alike) --
    a purely-missing check leaves that tuple empty, so neither can fire by
    construction. This PR must fall straight through to the passive
    ``janitor_blocked`` bookkeeping exactly as it did before this item's
    diff: no ``record_review`` decision, no ``review_dispatch_skipped_ci_red``
    event, no reviewer launch.

    ``review_dispatch`` is explicitly enabled here (``_required_checks_config()``
    alone leaves it at its ``enabled=False`` default) and the launch seam is
    driven for real via ``_fail_if_launched`` -- with dispatch left disabled,
    ``dispatch_reviews()`` returns ``launched_count == 0`` on its very first
    line for every fixture, sole-failure and co-occurring alike, which would
    make that assertion pass for any mutation of the gate. The positive
    control proving this zero is real, not vacuous, is the AC1 sibling
    ``test_janitor_all_checks_green_dispatches_reviewer_ci_red_kind_absent``,
    which drives the identical enabled seam and gets ``launched_count == 1``.
    """
    config = dataclasses.replace(
        _required_checks_config(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    launched = _fail_if_launched(monkeypatch)

    result = app.review(456)

    assert result.ok is False
    state = load_state(paths.state_file)
    pr_state = state["prs"]["456"]
    assert pr_state["status"] == "janitor_blocked"
    assert pr_state["is_missing_checks_only_block"] is True
    # Neither the pre-existing sole-failure short-circuit nor the new
    # co-occurring branch fired -- no request_changes decision was recorded.
    assert "decision" not in pr_state
    assert query_events(paths.state_file, kind="review_dispatch_skipped_ci_red") == []

    # No packet was written (the janitor gate blocked before packet-build),
    # so dispatch_reviews has structurally nothing to launch for this PR --
    # driven for real, with dispatch enabled and the launch seam wired to
    # fail loudly, not inferred from a disabled-dispatch early return.
    dispatch_result = app.dispatch_reviews()
    assert dispatch_result.ok is True
    assert dispatch_result.data["launched_count"] == 0
    assert dispatch_result.data["selected_count"] == 0
    assert launched == []


def test_janitor_required_check_rerun_api_error_falls_through_to_rework(tmp_path: Path) -> None:
    """Issue #391: a rerun API error surfaces as an event and falls through to rework."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        rerun_ok=False,
    )
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["decision"] == "request_changes"
    # The rerun attempt was not persisted because the API call failed.
    assert "check_rerun_attempts" not in state["prs"]["456"]
    assert any(event["kind"] == "flake_rerun_failed" for event in state.get("events", []))


def test_janitor_required_check_rerun_refused_while_sibling_running_defers_to_next_pass(
    tmp_path: Path,
) -> None:
    """Issue #992: a flake rerun refused because the containing workflow run is
    still in progress must defer to the next pass, not fall through to a
    request_changes verdict. The rerun attempt must not be consumed so a later
    pass can retry once the run has completed."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fixture = Path(__file__).parent / "fixtures" / "gh_run_rerun_already_running.json"
    rerun_error = json.loads(fixture.read_text(encoding="utf-8"))["error"]
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        rerun_ok=False,
        rerun_error=rerun_error,
    )
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("already_running") is True
    assert result.data.get("rerun_run_ids") == [12345]
    assert fake_gh.rerun_calls == [["run", "rerun", "12345", "--failed"]]
    state = load_state(paths.state_file)
    # The rerun attempt was not persisted and the PR was not routed to rework.
    assert "check_rerun_attempts" not in state["prs"].get("456", {})
    assert "decision" not in state["prs"].get("456", {})
    assert "request_changes_count" not in state["prs"].get("456", {})
    assert state.get("issues", {}).get("123", {}).get("status") != "rework_requested"
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    assert any(event["kind"] == "flake_rerun_failed" for event in state.get("events", []))


# Infra-failed (CANCELLED/INFRA_FAILURE/TIMED_OUT) required-check auto-rerun +
# escalation (issue #841). Before this, classify_check_failures only ever
# iterated summary.failed (a code push can't fix an infra kill), so a PR
# blocked on infra_failed sat there forever behind only a diagnostic
# merge_failed_attempt_alarm event -- no rerun, no rework, no escalation.


def test_janitor_infra_failed_first_cancel_triggers_rerun_without_failed_flag(
    tmp_path: Path,
) -> None:
    """Criterion 1: a CANCELLED required check whose run is the current head
    becomes eligible for exactly one auto-rerun. Also asserts the rerun is
    dispatched WITHOUT --failed: the job never completed (cancelled, not
    failed), so --failed's "rerun the failed jobs in this run" semantics do
    not apply -- verified against classify_check_failures' own --failed call,
    which this must NOT copy verbatim."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "CANCELLED", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data.get("infra_rerun_run_ids") == [12345]
    assert len(fake_gh.rerun_calls) == 1
    assert fake_gh.rerun_calls[0] == ["run", "rerun", "12345"]
    assert "--failed" not in fake_gh.rerun_calls[0]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_attempts"] == {
        "sha-abc123": {"Tests passed": {"12345": 1}}
    }
    assert any(event["kind"] == "infra_rerun_triggered" for event in state.get("events", []))
    # Not escalated, not routed to rework -- this pass only triggered the rerun.
    assert (123, config.labels.human_needed) not in fake_gh.labels_added
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    assert "123" not in state.get("issues", {})


def test_janitor_infra_failed_cap_exhausted_escalates_to_operator_queue(tmp_path: Path) -> None:
    """Criterion 2: once the infra rerun attempt cap (default 2) is exhausted,
    there is no code-fix rework path -- the PR must escalate instead of
    looping forever. Issue #1266: infra_rerun_cap_exceeded is a mechanical
    reason (a process-attempt-cap limit, not a judgment call), so it lands
    agent:operator-queue, not agent:human-needed."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "CANCELLED", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Pass 1: first cancel -> rerun attempt 1.
    result1 = app.review(456)
    assert result1.ok is False
    assert result1.data.get("infra_rerun_run_ids") == [12345]

    # Pass 2: STILL cancelled (the rerun itself timed out again, same run id,
    # per verified production behavior of `gh run rerun` reusing run ids) ->
    # rerun attempt 2 (at the default cap).
    result2 = app.review(456)
    assert result2.ok is False
    assert result2.data.get("infra_rerun_run_ids") == [12345]
    assert len(fake_gh.rerun_calls) == 2

    # Pass 3: STILL cancelled, cap (2) now exhausted -> escalate, not a third rerun.
    result3 = app.review(456)
    assert result3.ok is False
    assert result3.data.get("infra_escalated") is True
    assert len(fake_gh.rerun_calls) == 2  # no third rerun dispatched
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "infra_rerun_cap_exceeded"
    assert state["issues"]["123"]["reason_class"] == "mechanical"
    assert state["prs"]["456"]["status"] == "escalated"
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    assert any(event["kind"] == "infra_rerun_escalated" for event in state.get("events", []))


def test_janitor_infra_failed_rerun_api_error_does_not_consume_attempt_or_escalate(
    tmp_path: Path,
) -> None:
    """A `gh run rerun` API error must not silently consume the attempt (the
    next pass gets a fresh try) and must not escalate -- mirrors the genuine-
    failure flake-rerun error handling, minus the rework fallback (infra has
    none)."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    link = "https://github.com/owner/repo/actions/runs/12345/job/67890"
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "CANCELLED", "link": link},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        rerun_ok=False,
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    state = load_state(paths.state_file)
    # The attempt was not persisted because the API call failed.
    assert "infra_rerun_attempts" not in state["prs"].get("456", {})
    assert any(event["kind"] == "infra_rerun_failed" for event in state.get("events", []))
    # Not escalated -- an API error is not a genuine cap exhaustion.
    assert "123" not in state.get("issues", {})
    assert (123, config.labels.human_needed) not in fake_gh.labels_added


def test_janitor_mixed_genuine_failure_and_infra_failure_routes_to_rework_not_infra_rerun(
    tmp_path: Path,
) -> None:
    """Criterion 3: a genuine code FAILURE co-occurring with an infra-failed
    check must still route to rework -- the infra-rerun/escalation wiring
    must not shadow or double-dispatch. is_check_failure_block and
    is_infra_failure_block are both False here (mirrors the pre-existing
    is_draft-co-occurring precedent for is_check_failure_block), so this pass
    triggers neither remediation and falls through to the plain
    janitor_blocked path reporting both failures."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithRerunCapture(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "state": "CANCELLED"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert len(fake_gh.rerun_calls) == 0
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    failures = state["prs"]["456"]["janitor_failures"]
    assert any("Tests passed" in f for f in failures)
    assert any("infrastructure" in f.lower() and "Lint & Format" in f for f in failures)
    assert "123" not in state.get("issues", {})


def test_render_test_adequacy_section_unit() -> None:
    """Unit test for render_test_adequacy_section (issue #180)."""
    from charlie_work.janitor import TestAdequacyFacts
    from charlie_work.workflow import render_test_adequacy_section

    # Test with None (gate disabled)
    assert render_test_adequacy_section(None, ()) == ""

    # Test with populated facts
    facts = TestAdequacyFacts(
        added_product_loc=100,
        added_test_loc=50,
        assertion_count=10,
        test_files_changed=2,
        untested_product_files=("src/foo.py", "src/bar.py"),
        exempt=False,
        exempt_reason="",
    )
    warnings = ("Zero recognized assertions in added test lines",)

    section = render_test_adequacy_section(facts, warnings)
    assert "## Test-adequacy facts (Tier 1, deterministic)" in section
    assert "Added product LOC: 100" in section
    assert "Added test LOC: 50" in section
    assert "Assertion-bearing added test lines: 10" in section
    assert "Test files changed: 2" in section
    assert "Untested product files: src/foo.py, src/bar.py" in section
    assert "Zero recognized assertions in added test lines" in section

    # Test with empty warnings
    section_no_warnings = render_test_adequacy_section(facts, ())
    assert "Zero recognized assertions" not in section_no_warnings

    # Test with exempt claim
    facts_exempt = TestAdequacyFacts(
        added_product_loc=100,
        added_test_loc=0,
        assertion_count=0,
        test_files_changed=0,
        untested_product_files=(),
        exempt=True,
        exempt_reason="n/a - pure refactoring",
    )
    section_exempt = render_test_adequacy_section(facts_exempt, ())
    assert (
        'Test-exempt claim: "n/a - pure refactoring" (verify against the diff)' in section_exempt
    )


def test_test_adequacy_section_in_review_packet_when_enabled(tmp_path: Path) -> None:
    """Integration test: verify test_adequacy_section appears in review packet when gate is enabled and passes (issue #180)."""
    from unittest.mock import patch
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict

    config = OrchestratorConfig(
        test_adequacy=TestAdequacyConfig(
            enabled=True,
            exempt_marker="Test-exempt:",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # Mock check_test_adequacy to return a passing verdict with facts
    mock_facts = TestAdequacyFacts(
        added_product_loc=100,
        added_test_loc=50,
        assertion_count=10,
        test_files_changed=2,
        untested_product_files=(),
        exempt=False,
        exempt_reason="",
    )
    mock_verdict = TestAdequacyVerdict(
        ok=True,
        failures=(),
        warnings=(),
        facts=mock_facts,
    )

    with patch("charlie_work.workflow.check_test_adequacy", return_value=mock_verdict):
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)
        result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")

    # Verify the test-adequacy facts section appears in the packet
    assert "## Test-adequacy facts (Tier 1, deterministic)" in packet_text
    # Verify no unresolved placeholder
    assert "$test_adequacy_section" not in packet_text


def test_test_adequacy_section_not_in_review_packet_when_disabled(tmp_path: Path) -> None:
    """Integration test: verify test_adequacy_section does not appear in review packet when gate is disabled (issue #180)."""
    config = OrchestratorConfig(
        test_adequacy=TestAdequacyConfig(
            enabled=False,  # Gate disabled
            exempt_marker="Test-exempt:",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")

    # Verify the test-adequacy facts section does NOT appear in the packet
    assert "## Test-adequacy facts (Tier 1, deterministic)" not in packet_text
    # Verify no unresolved placeholder
    assert "$test_adequacy_section" not in packet_text


def test_render_static_probe_section_unit() -> None:
    """Unit test for render_static_probe_section (issues #1260/#1261)."""
    from charlie_work.diff_coverage_probe import (
        BranchCoverageFinding,
        StaticProbeVerdict,
        UnwiredSymbolFinding,
    )
    from charlie_work.workflow import render_static_probe_section

    # Disabled (probe never ran) -> "".
    assert render_static_probe_section(None) == ""

    # Enabled, zero findings, zero warnings -> explicit visible "no findings"
    # line, never a bare "" -- an advisory probe that goes silent on a clean
    # run must not read as "never ran".
    clean_section = render_static_probe_section(StaticProbeVerdict())
    assert clean_section == "Static probe: no findings.\n"

    # Enabled, internal error -> visible degradation warning, not silent-empty.
    degraded = StaticProbeVerdict(
        warnings=("static probe degraded: branch-coverage heuristic failed: boom",)
    )
    degraded_section = render_static_probe_section(degraded)
    assert "static probe degraded" in degraded_section

    # Enabled, findings present -> both W3 and W20 findings concatenated
    # into the one section.
    verdict = StaticProbeVerdict(
        branch_findings=(BranchCoverageFinding("src/foo.py", 3, 0, "no_test_adds"),),
        unwired_findings=(UnwiredSymbolFinding("helper", "src/bar.py", "function"),),
    )
    section = render_static_probe_section(verdict)
    assert "Branch-coverage heuristic (W3)" in section
    assert "src/foo.py" in section
    assert "Unwired-symbol probe (W20)" in section
    assert "helper" in section
    assert "src/bar.py" in section


def test_static_probe_section_not_in_review_packet_when_disabled(tmp_path: Path) -> None:
    """When coverage_probe.enabled=False (default), the computed section is
    empty and no dynamic probe content leaks into the packet. The STATIC
    '## Static probe' heading + rubric prose (W20 item 2) are permanent
    template text and remain present regardless -- mirrors the always-
    present '## Test adequacy' template heading precedent."""
    from charlie_work.config import CoverageProbeConfig

    config = OrchestratorConfig(coverage_probe=CoverageProbeConfig(enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = (
        "diff --git a/src/feature.py b/src/feature.py\n"
        "index 123..456 100644\n"
        "--- a/src/feature.py\n"
        "+++ b/src/feature.py\n"
        "@@ -1,2 +1,4 @@\n"
        " def feature():\n"
        "     pass\n"
        "+def new_feature(x):\n"
        "+    if x:\n"
        "+        return 1\n"
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")

    # Static template heading + rubric prose (W20 item 2) always present.
    assert "## Static probe" in packet_text
    assert "Name the production caller." in packet_text
    assert "Classify test strength." in packet_text
    assert "existence < type < status < value <" in packet_text
    # No dynamic probe content and no unresolved placeholder.
    assert "Branch-coverage heuristic (W3)" not in packet_text
    assert "Static probe: no findings." not in packet_text
    assert "$static_probe_section" not in packet_text


def test_static_probe_section_in_review_packet_when_enabled_with_findings(
    tmp_path: Path,
) -> None:
    """Integration test: findings from both probe halves land in
    $static_probe_section, adjacent to (not folded into) ## Test adequacy.

    ``repo_root`` (``tmp_path``) is given a real ``src/`` tree containing a
    file that does NOT reference the flagged symbol, so
    ``_collect_repo_referenced_names`` actually walks it and the line-295
    collision filter runs live (issue #1260/#1261 review finding A5) instead
    of short-circuiting on a missing ``src/`` directory. The flagged symbol
    uses a distinctive name (not ``helper``) precisely so it cannot
    accidentally collide with anything incidental in that tree.
    """
    from charlie_work.config import CoverageProbeConfig

    config = OrchestratorConfig(coverage_probe=CoverageProbeConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Real src/ tree under repo_root, unrelated to the diffed symbol below --
    # exercises the collision filter live rather than short-circuiting it.
    unrelated_src = tmp_path / "src" / "unrelated_module.py"
    unrelated_src.parent.mkdir(parents=True, exist_ok=True)
    unrelated_src.write_text(
        "def totally_unrelated_function(value):\n    return value * 2\n",
        encoding="utf-8",
    )

    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = (
        "diff --git a/src/feature.py b/src/feature.py\n"
        "index 123..456 100644\n"
        "--- a/src/feature.py\n"
        "+++ b/src/feature.py\n"
        "@@ -1,2 +1,5 @@\n"
        " def feature():\n"
        "     pass\n"
        "+def compute_shard_checksum(x):\n"
        "+    if x:\n"
        "+        return 1\n"
        "diff --git a/tests/test_feature.py b/tests/test_feature.py\n"
        "index 123..456 100644\n"
        "--- a/tests/test_feature.py\n"
        "+++ b/tests/test_feature.py\n"
        "@@ -1,2 +1,4 @@\n"
        " def test_existing():\n"
        "     pass\n"
        "+def test_compute_shard_checksum():\n"
        "+    assert compute_shard_checksum(True) == 1\n"
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")

    assert "## Static probe" in packet_text
    # compute_shard_checksum() is defined in src/feature.py, referenced only
    # from the test, and absent from the real (live-walked) src/ tree -- the
    # collision filter runs and correctly does not suppress it.
    assert "Unwired-symbol probe (W20)" in packet_text
    assert "compute_shard_checksum" in packet_text
    assert "$static_probe_section" not in packet_text
    assert packet_text.index("## Test adequacy") < packet_text.index("## Static probe")


def test_static_probe_section_no_findings_renders_visible_clean_line(tmp_path: Path) -> None:
    """Enabled + zero findings still renders visible text, not "" (mirrors
    render_test_adequacy_section's own always-visible-when-enabled shape)."""
    from charlie_work.config import CoverageProbeConfig

    config = OrchestratorConfig(coverage_probe=CoverageProbeConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = (
        "diff --git a/README.md b/README.md\n"
        "index 123..456 100644\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1 +1,2 @@\n"
        " Hello\n"
        "+World\n"
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")
    assert "Static probe: no findings." in packet_text


def test_static_probe_section_shows_visible_degradation_on_internal_error(
    tmp_path: Path, monkeypatch
) -> None:
    """An internal exception in the probe must render a visible warning
    line in the packet, never a silently empty section (design item 7)."""
    from charlie_work.config import CoverageProbeConfig

    def _boom(diff, config):
        raise ValueError("synthetic failure")

    monkeypatch.setattr("charlie_work.diff_coverage_probe.check_branch_coverage", _boom)

    config = OrchestratorConfig(coverage_probe=CoverageProbeConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")
    assert "static probe degraded" in packet_text
    assert "branch-coverage heuristic failed" in packet_text


def test_coverage_probe_never_called_when_disabled(tmp_path: Path, monkeypatch) -> None:
    """When coverage_probe.enabled=False (default), run_static_probe is
    never invoked -- mirrors test_review_test_adequacy_disabled_is_noop."""
    from charlie_work.config import CoverageProbeConfig

    calls = {"n": 0}

    def _fake_run_static_probe(diff, repo_root, config):
        calls["n"] += 1
        raise AssertionError("run_static_probe should not be called when disabled")

    monkeypatch.setattr("charlie_work.workflow.run_static_probe", _fake_run_static_probe)

    config = OrchestratorConfig(coverage_probe=CoverageProbeConfig(enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app.review(456)

    assert calls["n"] == 0
    assert result.ok is True


def test_reconcile_wiring_reports_clean_repo(tmp_path: Path) -> None:
    class QuietGitHub(FakeGitHub):
        def run(self, arguments, *, json_output=False, allow_failure=False):
            # Handle dependency API calls
            if "dependencies" in " ".join(arguments):
                return [] if json_output else ""
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
            # Handle dependency API calls
            if "dependencies" in " ".join(arguments):
                return [] if json_output else ""
            # paginated PR list from reconcile._fetch_prs
            if arguments[0] == "api" and "pulls?state=all" in arguments[1]:
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
            # paginated issue list from reconcile._fetch_issues
            if arguments[0] == "api" and "issues?state=all" in arguments[1]:
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
            # Handle dependency API calls
            if "dependencies" in " ".join(arguments):
                return [] if json_output else ""
            if arguments[0] == "api" and "pulls?state=all" in arguments[1]:
                return [self._pr]
            if arguments[0] == "api" and "issues?state=all" in arguments[1]:
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

        def close_issue(self, number: int) -> bool:
            # Mirror the label overrides above: a real close is visible to
            # the next issues snapshot, so flip the state this fake serves.
            ok = super().close_issue(number)
            if number == self._issue["number"]:
                self._issue["state"] = "CLOSED"
            return ok

    app = OrchestratorApp(
        tmp_path, runtime_paths(tmp_path, config.runtime.state_dir), config, DriftGitHub()
    )

    result = app.reconcile(fix=True)

    assert result.ok is True
    assert result.data["fixed"] is True
    assert result.data["drift_before"] == 1
    assert result.data["drift_after"] == 0
    assert result.data["remaining_drift"] == []


def test_reconcile_removes_mergequeue_label_via_full_stack(tmp_path: Path) -> None:
    """Wiring check for issue #819. Every other ``detect_mergequeue_not_approved``
    test (test_reconcile.py) calls the detector directly with a config it
    builds itself; none of them prove the detector is actually reached
    through ``app.reconcile()``. Every reconcile test *in this file* uses
    the default ``OrchestratorConfig()``, where ``auto_merge.mergequeue_label``
    is ``None`` -- so the detector's very first line (``if not
    mergequeue_label: return []``) short-circuits before doing anything,
    and green tests here would prove nothing about wiring (the exists/
    substantive/wired distinction). This test configures the label and
    drives a real ``request_changes``-at-head PR through
    ``app.reconcile(fix=True)`` end to end, asserting the label actually
    comes off via ``GitHub.remove_pr_label`` -- the same mechanical step
    that was missing when Aviator merged PR #695 over a standing
    request-changes verdict."""
    config = dataclasses.replace(
        OrchestratorConfig(),
        auto_merge=dataclasses.replace(
            OrchestratorConfig().auto_merge, mergequeue_label="mergequeue"
        ),
    )
    mergequeue_label = config.auto_merge.mergequeue_label
    assert mergequeue_label is not None

    class MergequeueGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self._pr = {
                "number": 819,
                "title": "fix",
                "url": "u",
                "headRefName": "agent/issue-819-x",
                "baseRefName": "main",
                "headRefOid": "sha-819-live",
                "body": "",
                "state": "OPEN",
                "labels": [{"name": mergequeue_label}],
                "isCrossRepository": False,
                "headRepositoryOwner": "owner",
                "baseRepositoryOwner": "owner",
            }
            self.pr_labels_removed: list[tuple[int, str]] = []

        def run(self, arguments, *, json_output=False, allow_failure=False):
            if "dependencies" in " ".join(arguments):
                return [] if json_output else ""
            if arguments[0] == "api" and "pulls?state=all" in arguments[1]:
                return [self._pr]
            if arguments[0] == "api" and "issues?state=all" in arguments[1]:
                return []
            return []

        def remove_pr_label(self, number: int, label: str) -> bool:
            self.pr_labels_removed.append((number, label))
            self._pr["labels"] = [item for item in self._pr["labels"] if item.get("name") != label]
            return True

    gh = MergequeueGitHub()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    pr_dir = paths.prs / "pr-819"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "sha-819-live"}),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, gh)

    app.reconcile(fix=True)

    assert gh.pr_labels_removed == [(819, mergequeue_label)]
    assert mergequeue_label not in [item.get("name") for item in gh._pr["labels"]]


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
            # Handle dependency API calls
            if "dependencies" in " ".join(arguments):
                return [] if json_output else ""
            if arguments[0] == "api" and "pulls?state=all" in arguments[1]:
                return []
            if arguments[0] == "api" and "issues?state=all" in arguments[1]:
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
    assert result.data["drift_before"] >= 1  # May be multiple if both adapters read the same issue
    assert result.data["drift_after"] >= 1
    assert len(result.data["remaining_drift"]) >= 1
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


def test_runtime_paths_warns_on_phantom_state_dir(tmp_path: Path, caplog: Any) -> None:
    """Issue #648: a state dir that exists with sibling artifacts but no
    state.json is a phantom signal — runtime_paths must warn (non-blocking)
    so the operator notices instead of seeing a silent 'all clear'."""
    from charlie_work.paths import runtime_paths

    state_dir = tmp_path / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    # Mimic the stray artifacts described in the issue.
    (state_dir / "events.db").write_bytes(b"")
    (state_dir / "state.json.lock").write_text("", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="charlie_work.paths"):
        runtime_paths(tmp_path, ".var/charlie-work")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(state_dir.resolve()) in warnings[0].message
    assert "state.json" in warnings[0].message


def test_runtime_paths_silent_without_sibling_artifacts(tmp_path: Path, caplog: Any) -> None:
    """Issue #648 review MINOR: a state dir that exists but has no state.json
    AND no sibling artifacts (events.db, state.json.lock) must NOT warn — it
    could be a pre-existing directory used for an unrelated purpose, not a
    phantom left by a misresolved invocation."""
    from charlie_work.paths import runtime_paths

    state_dir = tmp_path / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    # No sibling artifacts — just an empty dir.

    with caplog.at_level(logging.WARNING, logger="charlie_work.paths"):
        runtime_paths(tmp_path, ".var/charlie-work")

    assert not caplog.records


def test_runtime_paths_no_warn_for_absolute_unrelated_state_dir(
    tmp_path: Path, caplog: Any
) -> None:
    """Issue #648 review MINOR: an absolute state_dir pointing at a
    pre-existing directory without sibling artifacts must not trigger the
    phantom warning."""
    from charlie_work.paths import runtime_paths

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "some-file.txt").write_text("data", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="charlie_work.paths"):
        runtime_paths(tmp_path, str(unrelated))

    assert not caplog.records


def test_runtime_paths_silent_when_state_dir_absent(tmp_path: Path, caplog: Any) -> None:
    """A genuine first run has not created the state dir yet at runtime_paths
    call time — no phantom warning must fire."""
    from charlie_work.paths import runtime_paths

    with caplog.at_level(logging.WARNING, logger="charlie_work.paths"):
        runtime_paths(tmp_path, ".var/charlie-work")

    assert not caplog.records


def test_runtime_paths_silent_when_state_json_exists(tmp_path: Path, caplog: Any) -> None:
    """A populated state dir is the normal steady state — no warning."""
    from charlie_work.paths import runtime_paths

    state_dir = tmp_path / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="charlie_work.paths"):
        runtime_paths(tmp_path, ".var/charlie-work")

    assert not caplog.records


# --- adversarial-review fixes: regressions + coverage gaps ---------------------


def test_concurrent_dispatch_claims_prevent_double_launch(tmp_path: Path) -> None:
    """A dispatch_pending claim must block a second dispatch for the same issue."""
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(0)")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # First dispatch creates a dispatch_pending claim
    app.gh.prs[0]["state"] = "CLOSED"
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
    app.gh.prs[0]["state"] = "CLOSED"
    second_result = app.dispatch(limit=1)
    assert second_result.data["attempted_count"] == 0  # Blocked by claim

    # Verify the claim is still in place
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatch_pending"


def test_stale_dispatch_pending_claim_is_redispatchable(tmp_path: Path, monkeypatch) -> None:
    """A stale dispatch_pending claim (crashed phase-2) must be re-dispatchable."""
    from charlie_work.state import is_claim_stale

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(0)")),
        worker=WorkerRoleConfig(harness="command"),
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
    app.gh.prs[0]["state"] = "CLOSED"
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


def test_bootstrap_labels_descriptions_fit_github_limit(tmp_path: Path) -> None:
    """GitHub rejects label descriptions over 100 chars with HTTP 422.

    FakeGitHub doesn't enforce this, so a too-long description passes the
    other bootstrap tests while silently 422ing against the real API (the
    failure is swallowed by label_create's allow_failure=True) — 'complexity:high'
    shipped with a 121-char description and was missing from every repo as a
    result. Assert on the real descriptions directly so this can't recur.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.bootstrap_labels()

    too_long = {name: desc for name, _color, desc in fake_gh.labels_created if len(desc) > 100}
    assert too_long == {}


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


# ---------------------------------------------------------------------------
# Issue #1339: automatic startup label ensure (OrchestratorApp.ensure_labels)
# ---------------------------------------------------------------------------


def test_ensure_labels_creates_every_configured_label(tmp_path: Path) -> None:
    """ensure_labels is the automatic startup counterpart of bootstrap_labels.

    It must create every LabelConfig-derived label and record an ``ok`` event
    so convergence is observable in events.db without an operator running
    ``charlie bootstrap-labels``.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.ensure_labels()

    created = {name for name, _color, _desc in fake_gh.labels_created}
    assert created == set(config.labels.all)
    assert result.ok is True
    assert result.data["missing"] == []
    ok_events = query_events(paths.state_file, kind="label_ensure_ok")
    assert len(ok_events) == 1, ok_events
    assert ok_events[0]["level"] == "info"


def test_ensure_labels_extra_labelconfig_field_is_ensured(tmp_path: Path) -> None:
    """AC #3: a LabelConfig with an extra field results in that label being ensured.

    The ensure set must derive from LabelConfig fields, so a new field ships its
    label with no extra wiring. Verified by subclassing LabelConfig with an
    extra field included in ``all``.
    """

    @dataclasses.dataclass(frozen=True)
    class _ExtraLabelConfig(LabelConfig):
        extra_label: str = "agent:extra-test-field"

        @property
        def all(self) -> list[str]:
            return [*LabelConfig.all.fget(self), self.extra_label]  # type: ignore[misc]

    config = OrchestratorConfig(labels=_ExtraLabelConfig())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.ensure_labels()

    created = {name for name, _color, _desc in fake_gh.labels_created}
    assert "agent:extra-test-field" in created
    assert created == set(config.labels.all)
    assert result.ok is True


def test_ensure_labels_removal_of_field_does_not_delete(tmp_path: Path) -> None:
    """AC #3: removing a field from LabelConfig must not delete the label.

    ensure_labels only creates/updates; it never deletes. A label that exists
    on the repo but is no longer in ``LabelConfig.all`` must survive the ensure.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Pre-create a label that is NOT in the default LabelConfig.all, simulating
    # a label left behind after a field was removed from LabelConfig.
    orphan = "agent:removed-from-config"
    fake_gh.labels_created.append((orphan, "5319E7", "orphaned label"))
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.ensure_labels()

    assert result.ok is True
    live = {str(item.get("name") or "") for item in fake_gh.label_list()}
    # The orphan label survives — ensure never deletes.
    assert orphan in live


def test_ensure_labels_records_incomplete_event_on_missing(tmp_path: Path) -> None:
    """AC #2: ensure failures surface as events, not exceptions.

    When label_create silently fails (e.g. no auth), the ensure must record a
    ``label_ensure_incomplete`` warning event and return ok=False rather than
    raise.
    """

    class _FailingCreateGitHub(FakeGitHub):
        def label_create(self, label: str, color: str, description: str) -> None:
            pass  # silently drop all creates

        def label_list(self) -> list[dict[str, object]]:
            return []  # nothing was created

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, _FailingCreateGitHub())

    result = app.ensure_labels()

    assert result.ok is False
    assert result.data["missing"] == config.labels.all
    incomplete = query_events(paths.state_file, kind="label_ensure_incomplete")
    assert len(incomplete) == 1, incomplete
    assert incomplete[0]["level"] == "warning"


def test_ensure_labels_records_failed_event_on_list_error(tmp_path: Path) -> None:
    """AC #2: a verification (label_list) error surfaces as an error event."""
    from charlie_work.github import GitHubError

    class _ErrorListGitHub(FakeGitHub):
        def label_list(self) -> list[dict[str, object]]:
            raise GitHubError("could not list labels: HTTP 401")

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, _ErrorListGitHub())

    result = app.ensure_labels()

    assert result.ok is False
    assert "verification failed" in result.message
    failed = query_events(paths.state_file, kind="label_ensure_failed")
    assert len(failed) == 1, failed
    assert failed[0]["level"] == "error"


def test_ensure_labels_never_raises_on_unexpected_exception(tmp_path: Path) -> None:
    """AC #2: an unexpected exception from the gh impl is recorded, not raised."""

    class _RaisingCreateGitHub(FakeGitHub):
        def label_create(self, label: str, color: str, description: str) -> None:
            raise RuntimeError("boom")

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, _RaisingCreateGitHub())

    # Must not raise.
    result = app.ensure_labels()

    assert result.ok is False
    failed = query_events(paths.state_file, kind="label_ensure_failed")
    assert len(failed) == 1, failed
    assert failed[0]["level"] == "error"
    assert "RuntimeError" in failed[0]["payload"]["error"]


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


def test_branch_protection_caches_per_pass(monkeypatch, tmp_path: Path) -> None:
    """Issue #812: branch_protection() must cost exactly one `gh api` call per
    base ref per orchestrator pass, not one per PR -- N callers sharing a base
    (e.g. N open PRs against main in one merge_ready/broadcast-sweep pass) must
    collapse to a single underlying read. The cache lives in GitHub._list_cache
    (the same dict pr_list/issue_list already use) and is cleared only by
    invalidate_list_cache(), which the orchestrator calls once per pass.
    """
    calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        payload = json.dumps({"required_status_checks": {"strict": True}})
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = github_module.GitHub(repo_root=tmp_path)

    # Simulate N=5 PRs against the same base within one pass: 5 calls to the
    # method, but the underlying `gh api` subprocess must run exactly once.
    results = [gh.branch_protection("main") for _ in range(5)]
    assert all(r == {"required_status_checks": {"strict": True}} for r in results)
    assert len(calls) == 1
    # Pin the actual endpoint (and the {owner}/{repo} placeholder escaping),
    # not just "something got cached" -- a wrong URL would still pass a
    # call-count-only assertion.
    assert calls[0] == ["gh", "api", "repos/{owner}/{repo}/branches/main/protection"]

    # A different base ref is a distinct cache key, so it costs a fresh read.
    gh.branch_protection("develop")
    assert len(calls) == 2
    gh.branch_protection("develop")
    assert len(calls) == 2  # still cached

    # invalidate_list_cache() (called once at the top of every orchestrator
    # pass) must force a fresh read on the next call -- the cache is valid
    # only within a single pass, never leaking across passes.
    gh.invalidate_list_cache()
    gh.branch_protection("main")
    assert len(calls) == 3


def test_branch_protection_caches_failed_read_too(monkeypatch, tmp_path: Path) -> None:
    """A failed read (404/rate-limited) must also be cached as None for the
    rest of the pass -- otherwise every PR sharing a broken base ref retries
    the same doomed `gh api` call once each, turning one outage into N.
    """
    calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="HTTP 404: Not Found"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = github_module.GitHub(repo_root=tmp_path)

    assert gh.branch_protection("main") is None
    assert gh.branch_protection("main") is None
    assert gh.branch_protection("main") is None
    assert len(calls) == 1


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


def test_is_mutating_blocks_the_argv_delete_branch_actually_builds(
    monkeypatch, tmp_path: Path
) -> None:
    """#914/#917: `-X DELETE` classified as read-only, so `--dry-run` really deleted
    PR head branches.

    The argv is captured from `delete_branch` itself rather than written out by hand,
    so the gate cannot drift away from its most destructive caller: if someone changes
    how that call is spelled, this test follows it.
    """
    from charlie_work.github import _is_mutating

    captured: list[list[str]] = []
    # GitHub is a frozen dataclass -- patch the class, not the instance.
    monkeypatch.setattr(
        github_module.GitHub,
        "run",
        lambda self, args, **kwargs: captured.append(args) or "",
    )
    gh = github_module.GitHub(repo_root=tmp_path)

    assert gh.delete_branch("feature/x") is True
    assert len(captured) == 1
    assert _is_mutating(captured[0]) is True


def test_is_mutating_api_method_spellings_and_preserved_reads() -> None:
    """Every spelling `gh` accepts for a method must classify from that method, and
    an unparseable one must fail CLOSED.

    The read-only half is the half that protects `--dry-run` from being tightened into
    uselessness: these are real live call sites, and without them a future "just deny
    all `gh api`" change passes every other test in the suite.
    """
    from charlie_work.github import _is_mutating

    for mutating in (
        ["api", "-X", "DELETE", "repos/o/r/git/refs/heads/x"],
        ["api", "-X=DELETE", "repos/o/r/git/refs/heads/x"],
        ["api", "-XDELETE", "repos/o/r/git/refs/heads/x"],
        ["api", "-X", "POST", "repos/o/r/actions/runners/remove-token"],
        ["api", "--method", "PATCH", "repos/o/r/issues/1"],
        ["api", "--method=PUT", "repos/o/r/branches/main/protection"],
        ["api", "-X"],  # named but valueless -> fail closed, not open
        ["api", "--method"],
        ["api", "repos/o/r/issues", "-f", "title=x"],  # params switch gh to POST
        ["api", "repos/o/r/issues", "--field=labels[]=bug"],
        # pflag takes an attached shorthand value here too, exactly as for -X (#919).
        ["api", "repos/o/r/issues", "-ftitle=x"],
        ["api", "repos/o/r/issues", "-Flabels[]=bug"],
        ["api", "repos/o/r/issues", "-F"],
        ["api", "repos/o/r/issues", "--raw-field", "title=x"],
        ["api", "repos/o/r/issues", "--input", "body.json"],
        ["api", "repos/o/r/issues", "--input=-"],
    ):
        assert _is_mutating(mutating) is True, mutating

    for readonly in (
        ["api", "rate_limit"],
        ["api", "repos/o/r/commits/abc/check-runs"],
        ["api", "repos/o/r/compare/main...topic"],
        ["api", "repos/o/r/branches/main/protection"],
        ["api", "-X", "GET", "repos/o/r/issues"],
        ["api", "--method=HEAD", "repos/o/r"],
        # A header is not a method -- github.py:1033 fetches a diff this way.
        ["api", "repos/o/r/pulls/1", "-H", "Accept: application/vnd.github.v3.diff"],
        # Other shorthands must not be swept up by the -f/-F prefix match (#919).
        ["api", "repos/o/r/issues", "-q", ".[].number"],
        ["api", "repos/o/r/issues", "-t", "{{.number}}"],
        ["api", "repos/o/r/issues", "--paginate"],
    ):
        assert _is_mutating(readonly) is False, readonly


def test_token_minting_posts_do_not_retry_post_send_failures() -> None:
    """Credential-minting POSTs must retry only on provable pre-send failures.

    `_is_mutating` has a second consumer besides the --dry-run gate: `run()` feeds it
    to `_should_retry`, which grants reads an unconditional retry on any transient
    error and restricts mutations to pre-connection errors, so a request that may
    already have been applied is never re-sent.

    Before #918 these two argvs classified as *reads* (the `-X` spelling fell through
    the old enumeration), which put credential minting on the unconditional-retry
    path: a post-send timeout on a request GitHub had actually served would mint a
    second token. #918 fixed that as a side effect of the --dry-run work without
    naming it, so pin it here -- a future reclassification of `-X POST` would
    otherwise reopen the loop with every other test still green (#919).
    """
    from charlie_work.github import _is_mutating, _should_retry

    for argv in (
        ["api", "-X", "POST", "repos/{owner}/{repo}/actions/runners/remove-token"],
        [
            "api",
            "-X",
            "POST",
            "repos/{owner}/{repo}/actions/runners/registration-token",
        ],
    ):
        assert _is_mutating(argv) is True, argv
        # Ambiguous: the request may have been served before the read timed out.
        assert _should_retry(argv, "i/o timeout", _is_mutating(argv)) is False, argv
        # Provably pre-send -- no token can have been minted, so retrying is safe.
        assert _should_retry(argv, "dial tcp: connection refused", _is_mutating(argv)) is True, (
            argv
        )

    # Control: a read is still granted the unconditional retry, so the assertions
    # above are about the mutating classification and not about the error strings.
    read = ["api", "repos/{owner}/{repo}/actions/runners"]
    assert _should_retry(read, "i/o timeout", _is_mutating(read)) is True


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


# --- Issue #618: dry-run local-write sites -----------------------------------


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


def test_github_are_issues_open_normalizes_uppercase_state(monkeypatch, tmp_path: Path) -> None:
    """Issue #173: Regression test for are_issues_open with realistic uppercase state.

    Exercises the production ``are_issues_open`` per-issue fallback path with
    realistic uppercase state field values (as returned by the real GitHub API),
    ensuring the ``.upper()`` normalization cannot silently regress.

    Post-L07 (issue #1591) ``are_issues_open`` lives on the ``Issues`` capability
    collaborator, and its thread-pool fallback closure resolves ``self.issue_view``
    on that collaborator -- so a GitHub *subclass* override of ``issue_view`` no
    longer intercepts that internal call (the disclosed interception-path
    relocation). The mock is therefore installed at the ``run`` layer, which
    ``issue_view`` forwards to through the collaborator->owner seam, and the
    batched GraphQL path is forced to fail so the per-issue fallback -- the code
    performing the ``.upper()`` normalization -- is the path under test.
    """
    from charlie_work.github import GitHub as RealGitHub
    from charlie_work.github import GitHubError

    # Realistic API states: uppercase from the real API, plus one lowercase (400)
    # that must still count as open via the ``.upper()`` normalization.
    states = {100: "OPEN", 200: "CLOSED", 300: "OPEN", 400: "open"}

    def _fail_graphql(self, numbers):
        # Force are_issues_open onto its per-issue ``issue_view`` fallback.
        raise GitHubError("forced batched-state failure -> per-issue fallback")

    def _fake_run(self, args, **kwargs):
        # issue_view builds ["issue", "view", str(number), "--json", ISSUE_VIEW_FIELDS];
        # args[2] is the issue number. Raise ValueError (not KeyError) for an
        # unexpected number so the failure mode matches the original mock: the
        # fallback's ``except (GitHubError, ValueError, TypeError)`` swallows it to
        # ``is_open=False`` rather than propagating out of ``pool.map``.
        number = int(args[2])
        if number not in states:
            raise ValueError(f"Unexpected issue number: {number}")
        return {"number": number, "state": states[number]}

    monkeypatch.setattr(RealGitHub, "_graphql_issue_states", _fail_graphql)
    monkeypatch.setattr(RealGitHub, "run", _fake_run)

    real_gh = RealGitHub(repo_root=tmp_path)

    # Call are_issues_open with a mix of open/closed issues
    result = real_gh.are_issues_open([100, 200, 300, 400])

    # Only the OPEN-state issues are returned: 100 and 300 (uppercase OPEN) and
    # 400 (lowercase "open", normalized via .upper()). 200 is CLOSED.
    assert result == {100, 300, 400}, f"Expected {{100, 300, 400}}, got {result}"


def test_are_issues_open_caches_per_pass_and_dedupes_shared_numbers(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #870: are_issues_open() was a fully serial, uncached, one
    `gh issue view` per number loop. Every distinct blocker issue number must
    now cost exactly one `gh issue view` call per status()/orchestrator pass,
    no matter how many separate callers ask about it or how much the
    requested number lists overlap -- mirroring the existing per-pass cache
    contract already proven for branch_protection()
    (test_branch_protection_caches_per_pass).
    """
    calls: list[int] = []

    def fake_run(command, **kwargs):
        # command: ["gh", "issue", "view", "<number>", "--json", ...]
        number = int(command[3])
        calls.append(number)
        payload = json.dumps({"number": number, "state": "OPEN" if number != 200 else "CLOSED"})
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = github_module.GitHub(repo_root=tmp_path)

    # Two overlapping requests, as _filter_blocked_issues and _summarize_issue
    # would each independently make for two issues sharing a blocker.
    first = gh.are_issues_open([100, 200])
    second = gh.are_issues_open([100, 200, 300])

    assert first == {100}
    assert second == {100, 300}
    # 100 and 200 must not be re-fetched by the second, overlapping call;
    # only the genuinely new number (300) costs a live call.
    assert sorted(calls) == [100, 200, 300]

    # invalidate_list_cache() (called once per orchestrator pass) must force
    # a fresh read on the next call -- never leaking across passes.
    gh.invalidate_list_cache()
    gh.are_issues_open([100])
    assert calls.count(100) == 2


# --- Issue #18: idempotence of ship-it and loop --------------------------------


def test_loop_skips_review_for_approved_unmerged_pr(tmp_path: Path) -> None:
    """A second loop() pass over an approved-but-unmerged PR must NOT rewrite
    the review packet or re-fire label transitions — it should go straight to
    merge_ready."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubListingPRs(FakeGitHub):
        """loop() now runs a reconcile pass (merge-lane-recovery §6-B) that
        calls gh.run(["pr", "list", ...]) via reconcile._fetch_prs. The base
        FakeGitHub.run() generic fallback always returns [] for that query
        regardless of self.prs, which makes detect_drift see an empty GitHub
        snapshot against a non-empty tracked-PR state and misreport PR 456 as
        missing on GitHub. Reflect self.prs for real here so the reconcile
        pass sees the same PR the rest of this fake already knows about."""

        def run(self, args, *, json_output=False, allow_failure=False):
            if args[:2] == ["pr", "list"]:
                return list(self.prs) if json_output else ""
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubListingPRs()
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


def test_orchestrator_own_comment_is_not_reingested_as_external_finding(
    tmp_path: Path,
) -> None:
    """Issue #950 follow-up: the orchestrator's own PR comments must not come
    back as "external findings".

    The orchestrator authenticates with a *user* token -- ``gh api user``
    reports ``type=User`` -- so ``_is_bot_comment`` cannot see its output, and
    filtering by login would drop the genuine human findings this feature
    exists to ingest. Provenance therefore travels in the body via
    ``ORCHESTRATOR_COMMENT_MARKER``.

    This is deliberately a round trip rather than two assertions against a
    hardcoded marker string: the body is produced by the real ``_comment_pr``
    write path and then filtered by the real collection path, so the writer and
    the reader cannot drift apart without this failing.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Produce a comment body through the real posting path.
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    app._comment_pr(456, "## Request changes\n\nThe retry wrapper swallows the type.")
    posted_body = (pr_dir / "review-comment.md").read_text(encoding="utf-8")
    assert "Request changes" in posted_body, "sanity: the summary survived stamping"

    # Feed it back as GitHub reports it: authored by a User, not a Bot.
    fake_gh.pr_external_issue_comments[456] = [
        {"body": posted_body, "user": {"login": "orchestrator-operator", "type": "User"}},
        {
            "body": "The migration needs a rollback path before this can land.",
            "user": {"login": "a-real-human", "type": "User"},
        },
    ]

    result = app.record_review(
        456,
        "request_changes",
        summary="internal summary",
        required_changes=["keep me"],
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    # Issue #999: external findings ride in their own field.
    assert decision["required_changes"] == ["keep me"]
    external = decision["external_findings"]

    # The orchestrator's own echo is filtered out...
    assert not any("retry wrapper swallows the type" in item for item in external)
    # ...while a genuine human finding from an identical account type is kept.
    assert any("rollback path" in item for item in external)


def test_human_quote_reply_to_orchestrator_comment_is_still_ingested(
    tmp_path: Path,
) -> None:
    """The provenance marker must be matched as a *prefix*, not a substring.

    GitHub's "Quote reply" copies the raw markdown of the quoted comment --
    HTML comments included -- into a blockquote above the reply. Quote-replying
    to one of our ``request_changes`` comments is a natural way for a human to
    answer point by point, and it produces a body that *contains*
    ``ORCHESTRATOR_COMMENT_MARKER`` without starting with it.

    Under a substring test that comment is classified as ours and dropped, so
    the human's new finding below the quote never reaches ``required_changes``.
    Losing a genuine finding is the expensive direction of this filter, so the
    predicate must fail toward ingestion. This test fails if the check is
    weakened back to ``MARKER in body``.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    app._comment_pr(456, "## Request changes\n\nThe retry wrapper swallows the type.")
    posted_body = (pr_dir / "review-comment.md").read_text(encoding="utf-8")

    # Exactly what GitHub stores when a human uses "Quote reply": every line of
    # the quoted comment prefixed with "> ", then the human's own text.
    quoted = "\n".join(f"> {line}" for line in posted_body.splitlines())
    human_reply = f"{quoted}\n\nAgreed, and separately: the migration needs a rollback path."
    assert ORCHESTRATOR_COMMENT_MARKER in human_reply, (
        "sanity: the quote really does carry the marker -- otherwise this test "
        "would pass without exercising the prefix-vs-substring distinction"
    )
    assert not human_reply.startswith(ORCHESTRATOR_COMMENT_MARKER)

    fake_gh.pr_external_issue_comments[456] = [
        {"body": posted_body, "user": {"login": "orchestrator-operator", "type": "User"}},
        {"body": human_reply, "user": {"login": "a-real-human", "type": "User"}},
    ]

    result = app.record_review(
        456,
        "request_changes",
        summary="internal summary",
        required_changes=["keep me"],
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    # Issue #999: external findings ride in their own field.
    assert decision["required_changes"] == ["keep me"]
    external = decision["external_findings"]

    # The human's finding survives even though their comment embeds our marker.
    assert any("rollback path" in item for item in external)
    # Our own unquoted comment is still filtered out.
    assert not any(item.lstrip().startswith(ORCHESTRATOR_COMMENT_MARKER) for item in external)


def test_worker_rework_reply_is_not_ingested_as_external_finding(
    tmp_path: Path,
) -> None:
    """Issue #998: a worker's own rework reply is machine-generated, posted
    through the worker's path (no ``ORCHESTRATOR_COMMENT_MARKER``), and posted
    *after* the rework commit it describes. It must not come back as a
    "required change" on the next ``request_changes`` verdict -- that would
    tell the worker to address its own completion report.

    The cutoff is temporal, not identity-based: the worker posts through a
    user token (same account / ``type=User`` as the human whose findings #950
    exists to capture), so the upper bound is the ``reviewed_head_sha``'s
    committer date. A genuine human comment from the *same* account and same
    ``type=User``, posted *before* the reviewed head, is still ingested --
    this is the regression any identity-based shortcut would cause, asserted
    positively rather than assumed.

    Mutation check: disabling the ``before`` upper bound (reverting
    ``_collect_external_findings`` to its merge-base form) makes this test
    fail, because the worker reply is then ingested alongside the human
    finding.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # The rework commit the reviewer is about to read. Its committer date is
    # the ingestion upper bound. Set it up as the live PR head with full
    # commit metadata so _commit_timestamp can resolve it.
    rework_sha = "rework-head-sha"
    fake_gh.pr_head_shas[456] = rework_sha
    fake_gh.commits[rework_sha] = {
        "parents": [{"sha": "base-sha"}],
        "commit": {
            "author": {
                "name": "worker",
                "email": "w@example.test",
                "date": "2026-08-10T10:00:00Z",
            },
            "committer": {
                "name": "worker",
                "email": "w@example.test",
                "date": "2026-08-10T10:00:00Z",
            },
        },
    }

    # A genuine human finding from the SAME account and SAME type=User,
    # posted *before* the reviewed head commit -- must still be ingested.
    human_finding = "The migration script drops the index without a guard."
    # The worker pushed the rework at 10:00, then posted its completion reply
    # at 10:05 -- after the head commit, so outside the ingestion window. This
    # is exactly the real-world shape from PR #972's comment thread.
    worker_reply = (
        "Reworked in rework-head-sha. Summary of the changes addressing each "
        "point: added the missing rollback path and a regression test."
    )
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": human_finding,
            "user": {"login": "operator", "type": "User"},
            "created_at": "2026-08-09T12:00:00Z",
        },
        {
            "body": worker_reply,
            "user": {"login": "operator", "type": "User"},
            "created_at": "2026-08-10T10:05:00Z",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.record_review(
        456,
        "request_changes",
        summary="internal summary",
        required_changes=["keep me"],
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    # Issue #999: external findings ride in their own field, not required_changes.
    changes = decision["required_changes"]
    external = decision.get("external_findings", [])

    # The worker's own rework reply is NOT fed back as an external finding.
    assert not any("Reworked in rework-head-sha" in item for item in external), (
        "worker rework reply must not be ingested as an external finding"
    )
    # The genuine human finding from the same account/type=User IS ingested
    # into the external_findings field.
    assert any("migration script drops the index" in item for item in external), (
        "genuine human comment before the reviewed head must still be ingested"
    )
    # The internal finding survives untouched in required_changes.
    assert "keep me" in changes


def test_human_comment_in_before_to_reviewed_at_gap_surfaces_next_round(
    tmp_path: Path,
) -> None:
    """Issue #998 rework: a genuine human comment posted in the gap
    ``(before, reviewed_at]`` -- after the reviewed head commit landed but
    before the verdict was written -- is excluded by ``before`` this round
    and MUST surface as a required_change in the following round.

    The per-round ingestion windows must be contiguous: the next round's
    ``since`` is this round's persisted ``before`` (not its ``reviewed_at``).
    Deriving ``since`` from ``reviewed_at`` instead would drop a gap comment
    forever -- it satisfies ``item_dt <= reviewed_at``, so the lower bound
    skips it in every subsequent round -- silently violating the
    fail-toward-ingestion invariant.

    Mutation check: reverting the ``since`` derivation in ``record_review``
    to ``previous_decision.get("reviewed_at")`` (the merge-base form, without
    the ``before`` fallback) makes this test fail, because the gap comment is
    then dropped by ``since`` in round 2 and never surfaces.
    """
    # max_rework_cycles bumped past 2 so the second request_changes round does
    # not escalate -- escalation does not short-circuit required_changes
    # persistence (the ingestion block runs before the escalation check), but
    # keeping the verdict non-escalated makes the assertion target unambiguous.
    config = OrchestratorConfig(review=ReviewConfig(max_rework_cycles=10))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    base = datetime.now(UTC)
    # Round-1 head commit landed 2 hours ago -- well before the verdict write.
    round1_commit_dt = base - timedelta(hours=2)
    # A genuine human finding posted 1 hour ago: strictly AFTER the round-1
    # head commit (so ``before`` excludes it in round 1) and strictly BEFORE
    # round-1's reviewed_at (utc_now() during round 1's record_review, i.e.
    # ~base). This is the (before, reviewed_at] gap that the discontinuity
    # silently dropped.
    gap_comment_dt = base - timedelta(hours=1)
    gap_finding = "Gap comment: the rollback path leaks a file handle on early return."

    round1_sha = "round1-head-sha"
    fake_gh.pr_head_shas[456] = round1_sha
    fake_gh.commits[round1_sha] = {
        "parents": [{"sha": "base-sha"}],
        "commit": {
            "author": {
                "name": "worker",
                "email": "w@example.test",
                "date": round1_commit_dt.isoformat(),
            },
            "committer": {
                "name": "worker",
                "email": "w@example.test",
                "date": round1_commit_dt.isoformat(),
            },
        },
    }
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": gap_finding,
            "user": {"login": "operator", "type": "User"},
            "created_at": gap_comment_dt.isoformat(),
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Round 1: request_changes. The gap comment is after the round-1 head
    # commit, so ``before`` excludes it this round -- it must NOT appear yet.
    r1 = app.record_review(
        456,
        "request_changes",
        summary="round 1",
        required_changes=["internal-1"],
        verdict_provenance="fresh_llm_review",
    )
    assert r1.ok is True
    d1 = json.loads((paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8"))
    assert not any("Gap comment" in c for c in d1["required_changes"]), (
        "gap comment must be excluded by `before` in round 1 "
        "(it is strictly after the round-1 head commit)"
    )
    # The contiguity fix persists ``before`` so round 2 can derive ``since``
    # from it. This is the load-bearing persistence the next round reads back.
    assert d1.get("before") == round1_commit_dt.isoformat(), (
        "round-1 decision must persist the `before` upper bound for round-2 contiguity"
    )

    # Round 2: the worker pushed a new head. Its commit lands ~now (after the
    # gap comment), so ``before_2`` does not exclude the gap comment; and
    # ``since_2`` = round-1's persisted ``before`` = round1_commit_dt, which is
    # before the gap comment, so the lower bound does not exclude it either.
    round2_commit_dt = base
    round2_sha = "round2-head-sha"
    fake_gh.pr_head_shas[456] = round2_sha
    fake_gh.commits[round2_sha] = {
        "parents": [{"sha": round1_sha}],
        "commit": {
            "author": {
                "name": "worker",
                "email": "w@example.test",
                "date": round2_commit_dt.isoformat(),
            },
            "committer": {
                "name": "worker",
                "email": "w@example.test",
                "date": round2_commit_dt.isoformat(),
            },
        },
    }

    r2 = app.record_review(
        456,
        "request_changes",
        summary="round 2",
        required_changes=["internal-2"],
        verdict_provenance="fresh_llm_review",
    )
    assert r2.ok is True
    d2 = json.loads((paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8"))
    # Issue #999: external findings ride in their own field, not required_changes.
    changes2 = d2["required_changes"]
    external2 = d2.get("external_findings", [])

    # THE regression assertion: the gap comment surfaces in round 2 rather
    # than being permanently dropped.
    assert any("Gap comment" in c for c in external2), (
        "human comment in the (before, reviewed_at] gap must surface in the next "
        "round, not be silently dropped forever"
    )
    # The round-2 internal finding survives alongside it.
    assert "internal-2" in changes2


def test_human_quote_reply_to_a_crash_summary_is_still_ingested(tmp_path: Path) -> None:
    """A genuine human reply that GitHub-quotes a crash summary (to discuss
    or dispute it) is preserved -- mirrors
    test_human_quote_reply_to_orchestrator_comment_is_still_ingested's
    rationale for ORCHESTRATOR_COMMENT_MARKER, applied to the crash-heading
    prefix check instead. A substring match would wrongly discard this
    reply along with the quoted heading; the prefix check must not."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    quoted_reply = (
        f"> {REVIEW_SESSION_SUMMARY_HEADING}\n"
        "> \n"
        "> The automated reviewer ran for 4 turns...\n"
        "\n"
        "This looks like a session crash, not a real review -- can we re-run it?"
    )
    fake_gh.pr_external_issue_comments[456] = [
        {
            "body": quoted_reply,
            "user": {"login": "a-real-human", "type": "User"},
            "created_at": "2026-08-09T12:00:00Z",
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456,
        "request_changes",
        summary="internal summary",
        required_changes=["keep me"],
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    external = decision.get("external_findings", [])
    assert any("can we re-run it" in item for item in external), (
        "a genuine human reply quoting a crash summary must still be ingested"
    )


def test_refresh_pr_decision_cache_updates_disagreeing_tracked_pr(tmp_path: Path) -> None:
    """Issue #1362 Stage 3: a tracked PR whose cache disagrees with the
    file-first decision gets its three cache fields overwritten."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    seed = load_state(paths.state_file)
    seed["prs"]["456"] = {
        "status": "reviewing",
        "issue_number": 123,
        "decision": "pending",
        "reviewed_head_sha": "stale-sha",
        "decision_path": "stale-path",
    }
    save_state(paths.state_file, seed)

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = ReviewDecision(
        decision="approved",
        reviewed_head_sha="fresh-sha",
        recorded_at="2026-08-21T00:00:00Z",
        source_round=None,
        stale=False,
        missing=False,
    )
    app._refresh_pr_decision_cache(456, decision, decision_path)

    refreshed = load_state(paths.state_file)["prs"]["456"]
    assert refreshed["decision"] == "approved"
    assert refreshed["reviewed_head_sha"] == "fresh-sha"
    assert refreshed["decision_path"] == str(decision_path)
    # Non-decision fields (status, issue_number) must survive the mirror
    # write untouched -- the refresh must never clobber the rest of the entry.
    assert refreshed["status"] == "reviewing"
    assert refreshed["issue_number"] == 123


def test_refresh_pr_decision_cache_no_op_when_cache_already_agrees(tmp_path: Path) -> None:
    """Issue #1362 Stage 3: when the cache already agrees with the file, the
    refresh must not write state.json at all -- the docstring's promised
    short-circuit for the common (no verdict activity) case."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    seed = load_state(paths.state_file)
    seed["prs"]["456"] = {
        "status": "reviewing",
        "decision": "approved",
        "reviewed_head_sha": "fresh-sha",
        "decision_path": str(decision_path),
    }
    save_state(paths.state_file, seed)
    mtime_before = paths.state_file.stat().st_mtime_ns

    decision = ReviewDecision(
        decision="approved",
        reviewed_head_sha="fresh-sha",
        recorded_at="2026-08-21T00:00:00Z",
        source_round=None,
        stale=False,
        missing=False,
    )

    # Positive signal, not just an absence: wrap the gated writer so a call
    # that happens but happens to leave the mtime unchanged (e.g. a
    # sub-resolution clock or a write of byte-identical content) cannot
    # read as "no write occurred". An mtime check alone would pass even if
    # the short-circuit above it were deleted, as long as the resulting
    # write raced under the OS's mtime granularity.
    #
    # ``write_gate`` is a frozen dataclass (its instances reject attribute
    # assignment), so the wrap patches the *class* method rather than the
    # instance attribute.
    save_state_calls: list[dict] = []
    write_gate_cls = type(app.write_gate)
    original_save_state = write_gate_cls.save_state

    def _tracking_save_state(self: object, state: dict) -> None:
        save_state_calls.append(state)
        original_save_state(self, state)

    write_gate_cls.save_state = _tracking_save_state  # type: ignore[method-assign]
    try:
        app._refresh_pr_decision_cache(456, decision, decision_path)
    finally:
        write_gate_cls.save_state = original_save_state  # type: ignore[method-assign]

    assert save_state_calls == []
    assert paths.state_file.stat().st_mtime_ns == mtime_before


def test_refresh_pr_decision_cache_skips_pr_not_yet_in_state(tmp_path: Path) -> None:
    """Issue #1362 Stage 3 (review finding F3): a PR not yet tracked in
    state["prs"] must be left untouched by the refresh rather than
    materializing a decision-only partial entry with no status/counters."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    decision_path = paths.prs / "pr-999" / "review-decision.json"
    decision = ReviewDecision(
        decision="approved",
        reviewed_head_sha="fresh-sha",
        recorded_at="2026-08-21T00:00:00Z",
        source_round=None,
        stale=False,
        missing=False,
    )
    app._refresh_pr_decision_cache(999, decision, decision_path)

    state = load_state(paths.state_file)
    assert "999" not in state["prs"]


def test_loop_pass_refreshes_pr_decision_cache_to_file_value(tmp_path: Path) -> None:
    """Issue #1386: pin the _refresh_pr_decision_cache CALL SITE in loop()'s
    per-PR dispatch block.

    The three behavioral tests above cover the method itself (updates a
    disagreeing tracked PR, no-ops when the cache agrees, skips untracked
    PRs), but none of them exercise the call site -- deleting the
    ``self._refresh_pr_decision_cache(...)`` invocation from ``loop()``'s
    dispatch block would leave the whole suite green. This test closes that
    gap: it seeds a tracked PR whose state-side decision disagrees with its
    flat ``review-decision.json`` (the #1340 divergence shape: state lags a
    concurrent void/record_review), runs one ``loop()`` pass, and asserts
    ``state["prs"][N]`` was reconciled to the file value.

    The test runs in live (non-dry-run) mode. Dry-run would seem isolating
    but is not: ``_refresh_pr_decision_cache`` writes through
    ``self.write_gate.save_state``, which is a no-op under dry-run (the
    WriteGate's strict "zero writes under dry-run" invariant), so the
    refresh's disk write is invisible there. In live mode the merge success
    path (``merge_ready``) DOES write state, but it carries forward the
    existing decision cache fields via ``**state["prs"].get(...)``
    (workflow.py merge_success block) -- it does NOT re-derive
    ``decision``/``reviewed_head_sha``/``decision_path`` from the file. So
    the final state's cache fields are exactly what the refresh wrote: the
    file value if the refresh ran, the stale seed value if it did not.
    Deleting the call site leaves the seeded divergence unreconciled and
    this test fails.
    """
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubListingPRs(FakeGitHub):
        """loop()'s reconcile pass (merge-lane-recovery §6-B) queries
        gh.run(["pr", "list", ...]); the base fake's generic run() fallback
        returns [] regardless of self.prs, which misreports PR 456 as
        missing on GitHub. Reflect self.prs so the reconcile pass sees the
        same PR the rest of this fake knows about. Same override as
        test_loop_skips_review_for_approved_unmerged_pr."""

        def run(self, args, *, json_output=False, allow_failure=False):
            if args[:2] == ["pr", "list"]:
                return list(self.prs) if json_output else ""
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubListingPRs()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Seed a tracked PR whose state-side decision DISAGREES with the file
    # (the #1340 divergence shape: state says "pending" at a stale head while
    # the file has already been reset to "approved" at the live head).
    state = load_state(paths.state_file)
    state["prs"]["456"] = {
        "number": 456,
        "issue_number": 123,
        "status": "reviewing",
        "decision": "pending",
        "reviewed_head_sha": "stale-sha",
        "decision_path": "stale-path",
    }
    save_state(paths.state_file, state)

    # The flat file is authoritative: it says "approved" at the live head
    # (sha-abc123, matching FakeGitHub's default PR head). The file's
    # reviewed_head_sha matches the live head so the already_approved /
    # head_matches gates fire and merge_ready's merge-success path runs --
    # which carries forward the existing cache fields rather than
    # re-deriving them, isolating the refresh as the sole reconciler.
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    decision_path = decision_dir / "review-decision.json"
    decision_path.write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.loop(limit=0)

    # The merge must have run (confirms the per-PR dispatch block reached
    # merge_ready, which is downstream of the refresh call site).
    assert len(result.data["merges"]) == 1
    assert result.data["merges"][0]["merged"] is True

    refreshed = load_state(paths.state_file)["prs"]["456"]
    # The three cache fields must be reconciled to the file value by the
    # refresh -- merge_ready's carry-forward preserves whatever the refresh
    # wrote, so these fail if the refresh call site is deleted.
    assert refreshed["decision"] == "approved"
    assert refreshed["reviewed_head_sha"] == "sha-abc123"
    assert refreshed["decision_path"] == str(decision_path)
    # Non-decision fields survive: issue_number is carried forward by both
    # the refresh and merge_ready's spread. (status becomes "merged" after
    # the merge, which is expected and not a cache field.)
    assert refreshed["issue_number"] == 123


def test_standard_lifecycle_rework_dispatch_selects_issue(tmp_path: Path) -> None:
    """Issue #72 acceptance criterion 1: standard-lifecycle end-to-end rework dispatch.

    Fresh dispatch marks the issue dispatched → record_review(request_changes) →
    dispatch_rework SELECTS the issue and launches via a command-adapter fake,
    firing the rework_dispatched label transition.
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
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Step 1: Fresh dispatch marks the issue as dispatched
    app.gh.prs[0]["state"] = "CLOSED"
    dispatch_result = app.dispatch(limit=1)
    assert dispatch_result.ok is True
    assert dispatch_result.data["selected_count"] == 1

    # Verify issue is marked as dispatched in state
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"

    # Step 2: record_review(request_changes) updates issue status to rework_requested
    # Issue #1131: record_review now refuses on a terminal-state (CLOSED) PR,
    # so restore the PR to OPEN before recording the verdict -- the CLOSED
    # state above was a fixture trick to make dispatch select the issue, not
    # a real terminal state.
    app.gh.prs[0]["state"] = "OPEN"
    review_result = app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )
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
    app.gh.prs[0]["state"] = "OPEN"
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
        review=ReviewConfig(max_rework_cycles=2),
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
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Step 1: Fresh dispatch marks the issue as dispatched
    app.gh.prs[0]["state"] = "CLOSED"
    dispatch_result = app.dispatch(limit=1)
    assert dispatch_result.ok is True
    assert dispatch_result.data["selected_count"] == 1

    # Step 2: Record first request_changes (count = 1, not escalated, head = "sha-1")
    # Issue #1131: restore PR to OPEN before record_review -- the CLOSED state
    # was a fixture trick for dispatch, not a real terminal state.
    app.gh.prs[0]["state"] = "OPEN"
    fake_gh.pr_head_shas[456] = "sha-1"
    review_result_1 = app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )
    assert review_result_1.ok is True
    assert review_result_1.data["escalated"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert state["issues"]["123"]["status"] == "rework_requested"

    # Step 3: Record second request_changes (count = 2, not escalated yet, head = "sha-2")
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")
    fake_gh.pr_head_shas[456] = "sha-2"

    review_result_2 = app.record_review(
        456, "request_changes", summary="fix B", verdict_provenance="fresh_llm_review"
    )
    assert review_result_2.ok is True
    assert review_result_2.data["escalated"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["issues"]["123"]["status"] == "rework_requested"

    # Step 4: Record third request_changes (count stays at 2, escalated because max_rework_cycles = 2, head = "sha-3")
    # When escalated, the count is NOT incremented (see workflow.py line 731-734)
    fake_gh.pr_head_shas[456] = "sha-3"
    review_result_3 = app.record_review(
        456, "request_changes", summary="fix C", verdict_provenance="fresh_llm_review"
    )
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

    # Step 5: Verify the escalated label transition was fired (adds
    # operator_queue, removes reviewing). Issue #1266: max_rework_cycles_exceeded
    # is a mechanical escalation, so it now routes to operator_queue instead of
    # human_needed.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    # The escalated transition removes reviewing but does NOT remove needs_rework
    # (this is by design per labels.py's redispatch_escalated/escalated edges)

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


def test_request_changes_count_does_not_increment_on_unchanged_head(tmp_path: Path) -> None:
    """Issue #208: request_changes_count should only increment when PR head advances.

    When a worker dies orphaned and the PR head never advances, re-issuing
    request_changes should not consume the escalation budget.
    """
    config = OrchestratorConfig(
        review=ReviewConfig(max_rework_cycles=2),
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
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Step 1: Fresh dispatch
    app.gh.prs[0]["state"] = "CLOSED"
    dispatch_result = app.dispatch(limit=1)
    assert dispatch_result.ok is True

    # Step 2: Record first request_changes (count = 1, head = "sha-1")
    # Issue #1131: restore PR to OPEN before record_review -- the CLOSED state
    # was a fixture trick for dispatch, not a real terminal state.
    app.gh.prs[0]["state"] = "OPEN"
    fake_gh.pr_head_shas[456] = "sha-1"
    review_result_1 = app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )
    assert review_result_1.ok is True
    assert review_result_1.data["escalated"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-1"

    # Step 3: Record second request_changes with SAME head (count should stay at 1)
    # This simulates a worker dying orphaned - no rework was actually produced
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    review_result_2 = app.record_review(
        456, "request_changes", summary="fix B", verdict_provenance="fresh_llm_review"
    )
    assert review_result_2.ok is True
    assert review_result_2.data["escalated"] is False

    state = load_state(paths.state_file)
    # Count should NOT increment because head didn't advance
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-1"

    # Step 4: Record third request_changes with NEW head (count should increment to 2)
    fake_gh.pr_head_shas[456] = "sha-2"
    review_result_3 = app.record_review(
        456, "request_changes", summary="fix C", verdict_provenance="fresh_llm_review"
    )
    assert review_result_3.ok is True
    assert review_result_3.data["escalated"] is False

    state = load_state(paths.state_file)
    # Count should increment because head advanced
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-2"


def test_at_cap_request_changes_on_unchanged_head_does_not_escalate(
    tmp_path: Path,
) -> None:
    """Issue #1210: an at-cap request_changes verdict on an unchanged head must not escalate.

    The head_advanced guard (issue #208) previously protected only the counter
    increment; the escalation check ran unconditionally and fired on an at-cap
    verdict even when the head was unchanged (e.g. a worker died orphaned and
    pushed nothing). Such a round should re-issue request_changes without
    escalating, mirroring what already happens below-cap. The counter must be
    unchanged and the PR/issue status must NOT be escalated.
    """
    config = OrchestratorConfig(
        review=ReviewConfig(max_rework_cycles=2),
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
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Step 1: Fresh dispatch
    app.gh.prs[0]["state"] = "CLOSED"
    dispatch_result = app.dispatch(limit=1)
    assert dispatch_result.ok is True

    # Step 2: Drive request_changes_count up to the cap (2) over two advancing heads.
    # Issue #1131: restore PR to OPEN before record_review -- the CLOSED state
    # was a fixture trick for dispatch, not a real terminal state.
    app.gh.prs[0]["state"] = "OPEN"
    fake_gh.pr_head_shas[456] = "sha-1"
    review_result_1 = app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )
    assert review_result_1.ok is True
    assert review_result_1.data["escalated"] is False

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")

    fake_gh.pr_head_shas[456] = "sha-2"
    review_result_2 = app.record_review(
        456, "request_changes", summary="fix B", verdict_provenance="fresh_llm_review"
    )
    assert review_result_2.ok is True
    assert review_result_2.data["escalated"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-2"

    # Step 3: A request_changes verdict on the SAME head (sha-2) — the worker
    # died without pushing anything. request_changes_count is already at the
    # cap. This must NOT escalate and must NOT mutate the counter.
    review_result_3 = app.record_review(
        456, "request_changes", summary="fix C", verdict_provenance="fresh_llm_review"
    )
    assert review_result_3.ok is True
    assert review_result_3.data["escalated"] is False

    state = load_state(paths.state_file)
    # Counter unchanged — the round consumed no escalation budget.
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-2"
    # PR and issue status must reflect re-issued request_changes, not escalation.
    assert state["prs"]["456"]["status"] == "request_changes"
    assert state["issues"]["123"]["status"] == "rework_requested"
    # No human-needed label should have been added by this round.
    assert (123, "agent:human-needed") not in fake_gh.labels_added

    # Step 4: Sanity — once the head DOES advance, the at-cap verdict escalates
    # as before (existing behavior preserved for advanced heads).
    fake_gh.pr_head_shas[456] = "sha-3"
    review_result_4 = app.record_review(
        456, "request_changes", summary="fix D", verdict_provenance="fresh_llm_review"
    )
    assert review_result_4.ok is True
    assert review_result_4.data["escalated"] is True

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["issues"]["123"]["status"] == "escalated"


def test_loop_re_reviews_when_head_moved_after_approval(tmp_path: Path) -> None:
    config = dataclasses.replace(
        _required_checks_config(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
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
    fake_gh.pr_head_shas[456] = "sha-new-head"

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

    class FakeGitHubListingPRs(FakeGitHub):
        """See the identical override in
        test_loop_skips_review_for_approved_unmerged_pr: loop()'s reconcile
        pass (merge-lane-recovery §6-B) queries gh.run(["pr", "list", ...]),
        and the base fake's generic run() fallback returns [] regardless of
        self.prs, which misreports PR 456 as missing on GitHub."""

        def run(self, args, *, json_output=False, allow_failure=False):
            if args[:2] == ["pr", "list"]:
                return list(self.prs) if json_output else ""
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubListingPRs()
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


def test_loop_no_merge_evaluates_readiness_but_skips_gh_merge(tmp_path: Path) -> None:
    """bash-rats --no-merge: the pass reviews and evaluates merge readiness but
    never calls `gh pr merge` — operators sequencing same-surface cascades by
    hand rely on this to dispatch reworks without out-of-order merges."""
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

    result = app.loop(limit=0, merge=False)

    assert len(result.data["merges"]) == 1
    assert result.data["merges"][0]["merged"] is False
    assert fake_gh.merged == []


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
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
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
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Run a loop pass with limit=0 (no actual dispatch, just the classification logic).
    # `now` is frozen and injected (issue #822's clock seam -- see
    # workflow._classify_dead_sessions_and_update_throttle_state) so the throttle
    # timestamp assertion below is exact instead of racing wall-clock time under
    # CI runner contention. We don't assert result.ok because dispatch may fail
    # with no issues to process -- the key is that the classification logic runs
    # regardless.
    # frozen_now is offset 1 hour into the future (rather than the real instant)
    # so the throttle window below stays open for the dispatch-deferral check
    # further down regardless of how long the process stalls between here and
    # there -- the offset is arbitrary and only needs to exceed any plausible
    # CI stall; it does not affect the exact-equality assertion since both sides
    # derive from this same captured value.
    frozen_now = datetime.now(UTC) + timedelta(hours=1)
    app.loop(limit=0, now=frozen_now)

    # Verify throttled_until was set in state by the loop's classification pass
    state = load_state(paths.state_file)
    assert state.get("throttled_until") is not None

    # Verify the cooldown reflects the parsed 10 minutes plus the resume margin,
    # computed from the same frozen `now` the loop pass was given -- exact
    # equality, no wall-clock tolerance window.
    throttle_time = datetime.fromisoformat(state["throttled_until"].replace("Z", "+00:00"))
    expected_time = (
        frozen_now + timedelta(minutes=10, seconds=config.runtime.throttle_resume_margin_s)
    ).replace(microsecond=0)
    assert throttle_time == expected_time

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

    app.gh.prs[0]["state"] = "CLOSED"
    dispatch_result = app.dispatch(limit=1)
    # Dispatch should be deferred due to throttle (ok=False is expected for deferral)
    assert dispatch_result.ok is False
    assert "deferred" in dispatch_result.message.lower()
    # Should defer launch due to throttle
    assert dispatch_result.data["selected_count"] == 0


def test_loop_wires_persist_inconclusive_probe_counter_false_to_dead_lane(
    tmp_path: Path,
) -> None:
    """Issue #343 Finding 2 wiring test: loop() must call the dead lane with
    persist_inconclusive_probe_counter=False.

    The stall lane runs unconditionally at the top of loop() (line ~4100) and is
    the sole writer of the not-alive-worker inconclusive-probe deferral counter
    for that pass. If loop()'s call to _classify_dead_sessions_and_update_throttle_state
    (line ~4119) ever drops the persist_inconclusive_probe_counter=False keyword
    (e.g. reverted to the default True during a rebase), the dead lane would
    silently double-write that counter on top of the stall lane's write within a
    single loop() pass.

    This MUST fail if that call site's persist_inconclusive_probe_counter=False
    keyword is removed or flipped to True -- verified by temporarily reverting the
    call site during development of this test (not asserted here, since a
    mutation test would require editing production code from within a test).

    Deliberately does not assert on the inconclusive-probe counter's persisted
    value: the stall lane itself runs 3x per loop() pass (direct, plus via
    dispatch_rework() and dispatch()'s own internal calls -- a separate,
    pre-existing redundancy tracked outside this issue), which would make an
    end-to-end counter-value assertion through loop() fragile. Pinning the
    call-args of the dead lane directly is the precise, stable way to gate this
    specific wiring.
    """
    from charlie_work.workflow import (
        _classify_dead_sessions_and_update_throttle_state as real_classify_dead_sessions,
    )

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    with patch(
        "charlie_work.workflow._classify_dead_sessions_and_update_throttle_state",
        wraps=real_classify_dead_sessions,
    ) as mock_classify:
        app.loop(limit=0)

    mock_classify.assert_called_once()
    assert mock_classify.call_args.kwargs["persist_inconclusive_probe_counter"] is False


def test_loop_reaps_launch_failure_sidecar_and_reports_reaped(
    tmp_path: Path,
) -> None:
    """Issue #266: loop() reaps launch-failure sidecars (pid=None, error set)
    and reports them in the ``reaped`` section of the pass result.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(sessions_dir / "issue-42.log"),
        error="devin binary not found",
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    result = app.loop(limit=0)

    assert not sidecar_path.exists()
    reaped = result.data.get("reaped", [])
    assert len(reaped) == 1
    assert reaped[0]["issue_number"] == 42
    assert reaped[0]["failure_kind"] == "launch_failed"
    assert reaped[0]["error"] == "devin binary not found"


def test_loop_launch_failure_with_throttle_signature_persists_throttled_until(
    tmp_path: Path,
) -> None:
    """Issue #266 + cross-family finding: a throttle-caused LAUNCH failure must
    persist throttled_until exactly like the dead-session lane.

    A launch-failure sidecar (pid=None, error set) whose log carries the
    rate-limit signature is classified through the same failure classifier;
    discarding its throttled_until would relaunch straight into the throttled
    provider. This test MUST fail if the launch-failure branch drops the
    classifier's throttle window.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime, timedelta

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="Reached overall message rate limit. Your limit will reset in 10 minutes.",
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Issue #828: freeze the clock and thread it through loop() so the
    # assertion below can compare against an exact expected instant instead
    # of a wall-clock-tolerance window (a stall between loop() and the
    # assertion previously had ~5s to blow the tolerance and flake CI).
    frozen_now = datetime.now(UTC)
    result = app.loop(limit=0, now=frozen_now)

    # The launch-failure sidecar is reaped and reported
    reaped = result.data.get("reaped", [])
    matching = [entry for entry in reaped if entry["issue_number"] == 42]
    assert len(matching) == 1
    assert not sidecar_path.exists()

    # The throttle window from the classifier is persisted, same as the
    # dead-session lane, including the resume margin.
    state = load_state(paths.state_file)
    assert state.get("throttled_until") is not None
    throttle_time = datetime.fromisoformat(state["throttled_until"].replace("Z", "+00:00"))
    expected_time = (
        frozen_now + timedelta(minutes=10, seconds=config.runtime.throttle_resume_margin_s)
    ).replace(microsecond=0)
    assert throttle_time == expected_time


def test_loop_pid_none_no_error_not_classified_as_launch_failed(
    tmp_path: Path,
) -> None:
    """Issue #266: a pid=None + error=None sidecar is a dead session, not a launch failure.

    The launch-failure branch must not fire here; the dead-session branch handles
    it and does not tag it with failure_kind="launch_failed".
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(sessions_dir / "issue-42.log"),
        error=None,
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    result = app.loop(limit=0)

    reaped = result.data.get("reaped", [])
    # The record must actually be reaped (dead-session lane owns it) — a bare
    # loop over a possibly-empty list would pass vacuously if the sidecar were
    # skipped entirely, which is the old pin-the-loop-open behavior.
    matching = [entry for entry in reaped if entry["issue_number"] == 42]
    assert len(matching) == 1
    assert matching[0]["failure_kind"] != "launch_failed"


def test_worktree_unsafe_launch_failure_escalates_and_suppresses_redispatch(
    tmp_path: Path,
) -> None:
    """Issue #288: a launch result whose sidecar carries failure_kind=worktree_unsafe
    must escalate immediately, bypass the redispatch cap, and not be relabeled to ready.
    A subsequent dispatch pass must not select the issue.
    """
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state

    now = datetime.now(UTC)

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.ready}],
        }
    ]
    fake_gh.prs = []  # No open PR — the ordinary relabel path would fire here.

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("worktree contains local work, cannot reset\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Launch failure — process never started
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="worktree creation failed: worktree contains local work",
        failure_kind="worktree_unsafe_shim_dirt",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # No hot relabel-to-ready.
    assert (42, config.labels.ready) not in fake_gh.labels_added
    # Escalation transition added operator_queue. Issue #1266: worktree_unsafe
    # is mechanical, so it lands agent:operator-queue, not agent:human-needed.
    assert (42, config.labels.operator_queue) in fake_gh.labels_added
    # The launch never succeeded, so the issue should not be marked in_progress.
    assert (42, config.labels.in_progress) not in fake_gh.labels_added

    state = load_state(paths.state_file)
    issue_entry = state["issues"]["42"]
    assert issue_entry["status"] == "escalated"
    assert issue_entry["escalation_reason"] == "worktree_unsafe_shim_dirt"

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 42]
    assert "session_failed_relabeled" not in event_kinds
    assert "session_failed_escalated" in event_kinds

    fake_gh.issues[0]["labels"].append({"name": config.labels.operator_queue})
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)
    assert result.data["selected_count"] == 0


def test_worktree_probe_failed_launch_failure_does_not_escalate(
    tmp_path: Path,
) -> None:
    """PR #314 review follow-up to issue #288: a launch result whose sidecar
    carries failure_kind=worktree_probe_failed (the git status --porcelain
    safety probe itself failed -- index lock, I/O error, etc. -- NOT a
    confirmed-dirty worktree) must NOT escalate on first occurrence. It must
    take the ordinary redispatch-cap path so a subsequent dispatch pass can
    still select the issue.

    This is the mirror of
    test_worktree_unsafe_launch_failure_escalates_and_suppresses_redispatch:
    confirmed-dirty (worktree_unsafe) escalates immediately; a failed probe
    (worktree_probe_failed) must not, because it is transient contention an
    ordinary redispatch retry would plausibly heal.
    """
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state

    now = datetime.now(UTC)

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.ready}],
        }
    ]
    fake_gh.prs = []  # No open PR — the ordinary relabel path would fire here.

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("index.lock: File exists\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Launch failure — process never started
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="worktree creation failed: worktree status probe failed; treating as dirty",
        failure_kind="worktree_probe_failed",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # No escalation transition — human_needed must NOT be added, and the
    # issue must not be marked in_progress (the launch never succeeded).
    assert (42, config.labels.human_needed) not in fake_gh.labels_added
    assert (42, config.labels.in_progress) not in fake_gh.labels_added
    assert (42, config.labels.ready) not in fake_gh.labels_removed

    # No escalated status recorded in state for this issue.
    state = load_state(paths.state_file)
    issue_entry = state["issues"].get("42")
    assert issue_entry is None or issue_entry.get("status") != "escalated"

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 42]
    assert "session_failed_escalated" not in event_kinds

    # Because nothing removed the "ready" label or marked the issue escalated,
    # a subsequent dispatch pass must still be able to select it — the
    # opposite of the confirmed-dirty (worktree_unsafe) case above.
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)
    assert result.data["selected_count"] == 1


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
            "mergeStateStatus": "BEHIND",
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


def test_update_open_agent_prs_skips_request_changes_and_blocked(tmp_path: Path) -> None:
    """Issue #404: broadcast mode must not update-branch request_changes or blocked PRs.

    Rework or human intervention will replace the head, so the CI run would be
    guaranteed-wasted runner time.
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
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 101,
            "title": "Fix #125: blocked",
            "url": "https://example.test/pull/101",
            "headRefName": "agent/issue-125-blocked",
            "headRefOid": "sha-ghi789",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #125\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    # PR 456 approved and merged
    pr_456_decision_dir = paths.prs / "pr-456"
    pr_456_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_456_decision_dir / "review-decision.json").write_text(
        json.dumps(
            {"decision": "approved", "reviewed_head_sha": "sha-abc123"},
            indent=2,
        ),
        encoding="utf-8",
    )

    # PR 789 in rework (request_changes)
    pr_789_decision_dir = paths.prs / "pr-789"
    pr_789_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_789_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes"}, indent=2),
        encoding="utf-8",
    )

    # PR 101 blocked
    pr_101_decision_dir = paths.prs / "pr-101"
    pr_101_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_101_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "blocked"}, indent=2),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    results = app._update_open_agent_prs(merged_pr_number=456)

    assert len(results) == 2
    assert all(r["updated"] is False for r in results)
    assert {r["pr_number"] for r in results} == {789, 101}
    assert all(r["skipped_reason"] == "not_approved" for r in results)
    assert fake_gh.pr_update_branch_calls == []


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
            "mergeStateStatus": "BEHIND",
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


def test_update_open_agent_prs_skips_prs_with_pending_required_checks(tmp_path: Path) -> None:
    """Test that PRs with required checks in PENDING/IN_PROGRESS are skipped to avoid cancelling in-flight CI.

    Regression test for issue #209: when ship-it merges a PR with update_open_prs enabled,
    update-branch on sibling PRs cancels their in-flight CI, which can permanently wedge
    aggregate-gate checks. This test verifies the avoidance approach: skip update-branch
    for PRs whose required checks are in PENDING/IN_PROGRESS state.
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

    # Set up a PR with PENDING required checks
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "statusCheckRollup": [
                {
                    "__typename": "CheckRun",
                    "name": "Tests passed",
                    "status": "IN_PROGRESS",  # Required check is in-flight
                    "conclusion": "",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Lint & Format",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Pre-commit",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
            ],
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
            "statusCheckRollup": [
                {
                    "__typename": "CheckRun",
                    "name": "Tests passed",
                    "status": "QUEUED",  # Required check is pending
                    "conclusion": "",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Lint & Format",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Pre-commit",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
            ],
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Simulate merging a different PR: update remaining open PRs
    results = app._update_open_agent_prs(merged_pr_number=999)

    # Both PRs should be skipped due to pending required checks
    assert len(results) == 2
    assert results[0]["pr_number"] == 456
    assert results[0]["updated"] is False
    assert results[0]["skipped_reason"] == "pending-required-checks"
    assert results[1]["pr_number"] == 789
    assert results[1]["updated"] is False
    assert results[1]["skipped_reason"] == "pending-required-checks"

    # Verify update-branch was NOT called
    assert fake_gh.update_branch_ok is True  # Never set to False by a call


def test_update_open_agent_prs_updates_prs_with_completed_required_checks(tmp_path: Path) -> None:
    """Test that PRs with all required checks completed are still updated normally."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # Set up a PR with all required checks SUCCESS
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "statusCheckRollup": [
                {
                    "__typename": "CheckRun",
                    "name": "Tests passed",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Lint & Format",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Pre-commit",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
            ],
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Simulate merging a different PR: update remaining open PRs
    results = app._update_open_agent_prs(merged_pr_number=999)

    # PR should be updated normally
    assert len(results) == 1
    assert results[0]["pr_number"] == 456
    assert results[0]["updated"] is True
    assert "skipped_reason" not in results[0]


def test_is_base_currency_gated_protection_raises_never_lowers_config(tmp_path: Path) -> None:
    """Issue #875 truth table. The merge gate is ``require_current_base OR
    protection.strict`` -- protection can only ever raise the requirement.

    The asymmetry versus ``_is_base_freshness_required`` (which protection
    fully overrides) is deliberate: that one governs the broadcast sweep, whose
    write costs N CI cycles across every open PR, where this one costs a single
    cycle for the PR actually merging. Cheap enough to always pay; the broadcast
    is not, which is what issue #812 correctly established.
    """
    from charlie_work.config import AutoMergeConfig

    strict_true = {"required_status_checks": {"strict": True}}
    strict_false = {"required_status_checks": {"strict": False}}

    # (require_current_base, protection payload or None, expected gate)
    cases: list[tuple[bool, dict[str, Any] | None, bool]] = [
        # THE #875 FIX: config asks for the gate, protection must not veto it.
        (True, strict_false, True),
        # #812's direction still holds: protection alone can turn the gate on.
        (False, strict_true, True),
        # Both agree.
        (True, strict_true, True),
        (False, strict_false, False),
        # Unreadable protection falls back to config, in both directions.
        (True, None, True),
        (False, None, False),
    ]

    for require_current_base, payload, expected in cases:
        config = OrchestratorConfig(
            auto_merge=AutoMergeConfig(
                require_current_base=require_current_base,
                # strategy must stay on: require_current_base=True + "off" is
                # blocked by __post_init__ as an inescapable deferral loop.
                update_open_prs=True,
            )
        )
        paths = runtime_paths(tmp_path, config.runtime.state_dir)
        fake_gh = FakeGitHub()
        if payload is not None:
            fake_gh.branch_protection_overrides["main"] = payload
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)
        assert app._is_base_currency_gated("main") is expected, (
            f"require_current_base={require_current_base} payload={payload!r}"
        )


def test_is_base_currency_gated_skips_protection_read_when_config_requires(
    tmp_path: Path,
) -> None:
    """``require_current_base=True`` short-circuits before the protection read.

    Not merely an optimization: it means the merge gate cannot be disabled by a
    protection API outage, rate limit, or a repo whose protection is readable
    but shaped unexpectedly. The strongest form of failing closed is not
    depending on the remote read at all.

    Deliberately asserts on a recorded call rather than raising from
    ``branch_protection``: ``_is_base_freshness_required`` catches broad
    ``Exception``, so a raise would be swallowed and converted into the same
    ``True`` this test expects -- the assertion would hold whether or not the
    short-circuit existed, making it blind to the very regression it exists to
    catch.
    """
    config = OrchestratorConfig()  # require_current_base defaults to True
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # strict:false would DISABLE the gate if the protection read were consulted,
    # so this payload makes the read's influence observable in the return value
    # as well as in the call log.
    fake_gh.branch_protection_overrides["main"] = {"required_status_checks": {"strict": False}}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    assert app._is_base_currency_gated("main") is True
    assert fake_gh.branch_protection_calls == []


def test_is_base_freshness_required_fails_closed_when_protection_raises(tmp_path: Path) -> None:
    """Fail-closed shape 1/3: the protection read raising an exception must not
    propagate as an unhandled error, and must not be treated as "no freshness
    required" -- it falls back to the require_current_base config value.
    """

    class ExplodingProtectionGitHub(FakeGitHub):
        def branch_protection(self, base: str) -> dict[str, Any] | None:
            raise RuntimeError("simulated network failure")

    config = OrchestratorConfig()  # require_current_base defaults to True
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = ExplodingProtectionGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    assert app._is_base_freshness_required("main") is True


def test_is_base_freshness_required_fails_closed_when_protection_read_errors(
    tmp_path: Path,
) -> None:
    """Fail-closed shape 2/3: an error-value read (404 / rate-limited / gh
    unavailable) surfaces as branch_protection() returning None -- the fake's
    default for a base with no configured override, mirroring the real
    GitHub.branch_protection()'s contract. This must fall back to
    require_current_base, not be treated as "no freshness required".
    """
    config = OrchestratorConfig()  # require_current_base defaults to True
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # no override for "main" -> branch_protection("main") is None
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    assert app._is_base_freshness_required("main") is True


def test_is_base_freshness_required_fails_closed_on_malformed_protection_payload(
    tmp_path: Path,
) -> None:
    """Fail-closed shape 3/3: the protection payload is readable (200 OK) but
    `required_status_checks.strict` is absent or the wrong type -- e.g. a repo
    with protection configured for something other than status checks, or a
    GitHub API response shape this code doesn't anticipate. Every malformed
    shape must fall back to require_current_base, never silently disable the
    gate.
    """
    config = OrchestratorConfig()  # require_current_base defaults to True
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    malformed_payloads: list[dict[str, Any]] = [
        {},  # no required_status_checks key at all
        {"required_status_checks": {}},  # present but no "strict" key
        {"required_status_checks": {"strict": "yes"}},  # wrong type (str, not bool)
        {"required_status_checks": None},  # wrong type (None, not dict)
        {"enforce_admins": {"enabled": True}},  # unrelated protection field only
    ]
    for payload in malformed_payloads:
        fake_gh = FakeGitHub()
        fake_gh.branch_protection_overrides["main"] = payload
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)
        assert app._is_base_freshness_required("main") is True, f"payload={payload!r}"


def test_is_base_freshness_required_bidirectional_config_fallback(tmp_path: Path) -> None:
    """The fallback target on failure is require_current_base itself (per issue
    #812's non-goal: preserve it as the fallback), not a hardcoded True -- an
    operator who has explicitly opted out via config keeps that behavior when
    protection can't be read, matching pre-#812 semantics exactly.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(require_current_base=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # unreadable protection (no override -> None)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    assert app._is_base_freshness_required("main") is False


def test_update_open_agent_prs_broadcast_skips_when_protection_strict_false(
    tmp_path: Path,
) -> None:
    """Issue #812, second half: the broadcast sweep's pr_update_branch write was
    previously ungated by require_current_base entirely (only merge_ready's own
    deferral gate checked it). Prove the new gate now skips the compare-API
    read and the update-branch write together when protection says freshness
    isn't required, recording a distinct skipped_reason for telemetry.
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,  # broadcast
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.branch_protection_overrides["main"] = {"required_status_checks": {"strict": False}}
    fake_gh.prs = [
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "baseRefName": "main",
            "headRefOid": "sha-new456",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        }
    ]
    decision_dir = paths.prs / "pr-789"
    decision_dir.mkdir(parents=True, exist_ok=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-def456"}, indent=2),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    results = app._update_open_agent_prs(merged_pr_number=456)

    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is False
    assert results[0]["skipped_reason"] == "base_freshness_not_required"
    assert fake_gh.pr_update_branch_calls == []


def test_startup_death_does_not_consume_conflict_rework_cap(
    tmp_path: Path,
) -> None:
    """Issue #1106: a rework session that dies at CLI startup (before the
    worker's first tool action) must NOT consume the PR's no-op/conflict
    rework cap.  The cap counters should only count sessions that actually
    ran and produced no useful change.

    This test seeds a PR state with ``last_rework_was_startup_death=True``
    (the flag _reap_restore_rework_requested sets when a dead session is
    classified as a startup death) and verifies that
    _route_janitor_gate_failure_to_rework requeues without incrementing
    ``conflict_rework_attempts``.

    Mutation gate: removing the startup-death check in
    _route_janitor_gate_failure_to_rework makes this test fail (the counter
    increments to 1 instead of staying at 0).
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        ),
        review=ReviewConfig(max_conflict_rework_attempts=2),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # Seed the PR state with the startup-death flag, simulating a dead
    # rework session that was reaped by _reap_restore_rework_requested.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            **state.get("prs", {}).get("456", {}),
            "number": 456,
            "issue_number": 123,
            "last_rework_failure_kind": "launch_failed",
            "last_rework_was_startup_death": True,
        }
        # Issue must NOT be in a pending rework state, so the wrapper
        # reaches the counter-increment path (the startup-death check
        # is right before it).
        state["issues"]["123"] = {
            **state.get("issues", {}).get("123", {}),
            "number": 123,
            "status": "needs_review",
        }
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=False)
    assert result.ok is True
    assert result.data["merge_conflict"] is True

    state = load_state(paths.state_file)
    # The startup-death requeue must NOT have incremented the cap.
    assert state["prs"]["456"].get("conflict_rework_attempts", 0) == 0
    # The issue must have been routed back to rework_requested.
    assert state["issues"]["123"]["status"] == "rework_requested"
    # The startup-death flags must have been cleared.
    assert state["prs"]["456"]["last_rework_was_startup_death"] is False
    assert state["prs"]["456"]["last_rework_failure_kind"] is None


def test_startup_death_does_not_consume_no_op_rework_cap(
    tmp_path: Path,
) -> None:
    """Issue #1106: same as the conflict-rework variant, but for the no-op
    rework cap.  A startup-dead session requeued via the no-op-rework path
    must not increment ``no_op_rework_attempts``.
    """
    config = OrchestratorConfig(
        review=ReviewConfig(max_no_op_rework_attempts=2),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Force issue status to "reviewing" (the orphaned/stuck shape the
    # no-op route exists for — same as test_janitor_no_op_rework_routes_
    # to_rework in test_fix_janitor_routing.py).
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            **state.get("issues", {}).get("123", {}),
            "number": 123,
            "status": "reviewing",
        }
        # Seed the startup-death flag so the janitor gate's startup-death
        # check fires before the counter increment.
        state["prs"]["456"] = {
            **state.get("prs", {}).get("456", {}),
            "last_rework_failure_kind": "launch_failed",
            "last_rework_was_startup_death": True,
        }
        save_state(paths.state_file, state)

    # Same head, same diff as the recorded verdict: no actual content change,
    # so the janitor's no-op-rework signal fires and routes through
    # _route_janitor_gate_failure_to_rework with attempts_key=
    # "no_op_rework_attempts".
    result = app.review(456)
    assert result is not None
    assert result.ok is True
    assert result.data["routed_to_rework"] is True
    assert result.data.get("startup_death_requeue") is True

    state = load_state(paths.state_file)
    # The startup-death requeue must NOT have incremented the no-op cap.
    assert state["prs"]["456"].get("no_op_rework_attempts", 0) == 0
    # The issue must have been routed back to rework_requested.
    assert state["issues"]["123"]["status"] == "rework_requested"
    # The startup-death flags must have been cleared.
    assert state["prs"]["456"]["last_rework_was_startup_death"] is False
    assert state["prs"]["456"]["last_rework_failure_kind"] is None


def test_non_startup_death_still_consumes_conflict_rework_cap(
    tmp_path: Path,
) -> None:
    """Issue #1106 regression guard: a session that genuinely ran and died
    (NOT a startup death — e.g. ``stalled`` with a long runtime) must STILL
    consume the conflict rework cap.  The startup-death exemption must not
    be over-broad.
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        ),
        review=ReviewConfig(max_conflict_rework_attempts=2),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # Seed the PR state with a NON-startup death (stalled, but the flag
    # is False — the session ran long enough to be a genuine no-op).
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            **state.get("prs", {}).get("456", {}),
            "number": 456,
            "issue_number": 123,
            "last_rework_failure_kind": "stalled",
            "last_rework_was_startup_death": False,
        }
        state["issues"]["123"] = {
            **state.get("issues", {}).get("123", {}),
            "number": 123,
            "status": "needs_review",
        }
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=False)
    assert result.ok is True
    assert result.data["merge_conflict"] is True

    state = load_state(paths.state_file)
    # A non-startup death MUST still increment the cap.
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1


def test_is_startup_death_classification() -> None:
    """Issue #1106: unit test for the _is_startup_death classifier itself.

    ``launch_failed`` is always a startup death (the process never
    launched).  ``stalled`` is a startup death only under the threshold
    (the CLI exited before the worker did real work); a longer runtime
    means the worker genuinely ran and got stuck.  Unknown/None failure
    kinds are never startup deaths.
    """
    from charlie_work.workflow import (
        STARTUP_DEATH_THRESHOLD_SECONDS,
        _is_startup_death,
    )

    assert _is_startup_death("launch_failed", 0.0) is True
    assert _is_startup_death("launch_failed", 999.0) is True
    assert _is_startup_death("stalled", 1.0) is True
    assert _is_startup_death("stalled", float(STARTUP_DEATH_THRESHOLD_SECONDS)) is False
    assert _is_startup_death("stalled", float(STARTUP_DEATH_THRESHOLD_SECONDS) + 1) is False
    assert _is_startup_death(None, 0.0) is False
    assert _is_startup_death("worker_blocked", 0.0) is False
    assert _is_startup_death("rate_limited", 1.0) is False


def test_worker_death_bounded_runtime_last_activity_at_fallback(tmp_path: Path) -> None:
    """Issue #1106: ``_worker_death_bounded_runtime_seconds`` must fall back to
    the sidecar's ``last_activity_at`` timestamp when the log file is gone
    (``log_stat()`` returns None).

    The log file may be deleted after the CLI exits (cleanup, tmpfs recycle,
    operator intervention), but the sidecar's ``last_activity_at`` was updated
    each pass by ``update_worker_log_stat`` and is frozen at the last observed
    mtime.  The death-bounded runtime derived from it must be the gap between
    ``started_at`` and that last-activity timestamp — not 0.0 (which would
    only be correct when *no* activity was ever recorded).

    Mutation gate: removing the ``last_activity_at`` fallback branch (so the
    function returns 0.0 when ``log_stat()`` is None) makes this test fail
    (0.0 != ~5.0).
    """
    from charlie_work.worker import WorkerView
    from charlie_work.workflow import _worker_death_bounded_runtime_seconds

    now = datetime.now(UTC)
    started_at = now - timedelta(seconds=300)
    death_at = started_at + timedelta(seconds=5)

    # Log path does not exist — log_stat() will return None.
    missing_log = tmp_path / "nonexistent" / "issue-123.log"

    worker = WorkerView(
        adapter_kind="devin",
        issue_number=123,
        repo_key="",
        pid=99999,
        started_at=started_at.isoformat().replace("+00:00", "Z"),
        process_start_time=started_at.timestamp(),
        log_path=str(missing_log),
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        error=None,
        failure_kind=None,
        reclaimed=None,
        branch="agent/issue-123-fix-search",
        last_activity_at=death_at.isoformat().replace("+00:00", "Z"),
    )

    runtime = _worker_death_bounded_runtime_seconds(worker)
    # The fallback must derive ~5s from last_activity_at, not 0.0.
    assert runtime == pytest.approx(5.0, abs=0.01)


def test_worker_death_bounded_runtime_no_signal_returns_zero(tmp_path: Path) -> None:
    """Issue #1106: ``_worker_death_bounded_runtime_seconds`` must return 0.0
    when neither ``log_stat()`` nor ``last_activity_at`` is available — the CLI
    never wrote anything, which is a startup death by construction.

    Mutation gate: changing the final fallback to return a non-zero value
    (e.g. ``worker.runtime_seconds()``) makes this test fail.
    """
    from charlie_work.worker import WorkerView
    from charlie_work.workflow import _worker_death_bounded_runtime_seconds

    now = datetime.now(UTC)
    started_at = now - timedelta(seconds=300)

    missing_log = tmp_path / "nonexistent" / "issue-123.log"

    worker = WorkerView(
        adapter_kind="devin",
        issue_number=123,
        repo_key="",
        pid=99999,
        started_at=started_at.isoformat().replace("+00:00", "Z"),
        process_start_time=started_at.timestamp(),
        log_path=str(missing_log),
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        error=None,
        failure_kind=None,
        reclaimed=None,
        branch="agent/issue-123-fix-search",
        last_activity_at=None,  # no sidecar activity recorded
    )

    runtime = _worker_death_bounded_runtime_seconds(worker)
    assert runtime == 0.0


def test_unescalate_clears_conflict_cap_escalation_and_merge_ready_redispatches(
    tmp_path: Path,
) -> None:
    """Issue #776 follow-up: the new same-reason guard in
    _route_janitor_gate_failure_to_rework (which refuses to re-route once
    escalation_reason == f"{attempts_key}_cap_exceeded" is already recorded)
    must not become a NEW one-way door of its own. ``charlie unescalate`` is
    the sanctioned re-arm: it clears ``escalation_reason`` on both the issue
    and PR records (``_UNESCALATE_ISSUE_RESET_FIELDS`` /
    ``_UNESCALATE_PR_RESET_FIELDS`` both list it) and zeros
    ``conflict_rework_attempts`` on the PR record, so a PR that is STILL
    conflicting after a human re-arms it gets a genuinely fresh attempts
    budget rather than being silently re-refused by the guard or picking up
    where the exhausted counter left off.
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        ),
        review=ReviewConfig(max_conflict_rework_attempts=2),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # Directly construct the "this lane's own cap already exhausted,
    # escalated" state that _route_janitor_gate_failure_to_rework's
    # cap-exceeded branch produces (workflow.py ~11862-11878), merged over
    # whatever record_review() already wrote -- mirrors the construction
    # convention test_fix_unescalate.py uses rather than re-deriving the
    # escalation via a repeated merge_ready() loop.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            **state["prs"]["456"],
            "status": "escalated",
            "escalation_reason": "conflict_rework_attempts_cap_exceeded",
            "escalation_reasons_seen": ["conflict_rework_attempts_cap_exceeded"],
            "conflict_rework_attempts": 3,
        }
        state["issues"]["123"] = {
            **state["issues"]["123"],
            "status": "escalated",
            "escalation_reason": "conflict_rework_attempts_cap_exceeded",
            "escalation_reasons_seen": ["conflict_rework_attempts_cap_exceeded"],
        }
        save_state(paths.state_file, state)

    # Sanity check: while escalated for THIS lane's own reason, the new guard
    # refuses to re-route at all (the property test C already covers directly
    # -- reconfirmed here as a precondition for what unescalate() is about to
    # undo).
    precheck = app.merge_ready(456, merge=False)
    assert precheck.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 3

    unescalate_result = app.unescalate(issue_number=123)
    assert unescalate_result.ok is True
    assert unescalate_result.data["changed"] is True

    state = load_state(paths.state_file)
    assert "escalation_reason" not in state["prs"]["456"]
    assert "escalation_reason" not in state["issues"]["123"]
    assert "conflict_rework_attempts" not in state["prs"]["456"]
    assert state["prs"]["456"]["status"] == PASSIVE_OPEN_STATUS

    # The conflict is still present (PR still CONFLICTING/DIRTY on GitHub) --
    # a fresh merge_ready() pass must dispatch rework again with a genuinely
    # fresh attempts counter, not pick up where the exhausted counter (3)
    # left off and not be silently refused by the same-reason guard (which no
    # longer matches now that escalation_reason has been cleared).
    result = app.merge_ready(456, merge=False)
    assert result.ok is True
    assert result.data["merge_conflict"] is True

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["status"] == "rework_requested"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    dispatch_events = [
        e for e in state["events"] if e["kind"] == "merge_conflict_rework_requested"
    ]
    assert len(dispatch_events) == 1


def test_update_open_agent_prs_next_mode_syncs_stale_clean_base(tmp_path: Path) -> None:
    """Issue #334: next-mode update lane syncs a CLEAN-but-stale head candidate."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "baseRefName": "main",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    # The next candidate becomes stale organically once PR 456 is merged and
    # advances the fake base tip, even though mergeStateStatus reports CLEAN.
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.record_review(789, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    for idx, pr_number in enumerate((456, 789)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    result = app.merge_ready(456, merge=True)
    assert result.ok is True
    assert result.data["merged"] is True

    update_results = result.data["update_open_prs_results"]
    assert update_results is not None
    assert len(update_results) == 1
    assert update_results[0]["pr_number"] == 789
    assert update_results[0]["updated"] is True
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456-updated"


def test_update_open_agent_prs_all_mode_syncs_stale_clean_base(tmp_path: Path) -> None:
    """Issue #334: all-mode update lane syncs a PR with a CLEAN but stale merge-base."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "baseRefName": "main",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Merge PR 456 first so the base tip advances and the all-mode update lane
    # sees PR 789 as stale organically.
    fake_gh.merge_pr(456, "squash")

    results = app._update_open_agent_prs(merged_pr_number=456)

    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is True
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456-updated"


def test_update_open_agent_prs_next_mode_syncs_head_of_queue(tmp_path: Path) -> None:
    """In merge-train mode, post-merge only syncs the head of the approved queue."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
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
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 101,
            "title": "Fix #125: third",
            "url": "https://example.test/pull/101",
            "headRefName": "agent/issue-125-third",
            "headRefOid": "sha-ghi789",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #125\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Approve in order: 456 first (head), then 789, then 101.
    for pr_number in (456, 789, 101):
        app.record_review(
            pr_number, "approved", summary="lgtm", verdict_provenance="fresh_llm_review"
        )
    # Override timestamps so fast tests don't all land in the same second.
    for idx, pr_number in enumerate((456, 789, 101)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    # Merge the head of the queue.
    result = app.merge_ready(456, merge=True)
    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]

    # Post-merge only the next candidate (789) should be base-synced.
    update_results = result.data["update_open_prs_results"]
    assert update_results is not None
    assert len(update_results) == 1
    assert update_results[0]["pr_number"] == 789
    assert update_results[0]["updated"] is True
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456-updated"
    # The third PR should be untouched (not the head of the queue).
    assert fake_gh.prs[2]["headRefOid"] == "sha-ghi789"


def test_update_open_agent_prs_next_mode_skips_up_to_date_head(tmp_path: Path) -> None:
    """In merge-train mode, an up-to-date head candidate is not re-synced."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
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
            "mergeStateStatus": "CLEAN",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    # Model 789 as already rebased onto the post-merge base so the next
    # candidate is genuinely up-to-date and the merge-train skip path is exercised.
    post_merge_base = "main-merged-sha-abc123"
    fake_gh.commits[post_merge_base] = {"parents": [{"sha": "base-sha"}, {"sha": "sha-abc123"}]}
    fake_gh.commits["sha-def456"] = {"parents": [{"sha": post_merge_base}]}

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.record_review(789, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    # Ensure 456 is the head of the queue.
    for idx, pr_number in enumerate((456, 789)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    result = app.merge_ready(456, merge=True)
    assert result.ok is True
    assert result.data["merged"] is True

    update_results = result.data["update_open_prs_results"]
    assert update_results is not None
    assert len(update_results) == 1
    assert update_results[0]["pr_number"] == 789
    assert update_results[0]["updated"] is False
    assert update_results[0]["skipped_reason"] == "up-to-date"
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456"


def test_update_open_agent_prs_next_mode_reports_compare_unavailable(tmp_path: Path) -> None:
    """Issue #337 rework: a None compare() result must not be reported as up-to-date.

    When the GitHub compare API is unavailable, `_is_base_current` returns None
    and the branch is correctly never synced (fail-closed), but the reported
    reason must be distinct from a genuinely up-to-date branch — otherwise a
    compare-API outage silently masquerades as every PR being current.
    """
    from charlie_work.config import AutoMergeConfig

    class FakeGitHubCompareUnavailable(FakeGitHub):
        """compare() returns None only for the given head SHA, simulating a
        compare-API outage isolated to that candidate (the merged PR's own
        base-freshness check must still resolve normally).
        """

        def __init__(self, unavailable_head: str) -> None:
            super().__init__()
            self._unavailable_head = unavailable_head

        def compare(self, base: str, head: str) -> dict[str, Any] | None:
            if head == self._unavailable_head:
                return None
            return super().compare(base, head)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubCompareUnavailable(unavailable_head="sha-def456")
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
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
            "mergeStateStatus": "CLEAN",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.record_review(789, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    for idx, pr_number in enumerate((456, 789)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    result = app.merge_ready(456, merge=True)
    assert result.ok is True
    assert result.data["merged"] is True

    update_results = result.data["update_open_prs_results"]
    assert update_results is not None
    assert len(update_results) == 1
    assert update_results[0]["pr_number"] == 789
    assert update_results[0]["updated"] is False
    assert update_results[0]["skipped_reason"] == "compare_unavailable"
    # No update-branch call should have been made for the compare-unavailable PR.
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456"


def test_update_open_agent_prs_all_mode_reports_compare_unavailable(tmp_path: Path) -> None:
    """Issue #337 rework: all-mode update lane distinguishes compare-unavailable too."""
    from charlie_work.config import AutoMergeConfig

    class FakeGitHubCompareUnavailable(FakeGitHub):
        """compare() returns None only for the given head SHA, simulating a
        compare-API outage isolated to that candidate.
        """

        def __init__(self, unavailable_head: str) -> None:
            super().__init__()
            self._unavailable_head = unavailable_head

        def compare(self, base: str, head: str) -> dict[str, Any] | None:
            if head == self._unavailable_head:
                return None
            return super().compare(base, head)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubCompareUnavailable(unavailable_head="sha-def456")
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "baseRefName": "main",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    results = app._update_open_agent_prs(merged_pr_number=456)

    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is False
    assert results[0]["skipped_reason"] == "compare_unavailable"
    # No update-branch call should have been made for the compare-unavailable PR.
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456"


def test_front_of_train_only_updates_next_candidate(tmp_path: Path) -> None:
    """Issue #404: a single merge step updates only the new front candidate."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_branch_strategy="front_of_train",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
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
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 101,
            "title": "Fix #125: third",
            "url": "https://example.test/pull/101",
            "headRefName": "agent/issue-125-third",
            "headRefOid": "sha-ghi789",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #125\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    for pr_number in (456, 789, 101):
        app.record_review(
            pr_number, "approved", summary="lgtm", verdict_provenance="fresh_llm_review"
        )
    for idx, pr_number in enumerate((456, 789, 101)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    result = app.merge_ready(456, merge=True)
    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]

    # Exactly one post-merge update-branch, on the new front candidate (789).
    assert fake_gh.pr_update_branch_calls == [789]
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456-updated"
    # The third PR stays behind-base until it reaches the front.
    assert fake_gh.prs[2]["headRefOid"] == "sha-ghi789"


def test_front_of_train_carries_forward_approved_verdict_end_to_end(tmp_path: Path) -> None:
    """Issue #404: non-front approved PRs carry their verdict forward when they reach the front.

    After each merge, the front-of-train update rewrites the approved PR's
    reviewed_head_sha while preserving the patch-id-based verdict, so the next
    merge step can proceed without a re-review.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            update_branch_strategy="front_of_train",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "baseRefName": "main",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 101,
            "title": "Fix #125: third",
            "url": "https://example.test/pull/101",
            "headRefName": "agent/issue-125-third",
            "baseRefName": "main",
            "headRefOid": "sha-ghi789",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #125\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    for pr_number in (456, 789, 101):
        app.record_review(
            pr_number, "approved", summary="lgtm", verdict_provenance="fresh_llm_review"
        )
    for idx, pr_number in enumerate((456, 789, 101)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    # First merge: PR 456 is current, PR 789 is the new front and gets updated.
    result_456 = app.merge_ready(456, merge=True)
    assert result_456.data["merged"] is True
    fake_gh.prs[0]["state"] = "MERGED"

    decision_789 = json.loads((paths.prs / "pr-789" / "review-decision.json").read_text())
    assert decision_789["decision"] == "approved"
    assert decision_789["reviewed_head_sha"] == "sha-def456-updated"

    # Second merge: PR 789's carried-forward verdict lets it merge without re-review.
    result_789 = app.merge_ready(789, merge=True)
    assert result_789.data["merged"] is True
    fake_gh.prs[1]["state"] = "MERGED"

    decision_101 = json.loads((paths.prs / "pr-101" / "review-decision.json").read_text())
    assert decision_101["decision"] == "approved"
    assert decision_101["reviewed_head_sha"] == "sha-ghi789-updated"

    # Exactly two update-branch calls: one for each new front candidate.
    assert fake_gh.pr_update_branch_calls == [789, 101]
    assert fake_gh.merged == [(456, "squash"), (789, "squash")]


def test_front_of_train_skips_request_changes_and_blocked(tmp_path: Path) -> None:
    """Issue #404: front-of-train mode skips request_changes/blocked PRs and
    updates the next approved candidate instead."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_branch_strategy="front_of_train",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
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
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 101,
            "title": "Fix #125: third",
            "url": "https://example.test/pull/101",
            "headRefName": "agent/issue-125-third",
            "headRefOid": "sha-ghi789",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #125\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.record_review(
        789, "request_changes", summary="needs work", verdict_provenance="fresh_llm_review"
    )
    app.record_review(101, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    for idx, pr_number in enumerate((456, 789, 101)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    result = app.merge_ready(456, merge=True)
    assert result.ok is True
    assert result.data["merged"] is True
    # The request_changes PR is not the front; the next approved candidate is updated.
    assert fake_gh.pr_update_branch_calls == [101]


def test_update_open_agent_prs_merge_train_post_sync_head_race_rejected(
    tmp_path: Path,
) -> None:
    """If pr_view returns a non-qualifying head after update-branch, do not bless it.

    Regression test for the _update_open_agent_prs "next" path: a racing push
    must be rejected and the approved head left unchanged.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
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
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.record_review(789, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    for idx, pr_number in enumerate((456, 789)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    # Simulate a racing push that lands on PR 789's branch between update and view.
    fake_gh.pr_head_shas[789] = "racing-sha"
    fake_gh.commits["racing-sha"] = {
        "parents": [{"sha": "other-sha"}],
        "committer": {"login": "not-web-flow"},
        "commit": {"committer": {"name": "Not GitHub"}},
    }

    results = app._update_open_agent_prs(merged_pr_number=456)

    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is False
    assert results[0]["error"] == "post-sync head verification failed"
    # The approved head must remain unchanged.
    decision = json.loads(
        (paths.prs / "pr-789" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-def456"

    # dispatch_rework doesn't include skipped_issue_numbers in its result


def test_loop_surfaces_open_pr_backpressure_fields(tmp_path: Path) -> None:
    """Issue #1129 rework: loop() surfaces the dispatch-scoped open-PR fields.

    The clamp engages inside dispatch() (1 open PR, cap 1 -> dispatch_limit 0);
    _loop_body's 'prefer the dispatch-scoped governor values' copy loop must
    lift open_pr_count/open_pr_max to the top-level CommandResult.data exactly
    like the session-concurrency keys, so a loop() caller sees the backpressure
    that clamped this pass without digging into data["dispatch"].
    """

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_open_agent_prs=1, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.loop(merge=False)

    assert result.ok is True
    # The clamp engaged inside dispatch (dispatch-scoped values present).
    assert result.data["dispatch"]["open_pr_count"] == 1
    assert result.data["dispatch"]["open_pr_max"] == 1
    # And the copy loop lifted them to the top level.
    assert result.data["open_pr_count"] == 1
    assert result.data["open_pr_max"] == 1


def test_open_pr_backpressure_clamps_dispatch_end_to_end(tmp_path: Path) -> None:
    """Issue #1129: app.dispatch() (not just _apply_concurrency_governor) clamps
    fresh-issue dispatch to zero when open agent PRs meet the cap.

    This exercises the _dispatch_impl wiring -- the ``apply_open_pr_backpressure=True``
    argument on the governor call inside _dispatch_impl. Every other open-PR
    backpressure test calls ``_apply_concurrency_governor`` directly, so a
    regression that dropped or flipped that argument would pass all of them
    undetected. This test mirrors test_fleet_concurrency_governor_clamps_when_fleet_live_at_cap
    for the fleet governor: it goes through the public ``app.dispatch()`` entry
    point and asserts on ``result.data`` fields.
    """

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_open_agent_prs=1, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Default FakeGitHub has issue #123 (automated-ready, OPEN) and one open PR
    # #456 (headRefName "agent/issue-123-fix-search") linked to #123. #123 is
    # therefore excluded from candidates (it already has an open PR). Add a
    # second dispatchable issue #124 with no open PR so there is a genuine
    # candidate to clamp -- selected_count==0 proves the clamp engaged, not an
    # empty backlog.
    fake_gh = FakeGitHub()
    fake_gh.issues.append(
        {
            "number": 124,
            "title": "Fix telemetry",
            "url": "https://example.test/issues/124",
            "body": "Telemetry is broken",
            "labels": [{"name": "automated-ready"}],
            "state": "OPEN",
        }
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch()

    # The single open agent PR (#456) fills the max_open_agent_prs=1 cap, so
    # fresh dispatch is clamped to 0 even though issue #124 is dispatchable.
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["open_pr_count"] == 1
    assert result.data["open_pr_max"] == 1


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


def _reconcile_pass_app(
    tmp_path: Path,
    *,
    interval_minutes: int = 30,
    enabled: bool = True,
    gh: Any = None,
) -> OrchestratorApp:
    from charlie_work.config import ReconcilePassConfig

    config = OrchestratorConfig(
        reconcile_pass=ReconcilePassConfig(enabled=enabled, interval_minutes=interval_minutes)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    if gh is None:
        gh = FakeGitHub()
        # Wiring tests for _maybe_reconcile_drift expect a clean repo. The
        # default FakeGitHub carries a sample issue/PR that now resolve through
        # the paginated REST snapshots and would produce unrelated drift.
        gh.issues = []
        gh.prs = []
    return OrchestratorApp(tmp_path, paths, config, gh)


def test_loop_forwards_shared_now_to_cadence_gated_lanes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #828 (critical wiring test): a single frozen ``now`` passed to
    ``loop()`` must reach every cadence-gated lane's ``now`` keyword
    unchanged -- ``_maybe_probe_quota_recovery``, ``_maybe_reconcile_drift``,
    ``_maybe_reclaim_worktrees``, and the module-level
    ``_detect_and_handle_stalled_sessions`` reaper. Each of those functions
    is exercised directly (not via loop()) by its own dedicated test
    elsewhere in this file, so none of those tests would notice a future
    edit that deleted the ``now=now`` forwarding at loop()'s call sites in
    ``_loop_body`` -- this test exists specifically to catch that class of
    regression (L3 "wired", not merely L2 "exists and works standalone").
    Every lane is stubbed to a no-op recorder rather than asserted on
    cadence state, so the test is immune to each lane's own due/not-due
    gating and to the unrelated behavior each lane performs.
    """
    from charlie_work import workflow as workflow_module

    app = _reconcile_pass_app(tmp_path)
    frozen_now = datetime.now(UTC)
    received: dict[str, datetime | None] = {}

    def _record_probe(self: OrchestratorApp, *, now: datetime | None = None) -> None:
        received["probe_quota_recovery"] = now

    def _record_reconcile(self: OrchestratorApp, *, now: datetime | None = None) -> None:
        received["reconcile_drift"] = now

    def _record_reclaim(
        self: OrchestratorApp, *, now: datetime | None = None
    ) -> dict[str, Any] | None:
        received["reclaim_worktrees"] = now
        return None

    def _record_stalled_sessions(
        sessions_dir: Path,
        state_file: Path,
        config: OrchestratorConfig,
        *,
        write_gate: object,
        now: datetime | None = None,
    ) -> list[dict[str, int]]:
        received["stalled_sessions"] = now
        return []

    monkeypatch.setattr(OrchestratorApp, "_maybe_probe_quota_recovery", _record_probe)
    monkeypatch.setattr(OrchestratorApp, "_maybe_reconcile_drift", _record_reconcile)
    monkeypatch.setattr(OrchestratorApp, "_maybe_reclaim_worktrees", _record_reclaim)
    monkeypatch.setattr(
        workflow_module, "_detect_and_handle_stalled_sessions", _record_stalled_sessions
    )

    app.loop(limit=0, now=frozen_now)

    assert received.keys() == {
        "probe_quota_recovery",
        "reconcile_drift",
        "reclaim_worktrees",
        "stalled_sessions",
    }
    for lane, value in received.items():
        assert value is frozen_now, f"{lane} did not receive the pass's frozen now"


def test_maybe_reconcile_drift_noop_when_disabled(tmp_path: Path) -> None:
    """B-AC1/B-AC2: reconcile_pass.enabled=False must skip reconcile entirely --
    no call to reconcile(), no schedule armed, no summary event."""
    app = _reconcile_pass_app(tmp_path, enabled=False)

    def _fail_if_called(*, fix: bool = False) -> CommandResult:
        raise AssertionError("reconcile() must not be called when reconcile_pass is disabled")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(app, "reconcile", _fail_if_called)
    try:
        app._maybe_reconcile_drift()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    assert state.get("reconcile_pass", {}).get("next_reconcile_at") is None
    events = state.get("events", [])
    assert not [e for e in events if str(e.get("kind", "")).startswith("reconcile_pass")]


def test_maybe_reconcile_drift_runs_and_arms_schedule_when_due(tmp_path: Path) -> None:
    """B-AC1/B-AC2: enabled + due calls _reconcile_locked(fix=True) exactly
    once, arms the next-due schedule ~interval_minutes out, and records a
    single reconcile_pass_completed summary event carrying drift counts.

    Patches ``_reconcile_locked``, not ``reconcile`` (merge-lane-recovery
    D-8a): ``_maybe_reconcile_drift`` calls ``_reconcile_locked`` directly
    because loop()'s caller already holds supervisor.lock for the whole
    pass, and re-entering ``reconcile()`` would re-acquire that same
    non-reentrant lock and always no-op. Patching ``reconcile`` here would
    silently never be invoked and this test would falsely pass with 0 calls
    recorded as a bug, not a confirmation -- see
    test_maybe_reconcile_drift_runs_while_supervisor_lock_held for the test
    that actually exercises that lock-contention distinction end to end."""
    from datetime import UTC, datetime

    app = _reconcile_pass_app(tmp_path, interval_minutes=30)
    original_reconcile_locked = app._reconcile_locked
    calls: list[tuple[bool, bool]] = []

    def _counting_reconcile_locked(
        *,
        fix: bool = False,
        skip_dead_session_sweep: bool = False,
        dry_run: bool = False,
    ) -> CommandResult:
        calls.append((fix, skip_dead_session_sweep))
        return original_reconcile_locked(
            fix=fix,
            skip_dead_session_sweep=skip_dead_session_sweep,
            dry_run=dry_run,
        )

    # frozen_now (issue #828) injected so the schedule assertion below is
    # exact instead of racing _reconcile_locked's own duration (or a CI
    # stall). No downstream real-clock dependency follows in this test, so
    # no offset is needed.
    frozen_now = datetime.now(UTC)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(app, "_reconcile_locked", _counting_reconcile_locked)
    try:
        app._maybe_reconcile_drift(now=frozen_now)
    finally:
        monkeypatch.undo()

    # merge-lane-recovery §6-B follow-up: the in-loop caller must skip
    # reconcile's own dead-session sweep -- the loop's stall/dead lanes
    # (_detect_and_handle_stalled_sessions /
    # _classify_dead_sessions_and_update_throttle_state) already ran this
    # exact pass, immediately before this call, with grace-period semantics
    # (max_inconclusive_probe_deferrals) that reconcile.py's sweep does not
    # implement.
    assert calls == [(True, True)]

    state = load_state(app.paths.state_file)
    next_at = state["reconcile_pass"]["next_reconcile_at"]
    expected = (
        (frozen_now + timedelta(minutes=30))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    assert next_at == expected

    events = state.get("events", [])
    completed = [e for e in events if e.get("kind") == "reconcile_pass_completed"]
    assert len(completed) == 1
    assert completed[0]["payload"]["drift_detected"] == 0
    assert completed[0]["payload"]["drift_fixed"] == 0
    assert completed[0]["payload"]["drift_remaining"] == 0


def test_maybe_reconcile_drift_waits_until_due(tmp_path: Path) -> None:
    """The periodic cadence must not re-run reconcile before the armed
    next_reconcile_at timestamp."""
    from datetime import UTC, datetime

    from charlie_work.state import arm_reconcile_pass

    app = _reconcile_pass_app(tmp_path, interval_minutes=30)
    not_due = (
        (datetime.now(UTC) + timedelta(minutes=25))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = arm_reconcile_pass(load_state(app.paths.state_file), not_due)
    save_state(app.paths.state_file, state)

    def _fail_if_called(*, fix: bool = False) -> CommandResult:
        raise AssertionError("must not reconcile before the scheduled time")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(app, "reconcile", _fail_if_called)
    try:
        app._maybe_reconcile_drift()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    assert state["reconcile_pass"]["next_reconcile_at"] == not_due
    events = state.get("events", [])
    assert not [e for e in events if str(e.get("kind", "")).startswith("reconcile_pass")]


def test_maybe_reconcile_drift_defers_on_graphql_rate_limit(tmp_path: Path) -> None:
    """B-AC3: reconcile()'s existing GraphQL rate-limit deferral must be
    preserved, not bypassed -- surfaced as a distinguishable
    reconcile_pass_deferred event rather than a silent no-op."""

    class LowBudgetGitHub(FakeGitHub):
        def check_graphql_rate_limit(self, threshold: int) -> tuple[bool, int, int | None]:
            return (False, 50, 1234567890)

    app = _reconcile_pass_app(tmp_path, gh=LowBudgetGitHub())

    app._maybe_reconcile_drift()

    state = load_state(app.paths.state_file)
    assert state["reconcile_pass"]["next_reconcile_at"] is not None
    events = state.get("events", [])
    deferred = [e for e in events if e.get("kind") == "reconcile_pass_deferred"]
    assert len(deferred) == 1
    assert deferred[0]["payload"]["deferred_reason"] == "graphql_rate_limit"
    assert deferred[0]["payload"]["graphql_remaining"] == 50
    completed = [e for e in events if e.get("kind") == "reconcile_pass_completed"]
    assert completed == []


def test_loop_corrects_escalated_label_divergence_via_reconcile_pass(tmp_path: Path) -> None:
    """B-AC5 (critical wiring test): a state/label divergence of the
    escalated_labels_converged shape -- state says status "escalated", GitHub
    still carries the stale needs-rework label -- must be corrected by a
    single app.loop() pass. This proves _maybe_reconcile_drift is actually
    wired into _loop_body's production call path, not merely present and
    independently callable. Must FAIL if the wiring call site is removed;
    see the removal verification recorded in the PR description."""
    from charlie_work.config import ReconcilePassConfig

    class EscalatedDivergenceGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 40,
                    "title": "issue 40",
                    "url": "https://example.test/issues/40",
                    "body": "",
                    "labels": [{"name": "agent:needs-rework"}],
                    "state": "OPEN",
                }
            ]
            self.prs = []

        def run(
            self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
        ) -> Any:
            if args[:2] == ["issue", "list"]:
                return self.issues
            if args[:2] == ["pr", "list"]:
                return self.prs
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

        def issue_list(self, labels: Any = None, state: Any = None) -> list[dict[str, Any]]:
            return self.issues

        def pr_list(self) -> list[dict[str, Any]]:
            return self.prs

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return {
                issue["number"]
                for issue in self.issues
                if issue["number"] in issue_numbers and issue.get("state") == "OPEN"
            }

        def add_issue_label(self, number: int, label: str) -> bool:
            super().add_issue_label(number, label)
            for issue in self.issues:
                if issue["number"] == number:
                    names = {entry.get("name") for entry in issue["labels"]}
                    if label not in names:
                        issue["labels"].append({"name": label})
            return True

        def remove_issue_label(self, number: int, label: str) -> bool:
            super().remove_issue_label(number, label)
            for issue in self.issues:
                if issue["number"] == number:
                    issue["labels"] = [
                        entry for entry in issue["labels"] if entry.get("name") != label
                    ]
            return True

    gh = EscalatedDivergenceGitHub()
    config = OrchestratorConfig(
        reconcile_pass=ReconcilePassConfig(enabled=True, interval_minutes=30)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    app = OrchestratorApp(tmp_path, paths, config, gh)

    state = load_state(app.paths.state_file)
    state = {
        **state,
        "issues": {**state.get("issues", {}), "40": {"number": 40, "status": "escalated"}},
    }
    save_state(app.paths.state_file, state)

    app.loop(limit=1)

    assert (40, "agent:human-needed") in gh.labels_added
    assert (40, "agent:needs-rework") in gh.labels_removed

    # B-AC7 (critical safety invariant): reconcile must never rewrite an
    # open escalated issue's status to match labels -- only `charlie unescalate`
    # re-enters the machine.
    final_state = load_state(app.paths.state_file)
    assert final_state["issues"]["40"]["status"] == "escalated"


def test_reconcile_closed_unmerged_pr_does_not_drop_escalated_issue_status(
    tmp_path: Path,
) -> None:
    """D-2 regression guard for issue #1066: an OPEN escalated issue whose
    linked PR is CLOSED-unmerged must NOT have its ``status`` key dropped by
    the ``closed_unmerged_pr_issue_state_converged`` drift kind.

    Before #1066's ``DORMANT_CONVERGENCE_EXCLUDED_STATUSES`` exclusion, this
    path dropped the ``status`` key for any ``ACTIVE_STATE_STATUSES`` member
    -- including ``"escalated"`` -- silently detaching the state entry from
    its still-live ``agent:human-needed`` label with no repair path back into
    the human queue (fired in production: issue #894 via PR #948). The
    sibling ``issue_status_normalized`` sweep already excluded escalated via
    ``ORCHESTRATOR_OWNED_ISSUE_STATUSES``; the asymmetry was the defect.

    This test calls ``detect_drift``/``apply_fixes`` directly (the same
    surface the reviewer reproduced the bug on) and asserts both that no
    ``closed_unmerged_pr_issue_state_converged`` drift item is emitted for
    the escalated issue and that the ``status`` key survives ``apply_fixes``.
    """
    from charlie_work.reconcile import apply_fixes, detect_drift

    config = OrchestratorConfig()
    gh = FakeGitHub()
    # OPEN escalated issue with the terminal human-needed label already
    # present -- the exact live shape of issue #894 at the time of the
    # production incident.
    gh.issues = [
        {
            "number": 894,
            "title": "issue 894",
            "url": "https://example.test/issues/894",
            "body": "",
            "labels": [{"name": config.labels.human_needed}],
            "state": "OPEN",
        }
    ]
    # CLOSED-unmerged PR linked to issue #894 via the branch-name convention.
    gh.prs = [
        {
            "number": 948,
            "title": "Fix #894",
            "url": "https://example.test/pull/948",
            "headRefName": "agent/issue-894-fix",
            "baseRefName": "main",
            "headRefOid": "sha-948",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #894",
            "labels": [],
            "isCrossRepository": False,
            "state": "CLOSED",
        }
    ]

    state = empty_state()
    state["issues"]["894"] = {"number": 894, "status": "escalated"}
    state["prs"]["948"] = {"number": 948, "status": "reviewing", "issue_number": 894}
    state_file = tmp_path / "state.json"
    save_state(state_file, state)

    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # No closed_unmerged_pr_issue_state_converged drift item for the escalated
    # issue -- the DORMANT_CONVERGENCE_EXCLUDED_STATUSES exclusion (#1066)
    # prevents it.
    issue_converged = [
        d
        for d in drift
        if d.kind == "closed_unmerged_pr_issue_state_converged" and d.issue_number == 894
    ]
    assert issue_converged == [], (
        f"escalated issue should be excluded from closed_unmerged_pr_issue_state_"
        f"converged, got: {issue_converged}"
    )

    new_state = apply_fixes(gh, state, drift, config, repo_root=tmp_path, state_path=state_file)

    # D-2: the escalated issue's status key must survive -- not dropped to
    # None and not rewritten to any other value.
    assert new_state["issues"]["894"]["status"] == "escalated", (
        f"escalated issue status was rewritten to {new_state['issues']['894'].get('status')!r}"
    )


def test_maybe_reconcile_drift_runs_while_supervisor_lock_held(tmp_path: Path) -> None:
    """merge-lane-recovery D-8a: every production caller of loop() -- the
    cli.py bash-rats handler, fleet_dispatch.py, and supervise.py's
    `while True` -- already holds supervisor.lock for the whole call, so
    _maybe_reconcile_drift must make progress under that exact condition.

    The buggy predecessor called `self.reconcile(fix=True, ...)`, which
    re-acquires supervisor.lock as its first action. Byte-range locks taken
    via msvcrt.locking(LK_NBLCK) are per-handle and non-reentrant even
    within one process (file_lock.py keeps no reentrancy bookkeeping), so
    that reacquisition always failed and reconcile silently no-opped on
    every one of the fleet's loop passes (0 reconcile events across 9,848
    recorded events). The fix calls `self._reconcile_locked(...)` directly,
    bypassing the lock acquisition entirely, since the precondition (lock
    already held by loop()'s caller) is guaranteed by this method's callers.

    A test that does not hold supervisor.lock during the call cannot
    distinguish the two implementations -- both acquire/no-op fine when
    unlocked, which is exactly how this shipped undetected.
    """
    from charlie_work import layout
    from charlie_work.file_lock import try_acquire_byte_range_lock

    app = _reconcile_pass_app(tmp_path, interval_minutes=30)

    supervisor_lock = try_acquire_byte_range_lock(layout.supervisor_lock_path(app.paths.root))
    assert supervisor_lock is not None, "test setup failed to acquire the supervisor lock"
    try:
        app._maybe_reconcile_drift()
    finally:
        supervisor_lock.release()

    state = load_state(app.paths.state_file)
    events = state.get("events", [])

    completed = [e for e in events if e.get("kind") == "reconcile_pass_completed"]
    assert len(completed) == 1, (
        "reconcile_pass_completed must be recorded even while supervisor.lock "
        f"is held by the loop() caller; events were: {events}"
    )

    # The assertion that actually discriminates the bug: the buggy code path
    # (self.reconcile(fix=True, ...) re-acquiring the same non-reentrant
    # lock) always produced this event with this exact reason instead.
    lock_held_skips = [
        e
        for e in events
        if e.get("kind") == "reconcile_pass_skipped"
        and e.get("payload", {}).get("reason") == "supervisor_lock_held"
    ]
    assert lock_held_skips == [], (
        "reconcile must not report supervisor_lock_held when the lock is "
        "already held by this same in-process loop() call -- that reason is "
        "reserved for a genuinely concurrent mop-up --fix"
    )


def _main_ci_reclaim_app(
    tmp_path: Path,
    *,
    enabled: bool = True,
    workflow_filename: str = "ci.yml",
    gh: Any = None,
) -> OrchestratorApp:
    """Wiring-test fixture for _maybe_reclaim_superseded_main_ci (#863/#815).

    Deliberately mirrors _reconcile_pass_app's shape. These wiring tests
    monkeypatch charlie_work.workflow.reclaim_superseded_main_ci_runs itself
    rather than exercising real git/gh calls -- the safety-property logic
    (ancestor checks, tip exemption, race-safety re-fetch) already has
    dedicated coverage in tests/test_main_ci_reclaim.py. This fixture's job
    is only to prove the OrchestratorApp method is wired correctly: gated on
    config.enabled, records the right event for each outcome, and is
    actually called from loop().
    """
    from charlie_work.config import MainCiReclaimConfig

    config = OrchestratorConfig(
        main_ci_reclaim=MainCiReclaimConfig(enabled=enabled, workflow_filename=workflow_filename)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    return OrchestratorApp(tmp_path, paths, config, gh if gh is not None else FakeGitHub())


def test_maybe_reclaim_superseded_main_ci_noop_when_disabled(tmp_path: Path) -> None:
    from charlie_work import workflow as workflow_module

    app = _main_ci_reclaim_app(tmp_path, enabled=False)

    def _fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "reclaim_superseded_main_ci_runs must not be called when main_ci_reclaim is disabled"
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", _fail_if_called)
    try:
        app._maybe_reclaim_superseded_main_ci()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    events = state.get("events", [])
    assert not [e for e in events if str(e.get("kind", "")).startswith("main_ci_reclaim")]


def test_maybe_reclaim_superseded_main_ci_records_event_on_cancellation(
    tmp_path: Path,
) -> None:
    from charlie_work import workflow as workflow_module
    from charlie_work.main_ci_reclaim import MainCiReclaimResult, ReclaimedRun

    app = _main_ci_reclaim_app(tmp_path)
    canned = MainCiReclaimResult(
        ok=True,
        tip_sha="tip-sha",
        candidates_checked=2,
        cancelled=(
            ReclaimedRun(
                run_id=42, head_sha="old-sha", status_before_cancel="queued", created_at="t1"
            ),
        ),
        skipped_not_ancestor=1,
        skipped_started_before_cancel=0,
        cancel_errors=(),
    )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", lambda *a, **k: canned)
    try:
        app._maybe_reclaim_superseded_main_ci()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    events = state.get("events", [])
    cancelled_events = [e for e in events if e.get("kind") == "main_ci_reclaim_cancelled"]
    assert len(cancelled_events) == 1
    payload = cancelled_events[0]["payload"]
    assert payload["tip_sha"] == "tip-sha"
    assert payload["cancelled_run_ids"] == [42]
    assert payload["candidates_checked"] == 2
    assert payload["skipped_not_ancestor"] == 1


def test_maybe_reclaim_superseded_main_ci_no_event_when_nothing_to_reclaim(
    tmp_path: Path,
) -> None:
    """Deliberate no-noise policy (see the method's docstring): this lane has
    no cadence gate, so a durable event on every empty pass would flood
    events.db with zero diagnostic value. Only an actual cancellation or a
    pass-level failure is worth a durable record."""
    from charlie_work import workflow as workflow_module
    from charlie_work.main_ci_reclaim import MainCiReclaimResult

    app = _main_ci_reclaim_app(tmp_path)
    canned = MainCiReclaimResult(ok=True, tip_sha="tip-sha", candidates_checked=0, cancelled=())

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", lambda *a, **k: canned)
    try:
        app._maybe_reclaim_superseded_main_ci()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    events = state.get("events", [])
    assert not [e for e in events if str(e.get("kind", "")).startswith("main_ci_reclaim")]


def test_maybe_reclaim_superseded_main_ci_records_failed_event_on_pass_failure(
    tmp_path: Path,
) -> None:
    from charlie_work import workflow as workflow_module
    from charlie_work.main_ci_reclaim import MainCiReclaimResult

    app = _main_ci_reclaim_app(tmp_path)
    canned = MainCiReclaimResult(ok=False, error="git fetch origin main failed: boom")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", lambda *a, **k: canned)
    try:
        app._maybe_reclaim_superseded_main_ci()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    events = state.get("events", [])
    failed_events = [e for e in events if e.get("kind") == "main_ci_reclaim_failed"]
    assert len(failed_events) == 1
    assert "boom" in failed_events[0]["payload"]["error"]


def test_maybe_reclaim_superseded_main_ci_contains_exception_and_records_event(
    tmp_path: Path,
) -> None:
    """Exception containment is load-bearing: supervise.py's except Exception
    sits outside its while True, so an uncaught exception from this lane
    would kill the whole daemon rather than one pass."""
    from charlie_work import workflow as workflow_module

    app = _main_ci_reclaim_app(tmp_path)

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", _raise)
    try:
        app._maybe_reclaim_superseded_main_ci()  # must not raise
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    events = state.get("events", [])
    failed_events = [e for e in events if e.get("kind") == "main_ci_reclaim_failed"]
    assert len(failed_events) == 1
    assert "RuntimeError" in failed_events[0]["payload"]["error"]
    assert "boom" in failed_events[0]["payload"]["error"]


def test_loop_calls_maybe_reclaim_superseded_main_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L3 'wired' regression guard (#863/#815): proves _loop_body actually
    calls this lane, not just that the lane works standalone."""
    app = _main_ci_reclaim_app(tmp_path)
    calls = {"count": 0}

    def _record(self: OrchestratorApp) -> None:
        calls["count"] += 1

    monkeypatch.setattr(OrchestratorApp, "_maybe_reclaim_superseded_main_ci", _record)
    app.loop(limit=0)
    assert calls["count"] == 1


def test_maybe_reclaim_superseded_main_ci_dry_run_writes_nothing(
    tmp_path: Path,
) -> None:
    """Issue #1324: under dry_run=True, _maybe_reclaim_superseded_main_ci must
    not write any main_ci_reclaim_* event to state.json or events.db, and
    state.json must stay byte-identical to the pre-pass seed. Before the fix,
    _record_event called append_event directly (bypassing self.write_gate) and
    the paired save_state was also raw, so a dry-run pass that found a
    cancellation wrote a real main_ci_reclaim_cancelled event + state.json
    mutation even though nothing was actually cancelled GitHub-side."""
    from charlie_work import workflow as workflow_module
    from charlie_work.config import MainCiReclaimConfig
    from charlie_work.instrumentation import event_counts_by_kind
    from charlie_work.main_ci_reclaim import MainCiReclaimResult, ReclaimedRun
    from charlie_work.state import empty_state, save_state

    config = OrchestratorConfig(
        main_ci_reclaim=MainCiReclaimConfig(enabled=True, workflow_filename="ci.yml")
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    save_state(paths.state_file, empty_state())
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)

    canned = MainCiReclaimResult(
        ok=True,
        tip_sha="tip-sha",
        candidates_checked=2,
        cancelled=(
            ReclaimedRun(
                run_id=42,
                head_sha="old-sha",
                status_before_cancel="queued",
                created_at="t1",
            ),
        ),
        skipped_not_ancestor=1,
        skipped_started_before_cancel=0,
        cancel_errors=(),
    )

    before_bytes = paths.state_file.read_bytes()
    events_before = sum(event_counts_by_kind(paths.state_file).values())

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", lambda *a, **k: canned)
    try:
        app._maybe_reclaim_superseded_main_ci()
    finally:
        monkeypatch.undo()

    assert paths.state_file.read_bytes() == before_bytes, (
        "dry-run main_ci_reclaim pass must leave state.json byte-identical "
        "(issue #1324 WriteGate invariant)"
    )
    events_after = sum(event_counts_by_kind(paths.state_file).values())
    assert events_after == events_before, (
        f"dry-run main_ci_reclaim pass must not write any events.db row "
        f"(before={events_before}, after={events_after})"
    )
    state = load_state(paths.state_file)
    reclaim_events = [
        e for e in state.get("events", []) if str(e.get("kind", "")).startswith("main_ci_reclaim")
    ]
    assert reclaim_events == [], (
        f"dry-run main_ci_reclaim pass must not append any main_ci_reclaim_* "
        f"event to the state.json ring, found: {reclaim_events}"
    )


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


def test_count_live_sessions_corroborates_ghost_worker_via_state_json(
    tmp_path: Path,
) -> None:
    """Issue #343: a live ``worker_pid`` recorded in state.json with NO
    corresponding session sidecar (a "ghost") must still be counted against
    the concurrency governor.

    Before this fix, ``_count_live_sessions`` only counted sidecar files on
    disk. If a sidecar goes missing for a still-live process -- e.g. the
    dead-session reap lane removed it on ambiguous evidence, or any other
    path stranded state.json's dispatch record -- the live worker became
    invisible to the governor and looked like free capacity, letting the
    next dispatch pass launch past the configured concurrency cap even
    though the ghost's process was still actually running (issue #343's
    concrete production instance: pid 23440 verified alive via
    ``Get-Process`` with its sidecar already gone).

    This test uses the current test process's own real, genuinely-alive pid
    (recorded only in state.json, never in a sidecar) to prove the ghost is
    now counted, without needing to spawn or mock a child process.

    MUTATION GATE: removing the ``if state_file is not None:`` state.json
    corroboration block in ``_count_live_sessions``
    (src/charlie_work/workflow.py) makes this test fail -- the count would
    revert to 0 and the ghost worker would look like free capacity again.
    """
    from charlie_work.devin_shell import _get_process_start_time
    from charlie_work.workflow import _count_live_sessions

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    # No sidecar file is written for issue 343 -- this is the "ghost" case:
    # a live worker_pid with no session sidecar on disk at all.

    current_pid = os.getpid()
    current_start_time = _get_process_start_time(current_pid)
    state = load_state(paths.state_file)
    state["issues"]["343"] = {
        "status": "dispatched",
        "worker_pid": current_pid,
        "worker_process_start_time": current_start_time,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)

    count = _count_live_sessions(sessions_dir, paths.state_file)
    assert count == 1, "a ghost worker_pid that is genuinely alive must count against the cap"

    # Without state.json corroboration (the pre-fix behavior), the same ghost
    # is invisible -- pin the contrast so a future regression that silently
    # drops the state_file argument elsewhere is easy to diagnose.
    assert _count_live_sessions(sessions_dir) == 0


def test_loop_emits_concurrency_fields_when_governor_enabled(tmp_path: Path) -> None:
    """Regression test for issue #100: loop() must emit concurrency fields when governor is enabled,
    even when not clamped.

    On origin/main, loop() emitted concurrency_limit/live_session_count/available_slots whenever
    max_concurrent > 0 (governor enabled), regardless of whether it actually clamped anything.
    The initial PR implementation changed this to only emit when clamped, which was a silent behavior
    change. This test ensures the original semantics are preserved: fields appear when the governor
    is enabled, not only when it's actively throttling.
    """
    from charlie_work.config import DevinConfig, DispatchConfig

    # Configure with max_concurrent_sessions=5 (enabled but not clamping in this scenario)
    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=5),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Run loop with no live sessions (governor enabled but not clamped)
    result = app.loop(limit=0)

    # Assert that concurrency fields are present even though governor is not clamped
    assert "concurrency_limit" in result.data
    assert result.data["concurrency_limit"] == 5
    assert "live_session_count" in result.data
    assert result.data["live_session_count"] == 0
    assert "available_slots" in result.data


# --- Issue #108: dependency gate tests --------------------------------------


def test_parse_blockers_extracts_single_blocker() -> None:
    """Test that parse_blockers extracts a single blocker from issue body."""
    from charlie_work.github import parse_blockers

    body = "This issue is blocked by #743"
    blockers = parse_blockers(body)
    assert blockers == [743]


def test_parse_blockers_extracts_multiple_blockers() -> None:
    """Test that parse_blockers extracts multiple blockers from issue body."""
    from charlie_work.github import parse_blockers

    body = "Blocked by #743, #744"
    blockers = parse_blockers(body)
    assert blockers == [743, 744]


def test_parse_blockers_handles_various_patterns() -> None:
    """Test that parse_blockers handles different declaration patterns."""
    from charlie_work.github import parse_blockers

    # Test "Depends on" pattern
    assert parse_blockers("Depends on #123") == [123]

    # Test "Blocked-by:" pattern
    assert parse_blockers("Blocked-by: #456") == [456]

    # Test case insensitivity
    assert parse_blockers("BLOCKED BY #789") == [789]
    assert parse_blockers("depends on #100") == [100]


def test_parse_blockers_returns_empty_for_no_blockers() -> None:
    """Test that parse_blockers returns empty list when no blockers found."""
    from charlie_work.github import parse_blockers

    assert parse_blockers("No blockers here") == []
    assert parse_blockers("") == []
    assert parse_blockers(None) == []


def test_parse_blockers_deduplicates() -> None:
    """Test that parse_blockers deduplicates blocker numbers."""
    from charlie_work.github import parse_blockers

    body = "Blocked by #123, #123, #456"
    blockers = parse_blockers(body)
    assert blockers == [123, 456]


def test_parse_blockers_ignores_downstream_reference_to_self() -> None:
    """Issue #159 regression: prose describing OTHER issues as blocked by
    THIS issue must not be misread as a self-referencing blocker declaration.

    Real trip case from issue #159's "## Dependencies" section: the sentence
    describes #168/#169/#170 as blocked by #159, not #159 declaring its own
    blocker. Naively matching "blocked by #N" anywhere in the text extracted
    159 and treated it as #159 self-declaring a blocker on itself.
    """
    from charlie_work.github import parse_blockers

    body = (
        "## Dependencies\n\n"
        "None — greenfield, no blockers. Downstream: #168 (fleet status), "
        "#169 (global concurrency budget), and #170 (fleet dispatch) all "
        "build on this registry and are blocked by #159.\n\n"
        "_Filed from the fleet-management & worker-supervision design._\n"
    )
    assert parse_blockers(body) == []


def test_parse_blockers_quoted_backtick_phrase_does_not_self_block() -> None:
    """Issue #1454 regression: an issue whose body quotes ANOTHER issue's
    blocker declaration inside a Markdown backtick code span must not be
    classified as blocked by the quoted number.

    Reproduces the #1927 incident shape: a bug report ABOUT the parser
    flapping on #887/#888 quoted their trigger phrase on its own line, with
    no preceding issue ref in the clause, so the old backward-only guard
    could not suppress it and the describing issue self-gated on #886.
    """
    from charlie_work.github import parse_blockers

    body = (
        "## Symptom\n\n"
        "The parser flaps on #887 and #888.\n\n"
        "Their bodies contain the trigger phrase:\n\n"
        "`blocked by #886`\n\n"
        "which the parser reads as a self-declaration.\n"
    )
    assert parse_blockers(body) == []


def test_parse_blockers_quoted_double_quote_phrase_does_not_self_block() -> None:
    """Issue #1454: a trigger phrase inside straight double quotes is quoted
    prose, not a self-declaration."""
    from charlie_work.github import parse_blockers

    body = (
        "## Symptom\n\n"
        'The parser sees the literal phrase "blocked by #886" in #887\'s '
        "body and misreads it as a self-declaration.\n"
    )
    assert parse_blockers(body) == []


def test_parse_blockers_forward_foreign_ref_does_not_self_block() -> None:
    """Issue #1454: a match whose clause carries another #NNN AFTER it (e.g.
    an issue-referencing parenthetical) describes that other issue, not this
    one. The old guard only looked backward and missed this."""
    from charlie_work.github import parse_blockers

    body = (
        "## Symptom\n\n"
        "The trigger phrase blocked by #886 (see #887) appears verbatim in "
        "the upstream body.\n"
    )
    assert parse_blockers(body) == []


def test_parse_blockers_genuine_declaration_still_gates() -> None:
    """Issue #1454 regression: a genuine first-person blocker declaration
    (the #887/#888 shape) must still gate. The quoted-phrase fix must not
    suppress real declarations."""
    from charlie_work.github import parse_blockers

    assert parse_blockers("This issue is blocked by #743") == [743]
    assert parse_blockers("Blocked by #743, #744") == [743, 744]
    assert parse_blockers("Depends on #123") == [123]
    assert parse_blockers("Blocked-by: #456") == [456]
    # Genuine declaration with surrounding prose but no foreign issue ref.
    body = "## Summary\n\nFix the parser.\n\nBlocked by #886\n"
    assert parse_blockers(body) == [886]


def test_parse_blockers_stray_backtick_elsewhere_does_not_swallow_declaration() -> None:
    """Issue #1454 rework: an unrelated/unbalanced backtick ELSEWHERE in the
    body must not pair with a later backtick to form a code span that
    envelopes a genuine 'Blocked by #NNN' declaration and silently drop it.

    The body below has a stray opening backtick on the first line and a
    closing backtick on the last line. Against the whole-document span scan
    (the pre-rework guard 1) the regex ``(`+)(.+?)(\\1)`` with re.DOTALL
    matches one span whose content runs from "broken thing." through
    "Blocked by #159" through "See also ", so the declaration is
    misclassified as quoted prose and dropped -- a false negative. Scoping
    the span search to the containing clause (bounded by newlines) leaves
    the clause "Blocked by #159" with no backticks, so the declaration gates.
    """
    from charlie_work.github import parse_blockers

    body = "TODO: fix the `broken thing.\nBlocked by #159\nSee also `foo`.\n"
    assert parse_blockers(body) == [159]


def test_parse_blockers_stray_double_quote_elsewhere_does_not_swallow_declaration() -> None:
    """Issue #1454 rework: an unrelated/unbalanced straight double quote
    ELSEWHERE in the body must not pair with a later quote to envelope a
    genuine declaration. Same false-negative shape as the backtick case:
    ``"([^"]*)"`` matches from the first quote to the next, swallowing the
    declaration line in between when scanned over the whole document.
    Scoping to the clause leaves "Blocked by #743" with no quotes, so it
    gates.
    """
    from charlie_work.github import parse_blockers

    body = 'The error was "connection refused.\nBlocked by #743\nThen it said "done".\n'
    assert parse_blockers(body) == [743]


def test_parse_blockers_fenced_code_block_does_not_self_gate() -> None:
    """Issue #1454 rework round 2: a 'Blocked by #NNN' line inside a real
    multi-line triple-backtick fenced code block (fence markers on separate
    lines from the content) must NOT self-gate.

    The clause-scoped inline span guard (round 1) cannot detect this: clause
    bounds break on newlines, so the fenced content line ``blocked by #886``
    is its own clause with no fence markers in it, and the declaration is
    misclassified as a genuine self-declaration. The fenced-block check runs
    against the full document with absolute offsets and suppresses it.
    """
    from charlie_work.github import parse_blockers

    body = (
        "## Symptom\n\n"
        "The upstream issue's body contains:\n\n"
        "```python\n"
        "blocked by #886\n"
        "```\n\n"
        "which the parser used to misread as a self-declaration.\n"
    )
    assert parse_blockers(body) == []


def test_parse_blockers_fenced_code_block_tilde_fence_does_not_self_gate() -> None:
    """Issue #1454 rework round 2: ``~~~`` fences are equivalent to triple-
    backtick fences in CommonMark and must be detected the same way."""
    from charlie_work.github import parse_blockers

    body = "## Example\n\n~~~\nblocked by #886\n~~~\n"
    assert parse_blockers(body) == []


def test_parse_blockers_fenced_block_with_language_tag_does_not_self_gate() -> None:
    """Issue #1454 rework round 2: an opening fence carrying an info string
    (e.g. ```` ```bash ````) must still be recognized as a fence."""
    from charlie_work.github import parse_blockers

    body = "## Repro\n\n```bash\n$ echo 'blocked by #886'\n```\n"
    assert parse_blockers(body) == []


def test_parse_blockers_genuine_declaration_outside_fenced_block_still_gates() -> None:
    """Issue #1454 rework round 2: a genuine declaration on a line OUTSIDE a
    fenced block must still gate. The fenced-block guard must not over-suppress
    real declarations that merely share a document with a fenced block."""
    from charlie_work.github import parse_blockers

    body = "## Summary\n\nFix the parser.\n\n```python\nblocked by #886\n```\n\nBlocked by #743\n"
    assert parse_blockers(body) == [743]


def test_detect_prose_only_dependencies_do_not_dispatch_before() -> None:
    """Test detection of 'do not dispatch before' pattern (issue #225)."""
    from charlie_work.github import detect_prose_only_dependencies

    body = "Do not dispatch before P2-T2/P2-T3 have landed."
    assert detect_prose_only_dependencies(body) is True

    body = "DO NOT DISPATCH BEFORE #123 merges"
    assert detect_prose_only_dependencies(body) is True


def test_detect_prose_only_dependencies_task_references() -> None:
    """Test detection of task references like P2-T3 (issue #225)."""
    from charlie_work.github import detect_prose_only_dependencies

    body = "This depends on P2-T2 and P2-T3."
    assert detect_prose_only_dependencies(body) is True

    body = "Wait for P1-T5 to complete first."
    assert detect_prose_only_dependencies(body) is True


def test_detect_prose_only_dependencies_wait_for_pr() -> None:
    """Test detection of 'wait for PR' pattern (issue #225)."""
    from charlie_work.github import detect_prose_only_dependencies

    body = "Wait for this PR to merge before starting."
    assert detect_prose_only_dependencies(body) is True

    body = "wait for that PR to land"
    assert detect_prose_only_dependencies(body) is True


def test_detect_prose_only_dependencies_with_structured_blockers() -> None:
    """Test that issues with structured blockers are handled correctly (issue #225)."""
    from charlie_work.github import detect_prose_only_dependencies, parse_blockers

    body = "Blocked by #123"
    # Has structured blockers, so prose-only detection returns False
    # (doesn't match the prose patterns)
    assert detect_prose_only_dependencies(body) is False
    assert parse_blockers(body) == [123]

    # Test case with both prose and structured blockers
    body = "Do not dispatch before P2-T2. Blocked by #123"
    # Has prose pattern, so detection returns True
    # But caller will check parse_blockers and see structured blockers exist
    assert detect_prose_only_dependencies(body) is True
    assert parse_blockers(body) == [123]


def test_detect_prose_only_dependencies_no_match() -> None:
    """Test that normal issue bodies don't trigger false positives (issue #225)."""
    from charlie_work.github import detect_prose_only_dependencies

    body = "This is a normal issue with no dependencies."
    assert detect_prose_only_dependencies(body) is False

    body = "Fix the bug in the authentication module."
    assert detect_prose_only_dependencies(body) is False

    body = ""
    assert detect_prose_only_dependencies(body) is False


def test_detect_prose_only_dependencies_descriptive_task_refs_no_match() -> None:
    """REQUIRED 1 negative tests: bare/descriptive task-marker mentions must NOT match.

    Plan-generated issue bodies routinely mention task markers descriptively
    (e.g. 'implements P2-T4', title suffix '(P2-T4)', 'this task is P2-T3 of
    the plan'). These must not fire the detector — only dependency-context uses
    should match (REQUIRED 1, PR #230 rework).
    """
    from charlie_work.github import detect_prose_only_dependencies

    # "implements P2-T4" — descriptive, not a dependency declaration
    body = "This implements P2-T4 of the expiry plan."
    assert detect_prose_only_dependencies(body) is False

    # Commit-style title with inline task marker — descriptive reference in prose
    body = "fix(expiry): thread careers-match (P2-T4)"
    assert detect_prose_only_dependencies(body) is False

    # "this task is P2-T3 of the plan" — describes what the issue is, not what it depends on
    body = "This task is P2-T3 of the plan."
    assert detect_prose_only_dependencies(body) is False


def test_github_dependencies_404_tolerance(tmp_path: Path) -> None:
    """Test that 404 errors from dependencies API are handled gracefully (feature not available)."""
    from charlie_work.github import get_github_issue_dependencies

    class FakeGitHubWith404(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Simulate 404 response (feature not available)
            self.dependencies_response = {"message": "Not Found", "status": 404}

    fake_gh = FakeGitHubWith404()
    result = get_github_issue_dependencies(fake_gh, 123)
    assert result == []


def test_github_dependencies_transient_error_fail_open(tmp_path: Path) -> None:
    """Test that transient errors from dependencies API fail open with warning."""
    from charlie_work.github import get_github_issue_dependencies
    from unittest.mock import patch

    class FakeGitHubWithTransientError(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Simulate transient error (None return)
            self.dependencies_response = None

    fake_gh = FakeGitHubWithTransientError()

    # get_github_issue_dependencies moved to charlie_work.github_capabilities.issues
    # in L07 (issue #1591); its module logger moved with it, so the patch target
    # follows the symbol. The function object is still re-exported from
    # charlie_work.github (imported above), only its logger binding relocated.
    with patch("charlie_work.github_capabilities.issues.logger") as mock_logger:
        result = get_github_issue_dependencies(fake_gh, 123)
        assert result == []
        # Should have logged a warning about the transient error
        assert any(
            "returned None" in str(call.args[0]) for call in mock_logger.warning.call_args_list
        )


def test_github_dependencies_successful_parse(tmp_path: Path) -> None:
    """Test that successful dependencies API responses are parsed correctly."""
    from charlie_work.github import get_github_issue_dependencies

    class FakeGitHubWithDependencies(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Simulate successful response with dependencies
            self.dependencies_response = [
                {"number": 100, "url": "https://example.test/issues/100"},
                {"number": 200, "url": "https://example.test/issues/200"},
            ]

    fake_gh = FakeGitHubWithDependencies()
    result = get_github_issue_dependencies(fake_gh, 123)
    assert result == [100, 200]


def test_get_github_issue_dependencies_caches_successful_result_per_pass(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #870: get_github_issue_dependencies() made one live `gh api`
    call per invocation, with zero caching -- called once per ready issue
    from _get_open_blockers, and _get_open_blockers itself is called once per
    issue from *two* separate places (_filter_blocked_issues and
    _summarize_issue), so every ready issue's dependencies were fetched
    twice per status() call. A successful resolution for a given issue
    number must now cost exactly one live call per pass.

    Uses the real GitHub (not FakeGitHub, whose `run()` has its own
    dependencies_response stand-in and doesn't exercise the production
    _list_cache path) with subprocess.run mocked.
    """
    from charlie_work.github import get_github_issue_dependencies

    calls: list[str] = []

    def fake_run(command, **kwargs):
        # command: ["gh", "api", "repos/{owner}/{repo}/issues/<N>/dependencies/blocked_by"]
        calls.append(command[2])
        payload = json.dumps([{"number": 200}])
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = github_module.GitHub(repo_root=tmp_path)

    first = get_github_issue_dependencies(gh, 100)
    second = get_github_issue_dependencies(gh, 100)

    assert first == [200]
    assert second == [200]
    assert len(calls) == 1  # second call is a pure cache hit, no live fetch

    # A different issue number is a distinct cache key -- costs a fresh read.
    get_github_issue_dependencies(gh, 101)
    assert len(calls) == 2

    # invalidate_list_cache() (called once per orchestrator pass) must force
    # a fresh read on the next call for the same issue number.
    gh.invalidate_list_cache()
    get_github_issue_dependencies(gh, 100)
    assert len(calls) == 3


def test_github_dependencies_unexpected_type_fail_open(tmp_path: Path) -> None:
    """Test that unexpected return types from dependencies API fail open with warning."""
    from charlie_work.github import get_github_issue_dependencies
    from unittest.mock import patch

    class FakeGitHubWithUnexpectedType(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Simulate unexpected return type
            self.dependencies_response = "unexpected string"

    fake_gh = FakeGitHubWithUnexpectedType()

    # See the transient-error test above: the moved function's logger now lives
    # at charlie_work.github_capabilities.issues.logger (L07, issue #1591).
    with patch("charlie_work.github_capabilities.issues.logger") as mock_logger:
        result = get_github_issue_dependencies(fake_gh, 123)
        assert result == []
        # Should have logged a warning about the unexpected type
        assert any(
            "unexpected type" in str(call.args[0]) for call in mock_logger.warning.call_args_list
        )


def test_blocked_issue_does_not_consume_slot(tmp_path: Path) -> None:
    """Test that blocked issues don't consume dispatch slots (slot invariant).

    When a blocked issue is ordered ahead of an eligible candidate with
    dispatch_limit=1, the eligible one should dispatch and the blocked one
    should be in the skip event.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with blocked issue first, then eligible issue
    class FakeGitHubWithSlotTest(FakeGitHub):
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

    fake_gh = FakeGitHubWithSlotTest()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    # Only the eligible issue should dispatch (blocked issue doesn't consume slot)
    assert result.data["selected_count"] == 1
    assert result.data["attempted_count"] == 1

    # Verify the dispatched issue is exactly 101 (the eligible one), not 100 (blocked)
    assert len(result.data["sessions"]) == 1
    assert result.data["sessions"][0]["issue_number"] == 101
    assert len(result.data["dispatch_results"]) == 1
    assert result.data["dispatch_results"][0]["issue_number"] == 101

    # Verify issue 100 is absent from dispatch results and sessions
    dispatched_issue_numbers = {session["issue_number"] for session in result.data["sessions"]}
    assert 100 not in dispatched_issue_numbers
    assert 100 not in {result["issue_number"] for result in result.data["dispatch_results"]}

    # Check that the blocked issue was skipped
    state = load_state(paths.state_file)
    blocked_events = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_skip_blocked"
    ]
    assert len(blocked_events) == 1
    assert blocked_events[0]["payload"]["issue"] == 100
    assert blocked_events[0]["payload"]["blockers"] == [200]


def test_status_includes_blocked_section(tmp_path: Path) -> None:
    """Issue #108: status (roll-call) should include blocked section."""
    config = OrchestratorConfig(
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with a blocked issue
    class FakeGitHubWithBlockers(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Override with only the test issues
            self.issues = [
                {
                    "number": 752,
                    "title": "Dependent issue",
                    "url": "https://example.test/issues/752",
                    "body": "Blocked by #743",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
                {
                    "number": 743,
                    "title": "Blocker issue",
                    "url": "https://example.test/issues/743",
                    "body": "Foundation work",
                    "labels": [],  # Not ready
                    "state": "OPEN",
                },
            ]

        def issue_list(self, labels=None, state=None):
            # Support both old and new signature
            if isinstance(labels, str):
                ready_label = labels
                # Only return issue 752 (the dependent one)
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
            return {743}

    fake_gh = FakeGitHubWithBlockers()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.status()

    # Check that blocked section is present
    assert "blocked" in result.data
    assert len(result.data["blocked"]) == 1
    assert result.data["blocked"][0]["issue"] == 752
    assert result.data["blocked"][0]["blockers"] == [743]

    # available_issue_count should exclude blocked issues
    assert result.data["available_issue_count"] == 0


def test_status_prefetch_uses_batched_graphql_for_blocker_data(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #923: `fleet status --json` spawned 86 `gh` subprocesses, mostly
    one `gh api .../dependencies/blocked_by` per ready issue plus one `gh issue
    view` per unique blocker. That is replaced by a single batched GraphQL
    query per repo that fetches both the native blockedBy relationships and the
    blockers' open/closed states in one subprocess.

    This uses a real GitHub (not the hand-rolled FakeGitHub used elsewhere in
    this file) with subprocess.run mocked, so every `gh` invocation status()
    actually makes is visible and countable.

    Two ready issues (887, 888) both declare the same open blocker (886) via
    GitHub-native dependencies. Before this fix: 2 dependency-REST calls x2
    consumers = 4, plus 886's issue-view fetched twice per consumer = 4. After
    the batching fix: one GraphQL query fetches dependencies and blocker states
    for the whole set; no further `gh` calls are needed.
    """
    calls: list[list[str]] = []

    def make_issue(number: int) -> dict[str, Any]:
        return {
            "number": number,
            "title": f"Issue {number}",
            "url": f"https://example.test/issues/{number}",
            "body": "",
            "labels": [{"name": "automated-ready"}],
            "author": {"login": "tester"},
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
            "state": "OPEN",
        }

    ready_issues = [make_issue(887), make_issue(888)]

    def fake_run(command, **kwargs):
        args = command[1:]  # drop leading "gh"
        if args and args[-2:] == ["--json", "nonexistent"]:
            # OrchestratorApp.__init__'s validate_field_lists() startup probe
            all_field_list_constants = [
                "ISSUE_LIST_FIELDS",
                "ISSUE_VIEW_FIELDS",
                "PR_LIST_FIELDS",
                "MERGED_PR_LIST_FIELDS",
                "PR_VIEW_FIELDS",
                "PR_CHECKS_FIELDS",
                "LABEL_LIST_FIELDS",
                "RECONCILE_PR_FIELDS",
                "RECONCILE_ISSUE_FIELDS",
                "RUN_LIST_FIELDS",
            ]
            available_fields = sorted(
                {
                    field
                    for name in all_field_list_constants
                    for field in getattr(github_module, name).split(",")
                }
            )
            stderr = (
                'Unknown JSON field: "nonexistent"\nAvailable fields:\n  '
                + "\n  ".join(available_fields)
                + "\n"
            )
            return subprocess.CompletedProcess(
                args=command, returncode=1, stdout="", stderr=stderr
            )

        calls.append(command)
        if args[:2] == ["issue", "list"]:
            payload = json.dumps(ready_issues)
        elif args[:2] == ["pr", "list"]:
            payload = json.dumps([])
        elif args[0] == "api" and len(args) >= 2 and "pulls?state=closed" in args[1]:
            # Issue #1337: status() now calls merged_pr_list() to compute the
            # merged-PR coverage exclusion set for the reachability classifier.
            # No merged PRs in this test -> empty page breaks pagination.
            payload = json.dumps([])
        elif args[0] == "api" and len(args) >= 2 and args[1] == "graphql":
            payload = json.dumps(
                {
                    "data": {
                        "repository": {
                            "i_887": {
                                "number": 887,
                                "blockedBy": {
                                    "nodes": [{"number": 886, "state": "OPEN"}],
                                    "pageInfo": {"hasNextPage": False},
                                },
                            },
                            "i_888": {
                                "number": 888,
                                "blockedBy": {
                                    "nodes": [{"number": 886, "state": "OPEN"}],
                                    "pageInfo": {"hasNextPage": False},
                                },
                            },
                        }
                    }
                }
            )
        else:
            raise AssertionError(f"Unexpected gh command in status(): {command}")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    config = OrchestratorConfig(devin=DevinConfig())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = github_module.GitHub(repo_root=tmp_path)
    # Avoid a real `git remote` call; the owner/name are only used for the
    # GraphQL variables, and the mocked response is the same either way.
    gh._list_cache[("_repo_owner_name",)] = ("owner", "repo")
    app = OrchestratorApp(tmp_path, paths, config, gh)

    result = app.status()

    assert result.ok is True
    assert result.data["available_issue_count"] == 0
    blocked_by_issue = {b["issue"]: b["blockers"] for b in result.data["blocked"]}
    assert blocked_by_issue == {887: [886], 888: [886]}
    summaries = {s["number"]: s["dependencies"] for s in result.data["issues"]}
    assert summaries[887] == {"declared": [886], "open": [886]}
    assert summaries[888] == {"declared": [886], "open": [886]}

    graphql_calls = [c for c in calls if c[1:3] == ["api", "graphql"]]
    rest_dependency_calls = [
        c for c in calls if c[1] == "api" and "dependencies/blocked_by" in c[2]
    ]
    issue_view_calls = [c for c in calls if c[1:3] == ["issue", "view"]]

    # The whole dependency + blocker-state lookup is now one batched GraphQL
    # query for the two ready issues, not N per-issue REST calls + M issue views.
    assert len(graphql_calls) == 1
    assert len(rest_dependency_calls) == 0
    assert len(issue_view_calls) == 0


def test_status_includes_stalled_section(tmp_path: Path) -> None:
    """Issue #109: status (roll-call) should include stalled section."""
    from datetime import UTC, datetime, timedelta
    import os
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with a ready issue
    class FakeGitHubWithStalled(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 109,
                    "title": "Test issue",
                    "url": "https://example.test/issues/109",
                    "body": "Test body",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
            ]

        def issue_list(self, labels=None, state=None):
            # Support both old and new signature
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

    fake_gh = FakeGitHubWithStalled()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a fake stalled session sidecar
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file with old mtime (stalled by time)
    log_file = sessions_dir / "issue-109.log"
    log_file.write_text("working on issue\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a sidecar with a fake PID
    sidecar = sessions_dir / "issue-109.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 109,
                "branch": "agent/issue-109",
                "worktree_path": "/fake/path",
                "prompt_path": "/fake/prompt",
                "command": ["devin", "--print"],
                "pid": 99999,  # Fake PID that won't exist
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_file),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    # Mock is_session_alive to return True for PID 99999 so detection runs
    with patch("charlie_work.worker.is_session_alive", return_value=True):
        result = app.status()

    # Check that stalled section contains the issue number and pid
    assert "stalled" in result.data
    assert isinstance(result.data["stalled"], list)
    assert any(entry["issue"] == 109 for entry in result.data["stalled"])
    assert any(entry["pid"] == 99999 for entry in result.data["stalled"])


def test_status_includes_workers_section(tmp_path: Path) -> None:
    """Issue #167: status (roll-call) should include workers section with health classification."""
    from datetime import UTC, datetime
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub
    class FakeGitHubWithWorkers(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []

        def issue_list(self, labels=None, state=None):
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

    fake_gh = FakeGitHubWithWorkers()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a fake live session sidecar
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file with recent mtime (not stalled)
    log_file = sessions_dir / "issue-167.log"
    log_file.write_text("working on issue\n", encoding="utf-8")

    # Create a sidecar with a fake PID
    sidecar = sessions_dir / "issue-167.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 167,
                "branch": "agent/issue-167",
                "worktree_path": "/fake/path",
                "prompt_path": "/fake/prompt",
                "command": ["devin", "--print"],
                "pid": 12345,
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_file),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    # Mock is_session_alive to return True for PID 12345
    with patch("charlie_work.worker.is_session_alive", return_value=True):
        result = app.status()

    # Check that workers section is present
    assert "workers" in result.data
    assert isinstance(result.data["workers"], list)
    assert len(result.data["workers"]) == 1

    # Check worker entry has required fields
    worker = result.data["workers"][0]
    assert worker["issue"] == 167
    assert worker["adapter"] == "devin"
    assert worker["repo"] == "test-owner/test-repo"
    assert "health" in worker
    assert "runtime_seconds" in worker
    assert "last_activity_at" in worker
    assert worker["tool_calls"] is None  # Devin has no structured stream
    assert worker["tokens"] is None
    assert worker["cost_usd"] is None


def test_status_workers_section_claude_code_rework_layout(tmp_path: Path) -> None:
    """Issue #329 (F1): status()'s _summarize_worker must use the canonical
    events.jsonl derivation for rework-layout claude-code sessions too.

    A rework claude-code session logs to ``issue-<n>-rework.claude.log``, with
    its structured events at ``issue-<n>-rework.events.jsonl`` -- not
    ``issue-<n>.events.jsonl``, which the old rework=False-only
    ``_events_path(sessions_dir, issue_number)`` derivation would read instead
    (a stale prior attempt's tool_calls/tokens/cost_usd, or nothing). This
    test plants both a stale non-rework events.jsonl and the real rework
    sibling, and asserts the workers section reports the rework sibling's
    usage, not the stale one's.
    """
    from datetime import UTC, datetime

    config = OrchestratorConfig(devin=DevinConfig())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubWithWorkers(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []

        def issue_list(self, labels=None, state=None):
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

    fake_gh = FakeGitHubWithWorkers()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    issue_number = 329
    log_file = sessions_dir / f"issue-{issue_number}-rework.claude.log"
    log_file.write_text("reworking issue\n", encoding="utf-8")

    # Stale events.jsonl from a prior (non-rework) attempt: different usage.
    stale_events_file = sessions_dir / f"issue-{issue_number}.events.jsonl"
    stale_events_file.write_text(
        '{"type": "tool_call", "tokens": 111, "cost_usd": 0.11}\n',
        encoding="utf-8",
    )

    # The real rework events.jsonl sibling: the usage that must be reported.
    events_file = sessions_dir / f"issue-{issue_number}-rework.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 987654, "cost_usd": 12.34}\n{"type": "tool_call"}\n',
        encoding="utf-8",
    )

    record = ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path="/fake/path",
        prompt_path="/fake/prompt",
        command=("claude", "-p"),
        pid=54321,
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
    )
    sidecar = sessions_dir / f"issue-{issue_number}-rework.claude.json"
    sidecar.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        result = app.status()

    assert len(result.data["workers"]) == 1
    worker = result.data["workers"][0]
    assert worker["issue"] == issue_number
    assert worker["adapter"] == "claude-code"
    assert worker["tool_calls"] == 2
    assert worker["tokens"] == 987654
    assert worker["cost_usd"] == 12.34


@pytest.mark.real_activity_probe_live
def test_status_workers_not_killed_when_real_activity_probe_fresh(tmp_path: Path) -> None:
    """Issue #301 status()-path wiring: a claude-code worker whose sidecar log is
    frozen but whose events.jsonl sibling carries fresh activity must be
    reported healthy (not stalled) in the status() workers section (~1616).

    A future edit that drops the ``probe`` argument from the
    classify_worker_health call inside status(), or that neuters the
    claude_events_jsonl Source-3 construction in
    post_mortem.real_activity_for_worker, must make this test fail (the
    worker reports health="stalled") rather than silently reverting to
    mtime-only classification.

    Marked ``real_activity_probe_live`` so the autouse
    ``_stub_real_activity_probe_for_stalled_tests`` fixture leaves
    ``real_activity_probe_for`` unstubbed for this test only (rename-safe
    opt-out; issue #307 non-blocking cleanup).
    """
    from datetime import timedelta

    config = OrchestratorConfig(
        devin=DevinConfig(),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    issue_number = 303

    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    log_path.write_text("Working on task...\nLast line", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_path, (time.time(), old_time.timestamp()))

    events_path = sessions_dir / f"issue-{issue_number}.events.jsonl"
    fresh_time = datetime.now(UTC) - timedelta(minutes=1)
    events_path.write_text(
        f'{{"type": "tool_call", "timestamp": "{fresh_time.isoformat()}"}}\n',
        encoding="utf-8",
    )
    os.utime(events_path, (time.time(), fresh_time.timestamp()))

    sidecar_path = sessions_dir / f"issue-{issue_number}.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": issue_number,
                "branch": f"agent/issue-{issue_number}",
                "worktree_path": str(tmp_path / "worktree"),
                "prompt_path": str(tmp_path / "prompt.md"),
                "command": ["claude", "prompt.md"],
                "pid": 77777,
                "started_at": (datetime.now(UTC) - timedelta(minutes=10)).isoformat(),
                "log_path": str(log_path),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
            }
        ),
        encoding="utf-8",
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        result = app.status()

    workers = [w for w in result.data["workers"] if w["issue"] == issue_number]
    assert len(workers) == 1
    assert workers[0]["health"] == "healthy"


@pytest.mark.real_activity_probe_live
def test_status_workers_surfaces_corroboration_alive_but_polling(tmp_path: Path) -> None:
    """Issue #1346: an alive-but-polling worker (stale sidecar log mtime, fresh
    events.jsonl corroboration) must be visibly distinguishable from a genuinely
    stalled one (stale log AND stale corroboration) in the status()/roll-call
    workers section.

    Both workers share the same stale sidecar log mtime -- the only signal a
    log-mtime monitor sees -- so pre-#1346 they printed identically. After
    #1346 the workers section carries the watchdog's corroboration probe
    verdict (``last_corroborated_activity_at`` / ``last_corroborated_activity_source``
    / ``corroboration_fresh``) alongside ``health``, all derived from the same
    ``real_activity_probe_for`` + ``classify_worker_health`` code path the
    watchdog uses. The alive-but-polling worker reports ``health="healthy"`` +
    ``corroboration_fresh=True``; the stalled worker reports
    ``health="stalled"`` + ``corroboration_fresh=False``.

    Marked ``real_activity_probe_live`` so the autouse stale-probe stub fixture
    leaves ``real_activity_probe_for`` unstubbed and the real events.jsonl
    source is exercised.
    """
    from datetime import timedelta

    config = OrchestratorConfig(
        devin=DevinConfig(),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    old_time = datetime.now(UTC) - timedelta(minutes=30)
    fresh_time = datetime.now(UTC) - timedelta(minutes=1)

    def _plant_claude_worker(issue_number: int, *, fresh_corroboration: bool) -> None:
        log_path = sessions_dir / f"issue-{issue_number}.claude.log"
        log_path.write_text("Working on task...\nLast line", encoding="utf-8")
        # Stale sidecar log mtime for BOTH workers -- this is the signal that
        # log-mtime monitors cannot disambiguate.
        os.utime(log_path, (time.time(), old_time.timestamp()))

        events_path = sessions_dir / f"issue-{issue_number}.events.jsonl"
        ts = fresh_time if fresh_corroboration else old_time
        events_path.write_text(
            f'{{"type": "tool_call", "timestamp": "{ts.isoformat()}"}}\n',
            encoding="utf-8",
        )
        os.utime(events_path, (time.time(), ts.timestamp()))

        sidecar_path = sessions_dir / f"issue-{issue_number}.claude.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "issue_number": issue_number,
                    "branch": f"agent/issue-{issue_number}",
                    "worktree_path": str(tmp_path / f"worktree-{issue_number}"),
                    "prompt_path": str(tmp_path / "prompt.md"),
                    "command": ["claude", "prompt.md"],
                    "pid": 70000 + issue_number,
                    "started_at": (datetime.now(UTC) - timedelta(minutes=10)).isoformat(),
                    "log_path": str(log_path),
                    "error": None,
                    "failure_kind": None,
                    "process_start_time": 1710000000.0,
                    "reclaimed": None,
                    # last_activity_at is the sidecar's stored log mtime -- the
                    # stale signal a log-mtime monitor reads. Both workers
                    # carry the same stale value so the only disambiguator is
                    # the corroboration probe.
                    "last_activity_at": old_time.isoformat(),
                    "log_bytes": len("Working on task...\nLast line"),
                }
            ),
            encoding="utf-8",
        )

    # Worker 1346: alive-but-polling (stale log, fresh corroboration).
    _plant_claude_worker(1346, fresh_corroboration=True)
    # Worker 1347: genuinely stalled (stale log, stale corroboration).
    _plant_claude_worker(1347, fresh_corroboration=False)

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        result = app.status()

    by_issue = {w["issue"]: w for w in result.data["workers"]}
    assert set(by_issue) == {1346, 1347}

    alive_polling = by_issue[1346]
    # Acceptance criterion 1: visibly distinguishable. The stale log mtime is
    # identical to the stalled worker's, but the corroboration fields differ.
    assert alive_polling["health"] == "healthy"
    assert alive_polling["corroboration_fresh"] is True
    assert alive_polling["last_corroborated_activity_at"] is not None
    assert alive_polling["last_corroborated_activity_source"] is not None

    stalled = by_issue[1347]
    # Acceptance criterion 3: stale-both classifies stalled.
    assert stalled["health"] == "stalled"
    assert stalled["corroboration_fresh"] is False

    # The distinguishing signal: same stale last_activity_at (log mtime), but
    # the corroboration fields diverge -- exactly the gap #1346 closes.
    assert alive_polling["last_activity_at"] is not None
    assert stalled["last_activity_at"] is not None
    assert alive_polling["corroboration_fresh"] is not stalled["corroboration_fresh"]


def test_status_workers_empty_when_no_live_sessions(tmp_path: Path) -> None:
    """Issue #167: workers section should be empty list when no live sessions exist."""
    config = OrchestratorConfig(
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.status()

    # Check that workers section is present but empty
    assert "workers" in result.data
    assert isinstance(result.data["workers"], list)
    assert len(result.data["workers"]) == 0


def test_status_stalled_section_unchanged(tmp_path: Path) -> None:
    """Issue #167: stalled section keeps its base {issue, pid} shape. Issue #261
    intentionally extends each entry with a "health" field (STALLED vs DEAD) so
    digest callers can surface dead-worker terminal cause instead of collapsing
    everything to "STALLED"; terminal_tool/terminal_reason are added only for
    DEAD entries with a matching post-mortem. This test now pins that extended
    shape rather than the original byte-for-byte one.
    """
    from datetime import UTC, datetime, timedelta
    import os
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with a ready issue
    class FakeGitHubWithStalled(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 109,
                    "title": "Test issue",
                    "url": "https://example.test/issues/109",
                    "body": "Test body",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
            ]

        def issue_list(self, labels=None, state=None):
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

    fake_gh = FakeGitHubWithStalled()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a fake stalled session sidecar
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file with old mtime (stalled by time)
    log_file = sessions_dir / "issue-109.log"
    log_file.write_text("working on issue\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a sidecar with a fake PID
    sidecar = sessions_dir / "issue-109.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 109,
                "branch": "agent/issue-109",
                "worktree_path": "/fake/path",
                "prompt_path": "/fake/prompt",
                "command": ["devin", "--print"],
                "pid": 99999,
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_file),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    # Mock is_session_alive to return True for PID 99999 so detection runs
    with patch("charlie_work.worker.is_session_alive", return_value=True):
        result = app.status()

    # Check that the stalled section keeps its base shape plus the issue #261
    # "health" field. This fixture is live (mocked) with a stale log, so it
    # classifies as STALLED (not DEAD), which means no terminal_tool/
    # terminal_reason keys are added (those are DEAD-only, per
    # _detect_stalled_sessions).
    assert "stalled" in result.data
    assert isinstance(result.data["stalled"], list)
    assert any(entry["issue"] == 109 for entry in result.data["stalled"])
    assert any(entry["pid"] == 99999 for entry in result.data["stalled"])
    for entry in result.data["stalled"]:
        assert set(entry.keys()) == {"issue", "pid", "health"}
        assert entry["health"] == "STALLED"


def test_stalled_session_emits_event_with_required_fields(tmp_path: Path) -> None:
    """Issue #109: stalled session detection should emit session_stalled event with required fields."""
    from datetime import UTC, datetime, timedelta
    import os
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(
            harness="devin-shell"
        ),  # Use devin-shell adapter for watchdog support
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub
    class FakeGitHubForEvent(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []

        def issue_list(self, labels=None, state=None):
            # Support both old and new signature
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

    fake_gh = FakeGitHubForEvent()
    OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a fake stalled session sidecar
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file with old mtime (stalled by time)
    log_file = sessions_dir / "issue-109.log"
    log_file.write_text("working on issue\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a sidecar with a fake PID
    sidecar = sessions_dir / "issue-109.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 109,
                "branch": "agent/issue-109",
                "worktree_path": "/fake/path",
                "prompt_path": "/fake/prompt",
                "command": ["devin", "--print"],
                "pid": 99999,
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_file),
                "error": None,
                "process_start_time": 1234567890.0,  # Fake start time
            }
        ),
        encoding="utf-8",
    )

    # Mock read_session_records to return a fake record
    from charlie_work.devin_shell import SessionRecord

    fake_record = SessionRecord(
        issue_number=109,
        branch="agent/issue-109",
        worktree_path="/fake/path",
        prompt_path="/fake/prompt",
        command=("devin", "--print"),
        pid=99999,
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
        error=None,
        process_start_time=None,  # No start time verification in this test
    )

    # Mock is_session_alive to return True and kill_process_tree to return killed PIDs
    with (
        patch("charlie_work.devin_shell.read_session_records", return_value=[fake_record]),
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.write_gate.kill_process_tree", return_value=[99999]),
        patch(
            "charlie_work.dead_worker_reap.sweep_orphan_processes",
            return_value=[{"pid": 3492, "name": "python.exe", "command_line": "python worker.py"}],
        ),  # Fixed mock return
        patch(
            "charlie_work.devin_shell.update_session_record_with_failure_classification",
            return_value=(None, None),
        ),
    ):
        # Run the stall detection and handling
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        result = _detect_and_handle_stalled_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )

    # Check that the stalled issue was detected
    assert any(entry["issue"] == 109 for entry in result)

    # Load state and check for the event
    state = load_state(paths.state_file)
    events = state.get("events", [])

    # Find the session_stalled event
    stalled_events = [e for e in events if e.get("kind") == "session_stalled"]
    assert len(stalled_events) == 1

    event = stalled_events[0]
    # Check required fields (they're in the payload)
    payload = event.get("payload", {})
    assert payload.get("issue_number") == 109
    assert payload.get("pid") == 99999
    assert "log_mtime" in payload
    assert "last_log_line" in payload
    # killed_pids now includes both the session PID and any orphan PIDs
    # The mock returns [99999] for kill_process_tree, and sweep_orphan_processes
    # returns [3492] as a fixed mock value
    assert 99999 in payload.get("killed_pids", [])
    assert 3492 in payload.get("killed_pids", [])  # Orphan PID from mock
    # orphan_pids is included in the event payload with the exact mock value
    assert payload.get("orphan_pids") == [3492]


def test_sweep_orphan_processes_for_dead_sessions_unit(tmp_path: Path) -> None:
    """Unit test for _sweep_orphan_processes_for_dead_sessions (issue #139)."""
    from datetime import UTC, datetime
    from unittest.mock import patch, MagicMock
    import subprocess

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake session records
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.claude_code import ClaudeWorkerRecord

    dead_session = SessionRecord(
        issue_number=100,
        branch="agent/issue-100",
        worktree_path="/dead/worktree",
        prompt_path="/fake/prompt",
        command=("devin", "--print"),
        pid=1000,
        started_at=datetime.now(UTC).isoformat(),
        log_path="/fake/log",
        error=None,
        process_start_time=1234567890.0,
    )

    live_session = SessionRecord(
        issue_number=101,
        branch="agent/issue-101",
        worktree_path="/live/worktree",
        prompt_path="/fake/prompt",
        command=("devin", "--print"),
        pid=1001,
        started_at=datetime.now(UTC).isoformat(),
        log_path="/fake/log",
        error=None,
        process_start_time=1234567890.0,
    )

    dead_worker = ClaudeWorkerRecord(
        issue_number=102,
        branch="agent/issue-102",
        worktree_path="/dead/worker",
        prompt_path="/fake/prompt",
        command=("devin", "--print"),
        pid=1002,
        started_at=datetime.now(UTC).isoformat(),
        log_path="/fake/log",
        error=None,
        process_start_time=1234567890.0,
    )

    # Mock sweep_orphan_processes to return fixed orphan process details for dead worktrees
    def mock_sweep_orphan(worktree_path: str) -> list[dict[str, Any]]:
        if worktree_path == "/dead/worktree":
            return [
                {
                    "pid": 5000,
                    "name": "python.exe",
                    "command_line": "python script.py /dead/worktree",
                },
                {"pid": 5001, "name": "node.exe", "command_line": "node server.js /dead/worktree"},
            ]
        elif worktree_path == "/dead/worker":
            return [
                {
                    "pid": 6000,
                    "name": "python.exe",
                    "command_line": "python worker.py /dead/worker",
                },
            ]
        return []

    # Mock subprocess.run to track taskkill calls
    taskkill_calls = []
    original_run = subprocess.run

    def mock_subprocess_run(*args, **kwargs):
        if args and args[0] and args[0][0] == "taskkill":
            taskkill_calls.append(args[0])
            # Return a successful result
            return MagicMock(returncode=0, stdout="", stderr="")
        return original_run(*args, **kwargs)

    with (
        patch(
            "charlie_work.devin_shell.read_session_records",
            return_value=[dead_session, live_session],
        ),
        patch("charlie_work.claude_code.read_worker_records", return_value=[dead_worker]),
        patch("charlie_work.devin_shell.is_session_alive", side_effect=lambda r: r.pid != 1000),
        patch("charlie_work.claude_code.is_worker_alive", side_effect=lambda r: r.pid != 1002),
        patch(
            "charlie_work.dead_worker_reap.sweep_orphan_processes", side_effect=mock_sweep_orphan
        ),
        patch("os.name", "nt"),  # Force Windows path (os.name check lives in dead_worker_reap;
        # patching the os module directly avoids depending on which module happens to
        # `import os` into its own namespace)
        patch("subprocess.run", side_effect=mock_subprocess_run),
    ):
        from charlie_work.workflow import _sweep_orphan_processes_for_dead_sessions

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _sweep_orphan_processes_for_dead_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )

    # Verify taskkill was called for the orphan PIDs
    assert len(taskkill_calls) == 3
    killed_pids = [
        int(call[3]) for call in taskkill_calls
    ]  # Extract PID from taskkill /F /PID <pid>
    assert 5000 in killed_pids
    assert 5001 in killed_pids
    assert 6000 in killed_pids

    # Verify the event was logged
    state = load_state(paths.state_file)
    events = state.get("events", [])
    orphan_events = [e for e in events if e.get("kind") == "orphan_processes_killed"]
    assert len(orphan_events) == 2

    # Check the first event (dead/worktree)
    event1 = next(e for e in orphan_events if e["payload"]["worktree_path"] == "/dead/worktree")
    assert event1["payload"]["orphan_pids"] == [5000, 5001]
    assert event1["payload"]["killed_orphans"] == [5000, 5001]

    # Check the second event (dead/worker)
    event2 = next(e for e in orphan_events if e["payload"]["worktree_path"] == "/dead/worker")
    assert event2["payload"]["orphan_pids"] == [6000]
    assert event2["payload"]["killed_orphans"] == [6000]


def test_sweep_orphan_processes_called_from_production_loop(tmp_path: Path) -> None:
    """Integration test: verify _sweep_orphan_processes_for_dead_sessions is called from production loop (issue #139)."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub
    class FakeGitHubForSweep(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []

        def issue_list(self, labels=None, state=None):
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

        def pr_list(self):
            return []

    fake_gh = FakeGitHubForSweep()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Mock _sweep_orphan_processes_for_dead_sessions to track if it's called
    sweep_called = []

    def mock_sweep(*args, **kwargs):
        sweep_called.append(True)
        # Don't actually do anything

    with patch(
        "charlie_work.workflow._sweep_orphan_processes_for_dead_sessions", side_effect=mock_sweep
    ):
        # Run the production loop (loop calls the sweep)
        app.loop(limit=1)

    # Verify the sweep was called from the production loop
    assert len(sweep_called) == 1, (
        "Expected _sweep_orphan_processes_for_dead_sessions to be called from production loop"
    )


def test_watchdog_disabled_no_detection_no_kill_no_event(tmp_path: Path) -> None:
    """Issue #109: when watchdog.enabled=False, no detection, no kill, no event."""
    from datetime import UTC, datetime, timedelta
    import os
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        watchdog=WatchdogConfig(enabled=False, stall_minutes=20),  # Disabled
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub
    class FakeGitHubDisabled(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []

        def issue_list(self, labels=None, state=None):
            # Support both old and new signature
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

    fake_gh = FakeGitHubDisabled()
    OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a fake stalled session sidecar
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file with old mtime (stalled by time)
    log_file = sessions_dir / "issue-109.log"
    log_file.write_text("working on issue\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a sidecar with a fake PID
    sidecar = sessions_dir / "issue-109.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 109,
                "branch": "agent/issue-109",
                "worktree_path": "/fake/path",
                "prompt_path": "/fake/prompt",
                "command": ["devin", "--print"],
                "pid": 99999,
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_file),
                "error": None,
                "process_start_time": 1234567890.0,  # Fake start time
            }
        ),
        encoding="utf-8",
    )

    # Mock is_session_alive and kill_process_tree to track calls
    with (
        patch("charlie_work.worker.is_session_alive", return_value=True) as mock_alive,
        patch("charlie_work.process_utils.kill_process_tree", return_value=[]) as mock_kill,
    ):
        # Run the stall detection and handling
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        _detect_and_handle_stalled_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )

    # Check that is_session_alive was NOT called (detection skipped)
    mock_alive.assert_not_called()

    # Check that kill_process_tree was NOT called (no kill)
    mock_kill.assert_not_called()

    # Load state and check for the event
    state = load_state(paths.state_file)
    events = state.get("events", [])

    # No reap event of either kind may be emitted while the watchdog is off.
    #
    # This filter previously read e.get("type"), but events are keyed on
    # "kind" — so it matched nothing regardless of what was emitted and the
    # assertion below was true by construction. Found while splitting the
    # reap kinds for #873; fixed here rather than left as a silent no-op.
    # Both kinds are checked because #873 split the single "session_stalled"
    # into session_stalled (STALLED) + session_exited (DEAD), and a
    # disabled watchdog must emit neither.
    reap_events = [e for e in events if e.get("kind") in {"session_stalled", "session_exited"}]
    assert reap_events == []


def test_layered_config_logs_whether_global_layer_was_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The provenance line must report *readership*, not resolved values.

    A section that is missing from the global file and a global file that was
    never read produce byte-identical resolved config -- pristine dataclass
    defaults -- but demand opposite fixes. Issue #590 stalled on exactly that
    ambiguity, so this asserts the one fact that separates them survives in the
    log even when the resulting config looks entirely ordinary.
    """
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # Absent global layer: resolved config is all defaults, and the log says why.
    with caplog.at_level(logging.DEBUG, logger="charlie_work.global_config"):
        load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))
    absent_line = "\n".join(
        r.getMessage() for r in caplog.records if "Layered config" in r.getMessage()
    )
    assert "absent" in absent_line, f"absent global layer not reported: {absent_line!r}"
    assert str(fleet_dir_path / "config.yaml") in absent_line, "the path itself must be logged"

    # Present global layer: same call, and the log distinguishes it.
    caplog.clear()
    (fleet_dir_path / "config.yaml").write_text(
        "dispatch:\n  max_concurrent_sessions: 5\n", encoding="utf-8"
    )
    with caplog.at_level(logging.DEBUG, logger="charlie_work.global_config"):
        load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))
    present_line = "\n".join(
        r.getMessage() for r in caplog.records if "Layered config" in r.getMessage()
    )
    assert "present" in present_line, f"present global layer not reported: {present_line!r}"
    assert "absent" not in present_line, "a present layer must not read as absent"
    assert "dispatch" in present_line, "the sections actually read must be named"
    assert "bytes=" in present_line, "same vocabulary as the supervisor's INFO line"

    # An empty global file resolves to pristine defaults with zero sections --
    # indistinguishable from an absent one on the resolved config alone, and the
    # reason this line carries a byte count rather than a bare "present".
    caplog.clear()
    (fleet_dir_path / "config.yaml").write_text("", encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger="charlie_work.global_config"):
        load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))
    empty_line = "\n".join(
        r.getMessage() for r in caplog.records if "Layered config" in r.getMessage()
    )
    assert "bytes=0" in empty_line, f"a truncated global layer must be visible: {empty_line!r}"
    assert "absent" not in empty_line, "an empty file is present-but-empty, not absent"
    assert "(none)" in empty_line, "an empty file contributes no sections"


def test_describe_config_file_separates_absent_from_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config that could not be reached must not read as one that is absent.

    ``Path.exists()`` returns a bare False for every error in
    ``pathlib._ignore_error`` -- ENOENT, ENOTDIR, EBADF, ELOOP and the Windows
    unready-device / unresolvable-path winerrors. All of them take
    load_layered_config's silent-``{}`` branch and yield pristine dataclass
    defaults with no error raised, which is #590's symptom exactly. The cause
    must survive into the log instead of collapsing into "absent".

    ENOTDIR is used rather than EACCES deliberately: permission errors are *not*
    in the ignored set, so they raise instead of silently defaulting, which is
    why they cannot be #590's mechanism.
    """
    from charlie_work.global_config import describe_config_file

    missing = tmp_path / "nope.yaml"
    assert describe_config_file(missing) == "absent"

    present = tmp_path / "config.yaml"
    present.write_text("dispatch: {}\n", encoding="utf-8")
    assert describe_config_file(present) == f"present bytes={present.stat().st_size}"

    real_stat = Path.stat

    def not_a_dir(self: Path, *args: object, **kwargs: object) -> object:
        if self == present:
            raise NotADirectoryError(errno.ENOTDIR, "Not a directory")
        return real_stat(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", not_a_dir)

    # The precondition that makes this helper necessary: exists() hides this.
    assert present.exists() is False, "precondition: exists() collapses ENOTDIR to False"

    described = describe_config_file(present)
    assert described != "absent", "an unreachable config must not read as absent"
    assert described.startswith("UNREADABLE"), f"cause was lost: {described!r}"
    assert "NotADirectoryError" in described, "the failure cause must reach the log"


def test_cli_build_app_registers_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration test: cli.build_app registers repo in fleet.json."""
    from charlie_work.cli import build_app
    from charlie_work.github import GitHub

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir()  # Make it a git repo

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

        def validate_field_lists(self) -> None:
            # build_app is an integration test for fleet.json registration; the
            # gh --json field-list probe needs no real GitHub CLI here.
            pass

    # Monkeypatch GitHub to use our fake
    def fake_github(
        repo_root: Path, dry_run: bool = False, runtime: object | None = None
    ) -> GitHub:
        return FakeGitHub(repo_root=repo_root, dry_run=dry_run)

    monkeypatch.setattr("charlie_work.cli.GitHub", fake_github)

    # Redirect fleet_dir resolution to tmp_path via the env var fleet_paths.fleet_dir()
    # itself supports. Patching the module-level name directly no longer works since
    # fleet_registry composes fleet paths through layout.py, which binds its own
    # reference to fleet_paths.fleet_dir at import time.
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))

    # Build args
    import argparse

    args = argparse.Namespace(repo=repo_root, config=None, dry_run=False, fleet_dir=None)

    # Call build_app
    build_app(args)

    # Verify fleet.json was created
    fleet_json_path = tmp_path / "fleet" / "fleet.json"
    assert fleet_json_path.exists()

    # Verify registry entry
    import json

    registry = json.loads(fleet_json_path.read_text(encoding="utf-8"))
    assert "owner/repo" in registry["repos"]
    entry = registry["repos"]["owner/repo"]
    assert entry["repo_root"] == str(repo_root)
    assert entry["name_with_owner"] == "owner/repo"


def test_build_parser_fleet_subcommand() -> None:
    """Test that build_parser registers the fleet subcommand correctly."""
    parser = cli.build_parser()

    # Test fleet status parsing
    args = parser.parse_args(["fleet", "status"])
    assert args.command == "fleet"
    assert args.fleet_command == "status"

    # Test that existing subcommands still work
    args_roll_call = parser.parse_args(["roll-call"])
    assert args_roll_call.command == "roll-call"

    parser.parse_args(["doctor"])

    # Test fleet review-queue parsing
    args_review_queue = parser.parse_args(["fleet", "review-queue"])
    assert args_review_queue.command == "fleet"
    assert args_review_queue.fleet_command == "review-queue"

    # Test single-repo review-queue parsing
    args_single = parser.parse_args(["review-queue"])
    assert args_single.command == "review-queue"

    # Test fleet operator-queue parsing (issue #1314 item 1)
    args_fleet_operator_queue = parser.parse_args(["fleet", "operator-queue"])
    assert args_fleet_operator_queue.command == "fleet"
    assert args_fleet_operator_queue.fleet_command == "operator-queue"

    # Test single-repo operator-queue parsing (issue #1314 item 1)
    args_operator_queue = parser.parse_args(["operator-queue"])
    assert args_operator_queue.command == "operator-queue"


def test_run_command_dispatches_operator_queue(tmp_path: Path) -> None:
    """Issue #1314 item 1: ``run_command`` routes ``operator-queue`` to
    ``app.operator_queue()`` and returns its result verbatim.

    Mirrors the established convention that the CLI dispatch branch for a
    queue command is exercised end-to-end through ``run_command`` rather than
    only through the ``OrchestratorApp`` method in isolation — a broken
    dispatch branch (wrong ``args.command`` string, missing branch) would
    otherwise go undetected by the method-level tests.
    """
    args = cli.build_parser().parse_args(["operator-queue"])
    assert args.command == "operator-queue"

    dispatched: list[str] = []
    expected = cli.CommandResult(
        True, "operator queue: 0 issue(s) parked", {"queue": [], "depth": 0}
    )

    class _FakeApp:
        def operator_queue(self) -> cli.CommandResult:
            dispatched.append("operator_queue")
            return expected

    result = cli.run_command(_FakeApp(), args)  # type: ignore[arg-type]

    assert dispatched == ["operator_queue"]
    assert result is expected


def test_loop_reaps_stalled_session_with_no_candidates(tmp_path: Path) -> None:
    """Test that loop() reaps stalled sessions even with zero ready/rework candidates (issue #165)."""
    from datetime import UTC, datetime, timedelta
    from charlie_work.devin_shell import SessionRecord

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Issue #1325: dry_run=True suppresses sidecar writes (failure_kind stays
    # None), so this test must use dry_run=False to verify the sidecar is
    # actually classified. The stalled PID (99999) does not exist, so
    # kill_process_tree is a no-op — no real process is harmed.
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=False)

    # Create a session record for issue 123 with a live PID and stale log
    sessions_dir = app._layout.sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_file = sessions_dir / "issue-123.json"
    log_file = sessions_dir / "issue-123.log"

    # Write a log file with old mtime (stalled)
    log_file.write_text("working on issue\nmaking progress\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a session record with a fake PID
    session_record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(
            tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
        ),
        command=("devin", "--prompt-file", "{prompt_path}"),
        pid=99999,
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
        process_start_time=time.time(),
    )
    session_file.write_text(json.dumps(session_record.to_dict()), encoding="utf-8")

    # Ensure zero ready issues and zero rework candidates
    fake_gh.issues = []
    fake_gh.prs = []

    # Mock the liveness check to return True (simulating a live but stalled process)
    with patch("charlie_work.worker.is_session_alive", return_value=True):
        result = app.loop()

    # The loop should complete and the stalled session should be reaped
    assert result.ok is True
    # Verify the session file was marked with failure_kind: stalled
    updated_session = json.loads(session_file.read_text(encoding="utf-8"))
    assert updated_session.get("failure_kind") == "stalled"


def test_loop_advances_inconclusive_probe_deferral_counter_once_per_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One loop() pass advances the Signal-1 deferral counter at most once per worker.

    Regression test for the issue #343 Finding 2 follow-up: loop() runs the
    stall lane (_detect_and_handle_stalled_sessions) itself, and dispatch()/
    dispatch_rework() each used to re-run it internally — three sweeps per
    pass, each independently incrementing inconclusive_probe_deferred_count
    for a not-alive worker with an inconclusive real-activity probe. That
    collapsed max_inconclusive_probe_deferrals' "N passes of grace" into a
    single pass. loop() now hands its sweep result down to both dispatch
    lanes so the counter is written exactly once per pass.

    The dead-session lane is neutralized here: pre-PR-#352 it reaps any
    not-alive worker outright (deleting the sidecar mid-pass), and post-#352
    it defers with its own counter suppression — either way it is covered by
    its own tests, and this test pins the stall-lane/dispatch-lane interplay
    in isolation so it holds on both sides of that merge.
    """
    from datetime import UTC, datetime
    from charlie_work import workflow as workflow_module
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(
            enabled=True, stall_minutes=20, max_inconclusive_probe_deferrals=10
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = []
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=False)

    sessions_dir = app._layout.sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_file = sessions_dir / "issue-343.json"
    log_file = sessions_dir / "issue-343.log"
    # Fresh log so Signal 3 (progress staleness) never fires; the counter is
    # driven purely by Signal 1 (not alive) + inconclusive probe.
    log_file.write_text("working on issue\n", encoding="utf-8")

    session_record = SessionRecord(
        issue_number=343,
        branch="agent/issue-343-fix",
        worktree_path=str(tmp_path / "worktrees" / "agent-343"),
        prompt_path=str(
            tmp_path / ".var" / "charlie-work" / "issues" / "issue-343" / "worker-prompt.md"
        ),
        command=("devin", "--prompt-file", "{prompt_path}"),
        pid=99999,
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
        process_start_time=time.time(),
    )
    session_file.write_text(json.dumps(session_record.to_dict()), encoding="utf-8")

    def _inconclusive_probe(view: Any, cfg: Any, now: Any) -> RealActivityProbe:
        return RealActivityProbe(
            sources=(
                ActivitySource(
                    name="sessions.db",
                    timestamp=None,
                    staleness_seconds=None,
                    error="message_nodes query failed",
                ),
            )
        )

    # Worker process is gone; the probe cannot corroborate either way.
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: False)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _inconclusive_probe)
    # Neutralize the sibling lanes (see docstring) so only the stall-lane
    # sweeps driven by loop()/dispatch()/dispatch_rework() touch the sidecar.
    monkeypatch.setattr(
        workflow_module,
        "_classify_dead_sessions_and_update_throttle_state",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        workflow_module,
        "_sweep_orphan_processes_for_dead_sessions",
        lambda *args, **kwargs: None,
    )

    result = app.loop(limit=0)
    assert result.ok is True

    sidecar = json.loads(session_file.read_text(encoding="utf-8"))
    assert sidecar.get("inconclusive_probe_deferred_count") == 1
    assert sidecar.get("failure_kind") is None

    # Cross-pass accumulation still works: a second pass advances it once more.
    result = app.loop(limit=0)
    assert result.ok is True
    sidecar = json.loads(session_file.read_text(encoding="utf-8"))
    assert sidecar.get("inconclusive_probe_deferred_count") == 2
    assert sidecar.get("failure_kind") is None


def test_standalone_dispatch_and_rework_advance_inconclusive_probe_counter_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dispatch() and dispatch_rework() called standalone still run the stall lane once.

    Issue #356: loop() hands a pre-computed ``stalled_entries`` result to
    ``dispatch()`` and ``dispatch_rework()`` so the stall-lane sweep runs exactly
    once per ``loop()`` pass. Standalone callers (CLI ``work``, fleet
    ``work_only``) do not receive that result, so each must still run the sweep
    internally and advance a not-alive, inconclusive-probe worker's
    ``inconclusive_probe_deferred_count`` exactly once per call.
    """
    from datetime import UTC, datetime
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(
            enabled=True, stall_minutes=20, max_inconclusive_probe_deferrals=10
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = []
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=False)

    sessions_dir = app._layout.sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_file = sessions_dir / "issue-356.json"
    log_file = sessions_dir / "issue-356.log"
    log_file.write_text("working on issue\n", encoding="utf-8")

    session_record = SessionRecord(
        issue_number=356,
        branch="agent/issue-356-fix",
        worktree_path=str(tmp_path / "worktrees" / "agent-356"),
        prompt_path=str(
            tmp_path / ".var" / "charlie-work" / "issues" / "issue-356" / "worker-prompt.md"
        ),
        command=("devin", "--prompt-file", "{prompt_path}"),
        pid=99999,
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
        process_start_time=time.time(),
    )
    session_file.write_text(json.dumps(session_record.to_dict()), encoding="utf-8")

    def _inconclusive_probe(_view: Any, _cfg: Any, _now: Any) -> RealActivityProbe:
        return RealActivityProbe(
            sources=(
                ActivitySource(
                    name="sessions.db",
                    timestamp=None,
                    staleness_seconds=None,
                    error="message_nodes query failed",
                ),
            )
        )

    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda _record: False)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _inconclusive_probe)

    result = app.dispatch(limit=0)
    assert result.ok is True
    sidecar = json.loads(session_file.read_text(encoding="utf-8"))
    assert sidecar.get("inconclusive_probe_deferred_count") == 1
    assert sidecar.get("failure_kind") is None

    result = app.dispatch_rework(limit=0)
    assert result.ok is True
    sidecar = json.loads(session_file.read_text(encoding="utf-8"))
    assert sidecar.get("inconclusive_probe_deferred_count") == 2
    assert sidecar.get("failure_kind") is None

    # The reaper call should be before the governor call (unconditional vs gated)
    # This ensures it runs even when max_concurrent_sessions=0


def test_watchdog_config_additive_redispatch_fields(tmp_path: Path) -> None:
    """Test that WatchdogConfig loads with defaults when new fields are missing (issue #165)."""
    # Create a config file without the new fields
    config_path = tmp_path / "orchestrator.yaml"
    config_content = """
watchdog:
  enabled: true
  stall_minutes: 20
"""
    config_path.write_text(config_content, encoding="utf-8")

    # Load the config - should not raise ConfigError
    config = load_config(config_path)

    # Verify defaults are applied
    assert config.watchdog.redispatch_window_minutes == 240
    assert config.watchdog.max_auto_redispatch == 3


def test_redispatch_escalated_edge_clears_full_active_set(tmp_path: Path) -> None:
    """Test that redispatch_escalated edge clears all active labels (issue #165)."""
    from charlie_work.labels import _edges

    config = OrchestratorConfig()
    edges = _edges(config.labels)

    # Verify the edge exists
    assert "redispatch_escalated" in edges

    add, remove = edges["redispatch_escalated"]

    # Should add human_needed
    assert config.labels.human_needed in add

    # Should remove ALL other workflow labels (issue #215: terminal transitions clear siblings)
    assert set(remove) == config.labels.workflow_labels - {config.labels.human_needed}


def test_redispatch_within_window_does_not_escalate(tmp_path: Path) -> None:
    """Test that N-1 redispatches within the window does not escalate (issue #165)."""
    from datetime import UTC, datetime, timedelta

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            enabled=True,
            stall_minutes=20,
            redispatch_window_minutes=240,
            max_auto_redispatch=3,
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Setup state with 2 redispatches (within cap of 3)
    state = load_state(paths.state_file)
    now = datetime.now(UTC)
    state["issues"]["123"] = {
        "number": 123,
        "title": "Test issue",
        "url": "https://github.com/test/repo/issues/123",
        "status": "rework_requested",
        "redispatch_at": [
            (now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
            (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
        ],
    }
    save_state(paths.state_file, state)

    # Test the counting logic directly
    entry = state["issues"]["123"]
    window_start = now - timedelta(minutes=config.watchdog.redispatch_window_minutes)
    prior = [
        t
        for t in entry.get("redispatch_at", [])
        if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start
    ]
    redispatch_at = prior + [now.isoformat().replace("+00:00", "Z")]

    # Should not escalate - only 2 redispatches, cap is 3
    assert len(redispatch_at) == 3  # 2 prior + 1 new
    assert len(redispatch_at) <= config.watchdog.max_auto_redispatch


def test_redispatch_exceeding_cap_escalates(tmp_path: Path) -> None:
    """Test that exceeding max_auto_redispatch triggers escalation (issue #165)."""
    from datetime import UTC, datetime, timedelta

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            enabled=True,
            stall_minutes=20,
            redispatch_window_minutes=240,
            max_auto_redispatch=3,
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Setup state with 3 redispatches (at cap of 3)
    state = load_state(paths.state_file)
    now = datetime.now(UTC)
    state["issues"]["123"] = {
        "number": 123,
        "title": "Test issue",
        "url": "https://github.com/test/repo/issues/123",
        "status": "rework_requested",
        "redispatch_at": [
            (now - timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
            (now - timedelta(minutes=20)).isoformat().replace("+00:00", "Z"),
            (now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        ],
    }
    save_state(paths.state_file, state)

    # Test the counting logic directly
    entry = state["issues"]["123"]
    window_start = now - timedelta(minutes=config.watchdog.redispatch_window_minutes)
    prior = [
        t
        for t in entry.get("redispatch_at", [])
        if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start
    ]
    redispatch_at = prior + [now.isoformat().replace("+00:00", "Z")]

    # Should escalate - 4th redispatch exceeds cap of 3
    assert len(redispatch_at) == 4  # 3 prior + 1 new
    assert len(redispatch_at) > config.watchdog.max_auto_redispatch


def test_redispatch_timestamps_pruned_outside_window(tmp_path: Path) -> None:
    """Test that timestamps outside the window are pruned before counting (issue #165)."""
    from datetime import UTC, datetime, timedelta

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            enabled=True,
            stall_minutes=20,
            redispatch_window_minutes=240,
            max_auto_redispatch=3,
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Setup state with old redispatches outside the window
    state = load_state(paths.state_file)
    now = datetime.now(UTC)
    state["issues"]["123"] = {
        "number": 123,
        "title": "Test issue",
        "url": "https://github.com/test/repo/issues/123",
        "status": "rework_requested",
        "redispatch_at": [
            (now - timedelta(minutes=300)).isoformat().replace("+00:00", "Z"),  # Outside window
            (now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),  # Inside window
        ],
    }
    save_state(paths.state_file, state)

    # Test the counting logic directly
    entry = state["issues"]["123"]
    window_start = now - timedelta(minutes=config.watchdog.redispatch_window_minutes)
    prior = [
        t
        for t in entry.get("redispatch_at", [])
        if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start
    ]
    redispatch_at = prior + [now.isoformat().replace("+00:00", "Z")]

    # Old timestamp should be pruned, only 2 remain (1 in window + 1 new)
    assert len(redispatch_at) == 2
    assert len(redispatch_at) <= config.watchdog.max_auto_redispatch


def test_redispatch_at_only_written_by_known_call_sites(tmp_path: Path) -> None:
    """Test that redispatch_at is only written by known call sites."""
    # This test verifies by code inspection that redispatch_at is only written in:
    # 1. dispatch_rework normal paths (success + no-op rework pre-dispatch) --
    #    workflow.py, OrchestratorApp.dispatch_rework.
    # 2. _classify_dead_sessions_and_update_throttle_state normal paths --
    #    dead_worker_reap.py (issue #1317: moved verbatim from workflow.py).
    # 3. _reap_restore_rework_requested (issue #315 review finding 2) --
    #    dead_worker_reap.py (issue #1317: moved verbatim from workflow.py).
    # Escalated paths now consolidate on _escalate_issue and pass redispatch_at
    # through issue_extra, so direct entry["redispatch_at"] assignments only
    # remain in the non-escalated branches below.
    #
    # issue #1283 Phase A hazard (not in the recon's original list -- found by
    # the full-suite run for this split): AST-based real-assignment count, not
    # a raw substring count. `_windowed_redispatch_at`'s own docstring quotes
    # ``entry["redispatch_at"]`` in prose ("Normalizes ``entry["redispatch_at"]``
    # to a list of strings..."), and `.count()` over raw source text counted
    # that quote as a 5th "call site" for as long as the docstring lived in
    # workflow.py (confirmed: pre-split workflow.py already had exactly 4 real
    # assignments + this 1 docstring quote = 5 -- the miscounting predates
    # this extraction). Moving the function (and its docstring) to
    # dispatch_selection.py dropped the naive count to 4 with zero change to
    # any real write site -- a false regression signal, the opposite failure
    # mode from AC7's hazard test but the same root cause (name/text search
    # over source that doesn't distinguish code from prose). All three files
    # are scanned and summed via real ast.Assign nodes so neither a docstring
    # quote nor a future extraction of one of the three named call sites can
    # produce a false pass or a false failure here.
    #
    # issue #1317: the dead-worker/session-reap extraction moved call sites 2
    # and 3 above out of workflow.py into dead_worker_reap.py verbatim -- the
    # writers still exist, only their module changed (confirmed real split:
    # workflow.py=2, dispatch_selection.py=0, dead_worker_reap.py=2, total
    # unchanged at 4). dead_worker_reap.py is added to the scan so this guard
    # keeps failing if a genuinely NEW, unknown writer appears in any of the
    # three files, rather than going blind to two known call sites because
    # they changed address.
    #
    # issue #1645 (L01 batch 2): the OrchestratorApp delegation extraction moved
    # call site 1 above (dispatch_rework normal paths) out of workflow.py into
    # orchestration/state_dispatch_rework.py verbatim (_dispatch_rework_impl) --
    # the two writers still exist, only their module changed (confirmed real
    # split: workflow.py=0, dispatch_selection.py=0, dead_worker_reap.py=2,
    # state_dispatch_rework.py=2, total unchanged at 4). state_dispatch_rework.py
    # is added to the scan for the same reason dead_worker_reap.py was: keep
    # failing on a genuinely NEW writer rather than going blind to two known
    # call sites because they changed address.
    import ast

    workflow_path = Path(__file__).parents[1] / "src" / "charlie_work" / "workflow.py"
    dispatch_selection_path = (
        Path(__file__).parents[1] / "src" / "charlie_work" / "dispatch_selection.py"
    )
    dead_worker_reap_path = (
        Path(__file__).parents[1] / "src" / "charlie_work" / "dead_worker_reap.py"
    )
    state_dispatch_rework_path = (
        Path(__file__).parents[1]
        / "src"
        / "charlie_work"
        / "orchestration"
        / "state_dispatch_rework.py"
    )

    def _count_redispatch_at_assignments(path: Path) -> int:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "entry"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "redispatch_at"
                ):
                    count += 1
        return count

    # Any unexpected increase means a new call site is writing redispatch_at.
    redispatch_assignments = (
        _count_redispatch_at_assignments(workflow_path)
        + _count_redispatch_at_assignments(dispatch_selection_path)
        + _count_redispatch_at_assignments(dead_worker_reap_path)
        + _count_redispatch_at_assignments(state_dispatch_rework_path)
    )
    assert redispatch_assignments == 4, (
        'Expected 4 real entry["redispatch_at"] assignment statements across '
        "workflow.py, dispatch_selection.py, dead_worker_reap.py, and "
        "orchestration/state_dispatch_rework.py, found "
        f"{redispatch_assignments}"
    )


def test_dead_dispatched_worker_reaped_after_grace_period(tmp_path: Path) -> None:
    """Issue #654: a dead dispatched worker whose drift was already surfaced on
    a prior pass (``orphan_drift_at`` set) but whose PR state did not qualify
    for auto-reset (clean exit with no push -- the #773 no-op branch) must be
    escalated to ``agent:human-needed`` after ``dead_dispatched_reap_minutes``,
    not held in ``dispatched`` indefinitely.  This is the exact scenario from
    job-cannon #1408: the rework worker made 5 local commits, exited 0 without
    pushing, and the dispatch label held for 1+ hour because the #773 branch
    surfaces drift but never resets status or clears the label.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, dead_dispatched_reap_minutes=60),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Simulate the "second pass" state: the first pass already emitted drift
    # (dead_worker_clean_exit_no_op) and set orphan_drift_at.  The drift
    # fingerprint matches the #773 clean-exit branch so the specific sub-branch
    # would short-circuit to ``continue`` without this fix.
    old_drift_at = (datetime.now(UTC) - timedelta(minutes=120)).isoformat().replace("+00:00", "Z")
    fingerprint = json.dumps(
        {"reason": "dead_worker_clean_exit_no_op", "reviewed_head_sha": "abc123"},
        sort_keys=True,
        default=str,
    )
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "orphan_drift_at": old_drift_at,
        "orphan_drift_fingerprint": fingerprint,
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    # Issue #1229: the branch-issue validator threaded through
    # _detect_and_handle_orphaned_workers calls issue_list(state="open") and
    # rejects branch-name issue numbers absent from the open-issue set. The
    # default FakeGitHub.issues only carries #123, so #207 must be planted
    # here or the validator rejects the agent/issue-207 binding and the orphan
    # sweep cannot match the PR to the issue (pr_number resolves to None,
    # orphan_drift_at gets overwritten instead of preserved).
    fake_gh.issues.append(
        {"number": 207, "title": "test issue 207", "state": "OPEN", "labels": [], "body": ""}
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = sessions_dir / "issue-207.claude.terminal.json"
    terminal_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 0,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:05:00Z",
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # The time-based escape must have escalated the issue.
    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "dead_dispatched_worker_reap"
    assert entry.get("reason_class") == "mechanical"
    assert entry.get("dispatched_at") is None
    # The drift fingerprint/at are cleared so a de-escalated issue does not
    # immediately re-trigger the drift path.
    assert entry.get("orphan_drift_at") is None
    assert entry.get("orphan_drift_fingerprint") is None
    # Issue #282: the liveness fingerprint is preserved.
    assert entry["worker_pid"] == 99999

    # The ``escalated`` label edge must have been applied via transition().
    # Issue #1266: dead_dispatched_worker_reap is mechanical, so it lands
    # agent:operator-queue, not agent:human-needed.
    assert (207, config.labels.operator_queue) in fake_gh.labels_added
    assert (207, config.labels.in_progress) in fake_gh.labels_removed

    # A dedicated reap event must be recorded.
    reaped_events = [
        e for e in state.get("events", []) if e.get("kind") == "dead_dispatched_worker_reaped"
    ]
    assert len(reaped_events) == 1
    payload = reaped_events[0]["payload"]
    assert payload["issue_number"] == 207
    assert payload["pr_number"] == 100
    assert payload["previous_status"] == "dispatched"
    assert payload["reason"] == "dead_dispatched_worker_reap"
    assert payload["reap_minutes"] == 60
    assert payload["exit_code"] == 0


def test_dead_dispatched_worker_not_reaped_within_grace_period(tmp_path: Path) -> None:
    """Issue #654: a dead dispatched worker whose drift was surfaced recently
    (within ``dead_dispatched_reap_minutes``) must NOT be time-escalated.  The
    existing drift-only behavior (fingerprint match short-circuits to
    ``continue``) is preserved so a freshly-dead worker is not prematurely
    escalated before its specific sub-branch has had a chance to act.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, dead_dispatched_reap_minutes=60),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    recent_drift_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    fingerprint = json.dumps(
        {"reason": "dead_worker_clean_exit_no_op", "reviewed_head_sha": "abc123"},
        sort_keys=True,
        default=str,
    )
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "orphan_drift_at": recent_drift_at,
        "orphan_drift_fingerprint": fingerprint,
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)

    # Issue #1362 Stage 1: the last-review-decision read in
    # _detect_and_handle_orphaned_workers is now file-first, so the live
    # request_changes decision must exist on disk, not only in state.json,
    # or it resolves to "missing" and misses the clean-exit-no-op fingerprint
    # short-circuit this test is exercising.
    pr_decision_dir = paths.prs / "pr-100"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "abc123"}),
        encoding="utf-8",
    )

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    # Issue #1229: the branch-issue validator threaded through
    # _detect_and_handle_orphaned_workers calls issue_list(state="open") and
    # rejects branch-name issue numbers absent from the open-issue set. The
    # default FakeGitHub.issues only carries #123, so #207 must be planted
    # here or the validator rejects the agent/issue-207 binding, the orphan
    # sweep cannot match the PR to the issue, and the fingerprint short-circuit
    # (which requires the PR to be found) never fires — orphan_drift_at gets
    # overwritten with a fresh timestamp instead of being preserved.
    fake_gh.issues.append(
        {"number": 207, "title": "test issue 207", "state": "OPEN", "labels": [], "body": ""}
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = sessions_dir / "issue-207.claude.terminal.json"
    terminal_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 0,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:05:00Z",
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # Within the grace period: the existing drift-only behavior is preserved.
    # The fingerprint match short-circuits to ``continue`` without escalating.
    assert entry.get("status") == "dispatched"
    assert entry.get("escalation_reason") is None
    assert entry.get("orphan_drift_at") == recent_drift_at

    # No reap event, no label transition.
    reaped_events = [
        e for e in state.get("events", []) if e.get("kind") == "dead_dispatched_worker_reaped"
    ]
    assert reaped_events == []
    assert (207, config.labels.human_needed) not in fake_gh.labels_added


def test_dead_dispatched_worker_reap_disabled_by_config(tmp_path: Path) -> None:
    """Issue #654: ``dead_dispatched_reap_minutes=0`` disables the time-based
    escape, reverting to the pre-#654 hold-forever behavior.  A dead dispatched
    worker with old drift stays ``dispatched`` -- the operator explicitly opted
    out.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, dead_dispatched_reap_minutes=0),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    old_drift_at = (datetime.now(UTC) - timedelta(minutes=120)).isoformat().replace("+00:00", "Z")
    fingerprint = json.dumps(
        {"reason": "dead_worker_clean_exit_no_op", "reviewed_head_sha": "abc123"},
        sort_keys=True,
        default=str,
    )
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "orphan_drift_at": old_drift_at,
        "orphan_drift_fingerprint": fingerprint,
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = sessions_dir / "issue-207.claude.terminal.json"
    terminal_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 0,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:05:00Z",
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # With the escape disabled, the issue stays dispatched (pre-#654 behavior).
    assert entry.get("status") == "dispatched"
    assert entry.get("escalation_reason") is None
    reaped_events = [
        e for e in state.get("events", []) if e.get("kind") == "dead_dispatched_worker_reaped"
    ]
    assert reaped_events == []


# ---------------------------------------------------------------------------
# Issue #417: dead-session reclaim must be idempotent and resumable, not a
# one-shot handoff that permanently strands an issue if interrupted mid-way.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SupervisorConfig tests
# ---------------------------------------------------------------------------


def test_signature_rule_is_frozen() -> None:
    """SignatureRule is a frozen dataclass."""
    import dataclasses

    rule = SignatureRule(pattern="x", kind="worker_blocked")
    assert dataclasses.is_dataclass(rule)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        rule.kind = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# loop() additions: open_tracked_prs + same-head packet skip
# ---------------------------------------------------------------------------


def test_loop_open_tracked_prs_counted(tmp_path: Path) -> None:
    """loop() data includes open_tracked_prs = number of PRs with linked issues."""
    prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc",
            "body": "Closes #123",
            "labels": [],
            "isCrossRepository": False,
        },
        # This PR has no linked issue — should NOT count
        {
            "number": 999,
            "title": "Manual PR",
            "url": "https://example.test/pull/999",
            "headRefName": "manual-branch",
            "headRefOid": "sha-xyz",
            "body": "no issue link",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app, _ = _make_loop_app(tmp_path, prs=prs)
    result = app.loop(limit=0)
    assert "open_tracked_prs" in result.data
    assert result.data["open_tracked_prs"] == 1


def test_loop_open_tracked_prs_zero_when_no_prs(tmp_path: Path) -> None:
    """loop() returns open_tracked_prs=0 when there are no open PRs."""
    app, _ = _make_loop_app(tmp_path, prs=[])
    result = app.loop(limit=0)
    assert result.data["open_tracked_prs"] == 0


def test_loop_undecided_same_head_skips_review(tmp_path: Path) -> None:
    """Undecided PR with a same-head packet does NOT re-invoke review()."""
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": "sha-same",
        "body": "Closes #123",
        "labels": [],
        "isCrossRepository": False,
    }
    app, fake_gh = _make_loop_app(tmp_path, prs=[pr])

    # Pre-plant a pr.json packet with the same headRefOid as the live PR
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    import json as _json

    (pr_dir / "pr.json").write_text(
        _json.dumps({"number": 456, "headRefOid": "sha-same"}), encoding="utf-8"
    )
    # No review-decision.json → undecided

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    result = app.loop(limit=0)

    assert result.data["skipped_reviews"] == 1
    assert 456 not in review_calls


def test_loop_undecided_head_moved_invokes_review(tmp_path: Path) -> None:
    """Undecided PR whose head has advanced past the packet re-invokes review()."""
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": "sha-new",
        "body": "Closes #123",
        "labels": [],
        "isCrossRepository": False,
    }
    app, fake_gh = _make_loop_app(tmp_path, prs=[pr])

    # Packet has OLD sha — head has moved
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    import json as _json

    (pr_dir / "pr.json").write_text(
        _json.dumps({"number": 456, "headRefOid": "sha-old"}), encoding="utf-8"
    )

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    result = app.loop(limit=0)

    assert result.data["skipped_reviews"] == 0
    assert 456 in review_calls


def test_loop_undecided_same_head_skip_still_merges_on_approved_decision_file(
    tmp_path: Path,
) -> None:
    """Regression for review finding #7: an operator-written decision file
    must not stay invisible until the head moves, even when state.json
    hasn't caught up -- it should proceed straight to merge_ready(), same as
    the decided path.

    Pre-#1362 this was reached via the same-head packet-skip branch (which
    re-checked the decision file directly as a fallback, incrementing
    ``skipped_reviews``). Issue #1362 Stage 1 made ``already_approved``
    itself file-first, so the file-only approval is now caught one branch
    earlier -- ``already_approved`` is True immediately and routes straight
    to ``merge_ready()`` without ever reaching the packet-skip branch, so
    ``skipped_reviews`` stays 0. The observable guarantee this test protects
    (the approval is not invisible; the PR merges) is unchanged; only which
    internal branch reaches it is.
    """
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": "sha-abc123",
        "body": "Closes #123\n\nTests: regression coverage added.",
        "labels": [],
        "isCrossRepository": False,
    }
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [pr]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # State has NO decision recorded yet (undecided from state's perspective).
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    import json as _json

    # Packet head matches the live PR head → same-head skip branch fires.
    (pr_dir / "pr.json").write_text(
        _json.dumps({"number": 456, "headRefOid": "sha-abc123"}), encoding="utf-8"
    )
    # Operator wrote the decision file directly; state.json wasn't updated.
    (pr_dir / "review-decision.json").write_text(
        _json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    result = app.loop(limit=0)

    # Packet regeneration is still skipped (review() never called)...
    assert 456 not in review_calls
    # Issue #1362 Stage 1: already_approved is file-first, so this approval
    # is caught before the packet-skip branch runs at all -- skipped_reviews
    # stays 0 rather than incrementing (see docstring above).
    assert result.data["skipped_reviews"] == 0
    # ...but the approval is not left invisible: merge_ready() fires.
    assert len(result.data["merges"]) == 1
    assert result.data["merges"][0]["merged"] is True


def test_loop_undecided_no_packet_invokes_review(tmp_path: Path) -> None:
    """Undecided PR with no existing packet still invokes review()."""
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": "sha-abc",
        "body": "Closes #123",
        "labels": [],
        "isCrossRepository": False,
    }
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    # No packet at all

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    result = app.loop(limit=0)

    assert result.data["skipped_reviews"] == 0
    assert 456 in review_calls


def test_worktree_unsafe_launch_failure_with_commits_salvages_before_escalation(
    tmp_path: Path,
) -> None:
    """Issue #1130: a ``worktree_unsafe`` launch failure whose worktree has
    commits ahead of base must attempt salvage (push + PR) before escalating
    to ``agent:human-needed``. Salvage-the-commit is the cheap safe action;
    human adjudication is the fallback only when salvage fails."""

    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state

    now = datetime.now(UTC)

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    # Create a worktree with one commit beyond origin/main — the stranded
    # work that ``worktree_unsafe`` refused to reset.
    worktree_path, branch = _setup_completed_worktree(repo_root, 1130)

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub(repo_root=repo_root)
    fake_gh.issues = [
        {
            "number": 1130,
            "title": "Salvage test",
            "url": "https://example.test/issues/1130",
            "body": "Salvage does not fire",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []  # No open PR.
    fake_gh.pr_create_return = 200

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-1130.log"
    log_path.write_text("worktree contains local work, cannot reset\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-1130.json"
    record = SessionRecord(
        issue_number=1130,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Launch failure — process never started
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="worktree creation failed: worktree contains local work",
        failure_kind="worktree_unsafe_local_commits",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # Salvage fired: branch pushed and PR created.
    remote_refs = _git(remote, "show-ref")
    assert branch in remote_refs.stdout
    assert len(fake_gh.prs_created) == 1
    assert fake_gh.prs_created[0]["head"] == branch

    # Labels moved to pr_open, NOT human_needed.
    assert (1130, config.labels.in_progress) in fake_gh.labels_removed
    assert (1130, config.labels.pr_open) in fake_gh.labels_added
    assert (1130, config.labels.human_needed) not in fake_gh.labels_added

    state = load_state(paths.state_file)
    salvage_events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert len(salvage_events) == 1
    assert salvage_events[0]["payload"]["issue_number"] == 1130
    # No escalation event.
    escalate_events = [e for e in state["events"] if e["kind"] == "session_failed_escalated"]
    assert not escalate_events


def test_worktree_unsafe_launch_failure_no_commits_still_escalates(
    tmp_path: Path,
) -> None:
    """Issue #1130: a ``worktree_unsafe`` launch failure whose worktree has NO
    commits ahead of base (e.g. dirty working tree with no commits) still
    escalates to ``agent:human-needed``. Salvage is only attempted when there
    is committed work to push."""

    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state

    now = datetime.now(UTC)

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    # A worktree with no commits ahead, just a dirty file.
    branch = "agent/issue-1130-no-commits"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    worktree_path = info.path
    (worktree_path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub(repo_root=repo_root)
    fake_gh.issues = [
        {
            "number": 1130,
            "title": "Salvage test",
            "url": "https://example.test/issues/1130",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []
    fake_gh.pr_create_return = 200

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-1130.log"
    log_path.write_text("worktree contains local work, cannot reset\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-1130.json"
    record = SessionRecord(
        issue_number=1130,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="worktree creation failed: worktree contains local work",
        failure_kind="worktree_unsafe_shim_dirt",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # No salvage: no PR created.
    assert not fake_gh.prs_created
    # Escalation fired: a deterministic (mechanical) launch failure parks on
    # operator_queue (issue #1266), not human_needed.
    assert (1130, config.labels.operator_queue) in fake_gh.labels_added

    state = load_state(paths.state_file)
    salvage_events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert not salvage_events
    escalate_events = [e for e in state["events"] if e["kind"] == "session_failed_escalated"]
    assert len(escalate_events) == 1


def test_phantom_live_worker_preserves_sidecar_for_dirty_worktree_with_commits(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #1130: a phantom live worker whose worktree is PARTIAL (dirty
    working tree but commits ahead of base) must NOT have its sidecar reaped.
    The committed work is salvageable; reaping the sidecar would destroy the
    salvage path. This guards the relaxed ``ahead_count > 0`` preserve
    condition against the previous ``COMPLETED``-only check."""

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 1130, dirty=True)

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree_path),
            prompt_path=str(worktree_path / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=8282,
            started_at="2026-08-10T11:15:39Z",
            log_path=str(tmp_path / "log"),
            error="probe_error",
            failure_kind="live_worker_redispatch_averted",
            process_start_time=5_678_901.0,
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    monkeypatch.setattr("charlie_work.workflow.is_pid_alive", lambda pid, start: False)

    config = OrchestratorConfig(
        devin=DevinConfig(), worker=WorkerRoleConfig(harness="claude-code")
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.pr_list = lambda: []
    _original_issue_view = fake_gh.issue_view

    def _patched_issue_view(number: int):
        issue = _original_issue_view(number)
        return {
            **issue,
            "labels": [
                {"name": "automated-ready"},
                {"name": "agent:in-progress"},
            ],
        }

    fake_gh.issue_view = _patched_issue_view

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sessions_dir / "issue-123.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 123,
                "branch": branch,
                "worktree_path": str(worktree_path),
                "prompt_path": "",
                "command": ["claude", "-p"],
                "pid": 8282,
                "started_at": "2026-08-10T11:15:39Z",
                "log_path": str(tmp_path / "log"),
                "error": "probe_error",
                "failure_kind": "live_worker_redispatch_averted",
                "process_start_time": 5_678_901.0,
                "session_id": "test-session-1130",
            }
        ),
        encoding="utf-8",
    )

    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "branch_name": branch,
        "worker_pid": 8282,
        "worker_process_start_time": 5_678_901.0,
        "title": "Fix search",
        "url": "https://example.test/issues/123",
    }
    save_state(paths.state_file, seed)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    # The phantom live worker is detected but the sidecar is PRESERVED.
    assert result.data["phantom_live_worker_count"] == 1
    assert sidecar_path.exists(), (
        "Sidecar must not be reaped so the reaper lane can salvage the committed work"
    )

    # Labels are NOT stripped.
    assert (123, "agent:in-progress") not in fake_gh.labels_removed
    assert (123, "automated-ready") not in fake_gh.labels_removed

    state = load_state(paths.state_file)
    preserve_events = [
        e
        for e in state.get("events", [])
        if e["kind"] == "session_failed_relabeled"
        and e["payload"]["issue_number"] == 123
        and e["payload"]["reason"] == "phantom_live_worker_completed_work_preserved"
    ]
    assert len(preserve_events) == 1


def test_salvage_branch_empty_diff_returns_false_on_fetch_failure(
    tmp_path: Path,
) -> None:
    """Issue #1221 (check 3 fail-safe): ``salvage_branch_empty_diff`` returns
    False (do not skip salvage) when ``git fetch origin <base>`` fails -- a
    transient network error falls back to opening the PR, which a human reviews
    anyway. This is the fail-safe branch the design relies on but had no test.
    """
    from charlie_work.worktree import salvage_branch_empty_diff

    # A git repo with NO origin remote: ``git fetch origin main`` fails.
    repo_root = tmp_path / "no-remote-clone"
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "Test User")
    _git(repo_root, "config", "commit.gpgSign", "false")
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo_root, "add", "README.md")
    _git(repo_root, "commit", "-m", "initial commit")

    # No origin remote configured -- fetch will fail.
    result = salvage_branch_empty_diff(repo_root, "agent/issue-1221", "main")
    assert result is False


def test_salvage_already_landed_proceeds_when_empty_diff_fetch_fails(
    tmp_path: Path,
) -> None:
    """Issue #1221 (check 3 fail-safe, integration): when the git fetch inside
    ``salvage_branch_empty_diff`` fails, the function returns False and
    ``_salvage_already_landed`` returns ``(False, None)`` -- salvage proceeds
    (does not skip) instead of treating the git error as evidence the work
    already landed. A human reviews salvage PRs anyway.
    """
    from charlie_work.workflow import _salvage_already_landed

    # A git repo with NO origin remote: ``git fetch origin main`` fails.
    repo_root = tmp_path / "no-remote-clone"
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "Test User")
    _git(repo_root, "config", "commit.gpgSign", "false")
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo_root, "add", "README.md")
    _git(repo_root, "commit", "-m", "initial commit")

    config = OrchestratorConfig()

    class FakeGitHubEmptyMergeSearch(FakeGitHub):
        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            return github_module._MergedPRSearchResult([], ok=True)

    gh = FakeGitHubEmptyMergeSearch()

    # Issue is OPEN, no merged PR binds to it, and the fetch inside
    # salvage_branch_empty_diff fails (no origin remote). The fail-safe
    # must let salvage proceed: _salvage_already_landed returns (False, None).
    already_landed, reason = _salvage_already_landed(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch="agent/issue-1221",
        base_ref="main",
        issue_number=1221,
        issue={"state": "OPEN"},
    )
    assert already_landed is False
    assert reason is None


def test_session_failed_relabeled_payload_requires_reason() -> None:
    """Issue #978: the shared payload builder makes a relabel event without a
    ``reason`` unrepresentable -- calling it without ``reason`` raises
    TypeError, the same property ``_escalate_issue`` gives escalation."""
    from charlie_work.workflow import _session_failed_relabeled_payload

    # reason is a required keyword-only argument.
    with pytest.raises(TypeError):
        _session_failed_relabeled_payload(issue_number=42)  # type: ignore[call-arg]

    # With reason, the payload always carries it; failure_kind is optional.
    payload = _session_failed_relabeled_payload(issue_number=42, reason="dead_worker_no_open_pr")
    assert payload["reason"] == "dead_worker_no_open_pr"
    assert "failure_kind" not in payload

    payload = _session_failed_relabeled_payload(
        issue_number=42, reason="dead_worker_no_open_pr", failure_kind="stalled"
    )
    assert payload["reason"] == "dead_worker_no_open_pr"
    assert payload["failure_kind"] == "stalled"


# ---------------------------------------------------------------------------
# Issue #480: api-worker budget settlement wiring at the workflow reap sites
# ---------------------------------------------------------------------------
#
# _classify_dead_sessions_and_update_throttle_state has two production
# reap_sidecar call sites that wire ``api_config=config.api_worker,
# state_dir=state_file.parent`` so an api worker's spend is settled into the
# ledger before its sidecar is unlinked:
#   - the launch-failure lane (~workflow.py:2497, pid=None + error set)
#   - the confirmed-dead lane (~workflow.py:2652)
# Neither had any test coverage. A wiring regression at either site would
# silently disable budget tracking with no test failing. These two tests drive
# the real classification lane and assert the ledger is populated.


# ---------------------------------------------------------------------------
# Issue #484 review finding: regression tests for the orchestration wiring
# (the budget-kill block in _detect_and_handle_stalled_sessions and the
# ``elif w.adapter_kind == "api"`` branches threaded through workflow.py /
# reconcile.py). The pure helpers (_classify_session_failure,
# _api_session_over_budget) are tested in test_claude_code_adapter.py /
# test_worker_health_api_budget.py; these tests exercise the production call sites so a
# wiring regression that drops the api branch or the budget-kill block is
# caught.
# ---------------------------------------------------------------------------


def test_detect_drift_api_dead_session_provider_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #484 review finding: the ``elif w.adapter_kind == "api"`` branch in
    ``reconcile.detect_drift``'s dead-session lane classifies a dead api
    worker with a 401 log tail as ``provider_auth`` and emits a
    ``provider_throttle_detected`` drift item. A wiring regression that drops
    this branch leaves no throttle drift item. The sidecar is reaped by this
    lane, so the classification is asserted via the drift list.
    """
    from _api_budget_fixtures import api_worker_config, write_api_sidecar
    from charlie_work.reconcile import detect_drift

    config = OrchestratorConfig(
        api_worker=api_worker_config(),
        post_mortem=PostMortemConfig(enabled=False),
    )
    gh = FakeGitHub()
    gh.issues = [
        {
            "number": 4805,
            "title": "Test issue",
            "url": "https://example.test/issues/4805",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.prs = []
    state = empty_state()

    sessions_dir = resolved_layout(config, tmp_path).sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    write_api_sidecar(sessions_dir, 4805, provider="example", pid=99996)

    log_path = sessions_dir / "issue-4805.claude.log"
    log_path.write_text("Error: 401 Unauthorized. Invalid API key.\n", encoding="utf-8")

    # The dead-session lane fires only when the worker is not alive.
    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: False)

    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # A provider_throttle_detected drift item is emitted with provider_auth.
    throttle_items = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert len(throttle_items) == 1
    assert throttle_items[0].issue_number == 4805
    assert "provider_auth" in throttle_items[0].detail


@contextlib.contextmanager
def _hold_state_lock(lock_path: Path) -> Any:
    """Hold a real, competing byte-range/exclusive lock on the state lock file.

    This is used to force ``state_lock`` to time out without involving another
    process, while still exercising the real platform locking primitive.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not lock_path.exists():
        lock_path.write_bytes(b"\x00")
    handle = lock_path.open("r+b")
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("status", ()),
        ("intake", ()),
        ("dispatch", ()),
        ("dispatch_rework", ()),
        ("review", (456,)),
        ("merge_ready", (456,)),
    ],
)
def test_state_lock_guard_returns_skip_when_lock_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    args: tuple[Any, ...],
) -> None:
    """Issue #398: if the state lock is held, public state-writing methods
    return a clean skip CommandResult and leave state.json untouched.
    """
    monkeypatch.setattr(state_module, "_LOCK_TIMEOUT_SECONDS", 0.05)

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)

    # Issue #614: merge_ready's dry-run gate sits above the state_lock and
    # returns a read-only verdict without acquiring the lock — so the
    # state-lock guard is only reachable on the non-dry-run path.  The state
    # lock is the first thing that path hits, so no side effects occur before
    # the StateLockBusy exception.
    # Issue #617: review() now has the same shape — its dry-run gate sits
    # above the state_lock and returns a read-only plan without acquiring
    # the lock.
    # Issue #618: intake's dry-run gate also sits above the state_lock —
    # dry-run skips the state merge entirely, so the lock guard never fires.
    if method_name in ("merge_ready", "review", "intake"):
        app.dry_run = False

    state_path = paths.state_file
    state_path.parent.mkdir(parents=True, exist_ok=True)
    initial_state = {
        "version": 1,
        "generated_at": "2026-01-01T00:00:00Z",
        "issues": {},
        "prs": {},
        "events": [],
    }
    state_path.write_text(json.dumps(initial_state), encoding="utf-8")
    initial_mtime = state_path.stat().st_mtime
    initial_content = state_path.read_text(encoding="utf-8")

    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    with _hold_state_lock(lock_path):
        result = getattr(app, method_name)(*args)

    assert result.ok is True
    reason = result.data.get("reason") or result.data.get("deferred_reason")
    assert reason in {"state_lock_busy", "supervisor_lock_held", "graphql_rate_limit"}
    assert result.data.get("pass_skipped") is True or result.data.get("state_lock_busy") is True
    assert state_path.stat().st_mtime == initial_mtime
    assert state_path.read_text(encoding="utf-8") == initial_content


def test_is_pre_review_rework_candidate_detects_merge_conflict_and_stale_empty_checks() -> None:
    """Issue #439: the two pre-review rework predicates are detected independently."""
    from datetime import UTC, datetime, timedelta

    from charlie_work.workflow import _is_pre_review_rework_candidate

    config = OrchestratorConfig()
    now = datetime.now(UTC)

    # Merge conflict is an immediate rework trigger.
    assert _is_pre_review_rework_candidate({"mergeable": "CONFLICTING"}, config, now) == (
        True,
        "merge_conflict",
    )

    # mergeStateStatus DIRTY is also an immediate trigger with a distinct reason.
    assert _is_pre_review_rework_candidate({"mergeStateStatus": "DIRTY"}, config, now) == (
        True,
        "rework_branch_conflict",
    )

    old = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    stale_pr = {"statusCheckRollup": [], "updatedAt": old}
    assert _is_pre_review_rework_candidate(stale_pr, config, now) == (
        True,
        "stale_empty_checks",
    )

    # A fresh empty-rollup PR is not yet stale.
    fresh = (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    fresh_pr = {"statusCheckRollup": [], "updatedAt": fresh}
    assert _is_pre_review_rework_candidate(fresh_pr, config, now) == (False, "")

    # Any present check disqualifies the stale predicate.
    checks_pr = {"statusCheckRollup": [{"name": "Tests passed"}], "updatedAt": old}
    assert _is_pre_review_rework_candidate(checks_pr, config, now) == (False, "")


def test_is_pr_updated_at_older_than() -> None:
    """The shared updatedAt threshold helper parses, tz-normalizes, and compares."""
    from datetime import UTC, datetime, timedelta

    from charlie_work.workflow import _is_pr_updated_at_older_than

    now = datetime.now(UTC)
    stale = (now - timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    fresh = (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")

    assert _is_pr_updated_at_older_than({"updatedAt": stale}, now, 15) is True
    assert _is_pr_updated_at_older_than({"updatedAt": fresh}, now, 15) is False
    assert _is_pr_updated_at_older_than({}, now, 15) is False
    assert _is_pr_updated_at_older_than({"updatedAt": "not-a-date"}, now, 15) is False

    # Naive datetimes are normalized to UTC before comparison.
    naive_now = datetime.now()
    naive_updated = (naive_now - timedelta(minutes=20)).replace(microsecond=0)
    assert (
        _is_pr_updated_at_older_than({"updatedAt": naive_updated.isoformat()}, naive_now, 15)
        is True
    )


def test_is_readiness_no_ci_stall() -> None:
    """Issue #474: the readiness no-CI gate escalates only when required checks are missing and updatedAt is stale."""
    from datetime import UTC, datetime, timedelta

    from charlie_work.config import AutoMergeConfig
    from charlie_work.workflow import _is_readiness_no_ci_stall

    now = datetime.now(UTC)
    required = ("Tests passed", "Lint & Format")
    stale = (now - timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    fresh = (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    config = AutoMergeConfig(required_checks=required, readiness_no_ci_minutes=15)

    # Missing required checks and stale updatedAt.
    assert _is_readiness_no_ci_stall({"updatedAt": stale}, [], config, now) is True

    # A required check has appeared.
    assert (
        _is_readiness_no_ci_stall({"updatedAt": stale}, [{"name": "Tests passed"}], config, now)
        is False
    )

    # Missing checks but the PR was updated recently.
    assert _is_readiness_no_ci_stall({"updatedAt": fresh}, [], config, now) is False

    # Gate disabled by zero minutes.
    disabled = AutoMergeConfig(required_checks=required, readiness_no_ci_minutes=0)
    assert _is_readiness_no_ci_stall({"updatedAt": stale}, [], disabled, now) is False

    # No required checks configured: there is nothing to be missing.
    no_required = AutoMergeConfig(required_checks=(), readiness_no_ci_minutes=15)
    assert _is_readiness_no_ci_stall({"updatedAt": stale}, [], no_required, now) is False

    # Missing or malformed updatedAt is treated as not stale.
    assert _is_readiness_no_ci_stall({}, [], config, now) is False


def test_dry_run_dispatch_cross_repo_gate_reports_without_mutating(tmp_path: Path) -> None:
    """Issue #1010 wiring (dry-run): ``dispatch`` with ``dry_run=True`` reports
    which issues the cross-repo gate would escalate, without mutating state,
    labels, or events.

    Drives the dry-run branch of ``_dispatch_impl`` (the path that populates
    ``cross_repo_escalated_issue_numbers`` in the planning payload), not
    ``cross_repo_gate`` in isolation.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    fake_gh.issues[0]["body"] = _cross_repo_issue_body()
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


# --- Issue #507: record review verdicts from dead reviewer logs -------------


def test_parse_review_verdict_from_log_extracts_last_fenced_json(tmp_path: Path) -> None:
    """Issue #507: parse the last fenced JSON verdict block from a log."""
    log = tmp_path / "review.claude.log"
    log.write_text(
        'Some earlier output\n```json\n{"decision": "blocked", "summary": "old"}\n```\n'
        'Final verdict:\n```json\n{\n  "decision": "approved",\n  "summary": "lgtm",\n  "required_changes": []\n}\n```\n',
        encoding="utf-8",
    )

    verdict = _parse_review_verdict_from_log(log)

    assert verdict is not None
    assert verdict["decision"] == "approved"
    assert verdict["summary"] == "lgtm"
    assert verdict["required_changes"] == []


def test_parse_review_verdict_from_log_extracts_json_after_language_tagged_fence(
    tmp_path: Path,
) -> None:
    """Regression: ``_VERDICT_FENCE_RE`` previously only recognized an opening
    fence tagged bare or ``json`` (``` ```(?:json)?\\s*\\n `` ``), so a
    reviewer quoting evidence in a ```python fence before its final verdict
    fence would desync the pairing entirely -- the ```python fence's own
    opening backtick never matched, so its *closing* bare ``` got misread as
    a new opening and swallowed everything up to the *next* fence's opening,
    permanently misaligning the scan. This mirrors the exact structure that
    hid a real, well-formed verdict in a production cross-family report
    (PR #802); the same regex is duplicated here in ``workflow.py`` (kept
    latent so far by per-event stream-json decoding, but a real defect)."""
    log = tmp_path / "review.claude.log"
    log.write_text(
        "Citing the bug:\n```python\ndef broken():\n    return None\n```\n"
        "That's a real problem.\n\n"
        'Final verdict:\n```json\n{\n  "decision": "request_changes",\n'
        '  "summary": "broken() returns None instead of raising",\n'
        '  "required_changes": ["Raise instead of returning None"]\n}\n```\n',
        encoding="utf-8",
    )

    verdict = _parse_review_verdict_from_log(log)

    assert verdict is not None
    assert verdict["decision"] == "request_changes"
    assert verdict["summary"] == "broken() returns None instead of raising"
    assert verdict["required_changes"] == ["Raise instead of returning None"]


def test_parse_review_verdict_from_log_requires_valid_decision(tmp_path: Path) -> None:
    """Issue #507: only accepted decisions and non-empty summaries are valid."""
    log = tmp_path / "review.claude.log"
    log.write_text(
        '```json\n{"decision": "maybe", "summary": "?"}\n```\n',
        encoding="utf-8",
    )

    assert _parse_review_verdict_from_log(log) is None


def test_parse_review_verdict_from_log_rejects_empty_request_changes_summary(
    tmp_path: Path,
) -> None:
    """Issue #507: request_changes with an empty summary is not a valid verdict."""
    log = tmp_path / "review.claude.log"
    log.write_text(
        '```json\n{"decision": "request_changes", "summary": "   "}\n```\n',
        encoding="utf-8",
    )

    assert _parse_review_verdict_from_log(log) is None


def test_cross_family_request_changes_verdict_persists_required_changes(
    tmp_path: Path,
) -> None:
    """End-to-end: a rescue-tier request_changes verdict's decision/summary/
    required_changes, recorded the same way the rescue tier's own
    record_review call site does, ends up with a populated
    ``required_changes`` in review-decision.json -- the exact defect this fix
    closes (8 of 20 request_changes verdicts had it silently empty).

    The decision/summary/required_changes below are the literal values a
    JSON verdict block of
    ``{"decision": "request_changes", "summary": "file.py:10 has a real bug
    that breaks X", "required_changes": ["Fix the off-by-one in
    file.py:10", "Add a regression test for the empty-list case"]}``
    used to parse to, back when ``parse_cross_family_verdict`` (deleted in
    the role-config Phase 2 cleanup) extracted them from a report body."""
    verdict_decision = "request_changes"
    verdict_summary = "file.py:10 has a real bug that breaks X"
    verdict_required_changes = (
        "Fix the off-by-one in file.py:10",
        "Add a regression test for the empty-list case",
    )

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Mirrors the rescue tier's own record_review call site.
    # PR 456 is FakeGitHub's seeded default PR.
    result = app.record_review(
        456,
        verdict_decision,
        summary=verdict_summary,
        required_changes=verdict_required_changes,
        verdict_provenance="rescue_review",
    )
    assert result.ok

    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["required_changes"] == [
        "Fix the off-by-one in file.py:10",
        "Add a regression test for the empty-list case",
    ]
    assert decision["summary"] == "file.py:10 has a real bug that breaks X"


def test_cross_family_legacy_path_verdict_with_empty_required_changes_gets_derived(
    tmp_path: Path,
) -> None:
    """AC-6 (rescue-tier producer): a request_changes verdict with an empty
    required_changes and a real, non-vacuous summary -- the shape the legacy
    Markdown-only cross-family parse path (no JSON verdict block, deleted
    along with ``parse_cross_family_verdict`` in the role-config Phase 2
    cleanup) used to produce, since it only ever extracted a summary, never
    itemized findings -- always arrives at record_review as
    required_changes=(). This is the shape record_review's derivation
    exists for. Contrast with the (deleted) AC-8 malformed-verdict tests: a
    JSON verdict block declaring request_changes with an empty
    required_changes was diverted to MalformedCrossFamilyVerdict before ever
    reaching record_review (issue #795) -- this scenario has no JSON block
    at all, so that defense-in-depth layer never applied here and
    record_review's own derivation is what prevents the content-free
    outcome.

    ``verdict_summary`` below is the literal value
    ``"**MAJOR**\\nreal bug\\n\\nVerdict: MAJOR issues block merge"``
    used to parse to via the legacy path's verdict-line extraction."""
    verdict_decision = "request_changes"
    verdict_summary = "MAJOR issues block merge"
    verdict_required_changes: tuple[str, ...] = ()
    assert verdict_summary and not _summary_is_vacuous(verdict_summary)

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Mirrors the rescue tier's own record_review call site.
    result = app.record_review(
        456,
        verdict_decision,
        summary=verdict_summary,
        required_changes=verdict_required_changes,
        verdict_provenance="rescue_review",
    )
    assert result.ok is True

    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["required_changes"] == [verdict_summary]
    assert decision["findings_channel"] == "derived"


def test_render_required_changes_section_populated_list_unaffected_by_fallback() -> None:
    """Non-regression: when required_changes IS populated, the enumerated
    list renders exactly as before even though a summary is also present --
    the summary-fallback and escape-hatch tiers must never fire when there is
    real structured content to show."""
    decision = {
        "decision": "request_changes",
        "summary": "This summary must not appear in the rendered section.",
        "required_changes": ["fix the off-by-one", "add a regression test"],
    }

    section = _render_required_changes_section(decision)

    assert "## Required changes" in section
    assert "- fix the off-by-one" in section
    assert "- add a regression test" in section
    # The fallback/escape-hatch framing must not leak in alongside the list.
    assert "did not record a structured findings list" not in section
    assert "REVIEWER FINDINGS UNAVAILABLE" not in section
    assert "This summary must not appear" not in section


def test_render_required_changes_section_both_empty_renders_escape_hatch() -> None:
    """Hard requirement (F1): when both required_changes and summary are
    empty on a request_changes verdict, the section must never look like
    "nothing to change" -- it must loudly say the findings are unavailable
    and point the worker at the PR's review comments on GitHub instead of
    silently rendering an empty section."""
    decision = {"decision": "request_changes", "summary": "", "required_changes": []}

    section = _render_required_changes_section(decision)

    assert section.strip() != ""
    assert "REVIEWER FINDINGS UNAVAILABLE" in section
    assert "is NOT a signal that there is nothing to change" in section
    assert "GitHub" in section


def test_render_required_changes_section_blocked_both_empty_renders_escape_hatch() -> None:
    """Defensive extension of the same hard requirement to `blocked`: the
    decision-agnostic janitor-gate rework routes (merge-conflict / no-op
    repair) can carry forward whatever verdict was last on disk, including a
    `blocked` one. Suppressing the enumerated list / summary fallback for
    `blocked` (see the omits-for-approved-style suppression covered above)
    is deliberate, but suppressing it AND leaving the worker with zero signal
    that something was withheld is not -- the both-empty escape hatch still
    fires."""
    decision = {"decision": "blocked", "summary": "", "required_changes": []}

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section


# --------------------------------------------------------------------------
# Issue #792: verdicts recorded by the current record_review carry an
# explicit findings_channel marker ("vacuous" or "derived"). These tests
# cover the renderer's handling of that marker directly, independent of the
# shape-based (required_changes/summary) tiers above, which exist only to
# infer the same distinction for pre-#792 records with no marker at all.
# --------------------------------------------------------------------------


def test_render_required_changes_section_vacuous_marker_renders_escape_hatch() -> None:
    """A findings_channel="vacuous" verdict always renders tier 3, even
    though `summary` is technically non-blank (it may carry the historical
    placeholder) -- rendering it as real content (tier 2) would silently
    present content-free text as the reviewer's actual findings."""
    decision = {
        "decision": "request_changes",
        "summary": LEGACY_VACUOUS_SUMMARY,
        "required_changes": [],
        "findings_channel": "vacuous",
    }

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section
    assert LEGACY_VACUOUS_SUMMARY not in section


def test_render_required_changes_section_vacuous_marker_fires_for_blocked_too() -> None:
    """The vacuous marker's escape hatch is decision-agnostic -- it fires for
    `blocked` exactly as it does for `request_changes`, unlike the "derived"
    marker below which is request_changes-specific."""
    decision = {
        "decision": "blocked",
        "summary": LEGACY_VACUOUS_SUMMARY,
        "required_changes": [],
        "findings_channel": "vacuous",
    }

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section


def test_render_required_changes_section_derived_marker_renders_summary_verbatim() -> None:
    """A findings_channel="derived" request_changes verdict renders the
    tier-2-shaped verbatim-summary section even though required_changes is
    now populated (record_review copied summary into it) -- it must not
    fall into tier 1's bullet-list rendering, which would wrap an entire
    multi-sentence summary as a single bullet."""
    prose = "The retry wrapper swallows the exception type; callers cannot distinguish causes."
    decision = {
        "decision": "request_changes",
        "summary": prose,
        "required_changes": [prose],
        "findings_channel": "derived",
    }

    section = _render_required_changes_section(decision)

    assert prose in section
    assert f"- {prose}" not in section
    assert "did not record a structured findings list" in section


def test_render_required_changes_section_derived_marker_suppressed_for_blocked() -> None:
    """Unlike "vacuous", the "derived" marker's special-cased rendering only
    applies to request_changes (see the docstring's blocked-suppression
    rule) -- for `blocked` it falls through to the pre-existing shape-based
    tiers, where a populated required_changes on a blocked verdict is
    suppressed by design (blocked's "what must change before approval"
    framing doesn't fit blocked's routes)."""
    prose = "Security review flagged an unauthenticated endpoint."
    decision = {
        "decision": "blocked",
        "summary": prose,
        "required_changes": [prose],
        "findings_channel": "derived",
    }

    section = _render_required_changes_section(decision)

    assert section == ""


def test_defang_closing_keywords_strips_live_keyword_but_keeps_number_legible() -> None:
    # Issue #781 AC3: defang_closing_keywords rewrites `<keyword> #N` to
    # `<keyword> issue N` -- the rewritten text no longer matches the
    # closing-keyword regex (so it can no longer auto-close or falsely bind
    # via linked_issue_number), but the issue number stays legible to a
    # human reader.
    from charlie_work.github import defang_closing_keywords
    from charlie_work.issue_linking import _CLOSING_KEYWORD_REF

    text = "does not fix #649"
    defanged = defang_closing_keywords(text)

    assert _CLOSING_KEYWORD_REF.search(defanged) is None
    assert "649" in defanged, "issue number must remain legible to a human"
    assert defanged == "does not fix issue 649"


def test_defang_closing_keywords_preserves_non_keyword_hash_refs() -> None:
    # Bare `#N` (no preceding closing keyword) and `issue #N` mentions pass
    # through untouched -- defang targets only the live auto-close syntax.
    # Multiple keyword refs in one string are each independently defanged.
    from charlie_work.github import defang_closing_keywords

    assert defang_closing_keywords("see #5 for context") == "see #5 for context"
    assert defang_closing_keywords("related to issue #5") == "related to issue #5"
    assert (
        defang_closing_keywords("Closes #1 and also fixes #2")
        == "Closes issue 1 and also fixes issue 2"
    )


def test_render_required_changes_section_defangs_live_keyword_in_list_tier() -> None:
    # Issue #781 AC4: reviewer prose in the required_changes list tier must
    # not carry a live closing keyword into the rendered brief -- a worker
    # reads this brief and writes its own PR body from it, a boundary
    # linked_issue_number's hijack-safety check never sees.
    from charlie_work.issue_linking import _CLOSING_KEYWORD_REF

    decision = {
        "decision": "request_changes",
        "summary": "",
        "required_changes": ["this does not fix #649, dig deeper"],
    }

    section = _render_required_changes_section(decision)

    assert "does not fix issue 649" in section
    assert _CLOSING_KEYWORD_REF.search(section) is None, "live keyword survived"


def test_render_required_changes_section_defangs_live_keyword_in_summary_tier() -> None:
    # Same guarantee for the summary-fallback tier (tier 2).
    from charlie_work.issue_linking import _CLOSING_KEYWORD_REF

    decision = {
        "decision": "request_changes",
        "summary": "BLOCKER - does not fix #649. Still broken.",
        "required_changes": [],
    }

    section = _render_required_changes_section(decision)

    assert "does not fix issue 649" in section
    assert _CLOSING_KEYWORD_REF.search(section) is None, "live keyword survived"


# --------------------------------------------------------------------------
# Issue #1269 (W12): the render-side crash-signature guard is the single
# enforcement point that reaches records already persisted before the
# collector-side fix in _collect_external_findings shipped -- these tests
# cover both the new-shape (external_findings field) and old-shape
# (findings_channel == "external", merged into required_changes) paths.
# --------------------------------------------------------------------------


def test_render_required_changes_section_strips_crash_signature_from_external_findings() -> None:
    """New shape: a crash-comment body sitting in `external_findings` from
    before the collector-side fix is stripped at render time; a genuine
    external finding alongside it survives untouched."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": "Reviewer summary text.",
        "required_changes": ["fix the off-by-one"],
        "external_findings": ["A human found a real bug in the retry loop.", crash_body],
    }

    section = _render_required_changes_section(decision)

    assert "A human found a real bug in the retry loop." in section
    assert crash_body not in section
    assert "did not produce a structured verdict" not in section
    assert "## Findings posted on the PR itself" in section, "new_shape must still be True"


def test_render_required_changes_section_all_crash_external_findings_falls_back_to_pointer() -> (
    None
):
    """New shape, entirely crash noise: external_findings ends up empty
    after filtering, so `new_shape` becomes False and this falls to the
    ordinary pointer-style ending (_finish_required_changes_section) rather
    than rendering an external-findings section with nothing real in it."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": "",
        "required_changes": ["fix the off-by-one"],
        "external_findings": [crash_body],
    }

    section = _render_required_changes_section(decision)

    assert "fix the off-by-one" in section
    assert crash_body not in section
    assert "## Findings posted on the PR itself" not in section
    assert "none of which reach this brief" in section, (
        "the ordinary external-findings pointer must reappear once external_findings "
        "filters down to empty"
    )


def test_render_required_changes_section_old_shape_strips_crash_signature_from_changes() -> None:
    """Old shape (findings_channel == "external"): a crash comment merged
    into required_changes before the collector-side fix is stripped; the
    reviewer's own genuine item survives (defense-in-depth for a reopened
    old-shape PR, per issue #1269 open question 3)."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": "",
        "required_changes": ["fix the off-by-one", crash_body],
        "findings_channel": "external",
    }

    section = _render_required_changes_section(decision)

    assert "- fix the off-by-one" in section
    assert crash_body not in section


def test_render_required_changes_section_old_shape_vacuous_all_crash_changes_renders_tier3() -> (
    None
):
    """Old shape, all-crash: when the crash filter empties `changes`
    entirely AND the leftover summary is the vacuous placeholder
    (record_review's vacuous-replace path -- the only way this exact
    combination is reachable), the section must render tier 3, NOT fall
    through to tier 2 and present the content-free placeholder as if it
    were real findings. This is the failure mode the "vacuous" marker
    branch already guards against, reached here through the old-shape
    "external" path instead of a fresh "vacuous" marker."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": LEGACY_VACUOUS_SUMMARY,
        "required_changes": [crash_body],
        "findings_channel": "external",
    }

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section
    assert LEGACY_VACUOUS_SUMMARY not in section
    assert crash_body not in section
    assert "did not record a structured findings list" not in section, (
        "must not fall through to the tier-2 verbatim-summary rendering"
    )


def test_render_required_changes_section_old_shape_blank_summary_all_crash_changes_renders_tier3() -> (
    None
):
    """Same guard, other half of the "vacuous" OR-condition: a blank
    summary (not the LEGACY_VACUOUS_SUMMARY placeholder) alongside an
    all-crash `changes` list also renders tier 3, not tier 2's empty-prose
    rendering."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": "",
        "required_changes": [crash_body],
        "findings_channel": "external",
    }

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section
    assert crash_body not in section


def test_render_required_changes_section_old_shape_all_crash_changes_preserves_genuine_summary() -> (
    None
):
    """Guard against over-aggression: when the crash filter empties
    `changes` entirely but the summary is genuine, non-vacuous reviewer
    prose (the pre-#999 "reviewer chose prose over an itemized list"
    population -- see `_is_carry_forward_eligible`'s docstring), tier 2
    must still fire and render that real summary. The vacuous-neutralization
    guard must only fire on the two known placeholder shapes (blank or
    LEGACY_VACUOUS_SUMMARY), never on genuine content."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    genuine_summary = (
        "The retry wrapper swallows the exception type; callers cannot distinguish causes."
    )
    decision = {
        "decision": "request_changes",
        "summary": genuine_summary,
        "required_changes": [crash_body],
        "findings_channel": "external",
    }

    section = _render_required_changes_section(decision)

    assert genuine_summary in section
    assert crash_body not in section
    assert "REVIEWER FINDINGS UNAVAILABLE" not in section, (
        "a genuine, non-vacuous summary must not be discarded down to tier 3"
    )


def test_render_required_changes_section_vacuous_guard_does_not_drop_populated_external_findings() -> (
    None
):
    """Hardening (review of 63ce581): the vacuous-neutralization guard's
    `and not external_findings` clause must not let a co-persisted, populated
    `external_findings` field get silently discarded.

    No current writer produces this exact shape -- `findings_channel ==
    "external"` old-shape records and populated `external_findings` are
    disjoint in every writer today -- but the guard's condition must not
    silently assume that forever. Before the `and not external_findings`
    clause, an all-crash old-shape `changes` plus a vacuous summary would
    neutralize summary_text to "" regardless of `external_findings`, landing
    in the tier-3 "both empty" branch below -- which returns immediately
    without ever consulting `external_findings` at all, dropping genuine
    findings on the floor. With the clause, a populated `external_findings`
    keeps the guard from firing, so the section falls through to the
    tier-2 summary-fallback branch instead, which -- because `new_shape` is
    True -- still calls `_render_external_findings_section` and renders
    every genuine finding."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    genuine_external_finding = "the retry wrapper swallows the exception type"
    decision = {
        "decision": "request_changes",
        "summary": LEGACY_VACUOUS_SUMMARY,
        "required_changes": [crash_body],
        "findings_channel": "external",
        "external_findings": [genuine_external_finding],
    }

    section = _render_required_changes_section(decision)

    assert genuine_external_finding in section, (
        "populated external_findings must still be rendered, not dropped by "
        "the vacuous-neutralization guard"
    )
    assert crash_body not in section
    assert "REVIEWER FINDINGS UNAVAILABLE" not in section, (
        "must not fall through to tier 3 and discard the genuine external findings"
    )


# --------------------------------------------------------------------------
# Issue #1310: the tier-2 (summary-verbatim) render path -- both the
# marker-less `summary_text` fallback and the `"derived"` marker branch --
# must crash-filter `summary` exactly as W12 (#1269) filtered
# `external_findings`/`required_changes`. A crash-signature body or the
# LEGACY_VACUOUS_SUMMARY placeholder arriving as a verdict `summary` with
# an empty findings list must degrade to tier 3, not render verbatim.
# Production-unreachable today (crash bodies are posted as PR comments,
# never written into a verdict `summary`), but the suppression contract is
# "crash noise cannot reach prompt content through any render path."
# --------------------------------------------------------------------------


def test_render_required_changes_section_tier2_crash_summary_degrades_to_tier3() -> None:
    """Marker-less tier-2 path: a request_changes round whose `summary` is a
    crash body and whose findings list is empty must NOT render the crash
    text verbatim -- it degrades to tier 3 (the "findings unavailable"
    escape hatch), exactly as the other tiers do."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": crash_body,
        "required_changes": [],
    }

    section = _render_required_changes_section(decision)

    assert crash_body not in section, (
        "tier-2 verbatim emit must not render a crash-signature summary"
    )
    assert "REVIEWER FINDINGS UNAVAILABLE" in section, (
        "must degrade to tier 3 when the summary is a crash body"
    )
    assert "did not record a structured findings list" not in section, (
        "must not fall through to the tier-2 verbatim-summary rendering"
    )


def test_render_required_changes_section_tier2_vacuous_summary_degrades_to_tier3() -> None:
    """Marker-less tier-2 path: a request_changes round whose `summary` is
    the LEGACY_VACUOUS_SUMMARY placeholder and whose findings list is empty
    must degrade to tier 3, not present the content-free placeholder as if
    it were real findings."""
    decision = {
        "decision": "request_changes",
        "summary": LEGACY_VACUOUS_SUMMARY,
        "required_changes": [],
    }

    section = _render_required_changes_section(decision)

    assert LEGACY_VACUOUS_SUMMARY not in section, (
        "tier-2 verbatim emit must not render the vacuous placeholder"
    )
    assert "REVIEWER FINDINGS UNAVAILABLE" in section, (
        "must degrade to tier 3 when the summary is the vacuous placeholder"
    )
    assert "did not record a structured findings list" not in section


def test_render_required_changes_section_tier2_derived_crash_summary_degrades() -> None:
    """`"derived"` marker branch: a request_changes round stamped
    `findings_channel == "derived"` whose `summary` is a crash body and
    whose findings list is empty must degrade to tier 3, not render the
    crash text verbatim. Without the `and summary_text` guard on the
    derived branch, the crash guard would neutralize `summary_text` to ""
    but the branch would still fire and render an empty verbatim summary."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": crash_body,
        "required_changes": [],
        "findings_channel": "derived",
    }

    section = _render_required_changes_section(decision)

    assert crash_body not in section, (
        "derived tier-2 verbatim emit must not render a crash-signature summary"
    )
    assert "REVIEWER FINDINGS UNAVAILABLE" in section, (
        "must degrade to tier 3 when the derived summary is a crash body"
    )
    assert "did not record a structured findings list" not in section


def test_render_required_changes_section_tier2_genuine_summary_still_renders() -> None:
    """Guard against over-aggression: a genuine, non-crash, non-vacuous
    summary with an empty findings list must still render verbatim at
    tier 2. The crash/vacuous guard must only fire on crash-signature
    bodies or the LEGACY_VACUOUS_SUMMARY placeholder, never on real
    reviewer prose."""
    genuine_summary = (
        "The retry wrapper swallows the exception type; callers cannot distinguish causes."
    )
    decision = {
        "decision": "request_changes",
        "summary": genuine_summary,
        "required_changes": [],
    }

    section = _render_required_changes_section(decision)

    assert genuine_summary in section, "a genuine summary must still render verbatim at tier 2"
    assert "REVIEWER FINDINGS UNAVAILABLE" not in section, (
        "a genuine summary must not be discarded down to tier 3"
    )


def test_no_op_rework_repair_brief_preserves_reviewer_summary(tmp_path: Path) -> None:
    """F3: _request_no_op_rework_repair's hardcoded no-op note must not
    displace the reviewer's findings. It routes through _route_to_rework,
    which calls the shared _write_rework_prompt -- the same single point of
    enforcement (issue #632) every other rework route uses -- so the
    reviewer's on-disk verdict is read fresh from review-decision.json,
    independent of the `summary` argument this method builds. Verified
    concretely here rather than assumed: this test fails if that separation
    is ever broken (e.g. a future edit threads the no-op text into
    review-decision.json's own `summary`, or bypasses _write_rework_prompt)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    reviewer_summary = (
        "BLOCKER - does not fix #649. The pin is a no-op over uv's resolver; "
        "the underlying version conflict is still unresolved."
    )
    decision = {
        "decision": "request_changes",
        "summary": reviewer_summary,
        "required_changes": [],
    }
    (pr_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")
    pr = fake_gh.pr_view(456)

    result = app._request_no_op_rework_repair(pr, 123, decision)

    assert result is None, f"expected a clean label transition, got {result!r}"
    brief = (pr_dir / "rework-prompt.md").read_text(encoding="utf-8")
    # The no-op operational note (the "dispatch_note") is present...
    assert "no actual content change" in brief
    # ...and the reviewer's original findings are still present alongside it,
    # not replaced by it -- defanged of the live closing keyword (issue
    # #781 outbound fix) the same way as the summary-fallback test above.
    assert "does not fix issue 649" in brief
    assert "does not fix #649" not in brief, "live closing keyword leaked into brief"


def test_no_op_rework_repair_note_survives_dispatch_rework_regeneration(tmp_path: Path) -> None:
    """Issue #887: the operational note from _route_to_rework must survive
    a dispatch_rework re-render unchanged.

    _request_no_op_rework_repair routes through _route_to_rework, which uses
    the shared _write_rework_prompt. That writer also writes the
    rework-dispatch-note.txt sidecar. dispatch_rework re-renders stale briefs
    and replays the note from the sidecar; this test extends the same no-op
    repair setup through a full dispatch_rework call and a newer-verdict
    re-render, asserting the operational note is still present in the
    regenerated brief. A future writer that bypasses _write_rework_prompt and
    leaves the sidecar absent or stale would fail this test, because the
    re-render would either drop the note (mtime-gate empty note) or replay a
    stale one."""
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
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    reviewer_summary = (
        "BLOCKER - does not fix #649. The pin is a no-op over uv's resolver; "
        "the underlying version conflict is still unresolved."
    )
    decision = {
        "decision": "request_changes",
        "summary": reviewer_summary,
        "required_changes": [],
    }
    (pr_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")
    pr = fake_gh.pr_view(456)

    result = app._request_no_op_rework_repair(pr, 123, decision)
    assert result is None, f"expected a clean label transition, got {result!r}"

    brief_path = pr_dir / "rework-prompt.md"
    note_path = pr_dir / "rework-dispatch-note.txt"
    before = brief_path.read_text(encoding="utf-8")
    # The no-op operational note (the "dispatch_note") is present...
    assert "no actual content change" in before
    # ...and the reviewer's original findings are present alongside it.
    assert "does not fix issue 649" in before
    # _write_rework_prompt produced a sidecar for dispatch-time re-render.
    operational_note = (
        "The previous rework cycle produced no actual content change (the diff or head "
        "matches the last request_changes verdict). Check the branch worktree for "
        "unpushed commits and push the real fix, or explain in the PR body why no "
        "further change was needed."
    )
    assert note_path.read_text(encoding="utf-8") == operational_note

    # The operator / reviewer updates the verdict with a new finding and makes
    # it newer than the brief. This is the #632 axis that forces a re-render.
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": reviewer_summary,
                "required_changes": ["add a regression test for the empty-list case"],
            }
        ),
        encoding="utf-8",
    )
    now = time.time()
    os.utime(brief_path, (now, now))
    os.utime(note_path, (now, now))
    os.utime(pr_dir / "review-decision.json", (now + 10, now + 10))

    result = app.dispatch_rework()

    assert result.ok is True, result.message
    assert result.data["selected_count"] == 1
    after = brief_path.read_text(encoding="utf-8")
    # The regenerated brief picked up the new verdict finding.
    assert "add a regression test for the empty-list case" in after
    # The operational note from _route_to_rework was replayed from the sidecar
    # and survived the re-render.
    assert "no actual content change" in after
    # The sidecar itself is unchanged.
    assert note_path.read_text(encoding="utf-8") == operational_note
    # The regeneration is recorded with the correct axis.
    events = query_events(paths.state_file, kind="rework_brief_regenerated")
    assert len(events) == 1, events
    assert events[0]["payload"]["reason"] == "verdict_newer"
    assert events[0]["payload"]["pr_number"] == 456


def test_detect_and_handle_stalled_reviews_aggregates_same_pass_events(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #525: multiple stalled reviewer claims in one pass become one sweep event."""
    from charlie_work.worker import WorkerView

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    state_file = tmp_path / "state.json"
    config = OrchestratorConfig()

    prs = [100, 200, 300]
    old = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    state = empty_state()
    for pr in prs:
        state["prs"][str(pr)] = {
            "number": pr,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old,
            "reviewer_pid": 99999,
            "reviewer_process_start_time": 1.0,
        }
    save_state(state_file, state)

    for pr in prs:
        _make_dead_review_sidecar(reviews_dir, pr, "no verdict")

    monkeypatch.setattr(WorkerView, "is_alive", lambda self: False)
    monkeypatch.setattr("charlie_work.stalled_review_reap.is_pid_alive", lambda *_: False)
    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", lambda *a, **k: True
    )

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert {entry["pr"] for entry in stalled} == set(prs)
    state_after = load_state(state_file)
    events = state_after["events"]
    sweep = [e for e in events if e.get("kind") == "review_dispatch_stalled_sweep"]
    assert len(sweep) == 1
    assert sweep[0]["payload"]["count"] == len(prs)
    assert set(sweep[0]["payload"]["pr_numbers"]) == set(prs)


def test_orchestrator_app_init_wires_event_ring_size_from_config(tmp_path: Path) -> None:
    """Issue #525: OrchestratorApp.__init__ sets state.EVENT_RING_SIZE from
    RuntimeConfig.event_ring_size so the default append_event cap is
    config-driven. A regression here silently leaves the ring at the hardcoded
    default regardless of operator config."""
    from charlie_work.config import RuntimeConfig

    custom_size = 7777
    config = OrchestratorConfig(runtime=RuntimeConfig(event_ring_size=custom_size))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    # Snapshot the module global before construction and restore it after so
    # the test does not leak the override into other tests in the same process.
    saved = state_module.EVENT_RING_SIZE
    try:
        OrchestratorApp(tmp_path, paths, config, FakeGitHub())
        assert state_module.EVENT_RING_SIZE == custom_size
    finally:
        state_module.EVENT_RING_SIZE = saved


def test_salvage_rework_stranded_commits_skips_when_status_not_rework_requested(
    tmp_path: Path,
) -> None:
    """Issue #1239 round-3: the death-loop escalation gate's salvage call site
    (``_salvage_rework_stranded_commits``) receives ``issue_entry`` from the
    ``head_check_state`` snapshot loaded at the top of ``dispatch_rework``'s
    candidate loop.  Between that snapshot and this call the issue's status may
    have already moved off ``rework_requested`` (e.g. a concurrent loop pass
    dispatched it, escalated it, or the issue was closed).  In that case the
    salvage push to the shared origin remote MUST NOT be attempted — an
    unaudited push for a stale/no-longer-rework_requested issue leaves no event
    trail if it succeeds.  A fresh ``status == "rework_requested"`` precondition
    check (short state_lock scope, before any network push) gates the salvage,
    mirroring the ``_reap_restore_rework_requested`` precondition (which checks
    ``status == "dispatched"`` because that lane handles already-dispatched
    workers).
    """
    from datetime import UTC, datetime

    from charlie_work.paths import resolved_layout
    from charlie_work.worktree import push_branch, worktree_path_for_branch

    remote, repo_root = _init_repo_with_remote_inline(tmp_path)
    branch = "agent/issue-123-fix-search"

    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )

    # Create a branch from main and a worktree at the expected orchestrator path.
    run(["git", "branch", branch])
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
        worker=WorkerRoleConfig(harness="command"),
    )
    layout = resolved_layout(config, repo_root)
    wt_path = worktree_path_for_branch(repo_root, branch, layout.worktrees)
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", str(wt_path), branch])

    # Push the branch so the remote branch exists at the PR head sha.
    ok, error = push_branch(repo_root, branch, worktree_path=wt_path)
    assert ok, error
    pr_head_sha = run(["git", "rev-parse", branch]).stdout.strip()

    # Add a stranded commit to the worktree — the salvage WOULD push this if
    # the precondition check were absent.
    (wt_path / "fix.txt").write_text("fixed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "fix.txt"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "completed rework (died before push)"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )

    paths = runtime_paths(repo_root, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.repo_root = repo_root
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]
            self.prs[0]["headRefOid"] = pr_head_sha

    fake_gh = ReworkGitHub()
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    paths.root.mkdir(parents=True, exist_ok=True)
    # The issue's status has ALREADY moved off "rework_requested" — a
    # concurrent loop pass dispatched it (status is now "dispatched").  This
    # is the head_check_state/state.json decoupling the round-3 review flagged:
    # the snapshot still says rework_requested, but state.json has advanced.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "dispatched",
            "redispatch_at": [now_iso, now_iso],
            "worker_death_at": [now_iso, now_iso],
            "branch_name": branch,
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": pr_head_sha,
        }
        save_state(paths.state_file, state)

    app = OrchestratorApp(repo_root, paths, config, fake_gh)

    # The snapshot issue_entry still says rework_requested (stale) — this is
    # what dispatch_rework's candidate loop would pass.  The fresh check inside
    # _salvage_rework_stranded_commits must catch that state.json has moved.
    stale_issue_entry = {
        "number": 123,
        "status": "rework_requested",
        "branch_name": branch,
    }
    pr_data = fake_gh.prs[0]

    result = app._salvage_rework_stranded_commits(123, pr_data, stale_issue_entry)

    # The salvage must report no push.
    assert result is False

    # The remote branch head MUST NOT have advanced — no salvage push.
    remote_sha = subprocess.run(
        ["git", "rev-parse", branch],
        cwd=remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert remote_sha == pr_head_sha, (
        f"salvage push attempted for a non-rework_requested issue: "
        f"remote {remote_sha} != pr head {pr_head_sha}"
    )

    # The issue status must be unchanged (still dispatched, not reset).
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"

    # No salvage event must have been recorded.
    events = state.get("events", [])
    salvage_events = [e for e in events if e.get("kind") == "rework_stranded_commits_salvaged"]
    assert len(salvage_events) == 0


def test_windowed_redispatch_at_handles_corrupted_state(tmp_path: Path) -> None:
    """_windowed_redispatch_at must not crash when redispatch_at is corrupted
    (e.g., a string instead of a list). A string value would cause
    list("abc") → ['a', 'b', 'c'] which crashes datetime.fromisoformat.
    """
    from charlie_work.workflow import _windowed_redispatch_at

    # String instead of list — must return empty, not crash
    entry = {"redispatch_at": "2024-01-01T00:00:00Z"}
    result = _windowed_redispatch_at(entry, window_minutes=240)
    assert result == []

    # None
    entry = {"redispatch_at": None}
    result = _windowed_redispatch_at(entry, window_minutes=240)
    assert result == []

    # Missing key
    entry = {}
    result = _windowed_redispatch_at(entry, window_minutes=240)
    assert result == []

    # List with non-string entries
    entry = {"redispatch_at": [123, None, "not-a-date"]}
    result = _windowed_redispatch_at(entry, window_minutes=240)
    assert result == []

    # Valid list with recent timestamp
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    entry = {"redispatch_at": [now_iso]}
    result = _windowed_redispatch_at(entry, window_minutes=240)
    assert result == [now_iso]

    # Valid list with old timestamp (outside window)
    old_iso = (datetime.now(UTC) - timedelta(hours=10)).isoformat().replace("+00:00", "Z")
    entry = {"redispatch_at": [old_iso]}
    result = _windowed_redispatch_at(entry, window_minutes=240)
    assert result == []


def test_stalled_review_throttled_rolls_back_attempt_count(monkeypatch, tmp_path: Path) -> None:
    """A throttled reviewer death in _detect_and_handle_stalled_reviews must
    not consume the per-PR review_dispatch_attempt_count. The reviewer hit a
    provider limit, not a PR-specific failure.
    """
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    reviews_dir = app._layout.reviews_dir

    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    _make_dead_review_sidecar(
        reviews_dir, 100, "Error: usage limit exceeded; please try again later"
    )
    _set_review_dispatched_state(app, 100, 10, old_dispatched)

    # Set attempt_count to 1 to simulate a prior dispatch
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"]["review_dispatch_attempt_count"] = 1
        save_state(app.paths.state_file, state)

    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir,
        app.paths.state_file,
        app.config,
        repo_root,
        write_gate=_wg(app.paths.state_file),
    )
    assert any(entry.get("pr") == 100 for entry in stalled)

    state = load_state(app.paths.state_file)
    # Attempt count should be 0 (rolled back from 1)
    assert state["prs"]["100"].get("review_dispatch_attempt_count", 0) == 0
    # Claim should be cleared (rolled back, not failed)
    assert state["prs"]["100"].get("review_dispatch_status") is None


# ---------------------------------------------------------------------------
# Issue #626: rescue-tier combined-manifest adapter labeling
# ---------------------------------------------------------------------------


def test_manifest_adapter_label_helper() -> None:
    """Issue #626: ``manifest_adapter_label`` is the single point of label
    derivation. One kind → that kind; more than one → ``"mixed"``."""
    from charlie_work.adapters import manifest_adapter_label

    assert manifest_adapter_label({"devin-shell"}) == "devin-shell"
    assert manifest_adapter_label({"api"}) == "api"
    assert manifest_adapter_label({"claude-code"}) == "claude-code"
    assert manifest_adapter_label({"api", "devin-shell"}) == "mixed"
    assert manifest_adapter_label({"api", "claude-code", "devin-shell"}) == "mixed"


def test_sink_census_counts_escalated_and_blocked_only(tmp_path: Path) -> None:
    """Issue #1083: ``sink_census`` reads the sink population from state.

    The sink is exactly the issues whose ``status`` is ``escalated`` or
    ``blocked`` -- the in-state mirror of the ``agent:human-needed`` label.
    Other terminal-ish statuses (``done``, ``merged``) and non-digit keys
    are excluded so the census matches the de-escalation sweep's own
    selection query.
    """
    state = {
        "issues": {
            "101": {"number": 101, "status": "escalated", "reason_class": "judgment"},
            "102": {"number": 102, "status": "blocked", "reason_class": "mechanical"},
            "103": {"number": 103, "status": "done"},
            "104": {"number": 104, "status": PASSIVE_OPEN_STATUS},
            "105": {"number": 105, "status": "rework_requested"},
            "not-an-issue": {"number": 0, "status": "escalated"},
        }
    }
    assert sink_census(state) == {101, 102}
    # An empty / malformed issues map is safe.
    assert sink_census({}) == set()
    assert sink_census({"issues": "not-a-dict"}) == set()


def test_loop_records_sink_metric_in_completed_event_and_pass_row(
    tmp_path: Path,
) -> None:
    """Issue #1083: a loop pass reports the sink metric alongside autonomy.

    Autonomy (merge_count/review_count) must never be reported without its
    drop rate. This test asserts the ``loop_completed`` event payload and the
    ``loop_passes`` row both carry ``sink_population``, ``sink_arrivals``,
    and ``sink_clears`` for the pass, with arrivals derived from a
    before/after census diff around ``_loop_body``.

    ``_loop_body`` is replaced with a stub that escalates one fresh issue
    mid-pass, so the pass observes one pre-existing parked issue (population
    before) plus one arrival (population after = 2, arrivals = 1). The stub
    emits no ``deescalation_cleared`` event, so ``sink_clears`` is 0.
    """
    from charlie_work.instrumentation import _get_db

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # One issue already parked in the sink before the pass.
    state = load_state(paths.state_file)
    state["issues"]["100"] = {
        "number": 100,
        "status": "escalated",
        "reason_class": "judgment",
    }
    save_state(paths.state_file, state)

    # Stub _loop_body to escalate a second issue during the pass and return a
    # clean CommandResult without touching GitHub. This isolates the sink
    # census diff (the issue #1083 measurement) from the rest of the pass.
    def stub_body(limit: int | None, *, merge: bool | None, now=None) -> CommandResult:
        mid = load_state(paths.state_file)
        mid["issues"]["200"] = {
            "number": 200,
            "status": "blocked",
            "reason_class": "judgment",
        }
        save_state(paths.state_file, mid)
        return CommandResult(
            ok=True, message="stub", data={"errors": [], "merges": [], "reviews": []}
        )

    app._loop_body = stub_body  # type: ignore[assignment]
    app.loop(limit=0)

    completed = query_events(paths.state_file, kind="loop_completed")
    assert completed, "loop_completed event was not emitted"
    payload = completed[-1]["payload"]
    assert payload["sink_population"] == 2
    assert payload["sink_arrivals"] == 1
    assert payload["sink_clears"] == 0

    # The loop_passes row carries the same metric in queryable columns.
    conn = _get_db(paths.state_file)
    assert conn is not None
    row = conn.execute(
        "SELECT sink_population, sink_arrivals, sink_clears FROM loop_passes "
        "ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["sink_population"] == 2
    assert row["sink_arrivals"] == 1
    assert row["sink_clears"] == 0


# ---------------------------------------------------------------------------
# Issue #1363 PART 2: preflight gate wiring into OrchestratorApp.loop()
#
# Both tests below monkeypatch ``charlie_work.workflow.run_preflight``
# directly with a canned ``PreflightResult`` rather than driving the real
# disk/clock/venv/config probes through ``tmp_path``. That is the correct
# abstraction layer for a *wiring* test: run_preflight's own check logic
# (disk_floor math, venv_identity path matching, config_freshness
# once-per-change semantics, ...) is already exhaustively covered at the
# unit level in test_preflight.py. What is untested until now is whether
# OrchestratorApp._loop_impl reacts to a PreflightResult correctly -- and
# canning the result also makes these tests immune to the ambient
# sys.executable/orchestrator_root() of whatever environment happens to run
# them (a real venv-synced checkout under CI, a PYTHONPATH-overridden
# worktree locally, ...), which the real venv_identity check is otherwise
# sensitive to.
# ---------------------------------------------------------------------------


def test_loop_fatal_preflight_refusal_skips_loop_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2: a fatal preflight failure must refuse the pass -- _loop_body must
    never run, a loop_refused_preflight event must be recorded, and loop()
    must return a non-ok CommandResult naming the failing check -- without
    a loop_completed event ever appearing."""
    from charlie_work.preflight import PreflightCheck, PreflightResult

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    fatal_check = PreflightCheck(
        name="disk_floor", ok=False, detail="0.10 GB free (floor 5 GB)", fatal=True
    )
    fake_result = PreflightResult(checks=(fatal_check,))
    monkeypatch.setattr("charlie_work.workflow.run_preflight", lambda *args, **kwargs: fake_result)

    body_invoked = False

    def stub_body(limit: int | None, *, merge: bool | None, now=None) -> CommandResult:
        nonlocal body_invoked
        body_invoked = True
        return CommandResult(ok=True, message="stub", data={})

    app._loop_body = stub_body  # type: ignore[assignment]
    result = app.loop(limit=0)

    assert body_invoked is False, "_loop_body ran despite a fatal preflight refusal"
    assert result.ok is False
    assert result.data.get("pass_skipped") is True
    assert result.data.get("reason") == "preflight_refused"
    assert result.data.get("check") == "disk_floor"

    refused = query_events(paths.state_file, kind="loop_refused_preflight")
    assert refused, "loop_refused_preflight event was not emitted"
    assert refused[-1]["payload"]["check"] == "disk_floor"

    completed = query_events(paths.state_file, kind="loop_completed")
    assert not completed, "loop_completed must not fire when preflight refuses the pass"


def test_loop_healthy_preflight_proceeds_with_no_extra_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC3: a fully-passing preflight run must add nothing beyond the
    ordinary loop event sequence -- no loop_refused_preflight, no
    preflight_warning/preflight_config_stale noise -- and loop_completed
    must still fire normally. This is the "healthy host" regression control
    for AC2: proof that the gate's presence is invisible on a clean pass."""
    from charlie_work.preflight import PreflightCheck, PreflightResult

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    all_ok_result = PreflightResult(
        checks=(
            PreflightCheck(name="disk_floor", ok=True, detail="ok", fatal=True),
            PreflightCheck(name="clock_sanity", ok=True, detail="ok", fatal=False),
            PreflightCheck(name="venv_identity", ok=True, detail="ok", fatal=True),
            PreflightCheck(name="config_freshness", ok=True, detail="ok", fatal=False),
        )
    )
    monkeypatch.setattr(
        "charlie_work.workflow.run_preflight", lambda *args, **kwargs: all_ok_result
    )

    def stub_body(limit: int | None, *, merge: bool | None, now=None) -> CommandResult:
        return CommandResult(
            ok=True, message="stub", data={"errors": [], "merges": [], "reviews": []}
        )

    app._loop_body = stub_body  # type: ignore[assignment]
    result = app.loop(limit=0)

    assert result.ok is True
    assert not query_events(paths.state_file, kind="loop_refused_preflight")
    assert not query_events(paths.state_file, kind="preflight_warning")
    assert not query_events(paths.state_file, kind="preflight_config_stale")
    completed = query_events(paths.state_file, kind="loop_completed")
    assert completed, "loop_completed event was not emitted on a healthy preflight pass"


# ---------------------------------------------------------------------------
# Issue #1248: orphan-sweep integration with salvage_push_stranded_commits
# ---------------------------------------------------------------------------
#
# These monkeypatch `charlie_work.workflow.salvage_push_stranded_commits`
# directly -- no real git, no network -- following the existing orphan-sweep
# test setup (state_file/sessions_dir/FakeGitHub) used above.


def test_cleanup_stale_session_tmp_files_removes_stranded_tmp(
    tmp_path: Path,
) -> None:
    """Issue #1393: cleanup_stale_session_tmp_files removes stranded .json.tmp
    files from the sessions directory (left behind by an interrupted atomic
    write) without touching the valid .json sidecar.

    The tmp files are aged past the ``min_age_seconds`` threshold so the sweep
    treats them as genuinely stranded rather than in-flight.
    """
    import os

    from charlie_work.adapters import cleanup_stale_session_tmp_files

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)

    # A valid sidecar.
    (sessions_dir / "issue-100.json").write_text('{"ok": true}', encoding="utf-8")
    # A stranded tmp from an interrupted write.
    (sessions_dir / "issue-100.json.tmp").write_text('{"ok": ', encoding="utf-8")
    # Another stranded tmp for a different issue.
    (sessions_dir / "issue-200.json.tmp").write_text('{"partial": ', encoding="utf-8")

    # Age both tmp files beyond the default 60s threshold so the sweep
    # treats them as stranded, not in-flight.
    stale_mtime = time.time() - 120
    os.utime(sessions_dir / "issue-100.json.tmp", (stale_mtime, stale_mtime))
    os.utime(sessions_dir / "issue-200.json.tmp", (stale_mtime, stale_mtime))

    removed = cleanup_stale_session_tmp_files(sessions_dir)

    assert removed == 2
    assert (sessions_dir / "issue-100.json").exists()
    assert not (sessions_dir / "issue-100.json.tmp").exists()
    assert not (sessions_dir / "issue-200.json.tmp").exists()


def test_cleanup_stale_session_tmp_files_skips_fresh_tmp(tmp_path: Path) -> None:
    """Issue #1393 regression: a freshly-created (not-yet-replaced) .json.tmp
    file must survive cleanup_stale_session_tmp_files so the sweep cannot race
    a legitimate in-flight atomic write between its close() and replace()
    calls — unlinking the tmp there crashes the writer with FileNotFoundError.
    """
    from charlie_work.adapters import cleanup_stale_session_tmp_files

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)

    # A tmp file a legitimate writer just created and closed but has not yet
    # replaced — its mtime is "now", well within the 60s grace window.
    fresh = sessions_dir / "issue-300.json.tmp"
    fresh.write_text('{"in-flight": ', encoding="utf-8")

    removed = cleanup_stale_session_tmp_files(sessions_dir)

    assert removed == 0
    assert fresh.exists()
    assert fresh.read_text(encoding="utf-8") == '{"in-flight": '


def test_cleanup_stale_session_tmp_files_missing_dir(tmp_path: Path) -> None:
    """cleanup_stale_session_tmp_files is a no-op when the sessions dir
    does not exist (e.g. first-ever dispatch pass)."""
    from charlie_work.adapters import cleanup_stale_session_tmp_files

    removed = cleanup_stale_session_tmp_files(tmp_path / "nonexistent")
    assert removed == 0
