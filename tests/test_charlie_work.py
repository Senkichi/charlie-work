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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from _dead_session_fixtures import _git
from _dispatch_fixtures import _cross_repo_issue_body, _fail_if_launched
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub, FakeGitHubWithMissingRequired
from _helpers import EXAMPLES_DIR, _init_git_repo
from _merge_tripwire_fixtures import _arm_unauthorized_merge_tripwire
from _review_fixtures import (
    _approved_automerge,
    _dispatch_reviews_app,
    _fake_claude_worker_record,
    _make_dead_review_sidecar,
    _required_checks_config,
    _review_queue_carry_forward_app,
    _set_review_dispatched_state,
    _write_review_packet,
)
from _rework_dispatch_fixtures import _init_repo_with_remote_inline, _wg
from _worktree_fixtures import _init_bare_remote_and_clone, _setup_completed_worktree
from charlie_work import cli
from charlie_work import github as github_module
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import (
    AutoMergeConfig,
    ClaudeCodeConfig,
    ConfigError,
    DevinConfig,
    DispatchConfig,
    LabelConfig,
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
from charlie_work.devin_shell import SessionRecord
from charlie_work.github import issue_numbers_mentioned_by_pr, label_names
from charlie_work.instrumentation import query_events
from charlie_work.markdown_fence import fenced_block
from charlie_work.paths import resolved_layout, runtime_paths
from charlie_work.prompt_test_command import prompt_test_command_values
from charlie_work.prompts import render_prompt
from charlie_work.rescue_review import _CAVEAT, extract_report_body, report_body_is_valid
from charlie_work.review_decision import ReviewDecision
from charlie_work.state import (
    PASSIVE_OPEN_STATUS,
    empty_state,
    is_throttled,
    load_state,
    save_state,
    state_lock,
)
from charlie_work.verdict_parsing import REVIEW_SESSION_SUMMARY_HEADING
from charlie_work.workflow import (
    ORCHESTRATOR_COMMENT_MARKER,
    OrchestratorApp,
    _detect_and_handle_stalled_reviews,
    _parse_review_verdict_from_log,
    _summary_is_vacuous,
    sink_census,
    slugify,
)
from charlie_work.worktree import create_worktree
import charlie_work.state as state_module


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
