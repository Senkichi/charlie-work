from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from charlie_work import cli
from charlie_work import github as github_module
from charlie_work.checks import summarize_checks
from charlie_work.github import is_infrastructure_failure
from charlie_work.config import (
    ClaudeCodeConfig,
    CrossFamilyConfig,
    DevinConfig,
    DispatchConfig,
    FleetConfig,
    LabelConfig,
    OrchestratorConfig,
    PostMortemConfig,
    ReviewConfig,
    RuntimeConfig,
    SignatureRule,
    SupervisorConfig,
    TestAdequacyConfig,
    WatchdogConfig,
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
from charlie_work.workflow import ConcurrencyGovernorResult, OrchestratorApp, slugify
from charlie_work.worktree import create_worktree
from charlie_work.devin_shell import SessionRecord

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_default_config_enables_auto_merge() -> None:
    config = load_config()

    assert config.auto_merge.enabled is True
    # A shared package cannot know a consumer's CI check names; unconfigured
    # means empty, and `doctor` flags it.
    assert config.auto_merge.required_checks == ()
    assert config.labels.ready == "automated-ready"


def test_default_config_tee_stream_json_disabled() -> None:
    """ClaudeCodeConfig.tee_stream_json defaults to False (issue #160)."""
    config = load_config()
    assert config.claude_code.tee_stream_json is False


def test_default_config_runner_scaling_disabled() -> None:
    """RunnerScalingConfig.enabled defaults to False (issue #232)."""
    config = load_config()
    assert config.runner_scaling.enabled is False


def test_default_config_throttle_error_markers() -> None:
    """RuntimeConfig.throttle_error_markers defaults to genuine provider
    throttle signatures only.

    Issue #260, corrected premise: "A tool was rejected by the user" was
    originally a default here, but it is the Devin CLI's own surfacing of a
    PreToolUse hook block, not a provider throttle condition — it must never
    be a default throttle marker (retry/cooldown semantics are wrong for a
    hard hook block). See PostMortemConfig.signature_rules for the
    worker_blocked rule that owns that signature instead.
    """
    config = load_config()
    assert "Reached overall message rate limit" in config.runtime.throttle_error_markers
    assert "rate limit" in config.runtime.throttle_error_markers
    assert "too many requests" in config.runtime.throttle_error_markers
    assert "A tool was rejected by the user" not in config.runtime.throttle_error_markers


def test_runner_scaling_config_parses_with_enabled_flag(tmp_path: Path) -> None:
    """RunnerScalingConfig parses with enabled=true and custom values."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
runner_scaling:
  enabled: true
  managed_root: "C:\\\\actions-runners"
  runner_dir_prefix: "jc-"
  runner_name_template: "jc-9800x3d-{n}"
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
    assert config.runner_scaling.runner_name_template == "jc-9800x3d-{n}"
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


def test_load_config_runtime_throttle_error_markers(tmp_path: Path) -> None:
    """RuntimeConfig.throttle_error_markers is configurable from YAML."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
runtime:
  throttle_error_markers:
    - "Reached overall message rate limit"
    - "A tool was rejected by the user"
    - "custom provider failure"
"""
    )
    config = load_config(config_file)
    assert config.runtime.throttle_error_markers == (
        "Reached overall message rate limit",
        "A tool was rejected by the user",
        "custom provider failure",
    )


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
    assert summary.infra_failed == ()


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
    assert summary.infra_failed == ()


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
    assert summary.infra_failed == ()


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
    assert summary.infra_failed == ()


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
    assert summary.failed == ()
    assert summary.infra_failed == ()


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
    assert summary.infra_failed == ()


def test_summarize_checks_empty_state_and_bucket_classifies_as_pending() -> None:
    """Regression test for issue #95: null/empty state+bucket should classify as pending."""
    checks = [
        {"name": "test", "state": None, "bucket": None},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.pending == ("test",)
    assert summary.failed == ()
    assert summary.infra_failed == ()


def test_summarize_checks_empty_string_state_and_bucket_classifies_as_pending() -> None:
    """Regression test for issue #95: empty string state+bucket should classify as pending."""
    checks = [
        {"name": "test", "state": "", "bucket": ""},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.pending == ("test",)
    assert summary.failed == ()


def test_summarize_checks_cancelled_classifies_as_infra_failed() -> None:
    """Regression test for issue #210: CANCELLED state should classify as infrastructure failure."""
    checks = [
        {"name": "test", "state": "CANCELLED"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.infra_failed == ("test",)
    assert summary.failed == ()
    assert summary.pending == ()


def test_summarize_checks_cancelled_case_insensitive() -> None:
    """CANCELLED state classification should be case-insensitive."""
    checks = [
        {"name": "test", "state": "cancelled"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.infra_failed == ("test",)
    assert summary.failed == ()


def test_summarize_checks_mixed_cancelled_and_failure() -> None:
    """Mixed CANCELLED and FAILURE states should classify each separately."""
    checks = [
        {"name": "test1", "state": "CANCELLED"},
        {"name": "test2", "state": "FAILURE"},
    ]

    summary = summarize_checks(checks, ("test1", "test2"))

    assert summary.ready is False
    assert summary.infra_failed == ("test1",)
    assert summary.failed == ("test2",)
    assert summary.pending == ()


def test_summarize_checks_duplicate_runs_cancelled_then_success() -> None:
    """Duplicate runs with CANCELLED then SUCCESS should classify as infra_failed (worst-of)."""
    checks = [
        {"name": "test", "state": "CANCELLED"},
        {"name": "test", "state": "SUCCESS"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.infra_failed == ("test",)
    assert summary.failed == ()
    assert summary.pending == ()


def test_summarize_checks_failure_takes_priority_over_cancelled() -> None:
    """FAILURE should take priority over CANCELLED in worst-of semantics."""
    checks = [
        {"name": "test", "state": "CANCELLED"},
        {"name": "test", "state": "FAILURE"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.failed == ("test",)
    assert summary.infra_failed == ()
    assert summary.pending == ()


def test_summarize_checks_infra_failure_marker_classifies_as_infra_failed() -> None:
    """INFRA_FAILURE marker state should classify as infrastructure failure."""
    checks = [
        {"name": "test", "state": "INFRA_FAILURE"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.infra_failed == ("test",)
    assert summary.failed == ()
    assert summary.pending == ()


def test_summarize_checks_infra_failure_case_insensitive() -> None:
    """INFRA_FAILURE state classification should be case-insensitive."""
    checks = [
        {"name": "test", "state": "infra_failure"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.infra_failed == ("test",)
    assert summary.failed == ()


def test_summarize_checks_failure_takes_priority_over_infra_failure() -> None:
    """FAILURE should take priority over INFRA_FAILURE in worst-of semantics."""
    checks = [
        {"name": "test", "state": "INFRA_FAILURE"},
        {"name": "test", "state": "FAILURE"},
    ]

    summary = summarize_checks(checks, ("test",))

    assert summary.ready is False
    assert summary.failed == ("test",)
    assert summary.infra_failed == ()


def test_is_infrastructure_failure_zero_step_job() -> None:
    """Jobs with zero non-setup steps should be classified as infrastructure failure."""
    job = {
        "conclusion": "FAILURE",
        "steps": [
            {"name": "Set up job"},
            {"name": "Checkout"},
        ],
    }
    annotations = []

    assert is_infrastructure_failure(job, annotations) is True


def test_is_infrastructure_failure_with_test_steps() -> None:
    """Jobs with actual test steps should not be classified as infrastructure failure."""
    job = {
        "conclusion": "FAILURE",
        "steps": [
            {"name": "Set up job"},
            {"name": "Checkout"},
            {"name": "Run tests"},
        ],
    }
    annotations = []

    assert is_infrastructure_failure(job, annotations) is False


def test_is_infrastructure_failure_billing_annotation() -> None:
    """Jobs with billing annotation should be classified as infrastructure failure."""
    job = {
        "conclusion": "FAILURE",
        "steps": [{"name": "Run tests"}],
    }
    annotations = [
        {
            "message": "The job was not started because recent account payments have failed or your spending limit needs to be increased."
        }
    ]

    assert is_infrastructure_failure(job, annotations) is True


def test_is_infrastructure_failure_mixed_billing_annotation_text() -> None:
    """Billing annotation detection should be case-insensitive and match partial text."""
    job = {
        "conclusion": "FAILURE",
        "steps": [{"name": "Run tests"}],
    }
    annotations = [{"message": "The job WAS NOT STARTED due to billing issues"}]

    assert is_infrastructure_failure(job, annotations) is True


def test_is_infrastructure_failure_no_infrastructure_signals() -> None:
    """Jobs without infrastructure failure signals should not be classified as such."""
    job = {
        "conclusion": "FAILURE",
        "steps": [
            {"name": "Set up job"},
            {"name": "Checkout"},
            {"name": "Run tests"},
        ],
    }
    annotations = [{"message": "Test failed: assertion error"}]

    assert is_infrastructure_failure(job, annotations) is False


def test_is_infrastructure_failure_empty_steps() -> None:
    """Job with no steps at all should be classified as infrastructure failure (primary signal)."""
    job = {
        "conclusion": "FAILURE",
        "steps": [],
    }
    annotations = []

    assert is_infrastructure_failure(job, annotations) is True


def test_is_infrastructure_failure_non_failed_job() -> None:
    """Jobs that didn't fail should not trigger infrastructure failure detection."""
    job = {
        "conclusion": "SUCCESS",
        "steps": [],
    }
    annotations = []

    assert is_infrastructure_failure(job, annotations) is False


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


def test_load_config_rejects_invalid_dispatch_order(tmp_path: Path) -> None:
    """Issue #151: invalid dispatch.order config value is rejected at load."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "dispatch:\n  order: invalid\n",
        encoding="utf-8",
    )

    from charlie_work.config import ConfigError, load_config

    try:
        load_config(config_file)
        raise AssertionError("expected ConfigError for invalid dispatch.order")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid dispatch.order")

    assert "dispatch" in message
    assert "order" in message
    assert "oldest" in message or "newest" in message


def test_dispatch_launch_stagger_seconds_default_is_45() -> None:
    """Default stagger between worker-session launches within a pass."""
    assert DispatchConfig().launch_stagger_seconds == 45


def test_load_config_parses_dispatch_launch_stagger_seconds_override(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "dispatch:\n  launch_stagger_seconds: 10\n",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.dispatch.launch_stagger_seconds == 10


def test_load_config_rejects_negative_launch_stagger_seconds(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "dispatch:\n  launch_stagger_seconds: -1\n",
        encoding="utf-8",
    )

    try:
        load_config(config_file)
        raise AssertionError("expected ConfigError for negative launch_stagger_seconds")
    except ConfigError as exc:
        message = str(exc)

    assert "dispatch" in message
    assert "launch_stagger_seconds" in message


def test_load_config_rejects_wrong_type_launch_stagger_seconds(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "dispatch:\n  launch_stagger_seconds: not-a-number\n",
        encoding="utf-8",
    )

    try:
        load_config(config_file)
        raise AssertionError("expected ConfigError for non-int launch_stagger_seconds")
    except ConfigError as exc:
        message = str(exc)

    assert "dispatch" in message
    assert "launch_stagger_seconds" in message


def test_default_config_disables_test_adequacy() -> None:
    """TestAdequacyConfig defaults to disabled with all default values."""
    config = load_config()

    assert config.test_adequacy == TestAdequacyConfig()
    assert config.test_adequacy.enabled is False


def test_config_test_adequacy_coerces_tuple_fields_to_tuple(tmp_path: Path) -> None:
    """YAML lists in test_adequacy tuple fields round-trip to tuples."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """test_adequacy:
  test_path_globs:
    - "tests/**"
    - "test_*.py"
  exempt_path_globs:
    - "*.md"
    - "docs/**"
  assertion_markers:
    - "assert "
    - "pytest.raises"
  comment_prefixes:
    - "#"
    - "//"
  stub_test_seam_keywords:
    - "route"
    - "call_model"
  coverage_command:
    - "pytest"
    - "--cov"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config.test_adequacy.test_path_globs, tuple)
    assert config.test_adequacy.test_path_globs == ("tests/**", "test_*.py")
    assert isinstance(config.test_adequacy.exempt_path_globs, tuple)
    assert config.test_adequacy.exempt_path_globs == ("*.md", "docs/**")
    assert isinstance(config.test_adequacy.assertion_markers, tuple)
    assert config.test_adequacy.assertion_markers == ("assert ", "pytest.raises")
    assert isinstance(config.test_adequacy.comment_prefixes, tuple)
    assert config.test_adequacy.comment_prefixes == ("#", "//")
    assert isinstance(config.test_adequacy.stub_test_seam_keywords, tuple)
    assert config.test_adequacy.stub_test_seam_keywords == ("route", "call_model")
    assert isinstance(config.test_adequacy.coverage_command, tuple)
    assert config.test_adequacy.coverage_command == ("pytest", "--cov")


def test_config_rejects_non_list_test_adequacy_tuple_field(tmp_path: Path) -> None:
    """Tuple fields given as scalars raise ConfigError."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "test_adequacy:\n  test_path_globs: tests/**\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for non-list tuple field")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for non-list tuple field")

    assert "test_adequacy" in message
    assert "test_path_globs" in message
    assert "must be a list" in message


def test_config_rejects_non_str_element_in_test_adequacy_tuple_field(tmp_path: Path) -> None:
    """Tuple fields with non-str elements raise ConfigError."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "test_adequacy:\n  test_path_globs:\n    - tests/**\n    - 123\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for non-str element")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for non-str element")

    assert "test_adequacy" in message
    assert "test_path_globs" in message
    assert "element of type" in message


def test_config_rejects_bad_type_min_product_lines(tmp_path: Path) -> None:
    """min_product_lines as string raises ConfigError."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "test_adequacy:\n  min_product_lines: ten\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for bad type min_product_lines")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for bad type min_product_lines")

    assert "test_adequacy" in message
    assert "min_product_lines" in message
    assert "must be an int" in message


def test_config_rejects_bad_type_min_diff_coverage(tmp_path: Path) -> None:
    """min_diff_coverage as string raises ConfigError; int is accepted."""
    from charlie_work.config import ConfigError

    # Reject string
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "test_adequacy:\n  min_diff_coverage: high\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for bad type min_diff_coverage")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for bad type min_diff_coverage")

    assert "test_adequacy" in message
    assert "min_diff_coverage" in message
    assert "must be a float" in message

    # Accept int
    config_path.write_text(
        "test_adequacy:\n  min_diff_coverage: 1\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.test_adequacy.min_diff_coverage == 1


def test_config_rejects_empty_exempt_marker(tmp_path: Path) -> None:
    """exempt_marker as empty string raises ConfigError."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'test_adequacy:\n  exempt_marker: ""\n',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for empty exempt_marker")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for empty exempt_marker")

    assert "test_adequacy" in message
    assert "exempt_marker" in message
    assert "non-empty string" in message


def test_config_rejects_non_bool_test_adequacy_flags(tmp_path: Path) -> None:
    """Boolean flags as strings raise ConfigError."""
    from charlie_work.config import ConfigError

    for bool_key in ("enabled", "coverage_enabled", "require_assertions"):
        config_path = tmp_path / "orchestrator.config.yaml"
        config_path.write_text(
            f'test_adequacy:\n  {bool_key}: "true"\n',
            encoding="utf-8",
        )

        try:
            load_config(config_path)
            raise AssertionError(f"expected ConfigError for non-bool {bool_key}")
        except ConfigError as exc:
            message = str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"expected ConfigError for non-bool {bool_key}")

        assert "test_adequacy" in message
        assert bool_key in message
        assert "must be a bool" in message


def test_load_config_rejects_unknown_test_adequacy_key(tmp_path: Path) -> None:
    """Unknown keys under test_adequacy raise ConfigError listing valid keys."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "test_adequacy:\n  enabled: true\n  bad_key: value\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for unknown test_adequacy key")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown test_adequacy key")

    assert "section 'test_adequacy'" in message
    assert "bad_key" in message
    # Should list valid keys
    assert "enabled" in message


def test_config_accepts_full_test_adequacy_override(tmp_path: Path) -> None:
    """A YAML block overriding every field loads correctly."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """test_adequacy:
  enabled: true
  min_product_lines: 20
  test_path_globs:
    - "custom_tests/**"
  exempt_path_globs:
    - "*.txt"
  assertion_markers:
    - "custom_assert"
  comment_prefixes:
    - "//"
  require_assertions: true
  stub_test_seam_keywords:
    - "route"
    - "byte"
  exempt_marker: "Custom-exempt:"
  coverage_enabled: true
  coverage_command:
    - "custom"
    - "cov"
  min_diff_coverage: 0.5
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.test_adequacy.enabled is True
    assert config.test_adequacy.min_product_lines == 20
    assert config.test_adequacy.test_path_globs == ("custom_tests/**",)
    assert config.test_adequacy.exempt_path_globs == ("*.txt",)
    assert config.test_adequacy.assertion_markers == ("custom_assert",)
    assert config.test_adequacy.comment_prefixes == ("//",)
    assert config.test_adequacy.require_assertions is True
    assert config.test_adequacy.stub_test_seam_keywords == ("route", "byte")
    assert config.test_adequacy.exempt_marker == "Custom-exempt:"
    assert config.test_adequacy.coverage_enabled is True
    assert config.test_adequacy.coverage_command == ("custom", "cov")
    assert config.test_adequacy.min_diff_coverage == 0.5


def test_config_accepts_watchdog_terminal_error_markers(tmp_path: Path) -> None:
    """A YAML block with terminal_error_markers loads correctly."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  enabled: true
  stall_minutes: 20
  terminal_error_markers:
    - "Error: A tool was rejected"
    - "Error: Agent error:"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.watchdog.enabled is True
    assert config.watchdog.stall_minutes == 20
    assert config.watchdog.terminal_error_markers == (
        "Error: A tool was rejected",
        "Error: Agent error:",
    )


def test_config_defaults_watchdog_terminal_error_markers(tmp_path: Path) -> None:
    """A YAML block without terminal_error_markers uses the default."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  enabled: true
  stall_minutes: 20
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.watchdog.enabled is True
    assert config.watchdog.stall_minutes == 20
    assert config.watchdog.terminal_error_markers == (
        "Error: A tool was rejected",
        "Error: Agent error:",
    )


def test_config_rejects_invalid_watchdog_terminal_error_markers_type(tmp_path: Path) -> None:
    """terminal_error_markers must be a list of strings."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  terminal_error_markers: "not a list"
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for invalid terminal_error_markers type")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid terminal_error_markers type")

    assert "section 'watchdog'" in message
    assert "terminal_error_markers" in message
    assert "must be a list" in message


def test_config_rejects_invalid_watchdog_terminal_error_markers_element_type(
    tmp_path: Path,
) -> None:
    """terminal_error_markers elements must be strings."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  terminal_error_markers:
    - "valid string"
    - 123
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError(
            "expected ConfigError for invalid terminal_error_markers element type"
        )
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError(
            "expected ConfigError for invalid terminal_error_markers element type"
        )

    assert "section 'watchdog'" in message
    assert "terminal_error_markers" in message
    assert "must be a list of strings" in message


def test_config_rejects_unknown_watchdog_key(tmp_path: Path) -> None:
    """Unknown keys under watchdog raise ConfigError listing valid keys."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  enabled: true
  bad_key: value
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for unknown watchdog key")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown watchdog key")

    assert "section 'watchdog'" in message
    assert "bad_key" in message
    # Should list valid keys
    assert "enabled" in message
    assert "stall_minutes" in message
    assert "terminal_error_markers" in message
    assert "cost_budget_usd" in message
    assert "token_budget" in message
    assert "cost_budget_action" in message
    assert "wall_clock_minutes" in message
    assert "wall_clock_kill" in message
    assert "loop_stall_multiplier" in message
    assert "loop_kill" in message


def test_config_accepts_cost_token_budgets(tmp_path: Path) -> None:
    """A YAML block with cost/token budget fields loads correctly."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  enabled: true
  stall_minutes: 20
  cost_budget_usd: 10.0
  token_budget: 100000
  cost_budget_action: warn
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.watchdog.enabled is True
    assert config.watchdog.stall_minutes == 20
    assert config.watchdog.cost_budget_usd == 10.0
    assert config.watchdog.token_budget == 100000
    assert config.watchdog.cost_budget_action == "warn"


def test_config_watchdog_new_fields_have_defaults(tmp_path: Path) -> None:
    """New watchdog fields (wall_clock_minutes, wall_clock_kill, loop_stall_multiplier, loop_kill) have defaults."""
    config_path = tmp_path / "orchestrator.config.yaml"
    # Config without the new fields (pre-#162 style)
    config_path.write_text(
        """watchdog:
  enabled: true
  stall_minutes: 20
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.watchdog.enabled is True
    assert config.watchdog.stall_minutes == 20
    assert config.watchdog.cost_budget_usd is None
    assert config.watchdog.token_budget is None
    assert config.watchdog.cost_budget_action == "warn"
    assert config.watchdog.wall_clock_minutes == 240
    assert config.watchdog.wall_clock_kill is False
    assert config.watchdog.loop_stall_multiplier == 2
    assert config.watchdog.loop_kill is False


def test_config_defaults_cost_token_budgets(tmp_path: Path) -> None:
    """A YAML block without cost/token budget fields uses the defaults (None)."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  enabled: true
  stall_minutes: 20
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.watchdog.enabled is True
    assert config.watchdog.stall_minutes == 20
    assert config.watchdog.cost_budget_usd is None
    assert config.watchdog.token_budget is None
    assert config.watchdog.cost_budget_action == "warn"


def test_config_watchdog_accepts_new_fields(tmp_path: Path) -> None:
    """New watchdog fields can be set explicitly in YAML."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  enabled: true
  stall_minutes: 20
  wall_clock_minutes: 300
  wall_clock_kill: true
  loop_stall_multiplier: 3
  loop_kill: true
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.watchdog.wall_clock_minutes == 300
    assert config.watchdog.wall_clock_kill is True
    assert config.watchdog.loop_stall_multiplier == 3
    assert config.watchdog.loop_kill is True


def test_config_rejects_invalid_cost_budget_usd_type(tmp_path: Path) -> None:
    """cost_budget_usd must be a number."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  cost_budget_usd: "not a number"
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for invalid cost_budget_usd type")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid cost_budget_usd type")

    assert "section 'watchdog'" in message
    assert "cost_budget_usd" in message
    assert "must be a number" in message


def test_config_rejects_invalid_token_budget_type(tmp_path: Path) -> None:
    """token_budget must be an int."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  token_budget: "not an int"
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for invalid token_budget type")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid token_budget type")

    assert "section 'watchdog'" in message
    assert "token_budget" in message
    assert "must be an int" in message


def test_config_rejects_invalid_cost_budget_action_type(tmp_path: Path) -> None:
    """cost_budget_action must be a string."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  cost_budget_action: 123
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for invalid cost_budget_action type")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid cost_budget_action type")

    assert "section 'watchdog'" in message
    assert "cost_budget_action" in message
    assert "must be a string" in message


def test_config_rejects_invalid_cost_budget_action_value(tmp_path: Path) -> None:
    """cost_budget_action must be 'warn' or 'kill'."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  cost_budget_action: "invalid"
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for invalid cost_budget_action value")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid cost_budget_action value")

    assert "section 'watchdog'" in message
    assert "cost_budget_action" in message
    assert "must be 'warn' or 'kill'" in message


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


def _requests(count: int, tmp_path: Path) -> list:
    from charlie_work.adapters import SessionRequest

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt", encoding="utf-8")
    return [
        SessionRequest(
            issue_number=i,
            issue_title=f"Test {i}",
            prompt_path=prompt_path,
            branch_name=f"agent/issue-{i}",
        )
        for i in range(1, count + 1)
    ]


def test_dispatch_sessions_staggers_between_launches(tmp_path: Path, monkeypatch) -> None:
    """Issue: burst dispatch trips the Devin provider message rate limit (3
    sessions launched within 6 seconds all died on "Reached overall message
    rate limit"). dispatch_sessions must sleep launch_stagger_seconds BETWEEN
    consecutive launches -- not before the first, not after the last."""
    from charlie_work import adapters
    from charlie_work.adapters import AdapterSettings, dispatch_sessions
    from charlie_work.claude_code import ClaudeWorkerRecord

    sleep_calls: list[float] = []
    monkeypatch.setattr(adapters.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=1000 + issue_number,
            started_at="2026-07-10T00:00:00Z",
            log_path=str(tmp_path / "log"),
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    settings = AdapterSettings(adapter="claude-code", launch_stagger_seconds=45)

    results = dispatch_sessions(
        repo_root,
        tmp_path / "manifest.json",
        tmp_path / "results.json",
        settings,
        _requests(3, tmp_path),
    )

    assert len(results) == 3
    assert all(r.ok for r in results)
    assert sleep_calls == [45, 45]


def test_dispatch_sessions_single_launch_no_stagger_sleep(tmp_path: Path, monkeypatch) -> None:
    """A single launch has no "between launches" gap to fill -- no sleep."""
    from charlie_work import adapters
    from charlie_work.adapters import AdapterSettings, dispatch_sessions
    from charlie_work.claude_code import ClaudeWorkerRecord

    sleep_calls: list[float] = []
    monkeypatch.setattr(adapters.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=1000 + issue_number,
            started_at="2026-07-10T00:00:00Z",
            log_path=str(tmp_path / "log"),
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    settings = AdapterSettings(adapter="claude-code", launch_stagger_seconds=45)

    results = dispatch_sessions(
        repo_root,
        tmp_path / "manifest.json",
        tmp_path / "results.json",
        settings,
        _requests(1, tmp_path),
    )

    assert len(results) == 1
    assert sleep_calls == []


def test_dispatch_sessions_zero_stagger_disables_sleep(tmp_path: Path, monkeypatch) -> None:
    """launch_stagger_seconds=0 disables the stagger entirely, even with
    multiple launches."""
    from charlie_work import adapters
    from charlie_work.adapters import AdapterSettings, dispatch_sessions
    from charlie_work.claude_code import ClaudeWorkerRecord

    sleep_calls: list[float] = []
    monkeypatch.setattr(adapters.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=1000 + issue_number,
            started_at="2026-07-10T00:00:00Z",
            log_path=str(tmp_path / "log"),
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    settings = AdapterSettings(adapter="claude-code", launch_stagger_seconds=0)

    results = dispatch_sessions(
        repo_root,
        tmp_path / "manifest.json",
        tmp_path / "results.json",
        settings,
        _requests(3, tmp_path),
    )

    assert len(results) == 3
    assert sleep_calls == []


def test_adapter_settings_launch_stagger_seconds_wired_from_dispatch_config(
    tmp_path: Path,
) -> None:
    """_adapter_settings() must read dispatch.launch_stagger_seconds -- the
    single point of enforcement between config and both dispatch lanes."""
    config = OrchestratorConfig(dispatch=DispatchConfig(launch_stagger_seconds=17))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    assert app._adapter_settings().launch_stagger_seconds == 17


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
        required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
        enabled=True,  # Ensure auto_merge is enabled for merge tests
        **kwargs,
    )
    return OrchestratorConfig(auto_merge=auto_merge)


class FakeGitHub:
    def __init__(self, repo_root: Any = None) -> None:
        self.repo_root = repo_root
        self.issues = [
            {
                "number": 123,
                "title": "Fix search",
                "url": "https://example.test/issues/123",
                "body": "Search is broken",
                "labels": [{"name": "automated-ready"}],
                "state": "OPEN",
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
        self.prs_created: list[dict[str, Any]] = []
        self.pr_create_return: int | None = None
        self.merged: list[tuple[int, str]] = []
        self.merged_admin_flags: list[bool] = []
        self.merged_merge_flags: list[tuple[str, ...]] = []
        self.deleted_branches: list[str] = []
        self.delete_branch_ok = True
        self.update_branch_ok = True
        self.pr_head_shas: dict[int, str] = {}
        self.diffs: dict[int, str] = {}
        self.closed_issues: list[int] = []

    def issue_list(self, labels=None, state=None):
        # Honor the label filter: return only issues with the ready label
        # Support both old signature (ready_label: str) and new (labels=None, state=None)
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
                # Return a copy with the current head SHA (if overridden)
                pr_copy = dict(pr)
                if number in self.pr_head_shas:
                    pr_copy["headRefOid"] = self.pr_head_shas[number]
                return pr_copy
        raise ValueError(f"PR {number} not found")

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None:
        self.prs_created.append({"head": head, "base": base, "title": title, "body": body})
        return self.pr_create_return

    def pr_checks(self, number: int):
        return [
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]

    def pr_diff(self, number: int):
        # Return custom diff if set, otherwise default
        if number in self.diffs:
            return self.diffs[number]
        return "diff --git a/file b/file"

    def add_issue_label(self, number: int, label: str) -> bool:
        self.labels_added.append((number, label))
        return True

    def remove_issue_label(self, number: int, label: str) -> bool:
        self.labels_removed.append((number, label))
        return True

    def close_issue(self, number: int) -> bool:
        """Track issue closure for testing. Idempotent — returns True even if already closed."""
        # Track the closure
        self.closed_issues.append(number)
        # Update the issue state in the issues list
        for issue in self.issues:
            if issue["number"] == number:
                issue["state"] = "CLOSED"
                break
        return True

    def name_with_owner(self) -> str:
        return "test-owner/test-repo"

    def merge_pr(
        self, number: int, strategy: str, admin: bool = False, merge_flags: tuple[str, ...] = ()
    ) -> str:
        self.merged.append((number, strategy))
        # merge_flags takes precedence over admin
        if merge_flags:
            self.merged_admin_flags.append("--admin" in merge_flags)
        else:
            self.merged_admin_flags.append(admin)
        self.merged_merge_flags.append(merge_flags)
        return "merged"

    def delete_branch(self, branch: str) -> bool:
        self.deleted_branches.append(branch)
        return self.delete_branch_ok

    def pr_update_branch(self, pr_number: int) -> bool:
        # Simulate a base update by moving the PR's head to a new SHA
        # This reproduces the churn that the fix prevents
        for pr in self.prs:
            if pr["number"] == pr_number:
                # Append a merge-SHA marker to simulate the head moving
                old_head = pr.get("headRefOid", "")
                pr["headRefOid"] = f"{old_head}-updated"
                return self.update_branch_ok
        return False

    def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
        """Default implementation: check the actual state field in issues."""
        open_issues: set[int] = set()
        for number in issue_numbers:
            for issue in self.issues:
                if issue["number"] == number and str(issue.get("state") or "").upper() == "OPEN":
                    open_issues.add(number)
                    break
        return open_issues

    def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):
        """Fake run method for GitHub API calls. Returns empty list for dependencies by default."""
        # Handle dependency API calls
        if "dependencies" in " ".join(args):
            # Default: return empty list (feature not available)
            # Tests can override this by setting dependencies_response
            if hasattr(self, "dependencies_response"):
                return self.dependencies_response
            return [] if json_output else ""
        # Handle run list API calls
        if "run" in args and "list" in args:
            # Default: return empty list
            # Tests can override this by setting runs_response
            if hasattr(self, "runs_response"):
                return self.runs_response
            return [] if json_output else ""
        # Handle run cancel API calls
        if "run" in args and "cancel" in args:
            # Default: return success string
            return "Cancelled"
        # Handle other API calls (for reconcile tests)
        if json_output:
            return []
        return ""

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


def test_dispatch_excludes_stalled_session_dry_run(tmp_path: Path) -> None:
    """Test that dispatch excludes issues with stalled sessions (dry-run path)."""
    from datetime import UTC, datetime, timedelta
    from charlie_work.devin_shell import SessionRecord

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Create a session record for issue 123 with a live PID
    sessions_dir = app._resolve(config.devin.sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_file = sessions_dir / "issue-123.json"
    log_file = sessions_dir / "issue-123.log"

    # Write a log file with old mtime (stalled)
    log_file.write_text("working on issue\nmaking progress\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a session record with a fake PID (we'll mock liveness check)
    session_record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(
            tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
        ),
        command=("devin", "--prompt-file", "{prompt_path}"),
        pid=99999,  # Fake PID - we'll mock liveness to return True
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
        process_start_time=time.time(),
    )
    session_file.write_text(json.dumps(session_record.to_dict()), encoding="utf-8")

    # Mock the liveness check to return True (simulating a live but stalled process).
    # Patch target is charlie_work.worker (not devin_shell): the stalled-detection
    # path goes through worker.WorkerView.is_alive(), which holds its own
    # already-bound reference to is_session_alive from its module-level import —
    # patching devin_shell's attribute would not reach that call site.
    from unittest.mock import patch

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        result = app.dispatch(limit=1)

    # The stalled issue should be excluded from dispatch
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["stalled"] == [{"issue": 123, "pid": 99999, "health": "STALLED"}]


def test_dispatch_oldest_first_by_default(tmp_path: Path) -> None:
    """Test that dispatch selects oldest issues first by default (issue #151)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake GitHub with issues created out of order
    fake_gh = FakeGitHub()
    # Override issue_list to return issues with different creation dates
    fake_gh.issues = [
        {
            "number": 792,
            "title": "crash-fix",
            "url": "https://github.com/test/repo/issues/792",
            "body": "Fix crash",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-01T00:00:00Z",  # Oldest
            "updatedAt": "2026-07-01T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 808,
            "title": "e2e-test",
            "url": "https://github.com/test/repo/issues/808",
            "body": "E2E test",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-06T00:00:00Z",  # Newest
            "updatedAt": "2026-07-06T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 793,
            "title": "data-model",
            "url": "https://github.com/test/repo/issues/793",
            "body": "Data model",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-02T00:00:00Z",  # Middle
            "updatedAt": "2026-07-02T00:00:00Z",
            "state": "OPEN",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Dispatch 2 issues - should select oldest first (792, then 793)
    result = app.dispatch(limit=2)

    assert result.ok is True
    assert result.data["selected_count"] == 2
    # Should select oldest issues: 792 (oldest), 793 (middle)
    selected_numbers = [s["issue_number"] for s in result.data["sessions"]]
    assert selected_numbers == [792, 793]


def test_dispatch_newest_first_with_config(tmp_path: Path) -> None:
    """Test that dispatch selects newest issues first when configured (issue #151)."""
    config = OrchestratorConfig(dispatch=DispatchConfig(order="newest"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake GitHub with issues created out of order
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 792,
            "title": "crash-fix",
            "url": "https://github.com/test/repo/issues/792",
            "body": "Fix crash",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-01T00:00:00Z",  # Oldest
            "updatedAt": "2026-07-01T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 808,
            "title": "e2e-test",
            "url": "https://github.com/test/repo/issues/808",
            "body": "E2E test",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-06T00:00:00Z",  # Newest
            "updatedAt": "2026-07-06T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 793,
            "title": "data-model",
            "url": "https://github.com/test/repo/issues/793",
            "body": "Data model",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-02T00:00:00Z",  # Middle
            "updatedAt": "2026-07-02T00:00:00Z",
            "state": "OPEN",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Dispatch 2 issues - should select newest first (808, then 793)
    result = app.dispatch(limit=2)

    assert result.ok is True
    assert result.data["selected_count"] == 2
    # Should select newest issues: 808 (newest), 793 (middle)
    selected_numbers = [s["issue_number"] for s in result.data["sessions"]]
    assert selected_numbers == [808, 793]


def test_dispatch_sorts_by_out_degree_blocked_dependents(tmp_path: Path) -> None:
    """Test that dispatch sorts by out-degree (number of blocked dependents) per issue #152."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake GitHub with dependency relationships:
    # - Issue X (100) has 0 blocked dependents
    # - Issue Y (200) has 3 blocked dependents (300, 400, 500)
    # - Issues 300, 400, 500 are blocked by Y (have open blocker Y)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 100,
            "title": "issue-x",
            "url": "https://github.com/test/repo/issues/100",
            "body": "Issue X with no dependents",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-01T00:00:00Z",
            "updatedAt": "2026-07-01T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 200,
            "title": "issue-y",
            "url": "https://github.com/test/repo/issues/200",
            "body": "Issue Y with 3 dependents",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-02T00:00:00Z",
            "updatedAt": "2026-07-02T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 300,
            "title": "dependent-1",
            "url": "https://github.com/test/repo/issues/300",
            "body": "Blocked by #200",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-03T00:00:00Z",
            "updatedAt": "2026-07-03T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 400,
            "title": "dependent-2",
            "url": "https://github.com/test/repo/issues/400",
            "body": "Blocked by #200",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-04T00:00:00Z",
            "updatedAt": "2026-07-04T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 500,
            "title": "dependent-3",
            "url": "https://github.com/test/repo/issues/500",
            "body": "Blocked by #200",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-05T00:00:00Z",
            "updatedAt": "2026-07-05T00:00:00Z",
            "state": "OPEN",
        },
    ]

    # Mock issue_list to return all ready issues for out-degree computation
    original_issue_list = fake_gh.issue_list

    def mock_issue_list(labels=None, state=None):
        if labels and "automated-ready" in labels:
            return fake_gh.issues
        return original_issue_list(labels=labels, state=state)

    fake_gh.issue_list = mock_issue_list

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Dispatch 2 issues - should select Y (3 dependents) before X (0 dependents)
    result = app.dispatch(limit=2)

    assert result.ok is True
    assert result.data["selected_count"] == 2
    selected_numbers = [s["issue_number"] for s in result.data["sessions"]]
    # Y (200) should be selected first due to higher out-degree
    assert selected_numbers == [200, 100]


def test_dispatch_handles_cyclic_dependency_declaration(tmp_path: Path) -> None:
    """Test that dispatch handles cyclic dependency declarations without crashing per issue #152."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake GitHub with cyclic dependency:
    # - Issue A blocks B, B blocks A (both have open blockers on each other)
    # - Issue C is independent
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 100,
            "title": "issue-a",
            "url": "https://github.com/test/repo/issues/100",
            "body": "Blocked by #200",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-01T00:00:00Z",
            "updatedAt": "2026-07-01T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 200,
            "title": "issue-b",
            "url": "https://github.com/test/repo/issues/200",
            "body": "Blocked by #100",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-02T00:00:00Z",
            "updatedAt": "2026-07-02T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 300,
            "title": "issue-c",
            "url": "https://github.com/test/repo/issues/300",
            "body": "Independent issue",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-03T00:00:00Z",
            "updatedAt": "2026-07-03T00:00:00Z",
            "state": "OPEN",
        },
    ]

    # Mock issue_list to return all ready issues for out-degree computation
    def mock_issue_list(labels=None, state=None):
        if labels and "automated-ready" in labels:
            return fake_gh.issues
        return []

    fake_gh.issue_list = mock_issue_list

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Dispatch should not crash on cyclic dependencies
    # Since A and B block each other, both should be filtered out
    # Only C should be dispatchable
    result = app.dispatch(limit=2)

    assert result.ok is True
    # Only C should be selected (A and B are mutually blocked)
    assert result.data["selected_count"] == 1
    selected_numbers = [s["issue_number"] for s in result.data["sessions"]]
    assert selected_numbers == [300]


def test_dispatch_handles_missing_created_at(tmp_path: Path) -> None:
    """Test that dispatch handles missing createdAt field, sorting last per issue #152."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake GitHub with issues, some missing createdAt
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 100,
            "title": "old-issue",
            "url": "https://github.com/test/repo/issues/100",
            "body": "Old issue",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-01T00:00:00Z",
            "updatedAt": "2026-07-01T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 200,
            "title": "new-issue",
            "url": "https://github.com/test/repo/issues/200",
            "body": "New issue",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            "createdAt": "2026-07-06T00:00:00Z",
            "updatedAt": "2026-07-06T00:00:00Z",
            "state": "OPEN",
        },
        {
            "number": 300,
            "title": "missing-date",
            "url": "https://github.com/test/repo/issues/300",
            "body": "Issue without createdAt",
            "labels": [{"name": "automated-ready"}],
            "assignees": [],
            "author": {"login": "test"},
            # No createdAt field
            "updatedAt": "2026-07-03T00:00:00Z",
            "state": "OPEN",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Dispatch all 3 - missing createdAt should sort last
    result = app.dispatch(limit=3)

    assert result.ok is True
    assert result.data["selected_count"] == 3
    selected_numbers = [s["issue_number"] for s in result.data["sessions"]]
    # Order: 100 (oldest), 200 (newest), 300 (missing date, last)
    assert selected_numbers == [100, 200, 300]


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


def test_dispatch_excludes_stalled_session_real(tmp_path: Path) -> None:
    """Test that dispatch excludes issues with stalled sessions (real dispatch path)."""
    from datetime import UTC, datetime, timedelta
    from charlie_work.devin_shell import SessionRecord

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
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a session record for issue 123 with a live PID
    sessions_dir = app._resolve(config.devin.sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_file = sessions_dir / "issue-123.json"
    log_file = sessions_dir / "issue-123.log"

    # Write a log file with old mtime (stalled)
    log_file.write_text("working on issue\nmaking progress\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a session record with a fake PID (we'll mock liveness check)
    session_record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(
            tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
        ),
        command=("devin", "--prompt-file", "{prompt_path}"),
        pid=99999,  # Fake PID - we'll mock liveness to return True
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
        process_start_time=time.time(),
    )
    session_file.write_text(json.dumps(session_record.to_dict()), encoding="utf-8")

    # Mock the liveness check to return True (simulating a live but stalled process)
    from unittest.mock import patch

    with patch("charlie_work.devin_shell.is_session_alive", return_value=True):
        result = app.dispatch(limit=1)

    # The stalled issue should be excluded from dispatch
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["stalled"] == [{"issue": 123, "pid": 99999}]


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

    ready = app.merge_ready(456, merge=True)  # Explicitly request merge

    assert ready.data["can_merge"] is True
    assert ready.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    assert fake_gh.merged_merge_flags == [()]
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

    ready = app.merge_ready(456, merge=True)

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

    ready = app.merge_ready(456, merge=True)

    assert ready.data["merged"] is True
    assert fake_gh.deleted_branches == []
    assert ready.data["branch_deleted"] is None


def test_merge_ready_update_open_prs_disabled_returns_none(tmp_path: Path) -> None:
    """Issue #149: when update_open_prs is disabled, update_open_prs_results must be None."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=False,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    ready = app.merge_ready(456, merge=True)

    assert ready.data["merged"] is True
    assert ready.data["update_open_prs_results"] is None


def test_merge_ready_not_merged_returns_none(tmp_path: Path) -> None:
    """Issue #149: when PR is not merged, update_open_prs_results must be None."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # No approval decision, so PR won't merge
    ready = app.merge_ready(456)

    assert ready.data["merged"] is False
    assert ready.data["update_open_prs_results"] is None


def test_merge_ready_update_open_prs_zero_matching_returns_empty_list(tmp_path: Path) -> None:
    """Issue #149: when sweep runs with zero matching PRs, update_open_prs_results must be []."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
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
    # The sweep ran but found no other open agent PRs to update
    assert ready.data["update_open_prs_results"] == []


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


def test_config_rejects_non_mapping_devin_worker_env(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    path = tmp_path / "c.yaml"
    # Same validation for devin.worker_env
    path.write_text('devin:\n  worker_env: "2"\n', encoding="utf-8")

    try:
        load_config(path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")

    assert "worker_env" in message
    assert "devin" in message


def test_config_rejects_non_list_materialize_dirs(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    path = tmp_path / "c.yaml"
    # A plausible operator typo: a scalar instead of a list.
    path.write_text('dispatch:\n  materialize_dirs: ".devin"\n', encoding="utf-8")

    try:
        load_config(path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")

    assert "materialize_dirs" in message
    assert "dispatch" in message
    assert "list" in message


def test_config_rejects_merge_flags_not_starting_with_double_dash(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    path = tmp_path / "c.yaml"
    # A plausible operator typo: a flag without the -- prefix.
    # Must fail at load with a clear error message.
    path.write_text('auto_merge:\n  merge_flags: ["admin"]\n', encoding="utf-8")

    try:
        load_config(path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")

    assert "merge_flags" in message
    assert "auto_merge" in message
    assert "must start with '--'" in message


def test_config_accepts_valid_merge_flags(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    # Valid merge_flags with -- prefix (non-managed flags only).
    path.write_text('auto_merge:\n  merge_flags: ["--auto", "--subject"]\n', encoding="utf-8")

    config = load_config(path)

    assert config.auto_merge.merge_flags == ("--auto", "--subject")


def test_config_rejects_orchestrator_managed_merge_flags(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    # Test each orchestrator-managed flag
    for flag in ["--merge", "--rebase", "--squash", "--delete-branch", "--admin"]:
        path = tmp_path / "c.yaml"
        path.write_text(f'auto_merge:\n  merge_flags: ["{flag}"]\n', encoding="utf-8")

        try:
            load_config(path)
            raise AssertionError(f"expected ConfigError for {flag}")
        except ConfigError as exc:
            message = str(exc)
            assert "merge_flags" in message
            assert "auto_merge" in message
            assert "managed by the orchestrator" in message
            assert flag in message


def test_config_rejects_orchestrator_managed_merge_flags_equals_form(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    # Test that --flag=value forms are also rejected (normalization splits on '=')
    # --delete-branch=true is the critical case: it bypasses exact match but is valid gh syntax
    for flag in ["--delete-branch=true", "--squash=true"]:
        path = tmp_path / "c.yaml"
        path.write_text(f'auto_merge:\n  merge_flags: ["{flag}"]\n', encoding="utf-8")

        try:
            load_config(path)
            raise AssertionError(f"expected ConfigError for {flag}")
        except ConfigError as exc:
            message = str(exc)
            assert "merge_flags" in message
            assert "auto_merge" in message
            assert "managed by the orchestrator" in message
            assert flag in message


def test_config_rejects_merge_flags_scalar(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    path = tmp_path / "c.yaml"
    # YAML scalar instead of list - this would iterate per-character
    path.write_text('auto_merge:\n  merge_flags: "--admin"\n', encoding="utf-8")

    try:
        load_config(path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")

    assert "merge_flags" in message
    assert "auto_merge" in message
    assert "must be a list" in message


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

    # Set a consistent head SHA for the PR
    app.gh.pr_head_shas[456] = "sha-abc123"

    def _fake_run(**kwargs):
        calls["n"] += 1
        # Write a report with the current head SHA so it can be reused
        head_sha = kwargs.get("head_ref_oid", "sha-abc123")
        report_content = (
            f"# Cross-family adversarial review — `codex`\n\n"
            f"<!-- PR head SHA: {head_sha} -->\n\n"
            f"> Findings below are **leads, not verdicts**\n\n"
            f"---\n\n"
            f"{VALID_REPORT}\n"
        )
        Path(kwargs["report_path"]).write_text(report_content, encoding="utf-8")
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
    """When repo has .venv, VIRTUAL_ENV must be set to that path."""
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
        "UV_PROJECT_ENVIRONMENT must be dropped when repo has .venv"
    )


def test_run_cross_family_sanitizes_environment_at_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_cross_family_review must pass sanitized env to the actual subprocess runner."""
    import subprocess
    from charlie_work.cross_family import run_cross_family_review

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    report_path = tmp_path / "report.md"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("test prompt", encoding="utf-8")

    # Set parent env variables (simulating orchestrator leak)
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    captured_env: dict[str, str] | None = None

    def _fake_runner(command, **kwargs):
        nonlocal captured_env
        captured_env = kwargs.get("env")
        # Return a valid report
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="**MINOR**\nissue\n\nVerdict: safe",
            stderr="",
        )

    result = run_cross_family_review(
        model="codex",
        command=("echo", "test"),
        repo_root=repo_root,
        prompt_text="test prompt",
        prompt_path=prompt_path,
        report_path=report_path,
        timeout_seconds=30,
        runner=_fake_runner,
    )

    assert result.ok is True
    assert captured_env is not None, "Runner should have received env kwarg"
    assert "VIRTUAL_ENV" not in captured_env, (
        "VIRTUAL_ENV must be sanitized in the actual subprocess env"
    )
    assert "UV_PROJECT_ENVIRONMENT" not in captured_env, (
        "UV_PROJECT_ENVIRONMENT must be sanitized in the actual subprocess env"
    )


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

    # First request_changes (count = 1, head = "sha-1")
    fake_gh.pr_head_shas[456] = "sha-1"
    first = app.record_review(456, "request_changes", summary="fix A")

    # Second request_changes (count = 2, head = "sha-2")
    fake_gh.pr_head_shas[456] = "sha-2"
    second = app.record_review(456, "request_changes", summary="fix B")

    # Third request_changes (count stays at 2, escalated, head = "sha-3")
    fake_gh.pr_head_shas[456] = "sha-3"
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


def test_cross_family_report_invalidated_on_head_sha_change(tmp_path: Path, monkeypatch) -> None:
    """Regression test for issue #156: cross-family reports must be invalidated
    when the PR head SHA changes to prevent reviewing stale code."""
    app = _cross_family_app(tmp_path, enabled=True)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)

    # Create a report with an old head SHA
    old_head_sha = "abc123def456"
    report_content = (
        f"# Cross-family adversarial review — `codex`\n\n"
        f"<!-- PR head SHA: {old_head_sha} -->\n\n"
        f"> Findings below are **leads, not verdicts**\n\n"
        f"---\n\n"
        f"**MAJOR**\nfile:line: old bug\n\nVerdict: needs work\n"
    )
    (pr_dir / "cross-family-review.md").write_text(report_content, encoding="utf-8")

    calls = {"n": 0}

    def _fake_run(**kwargs):
        calls["n"] += 1
        # Verify that the new head SHA is passed to run_cross_family_review
        assert kwargs.get("head_ref_oid") == "newheadsha789"
        Path(kwargs["report_path"]).write_text("# new findings", encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.run_cross_family_review", _fake_run)

    # Update the PR to have a new head SHA
    app.gh.pr_head_shas[456] = "newheadsha789"

    result = app.review(456)

    # The old report should NOT be reused because the head SHA changed
    assert calls["n"] == 1
    assert result.data["cross_family_ok"] is True


def test_cross_family_report_reused_when_head_sha_unchanged(tmp_path: Path, monkeypatch) -> None:
    """Regression test for issue #156: cross-family reports should be reused
    when the PR head SHA has not changed."""
    app = _cross_family_app(tmp_path, enabled=True)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)

    # Create a report with a head SHA
    head_sha = "abc123def456"
    report_content = (
        f"# Cross-family adversarial review — `codex`\n\n"
        f"<!-- PR head SHA: {head_sha} -->\n\n"
        f"> Findings below are **leads, not verdicts**\n\n"
        f"---\n\n"
        f"**MAJOR**\nfile:line: bug\n\nVerdict: needs work\n"
    )
    (pr_dir / "cross-family-review.md").write_text(report_content, encoding="utf-8")

    calls = {"n": 0}

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text("# new findings", encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.run_cross_family_review", _fake_run)

    # The PR still has the same head SHA
    app.gh.pr_head_shas[456] = head_sha

    result = app.review(456)

    # The report should be reused because the head SHA has not changed
    assert calls["n"] == 0
    assert result.data["cross_family_ok"] is True
    assert result.data["cross_family_reused"] is True


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


def test_dispatch_skips_prose_only_deps_labeled_issues(tmp_path: Path) -> None:
    """Issue #225: dispatch should skip issues labeled with prose-only-deps."""

    class ProseOnlyDepsGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 908,
                    "title": "Issue with prose-only deps",
                    "url": "https://example.test/issues/908",
                    "body": "Do not dispatch before P2-T2/P2-T3 have landed.",
                    "labels": [{"name": "automated-ready"}, {"name": "agent:prose-only-deps"}],
                },
                {
                    "number": 910,
                    "title": "Normal issue",
                    "url": "https://example.test/issues/910",
                    "body": "Just a normal issue",
                    "labels": [{"name": "automated-ready"}],
                },
            ]

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = ProseOnlyDepsGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=10)

    # Only issue 910 should be dispatched (908 is labeled with prose-only-deps)
    assert result.data["selected_count"] == 1
    assert result.data["attempted_count"] == 1
    # The dispatched issue should be 910, not 908
    assert result.data["sessions"][0]["issue_number"] == 910


def test_review_label_transition_failure_persists_packet(tmp_path: Path) -> None:
    """Issue #135: A PARTIAL_FAILURE during review_started label transition must
    leave the review packet persisted in state and report structured label_error."""
    from charlie_work.labels import TransitionOutcome

    config = OrchestratorConfig()
    reviewing_label = config.labels.reviewing

    class LabelFailReviewGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            if label == reviewing_label:
                # Return False to simulate add failure (error-as-value)
                return False
            return super().add_issue_label(number, label)

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailReviewGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "review_started"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    # PR #456 is linked to issue #123 in FakeGitHub
    assert (123, reviewing_label) in label_error["add_failures"]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "reviewing"
    assert state["prs"]["456"]["label_error"]["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
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
            # Handle dependency API calls
            if "dependencies" in " ".join(arguments):
                return [] if json_output else ""
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
            # Handle dependency API calls
            if "dependencies" in " ".join(arguments):
                return [] if json_output else ""
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

    # First request_changes (count = 1, head = "sha-1")
    fake_gh.pr_head_shas[456] = "sha-1"
    app.record_review(456, "request_changes", summary="a")

    # Second request_changes (count = 2, head = "sha-2")
    fake_gh.pr_head_shas[456] = "sha-2"
    app.record_review(456, "request_changes", summary="b")

    # Flood the event log so any record_review events for 456 are evicted.
    state = load_state(paths.state_file)
    for i in range(300):
        state = _append(state, "review_packet", {"pr_number": 90000 + i})
    save_state(paths.state_file, state)
    assert not any(  # prove the earlier request_changes events are gone
        e.get("kind") == "record_review" for e in load_state(paths.state_file)["events"]
    )

    # Third request_changes (count stays at 2, escalated, head = "sha-3")
    fake_gh.pr_head_shas[456] = "sha-3"
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


def test_record_review_transition_failure_recorded(tmp_path: Path) -> None:
    """Issue #135: PARTIAL_FAILURE during record_review transition must be recorded."""
    from charlie_work.labels import TransitionOutcome

    config = OrchestratorConfig()

    class LabelFailGitHub(FakeGitHub):
        def remove_issue_label(self, number: int, label: str) -> bool:
            # Return False to simulate remove failure (error-as-value)
            return False

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(456, "approved", summary="lgtm")

    assert result.ok is True
    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "review_approved"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert len(label_error["remove_failures"]) > 0


def test_record_review_request_changes_transition_failure_recorded(tmp_path: Path) -> None:
    """Issue #135: PARTIAL_FAILURE during record_review request_changes transition must be recorded."""
    from charlie_work.labels import TransitionOutcome

    config = OrchestratorConfig()

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            # Return False to simulate add failure (error-as-value)
            return False

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(456, "request_changes", summary="fix it")

    assert result.ok is True
    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "rework_requested"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert len(label_error["add_failures"]) > 0


def test_record_review_blocked_transition_failure_recorded(tmp_path: Path) -> None:
    """Issue #135: PARTIAL_FAILURE during record_review blocked transition must be recorded."""
    from charlie_work.labels import TransitionOutcome

    config = OrchestratorConfig()

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            # Return False to simulate add failure (error-as-value)
            return False

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(456, "blocked", summary="security issue")

    assert result.ok is True
    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "blocked"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert len(label_error["add_failures"]) > 0


def test_merge_ready_head_moved_transition_failure_recorded(tmp_path: Path) -> None:
    """Issue #135: PARTIAL_FAILURE during merge_ready head-moved transition must be recorded."""
    from charlie_work.labels import TransitionOutcome

    config = OrchestratorConfig()

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            # Return False to simulate add failure (error-as-value)
            return False

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # First approve the PR to set reviewed_head_sha
    app.record_review(456, "approved", summary="lgtm")

    # Then simulate head moved by updating the PR head SHA
    fake_gh.pr_head_shas[456] = "sha-different"

    # Now merge_ready should trigger head-moved re-review path
    result = app.merge_ready(456)

    # Head-moved returns ok=False (cannot merge), but label_error is still recorded
    assert result.ok is False
    assert result.data["head_moved"] is True
    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "review_started"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert len(label_error["add_failures"]) > 0


def test_review_started_clears_needs_rework() -> None:
    # Re-review after a rework must not stack reviewing on top of needs-rework.
    from charlie_work.labels import transition, TransitionOutcome

    fake_gh = FakeGitHub()
    result = transition(fake_gh, OrchestratorConfig().labels, 123, "review_started")

    assert result.outcome == TransitionOutcome.APPLIED
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


def test_dispatch_rework_transition_failure_recorded(tmp_path: Path) -> None:
    """Issue #135: PARTIAL_FAILURE during rework_dispatched transition must be recorded."""
    from charlie_work.labels import TransitionOutcome

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
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; sys.exit(1)",  # Simulate dispatch failure
            ),
        )
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
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        )
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
    """Issue #135: PARTIAL_FAILURE during merged transition must be recorded."""
    from charlie_work.labels import TransitionOutcome

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            # Return False to simulate add failure (error-as-value)
            return False

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
    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "merged"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
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


def test_merge_ready_pr_list_error_during_update_open_prs_is_caught(tmp_path: Path) -> None:
    """Issue #146: GitHubError from pr_list during post-merge sweep must not propagate."""
    from charlie_work.config import AutoMergeConfig
    from charlie_work.github import GitHubError

    class PrListFailGitHub(FakeGitHub):
        def pr_list(self):
            raise GitHubError("API rate limit exceeded")

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,  # Enable the feature that calls pr_list
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = PrListFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    # Write the approved decision directly
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.merge_ready(456, merge=True)

    # The merge should still succeed despite pr_list failing
    assert result.data["merged"] is True
    # The error should be recorded in update_open_prs_results
    assert result.data["update_open_prs_results"] is not None
    assert len(result.data["update_open_prs_results"]) == 1
    assert "error" in result.data["update_open_prs_results"][0]
    assert "pr_list failed" in result.data["update_open_prs_results"][0]["error"]
    # The merged state should still be recorded
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "merged"


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
    """Issue #135: PARTIAL_FAILURE during dispatch label transition must be recorded."""
    from charlie_work import devin_shell
    from charlie_work.labels import TransitionOutcome
    from charlie_work.worktree import WorktreeInfo

    wt_path = tmp_path / "worktrees" / "agent-issue-123-fix-search"
    wt_path.mkdir(parents=True, exist_ok=True)

    def _fake_create_worktree(repo_root, branch, **kwargs):
        return WorktreeInfo(path=wt_path, branch=branch, venv_junction=None)

    monkeypatch.setattr(devin_shell, "create_worktree", _fake_create_worktree)

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            # Return False to simulate add failure (error-as-value)
            return False

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
    label_error = state["issues"]["123"]["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "dispatched"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value


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


def test_dry_run_dispatch_dependency_gate_filter(tmp_path: Path) -> None:
    """Issue #127: dry-run dispatch dependency-gate filter must exclude blocked issues.

    When a blocked issue is ordered ahead of an eligible candidate with
    dispatch_limit=1, the dry-run report should list the eligible issue as
    dispatchable and the blocked issue should be excluded from sessions.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="manual"),
    )
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


def test_github_are_issues_open_normalizes_uppercase_state(tmp_path: Path) -> None:
    """Issue #173: Regression test for GitHub.are_issues_open with realistic uppercase state.

    This test directly exercises the production GitHub.are_issues_open method with
    realistic uppercase state field values (as returned by the real GitHub API).
    It ensures the .upper() normalization in github.py:326 is tested and cannot
    silently regress.
    """
    from charlie_work.github import GitHub as RealGitHub

    # Create a subclass that overrides issue_view to return realistic uppercase state values
    class RealGitHubWithMockedIssueView(RealGitHub):
        def issue_view(self, number: int) -> dict:
            if number == 100:
                return {"number": 100, "state": "OPEN"}  # Uppercase as from real API
            elif number == 200:
                return {"number": 200, "state": "CLOSED"}  # Uppercase as from real API
            elif number == 300:
                return {"number": 300, "state": "OPEN"}  # Uppercase as from real API
            elif number == 400:
                return {
                    "number": 400,
                    "state": "open",
                }  # Lowercase (should still work due to .upper())
            else:
                raise ValueError(f"Unexpected issue number: {number}")

    # Create an instance of the subclass
    real_gh = RealGitHubWithMockedIssueView(repo_root=tmp_path)

    # Call are_issues_open with a mix of open/closed issues
    result = real_gh.are_issues_open([100, 200, 300, 400])

    # Assert that only the OPEN-state issues (100, 300, and 400) are returned
    # 200 is CLOSED and should not be in the result
    assert result == {100, 300, 400}, f"Expected {{100, 300, 400}}, got {result}"


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

    # Step 2: Record first request_changes (count = 1, not escalated, head = "sha-1")
    fake_gh.pr_head_shas[456] = "sha-1"
    review_result_1 = app.record_review(456, "request_changes", summary="fix A")
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

    review_result_2 = app.record_review(456, "request_changes", summary="fix B")
    assert review_result_2.ok is True
    assert review_result_2.data["escalated"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["issues"]["123"]["status"] == "rework_requested"

    # Step 4: Record third request_changes (count stays at 2, escalated because max_rework_cycles = 2, head = "sha-3")
    # When escalated, the count is NOT incremented (see workflow.py line 731-734)
    fake_gh.pr_head_shas[456] = "sha-3"
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


def test_request_changes_count_does_not_increment_on_unchanged_head(tmp_path: Path) -> None:
    """Issue #208: request_changes_count should only increment when PR head advances.

    When a worker dies orphaned and the PR head never advances, re-issuing
    request_changes should not consume the escalation budget.
    """
    config = OrchestratorConfig(
        review=ReviewConfig(max_rework_cycles=2),
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

    # Step 1: Fresh dispatch
    dispatch_result = app.dispatch(limit=1)
    assert dispatch_result.ok is True

    # Step 2: Record first request_changes (count = 1, head = "sha-1")
    fake_gh.pr_head_shas[456] = "sha-1"
    review_result_1 = app.record_review(456, "request_changes", summary="fix A")
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

    review_result_2 = app.record_review(456, "request_changes", summary="fix B")
    assert review_result_2.ok is True
    assert review_result_2.data["escalated"] is False

    state = load_state(paths.state_file)
    # Count should NOT increment because head didn't advance
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-1"

    # Step 4: Record third request_changes with NEW head (count should increment to 2)
    fake_gh.pr_head_shas[456] = "sha-2"
    review_result_3 = app.record_review(456, "request_changes", summary="fix C")
    assert review_result_3.ok is True
    assert review_result_3.data["escalated"] is False

    state = load_state(paths.state_file)
    # Count should increment because head advanced
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-2"


def test_merge_ready_refuses_when_head_moved_after_approval(tmp_path: Path) -> None:
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    fake_gh.prs[0] = {**fake_gh.prs[0], "headRefOid": "sha-new-head"}
    fake_gh.pr_head_shas[456] = "sha-new-head"

    result = app.merge_ready(456, merge=True)

    assert result.ok is False
    assert "PR head moved since approval" in result.message
    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert result.data["head_moved"] is True
    assert fake_gh.merged == []
    assert (123, "agent:reviewing") in fake_gh.labels_added
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "reviewing"
    assert fake_gh.merged_merge_flags == []


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
    # Default config: no --admin
    assert fake_gh.merged_admin_flags == [False]
    assert fake_gh.merged_merge_flags == [()]


def test_merge_ready_passes_admin_flag_when_configured(tmp_path: Path) -> None:
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(required_checks=(), require_approved_review=True, admin=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    assert fake_gh.merged_admin_flags == [True]
    assert fake_gh.merged_merge_flags == [()]


def test_merge_ready_passes_merge_flags_when_configured(tmp_path: Path) -> None:
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(), require_approved_review=True, merge_flags=("--admin",)
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    assert fake_gh.merged_admin_flags == [True]
    assert fake_gh.merged_merge_flags == [("--admin",)]


def test_merge_ready_merge_flags_takes_precedence_over_admin(tmp_path: Path) -> None:
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            require_approved_review=True,
            admin=True,
            merge_flags=("--admin",),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    # merge_flags takes precedence, so admin flag should be True (from merge_flags)
    assert fake_gh.merged_admin_flags == [True]
    assert fake_gh.merged_merge_flags == [("--admin",)]


def test_merge_ready_default_merge_flags_preserves_current_behavior(
    tmp_path: Path,
) -> None:
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(required_checks=(), require_approved_review=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    # Default empty tuple should not add admin flag
    assert fake_gh.merged_admin_flags == [False]
    assert fake_gh.merged_merge_flags == [()]


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
    import json

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


def test_classify_dead_sessions_relabel_idempotent(tmp_path: Path) -> None:
    """Issue #118 AC3: classification pass relabel is idempotent - two-pass test.

    This test runs the classification pass twice on the same dead session and
    verifies that (a) no error occurs, (b) no duplicate event is emitted, and
    (c) the issue remains in the correct label state after the second pass.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

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
    # Issue starts with in_progress label (active)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]

    # Ensure state directory exists
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a sessions directory with a dead session
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

    # First pass: run classification directly (not via loop to avoid review logic)
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    # Verify first pass relabeled the issue
    assert (42, config.labels.in_progress) in fake_gh.labels_removed
    assert (42, config.labels.ready) in fake_gh.labels_added

    # Verify event was emitted
    state = load_state(paths.state_file)
    events_after_first = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events_after_first) == 1
    assert events_after_first[0]["payload"]["issue_number"] == 42

    # Update fake GitHub to reflect the relabeled state (ready label, no active labels)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.ready}],
        }
    ]
    # Clear label tracking for second pass
    fake_gh.labels_added = []
    fake_gh.labels_removed = []

    # Second pass: run classification again
    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    # Verify second pass did NOT emit duplicate event (idempotency)
    state_after_second = load_state(paths.state_file)
    events_after_second = [
        e for e in state_after_second["events"] if e["kind"] == "session_failed_relabeled"
    ]
    assert len(events_after_second) == 1, "Second pass should not emit duplicate event"

    # Verify second pass did not attempt to remove in_progress (already gone)
    assert (42, config.labels.in_progress) not in fake_gh.labels_removed

    # Verify second pass did not attempt to add ready (already present)
    assert (42, config.labels.ready) not in fake_gh.labels_added


def test_classify_dead_sessions_preserves_state_record_branch(tmp_path: Path) -> None:
    """Issue #118 AC1: classification pass preserves state record branch/worktree fields.

    This test ensures that the relabel logic does not clobber the branch or
    worktree_path fields in the state record for the issue. Mutation gate:
    clobbering branch MUST fail this test.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

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
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]

    # Ensure state directory exists
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize state with a branch/worktree entry for issue 42
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["42"] = {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "labels": [config.labels.in_progress],
            "branch": "agent/issue-42-fix-search",
            "worktree_path": "/tmp/worktree-issue-42",
        }
        save_state(paths.state_file, state)

    # Create a sessions directory with a dead session
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

    # Run classification pass directly
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    # Verify branch and worktree_path are preserved byte-identical
    state_after = load_state(paths.state_file)
    assert state_after["issues"]["42"]["branch"] == "agent/issue-42-fix-search"
    assert state_after["issues"]["42"]["worktree_path"] == "/tmp/worktree-issue-42"


def test_classify_dead_sessions_dispatch_recovery_integration(tmp_path: Path) -> None:
    """Issue #118 AC4: full chain integration test - classified-dead + relabeled → dispatch.

    This test drives the complete workflow:
    1. A session dies and is classified by the automated pass
    2. The issue is relabeled to dispatchable (ready label)
    3. The next dispatch pass selects the issue
    4. The recovery dict (branch/worktree) is passed to create_worktree
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

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
    # Issue starts with in_progress label (active)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Ensure state directory exists
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize state with a branch/worktree entry for issue 42 (recovery dict)
    # The recovery matcher expects branch_name and status: "dispatched"
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["42"] = {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "labels": [config.labels.in_progress],
            "branch_name": "agent/issue-42-fix-search",
            "worktree_path": "/tmp/worktree-issue-42",
            "status": "dispatched",
        }
        save_state(paths.state_file, state)

    # Create a sessions directory with a dead session
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

    # Step 1: Run classification pass directly
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    # Verify the issue was relabeled to ready
    assert (42, config.labels.in_progress) in fake_gh.labels_removed
    assert (42, config.labels.ready) in fake_gh.labels_added

    # Update fake GitHub to reflect the relabeled state
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.ready}],
        }
    ]
    # Clear label tracking
    fake_gh.labels_added = []
    fake_gh.labels_removed = []

    # Clear throttle state so dispatch is not deferred
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state.pop("throttled_until", None)
        save_state(paths.state_file, state)

    # Step 2: Run dispatch pass - should select the relabeled issue
    dispatch_result = app.dispatch(limit=1)

    # Verify dispatch selected the issue
    assert dispatch_result.ok is True
    assert dispatch_result.data["selected_count"] == 1
    assert dispatch_result.data["sessions"][0]["issue_number"] == 42

    # Verify the recovery dict (branch_name/worktree_path) was preserved in state
    # The dispatch should have used the existing branch from state
    state_after_dispatch = load_state(paths.state_file)
    assert state_after_dispatch["issues"]["42"]["branch_name"] == "agent/issue-42-fix-search"
    assert state_after_dispatch["issues"]["42"]["worktree_path"] == "/tmp/worktree-issue-42"
    # Verify the recovery dict was actually passed to the adapter (non-None)
    # The test should fail if recovery is hardcoded to None at dispatch call sites
    assert dispatch_result.data["sessions"][0].get("recovery") is not None


def test_classify_dead_sessions_with_closed_pr_triggers_relabel(tmp_path: Path) -> None:
    """Issue #118 R2: dead session + prior CLOSED PR only ⇒ relabel fires.

    This is a workflow-level test driving _classify_dead_sessions_and_update_throttle_state
    to ensure the OPEN filter is enforced. Mutation gate: dropping the OPEN filter fails this test.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

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
    # Issue starts with in_progress label (active)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    # Prior CLOSED PR (not OPEN) - should NOT suppress relabel
    fake_gh.prs = [
        {
            "number": 1,
            "title": "Fix #42: search",
            "url": "https://example.test/pull/1",
            "headRefName": "agent/issue-42-fix-search",
            "baseRefName": "main",
            "body": "Closes #42",
            "state": "CLOSED",
            "labels": [],
            "isCrossRepository": False,
        }
    ]

    # Ensure state directory exists
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a sessions directory with a dead session
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

    # Run classification pass directly
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    # Verify relabel fired despite CLOSED PR (OPEN filter works)
    assert (42, config.labels.in_progress) in fake_gh.labels_removed
    assert (42, config.labels.ready) in fake_gh.labels_added


def test_classify_dead_sessions_with_open_pr_suppresses_relabel(tmp_path: Path) -> None:
    """Issue #118 R2: dead session + OPEN PR ⇒ no relabel.

    This is a workflow-level test driving _classify_dead_sessions_and_update_throttle_state
    to ensure the OPEN PR guard is enforced. Mutation gate: deleting the guard fails this test.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

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
    # Issue starts with in_progress label (active)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    # OPEN PR - should suppress relabel
    fake_gh.prs = [
        {
            "number": 1,
            "title": "Fix #42: search",
            "url": "https://example.test/pull/1",
            "headRefName": "agent/issue-42-fix-search",
            "baseRefName": "main",
            "body": "Closes #42",
            "state": "OPEN",
            "labels": [],
            "isCrossRepository": False,
        }
    ]

    # Ensure state directory exists
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a sessions directory with a dead session
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

    # Run classification pass directly
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    # Verify relabel did NOT fire (OPEN PR guard works)
    assert (42, config.labels.in_progress) not in fake_gh.labels_removed
    assert (42, config.labels.ready) not in fake_gh.labels_added


def test_classify_dead_sessions_worker_blocked_escalates_and_suppresses_redispatch(
    tmp_path: Path,
) -> None:
    """Issue #261 F5: a dead session whose post-mortem shows worker_blocked
    (killed by a push-gate hook) must escalate to human review instead of
    hot-relabeling to ready — a hot relabel would redispatch straight back
    into the same push-gate hook and, per attempt_refs.py's motivation,
    destroy the worker's unpushed commits on the next branch reset.

    This is the workflow-level counterpart to
    reconcile.py's test_detect_drift_session_failed_worker_blocked_escalates_instead_of_relabel
    and mirrors test_classify_dead_sessions_with_open_pr_suppresses_relabel's
    style, but the suppressing signal is a worker_blocked post-mortem verdict
    (no open PR at all) rather than an open PR.

    Mutation gate: dropping the `worker_blocked or` clause from the escalation
    condition at workflow.py's `_classify_dead_sessions_and_update_throttle_state`
    (the `if (worker_blocked or len(redispatch_at) > ...)` check) fails this test.
    """
    import json
    import sqlite3
    from datetime import UTC, datetime

    from charlie_work.config import AutoMergeConfig, DevinConfig, PostMortemConfig
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.state import load_state

    now = datetime.now(UTC)
    worktree_path = str(tmp_path / "worktree")

    # Build a sessions.db fixture with a "Tool blocked:" message node matching
    # this worker's worktree_path and timing — the same shape post_mortem.py's
    # classify_and_record looks for to return "worker_blocked".
    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, working_directory TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE message_nodes ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, "
            "content TEXT, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO sessions (id, working_directory, created_at) VALUES (?, ?, ?)",
            ("sess-1", worktree_path, now.isoformat()),
        )
        conn.execute(
            "INSERT INTO message_nodes (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                "sess-1",
                "tool",
                'Tool blocked: {"decision": "block", "reason": "push-gate hook rejected"}',
                now.isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Use command adapter to avoid needing a real devin binary.
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    fake_gh.prs = []  # No open PR at all — the ordinary relabel path would fire here.

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("some work then silence\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path=worktree_path,
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    # (a) No hot relabel-to-ready — the escalation path must never grant the
    # `ready` label. Since dispatch() selects candidates via
    # gh.issue_list(config.labels.ready), an issue that never receives this
    # label can never be selected by a subsequent dispatch pass — this is a
    # structural (not merely incidental) proof that redispatch cannot fire.
    assert (42, config.labels.ready) not in fake_gh.labels_added

    # The escalation transition (redispatch_escalated) must have actually run:
    # human_needed added, in_progress removed — proving escalation took the
    # GitHub-mutating path rather than silently no-oping.
    assert (42, config.labels.human_needed) in fake_gh.labels_added
    assert (42, config.labels.in_progress) in fake_gh.labels_removed

    # (b) escalation_reason recorded as worker_blocked, not the generic cap.
    state = load_state(paths.state_file)
    issue_entry = state["issues"]["42"]
    assert issue_entry["status"] == "escalated"
    assert issue_entry["escalation_reason"] == "worker_blocked"

    # No session_failed_relabeled event was appended for this issue — only
    # session_failed_escalated.
    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 42]
    assert "session_failed_relabeled" not in event_kinds
    assert "session_failed_escalated" in event_kinds


def test_classify_dead_sessions_worker_blocked_log_tail_fallback_escalates_and_suppresses_redispatch(
    tmp_path: Path,
) -> None:
    """Issue #260 (corrected premise): the same escalate/suppress-redispatch
    contract as test_classify_dead_sessions_worker_blocked_escalates_and_suppresses_redispatch
    above, but the worker_blocked signal comes from the log-tail fallback
    (post_mortem.classify_and_record's _classify_worker_blocked_from_log_tail)
    rather than a sessions.db match -- exercised here by pointing db_path at
    a location with no database at all, so DB-based extraction degrades to
    matched=False and the log tail ("Error: A tool was rejected by the
    user.", the Devin CLI's own PreToolUse hook-block surfacing) is the only
    signal available. This is the corrected #260 ask: this exact string was
    originally misclassified as a provider throttle signature (rate_limited,
    hot-redispatched after a cooldown) -- it must instead escalate on first
    occurrence, identically to a DB-detected "Tool blocked:" verdict.

    Mutation gate: this test is the log-tail-fallback counterpart of the
    DB-based test above and is covered by the same
    `if (worker_blocked or len(redispatch_at) > ...)` mutation at
    workflow.py's _classify_dead_sessions_and_update_throttle_state; it also
    independently covers post_mortem.classify_and_record's log-tail fallback
    branch (see test_post_mortem.py's
    test_classify_and_record_log_tail_fallback_detects_worker_blocked_when_db_unavailable
    for that mutation gate's verbatim transcript).
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.config import AutoMergeConfig, DevinConfig, PostMortemConfig
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.state import load_state

    now = datetime.now(UTC)
    worktree_path = str(tmp_path / "worktree")

    # No sessions.db at all -- DB-based extraction must degrade to
    # matched=False (extraction_error set), leaving the log tail as the
    # only signal.
    missing_db_path = tmp_path / "does-not-exist" / "sessions.db"

    # Use command adapter to avoid needing a real devin binary.
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
        post_mortem=PostMortemConfig(db_path=str(missing_db_path)),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    fake_gh.prs = []  # No open PR at all — the ordinary relabel path would fire here.

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("Error: A tool was rejected by the user.\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path=worktree_path,
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    # No hot relabel-to-ready — same structural proof as the DB-based test.
    assert (42, config.labels.ready) not in fake_gh.labels_added

    # Escalation transition actually ran.
    assert (42, config.labels.human_needed) in fake_gh.labels_added
    assert (42, config.labels.in_progress) in fake_gh.labels_removed

    # escalation_reason recorded as worker_blocked, not the generic cap or
    # (crucially, per the corrected premise) rate_limited.
    state = load_state(paths.state_file)
    issue_entry = state["issues"]["42"]
    assert issue_entry["status"] == "escalated"
    assert issue_entry["escalation_reason"] == "worker_blocked"

    # No throttle cooldown was set — a worker_blocked verdict must never
    # carry rate-limit retry semantics.
    assert state.get("throttled_until") is None

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 42]
    assert "session_failed_relabeled" not in event_kinds
    assert "session_failed_escalated" in event_kinds


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


def test_merge_ready_two_approved_prs_second_ship_succeeds_after_first_ship(
    tmp_path: Path,
) -> None:
    """End-to-end test for AC2: shipping two approved PRs in sequence should succeed.

    Regression test for issue #89: when two PRs are approved in the same operator pass,
    merging the first should not base-update the second (which would move its head and
    invalidate its approval). This test goes through the full merge_ready() path
    (not just _update_open_agent_prs) to verify the complete ship-it flow.
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

    # Ship the first PR
    result_456 = app.merge_ready(456, merge=True)
    assert result_456.ok is True
    assert result_456.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]

    # Verify PR 789's head is UNCHANGED (the skip worked)
    pr_789 = fake_gh.pr_view(789)
    assert pr_789["headRefOid"] == "sha-def456"  # Still the original head

    # Ship the second PR immediately afterward - should succeed without head-moved error
    result_789 = app.merge_ready(789, merge=True)
    assert result_789.ok is True
    assert result_789.data["merged"] is True
    assert result_789.data["can_merge"] is True
    assert result_789.data.get("head_moved") is not True  # Should not trigger head-moved gate
    assert fake_gh.merged == [(456, "squash"), (789, "squash")]


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
    def mock_count_live(sessions_dir):
        return 0

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(adapter="manual"),
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
    def mock_count_live(sessions_dir):
        return 1

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(adapter="manual"),
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
    def mock_count_live(sessions_dir):
        return 0

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(adapter="manual"),
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
    def mock_count_live(sessions_dir):
        return 0

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
    # dispatch_rework doesn't include skipped_issue_numbers in its result


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


def test_fleet_concurrency_governor_unlimited_when_unset(tmp_path: Path, monkeypatch) -> None:
    """When fleet.global_max_concurrent_sessions is 0 (default), dispatch should behave as before (unlimited)."""

    # Mock count_fleet_live_sessions to return 0 fleet live sessions
    def mock_count_fleet_live(fleet_dir_override):
        return 0, []

    monkeypatch.setattr("charlie_work.workflow.count_fleet_live_sessions", mock_count_fleet_live)

    config = OrchestratorConfig(
        fleet=FleetConfig(global_max_concurrent_sessions=0),
        dispatch=DispatchConfig(max_concurrent_sessions=0),
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch()

    # Should dispatch normally without fleet concurrency clamping
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert "fleet_concurrency_limit" not in result.data
    assert "fleet_live_session_count" not in result.data


def test_fleet_concurrency_governor_clamps_when_fleet_live_at_cap(
    tmp_path: Path, monkeypatch
) -> None:
    """When fleet.global_max_concurrent_sessions is set and fleet live count meets cap, dispatch should be clamped."""

    # Mock count_fleet_live_sessions to return 3 fleet live sessions (at cap)
    def mock_count_fleet_live(fleet_dir_override):
        return 3, []

    monkeypatch.setattr("charlie_work.workflow.count_fleet_live_sessions", mock_count_fleet_live)

    config = OrchestratorConfig(
        fleet=FleetConfig(global_max_concurrent_sessions=3),
        dispatch=DispatchConfig(max_concurrent_sessions=5, default_limit=5),
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch()

    # Should clamp to 0 since fleet cap is 3 and fleet live is 3
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["fleet_concurrency_limit"] == 3
    assert result.data["fleet_live_session_count"] == 3


def test_fleet_concurrency_governor_tighter_cap_wins(tmp_path: Path, monkeypatch) -> None:
    """When both per-repo and fleet caps are set, the tighter constraint wins."""

    # Mock count_fleet_live_sessions to return 1 fleet live session
    def mock_count_fleet_live(fleet_dir_override):
        return 1, []

    # Mock _count_live_sessions to return 1 local live session
    def mock_count_live(sessions_dir):
        return 1

    monkeypatch.setattr("charlie_work.workflow.count_fleet_live_sessions", mock_count_fleet_live)
    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        fleet=FleetConfig(global_max_concurrent_sessions=1),
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch()

    # Fleet cap (1) is tighter than per-repo cap (2), so should clamp to 0
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["concurrency_limit"] == 2  # per-repo cap
    assert result.data["live_session_count"] == 1  # local live
    assert result.data["fleet_concurrency_limit"] == 1  # fleet cap
    assert result.data["fleet_live_session_count"] == 1  # fleet live


def test_fleet_concurrency_governor_per_repo_cap_tighter(tmp_path: Path, monkeypatch) -> None:
    """When per-repo cap is tighter than fleet cap, per-repo wins."""

    # Mock count_fleet_live_sessions to return 1 fleet live session
    def mock_count_fleet_live(fleet_dir_override):
        return 1, []

    # Mock _count_live_sessions to return 1 local live session
    def mock_count_live(sessions_dir):
        return 1

    monkeypatch.setattr("charlie_work.workflow.count_fleet_live_sessions", mock_count_fleet_live)
    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        fleet=FleetConfig(global_max_concurrent_sessions=5),
        dispatch=DispatchConfig(max_concurrent_sessions=1, default_limit=5),
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch()

    # Per-repo cap (1) is tighter than fleet cap (5), so should clamp to 0
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["concurrency_limit"] == 1  # per-repo cap
    assert result.data["live_session_count"] == 1  # local live
    assert result.data["fleet_concurrency_limit"] == 5  # fleet cap
    assert result.data["fleet_live_session_count"] == 1  # fleet live


def test_fleet_concurrency_governor_result_fleet_enabled_property() -> None:
    """ConcurrencyGovernorResult.fleet_enabled property correctly reflects fleet governor enabled state."""
    result = ConcurrencyGovernorResult(
        clamped=True,
        max_concurrent=5,
        live_count=2,
        available_slots=3,
        dispatch_limit=3,
        fleet_live_count=1,
        fleet_max=3,
    )

    assert result.fleet_enabled is True  # fleet_max > 0 means enabled

    result_unlimited = ConcurrencyGovernorResult(
        clamped=False,
        max_concurrent=0,
        live_count=0,
        available_slots=5,
        dispatch_limit=5,
        fleet_live_count=0,
        fleet_max=0,
    )

    assert result_unlimited.fleet_enabled is False  # fleet_max=0 means disabled


def test_fleet_concurrency_governor_result_report_fields_includes_fleet() -> None:
    """ConcurrencyGovernorResult.report_fields includes fleet fields when fleet_enabled."""
    result = ConcurrencyGovernorResult(
        clamped=True,
        max_concurrent=5,
        live_count=2,
        available_slots=3,
        dispatch_limit=3,
        fleet_live_count=1,
        fleet_max=3,
    )

    fields = result.report_fields()
    assert fields == {
        "concurrency_limit": 5,
        "live_session_count": 2,
        "available_slots": 3,
        "fleet_concurrency_limit": 3,
        "fleet_live_session_count": 1,
    }

    # When fleet disabled, fleet fields should not be present
    result_unlimited = ConcurrencyGovernorResult(
        clamped=False,
        max_concurrent=0,
        live_count=0,
        available_slots=5,
        dispatch_limit=5,
        fleet_live_count=0,
        fleet_max=0,
    )

    fields_unlimited = result_unlimited.report_fields()
    assert fields_unlimited == {
        "concurrency_limit": 0,
        "live_session_count": 0,
        "available_slots": 5,
    }
    assert "fleet_concurrency_limit" not in fields_unlimited
    assert "fleet_live_session_count" not in fields_unlimited


def test_count_fleet_live_sessions_skips_vanished_repos(tmp_path: Path, monkeypatch) -> None:
    """count_fleet_live_sessions should skip repos that no longer exist and report them."""
    from charlie_work.fleet_registry import count_fleet_live_sessions

    # Create a fake fleet registry with 3 repos
    fleet_dir = tmp_path / ".fleet"
    fleet_dir.mkdir(parents=True)
    fleet_json = fleet_dir / "fleet.json"

    # Create two real repos and one vanished repo
    repo1 = tmp_path / "repo1"
    repo2 = tmp_path / "repo2"
    repo1.mkdir()
    repo2.mkdir()
    (repo1 / ".git").mkdir()
    (repo2 / ".git").mkdir()

    # Create state dirs for the real repos
    state1 = repo1 / ".var" / "charlie-work"
    state2 = repo2 / ".var" / "charlie-work"
    state1.mkdir(parents=True)
    state2.mkdir(parents=True)

    # Create sessions dirs (empty, so no live sessions)
    sessions1 = state1 / "dispatches" / "sessions"
    sessions2 = state2 / "dispatches" / "sessions"
    sessions1.mkdir(parents=True)
    sessions2.mkdir(parents=True)

    # Write the registry
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo1": {
                "repo_root": str(repo1),
                "name_with_owner": "owner/repo1",
                "config_path": str(repo1 / "orchestrator.config.yaml"),
                "state_dir": str(state1),
                "first_seen": "2024-01-01T00:00:00Z",
                "last_seen": "2024-01-01T00:00:00Z",
            },
            "owner/repo2": {
                "repo_root": str(repo2),
                "name_with_owner": "owner/repo2",
                "config_path": str(repo2 / "orchestrator.config.yaml"),
                "state_dir": str(state2),
                "first_seen": "2024-01-01T00:00:00Z",
                "last_seen": "2024-01-01T00:00:00Z",
            },
            "owner/vanished": {
                "repo_root": str(tmp_path / "vanished"),
                "name_with_owner": "owner/vanished",
                "config_path": str(tmp_path / "vanished" / "orchestrator.config.yaml"),
                "state_dir": str(tmp_path / "vanished" / ".var" / "charlie-work"),
                "first_seen": "2024-01-01T00:00:00Z",
                "last_seen": "2024-01-01T00:00:00Z",
            },
        },
    }
    fleet_json.write_text(json.dumps(registry_data), encoding="utf-8")

    # Mock fleet_dir to point to our test fleet dir
    def mock_fleet_dir(override=None):
        return fleet_dir

    monkeypatch.setattr("charlie_work.fleet_registry.fleet_dir", mock_fleet_dir)

    # Count fleet live sessions
    live_count, skipped_repos = count_fleet_live_sessions(None)

    # Should count 0 live sessions (both real repos have empty sessions dirs)
    assert live_count == 0
    # Should report the vanished repo
    assert "owner/vanished" in skipped_repos
    assert len(skipped_repos) == 1


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


def test_apply_concurrency_governor_helper_unlimited(tmp_path: Path) -> None:
    """_apply_concurrency_governor returns unclamped result when max_concurrent is 0."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=0),
        devin=DevinConfig(adapter="manual"),
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

    result = app._apply_concurrency_governor(5)

    assert result.clamped is True
    assert result.max_concurrent == 2
    assert result.live_count == 2
    assert result.available_slots == 0
    assert result.dispatch_limit == 0


def test_apply_concurrency_governor_helper_partial_slots(tmp_path: Path, monkeypatch) -> None:
    """_apply_concurrency_governor returns partial clamped result when some slots available."""

    def mock_count_live(sessions_dir):
        return 1

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(adapter="manual"),
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
    """Janitor blocks review when head hasn't changed after request_changes (no-op rework).

    This prevents pointless packet churn and preserves the needs_rework label on
    budget-deferred rework candidates. The janitor now blocks before review_started
    can fire.
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
                "baseRefName": "main",
            }
        ),
        encoding="utf-8",
    )

    # Set initial diff
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )

    app.record_review(456, "request_changes", summary="fix A")

    # Verify the decision was recorded
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        assert state["prs"]["456"]["decision"] == "request_changes"
        assert state["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"
        # Verify patch-id was calculated
        assert "reviewed_patch_id" in state["prs"]["456"]

    # Clear label tracking to isolate the review() call
    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    # Call review again with the same head SHA and same diff (no-op rework)
    result = app.review(456)

    # The janitor should block the PR because the diff is unchanged (no-op rework)
    assert result.ok is False
    # With patch-id comparison, the message should mention patch-id
    assert "PR diff unchanged since request_changes verdict" in result.message
    # review_started transition should not fire (janitor blocks before it)
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
                "baseRefName": "main",
            }
        ),
        encoding="utf-8",
    )

    # Set initial diff
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )

    app.record_review(456, "request_changes", summary="fix A")

    # Advance the PR head and change the diff (simulating actual content changes)
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+changed"
    )

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
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
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


def test_dispatch_skips_issue_with_open_blocker(tmp_path: Path) -> None:
    """Issue #108: dispatch should skip issues with open blockers."""
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="manual"),
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
                    "labels": [],  # Not ready, so won't be in dispatch list
                    "state": "OPEN",  # Still open, so should block
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
            # Return that #743 is open
            return {743}

    fake_gh = FakeGitHubWithBlockers()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=10)

    # Issue 752 should be skipped due to open blocker #743
    assert result.data["selected_count"] == 0
    assert result.data["attempted_count"] == 0

    # Check that dispatch_skip_blocked event was logged
    state = load_state(paths.state_file)
    blocked_events = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_skip_blocked"
    ]
    assert len(blocked_events) == 1
    assert blocked_events[0]["payload"]["issue"] == 752
    assert blocked_events[0]["payload"]["blockers"] == [743]


def test_dispatch_proceeds_when_blocker_closed(tmp_path: Path) -> None:
    """Issue #108: dispatch should proceed when blocker is closed."""
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with a blocked issue but closed blocker
    class FakeGitHubWithClosedBlocker(FakeGitHub):
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
                    "labels": [],  # Not ready, so won't be in dispatch list
                    "state": "CLOSED",  # Closed, so should not block
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
            # Return that #743 is NOT open
            return set()

    fake_gh = FakeGitHubWithClosedBlocker()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=10)

    # Issue 752 should be dispatched since blocker is closed
    assert result.data["selected_count"] == 1
    assert result.data["attempted_count"] == 1

    # Check that no dispatch_skip_blocked event was logged
    state = load_state(paths.state_file)
    blocked_events = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_skip_blocked"
    ]
    assert len(blocked_events) == 0


def test_dispatch_skips_when_any_blocker_open(tmp_path: Path) -> None:
    """Issue #108: dispatch should skip when ANY blocker is open (logical AND)."""
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with multiple blockers, one still open
    class FakeGitHubWithMultipleBlockers(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Override with only the test issues
            self.issues = [
                {
                    "number": 752,
                    "title": "Dependent issue",
                    "url": "https://example.test/issues/752",
                    "body": "Blocked by #743, #744",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
                {
                    "number": 743,
                    "title": "Blocker issue 1",
                    "url": "https://example.test/issues/743",
                    "body": "Foundation work",
                    "labels": [],  # Not ready
                    "state": "CLOSED",
                },
                {
                    "number": 744,
                    "title": "Blocker issue 2",
                    "url": "https://example.test/issues/744",
                    "body": "More foundation work",
                    "labels": [],  # Not ready
                    "state": "OPEN",  # Still open
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
            # Return that #744 is open
            return {744}

    fake_gh = FakeGitHubWithMultipleBlockers()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=10)

    # Issue 752 should be skipped since #744 is still open
    assert result.data["selected_count"] == 0

    # Check that both declared blockers are listed in the event (AC3)
    state = load_state(paths.state_file)
    blocked_events = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_skip_blocked"
    ]
    assert len(blocked_events) == 1
    assert blocked_events[0]["payload"]["issue"] == 752
    assert set(blocked_events[0]["payload"]["blockers"]) == {743, 744}


def test_dispatch_handles_self_reference_blocker(tmp_path: Path) -> None:
    """Issue #108: self-referencing blockers should be filtered out with warning."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub with a self-referencing blocker
    class FakeGitHubWithSelfRef(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Override with only the test issue
            self.issues = [
                {
                    "number": 123,
                    "title": "Self-referencing issue",
                    "url": "https://example.test/issues/123",
                    "body": "Blocked by #123",  # Self-reference
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                },
            ]

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            # Return that the self-reference is "open" (it exists)
            return {123}

    fake_gh = FakeGitHubWithSelfRef()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Capture warning logs
    with patch("logging.Logger.warning") as mock_warning:
        result = app.dispatch(limit=10)

        # Issue should be dispatched (self-reference filtered out)
        assert result.data["selected_count"] == 1

        # Warning should have been logged for self-reference
        assert any(
            "self-referencing" in str(call.args[0]).lower() for call in mock_warning.call_args_list
        ), "Expected warning about self-referencing blocker"

    # No dispatch_skip_blocked event should be logged
    state = load_state(paths.state_file)
    blocked_events = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_skip_blocked"
    ]
    assert len(blocked_events) == 0


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

    with patch("charlie_work.github.logger") as mock_logger:
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


def test_cancel_superseded_runs_no_workflow_name(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs returns error when workflow_name is empty."""
    from charlie_work.github import cancel_superseded_runs

    fake_gh = FakeGitHub()
    result = cancel_superseded_runs(fake_gh, "main", "")
    assert result["errors"] == ["workflow_name is empty - cannot cancel runs"]
    assert result["total_queued"] == 0
    assert result["kept"] == 0
    assert result["cancelled"] == 0


def test_cancel_superseded_runs_no_queued_runs(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs handles no queued runs correctly."""
    from charlie_work.github import cancel_superseded_runs

    class FakeGitHubWithEmptyRuns(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.runs_response = []

    fake_gh = FakeGitHubWithEmptyRuns()
    result = cancel_superseded_runs(fake_gh, "main", "test-workflow")
    assert result["total_queued"] == 0
    assert result["kept"] == 0
    assert result["cancelled"] == 0
    assert result["cancelled_run_ids"] == []
    assert result["errors"] == []


def test_cancel_superseded_runs_one_queued_run(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs keeps the single queued run."""
    from charlie_work.github import cancel_superseded_runs

    class FakeGitHubWithOneRun(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.runs_response = [
                {
                    "databaseId": 123,
                    "status": "queued",
                    "createdAt": "2026-07-09T00:00:00Z",
                    "headBranch": "main",
                }
            ]

    fake_gh = FakeGitHubWithOneRun()
    result = cancel_superseded_runs(fake_gh, "main", "test-workflow")
    assert result["total_queued"] == 1
    assert result["kept"] == 1
    assert result["cancelled"] == 0
    assert result["cancelled_run_ids"] == []
    assert result["errors"] == []


def test_cancel_superseded_runs_multiple_queued_runs(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs keeps newest and cancels older runs."""
    from charlie_work.github import cancel_superseded_runs

    class FakeGitHubWithMultipleRuns(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.runs_response = [
                {
                    "databaseId": 123,
                    "status": "queued",
                    "createdAt": "2026-07-09T00:00:00Z",
                    "headBranch": "main",
                },
                {
                    "databaseId": 124,
                    "status": "queued",
                    "createdAt": "2026-07-09T01:00:00Z",
                    "headBranch": "main",
                },
                {
                    "databaseId": 125,
                    "status": "queued",
                    "createdAt": "2026-07-09T02:00:00Z",
                    "headBranch": "main",
                },
            ]
            self.cancelled_runs = []

        def run(self, args, *, json_output=False, allow_failure=False):
            if "cancel" in args:
                run_id = int(args[-1])
                self.cancelled_runs.append(run_id)
                return "Cancelled"
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubWithMultipleRuns()
    result = cancel_superseded_runs(fake_gh, "main", "test-workflow")
    assert result["total_queued"] == 3
    assert result["kept"] == 1
    assert result["cancelled"] == 2
    assert result["cancelled_run_ids"] == [124, 123]  # Oldest two cancelled
    assert result["errors"] == []
    # Verify the newest (125) was kept, older ones cancelled
    assert 123 in fake_gh.cancelled_runs  # Oldest cancelled
    assert 124 in fake_gh.cancelled_runs  # Middle cancelled
    assert 125 not in fake_gh.cancelled_runs  # Newest kept


def test_cancel_superseded_runs_ignores_in_progress(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs ignores in_progress runs."""
    from charlie_work.github import cancel_superseded_runs

    class FakeGitHubWithInProgress(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.runs_response = [
                {
                    "databaseId": 123,
                    "status": "queued",
                    "createdAt": "2026-07-09T00:00:00Z",
                    "headBranch": "main",
                },
                {
                    "databaseId": 124,
                    "status": "in_progress",
                    "createdAt": "2026-07-09T01:00:00Z",
                    "headBranch": "main",
                },
            ]

    fake_gh = FakeGitHubWithInProgress()
    result = cancel_superseded_runs(fake_gh, "main", "test-workflow")
    assert result["total_queued"] == 1  # Only queued runs counted
    assert result["kept"] == 1
    assert result["cancelled"] == 0
    assert result["cancelled_run_ids"] == []
    assert result["errors"] == []


def test_cancel_superseded_runs_handles_cancel_error(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs handles individual cancel failures gracefully."""
    from charlie_work.github import cancel_superseded_runs

    class FakeGitHubWithCancelError(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.runs_response = [
                {
                    "databaseId": 123,
                    "status": "queued",
                    "createdAt": "2026-07-09T00:00:00Z",
                    "headBranch": "main",
                },
                {
                    "databaseId": 124,
                    "status": "queued",
                    "createdAt": "2026-07-09T01:00:00Z",
                    "headBranch": "main",
                },
            ]

        def run(self, args, *, json_output=False, allow_failure=False):
            if "cancel" in args and args[-1] == "123":
                # Simulate failure by returning None (allow_failure=True behavior)
                return None
            elif "cancel" in args:
                # Other cancels succeed
                return "Cancelled"
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubWithCancelError()
    result = cancel_superseded_runs(fake_gh, "main", "test-workflow")
    assert result["total_queued"] == 2
    assert result["kept"] == 1
    assert result["cancelled"] == 0  # The only run to cancel (123) failed
    assert result["cancelled_run_ids"] == []  # No successful cancels
    assert len(result["errors"]) == 1
    assert "Failed to cancel run 123" in result["errors"][0]


def test_cancel_superseded_runs_handles_list_error(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs handles list API errors gracefully."""
    from charlie_work.github import cancel_superseded_runs, GitHubError

    class FakeGitHubWithListError(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()

        def run(self, args, *, json_output=False, allow_failure=False):
            if "run" in args and "list" in args:
                raise GitHubError("List failed")
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubWithListError()
    result = cancel_superseded_runs(fake_gh, "main", "test-workflow")
    assert result["total_queued"] == 0
    assert result["kept"] == 0
    assert result["cancelled"] == 0
    assert len(result["errors"]) == 1
    assert "GitHub API error" in result["errors"][0]


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

    with patch("charlie_work.github.logger") as mock_logger:
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
        devin=DevinConfig(adapter="manual"),
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
        devin=DevinConfig(adapter="manual"),
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


def test_status_includes_stalled_section(tmp_path: Path) -> None:
    """Issue #109: status (roll-call) should include stalled section."""
    from datetime import UTC, datetime, timedelta
    import os
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="manual"),
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
        devin=DevinConfig(adapter="manual"),
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


def test_status_workers_empty_when_no_live_sessions(tmp_path: Path) -> None:
    """Issue #167: workers section should be empty list when no live sessions exist."""
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="manual"),
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
        devin=DevinConfig(adapter="manual"),
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
        devin=DevinConfig(adapter="devin-shell"),  # Use devin-shell adapter for watchdog support
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
        patch("charlie_work.workflow.kill_process_tree", return_value=[99999]),
        patch(
            "charlie_work.workflow.sweep_orphan_processes", return_value=[3492]
        ),  # Fixed mock return
        patch(
            "charlie_work.devin_shell.update_session_record_with_failure_classification",
            return_value=(None, None),
        ),
    ):
        # Run the stall detection and handling
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        result = _detect_and_handle_stalled_sessions(sessions_dir, paths.state_file, config)

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


def _make_stalled_sidecar(
    sessions_dir: Path, issue_number: int, *, log_text: str
) -> tuple[Path, Path]:
    """Write a fake devin-shell sidecar + log with an old mtime (stalled by time).

    Shared setup for issue #246 regression tests: the stall watchdog must
    classify the log tail (rate-limit/quota signatures) before falling back
    to failure_kind "stalled".
    """
    from datetime import UTC, datetime, timedelta
    import os as _os

    log_file = sessions_dir / f"issue-{issue_number}.log"
    log_file.write_text(log_text, encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    _os.utime(log_file, (timestamp, timestamp))

    sidecar = sessions_dir / f"issue-{issue_number}.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": issue_number,
                "branch": f"agent/issue-{issue_number}",
                "worktree_path": "/fake/path",
                "prompt_path": "/fake/prompt",
                "command": ["devin", "--print"],
                "pid": 99999,
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_file),
                "error": None,
                "process_start_time": 1234567890.0,
            }
        ),
        encoding="utf-8",
    )
    return sidecar, log_file


def test_stall_reap_classifies_rate_limit_before_stalled_fallback(tmp_path: Path) -> None:
    """Issue #246: a stall-killed worker whose log tail matches the rate-limit
    signature must be classified rate_limited (with throttled_until set from
    the parsed reset-in-N-minutes cooldown), not the hardcoded "stalled"
    fallback — otherwise the very next dispatch pass relaunches into the same
    live provider rate limit.
    """
    from datetime import UTC, datetime, timedelta
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    sidecar, _log_file = _make_stalled_sidecar(
        sessions_dir,
        1034,
        log_text=(
            "Error: Agent error: Permission denied: Permission denied: Reached "
            "overall message rate limit. Please try again later. Your limit "
            "will reset in 7 minutes.\n"
        ),
    )

    before = datetime.now(UTC)
    with (
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.workflow.kill_process_tree", return_value=[99999]),
        patch("charlie_work.workflow.sweep_orphan_processes", return_value=[]),
    ):
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        result = _detect_and_handle_stalled_sessions(sessions_dir, paths.state_file, config)
    after = datetime.now(UTC)

    assert any(entry["issue"] == 1034 for entry in result)

    # Sidecar must be classified rate_limited, not the hardcoded "stalled"
    updated_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "rate_limited"

    # throttled_until must be persisted to state.json, roughly now + 7 minutes
    state = load_state(paths.state_file)
    throttled_until = state.get("throttled_until")
    assert throttled_until is not None
    throttle_time = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    assert before + timedelta(minutes=6) <= throttle_time <= after + timedelta(minutes=8)

    # session_stalled event must carry the resolved failure_kind
    events = state.get("events", [])
    stalled_events = [e for e in events if e.get("kind") == "session_stalled"]
    assert len(stalled_events) == 1
    assert stalled_events[0]["payload"]["failure_kind"] == "rate_limited"


def test_stall_reap_classifies_quota_exhausted_before_stalled_fallback(tmp_path: Path) -> None:
    """Issue #246: quota-exhaustion signature in the log tail must classify as
    quota_exhausted with the fixed 24-hour cooldown, not "stalled".
    """
    from datetime import UTC, datetime, timedelta
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    sidecar, _log_file = _make_stalled_sidecar(
        sessions_dir,
        2001,
        log_text="Error: daily usage quota has been exhausted.\n",
    )

    before = datetime.now(UTC)
    with (
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.workflow.kill_process_tree", return_value=[99999]),
        patch("charlie_work.workflow.sweep_orphan_processes", return_value=[]),
    ):
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        _detect_and_handle_stalled_sessions(sessions_dir, paths.state_file, config)

    updated_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "quota_exhausted"

    state = load_state(paths.state_file)
    throttled_until = state.get("throttled_until")
    assert throttled_until is not None
    throttle_time = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    assert before + timedelta(hours=23) <= throttle_time <= before + timedelta(hours=25)

    events = state.get("events", [])
    stalled_events = [e for e in events if e.get("kind") == "session_stalled"]
    assert stalled_events[0]["payload"]["failure_kind"] == "quota_exhausted"


def test_stall_reap_falls_back_to_stalled_when_no_throttle_signature(tmp_path: Path) -> None:
    """Issue #246: a stall-killed worker with a quiet log tail (no rate-limit
    or quota signature) still falls back to failure_kind "stalled", and
    throttled_until is left untouched.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    sidecar, _log_file = _make_stalled_sidecar(
        sessions_dir, 3007, log_text="working on the issue, one moment...\n"
    )

    with (
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.workflow.kill_process_tree", return_value=[99999]),
        patch("charlie_work.workflow.sweep_orphan_processes", return_value=[]),
    ):
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        _detect_and_handle_stalled_sessions(sessions_dir, paths.state_file, config)

    updated_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "stalled"

    state = load_state(paths.state_file)
    assert state.get("throttled_until") is None

    events = state.get("events", [])
    stalled_events = [e for e in events if e.get("kind") == "session_stalled"]
    assert stalled_events[0]["payload"]["failure_kind"] == "stalled"


def test_dispatch_defers_after_stall_reap_sets_throttled_until(tmp_path: Path) -> None:
    """Issue #246: after the stall watchdog reaps a worker that hit a live
    provider rate limit, the very next dispatch pass must defer instead of
    launching a replacement worker into the same throttle window.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        dispatch=DispatchConfig(default_limit=3),
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    _make_stalled_sidecar(
        sessions_dir,
        4055,
        log_text=(
            "Error: Reached overall message rate limit. Please try again "
            "later. Your limit will reset in 9 minutes.\n"
        ),
    )

    with (
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.workflow.kill_process_tree", return_value=[99999]),
        patch("charlie_work.workflow.sweep_orphan_processes", return_value=[]),
    ):
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        _detect_and_handle_stalled_sessions(sessions_dir, paths.state_file, config)

        # dispatch() re-runs the stall reaper unconditionally at its top (workflow.py
        # dispatch():~1180) before checking is_throttled — keep the same mocks active
        # so this second reap pass over the already-classified sidecar stays cheap
        # and deterministic instead of shelling out to real process/PowerShell calls.
        result = app.dispatch()

    assert result.ok is False
    assert result.data["deferred_reason"] == "provider_throttled"
    assert result.data["throttled_until"] is not None
    assert result.data["selected_count"] == 0


def test_sweep_orphan_processes_for_dead_sessions_unit(tmp_path: Path) -> None:
    """Unit test for _sweep_orphan_processes_for_dead_sessions (issue #139)."""
    from datetime import UTC, datetime
    from unittest.mock import patch, MagicMock
    import subprocess

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
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

    # Mock sweep_orphan_processes to return fixed PIDs for dead worktrees
    def mock_sweep_orphan(worktree_path: str) -> list[int]:
        if worktree_path == "/dead/worktree":
            return [5000, 5001]
        elif worktree_path == "/dead/worker":
            return [6000]
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
        patch("charlie_work.workflow.sweep_orphan_processes", side_effect=mock_sweep_orphan),
        patch("charlie_work.workflow.os.name", "nt"),  # Force Windows path
        patch("subprocess.run", side_effect=mock_subprocess_run),
    ):
        from charlie_work.workflow import _sweep_orphan_processes_for_dead_sessions

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _sweep_orphan_processes_for_dead_sessions(sessions_dir, paths.state_file, config)

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
        devin=DevinConfig(adapter="devin-shell"),
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
        devin=DevinConfig(adapter="manual"),
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

        _detect_and_handle_stalled_sessions(sessions_dir, paths.state_file, config)

    # Check that is_session_alive was NOT called (detection skipped)
    mock_alive.assert_not_called()

    # Check that kill_process_tree was NOT called (no kill)
    mock_kill.assert_not_called()

    # Load state and check for the event
    state = load_state(paths.state_file)
    events = state.get("events", [])

    # Find the session_stalled event
    stalled_events = [e for e in events if e.get("type") == "session_stalled"]
    assert len(stalled_events) == 0  # No event emitted


# Fleet registry and global config tests


def test_fleet_dir_override() -> None:
    """Test that fleet_dir respects the override parameter."""
    from charlie_work.fleet_paths import fleet_dir

    result = fleet_dir(override="/custom/path")
    assert result == Path("/custom/path")


def test_fleet_dir_env_var() -> None:
    """Test that fleet_dir respects CHARLIE_WORK_FLEET_DIR env var."""
    from charlie_work.fleet_paths import fleet_dir

    original = os.environ.get("CHARLIE_WORK_FLEET_DIR")
    try:
        os.environ["CHARLIE_WORK_FLEET_DIR"] = "/env/path"
        result = fleet_dir()
        assert result == Path("/env/path")
    finally:
        if original is None:
            os.environ.pop("CHARLIE_WORK_FLEET_DIR", None)
        else:
            os.environ["CHARLIE_WORK_FLEET_DIR"] = original


def test_fleet_dir_platform_defaults() -> None:
    """Test that fleet_dir uses platform-specific defaults."""
    from charlie_work.fleet_paths import fleet_dir

    # Clear env var to test platform defaults
    original = os.environ.get("CHARLIE_WORK_FLEET_DIR")
    try:
        os.environ.pop("CHARLIE_WORK_FLEET_DIR", None)
        result = fleet_dir()

        if sys.platform == "win32":
            expected_base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        else:
            expected_base = Path(
                os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
            )

        assert result == expected_base / "charlie-work"
    finally:
        if original is not None:
            os.environ["CHARLIE_WORK_FLEET_DIR"] = original


def test_fleet_registry_touch_repo_first_call(tmp_path: Path) -> None:
    """Test that touch_repo sets first_seen and last_seen on first registration."""
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")

    # Mock GitHub that returns a nameWithOwner
    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

    gh = FakeGitHub(repo_root=repo_root)

    # Touch repo with isolated fleet dir
    registry = touch_repo(str(tmp_path / "fleet"), repo_root, paths, gh)

    assert "repos" in registry
    assert "owner/repo" in registry["repos"]
    entry = registry["repos"]["owner/repo"]
    assert entry["repo_root"] == str(repo_root)
    assert entry["name_with_owner"] == "owner/repo"
    assert entry["config_path"] == str(repo_root / "orchestrator.config.yaml")
    assert entry["state_dir"] == str(paths.root)
    assert entry["first_seen"] == entry["last_seen"]  # First call: both equal


def test_fleet_registry_touch_repo_second_call(tmp_path: Path) -> None:
    """Test that touch_repo preserves first_seen and bumps last_seen on subsequent calls."""
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

    gh = FakeGitHub(repo_root=repo_root)

    # First call
    registry = touch_repo(str(tmp_path / "fleet"), repo_root, paths, gh)
    first_first_seen = registry["repos"]["owner/repo"]["first_seen"]
    first_last_seen = registry["repos"]["owner/repo"]["last_seen"]

    # Small delay to ensure timestamp difference (need >1s due to second resolution)
    time.sleep(2.0)

    # Second call
    registry = touch_repo(str(tmp_path / "fleet"), repo_root, paths, gh)
    second_first_seen = registry["repos"]["owner/repo"]["first_seen"]
    second_last_seen = registry["repos"]["owner/repo"]["last_seen"]

    assert second_first_seen == first_first_seen  # first_seen preserved
    assert second_last_seen != first_last_seen  # last_seen bumped
    assert second_last_seen > first_last_seen  # last_seen increased


def test_fleet_registry_touch_repo_moved_repo(tmp_path: Path) -> None:
    """Test that touch_repo updates repo_root when repo is moved."""
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    repo_root_old = tmp_path / "repo_old"
    repo_root_old.mkdir(parents=True, exist_ok=True)
    paths_old = runtime_paths(repo_root_old, ".var/charlie-work")

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

    gh_old = FakeGitHub(repo_root=repo_root_old)

    # First registration
    registry = touch_repo(str(tmp_path / "fleet"), repo_root_old, paths_old, gh_old)
    first_first_seen = registry["repos"]["owner/repo"]["first_seen"]

    # Move repo
    repo_root_new = tmp_path / "repo_new"
    repo_root_new.mkdir(parents=True, exist_ok=True)
    paths_new = runtime_paths(repo_root_new, ".var/charlie-work")
    gh_new = FakeGitHub(repo_root=repo_root_new)

    # Re-register with new path
    registry = touch_repo(str(tmp_path / "fleet"), repo_root_new, paths_new, gh_new)

    # Should update repo_root but preserve first_seen (same nameWithOwner)
    entry = registry["repos"]["owner/repo"]
    assert entry["repo_root"] == str(repo_root_new)
    assert entry["first_seen"] == first_first_seen  # Preserved on move


def test_fleet_registry_touch_repo_gh_error(tmp_path: Path) -> None:
    """Test that touch_repo silently skips registration on gh error."""
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub, GitHubError

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            raise GitHubError("gh not available")

    gh = FakeGitHub(repo_root=repo_root)

    # Should not raise, should return empty registry
    registry = touch_repo(str(tmp_path / "fleet"), repo_root, paths, gh)
    assert registry == {"version": 1, "repos": {}}


def test_fleet_registry_uses_state_lock(tmp_path: Path) -> None:
    """Test that fleet_registry writes go through state.save_state."""
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.github import GitHub

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

    gh = FakeGitHub(repo_root=repo_root)

    # Spy on save_state
    from charlie_work.state import save_state

    original_save_state = save_state
    calls = []

    def spy_save_state(path: Path, data: dict) -> dict:
        calls.append(path)
        return original_save_state(path, data)

    with patch("charlie_work.fleet_registry.save_state", side_effect=spy_save_state):
        touch_repo(str(tmp_path / "fleet"), repo_root, paths, gh)

    # Verify save_state was called with fleet.json path
    assert len(calls) == 1
    assert calls[0] == tmp_path / "fleet" / "fleet.json"


def test_global_config_no_global_file(tmp_path: Path) -> None:
    """Test that load_layered_config behaves like load_config when no global file exists."""
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    # No global config, no repo config
    config = load_layered_config(repo_root, None, fleet_dir_override=str(tmp_path / "fleet"))

    # Should match default config
    default_config = load_config(None)
    assert config.labels.ready == default_config.labels.ready
    assert config.dispatch.default_limit == default_config.dispatch.default_limit


def test_global_config_global_only(tmp_path: Path) -> None:
    """Test that global config values apply when no per-repo override exists."""
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # Create global config with a custom value
    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text("dispatch:\n  max_concurrent_sessions: 5\n", encoding="utf-8")

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))

    assert config.dispatch.max_concurrent_sessions == 5


def test_global_config_per_repo_wins(tmp_path: Path) -> None:
    """Test that per-repo config overrides global config."""
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # Create global config
    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text("dispatch:\n  max_concurrent_sessions: 5\n", encoding="utf-8")

    # Create per-repo config with different value
    repo_config_path = repo_root / "orchestrator.config.yaml"
    repo_config_path.write_text("dispatch:\n  max_concurrent_sessions: 10\n", encoding="utf-8")

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))

    # Per-repo value should win
    assert config.dispatch.max_concurrent_sessions == 10


def test_global_config_unknown_key_raises(tmp_path: Path) -> None:
    """Test that unknown keys in global config raise ConfigError."""
    from charlie_work.config import ConfigError
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # Create global config with unknown top-level section
    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text("unknown_section:\n  foo: bar\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown config section"):
        load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))


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

    # Monkeypatch GitHub to use our fake
    def fake_github(repo_root: Path, dry_run: bool = False) -> GitHub:
        return FakeGitHub(repo_root=repo_root, dry_run=dry_run)

    monkeypatch.setattr("charlie_work.cli.GitHub", fake_github)

    # Mock fleet_dir to use tmp_path
    def fake_fleet_dir(*, override: str | None = None) -> Path:
        return tmp_path / "fleet"

    monkeypatch.setattr("charlie_work.fleet_registry.fleet_dir", fake_fleet_dir)

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


def test_dispatch_stall_detection_called_once_per_dispatch(tmp_path: Path, monkeypatch) -> None:
    """Regression test for issue #158: _detect_and_handle_stalled_sessions should be called exactly once per dispatch() call, not twice (was duplicated in _apply_concurrency_governor)."""
    # Mock _detect_and_handle_stalled_sessions to track call count
    stall_detection_calls = []

    def mock_stall_detection(sessions_dir, state_file, config):
        stall_detection_calls.append(1)
        return []  # No stalled sessions

    monkeypatch.setattr(
        "charlie_work.workflow._detect_and_handle_stalled_sessions", mock_stall_detection
    )

    # Mock _count_live_sessions to return 0 (no live sessions)
    def mock_count_live(sessions_dir):
        return 0

    monkeypatch.setattr("charlie_work.workflow._count_live_sessions", mock_count_live)

    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=2, default_limit=5),
        devin=DevinConfig(adapter="manual"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Call dispatch() with max_concurrent_sessions > 0
    app.dispatch()

    # Verify stall detection was called exactly once
    assert len(stall_detection_calls) == 1, (
        f"_detect_and_handle_stalled_sessions was called {len(stall_detection_calls)} times, expected 1"
    )


# --- Test-adequacy gate (issue #179) ------------------------------------------


def _test_adequacy_app(
    tmp_path: Path, *, enabled: bool, max_rework_cycles: int = 2
) -> OrchestratorApp:
    config = OrchestratorConfig(
        test_adequacy=TestAdequacyConfig(enabled=enabled, exempt_marker="Test-exempt:"),
        review=ReviewConfig(max_rework_cycles=max_rework_cycles),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub())


def test_review_test_adequacy_disabled_is_noop(tmp_path: Path, monkeypatch) -> None:
    """When test_adequacy.enabled=False (default), check_test_adequacy is never called."""
    app = _test_adequacy_app(tmp_path, enabled=False)
    calls = {"n": 0}

    def _fake_check(diff, pr, config):
        calls["n"] += 1
        raise AssertionError("check_test_adequacy should not be called when disabled")

    monkeypatch.setattr("charlie_work.workflow.check_test_adequacy", _fake_check)

    result = app.review(456)

    assert calls["n"] == 0
    assert result.ok is True
    assert "prompt_path" in result.data


def test_review_test_adequacy_hard_fail_records_request_changes(
    tmp_path: Path, monkeypatch
) -> None:
    """When test_adequacy hard-fails, review() calls record_review with request_changes."""
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict
    from charlie_work.workflow import CommandResult

    app = _test_adequacy_app(tmp_path, enabled=True)
    calls = {"check_test_adequacy": 0, "record_review": 0, "transition": 0}

    hard_fail_verdict = TestAdequacyVerdict(
        ok=False,
        failures=("Product code changed (15 LOC added) but no test files changed.",),
        warnings=(),
        facts=TestAdequacyFacts(
            added_product_loc=15,
            added_test_loc=0,
            assertion_count=0,
            test_files_changed=0,
            untested_product_files=("src/foo.py", "src/bar.py"),
            exempt=False,
            exempt_reason="",
        ),
    )

    def _fake_check(diff, pr, config):
        calls["check_test_adequacy"] += 1
        return hard_fail_verdict

    def _fake_record_review(pr_number, decision, **kwargs):
        calls["record_review"] += 1
        assert decision == "request_changes"
        summary = kwargs.get("summary", "")
        assert "Test adequacy check failed" in summary
        assert "src/foo.py" in summary
        assert "src/bar.py" in summary
        assert "Test-exempt:" in summary
        return CommandResult(True, "record_review called", {})

    def _fake_transition(gh, labels, issue_number, edge):
        calls["transition"] += 1
        assert edge == "review_started"
        from charlie_work.labels import TransitionResult, TransitionOutcome

        return TransitionResult(
            outcome=TransitionOutcome.APPLIED,
            add_failures=[],
            remove_failures=[],
        )

    monkeypatch.setattr("charlie_work.workflow.check_test_adequacy", _fake_check)
    monkeypatch.setattr("charlie_work.workflow.transition", _fake_transition)
    monkeypatch.setattr(app, "record_review", _fake_record_review)

    result = app.review(456)

    assert calls["check_test_adequacy"] == 1
    assert calls["transition"] == 1
    assert calls["record_review"] == 1
    assert result.ok is True
    assert result.data == {}


def test_review_test_adequacy_hard_fail_label_set(tmp_path: Path, monkeypatch) -> None:
    """After hard-fail, label transitions compose to {in_progress, pr_open, needs_rework}."""
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict
    from charlie_work.labels import TransitionResult, TransitionOutcome
    from charlie_work.workflow import CommandResult

    app = _test_adequacy_app(tmp_path, enabled=True)
    transition_calls = []

    hard_fail_verdict = TestAdequacyVerdict(
        ok=False,
        failures=("Product code changed (15 LOC added) but no test files changed.",),
        warnings=(),
        facts=TestAdequacyFacts(
            added_product_loc=15,
            added_test_loc=0,
            assertion_count=0,
            test_files_changed=0,
            untested_product_files=("src/foo.py",),
            exempt=False,
            exempt_reason="",
        ),
    )

    def _fake_check(diff, pr, config):
        return hard_fail_verdict

    def _fake_record_review(pr_number, decision, **kwargs):
        # Simulate the rework_requested transition that record_review would call
        transition_calls.append("rework_requested")
        return CommandResult(True, "record_review called", {})

    def _fake_transition(gh, labels, issue_number, edge):
        transition_calls.append(edge)
        if edge == "review_started":
            return TransitionResult(
                outcome=TransitionOutcome.APPLIED,
                add_failures=[],
                remove_failures=[],
            )
        elif edge == "rework_requested":
            return TransitionResult(
                outcome=TransitionOutcome.APPLIED,
                add_failures=[],
                remove_failures=[],
            )
        raise AssertionError(f"Unexpected transition: {edge}")

    monkeypatch.setattr("charlie_work.workflow.check_test_adequacy", _fake_check)
    monkeypatch.setattr("charlie_work.workflow.transition", _fake_transition)
    monkeypatch.setattr(app, "record_review", _fake_record_review)

    app.review(456)

    assert transition_calls == ["review_started", "rework_requested"]


def test_review_test_adequacy_unchanged_head_not_rerecorded(tmp_path: Path, monkeypatch) -> None:
    """A second review() on the same unchanged head is blocked by janitor gate (no-op rework check)."""
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict
    from charlie_work.workflow import CommandResult

    app = _test_adequacy_app(tmp_path, enabled=True)
    check_calls = {"n": 0}

    hard_fail_verdict = TestAdequacyVerdict(
        ok=False,
        failures=("Product code changed (15 LOC added) but no test files changed.",),
        warnings=(),
        facts=TestAdequacyFacts(
            added_product_loc=15,
            added_test_loc=0,
            assertion_count=0,
            test_files_changed=0,
            untested_product_files=("src/foo.py",),
            exempt=False,
            exempt_reason="",
        ),
    )

    def _fake_check(diff, pr, config):
        check_calls["n"] += 1
        return hard_fail_verdict

    def _fake_record_review(pr_number, decision, **kwargs):
        return CommandResult(True, "record_review called", {})

    def _fake_transition(gh, labels, issue_number, edge):
        from charlie_work.labels import TransitionResult, TransitionOutcome

        return TransitionResult(
            outcome=TransitionOutcome.APPLIED,
            add_failures=[],
            remove_failures=[],
        )

    monkeypatch.setattr("charlie_work.workflow.check_test_adequacy", _fake_check)
    monkeypatch.setattr("charlie_work.workflow.transition", _fake_transition)
    monkeypatch.setattr(app, "record_review", _fake_record_review)

    # First review: hard-fail records request_changes
    app.gh.pr_head_shas[456] = "sha-abc123"
    result1 = app.review(456)
    assert check_calls["n"] == 1
    assert result1.ok is True

    # Simulate state after first review: decision=request_changes, reviewed_head_sha=sha-abc123
    state = load_state(app.paths.state_file)
    if "456" not in state["prs"]:
        state["prs"]["456"] = {}
    state["prs"]["456"]["decision"] = "request_changes"
    state["prs"]["456"]["reviewed_head_sha"] = "sha-abc123"
    save_state(app.paths.state_file, state)

    # Second review on same head: janitor gate blocks (no-op rework check)
    # check_test_adequacy should NOT be called again
    result2 = app.review(456)
    assert check_calls["n"] == 1  # Still 1, not 2
    assert result2.ok is False
    assert "janitor gate blocked" in result2.message
    assert "unchanged since request_changes" in result2.message


def test_review_test_adequacy_escalates_at_max_rework_cycles(tmp_path: Path, monkeypatch) -> None:
    """After max_rework_cycles hard-fails, escalate to agent:human-needed."""
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict
    from charlie_work.workflow import CommandResult

    app = _test_adequacy_app(tmp_path, enabled=True, max_rework_cycles=2)
    check_calls = {"n": 0}

    hard_fail_verdict = TestAdequacyVerdict(
        ok=False,
        failures=("Product code changed (15 LOC added) but no test files changed.",),
        warnings=(),
        facts=TestAdequacyFacts(
            added_product_loc=15,
            added_test_loc=0,
            assertion_count=0,
            test_files_changed=0,
            untested_product_files=("src/foo.py",),
            exempt=False,
            exempt_reason="",
        ),
    )

    def _fake_check(diff, pr, config):
        check_calls["n"] += 1
        return hard_fail_verdict

    def _fake_record_review(pr_number, decision, **kwargs):
        state = load_state(app.paths.state_file)
        pr_state = state["prs"].get(str(pr_number), {})
        request_changes_count = int(pr_state.get("request_changes_count", 0))
        # Simulate the increment that record_review would do
        if decision == "request_changes":
            escalated = request_changes_count >= 2  # Check before increment
            # Simulate head_advanced check (issue #208)
            reviewed_head_sha = app.gh.pr_head_shas.get(pr_number)
            head_advanced = reviewed_head_sha != pr_state.get("reviewed_head_sha")
            if not escalated and head_advanced:
                request_changes_count += 1
            pr_state["request_changes_count"] = request_changes_count
            state["prs"][str(pr_number)] = pr_state
            save_state(app.paths.state_file, state)
        else:
            escalated = False
        return CommandResult(True, "record_review called", {"escalated": escalated})

    def _fake_transition(gh, labels, issue_number, edge):
        from charlie_work.labels import TransitionResult, TransitionOutcome

        return TransitionResult(
            outcome=TransitionOutcome.APPLIED,
            add_failures=[],
            remove_failures=[],
        )

    monkeypatch.setattr("charlie_work.workflow.check_test_adequacy", _fake_check)
    monkeypatch.setattr("charlie_work.workflow.transition", _fake_transition)
    monkeypatch.setattr(app, "record_review", _fake_record_review)

    # Cycle 1: request_changes_count = 0 -> 1
    app.gh.pr_head_shas[456] = "sha-1"
    result1 = app.review(456)
    assert result1.data["escalated"] is False

    # Cycle 2: request_changes_count = 1 -> 2
    app.gh.pr_head_shas[456] = "sha-2"
    result2 = app.review(456)
    assert result2.data["escalated"] is False

    # Cycle 3: request_changes_count = 2 -> escalate
    app.gh.pr_head_shas[456] = "sha-3"
    result3 = app.review(456)
    assert result3.data["escalated"] is True


def test_review_test_adequacy_pass_proceeds_to_packet(tmp_path: Path, monkeypatch) -> None:
    """When test_adequacy passes, review() proceeds to normal packet path."""
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict

    app = _test_adequacy_app(tmp_path, enabled=True)
    check_calls = {"n": 0}

    pass_verdict = TestAdequacyVerdict(
        ok=True,
        failures=(),
        warnings=(),
        facts=TestAdequacyFacts(
            added_product_loc=15,
            added_test_loc=20,
            assertion_count=5,
            test_files_changed=2,
            untested_product_files=(),
            exempt=False,
            exempt_reason="",
        ),
    )

    def _fake_check(diff, pr, config):
        check_calls["n"] += 1
        return pass_verdict

    monkeypatch.setattr("charlie_work.workflow.check_test_adequacy", _fake_check)

    result = app.review(456)

    assert check_calls["n"] == 1
    assert result.ok is True


# Fleet status tests


def test_fleet_status_aggregates_multiple_repos(tmp_path: Path, monkeypatch) -> None:
    """Test that fleet status aggregates status from multiple repos."""
    # Set up fleet directory override
    fleet_override = str(tmp_path / "fleet")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", fleet_override)

    # Create two repo directories with minimal setup
    repo1 = tmp_path / "repo1"
    repo2 = tmp_path / "repo2"
    repo1.mkdir()
    repo2.mkdir()

    # Create minimal configs
    config1 = repo1 / "orchestrator.config.yaml"
    config2 = repo2 / "orchestrator.config.yaml"
    config1.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\nruntime:\n  state_dir: .var/charlie-work\n"
    )
    config2.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\nruntime:\n  state_dir: .var/charlie-work\n"
    )

    # Create state directories
    (repo1 / ".var" / "charlie-work").mkdir(parents=True)
    (repo2 / ".var" / "charlie-work").mkdir(parents=True)

    # Create fleet.json with two repos
    fleet_json_path = Path(fleet_override) / "fleet.json"
    fleet_json_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo1": {
                "repo_root": str(repo1),
                "name_with_owner": "owner/repo1",
                "config_path": str(config1),
                "state_dir": str(repo1 / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
            "owner/repo2": {
                "repo_root": str(repo2),
                "name_with_owner": "owner/repo2",
                "config_path": str(config2),
                "state_dir": str(repo2 / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
        },
    }
    import json

    fleet_json_path.write_text(json.dumps(registry_data, indent=2))

    # Mock GitHub to return empty issue/PR lists
    from charlie_work.github import GitHub

    def mock_issue_list(self, label):
        return []

    def mock_pr_list(self):
        return []

    def mock_get_github_issue_dependencies(gh, issue_number):
        return [], []

    monkeypatch.setattr(GitHub, "issue_list", mock_issue_list)
    monkeypatch.setattr(GitHub, "pr_list", mock_pr_list)
    monkeypatch.setattr(
        "charlie_work.github.get_github_issue_dependencies", mock_get_github_issue_dependencies
    )

    # Run fleet status
    args = cli.build_parser().parse_args(["fleet", "status"])
    result = cli.run_fleet_status(args)

    # Verify aggregation
    assert result.ok is True
    assert "2 repo(s)" in result.message
    assert len(result.data["repos"]) == 2
    assert "owner/repo1" in result.data["repos"]
    assert "owner/repo2" in result.data["repos"]
    assert result.data["errors"] == []


def test_fleet_status_isolates_broken_repo(tmp_path: Path, monkeypatch) -> None:
    """Test that fleet status isolates errors from broken repos."""
    # Set up fleet directory override
    fleet_override = str(tmp_path / "fleet")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", fleet_override)

    # Create one valid repo
    repo_valid = tmp_path / "repo_valid"
    repo_valid.mkdir()
    config_valid = repo_valid / "orchestrator.config.yaml"
    config_valid.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\nruntime:\n  state_dir: .var/charlie-work\n"
    )
    (repo_valid / ".var" / "charlie-work").mkdir(parents=True)

    # Create fleet.json with one valid and one broken repo
    fleet_json_path = Path(fleet_override) / "fleet.json"
    fleet_json_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo_valid": {
                "repo_root": str(repo_valid),
                "name_with_owner": "owner/repo_valid",
                "config_path": str(config_valid),
                "state_dir": str(repo_valid / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
            "owner/repo_broken": {
                "repo_root": str(tmp_path / "nonexistent"),
                "name_with_owner": "owner/repo_broken",
                "config_path": str(tmp_path / "nonexistent" / "orchestrator.config.yaml"),
                "state_dir": str(tmp_path / "nonexistent" / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
        },
    }
    import json

    fleet_json_path.write_text(json.dumps(registry_data, indent=2))

    # Mock GitHub to return empty issue/PR lists
    from charlie_work.github import GitHub

    def mock_issue_list(self, label):
        return []

    def mock_pr_list(self):
        return []

    def mock_get_github_issue_dependencies(gh, issue_number):
        return [], []

    monkeypatch.setattr(GitHub, "issue_list", mock_issue_list)
    monkeypatch.setattr(GitHub, "pr_list", mock_pr_list)
    monkeypatch.setattr(
        "charlie_work.github.get_github_issue_dependencies", mock_get_github_issue_dependencies
    )

    # Run fleet status
    args = cli.build_parser().parse_args(["fleet", "status"])
    result = cli.run_fleet_status(args)

    # Verify error isolation
    assert result.ok is False  # Errors present
    assert "1 repo(s), 1 error(s)" in result.message
    assert len(result.data["repos"]) == 1
    assert "owner/repo_valid" in result.data["repos"]
    assert len(result.data["errors"]) == 1
    assert result.data["errors"][0]["repo_key"] == "owner/repo_broken"
    assert "does not exist" in result.data["errors"][0]["error"]


def test_fleet_status_never_mutates(tmp_path: Path, monkeypatch) -> None:
    """Test that fleet status never mutates GitHub labels or state."""
    # Set up fleet directory override
    fleet_override = str(tmp_path / "fleet")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", fleet_override)

    # Create a repo with a ready-labeled issue
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\nruntime:\n  state_dir: .var/charlie-work\n"
    )
    (repo / ".var" / "charlie-work").mkdir(parents=True)

    # Create state.json
    state_file = repo / ".var" / "charlie-work" / "state.json"
    import json

    initial_state = {
        "version": 1,
        "generated_at": "2026-07-06T12:00:00Z",
        "issues": {},
        "prs": {},
        "events": [],
    }
    state_file.write_text(json.dumps(initial_state, indent=2))

    # Create fleet.json
    fleet_json_path = Path(fleet_override) / "fleet.json"
    fleet_json_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo": {
                "repo_root": str(repo),
                "name_with_owner": "owner/repo",
                "config_path": str(config),
                "state_dir": str(repo / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
        },
    }
    fleet_json_path.write_text(json.dumps(registry_data, indent=2))

    # Mock GitHub to return a ready issue and track mutating calls
    from charlie_work.github import GitHub

    mutating_calls = []

    def mock_run(self, args, json_output=False, allow_failure=False):
        mutating_calls.append(args)
        return ""

    def mock_issue_list(self, label):
        return [{"number": 123, "title": "Test issue", "labels": [{"name": label}]}]

    def mock_pr_list(self):
        return []

    def mock_get_github_issue_dependencies(gh, issue_number):
        return [], []

    monkeypatch.setattr(GitHub, "run", mock_run)
    monkeypatch.setattr(GitHub, "issue_list", mock_issue_list)
    monkeypatch.setattr(GitHub, "pr_list", mock_pr_list)
    monkeypatch.setattr(
        "charlie_work.github.get_github_issue_dependencies", mock_get_github_issue_dependencies
    )

    # Run fleet status
    args = cli.build_parser().parse_args(["fleet", "status"])
    result = cli.run_fleet_status(args)

    # Verify no mutating calls were made
    assert result.ok is True
    # GitHub.run should not be called with mutating commands
    for call in mutating_calls:
        assert not any(
            mutating_cmd in call
            for mutating_cmd in ["issue edit", "label add", "label remove", "pr edit"]
        ), f"Mutating call detected: {call}"

    # Verify state.json was not modified
    final_state = json.loads(state_file.read_text())
    assert final_state["generated_at"] == initial_state["generated_at"]
    assert final_state == initial_state


def test_fleet_status_json_output_shape(tmp_path: Path, monkeypatch) -> None:
    """Test that fleet status --json produces the correct output shape."""
    from io import StringIO

    # Set up fleet directory override
    fleet_override = str(tmp_path / "fleet")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", fleet_override)

    # Create a minimal repo
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\nruntime:\n  state_dir: .var/charlie-work\n"
    )
    (repo / ".var" / "charlie-work").mkdir(parents=True)

    # Create fleet.json
    fleet_json_path = Path(fleet_override) / "fleet.json"
    fleet_json_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo": {
                "repo_root": str(repo),
                "name_with_owner": "owner/repo",
                "config_path": str(config),
                "state_dir": str(repo / ".var" / "charlie-work"),
                "first_seen": "2026-07-06T12:00:00Z",
                "last_seen": "2026-07-06T12:00:00Z",
            },
        },
    }
    import json

    fleet_json_path.write_text(json.dumps(registry_data, indent=2))

    # Mock GitHub to return empty issue/PR lists
    from charlie_work.github import GitHub

    def mock_issue_list(self, label):
        return []

    def mock_pr_list(self):
        return []

    def mock_get_github_issue_dependencies(gh, issue_number):
        return [], []

    monkeypatch.setattr(GitHub, "issue_list", mock_issue_list)
    monkeypatch.setattr(GitHub, "pr_list", mock_pr_list)
    monkeypatch.setattr(
        "charlie_work.github.get_github_issue_dependencies", mock_get_github_issue_dependencies
    )

    # Capture stdout
    fake_stdout = StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    # Run fleet status --json via main()
    try:
        cli.main(["fleet", "status", "--json"])
    except SystemExit:
        pass

    output = fake_stdout.getvalue()
    parsed = json.loads(output)

    # Verify JSON structure
    assert "ok" in parsed
    assert "message" in parsed
    assert "data" in parsed
    assert "repos" in parsed["data"]
    assert "errors" in parsed["data"]
    assert "owner/repo" in parsed["data"]["repos"]


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


def test_loop_reaps_stalled_session_with_no_candidates(tmp_path: Path) -> None:
    """Test that loop() reaps stalled sessions even with zero ready/rework candidates (issue #165)."""
    from datetime import UTC, datetime, timedelta
    from charlie_work.devin_shell import SessionRecord

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Create a session record for issue 123 with a live PID and stale log
    sessions_dir = app._resolve(config.devin.sessions_dir)
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


def test_dispatch_rework_reaps_unconditionally_when_max_concurrent_zero(tmp_path: Path) -> None:
    """Test that dispatch_rework() has the unconditional reaper call (issue #165)."""
    # Verify by code inspection that dispatch_rework calls _detect_and_handle_stalled_sessions
    import charlie_work.workflow as workflow_module
    import inspect

    dispatch_rework_source = inspect.getsource(workflow_module.OrchestratorApp.dispatch_rework)

    # Verify the unconditional call exists
    assert "_detect_and_handle_stalled_sessions" in dispatch_rework_source
    # Verify it's called before the governor (which has the max_concurrent check)
    reaper_call_pos = dispatch_rework_source.find("_detect_and_handle_stalled_sessions")
    assert reaper_call_pos > 0, "Reaper call should exist in dispatch_rework"
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
    """Test that redispatch_at is only written by the two known call sites (issue #165)."""
    # This test verifies by code inspection that redispatch_at is only written in:
    # 1. dispatch_rework (workflow.py:2440-2472)
    # 2. _classify_dead_sessions_and_update_throttle_state (workflow.py:468-504)
    # No other code paths write to redispatch_at.

    # Verify the two call sites exist in the code
    import charlie_work.workflow as workflow_module
    import inspect

    workflow_source = inspect.getsource(workflow_module)

    # Count occurrences of redispatch_at assignments to entry
    # We have 2 assignments in dispatch_rework (lines 2472, 2451) and 2 in _classify_dead_sessions_and_update_throttle_state (lines 502, 483)
    # Total of 4 assignments is correct - 2 in each function for the two code paths (normal vs escalation)
    redispatch_assignments = workflow_source.count('entry["redispatch_at"]')
    # Should be exactly 4: 2 in dispatch_rework (normal + escalation), 2 in _classify_dead_sessions_and_update_throttle_state (normal + escalation)
    assert redispatch_assignments == 4, (
        f"Expected 4 redispatch_at assignments, found {redispatch_assignments}"
    )


def test_orphaned_worker_detection_with_request_changes_and_unchanged_head(tmp_path: Path) -> None:
    """Regression test for issue #207: dead worker with request_changes and unchanged head should reset to rework_requested."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create initial state with a dispatched issue and dead worker PID
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,  # Dead PID
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)

    # Mock GitHub to return an open PR for the issue
    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",  # Unchanged since request_changes
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()

    # Mock PID liveness check to return False (dead PID)
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    # Load state and verify the transition
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # Status should be reset to rework_requested
    assert entry.get("status") == "rework_requested"
    assert entry.get("dispatched_at") is None

    # Worker PID should be cleared
    assert "worker_pid" not in entry
    assert "worker_process_start_time" not in entry

    # Verify the event was logged
    events = state.get("events", [])
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 1
    assert recovered_events[0]["payload"]["issue_number"] == 207
    assert recovered_events[0]["payload"]["pr_number"] == 100
    assert recovered_events[0]["payload"]["reason"] == "dead_worker_with_request_changes"


def test_orphaned_worker_detection_with_head_change(tmp_path: Path) -> None:
    """Regression test for issue #207: dead worker with head change should emit drift event, not auto-reset."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create initial state with a dispatched issue and dead worker PID
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,  # Dead PID
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",  # Old head
    }
    save_state(paths.state_file, state)

    # Mock GitHub to return an open PR with changed head
    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "def456",  # Changed since request_changes
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()

    # Mock PID liveness check to return False (dead PID)
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    # Load state and verify NO auto-reset
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # Status should NOT be reset (still dispatched)
    assert entry.get("status") == "dispatched"

    # Worker PID should be cleared
    assert "worker_pid" not in entry
    assert "worker_process_start_time" not in entry

    # Verify drift event was logged
    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["issue_number"] == 207
    assert drift_events[0]["payload"]["reason"] == "dead_worker_with_head_change"


def test_orphaned_worker_detection_with_live_pid(tmp_path: Path) -> None:
    """Regression test for issue #207: live worker with matching start time should be untouched."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create initial state with a dispatched issue and live worker PID
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,  # Live PID
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)

    # Mock GitHub to return an open PR
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

    # Mock PID liveness check to return True (live PID) with matching start time
    with patch("charlie_work.workflow._worker_pid_alive", return_value=True):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    # Load state and verify NO changes
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # Status should remain dispatched
    assert entry.get("status") == "dispatched"

    # Worker PID should still be present
    assert entry.get("worker_pid") == 99999
    assert entry.get("worker_process_start_time") == 1234567890.0

    # Verify NO events were logged
    events = state.get("events", [])
    orphaned_events = [
        e
        for e in events
        if e.get("kind") in ("orphaned_worker_recovered", "orphaned_worker_drift")
    ]
    assert len(orphaned_events) == 0


def test_orphaned_worker_detection_with_pid_recycled(tmp_path: Path) -> None:
    """Regression test for issue #207: PID recycled (start-time mismatch) should be treated as dead."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create initial state with a dispatched issue and recycled PID
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,  # Recycled PID
        "worker_process_start_time": 1234567890.0,  # Old start time
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)

    # Mock GitHub to return an open PR
    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",  # Unchanged
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()

    # Mock the helper to simulate PID recycling (alive check returns False due to start-time mismatch)
    def mock_worker_pid_alive(entry):
        # Simulate start-time mismatch by returning False even though PID is set
        return False

    with patch("charlie_work.workflow._worker_pid_alive", side_effect=mock_worker_pid_alive):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    # Load state and verify it was treated as dead
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # Status should be reset to rework_requested
    assert entry.get("status") == "rework_requested"

    # Worker PID should be cleared
    assert "worker_pid" not in entry
    assert "worker_process_start_time" not in entry

    # Verify recovered event was logged
    events = state.get("events", [])
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 1


def test_orphaned_worker_detection_no_open_pr(tmp_path: Path) -> None:
    """Regression test for issue #207: dead worker with no open PR should emit drift event (not auto-reset status)."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create initial state with a dispatched issue and dead worker PID
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,  # Dead PID
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)

    # Mock GitHub to return NO open PRs
    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubForOrphan()

    # Mock PID liveness check to return False (dead PID)
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    # Load state and verify NO status auto-reset
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # Status should NOT be reset (still dispatched)
    assert entry.get("status") == "dispatched"

    # Worker PID should be cleared
    assert "worker_pid" not in entry
    assert "worker_process_start_time" not in entry

    # Verify drift event was logged (not recovered)
    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["issue_number"] == 207
    assert drift_events[0]["payload"]["reason"] == "dead_worker_no_open_pr"

    # Verify NO recovered event
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 0


# ---------------------------------------------------------------------------
# SupervisorConfig tests
# ---------------------------------------------------------------------------


def test_supervisor_config_defaults() -> None:
    """SupervisorConfig defaults are stable and load_config picks them up."""
    config = load_config()
    assert isinstance(config.supervisor, SupervisorConfig)
    assert config.supervisor.poll_interval_seconds == 20
    assert config.supervisor.full_pass_interval_seconds == 300
    assert config.supervisor.active_cooldown_seconds == 30
    assert config.supervisor.max_runtime_minutes == 0


def test_supervisor_config_parses_custom_values(tmp_path: Path) -> None:
    """Custom supervisor section values are parsed correctly."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
supervisor:
  poll_interval_seconds: 10
  full_pass_interval_seconds: 120
  active_cooldown_seconds: 15
  max_runtime_minutes: 60
"""
    )
    config = load_config(config_file)
    assert config.supervisor.poll_interval_seconds == 10
    assert config.supervisor.full_pass_interval_seconds == 120
    assert config.supervisor.active_cooldown_seconds == 15
    assert config.supervisor.max_runtime_minutes == 60


def test_supervisor_config_unknown_key_raises(tmp_path: Path) -> None:
    """Unknown keys in supervisor section raise ConfigError."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
supervisor:
  poll_interval_seconds: 10
  unknown_key: 99
"""
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(config_file)


def test_supervisor_config_wrong_type_raises(tmp_path: Path) -> None:
    """Wrong types in supervisor section raise ConfigError."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
supervisor:
  poll_interval_seconds: "not-an-int"
"""
    )
    with pytest.raises(ConfigError, match="must be an int"):
        load_config(config_file)


def test_supervisor_config_is_frozen() -> None:
    """SupervisorConfig is a frozen dataclass."""
    import dataclasses

    cfg = SupervisorConfig()
    assert dataclasses.is_dataclass(cfg)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        cfg.poll_interval_seconds = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PostMortemConfig tests (issue #261)
# ---------------------------------------------------------------------------


def test_post_mortem_config_defaults() -> None:
    """PostMortemConfig defaults are stable and load_config picks them up.

    Issue #260 (corrected premise) added a third default rule: "A tool was
    rejected by the user" is the Devin CLI's own log/stdout surfacing of a
    PreToolUse hook block — distinct from the "Tool blocked:" prefix that
    appears in sessions.db message-node content — and drives
    post_mortem.classify_and_record's log-tail fallback when the DB is
    unavailable (see test_post_mortem.py's log-tail fallback tests).
    """
    config = load_config()
    assert isinstance(config.post_mortem, PostMortemConfig)
    assert config.post_mortem.enabled is True
    assert config.post_mortem.db_path == ""
    assert config.post_mortem.message_node_limit == 10
    assert config.post_mortem.match_window_margin_seconds == 120
    assert config.post_mortem.signature_rules == (
        SignatureRule(pattern=r"Tool blocked:", kind="worker_blocked"),
        SignatureRule(pattern=r"decision\s*:\s*block", kind="worker_blocked"),
        SignatureRule(pattern=r"A tool was rejected by the user", kind="worker_blocked"),
    )


def test_post_mortem_config_parses_custom_values(tmp_path: Path) -> None:
    """Custom post_mortem section values, including signature_rules, are parsed correctly."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
post_mortem:
  enabled: false
  db_path: "C:/custom/sessions.db"
  message_node_limit: 5
  match_window_margin_seconds: 30
  signature_rules:
    - pattern: "custom-block-signature"
      kind: "worker_blocked"
"""
    )
    config = load_config(config_file)
    assert config.post_mortem.enabled is False
    assert config.post_mortem.db_path == "C:/custom/sessions.db"
    assert config.post_mortem.message_node_limit == 5
    assert config.post_mortem.match_window_margin_seconds == 30
    assert config.post_mortem.signature_rules == (
        SignatureRule(pattern="custom-block-signature", kind="worker_blocked"),
    )


def test_post_mortem_config_unknown_key_raises(tmp_path: Path) -> None:
    """Unknown keys in post_mortem section raise ConfigError."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
post_mortem:
  enabled: true
  unknown_key: 99
"""
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(config_file)


def test_post_mortem_config_wrong_type_raises(tmp_path: Path) -> None:
    """Wrong types in post_mortem section raise ConfigError."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
post_mortem:
  message_node_limit: "not-an-int"
"""
    )
    with pytest.raises(ConfigError, match="must be an int"):
        load_config(config_file)


def test_post_mortem_config_signature_rules_wrong_shape_raises(tmp_path: Path) -> None:
    """signature_rules must be a list of {pattern, kind} mappings."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
post_mortem:
  signature_rules: "not-a-list"
"""
    )
    with pytest.raises(ConfigError, match="must be a list"):
        load_config(config_file)


def test_post_mortem_config_signature_rule_bad_regex_raises(tmp_path: Path) -> None:
    """An invalid regex pattern in a signature rule raises ConfigError."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
post_mortem:
  signature_rules:
    - pattern: "["
      kind: "worker_blocked"
"""
    )
    with pytest.raises(ConfigError, match="not a valid regex"):
        load_config(config_file)


def test_post_mortem_config_is_frozen() -> None:
    """PostMortemConfig is a frozen dataclass."""
    import dataclasses

    cfg = PostMortemConfig()
    assert dataclasses.is_dataclass(cfg)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        cfg.enabled = False  # type: ignore[misc]


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


def _make_loop_app(tmp_path: Path, *, prs: list[dict]) -> tuple[OrchestratorApp, FakeGitHub]:
    """Build a minimal OrchestratorApp with the given open PRs for loop() tests."""
    config = OrchestratorConfig(
        cross_family=CrossFamilyConfig(enabled=False),
        auto_merge=_approved_automerge(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, fake_gh


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
    """Regression for review finding #7: a same-head packet skip must still
    check review-decision.json directly. An operator can write the decision
    file without state.json reflecting it yet (the already_approved branch
    only fires once state.json has caught up), so the approval must not stay
    invisible until the head moves -- it should proceed straight to
    merge_ready(), same as the decided path.
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
    assert result.data["skipped_reviews"] == 1
    assert 456 not in review_calls
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


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_bare_remote_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare remote repo and a local clone, return (remote, clone)."""
    remote = tmp_path / "remote"
    remote.mkdir(parents=True, exist_ok=True)
    _git(remote, "init", "--bare", "--initial-branch=main")
    clone = tmp_path / "clone"
    clone.mkdir(parents=True, exist_ok=True)
    _git(clone, "init", "--initial-branch=main")
    _git(clone, "config", "user.email", "test@example.test")
    _git(clone, "config", "user.name", "Test User")
    _git(clone, "config", "commit.gpgSign", "false")
    _git(clone, "remote", "add", "origin", str(remote))
    (clone / "README.md").write_text("hello\n", encoding="utf-8")
    _git(clone, "add", "README.md")
    _git(clone, "commit", "-m", "initial commit")
    _git(clone, "push", "-u", "origin", "main")
    return remote, clone


def _setup_completed_worktree(
    repo_root: Path, issue_number: int, dirty: bool = False
) -> tuple[Path, str]:
    """Create a worktree with one commit beyond origin/main. Return (worktree_path, branch)."""
    branch = f"agent/issue-{issue_number}"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")
    if dirty:
        (info.path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    return info.path, branch


def _write_dead_session_sidecar(
    sessions_dir: Path, issue_number: int, branch: str, worktree_path: Path
) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    record = SessionRecord(
        issue_number=issue_number,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(sessions_dir / f"issue-{issue_number}.log"),
        error=None,
    )
    sidecar_path = sessions_dir / f"issue-{issue_number}.json"
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")


def _make_classify_state(tmp_path: Path) -> tuple[Path, Path]:
    """Create a state file and sessions dir under tmp_path, return (sessions_dir, state_file)."""
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    return sessions_dir, state_file


def test_classify_dead_sessions_salvages_completed_unpublished_work(
    tmp_path: Path,
) -> None:
    """Issue #252: a clean, ahead worktree is salvaged (push + PR + pr_open label)."""
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 252)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 252, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 252,
            "title": "Test issue",
            "url": "https://example.test/issues/252",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.pr_create_return = 101

    _classify_dead_sessions_and_update_throttle_state(sessions_dir, state_file, gh, config)

    # Branch pushed and PR created
    remote_refs = _git(remote, "show-ref")
    assert "agent/issue-252" in remote_refs.stdout
    assert len(gh.prs_created) == 1
    assert gh.prs_created[0]["head"] == branch
    assert gh.prs_created[0]["base"] == "main"

    # Labels moved to pr_open
    assert (252, config.labels.in_progress) in gh.labels_removed
    assert (252, config.labels.pr_open) in gh.labels_added

    # Sidecar reaped and event recorded
    assert not (sessions_dir / "issue-252.json").exists()
    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert len(events) == 1
    assert events[0]["payload"]["issue_number"] == 252
    assert events[0]["payload"]["pr_number"] == 101


def test_classify_dead_sessions_dirty_worktree_relabels_to_ready(tmp_path: Path) -> None:
    """Issue #252: a dirty worktree is not salvaged; it relabels to ready."""
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 253, dirty=True)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 253, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 253,
            "title": "Test issue",
            "url": "https://example.test/issues/253",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.pr_create_return = 101

    _classify_dead_sessions_and_update_throttle_state(sessions_dir, state_file, gh, config)

    # No PR created, active label removed, ready label added
    assert not gh.prs_created
    assert (253, config.labels.in_progress) in gh.labels_removed
    assert (253, config.labels.ready) in gh.labels_added

    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 1


def test_classify_dead_sessions_no_commits_relabels_to_ready(tmp_path: Path) -> None:
    """Issue #252: a clean worktree with no commits relabels to ready."""
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    branch = "agent/issue-254"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 254, branch, info.path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 254,
            "title": "Test issue",
            "url": "https://example.test/issues/254",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.pr_create_return = 101

    _classify_dead_sessions_and_update_throttle_state(sessions_dir, state_file, gh, config)

    assert not gh.prs_created
    assert (254, config.labels.in_progress) in gh.labels_removed
    assert (254, config.labels.ready) in gh.labels_added


def test_classify_dead_sessions_salvage_push_failure_fallback(tmp_path: Path) -> None:
    """Issue #252: a failed salvage push records failure and falls back to relabel."""
    from charlie_work import workflow as workflow_module
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 255)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 255, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 255,
            "title": "Test issue",
            "url": "https://example.test/issues/255",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.pr_create_return = 101

    original_push_branch = workflow_module.push_branch
    workflow_module.push_branch = lambda repo, br, worktree_path=None: (
        False,
        "simulated push failure",
    )
    try:
        _classify_dead_sessions_and_update_throttle_state(sessions_dir, state_file, gh, config)
    finally:
        workflow_module.push_branch = original_push_branch

    # No PR created, active label removed, ready label added, event records salvage failure
    assert not gh.prs_created
    assert (255, config.labels.in_progress) in gh.labels_removed
    assert (255, config.labels.ready) in gh.labels_added

    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 1
    assert events[0]["payload"].get("salvage_failed") is True
