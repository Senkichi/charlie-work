from __future__ import annotations

import contextlib
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

from _sessions_db_fixtures import make_sessions_db
from charlie_work import cli
from charlie_work import github as github_module
from charlie_work.checks import _run_id_from_link, summarize_checks
from charlie_work.github import _job_id_from_link, is_infrastructure_failure
from charlie_work.config import (
    AutoMergeConfig,
    ClaudeCodeConfig,
    ConfigError,
    CrossFamilyConfig,
    DevinConfig,
    DispatchConfig,
    FleetConfig,
    LabelConfig,
    OrchestratorConfig,
    PostMortemConfig,
    ReviewConfig,
    ReviewDispatchConfig,
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
from charlie_work.github import issue_numbers_mentioned_by_pr, label_names, linked_issue_number
from charlie_work.paths import runtime_paths
from charlie_work.prompts import render_prompt
from charlie_work.state import (
    append_event,
    is_throttled,
    load_state,
    save_state,
    set_throttled_until,
    state_lock,
)
import charlie_work.state as state_module
from charlie_work.workflow import (
    CommandResult,
    ConcurrencyGovernorResult,
    OrchestratorApp,
    slugify,
)
from charlie_work.worktree import create_worktree
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.devin_shell import SessionRecord

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(autouse=True)
def _stub_real_activity_probe_for_stalled_tests(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Issue #307: stall-detection tests need a stale real-activity probe.

    Without this fixture, ``real_activity_probe_for`` reaches the host's real
    ``sessions.db`` and per-PID logs and returns an all-errored probe. Issue #307
    makes an all-errored probe fail-open (defer), so stall tests that are not
    about real-activity corroboration would otherwise return HEALTHY. Tests that
    intentionally exercise a live (unstubbed) probe opt out via the
    ``real_activity_probe_live`` marker instead of a rename-fragile name match.
    """
    if request.node.get_closest_marker("real_activity_probe_live") is not None:
        return

    from datetime import datetime, timedelta, UTC
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe

    def _stale_probe(*_args: object, **_kwargs: object) -> RealActivityProbe:
        now = datetime.now(UTC)
        timestamp = now - timedelta(minutes=30)
        return RealActivityProbe(
            sources=(
                ActivitySource(
                    name="devin_per_pid_log",
                    timestamp=timestamp,
                    staleness_seconds=(now - timestamp).total_seconds(),
                    error=None,
                ),
            )
        )

    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_probe)


def test_default_config_enables_auto_merge() -> None:
    config = load_config()

    assert config.auto_merge.enabled is True
    # A shared package cannot know a consumer's CI check names; unconfigured
    # means empty, and `doctor` flags it.
    assert config.auto_merge.required_checks == ()
    assert config.labels.ready == "automated-ready"


def test_default_config_failed_attempt_alarm() -> None:
    """Issue #254: default merge attempt alarm threshold is 3."""
    config = load_config()
    assert config.auto_merge.failed_attempt_alarm == 3


def test_default_config_update_open_prs_is_next() -> None:
    """Default update_open_prs is merge-train mode."""
    config = load_config()
    assert config.auto_merge.update_open_prs == "next"


def test_default_config_update_branch_strategy_is_front_of_train() -> None:
    """Issue #404: default update_branch_strategy is front-of-train."""
    config = load_config()
    assert config.auto_merge.update_branch_strategy == "front_of_train"


def test_default_config_require_current_base() -> None:
    """Default require_current_base is True."""
    config = load_config()
    assert config.auto_merge.require_current_base is True


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


def test_load_config_rejects_non_int_failed_attempt_alarm(tmp_path: Path) -> None:
    """Issue #254: auto_merge.failed_attempt_alarm must be an int."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  failed_attempt_alarm: "three"
"""
    )
    with pytest.raises(ConfigError, match="failed_attempt_alarm.*must be an int"):
        load_config(config_file)


def test_load_config_update_open_prs_boolean_aliases(tmp_path: Path) -> None:
    """update_open_prs boolean aliases are normalized for backward compatibility."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  update_open_prs: true
"""
    )
    config = load_config(config_file)
    assert config.auto_merge.update_open_prs == "all"

    config_file.write_text(
        """
auto_merge:
  update_open_prs: false
  require_current_base: false
"""
    )
    config = load_config(config_file)
    assert config.auto_merge.update_open_prs == "off"


def test_load_config_update_open_prs_string_values(tmp_path: Path) -> None:
    """update_open_prs accepts all/next/off string values."""
    config_file = tmp_path / "orchestrator.config.yaml"
    for value in ("all", "next", "off", "ALL", "Next", "OFF"):
        config_file.write_text(
            f"""
auto_merge:
  update_open_prs: {value}
  require_current_base: false
"""
        )
        config = load_config(config_file)
        assert config.auto_merge.update_open_prs == value.lower()


def test_load_config_update_open_prs_rejects_invalid_value(tmp_path: Path) -> None:
    """update_open_prs rejects unknown string values."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  update_open_prs: sometimes
"""
    )
    with pytest.raises(ConfigError, match="update_open_prs.*'all', 'next', 'off'"):
        load_config(config_file)


def test_load_config_update_branch_strategy_values(tmp_path: Path) -> None:
    """Issue #404: update_branch_strategy accepts front_of_train/broadcast/off."""
    from charlie_work.config import ConfigError, load_config

    config_file = tmp_path / "orchestrator.config.yaml"
    for value in ("front_of_train", "broadcast", "off"):
        config_file.write_text(
            f"""
auto_merge:
  update_branch_strategy: {value}
  require_current_base: false
"""
        )
        config = load_config(config_file)
        assert config.auto_merge.update_branch_strategy == value

    config_file.write_text(
        """
auto_merge:
  update_branch_strategy: sometimes
"""
    )
    with pytest.raises(
        ConfigError, match="update_branch_strategy.*'front_of_train', 'broadcast', or 'off'"
    ):
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


def test_load_config_rejects_non_string_mergequeue_label(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  mergequeue_label: 123
"""
    )
    with pytest.raises(ConfigError, match="mergequeue_label.*must be a string"):
        load_config(config_file)


def test_load_config_rejects_empty_mergequeue_label(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  mergequeue_label: "   "
"""
    )
    with pytest.raises(ConfigError, match="mergequeue_label.*must not be empty"):
        load_config(config_file)


def test_load_config_accepts_valid_mergequeue_label(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  mergequeue_label: mergequeue
"""
    )
    config = load_config(config_file)
    assert config.auto_merge.mergequeue_label == "mergequeue"


def test_load_config_strips_mergequeue_label_whitespace(tmp_path: Path) -> None:
    """Adversarial review finding #3: mergequeue_label.strip() is used only to
    validate truthiness, but the unstripped value must not thread verbatim
    into `gh pr edit --add-label` — surrounding whitespace is not a valid (or
    intended) part of a GitHub label name."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  mergequeue_label: "  mergequeue  "
"""
    )
    config = load_config(config_file)
    assert config.auto_merge.mergequeue_label == "mergequeue"


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


def test_summarize_checks_none_returns_unavailable_required_checks() -> None:
    """Command-level gh failure (checks=None) marks every required check unavailable."""
    summary = summarize_checks(None, ("Tests",))

    assert summary.ready is False
    assert summary.unavailable == ("Tests",)
    assert summary.passed == ()
    assert summary.pending == ()
    assert summary.failed == ()
    assert summary.missing == ()


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


def test_github_merged_pr_list_uses_scoped_field_set(monkeypatch, tmp_path: Path) -> None:
    """merged_pr_list()'s sole consumer (workflow._merged_pr_referenced_issue_numbers,
    via linked_issue_number()/issue_numbers_mentioned_by_pr()) only reads
    state/headRefName/title/body/isCrossRepository. It must not request the
    broader PR_LIST_FIELDS set — in particular not `statusCheckRollup`, which
    forces gh's GraphQL query to walk each PR's check-run connection and is
    the root cause of intermittent gateway 502s at ~500-merged-PR scale.
    """
    captured_args: list[list[str]] = []
    rate_limit_payload = json.dumps({"resources": {"graphql": {"remaining": 10000, "reset": 0}}})

    def fake_run(cmd, *args, **kwargs):
        captured_args.append(cmd)
        if cmd[:3] == ["gh", "api", "rate_limit"]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=rate_limit_payload, stderr=""
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    gh.merged_pr_list()

    assert len(captured_args) == 2
    args = captured_args[1]
    assert args[:5] == ["gh", "pr", "list", "--state", "merged"]
    fields = args[args.index("--json") + 1].split(",")
    assert set(fields) == set(github_module.MERGED_PR_LIST_FIELDS.split(","))
    for unused_field in (
        "statusCheckRollup",
        "reviewDecision",
        "labels",
        "author",
        "updatedAt",
        "url",
        "baseRefName",
        "mergeStateStatus",
        "headRefOid",
        "isDraft",
    ):
        assert unused_field not in fields


def test_github_merged_pr_list_retries_on_transient_gateway_error(
    monkeypatch, tmp_path: Path
) -> None:
    """A transient 502/503/504 from the GraphQL-backed listing endpoint retries
    in-pass (bounded) instead of immediately failing the whole fleet pass for
    that repo (issue #361). Succeeds on the 2nd attempt here.
    """
    call_count = 0
    sleeps: list[float] = []
    rate_limit_payload = json.dumps({"resources": {"graphql": {"remaining": 10000, "reset": 0}}})

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if cmd[:3] == ["gh", "api", "rate_limit"]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=rate_limit_payload, stderr=""
            )
        if call_count == 2:
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
    assert call_count == 3
    assert len(sleeps) == 1


def test_github_merged_pr_list_gives_up_after_max_retries(monkeypatch, tmp_path: Path) -> None:
    """Persistent 502s must eventually raise GitHubError — never hang or retry
    forever — so the per-repo fleet-pass boundary can still catch it and move
    on to the next repo.
    """
    call_count = 0
    rate_limit_payload = json.dumps({"resources": {"graphql": {"remaining": 10000, "reset": 0}}})

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if cmd[:3] == ["gh", "api", "rate_limit"]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=rate_limit_payload, stderr=""
            )
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

    assert call_count == 4


def test_github_merged_pr_list_does_not_retry_non_transient_error(
    monkeypatch, tmp_path: Path
) -> None:
    """A non-gateway error (e.g. bad credentials) must fail immediately rather
    than be swallowed into the transient-gateway retry loop.
    """
    call_count = 0
    rate_limit_payload = json.dumps({"resources": {"graphql": {"remaining": 10000, "reset": 0}}})

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if cmd[:3] == ["gh", "api", "rate_limit"]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=rate_limit_payload, stderr=""
            )
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="HTTP 401: Bad credentials"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    with pytest.raises(github_module.GitHubError):
        gh.merged_pr_list()

    assert call_count == 2


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
                "baseRefName": "main",
                "headRefOid": "sha-abc123",
                "mergeStateStatus": "CLEAN",
                "body": "Closes #123\n\nTests: regression coverage added.",
                "labels": [],
                "isCrossRepository": False,
                "state": "OPEN",
            }
        ]
        self.labels_added: list[tuple[int, str]] = []
        self.labels_removed: list[tuple[int, str]] = []
        self.labels_created: list[tuple[str, str, str]] = []
        self.pr_labels_added: list[tuple[int, str]] = []
        self.add_pr_label_ok = True
        self.prs_created: list[dict[str, Any]] = []
        self.pr_create_return: int | None = None
        self.merged: list[tuple[int, str]] = []
        self.merged_admin_flags: list[bool] = []
        self.merged_merge_flags: list[tuple[str, ...]] = []
        self.deleted_branches: list[str] = []
        self.delete_branch_ok = True
        self.update_branch_ok = True
        self.pr_update_branch_calls: list[int] = []
        self.pr_head_shas: dict[int, str] = {}
        self.diffs: dict[int, str] = {}
        self.closed_issues: list[int] = []
        self.commits: dict[str, dict[str, Any]] = {}
        # Default base head and per-(base,head) compare overrides for testing
        # the merge-base freshness gate.
        self.base_head_sha = "base-sha"
        self.compare_overrides: dict[tuple[str, str], dict[str, Any] | None] = {}
        self._record_pr_heads(self.prs)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "base_head_sha" and hasattr(self, "commits"):
            if value not in self.commits:
                self.commits[value] = {"parents": []}
        elif name == "prs" and hasattr(self, "commits") and hasattr(self, "base_head_sha"):
            self._record_pr_heads(value)

    def _record_pr_heads(self, prs: list[dict[str, Any]]) -> None:
        """Index PR head SHAs as commits rooted at the current base tip."""
        base = self.base_head_sha
        for pr in prs:
            head = pr.get("headRefOid")
            if head and head not in self.commits:
                self.commits[head] = {"parents": [{"sha": base}]}

    def check_graphql_rate_limit(self, threshold: int) -> tuple[bool, int, int | None]:
        return (True, 10000, 0)

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
        return [pr for pr in self.prs if pr.get("state", "OPEN").upper() == "OPEN"]

    def merged_pr_list(self):
        return [pr for pr in self.prs if pr.get("state", "OPEN").upper() == "MERGED"]

    def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
        matched = []
        for pr in self.prs:
            if pr.get("state", "OPEN").upper() != "MERGED":
                continue
            bound = linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=branch_prefix,
            )
            if bound == issue_number:
                matched.append(pr)
        return matched

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

    def add_pr_label(self, number: int, label: str) -> bool:
        self.pr_labels_added.append((number, label))
        return self.add_pr_label_ok

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

        # Model the real effect of a merge: the base branch tip advances to a
        # merge commit whose parents are the previous base tip and the merged PR
        # head. This lets stale-base tests derive base movement organically from
        # recorded merges instead of hand-feeding compare_overrides.
        pr: dict[str, Any] | None = None
        for candidate in self.prs:
            if candidate.get("number") == number:
                pr = candidate
                break
        if pr is not None:
            base_ref = pr.get("baseRefName") or "main"
            head_sha = pr.get("headRefOid")
            old_base = self.base_head_sha
            merge_sha = f"{base_ref}-merged-{head_sha}"
            self.commits[merge_sha] = {
                "parents": [{"sha": old_base}, {"sha": head_sha}],
                "committer": {"login": "web-flow"},
                "commit": {"committer": {"name": "GitHub"}},
            }
            self.base_head_sha = merge_sha

        return "merged"

    def delete_branch(self, branch: str) -> bool:
        self.deleted_branches.append(branch)
        return self.delete_branch_ok

    def pr_update_branch(self, pr_number: int) -> bool:
        self.pr_update_branch_calls.append(pr_number)
        # Simulate a base update by moving the PR's head to a new SHA
        # This reproduces the churn that the fix prevents
        for pr in self.prs:
            if pr["number"] == pr_number:
                # Append a merge-SHA marker to simulate the head moving
                old_head = pr.get("headRefOid", "")
                new_head = f"{old_head}-updated"
                pr["headRefOid"] = new_head
                # A real update-branch makes the PR current with its base. Future
                # compare calls for the new head should see the current base tip.
                pr["mergeStateStatus"] = "CLEAN"
                base_ref = pr.get("baseRefName") or "main"
                base_head = self.base_head_sha
                self.compare_overrides[(base_ref, new_head)] = {
                    "base_commit": {"sha": base_head},
                    "merge_base_commit": {"sha": base_head},
                }
                # Record the fake commit metadata so the post-sync verification
                # helper sees a valid GitHub web-flow merge commit.
                self.commits[new_head] = {
                    "parents": [
                        {"sha": old_head},
                        {"sha": base_head},
                    ],
                    "committer": {"login": "web-flow"},
                    "commit": {"committer": {"name": "GitHub"}},
                }
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

    def commit(self, sha: str) -> dict[str, Any] | None:
        return self.commits.get(sha)

    def _ancestors(self, sha: str) -> set[str]:
        """Return all ancestors of ``sha`` (including ``sha`` itself)."""
        seen: set[str] = set()
        stack = [sha]
        while stack:
            current = stack.pop()
            if current in seen or not current:
                continue
            seen.add(current)
            commit = self.commits.get(current)
            if not isinstance(commit, dict):
                continue
            for parent in commit.get("parents", []):
                if isinstance(parent, dict):
                    parent_sha = parent.get("sha")
                else:
                    parent_sha = parent
                if parent_sha:
                    stack.append(parent_sha)
        return seen

    def _merge_base(self, base_sha: str, head_sha: str) -> str | None:
        """Return the best common ancestor of ``base_sha`` and ``head_sha``.

        The best common ancestor is a common ancestor that is not itself an
        ancestor of another common ancestor. For linear DAGs this is the usual
        merge-base; the simple filter works for the small graphs in these tests.
        """
        base_ancestors = self._ancestors(base_sha)
        head_ancestors = self._ancestors(head_sha)
        common = base_ancestors & head_ancestors
        if not common:
            return None
        best = [
            sha
            for sha in common
            if not any(sha in self._ancestors(other) and sha != other for other in common)
        ]
        if not best:
            best = list(common)

        # Deterministic tie-break: prefer the ancestor closest to the base tip.
        def _distance(source: str, target: str) -> int:
            if source == target:
                return 0
            visited: set[str] = {source}
            queue: list[tuple[str, int]] = [(source, 0)]
            while queue:
                current, dist = queue.pop(0)
                commit = self.commits.get(current)
                if not isinstance(commit, dict):
                    continue
                for parent in commit.get("parents", []):
                    if isinstance(parent, dict):
                        parent_sha = parent.get("sha")
                    else:
                        parent_sha = parent
                    if parent_sha == target:
                        return dist + 1
                    if parent_sha and parent_sha not in visited:
                        visited.add(parent_sha)
                        queue.append((parent_sha, dist + 1))
            return len(self.commits)

        best.sort(key=lambda sha: (_distance(base_sha, sha), _distance(head_sha, sha), sha))
        return best[0]

    def compare(self, base: str, head: str) -> dict[str, Any] | None:
        override = self.compare_overrides.get((base, head))
        if override is not None:
            return override
        base_head = self.base_head_sha

        # Find the matching PR so we can honor mergeStateStatus hints when the
        # graph is not enough or contradicts a BEHIND signal.
        matching_pr = None
        for pr in self.prs:
            if pr.get("headRefOid") == head:
                matching_pr = pr
                break
        if matching_pr is None:
            for pr_number, pr_head in self.pr_head_shas.items():
                if pr_head == head:
                    for pr in self.prs:
                        if pr.get("number") == pr_number:
                            matching_pr = pr
                            break
                    break

        # If we have a commit graph for both the current base tip and the head,
        # derive the merge base from recorded merges. This is the path that lets
        # merge tests prove ``merge advances main`` organically.
        if base_head in self.commits and head in self.commits:
            merge_base = self._merge_base(base_head, head)
            base_current = merge_base == base_head
            if base_current and str(matching_pr.get("mergeStateStatus") or "").upper() == "BEHIND":
                # A BEHIND mergeStateStatus is a stronger stale signal than the
                # current graph, so tests can still simulate a stale branch by
                # setting mergeStateStatus to BEHIND.
                return {
                    "base_commit": {"sha": base_head},
                    "merge_base_commit": {"sha": f"{base_head}-stale"},
                }
            return {
                "base_commit": {"sha": base_head},
                "merge_base_commit": {"sha": merge_base if merge_base else ""},
            }

        # If no graph is available, fall back to the PR's mergeStateStatus when
        # it is known. Tests can still use compare_overrides to model exceptional
        # cases (e.g. CLEAN-but-stale where mergeStateStatus lags).
        if (
            matching_pr is not None
            and str(matching_pr.get("mergeStateStatus") or "").upper() == "BEHIND"
        ):
            return {
                "base_commit": {"sha": base_head},
                "merge_base_commit": {"sha": f"{base_head}-stale"},
            }
        # Default: the PR's merge-base is the current base tip.
        return {
            "base_commit": {"sha": base_head},
            "merge_base_commit": {"sha": base_head},
        }

    def label_create(self, label: str, color: str, description: str) -> None:
        self.labels_created.append((label, color, description))

    def label_list(self) -> list[dict[str, object]]:
        # Return all labels that have been created — simulates creation success.
        return [{"name": name} for name, _color, _desc in self.labels_created]

    def pr_comment(self, number: int, body_file: Path) -> None:
        pass


class FakeGitHubWithChecks(FakeGitHub):
    """FakeGitHub whose pr_checks returns a configurable list."""

    def __init__(self, checks: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self.checks = checks if checks is not None else []

    def pr_checks(self, number: int) -> list[dict[str, Any]]:
        # Mirror production GitHub.pr_checks: inject databaseId/runId from the
        # check link, but only when not already provided by the test.
        return [
            {
                **check,
                "databaseId": check.get("databaseId", _job_id_from_link(check.get("link"))),
                "runId": check.get("runId", _run_id_from_link(check.get("link"))),
            }
            for check in self.checks
        ]


class FakeGitHubWithMissingRequired(FakeGitHubWithChecks):
    """No required checks present at all, so every required check is missing."""

    def __init__(self) -> None:
        super().__init__(checks=[])


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
    """
    tests_dir = Path(__file__).parent
    repo_root = tests_dir.parent
    code = "\n".join(
        [
            "import os, sys",
            "sys.path.insert(0, sys.argv[1])",
            "from test_charlie_work import FakeGitHub",
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
    results: set[str] = set()
    for seed in range(5):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        proc = subprocess.run(
            [sys.executable, "-c", code, str(tests_dir)],
            env=env,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        results.add(proc.stdout.strip())
    assert len(results) == 1, f"merge_base varied across hash seeds: {results}"


def test_dispatch_writes_worker_prompt_and_session_manifest(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
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


def test_dispatch_excludes_issue_with_open_tracked_pr(tmp_path: Path) -> None:
    """Issue #257: a labeled issue with an open tracked PR must never be a
    dispatch candidate, even with no state.json entry (label drift after
    manual salvage or escalation churn) — GitHub's open-PR set is the
    ground truth, not labels or state."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # The default FakeGitHub fixture is exactly the hazard case: issue 123 is
    # labeled ready and has NO state entry, while open PR 456 tracks it.
    assert app.gh.prs[0]["state"] == "OPEN"
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    assert not prompt_path.exists()
    assert (123, "agent:queued") not in fake_gh.labels_added
    assert (123, "agent:in-progress") not in fake_gh.labels_added


def test_dispatch_skips_ready_issue_with_merged_pr_reference(tmp_path: Path) -> None:
    """Issue #203: a ready issue whose merged PR is hijack-safely *bound* to it
    (here: the default fixture's ``agent/issue-123-...`` head branch matches the
    configured branch prefix) must not be dispatched, and — because binding is
    the same trust level issue #220 uses at merge time — the issue should be
    closed and labeled done as a belt-and-suspenders retry.

    Note: this fixture's PR title/body also happens to contain an "issue #123"
    text mention, but that is incidental to this test — the branch binding
    alone is sufficient to authorize the close. The mention-only path (no
    branch/closing-keyword binding) is isolated separately in
    test_dispatch_flags_but_does_not_close_ready_issue_with_bare_mention,
    which is the actual regression coverage for a bare-mention reference.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # PR #456 is merged; its headRefName ("agent/issue-123-fix-search", from
    # the default fixture) safely binds it to issue #123 via branch prefix.
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["title"] = "fix(scope): reap sidecar files on session exit (issue #123)"
    fake_gh.prs[0]["body"] = "This PR addresses issue #123."
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["merged_pr_referenced_issue_numbers"] == [123]
    assert result.data["merged_pr_closed_issue_numbers"] == [123]
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert 123 in fake_gh.closed_issues
    assert (123, "agent:done") in fake_gh.labels_added
    assert (123, "agent:queued") not in fake_gh.labels_added
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    assert not prompt_path.exists()


def test_dispatch_finalizes_closed_issue_for_aviator_merge(tmp_path: Path) -> None:
    """Issue #427: an Aviator-mergequeue merged PR closes its linked issue via
    GitHub's 'Closes #N'. The issue may already be CLOSED while still carrying
    stale automated-ready and agent:pr-open labels. dispatch() must scan
    merged PRs regardless of the issue's open/closed state, run the merged
    label transition, and clean up state.json.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubRespectingState(FakeGitHub):
        def issue_list(self, labels=None, state=None):
            issues = super().issue_list(labels=labels, state=state)
            if state is None:
                state = "OPEN"
            if state.upper() == "ALL":
                return issues
            return [i for i in issues if (i.get("state") or "OPEN").upper() == state.upper()]

    fake_gh = FakeGitHubRespectingState()

    # Seed state as if merge_ready handed the PR off to Aviator.
    seed = load_state(paths.state_file)
    seed["prs"]["456"] = {"status": "mergequeue", "issue_number": 123}
    seed["issues"]["123"] = {"status": "reviewing", "number": 123}
    save_state(paths.state_file, seed)

    # Aviator merged the PR; GitHub auto-closed the issue.
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["labels"] = [{"name": "mergequeue"}]
    fake_gh.issues[0]["state"] = "CLOSED"
    fake_gh.issues[0]["labels"] = [
        {"name": config.labels.ready},
        {"name": config.labels.pr_open},
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["merged_pr_referenced_issue_numbers"] == [123]
    assert result.data["merged_pr_closed_issue_numbers"] == [123]
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert (123, config.labels.done) in fake_gh.labels_added
    assert (123, config.labels.pr_open) in fake_gh.labels_removed
    assert (123, config.labels.ready) in fake_gh.labels_removed
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "merged"
    assert state["prs"]["456"]["merged"] is True
    assert state["issues"]["123"]["status"] == "closed"


def test_dispatch_finalizes_closed_issue_with_pr_outside_merged_pr_list_window(
    tmp_path: Path,
) -> None:
    """Issue #433: a closed ready-labeled issue whose merged PR is older than the
    most-recent-500 window of ``merged_pr_list()`` must still be finalized.

    This simulates a repo with 1250+ merged PRs where the global list cannot see
    the linked PR, but a per-issue ``gh pr list --search`` lookup finds it.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubOutsideWindow(FakeGitHub):
        def issue_list(self, labels=None, state=None):
            issues = super().issue_list(labels=labels, state=state)
            if state is None:
                state = "OPEN"
            if state.upper() == "ALL":
                return issues
            return [i for i in issues if (i.get("state") or "OPEN").upper() == state.upper()]

        def merged_pr_list(self):
            # Simulate the 500-window truncation: the real merged PR is not visible.
            return []

    fake_gh = FakeGitHubOutsideWindow()

    seed = load_state(paths.state_file)
    seed["prs"]["456"] = {"status": "mergequeue", "issue_number": 123}
    seed["issues"]["123"] = {"status": "reviewing", "number": 123}
    save_state(paths.state_file, seed)

    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["labels"] = [{"name": "mergequeue"}]
    fake_gh.issues[0]["state"] = "CLOSED"
    fake_gh.issues[0]["labels"] = [
        {"name": config.labels.ready},
        {"name": config.labels.pr_open},
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["merged_pr_referenced_issue_numbers"] == [123]
    assert result.data["merged_pr_closed_issue_numbers"] == [123]
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert (123, config.labels.done) in fake_gh.labels_added
    assert (123, config.labels.pr_open) in fake_gh.labels_removed
    assert (123, config.labels.ready) in fake_gh.labels_removed
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "merged"
    assert state["prs"]["456"]["merged"] is True
    assert state["issues"]["123"]["status"] == "closed"


def test_dispatch_caps_merge_finalization_per_pass(tmp_path: Path) -> None:
    """Issue #432: merge-finalization is capped at dispatch.finalize_limit per pass.

    A backlog of bound-and-closed issues carrying a stale ready marker drains
    oldest-first across passes; no single pass can monopolize the pass budget.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(finalize_limit=2))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class LabelMutatingFakeGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            super().add_issue_label(number, label)
            for issue in self.issues:
                if issue["number"] != number:
                    continue
                names = {item.get("name") for item in issue["labels"]}
                if label not in names:
                    issue["labels"].append({"name": label})
                break
            return True

        def remove_issue_label(self, number: int, label: str) -> bool:
            super().remove_issue_label(number, label)
            for issue in self.issues:
                if issue["number"] == number:
                    issue["labels"] = [
                        item for item in issue["labels"] if item.get("name") != label
                    ]
                    break
            return True

    fake_gh = LabelMutatingFakeGitHub()
    fake_gh.issues = [
        {
            "number": 101,
            "title": "Fix one",
            "url": "https://example.test/issues/101",
            "body": "body one",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-01T00:00:00Z",
        },
        {
            "number": 102,
            "title": "Fix two",
            "url": "https://example.test/issues/102",
            "body": "body two",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-02T00:00:00Z",
        },
        {
            "number": 103,
            "title": "Fix three",
            "url": "https://example.test/issues/103",
            "body": "body three",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-03T00:00:00Z",
        },
    ]
    fake_gh.prs = [
        {
            "number": 201,
            "title": "Fix #101",
            "url": "https://example.test/pull/201",
            "headRefName": "agent/issue-101-fix-one",
            "baseRefName": "main",
            "headRefOid": "sha-101",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #101",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
        {
            "number": 202,
            "title": "Fix #102",
            "url": "https://example.test/pull/202",
            "headRefName": "agent/issue-102-fix-two",
            "baseRefName": "main",
            "headRefOid": "sha-102",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #102",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
        {
            "number": 203,
            "title": "Fix #103",
            "url": "https://example.test/pull/203",
            "headRefName": "agent/issue-103-fix-three",
            "baseRefName": "main",
            "headRefOid": "sha-103",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #103",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result1 = app.dispatch()

    assert result1.ok is True
    assert result1.data["selected_count"] == 0
    assert result1.data["merged_pr_referenced_issue_numbers"] == [101, 102, 103]
    assert result1.data["merged_pr_closed_issue_numbers"] == [101, 102]
    assert 101 in fake_gh.closed_issues
    assert 102 in fake_gh.closed_issues
    assert 103 not in fake_gh.closed_issues
    assert (101, config.labels.done) in fake_gh.labels_added
    assert (102, config.labels.done) in fake_gh.labels_added
    assert (103, config.labels.done) not in fake_gh.labels_added
    state = load_state(paths.state_file)
    assert state["issues"]["101"]["status"] == "closed"
    assert state["issues"]["102"]["status"] == "closed"
    assert "103" not in state["issues"]

    # Second pass drains the remaining bound-and-closed issue.
    result2 = app.dispatch()
    assert result2.ok is True
    assert result2.data["selected_count"] == 0
    assert result2.data["merged_pr_closed_issue_numbers"] == [103]
    assert 103 in fake_gh.closed_issues
    assert (103, config.labels.done) in fake_gh.labels_added
    state = load_state(paths.state_file)
    assert state["issues"]["103"]["status"] == "closed"


def test_dispatch_caps_externally_merged_issue_lookups_at_finalize_limit(
    tmp_path: Path,
) -> None:
    """Issue #433 review: per-issue merged-PR lookups are capped at
    ``dispatch.finalize_limit`` and processed oldest-first.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(finalize_limit=2))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class FakeGitHubCappedOutsideWindow(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_prs_for_issue_calls: list[int] = []

        def merged_pr_list(self):
            # Outside the 500-window: the cheap list cannot see any merged PRs.
            return []

        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            self.merged_prs_for_issue_calls.append(issue_number)
            return github_module._MergedPRSearchResult(
                super().merged_prs_for_issue(issue_number, branch_prefix),
                ok=True,
            )

        def add_issue_label(self, number: int, label: str) -> bool:
            super().add_issue_label(number, label)
            for issue in self.issues:
                if issue["number"] != number:
                    continue
                names = {item.get("name") for item in issue["labels"]}
                if label not in names:
                    issue["labels"].append({"name": label})
                break
            return True

        def remove_issue_label(self, number: int, label: str) -> bool:
            super().remove_issue_label(number, label)
            for issue in self.issues:
                if issue["number"] == number:
                    issue["labels"] = [
                        item for item in issue["labels"] if item.get("name") != label
                    ]
                    break
            return True

    fake_gh = FakeGitHubCappedOutsideWindow()
    fake_gh.issues = [
        {
            "number": 101,
            "title": "Fix one",
            "url": "https://example.test/issues/101",
            "body": "body one",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-03T00:00:00Z",
        },
        {
            "number": 102,
            "title": "Fix two",
            "url": "https://example.test/issues/102",
            "body": "body two",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-01T00:00:00Z",
        },
        {
            "number": 103,
            "title": "Fix three",
            "url": "https://example.test/issues/103",
            "body": "body three",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-02T00:00:00Z",
        },
    ]
    fake_gh.prs = [
        {
            "number": 201,
            "title": "Fix #101",
            "url": "https://example.test/pull/201",
            "headRefName": "agent/issue-101-fix-one",
            "baseRefName": "main",
            "headRefOid": "sha-101",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #101",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
        {
            "number": 202,
            "title": "Fix #102",
            "url": "https://example.test/pull/202",
            "headRefName": "agent/issue-102-fix-two",
            "baseRefName": "main",
            "headRefOid": "sha-102",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #102",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
        {
            "number": 203,
            "title": "Fix #103",
            "url": "https://example.test/pull/203",
            "headRefName": "agent/issue-103-fix-three",
            "baseRefName": "main",
            "headRefOid": "sha-103",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #103",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result1 = app.dispatch()

    assert result1.ok is True
    assert result1.data["selected_count"] == 0
    # Oldest issues first: 102 (Jan 1), 103 (Jan 2); 101 (Jan 3) is deferred.
    assert result1.data["merged_pr_closed_issue_numbers"] == [102, 103]
    assert fake_gh.merged_prs_for_issue_calls == [102, 103]
    assert 101 not in fake_gh.closed_issues
    assert 102 in fake_gh.closed_issues
    assert 103 in fake_gh.closed_issues

    # Second pass drains the remaining issue.
    result2 = app.dispatch()
    assert result2.ok is True
    assert result2.data["selected_count"] == 0
    assert result2.data["merged_pr_closed_issue_numbers"] == [101]
    assert fake_gh.merged_prs_for_issue_calls == [102, 103, 101]
    assert 101 in fake_gh.closed_issues


def test_dispatch_circuit_breaker_stops_externally_merged_issue_lookups(
    tmp_path: Path,
) -> None:
    """Issue #433 review: 3 consecutive merged_prs_for_issue failures stop the
    pass; the next pass resumes from the remaining issues.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(finalize_limit=10))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class FakeGitHubWithFailingSearch(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_prs_for_issue_calls: list[int] = []

        def merged_pr_list(self):
            return []

        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            self.merged_prs_for_issue_calls.append(issue_number)
            if issue_number in {101, 102, 103}:
                return github_module._MergedPRSearchResult([], ok=False)
            return github_module._MergedPRSearchResult(
                super().merged_prs_for_issue(issue_number, branch_prefix),
                ok=True,
            )

    fake_gh = FakeGitHubWithFailingSearch()
    fake_gh.issues = [
        {
            "number": 101,
            "title": "Fix one",
            "url": "https://example.test/issues/101",
            "body": "body one",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-01T00:00:00Z",
        },
        {
            "number": 102,
            "title": "Fix two",
            "url": "https://example.test/issues/102",
            "body": "body two",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-02T00:00:00Z",
        },
        {
            "number": 103,
            "title": "Fix three",
            "url": "https://example.test/issues/103",
            "body": "body three",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-03T00:00:00Z",
        },
        {
            "number": 104,
            "title": "Fix four",
            "url": "https://example.test/issues/104",
            "body": "body four",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-04T00:00:00Z",
        },
    ]
    fake_gh.prs = [
        {
            "number": 204,
            "title": "Fix #104",
            "url": "https://example.test/pull/204",
            "headRefName": "agent/issue-104-fix-four",
            "baseRefName": "main",
            "headRefOid": "sha-104",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #104",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    # 101-103 all fail; the circuit breaker opens before 104 is searched.
    assert fake_gh.merged_prs_for_issue_calls == [101, 102, 103]
    assert result.data["merged_pr_closed_issue_numbers"] == []
    assert 104 not in fake_gh.closed_issues

    # Next pass resumes from the beginning; if failures continue it stops again.
    result2 = app.dispatch()
    assert fake_gh.merged_prs_for_issue_calls == [101, 102, 103, 101, 102, 103]
    assert result2.data["merged_pr_closed_issue_numbers"] == []


def test_dispatch_strips_ready_from_closed_unmerged_issue(tmp_path: Path) -> None:
    """Issue #429: a closed ready issue with no merged PR binding it is stripped.

    Uses fully self-contained fixtures so the test is hermetic and does not
    depend on sibling tests or module-level FakeGitHub defaults.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 123,
            "title": "Search is broken",
            "url": "https://example.test/issues/123",
            "body": "Search is broken",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-07-01T00:00:00Z",
        }
    ]
    fake_gh.prs = []

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["merged_pr_referenced_issue_numbers"] == []
    assert result.data["merged_pr_closed_issue_numbers"] == []
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert 123 not in fake_gh.closed_issues
    assert (123, ready) in fake_gh.labels_removed
    assert fake_gh.labels_added == []
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "closed"
    stripped_events = [
        e
        for e in state.get("events", [])
        if e.get("kind") == "dispatch_closed_unmerged_ready_stripped"
    ]
    assert len(stripped_events) == 1
    assert stripped_events[0]["payload"]["issue_numbers"] == [123]


def test_dispatch_closed_unmerged_skips_candidate_on_lookup_failure(
    tmp_path: Path,
) -> None:
    """If merged_prs_for_issue fails for a closed-unmerged candidate, skip it."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class FakeGitHubFailingSearch(FakeGitHub):
        def merged_pr_list(self):
            return []

        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            return github_module._MergedPRSearchResult([], ok=False)

    fake_gh = FakeGitHubFailingSearch()
    fake_gh.issues = [
        {
            "number": 123,
            "title": "Search is broken",
            "url": "https://example.test/issues/123",
            "body": "Search is broken",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-07-01T00:00:00Z",
        }
    ]
    fake_gh.prs = []

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert (123, ready) not in fake_gh.labels_removed
    assert fake_gh.labels_added == []
    state = load_state(paths.state_file)
    stripped_events = [
        e
        for e in state.get("events", [])
        if e.get("kind") == "dispatch_closed_unmerged_ready_stripped"
    ]
    assert stripped_events == []


def test_dispatch_closed_unmerged_capped_at_finalize_limit(
    tmp_path: Path,
) -> None:
    """Issue #432/#433: closed-unmerged stripping is capped at dispatch.finalize_limit
    and processed oldest-first.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(finalize_limit=2))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class FakeGitHubCapped(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_prs_for_issue_calls: list[int] = []

        def merged_pr_list(self):
            return []

        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            self.merged_prs_for_issue_calls.append(issue_number)
            return github_module._MergedPRSearchResult([], ok=True)

    fake_gh = FakeGitHubCapped()
    fake_gh.issues = [
        {
            "number": 101,
            "title": "Fix one",
            "url": "https://example.test/issues/101",
            "body": "body one",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-03T00:00:00Z",
        },
        {
            "number": 102,
            "title": "Fix two",
            "url": "https://example.test/issues/102",
            "body": "body two",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-01T00:00:00Z",
        },
        {
            "number": 103,
            "title": "Fix three",
            "url": "https://example.test/issues/103",
            "body": "body three",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-02T00:00:00Z",
        },
    ]
    fake_gh.prs = []

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    # Oldest issues first: 102 (Jan 1), 103 (Jan 2); 101 (Jan 3) is deferred.
    assert fake_gh.merged_prs_for_issue_calls == [102, 103]
    state = load_state(paths.state_file)
    stripped_events = [
        e
        for e in state.get("events", [])
        if e.get("kind") == "dispatch_closed_unmerged_ready_stripped"
    ]
    assert len(stripped_events) == 1
    assert stripped_events[0]["payload"]["issue_numbers"] == [102, 103]


def test_dispatch_merged_pr_list_called_once_per_pass(tmp_path: Path) -> None:
    """Issue #446: merged_pr_list() is listed once per dispatch pass even when
    both finalization and the dispatch binding step need it.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(default_limit=1))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class CountingFakeGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_pr_list_calls = 0

        def merged_pr_list(self):
            self.merged_pr_list_calls += 1
            return super().merged_pr_list()

    fake_gh = CountingFakeGitHub()
    fake_gh.issues = [
        {
            "number": 101,
            "title": "Closed issue",
            "url": "https://example.test/issues/101",
            "body": "body one",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": "2026-01-01T00:00:00Z",
        },
        {
            "number": 102,
            "title": "Open issue",
            "url": "https://example.test/issues/102",
            "body": "body two",
            "labels": [{"name": ready}],
            "state": "OPEN",
            "createdAt": "2026-01-02T00:00:00Z",
        },
    ]
    fake_gh.prs = [
        {
            "number": 201,
            "title": "Fix #101",
            "url": "https://example.test/pull/201",
            "headRefName": "agent/issue-101-fix",
            "baseRefName": "main",
            "headRefOid": "sha-101",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #101",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch()

    assert result.ok is True
    assert fake_gh.merged_pr_list_calls == 1
    assert result.data["merged_pr_closed_issue_numbers"] == [101]
    assert result.data["selected_count"] == 1


def test_dispatch_circuit_breaker_resets_on_interleaved_success(tmp_path: Path) -> None:
    """Issue #446: the consecutive-failure circuit breaker resets on success.

    Sequence fail, fail, success, fail, fail, fail must not trip at the third
    candidate (the success resets the counter) and must trip after the final
    three consecutive failures, leaving the next candidate untouched.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(finalize_limit=10))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    ready = config.labels.ready

    class FakeGitHubWithInterleavedSearch(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_prs_for_issue_calls: list[int] = []

        def merged_pr_list(self):
            return []

        def merged_prs_for_issue(self, issue_number: int, branch_prefix: str):
            self.merged_prs_for_issue_calls.append(issue_number)
            if issue_number == 103:
                return github_module._MergedPRSearchResult(
                    [
                        {
                            "number": 203,
                            "title": "Fix #103",
                            "url": "https://example.test/pull/203",
                            "headRefName": "agent/issue-103-fix",
                            "baseRefName": "main",
                            "headRefOid": "sha-103",
                            "mergeStateStatus": "CLEAN",
                            "body": "Closes #103",
                            "labels": [],
                            "isCrossRepository": False,
                            "state": "MERGED",
                        }
                    ],
                    ok=True,
                )
            if issue_number in {101, 102, 104, 105, 106}:
                return github_module._MergedPRSearchResult([], ok=False)
            if issue_number == 107:
                return github_module._MergedPRSearchResult(
                    [
                        {
                            "number": 207,
                            "title": "Fix #107",
                            "url": "https://example.test/pull/207",
                            "headRefName": "agent/issue-107-fix",
                            "baseRefName": "main",
                            "headRefOid": "sha-107",
                            "mergeStateStatus": "CLEAN",
                            "body": "Closes #107",
                            "labels": [],
                            "isCrossRepository": False,
                            "state": "MERGED",
                        }
                    ],
                    ok=True,
                )
            return github_module._MergedPRSearchResult(
                super().merged_prs_for_issue(issue_number, branch_prefix), ok=True
            )

    fake_gh = FakeGitHubWithInterleavedSearch()
    fake_gh.issues = [
        {
            "number": n,
            "title": f"Issue {n}",
            "url": f"https://example.test/issues/{n}",
            "body": f"body {n}",
            "labels": [{"name": ready}],
            "state": "CLOSED",
            "createdAt": f"2026-01-{i:02d}T00:00:00Z",
        }
        for i, n in enumerate(range(101, 108), start=1)
    ]
    fake_gh.prs = []

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert fake_gh.merged_prs_for_issue_calls == [101, 102, 103, 104, 105, 106]
    assert 107 not in fake_gh.merged_prs_for_issue_calls
    assert result.data["merged_pr_closed_issue_numbers"] == [103]
    assert 103 in fake_gh.closed_issues
    assert 107 not in fake_gh.closed_issues
    state = load_state(paths.state_file)
    assert state["issues"]["103"]["status"] == "closed"
    assert "107" not in state["issues"]


def test_dispatch_flags_but_does_not_close_ready_issue_with_bare_mention(
    tmp_path: Path,
) -> None:
    """Issue #203 (review redesign): a merged PR that only *mentions* the issue
    in free text — no branch-prefix binding, no closing keyword — must never
    authorize an autonomous close. It must be excluded from dispatch and
    flagged with the human-needed label, and the issue must stay open, for
    the operator to decide.

    Isolates the mention-only path from the hijack-safe binding path: unlike
    test_dispatch_skips_ready_issue_with_merged_pr_reference (which reuses the
    default fixture's agent/issue-123-... head branch), this PR's head branch
    does not match the configured branch prefix and its body contains no
    closing keyword, so linked_issue_number() cannot bind it — only the loose
    "issue #N" text scan can find it.

    Mutation-verify: gutting issue_numbers_mentioned_by_pr() to return an
    empty set makes this test fail (selected_count would become 1 and no
    human-needed label would be added).
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # PR #456 is merged, same-repo, but its head branch does not match the
    # "agent/issue" prefix and neither title nor body has a closing keyword —
    # only a bare "issue #123" mention.
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "chore: unrelated cleanup"
    fake_gh.prs[0]["body"] = "While in the area, this also happens to fix issue #123."
    assert fake_gh.prs[0]["isCrossRepository"] is False

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["merged_pr_referenced_issue_numbers"] == [123]
    assert result.data["merged_pr_closed_issue_numbers"] == []
    assert result.data["merged_pr_flagged_issue_numbers"] == [123]
    assert 123 not in fake_gh.closed_issues
    assert (123, "agent:human-needed") in fake_gh.labels_added
    assert (123, "agent:done") not in fake_gh.labels_added
    assert (123, "agent:queued") not in fake_gh.labels_added
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    # The issue itself is left open — only the label lifecycle is touched.
    assert fake_gh.issue_view(123)["state"] == "OPEN"


def test_dispatch_ignores_cross_repo_pr_mentioning_ready_issue(tmp_path: Path) -> None:
    """Regression for the isCrossRepository guard (workflow.py,
    _merged_pr_referenced_issue_numbers): a merged PR whose provenance is
    cross-repo (isCrossRepository=True) must never have its free-text "issue
    #N" mention counted, even though the text itself is indistinguishable
    from a same-repo mention. isCrossRepository describes head-branch
    provenance, not which repo the text refers to, but it is the only
    provenance signal available and must still gate the mention scan.

    Before this test, mutating the guard to unconditionally count mentions
    (e.g. `if True:` regardless of isCrossRepository) passed every existing
    test — this pins the behavior so that mutation is caught: issue #123
    must remain a normal, undisturbed dispatch candidate.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Merged PR #456 is cross-repo (e.g. a fork), has no branch/closing-keyword
    # binding, but its body text mentions "issue #123".
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "some-fork-branch"
    fake_gh.prs[0]["title"] = "unrelated fork PR"
    fake_gh.prs[0]["body"] = "Unrelated change; see issue #123 in a different project."
    fake_gh.prs[0]["isCrossRepository"] = True

    result = app.dispatch(limit=1)

    assert result.ok is True
    # Issue #123 is dispatched normally — the cross-repo mention must not
    # exclude or flag it.
    assert result.data["selected_count"] == 1
    assert result.data["merged_pr_referenced_issue_numbers"] == []
    assert result.data["merged_pr_closed_issue_numbers"] == []
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert 123 not in fake_gh.closed_issues
    assert (123, "agent:human-needed") not in fake_gh.labels_added


def test_dispatch_skips_merged_pr_list_query_when_no_ready_issues(tmp_path: Path) -> None:
    """Issue #361: when there are no ready-labeled issues to consider,
    merged_pr_list() must not be called at all.
    _merged_pr_referenced_issue_numbers() intersects its scan against the
    ready-issue-number set, so with zero ready issues it always returns empty
    sets regardless of what merged_pr_list() would have returned — the query
    is pure waste on every pass with an empty ready queue, and this is the
    dominant case the guard is meant to eliminate.
    """

    class FakeGitHubCountingMergedPrCalls(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_pr_list_calls = 0
            self.issues = []  # no ready-labeled issues at all

        def merged_pr_list(self):
            self.merged_pr_list_calls += 1
            return super().merged_pr_list()

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubCountingMergedPrCalls()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert fake_gh.merged_pr_list_calls == 0


def test_dispatch_still_queries_merged_pr_list_when_ready_issues_exist(tmp_path: Path) -> None:
    """Sanity counterpart to the guard above: when a ready issue IS in the
    queue, merged_pr_list() must still be called — issue #203's guarantee
    (never re-dispatch an issue a merged PR already covers) depends on it.
    """

    class FakeGitHubCountingMergedPrCalls(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_pr_list_calls = 0

        def merged_pr_list(self):
            self.merged_pr_list_calls += 1
            return super().merged_pr_list()

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubCountingMergedPrCalls()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert fake_gh.merged_pr_list_calls == 1


def test_dispatch_only_issues_selects_explicit_subset(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Numbers not among the dispatchable candidates are skipped; only the
    # explicit, dispatchable match is selected (dependency-ordered waves).
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(only_issues="999, 123")

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert (123, "agent:queued") in fake_gh.labels_added


def test_dispatch_worker_template_selects_claude_code_variant(tmp_path: Path) -> None:
    config = OrchestratorConfig(dispatch=DispatchConfig(worker_template="worker_claude_code.md"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    app.gh.prs[0]["state"] = "CLOSED"
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

    app.gh.prs[0]["state"] = "CLOSED"
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
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; sys.exit(7)"),
        )
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
        app.gh.prs[0]["state"] = "CLOSED"
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
    app.gh.prs[0]["state"] = "CLOSED"
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
    app.gh.prs[0]["state"] = "CLOSED"
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
    app.gh.prs[0]["state"] = "CLOSED"
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
    app.gh.prs[0]["state"] = "CLOSED"
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
    app.gh.prs[0]["state"] = "CLOSED"
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
        app.gh.prs[0]["state"] = "CLOSED"
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
            require_current_base=False,
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


def test_merge_ready_checks_unavailable_returns_false(tmp_path: Path) -> None:
    """gh pr checks command failure must be reported as checks unavailable, not merge."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubWithChecksUnavailable(FakeGitHub):
        def pr_checks(self, number: int):
            return None

    fake_gh = FakeGitHubWithChecksUnavailable()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.merge_ready(456, merge=True)

    assert result.ok is False
    assert result.data["checks_unavailable"] is True
    assert result.data["can_merge"] is False
    assert result.data["merged"] is False
    assert fake_gh.merged == []


def _review_queue_app(
    tmp_path: Path, *, prs: list[dict[str, Any]] | None = None
) -> OrchestratorApp:
    """Build an OrchestratorApp with FakeGitHub and a default state file."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    if prs is not None:
        fake_gh.prs = prs
    return OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)


def _write_review_packet(
    tmp_path: Path,
    pr_number: int,
    packet_head_sha: str,
    decision: dict[str, Any] | None = None,
) -> Path:
    """Create a review packet fixture for a PR."""
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps({"number": pr_number, "headRefOid": packet_head_sha}),
        encoding="utf-8",
    )
    (pr_dir / "review-prompt.md").write_text(
        f"review prompt for PR #{pr_number}",
        encoding="utf-8",
    )
    if decision is not None:
        (pr_dir / "review-decision.json").write_text(
            json.dumps(decision),
            encoding="utf-8",
        )
    return pr_dir


def test_review_queue_includes_missing_pending_and_stale_decisions(
    tmp_path: Path,
) -> None:
    """Issue #369: review_queue enumerates current packets awaiting a verdict."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10: missing decision",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
        {
            "number": 200,
            "title": "Fix #20: pending decision",
            "url": "https://example.test/pull/200",
            "headRefName": "agent/issue-20-fix",
            "baseRefName": "main",
            "headRefOid": "sha-200",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #20",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
        {
            "number": 300,
            "title": "Fix #30: stale request_changes",
            "url": "https://example.test/pull/300",
            "headRefName": "agent/issue-30-fix",
            "baseRefName": "main",
            "headRefOid": "sha-300-new",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #30",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
        {
            "number": 400,
            "title": "Fix #40: approved on current head",
            "url": "https://example.test/pull/400",
            "headRefName": "agent/issue-40-fix",
            "baseRefName": "main",
            "headRefOid": "sha-400",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #40",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
        {
            "number": 500,
            "title": "Fix #50: stale packet",
            "url": "https://example.test/pull/500",
            "headRefName": "agent/issue-50-fix",
            "baseRefName": "main",
            "headRefOid": "sha-500-new",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #50",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
        {
            "number": 600,
            "title": "No linked issue",
            "url": "https://example.test/pull/600",
            "headRefName": "feature/unlinked",
            "baseRefName": "main",
            "headRefOid": "sha-600",
            "mergeStateStatus": "CLEAN",
            "body": "Some feature",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _review_queue_app(tmp_path, prs=prs)

    # PR 100: current packet, no decision -> decision missing
    _write_review_packet(tmp_path, 100, "sha-100")
    # PR 200: current packet, pending decision
    _write_review_packet(tmp_path, 200, "sha-200", {"decision": "pending"})
    # PR 300: current packet, request_changes from prior head -> stale
    _write_review_packet(
        tmp_path,
        300,
        "sha-300-new",
        {"decision": "request_changes", "reviewed_head_sha": "sha-300-old"},
    )
    # PR 400: current packet, approved on current head -> excluded
    _write_review_packet(
        tmp_path,
        400,
        "sha-400",
        {"decision": "approved", "reviewed_head_sha": "sha-400"},
    )
    # PR 500: stale packet (recorded head differs from live head) -> excluded
    _write_review_packet(tmp_path, 500, "sha-500-old")
    # PR 600: unlinked PR has no packet, so excluded

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": 100,
            "issue": 10,
            "packet_head_sha": "sha-100",
            "decision": "missing",
            "reviewed_head_sha": None,
        },
        {
            "pr": 200,
            "issue": 20,
            "packet_head_sha": "sha-200",
            "decision": "pending",
            "reviewed_head_sha": None,
        },
        {
            "pr": 300,
            "issue": 30,
            "packet_head_sha": "sha-300-new",
            "decision": "stale",
            "reviewed_head_sha": "sha-300-old",
        },
    ]


def test_review_queue_is_read_only(tmp_path: Path) -> None:
    """Issue #369: review_queue must not mutate state.json or PR-directory files."""
    app = _review_queue_app(tmp_path)
    _write_review_packet(tmp_path, 456, "sha-abc123")
    before_state = json.loads(app.paths.state_file.read_text(encoding="utf-8"))
    pr_json = (app.paths.prs / "pr-456" / "pr.json").read_text(encoding="utf-8")
    prompt = (app.paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8")

    result = app.review_queue()

    assert result.ok is True
    after_state = json.loads(app.paths.state_file.read_text(encoding="utf-8"))
    assert after_state == before_state
    assert (app.paths.prs / "pr-456" / "pr.json").read_text(encoding="utf-8") == pr_json
    assert (app.paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8") == prompt


def _review_queue_carry_forward_app(
    tmp_path: Path,
    *,
    prs: list[dict[str, Any]] | None = None,
    dry_run: bool = False,
) -> OrchestratorApp:
    """Build an OrchestratorApp for carry-forward tests."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    if prs is not None:
        fake_gh.prs = prs
    return OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=dry_run)


def test_review_queue_carries_forward_approved_on_identical_patch_id(tmp_path: Path) -> None:
    """Issue #411: an approved verdict whose cumulative patch-id is unchanged
    should be carried forward to the new head and not reported as stale."""
    from charlie_work.janitor import _calculate_patch_id

    diff_text = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+line3\n"
        " line4\n"
    )
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-abc123"
    new_head = "sha-rebased123"
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
    fake_gh.diffs[pr_number] = diff_text

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []

    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == new_head
    assert decision["reviewed_patch_id"] == patch_id
    # Issue #414 (d): the tier-1 fast path is unchanged and tags its own
    # carry-forwards distinctly from tier 2.
    assert decision["carry_forward_tier"] == "patch-id"
    assert old_head in decision["carried_forward_from"]

    state = load_state(app.paths.state_file)
    assert state["prs"][str(pr_number)]["reviewed_head_sha"] == new_head
    assert state["prs"][str(pr_number)]["carried_forward_from"] == [old_head]


def test_review_queue_carries_forward_request_changes_on_identical_patch_id(
    tmp_path: Path,
) -> None:
    """Issue #411: a request_changes verdict is also valid when the patch is identical."""
    from charlie_work.janitor import _calculate_patch_id

    diff_text = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 modified\n"
    )
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-old-head"
    new_head = "sha-sync-merge-head"
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
    fake_gh.diffs[pr_number] = diff_text

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "request_changes",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []

    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == new_head
    assert decision["carried_forward_from"] == [old_head]


def test_review_queue_carries_forward_blocked_on_identical_patch_id(tmp_path: Path) -> None:
    """Issue #413: a blocked verdict whose cumulative patch-id is unchanged
    should be carried forward to the new head and not reported as stale."""
    from charlie_work.janitor import _calculate_patch_id

    diff_text = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 blocked\n"
    )
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-old-head"
    new_head = "sha-sync-merge-head"
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
    fake_gh.diffs[pr_number] = diff_text

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "blocked",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []

    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == new_head
    assert decision["carried_forward_from"] == [old_head]

    state = load_state(app.paths.state_file)
    assert state["prs"][str(pr_number)]["reviewed_head_sha"] == new_head
    assert state["prs"][str(pr_number)]["carried_forward_from"] == [old_head]


def test_review_queue_reports_stale_on_different_patch_id(tmp_path: Path) -> None:
    """Issue #411: a head move that changes the cumulative diff is still stale."""
    from charlie_work.janitor import _calculate_patch_id

    old_diff = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 old\n"
    )
    new_diff = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 new\n"
    )
    old_patch_id = _calculate_patch_id(old_diff)
    new_head = "sha-new-head"
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
    fake_gh.diffs[pr_number] = new_diff

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": "sha-old-head",
            "reviewed_patch_id": old_patch_id,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": "sha-old-head",
        }
    ]

    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-old-head"


def test_review_queue_git_failure_falls_back_to_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #411: if git patch-id computation fails, treat the verdict as stale."""
    from charlie_work import workflow as workflow_module

    old_head = "sha-old-head"
    new_head = "sha-new-head"
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
    fake_gh.diffs[pr_number] = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 new\n"
    )

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": "known-patch-id",
            "carried_forward_from": [],
        },
    )

    monkeypatch.setattr(workflow_module, "_calculate_patch_id", lambda _diff: "")

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": old_head,
        }
    ]


def test_review_queue_dry_run_skips_carry_forward_write_but_not_stale_check(
    tmp_path: Path,
) -> None:
    """Issue #411: dry-run review-queue must not write but still hide stale verdicts
    when the cumulative patch-id is unchanged."""
    from charlie_work.janitor import _calculate_patch_id

    diff_text = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+line3\n"
        " line4\n"
    )
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-abc123"
    new_head = "sha-rebased123"
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
    app = _review_queue_carry_forward_app(tmp_path, prs=prs, dry_run=True)
    fake_gh = app.gh
    fake_gh.diffs[pr_number] = diff_text

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "carried_forward_from": [],
        },
    )
    before_decision = (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(
        encoding="utf-8"
    )
    before_state = app.paths.state_file.read_text(encoding="utf-8")

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []
    assert (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(
        encoding="utf-8"
    ) == before_decision
    assert app.paths.state_file.read_text(encoding="utf-8") == before_state


def test_review_queue_carries_forward_on_tier2_line_content_after_main_advance(
    tmp_path: Path,
) -> None:
    """Issue #414: patch-id is unstable across every main advance because the
    merge-base moves, which can shift a hunk's CONTEXT lines even when the
    PR's own +/- content is untouched. Tier 2 recognizes this via the
    ordered +/- line stream and changed-file set, ignoring context drift."""
    from charlie_work.janitor import _calculate_patch_id, _diff_content_signature

    # Reviewed diff: the PR added "+gamma" between unchanged "beta"/"delta".
    reviewed_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " alpha\n"
        " beta\n"
        "+gamma\n"
        " delta\n"
    )
    # Live diff: main advanced and changed the *context* line "beta" ->
    # "beta-updated" between the old and new merge-base. The PR's own
    # "+gamma" contribution is byte-identical and in the same position, but
    # git patch-id --stable hashes context text too, so the cumulative
    # patch-id differs even though nothing the PR authored changed.
    live_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " alpha\n"
        " beta-updated\n"
        "+gamma\n"
        " delta\n"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    live_patch_id = _calculate_patch_id(live_diff)
    assert reviewed_patch_id != live_patch_id, "test fixture must reproduce patch-id drift"

    reviewed_signature = _diff_content_signature(reviewed_diff)
    live_signature = _diff_content_signature(live_diff)
    assert reviewed_signature == live_signature, "tier-2 signature must ignore context drift"

    old_head = "sha-old-head"
    new_head = "sha-new-head-after-main-advance"
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
            "reviewed_changed_lines": list(reviewed_signature.changed_lines),
            "reviewed_changed_files": sorted(reviewed_signature.changed_files),
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []

    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == new_head
    assert decision["carry_forward_tier"] == "line-content"
    assert decision["reviewed_patch_id"] == live_patch_id
    assert old_head in decision["carried_forward_from"]

    state = load_state(app.paths.state_file)
    assert state["prs"][str(pr_number)]["reviewed_head_sha"] == new_head
    assert state["prs"][str(pr_number)]["carry_forward_tier"] == "line-content"
    assert state["prs"][str(pr_number)]["carried_forward_from"] == [old_head]


def test_review_queue_reports_stale_on_reordered_changed_lines(tmp_path: Path) -> None:
    """Issue #414: the same +/- lines in a different ORDER is a real semantic
    change and must not carry forward via tier 2 (ordered, not sorted)."""
    from charlie_work.janitor import _calculate_patch_id

    reviewed_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+first\n"
        "+second\n"
        " delta\n"
    )
    live_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+second\n"
        "+first\n"
        " delta\n"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    old_head = "sha-old-head"
    new_head = "sha-new-head"
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
            "reviewed_changed_lines": ["+first", "+second"],
            "reviewed_changed_files": ["file"],
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": old_head,
        }
    ]
    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == old_head
    assert "carry_forward_tier" not in decision


def test_review_queue_reports_stale_on_changed_file_set(tmp_path: Path) -> None:
    """Issue #414: an identical line stream but a different set of changed
    files (a file added/removed) is a real change and must not carry
    forward via tier 2."""
    from charlie_work.janitor import _calculate_patch_id

    reviewed_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+gamma\n"
        " delta\n"
    )
    live_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+gamma\n"
        " delta\n"
        "diff --git a/other b/other\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/other\n"
        "@@ -0,0 +1,1 @@\n"
        "+extra file\n"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    old_head = "sha-old-head"
    new_head = "sha-new-head"
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
            "reviewed_changed_lines": ["+gamma"],
            "reviewed_changed_files": ["file"],
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": old_head,
        }
    ]


def test_review_queue_git_failure_in_tier2_falls_back_to_stale(tmp_path: Path) -> None:
    """Issue #414: if the live diff cannot be fetched at all (gh/git failure),
    tier 2 must fail closed to stale even when a tier-2 baseline is recorded
    — never carry forward on uncertainty."""
    from charlie_work.janitor import _calculate_patch_id

    class FakeGitHubEmptyDiff(FakeGitHub):
        """Simulates a gh/git failure: pr_diff always returns empty."""

        def pr_diff(self, number: int) -> str:
            return ""

    reviewed_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+gamma\n"
        " delta\n"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    old_head = "sha-old-head"
    new_head = "sha-new-head"
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
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHubEmptyDiff()
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": reviewed_patch_id,
            "reviewed_changed_lines": ["+gamma"],
            "reviewed_changed_files": ["file"],
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": old_head,
        }
    ]
    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == old_head


def test_merge_ready_carries_forward_approved_verdict_on_tier2_line_content(
    tmp_path: Path,
) -> None:
    """Issue #414: the ship-it merge gate also carries forward via tier 2
    when patch-ids differ due to main-advance context drift but the ordered
    +/- lines and changed-file set are identical."""
    from charlie_work.janitor import _calculate_patch_id, _diff_content_signature

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    reviewed_diff = (
        "diff --git a/file b/file\n"
        "index 123..456 78910\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " alpha\n"
        " beta\n"
        "+gamma\n"
        " delta\n"
    )
    live_diff = (
        "diff --git a/file b/file\n"
        "index 123..789 78910\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " alpha\n"
        " beta-updated\n"
        "+gamma\n"
        " delta\n"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    live_patch_id = _calculate_patch_id(live_diff)
    assert reviewed_patch_id != live_patch_id, "test fixture must reproduce patch-id drift"
    reviewed_signature = _diff_content_signature(reviewed_diff)

    old_head = "sha-abc123"
    new_head = "sha-rebased123"

    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": old_head,
                "reviewed_patch_id": reviewed_patch_id,
                "reviewed_changed_lines": list(reviewed_signature.changed_lines),
                "reviewed_changed_files": sorted(reviewed_signature.changed_files),
                "summary": "lgtm",
            }
        ),
        encoding="utf-8",
    )

    # Simulate a rebase-style head move (not a 2-parent web-flow merge commit,
    # so _verify_synced_head would reject it) with genuine patch-id drift
    # from an intervening main advance, but content-identical +/- lines.
    fake_gh.pr_head_shas[456] = new_head
    fake_gh.diffs[456] = live_diff
    fake_gh.compare_overrides[("main", new_head)] = {
        "base_commit": {"sha": fake_gh.base_head_sha},
        "merge_base_commit": {"sha": fake_gh.base_head_sha},
    }
    fake_gh.commits[new_head] = {
        "parents": [{"sha": old_head}],
        "committer": {"login": "someone"},
        "commit": {"committer": {"name": "Not GitHub"}},
    }

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["can_merge"] is True
    assert result.data["merged"] is True
    assert result.data.get("head_moved") is not True
    assert fake_gh.merged == [(456, "squash")]

    decision = json.loads((decision_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert decision["reviewed_head_sha"] == new_head
    assert decision["carry_forward_tier"] == "line-content"
    assert decision["reviewed_patch_id"] == live_patch_id
    assert decision["carried_forward_from"] == [old_head]

    state = load_state(paths.state_file)
    pr_state = state["prs"]["456"]
    assert pr_state["reviewed_head_sha"] == new_head
    assert pr_state["carry_forward_tier"] == "line-content"
    assert pr_state["carried_forward_from"] == [old_head]
    assert pr_state["status"] != "reviewing"

    carry_events = [
        e for e in state["events"] if e["kind"] == "verdict_carried_forward_line_content"
    ]
    assert len(carry_events) == 1
    payload = carry_events[0]["payload"]
    assert payload["pr_number"] == 456
    assert payload["old_reviewed_head_sha"] == old_head
    assert payload["new_head_sha"] == new_head
    assert payload["patch_id"] == live_patch_id
    assert payload["carry_forward_tier"] == "line-content"
    assert payload["carried_forward_from"] == [old_head]

    # No review_started transition should fire for a tier-2 carry-forward.
    assert (123, "agent:reviewing") not in fake_gh.labels_added


def test_review_queue_reports_stale_on_mixed_binary_content_change(tmp_path: Path) -> None:
    """Issue #414 (review follow-up): a binary file's payload emits no +/-
    lines, so a text hunk that's byte-identical alongside a binary asset
    whose content genuinely changed (different git index blob hashes) must
    NOT carry forward via tier 2 — the signature can't see binary content
    and must fail closed instead of silently ignoring it."""
    from charlie_work.janitor import _calculate_patch_id, _diff_content_signature

    reviewed_diff = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+gamma\n"
        " delta\n"
        "diff --git a/logo.png b/logo.png\n"
        "index aaa1111..bbb2222 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    live_diff = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+gamma\n"
        " delta\n"
        "diff --git a/logo.png b/logo.png\n"
        "index ccc3333..ddd4444 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    live_patch_id = _calculate_patch_id(live_diff)
    assert reviewed_patch_id != live_patch_id, "test fixture must have differing patch-ids"
    reviewed_signature = _diff_content_signature(reviewed_diff)
    live_signature = _diff_content_signature(live_diff)
    # The bug this test guards against: without the has_binary gate, these
    # signatures compare equal despite genuinely different binary content.
    assert reviewed_signature.changed_lines == live_signature.changed_lines
    assert reviewed_signature.changed_files == live_signature.changed_files
    assert reviewed_signature.has_binary is True
    assert live_signature.has_binary is True

    old_head = "sha-old-head"
    new_head = "sha-new-head"
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
            "reviewed_changed_lines": list(reviewed_signature.changed_lines),
            "reviewed_changed_files": sorted(reviewed_signature.changed_files),
            "reviewed_has_binary": reviewed_signature.has_binary,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": old_head,
        }
    ]
    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == old_head
    assert "carry_forward_tier" not in decision


def test_review_queue_reports_stale_on_binary_only_content_change(tmp_path: Path) -> None:
    """Issue #414 (review follow-up): a pure binary-only diff (no hunks at
    all, so patch-id is empty on both sides) whose binary content genuinely
    changed must still report stale, not silently carry forward through the
    tier-2 signature fields (which never had any content to compare)."""
    from charlie_work.janitor import _calculate_patch_id, _diff_content_signature

    reviewed_diff = (
        "diff --git a/logo.png b/logo.png\n"
        "index aaa1111..bbb2222 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    live_diff = (
        "diff --git a/logo.png b/logo.png\n"
        "index ccc3333..ddd4444 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    assert reviewed_patch_id == "", "a hunk-less binary diff must not produce a patch-id"
    reviewed_signature = _diff_content_signature(reviewed_diff)

    old_head = "sha-old-head"
    new_head = "sha-new-head"
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
            "reviewed_changed_lines": list(reviewed_signature.changed_lines),
            "reviewed_changed_files": sorted(reviewed_signature.changed_files),
            "reviewed_has_binary": reviewed_signature.has_binary,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": old_head,
        }
    ]


def test_review_queue_carries_forward_identical_mixed_binary_and_text_via_tier1(
    tmp_path: Path,
) -> None:
    """Issue #414 (review follow-up): a byte-identical mixed text+binary diff
    (same index hash, same text) still carries forward — via tier 1's
    patch-id match, since tier 2 is never reached when tier 1 already
    succeeds. Confirms the has_binary gate does not regress the ordinary
    identical-content case."""
    from charlie_work.janitor import _calculate_patch_id

    diff_text = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+gamma\n"
        " delta\n"
        "diff --git a/logo.png b/logo.png\n"
        "index aaa1111..bbb2222 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-old-head"
    new_head = "sha-new-head"
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
    fake_gh.diffs[pr_number] = diff_text

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []
    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == new_head
    assert decision["carry_forward_tier"] == "patch-id"


def test_review_queue_stays_stale_with_empty_patch_id_from_rename(
    tmp_path: Path,
) -> None:
    """Issue #414 (review follow-up, deliberately NOT fixed): a pure-rename
    (100% similarity, no hunk) diff has no patch-id at all, even though it
    has a valid tier-2 signature on file. Eligibility for both tiers gates
    on ``reviewed_patch_id`` being recorded (matching #412 exactly) — an
    earlier attempt to gate on the signature fields' presence instead was
    reverted because ``record_review`` unconditionally records a signature
    (possibly trivially empty) for every approved/request_changes decision,
    which made unrelated no-op/placeholder diffs look like valid tier-2
    baselines and wrongly carried forward verdicts whose head had actually
    moved to different content (test_merge_ready_refuses_when_head_moved_
    after_approval and siblings). This case stays conservatively stale;
    tracked as a narrow follow-up rather than fixed here."""
    from charlie_work.janitor import _calculate_patch_id, _diff_content_signature

    rename_diff = (
        "diff --git a/old.py b/new.py\n"
        "similarity index 100%\n"
        "rename from old.py\n"
        "rename to new.py\n"
    )
    reviewed_patch_id = _calculate_patch_id(rename_diff)
    assert reviewed_patch_id == "", "a hunk-less rename diff must not produce a patch-id"
    reviewed_signature = _diff_content_signature(rename_diff)
    assert reviewed_signature.changed_files == frozenset({"new.py"})

    old_head = "sha-old-head"
    new_head = "sha-new-head"
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
    fake_gh.diffs[pr_number] = rename_diff

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": reviewed_patch_id,
            "reviewed_changed_lines": list(reviewed_signature.changed_lines),
            "reviewed_changed_files": sorted(reviewed_signature.changed_files),
            "reviewed_has_binary": reviewed_signature.has_binary,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": old_head,
        }
    ]
    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == old_head
    assert "carry_forward_tier" not in decision


def _dispatch_reviews_app(
    tmp_path: Path, *, prs: list[dict[str, Any]] | None = None
) -> OrchestratorApp:
    """Build an OrchestratorApp with review_dispatch enabled and an empty state file."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    if prs is not None:
        fake_gh.prs = prs
    return OrchestratorApp(tmp_path, paths, config, fake_gh)


def _fake_claude_worker_record(pr_number: int, branch: str) -> ClaudeWorkerRecord:
    """Return a successful Claude worker record for monkeypatched launches."""
    return ClaudeWorkerRecord(
        issue_number=pr_number,
        branch=branch,
        worktree_path="/fake/worktree",
        prompt_path="/fake/prompt.md",
        command=("claude", "-p"),
        pid=12345,
        started_at="2026-07-06T12:00:00Z",
        log_path="/fake/log.log",
        error=None,
        process_start_time=1.0,
    )


def test_dispatch_reviews_launches_for_all_queued_prs(monkeypatch, tmp_path: Path) -> None:
    """Issue #370: dispatch_reviews launches a Claude reviewer for every queued PR."""
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
        },
        {
            "number": 200,
            "title": "Fix #20",
            "url": "https://example.test/pull/200",
            "headRefName": "agent/issue-20-fix",
            "baseRefName": "main",
            "headRefOid": "sha-200",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #20",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    _write_review_packet(tmp_path, 200, "sha-200")

    launched: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append((args, kwargs))
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["launched_count"] == 2
    assert result.data["failed_count"] == 0
    assert result.data["selected_count"] == 2
    assert len(launched) == 2
    for _args, kwargs in launched:
        assert kwargs.get("review") is True
        assert "reviews" in str(kwargs.get("sessions_dir", ""))

    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert state["prs"]["100"]["reviewer_pid"] == 12345
    assert state["prs"]["200"]["review_dispatch_status"] == "review_dispatch_dispatched"


def test_dispatch_reviews_prevents_double_dispatch(monkeypatch, tmp_path: Path) -> None:
    """Issue #370: a live reviewer blocks re-dispatch of the same PR."""
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

    launched: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append((args, kwargs))
        return _fake_claude_worker_record(100, "agent/issue-10-fix")

    def fake_is_pid_alive(pid: int, *_args: Any, **_kwargs: Any) -> bool:
        # Pretend the fake reviewer PID is still alive.
        return pid == 12345

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    monkeypatch.setattr("charlie_work.workflow.is_pid_alive", fake_is_pid_alive)

    first = app.dispatch_reviews()
    assert first.data["launched_count"] == 1
    assert len(launched) == 1

    second = app.dispatch_reviews()
    assert second.data["launched_count"] == 0
    assert len(launched) == 1
    assert second.data["deferred_count"] == 1


def test_dispatch_reviews_respects_local_process_cap(monkeypatch, tmp_path: Path) -> None:
    """Issue #370: max_local_review_processes caps concurrent reviewer launches."""
    prs = [
        {
            "number": i,
            "title": f"Fix #{i}",
            "url": f"https://example.test/pull/{i}",
            "headRefName": f"agent/issue-{i}-fix",
            "baseRefName": "main",
            "headRefOid": f"sha-{i}",
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{i}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
        for i in range(300, 303)
    ]
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_local_review_processes=2),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    for pr in prs:
        _write_review_packet(tmp_path, pr["number"], pr["headRefOid"])

    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["launched_count"] == 2
    assert result.data["selected_count"] == 2
    assert result.data["skipped_count"] == 1
    assert result.data["max_local_review_processes"] == 2
    assert len(launched) == 2


def test_dispatch_reviews_redispatches_stalled_reviews(monkeypatch, tmp_path: Path) -> None:
    """Issue #370: a dead/stale reviewer claim is reaped and the PR re-dispatched."""
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

    # Seed a stale reviewer sidecar + state claim.
    reviews_dir = app._resolve(app.config.review_dispatch.reviews_dir)
    reviews_dir.mkdir(parents=True, exist_ok=True)
    old_started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    old_dispatched = old_started
    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-10-fix",
        "worktree_path": str(tmp_path / "worktrees" / "issue-100"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 99999,
        "started_at": old_started,
        "log_path": str(tmp_path / "log.log"),
        "error": None,
        "process_start_time": 1.0,
    }
    (reviews_dir / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old_dispatched,
            "reviewer_pid": 99999,
            "reviewer_process_start_time": 1.0,
        }
        save_state(app.paths.state_file, state)

    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(100, "agent/issue-10-fix")

    def fake_is_worker_alive(record: ClaudeWorkerRecord, **_kwargs: Any) -> bool:
        return False

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", fake_is_worker_alive)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["launched_count"] == 1
    assert launched == [100]
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert state["prs"]["100"]["reviewer_pid"] == 12345


def _init_git_repo(repo_root: Path) -> None:
    import subprocess

    repo_root.mkdir(parents=True, exist_ok=True)
    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )
    run(["git", "init", "--initial-branch=main"])
    run(["git", "config", "user.email", "test@example.test"])
    run(["git", "config", "user.name", "Test User"])
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    run(["git", "add", "README.md"])
    run(["git", "commit", "-m", "initial commit"])


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

    stalled = _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

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


def test_reap_completed_review_checkouts_removes_checkout_once_reviewer_exited(
    tmp_path: Path,
) -> None:
    """Issue #397: once record_review has recorded a verdict
    (review_dispatch_completed) and the reviewer's own sidecar process is no
    longer alive, the isolated review checkout is reaped. Liveness must be
    checked via the sidecar in reviews_dir (iter_workers), since record_review
    already cleared state.json's reviewer_pid by this point."""
    from datetime import timedelta

    from charlie_work.workflow import _reap_completed_review_checkouts
    from charlie_work.worktree import create_review_checkout

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"

    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    checkout = create_review_checkout(repo_root, 200, head_sha, reviews_dir=reviews_dir)
    assert checkout.path.exists()

    # A sidecar recording a definitely-dead pid (record_review does not
    # delete the sidecar itself, only clears state.json's own pid fields).
    reviews_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "issue_number": 200,
        "branch": "agent/issue-20-fix",
        "worktree_path": str(checkout.path),
        "prompt_path": str(checkout.path / ".orchestrator-prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,
        "started_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "log_path": str(reviews_dir / "issue-200.claude.log"),
        "error": None,
        "process_start_time": 1.0,
    }
    (reviews_dir / "issue-200.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {},
                "prs": {
                    "200": {
                        "number": 200,
                        "review_dispatch_status": "review_dispatch_completed",
                        "reviewer_pid": None,
                        "reviewer_process_start_time": None,
                    }
                },
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    reaped = _reap_completed_review_checkouts(repo_root, reviews_dir, state_file)

    assert reaped == [200]
    assert not checkout.path.exists()


def test_reap_completed_review_checkouts_skips_while_reviewer_still_alive(
    tmp_path: Path,
) -> None:
    """A completed-verdict PR whose reviewer sidecar is still alive must NOT
    have its checkout removed out from under the exiting process.

    Uses this test process's own PID/start-time as the sidecar's recorded
    identity, so the real (non-monkeypatched) claude_code.is_worker_alive
    liveness+identity check reports it genuinely alive — matching how
    test_count_live_sessions_ghost_worker_pid_corroborated_by_state (same
    file) proves a "ghost" liveness case elsewhere in this suite.
    """
    from charlie_work.claude_code import _get_process_start_time
    from charlie_work.workflow import _reap_completed_review_checkouts
    from charlie_work.worktree import create_review_checkout

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"

    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    checkout = create_review_checkout(repo_root, 201, head_sha, reviews_dir=reviews_dir)

    reviews_dir.mkdir(parents=True, exist_ok=True)
    current_pid = os.getpid()
    sidecar = {
        "issue_number": 201,
        "branch": "agent/issue-21-fix",
        "worktree_path": str(checkout.path),
        "prompt_path": str(checkout.path / ".orchestrator-prompt.md"),
        "command": ["claude", "-p"],
        "pid": current_pid,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "log_path": str(reviews_dir / "issue-201.claude.log"),
        "error": None,
        "process_start_time": _get_process_start_time(current_pid),
    }
    (reviews_dir / "issue-201.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {},
                "prs": {
                    "201": {
                        "number": 201,
                        "review_dispatch_status": "review_dispatch_completed",
                        "reviewer_pid": None,
                        "reviewer_process_start_time": None,
                    }
                },
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    reaped = _reap_completed_review_checkouts(repo_root, reviews_dir, state_file)

    assert reaped == []
    assert checkout.path.exists()


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


def test_review_checks_unavailable_blocks_and_preserves_labels(tmp_path: Path) -> None:
    """gh pr checks command failure must block review, leave labels unchanged, and surface checks_unavailable."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubWithChecksUnavailable(FakeGitHub):
        def pr_checks(self, number: int):
            return None

    fake_gh = FakeGitHubWithChecksUnavailable()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is False
    assert result.data["checks_unavailable"] is True
    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"


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


def test_spec_review_missing_file_propagates_os_error(tmp_path: Path) -> None:
    """A missing spec file raises naturally from read_text; the CLI boundary converts."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    with pytest.raises(OSError):
        app.spec_review(tmp_path / "nope.md")

    assert not (tmp_path / ".var" / "charlie-work" / "cross-family").exists()


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

    app.gh.prs[0]["state"] = "CLOSED"
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

    app.gh.prs[0]["state"] = "CLOSED"
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
    config = OrchestratorConfig(devin=DevinConfig(adapter="claude-code"))
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
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    assert result.ok is True
    # The critical assertion: recovery record must be passed to the adapter
    assert captured["recovery"] is not None
    assert captured["recovery"]["status"] == "dispatched"
    assert captured["recovery"]["branch_name"] == "agent/issue-123-fix-search"
    assert (123, "agent:in-progress") in fake_gh.labels_added


def test_dispatch_recovery_aborts_for_live_worker_and_restores_in_progress(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #282: a recovery redispatch that detects a live worker must abort
    and restore the in-progress label, not clobber the worktree."""

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=4242,
            started_at="2026-07-02T00:00:00Z",
            log_path=str(tmp_path / "log"),
            error="pid_alive",
            failure_kind="live_worker_redispatch_averted",
            process_start_time=1_234_567.0,
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    config = OrchestratorConfig(devin=DevinConfig(adapter="claude-code"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.pr_list = lambda: []

    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "branch_name": "agent/issue-123-fix-search",
        "worker_pid": 4242,
        "worker_process_start_time": 1_234_567.0,
        "title": "Fix search",
        "url": "https://example.test/issues/123",
    }
    save_state(paths.state_file, seed)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"
    assert (123, "agent:in-progress") in fake_gh.labels_added
    assert any(
        event["kind"] == "live_worker_redispatch_averted"
        and event["payload"]["issue_number"] == 123
        for event in state.get("events", [])
    )


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
    """Issue #376: an infrastructure check failure (CANCELLED) is not routed to rework."""
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
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


def test_janitor_required_check_repeated_failure_escalates(tmp_path: Path) -> None:
    """Issue #376: repeated check-failure reworks escalate to human_needed via the request_changes cap."""
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
    assert (123, config.labels.human_needed) in fake_gh.labels_added


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
        self, checks: list[dict[str, Any]] | None = None, *, rerun_ok: bool = True
    ) -> None:
        super().__init__(checks)
        self.rerun_ok = rerun_ok
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
                stderr="rate limit exceeded",
                value=None,
                error="rate limit exceeded",
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
    """Issue #391: the same check failing again on the same head is definitive and routes to rework."""
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

    result2 = app.review(456)
    assert result2.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["decision"] == "request_changes"
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert (123, config.labels.needs_rework) in fake_gh.labels_added
    # No additional rerun was triggered on the second pass.
    assert len(fake_gh.rerun_calls) == 1


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


def _mergequeue_automerge(label: str = "mergequeue"):
    from charlie_work.config import AutoMergeConfig

    # No required checks -> the check gate is vacuously satisfied, isolating the
    # approved-decision path for Aviator MergeQueue handoff tests (task #10).
    return AutoMergeConfig(
        required_checks=(), require_approved_review=True, mergequeue_label=label
    )


def test_merge_ready_mergequeue_mode_labels_instead_of_merging(tmp_path: Path) -> None:
    """Aviator MergeQueue handoff (task #10): when auto_merge.mergequeue_label
    is set, an approved+green PR is labeled for the queue INSTEAD of being
    self-merged, and state records a distinct 'mergequeue' status — never
    'merged', so the merge_ready idempotency short-circuit does not fire while
    Aviator's async merge is still pending."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok")

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is True
    assert (456, "mergequeue") in fake_gh.pr_labels_added
    assert fake_gh.merged == []
    assert result.data["merged"] is False
    assert result.data["mergequeue_label_applied"] is True
    persisted = load_state(paths.state_file)["prs"]["456"]
    assert persisted["status"] == "mergequeue"
    assert persisted["status"] != "merged"


def test_merge_ready_mergequeue_mode_unapproved_pr_not_labeled(tmp_path: Path) -> None:
    """An unapproved PR must never be labeled for the merge queue — the
    approval gate (can_merge) is upstream of the mergequeue branch, exactly as
    it is upstream of the self-merge branch today."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is False
    assert fake_gh.pr_labels_added == []
    assert fake_gh.merged == []
    assert result.data.get("mergequeue_label_applied") is None


def _second_mergequeue_pr(fake_gh) -> None:
    """Add a second approved-candidate issue/PR pair (124/789) to a FakeGitHub
    fixture, reviewed after the default 123/456 pair."""
    fake_gh.issues.append(
        {
            "number": 124,
            "title": "Fix parsing",
            "url": "https://example.test/issues/124",
            "body": "Parsing is broken",
            "labels": [{"name": "automated-ready"}],
            "state": "OPEN",
        }
    )
    fake_gh.prs.append(
        {
            "number": 789,
            "title": "Fix #124: parsing",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-parsing",
            "baseRefName": "main",
            "headRefOid": "sha-def789",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #124\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    )


def test_merge_ready_mergequeue_parked_pr_excluded_from_merge_train_head(
    tmp_path: Path,
) -> None:
    """Adversarial review finding #1a: once PR #456 is parked in Aviator's
    queue (state status 'mergequeue'), it must not keep winning
    front-of-train's merge-train head on every subsequent poll — Aviator now
    owns its serialization. Without excluding 'mergequeue'-status PRs from
    _merge_train_candidates, #456 (reviewed first, still approved, still
    head-SHA-matching) would win merge-train head forever and PR #789 would
    never be attempted, even though #789 is independently approved and green."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    _second_mergequeue_pr(fake_gh)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok")
    app.record_review(789, "approved", summary="ok")

    # First poll: 456 was reviewed first (and sorts first on a tie), so it
    # wins merge-train head and gets parked into the mergequeue.
    first = app.merge_ready(456, merge=True)
    assert first.data["can_merge"] is True
    assert first.data["mergequeue_label_applied"] is True
    assert (456, "mergequeue") in fake_gh.pr_labels_added

    # Second poll: 789 must now become merge-train head. Before the fix, 456
    # (still "approved" + head-SHA-matching from _merge_train_candidates'
    # point of view) keeps winning head, so 789 gets bounced as "not the
    # head of the merge-train queue" (can_merge False) forever.
    second = app.merge_ready(789, merge=True)
    assert second.data["can_merge"] is True
    assert second.data["mergequeue_label_applied"] is True
    assert (789, "mergequeue") in fake_gh.pr_labels_added


def test_merge_train_candidates_no_state_read_when_mergequeue_label_unset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Issue #421: when auto_merge.mergequeue_label is None (the default),
    _merge_train_candidates must not call load_state_locked. The mergequeue
    handoff feature is disabled, so no PR can have status 'mergequeue' and the
    state read is pure hot-path overhead that widens the StateLockBusy window.
    """
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok")

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("load_state_locked called with mergequeue_label unset")

    monkeypatch.setattr("charlie_work.workflow.load_state_locked", _fail_if_called)

    candidates = app._merge_train_candidates(prs=fake_gh.prs)

    pr_numbers = [pr_number for _sort_key, pr_number, _pr, _decision, _head in candidates]
    assert 456 in pr_numbers


def test_merge_ready_mergequeue_parked_pr_skips_charlie_branch_sync(
    tmp_path: Path,
) -> None:
    """Adversarial review finding #1b: once a PR is parked in Aviator's queue
    (state status 'mergequeue'), charlie must stop calling pr_update_branch
    for it on every subsequent poll — Aviator now owns rebasing queued PRs.
    This repo's live orchestrator.config.yaml sets update_open_prs: true
    (broadcast), so without this fix charlie would race Aviator's own rebase
    as a second writer on the same ref on every poll while the base is stale."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok")

    # First poll parks the PR.
    first = app.merge_ready(456, merge=True)
    assert first.data["mergequeue_label_applied"] is True
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "mergequeue"

    # Simulate main having advanced past this PR's merge-base — a stale base,
    # exactly as would happen once Aviator (or anything else) merges another
    # PR into main while #456 sits in the queue.
    fake_gh.compare_overrides[("main", "sha-abc123")] = {
        "base_commit": {"sha": "new-main-tip"},
        "merge_base_commit": {"sha": "stale-ancestor"},
    }

    second = app.merge_ready(456, merge=True)

    assert second.data["can_merge"] is False
    assert fake_gh.pr_update_branch_calls == []


def test_merge_ready_mergequeue_label_add_failure_does_not_advance_status(
    tmp_path: Path,
) -> None:
    """Adversarial review finding #2: add_pr_label IS the entire handoff. A
    failed label add must not be silently treated like a best-effort cleanup
    step — it must not advance state to 'mergequeue' (that would orphan the
    PR: never self-merged, never picked up by Aviator, and nothing would ever
    look wrong to state). It must instead increment
    consecutive_failed_merge_attempts (so the failure retries and can
    escalate) and surface the failure in the result message."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.add_pr_label_ok = False
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok")

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is True
    assert result.data["mergequeue_label_applied"] is False
    assert (456, "mergequeue") in fake_gh.pr_labels_added  # attempted
    assert fake_gh.merged == []
    assert "FAILED to apply" in result.message
    persisted = load_state(paths.state_file)["prs"]["456"]
    assert persisted["status"] == "approved"
    assert persisted["status"] != "mergequeue"
    assert persisted["consecutive_failed_merge_attempts"] == 1


def test_merge_ready_mergequeue_label_add_failure_alarm_fires_at_threshold(
    tmp_path: Path,
) -> None:
    """The failed-attempt alarm must be able to fire for a handoff-label
    failure too. Before this fix, can_merge is True whenever checks are green
    and the PR is approved (that is the whole point of reaching the
    mergequeue branch), so the pre-existing 'approved and not can_merge'
    alarm gate could never trigger for a persistently failing label add — a
    typo'd label or a missing repo label would retry forever with zero
    escalation."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            require_approved_review=True,
            mergequeue_label="mergequeue",
            failed_attempt_alarm=2,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.add_pr_label_ok = False
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok")

    first = app.merge_ready(456, merge=True)
    assert first.data["merge_attempt_alarm"] is False
    assert first.data["consecutive_failed_merge_attempts"] == 1

    second = app.merge_ready(456, merge=True)
    assert second.data["merge_attempt_alarm"] is True
    assert second.data["consecutive_failed_merge_attempts"] == 2
    assert "mergequeue" in (second.data["merge_attempt_warning"] or "")


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


def test_merge_ready_carries_forward_approved_verdict_on_clean_rebase(tmp_path: Path) -> None:
    """Issue #375: a clean rebase with unchanged cumulative patch-id carries the
    approved verdict forward and lets the PR merge once CI/base checks pass."""
    from charlie_work.janitor import _calculate_patch_id

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    original_diff = (
        "diff --git a/file b/file\n"
        "index 123..456 78910\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+line3\n"
        " line4\n"
    )
    patch_id = _calculate_patch_id(original_diff)
    old_head = "sha-abc123"
    new_head = "sha-rebased123"

    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": old_head,
                "reviewed_patch_id": patch_id,
                "summary": "lgtm",
            }
        ),
        encoding="utf-8",
    )

    # Simulate a rebase-style head move: the new head is not a 2-parent web-flow
    # merge commit, so _verify_synced_head would reject it. The cumulative diff
    # is unchanged, so patch-id carry-forward should keep the approval valid.
    fake_gh.pr_head_shas[456] = new_head
    fake_gh.diffs[456] = original_diff
    fake_gh.compare_overrides[("main", new_head)] = {
        "base_commit": {"sha": fake_gh.base_head_sha},
        "merge_base_commit": {"sha": fake_gh.base_head_sha},
    }
    fake_gh.commits[new_head] = {
        "parents": [{"sha": old_head}],
        "committer": {"login": "someone"},
        "commit": {"committer": {"name": "Not GitHub"}},
    }

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["can_merge"] is True
    assert result.data["merged"] is True
    assert result.data.get("head_moved") is not True
    assert fake_gh.merged == [(456, "squash")]

    decision = json.loads((decision_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert decision["reviewed_head_sha"] == new_head
    assert decision["reviewed_patch_id"] == patch_id
    # Issue #414 (d): the tier-1 fast path is unchanged and tags its own
    # carry-forwards distinctly from tier 2.
    assert decision["carry_forward_tier"] == "patch-id"
    assert decision["carried_forward_from"] == [old_head]

    state = load_state(paths.state_file)
    pr_state = state["prs"]["456"]
    assert pr_state["reviewed_head_sha"] == new_head
    assert pr_state["carry_forward_tier"] == "patch-id"
    assert pr_state["carried_forward_from"] == [old_head]
    # The approval was carried forward (not reset to "reviewing") and the PR
    # proceeded to merge on the same poll.
    assert pr_state["status"] != "reviewing"

    carry_events = [
        e for e in state["events"] if e["kind"] == "verdict_carried_forward_clean_rebase"
    ]
    assert len(carry_events) == 1
    payload = carry_events[0]["payload"]
    assert payload["pr_number"] == 456
    assert payload["issue_number"] == 123
    assert payload["old_reviewed_head_sha"] == old_head
    assert payload["new_head_sha"] == new_head
    assert payload["patch_id"] == patch_id
    assert payload["carried_forward_from"] == [old_head]

    # No review_started transition should fire for a clean rebase.
    assert (123, "agent:reviewing") not in fake_gh.labels_added


def test_merge_ready_changed_patch_id_resets_to_pending(tmp_path: Path) -> None:
    """Issue #375: if the cumulative diff changes, the approval is voided."""
    from charlie_work.janitor import _calculate_patch_id

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    original_diff = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+original\n"
    )
    changed_diff = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+changed\n"
    )
    patch_id = _calculate_patch_id(original_diff)
    old_head = "sha-abc123"
    new_head = "sha-new-head"

    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": old_head,
                "reviewed_patch_id": patch_id,
                "summary": "lgtm",
            }
        ),
        encoding="utf-8",
    )

    fake_gh.pr_head_shas[456] = new_head
    fake_gh.diffs[456] = changed_diff

    result = app.merge_ready(456)

    assert result.ok is False
    assert result.data["head_moved"] is True
    assert result.data["can_merge"] is False
    assert result.data["merged"] is False
    assert fake_gh.merged == []
    assert (123, "agent:reviewing") in fake_gh.labels_added

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "reviewing"
    assert not any(e["kind"] == "verdict_carried_forward_clean_rebase" for e in state["events"])


def test_merge_ready_missing_patch_id_falls_back_to_pending(tmp_path: Path) -> None:
    """Issue #375: an old approved decision without reviewed_patch_id falls back to
    the legacy head-SHA reset."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    old_head = "sha-abc123"
    new_head = "sha-new-head"

    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": old_head,
                "summary": "lgtm",
            }
        ),
        encoding="utf-8",
    )

    fake_gh.pr_head_shas[456] = new_head
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new\n"
    )

    result = app.merge_ready(456)

    assert result.ok is False
    assert result.data["head_moved"] is True
    assert result.data["can_merge"] is False
    assert fake_gh.merged == []


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


def test_dispatch_rework_failure_reason_in_event_payload(tmp_path: Path) -> None:
    """Issue #448: failed rework dispatch must record the per-issue reason in the event payload."""
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; sys.exit(1)",
            ),
        )
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


def test_merge_ready_failed_attempt_alarm_fires_once_at_threshold(tmp_path: Path) -> None:
    """Issue #254: after N approved-but-unmergeable passes, emit an alarm once."""
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")

    # First two failed attempts do not alarm.
    result1 = app.merge_ready(456, merge=False)
    assert result1.data["can_merge"] is False
    assert result1.data["consecutive_failed_merge_attempts"] == 1
    assert result1.data["merge_attempt_alarm"] is False
    assert result1.data["merge_attempt_warning"] is None

    result2 = app.merge_ready(456, merge=False)
    assert result2.data["consecutive_failed_merge_attempts"] == 2
    assert result2.data["merge_attempt_alarm"] is False

    # Third attempt crosses the threshold.
    result3 = app.merge_ready(456, merge=False)
    assert result3.data["consecutive_failed_merge_attempts"] == 3
    assert result3.data["merge_attempt_alarm"] is True
    warning = result3.data["merge_attempt_warning"]
    assert warning is not None
    assert "PR #456 approved but unmergeable for 3 passes" in warning
    assert "required checks missing while GitHub shows the PR open" in warning

    # Fourth attempt is still unmergeable but does not re-alarm.
    result4 = app.merge_ready(456, merge=False)
    assert result4.data["consecutive_failed_merge_attempts"] == 4
    assert result4.data["merge_attempt_alarm"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 4
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert len(alarm_events) == 1
    assert alarm_events[0]["payload"]["pr_number"] == 456
    assert alarm_events[0]["payload"]["attempts"] == 3
    assert set(alarm_events[0]["payload"]["checks_summary"].keys()) == {
        "required",
        "passed",
        "pending",
        "failed",
        "missing",
        "infra_failed",
        "unavailable",
    }


def test_merge_ready_failed_attempt_alarm_resets_on_merge(tmp_path: Path) -> None:
    """Issue #254: a successful merge resets the failed attempt counter."""
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    missing_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, missing_gh)

    app.record_review(456, "approved", summary="lgtm")
    for _ in range(3):
        app.merge_ready(456, merge=False)

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 3

    # Now the checks turn green and merge succeeds.
    passing_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, passing_gh)
    result = app.merge_ready(456, merge=True)
    assert result.data["merged"] is True

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 0
    assert state["issues"]["123"]["merge_alert"] == "OK"


def test_merge_ready_failed_attempt_alarm_resets_on_head_move(tmp_path: Path) -> None:
    """Issue #254: a head move after approval resets the failed attempt counter."""
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    app.merge_ready(456, merge=False)
    app.merge_ready(456, merge=False)

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 2

    # Simulate the PR head advancing on GitHub.
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    result = app.merge_ready(456, merge=False)
    assert result.data["head_moved"] is True

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 0
    assert state["issues"]["123"]["merge_alert"] == "OK"


def test_merge_ready_failed_attempt_alarm_resets_on_decision_change(tmp_path: Path) -> None:
    """Issue #254: a decision change resets the failed attempt counter."""
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    app.merge_ready(456, merge=False)
    app.merge_ready(456, merge=False)

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 2

    # Operator changes decision to request_changes.
    app.record_review(456, "request_changes", summary="needs work")
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 0
    assert state["issues"]["123"]["merge_alert"] == "OK"


def test_merge_ready_failed_attempt_alarm_skips_pending_only_checks(tmp_path: Path) -> None:
    """Issue #254: pending-only checks must not count toward failed merge attempts."""
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    pending_checks = [
        {"name": "Tests passed", "state": "PENDING"},
        {"name": "Lint & Format", "state": "PENDING"},
        {"name": "Pre-commit", "state": "PENDING"},
    ]
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(checks=pending_checks)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")

    for _ in range(3):
        result = app.merge_ready(456, merge=False)
        assert result.data["can_merge"] is False
        assert result.data["merge_attempt_alarm"] is False
        assert result.data["merge_attempt_warning"] is None
        assert result.data["consecutive_failed_merge_attempts"] == 0

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 0
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert len(alarm_events) == 0


def test_merge_ready_stale_base_alarm_fires_after_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #368: an operator alarm is emitted after N consecutive base_stale
    deferrals for the same PR.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # The default base_head_sha is base-sha, which is already the merge-base of
    # sha-abc123. Advance it to a post-merge tip whose merge-base with sha-abc123
    # is still base-sha, so the freshness gate sees a stale base.
    post_merge_base = "main-merged-sha-abc123"
    fake_gh.base_head_sha = post_merge_base
    fake_gh.commits[post_merge_base] = {"parents": [{"sha": "base-sha"}, {"sha": "sha-abc123"}]}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Simulate a base-sync that reports success but does not advance the head.
    monkeypatch.setattr(fake_gh, "pr_update_branch", lambda pr_number: True)

    app.record_review(456, "approved", summary="lgtm")

    for attempt in range(1, 4):
        result = app.merge_ready(456, merge=False)
        assert result.data["can_merge"] is False
        assert result.data["merged"] is False
        assert result.data.get("stale_base") is True
        assert result.data["consecutive_stale_base_deferrals"] == attempt
        if attempt < 3:
            assert result.data["merge_attempt_alarm"] is False
            assert result.data["merge_attempt_warning"] is None
        else:
            assert result.data["merge_attempt_alarm"] is True
            assert result.data["merge_attempt_warning"] is not None
            assert "base is stale" in result.data["merge_attempt_warning"]

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_stale_base_deferrals"] == 3
    stale_events = [e for e in state["events"] if e["kind"] == "merge_deferred_stale_base"]
    assert len(stale_events) == 3
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_deferred_stale_base_alarm"]
    assert len(alarm_events) == 1
    assert alarm_events[0]["payload"]["pr_number"] == 456
    assert alarm_events[0]["payload"]["reason"] == "base_stale"
    assert alarm_events[0]["payload"]["attempts"] == 3
    assert alarm_events[0]["payload"]["threshold"] == 3

    # A fourth deferral is still counted but does not re-fire the alarm.
    result = app.merge_ready(456, merge=False)
    assert result.data["consecutive_stale_base_deferrals"] == 4
    assert result.data["merge_attempt_alarm"] is False
    assert result.data["merge_attempt_warning"] is None


def test_merge_ready_merge_alert_refires_after_can_merge_recovery(tmp_path: Path) -> None:
    """Issue #254: merge=False recovery resets merge_alert so a second degradation
    can re-fire the notify digest.
    """
    from charlie_work.config import AutoMergeConfig
    from charlie_work.workflow import _build_attention_digest

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    failing_checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "state": "FAILURE"},
        {"name": "Pre-commit", "state": "FAILURE"},
    ]
    passing_checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "state": "SUCCESS"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    failing_gh = FakeGitHubWithChecks(checks=failing_checks)
    app = OrchestratorApp(tmp_path, paths, config, failing_gh)
    app.record_review(456, "approved", summary="lgtm")

    # First degradation to threshold.
    for _ in range(3):
        result = app.merge_ready(456, merge=False)
    warning = result.data["merge_attempt_warning"]
    assert warning is not None

    # Simulate the loop digest that would set merge_alert to MERGE_BLOCKED.
    _build_attention_digest(
        paths.state_file,
        {
            123: {
                "adapter_kind": "unknown",
                "health": "MERGE_BLOCKED",
                "last_log_line": None,
                "pid": None,
                "terminal_tool": None,
                "terminal_reason": warning,
            }
        },
        repo="test-repo",
        state_field="merge_alert",
    )
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["merge_alert"] == "MERGE_BLOCKED"

    # Recovery: can_merge=True but no merge attempted (merge=False).
    passing_gh = FakeGitHubWithChecks(checks=passing_checks)
    app = OrchestratorApp(tmp_path, paths, config, passing_gh)
    result = app.merge_ready(456, merge=False)
    assert result.data["can_merge"] is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["merge_alert"] == "OK"

    # Second degradation to threshold.
    app = OrchestratorApp(tmp_path, paths, config, failing_gh)
    for _ in range(3):
        result = app.merge_ready(456, merge=False)
    warning = result.data["merge_attempt_warning"]
    assert warning is not None

    # The digest should fire again because merge_alert moved OK -> MERGE_BLOCKED.
    digest = _build_attention_digest(
        paths.state_file,
        {
            123: {
                "adapter_kind": "unknown",
                "health": "MERGE_BLOCKED",
                "last_log_line": None,
                "pid": None,
                "terminal_tool": None,
                "terminal_reason": warning,
            }
        },
        repo="test-repo",
        state_field="merge_alert",
    )
    assert digest is not None
    assert len(digest.transitions) == 1
    assert digest.transitions[0].health == "MERGE_BLOCKED"
    assert digest.transitions[0].previous_health == "OK"
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["merge_alert"] == "MERGE_BLOCKED"


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

        app.gh.prs[0]["state"] = "OPEN"
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
    # Mark the default PR as closed so the issue is considered dispatchable.
    fake_gh.prs[0]["state"] = "CLOSED"
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

    app.gh.prs[0]["state"] = "CLOSED"
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
            "state": "OPEN",
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

    app.gh.prs[0]["state"] = "OPEN"
    result = app.dispatch(limit=3)

    # The issue should NOT be re-dispatched since there's an open PR
    assert result.data["attempted_count"] == 0


def test_dispatch_clears_stale_orphan_flagged_at(tmp_path: Path) -> None:
    """Issue #259 review: a fresh dispatch must clear a stale orphan flag."""
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="manual",  # Use manual to avoid actual worker launch
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Close the default PR so the issue is dispatchable.
    fake_gh.prs[0]["state"] = "CLOSED"
    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "orphan_flagged_at": "2024-01-01T00:00:00Z",
        "title": "Test issue",
        "url": "https://github.com/test/repo/issues/123",
    }
    save_state(paths.state_file, seed)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch()

    assert result.data["attempted_count"] == 1
    assert result.data["selected_count"] == 1
    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry.get("status") == "manifest_written"
    assert "orphan_flagged_at" not in entry


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

    app.gh.prs[0]["state"] = "CLOSED"
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

    app.gh.prs[0]["state"] = "CLOSED"
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


def test_record_review_pins_reviewed_head_sha_to_packet_not_live_fetch(tmp_path: Path) -> None:
    """A commit landing between review() (packet generation) and record_review()
    (verdict recording) must not reattribute the approval to a head/diff that
    was never reviewed: reviewed_head_sha and reviewed_patch_id must come from
    the packet the reviewer actually read, not a fresh fetch at verdict time.
    """
    from charlie_work.janitor import _calculate_patch_id

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = "diff --git a/file b/file\n+packet diff"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    review_result = app.review(456)
    assert review_result.ok is True
    packet_patch_id = _calculate_patch_id(fake_gh.diffs[456])

    # Simulate a new commit landing after the packet was generated but before
    # the verdict is recorded.
    fake_gh.pr_head_shas[456] = "sha-new789"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+unreviewed change"

    result = app.record_review(456, "approved", summary="lgtm")

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["reviewed_head_sha"] == "sha-abc123"
    assert decision["reviewed_patch_id"] == packet_patch_id
    assert load_state(paths.state_file)["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"
    assert result.data["reviewed_head_sha"] == "sha-abc123"


def test_record_review_blocked_persists_reviewed_patch_id(tmp_path: Path) -> None:
    """Issue #413: blocked decisions must persist reviewed_patch_id so the
    review-queue enumerator can carry them forward on content-identical heads."""
    from charlie_work.janitor import _calculate_patch_id

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 blocked\n"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    review_result = app.review(456)
    assert review_result.ok is True
    packet_patch_id = _calculate_patch_id(fake_gh.diffs[456])

    result = app.record_review(456, "blocked", summary="security concern")

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "blocked"
    assert decision["reviewed_patch_id"] == packet_patch_id
    assert load_state(paths.state_file)["prs"]["456"]["reviewed_patch_id"] == packet_patch_id
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


def test_record_review_persists_escalated_in_decision_file(tmp_path: Path) -> None:
    """Issue #407: review-decision.json must include the correct escalated value.

    The decision payload is fully built before the single atomic write, so
    re-reading the persisted file returns the same escalated flag as the
    in-memory result. Non-escalated request_changes and escalated
    request_changes must both persist the correct value.
    """
    config = OrchestratorConfig(review=ReviewConfig(max_rework_cycles=2))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    decision_path = paths.prs / "pr-456" / "review-decision.json"

    fake_gh.pr_head_shas[456] = "sha-1"
    app.record_review(456, "request_changes", summary="fix A")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert "escalated" in decision
    assert decision["escalated"] is False
    assert app._review_decision(456)["escalated"] is False

    fake_gh.pr_head_shas[456] = "sha-2"
    app.record_review(456, "request_changes", summary="fix B")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["escalated"] is False

    fake_gh.pr_head_shas[456] = "sha-3"
    app.record_review(456, "request_changes", summary="fix C")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["escalated"] is True
    # _review_decision is the reader used by merge_ready and merge-train
    # eligibility; it must see the persisted escalated value.
    assert app._review_decision(456)["escalated"] is True


def test_merge_ready_reads_escalated_from_persisted_decision(tmp_path: Path) -> None:
    """Issue #407: merge_ready (via _review_decision and merge-train
    eligibility) must see the escalated flag from the persisted decision file.
    """
    config = OrchestratorConfig(review=ReviewConfig(max_rework_cycles=1))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # First request_changes is not escalated (count 0 -> 1).
    fake_gh.pr_head_shas[456] = "sha-1"
    app.record_review(456, "request_changes", summary="fix A")
    # Second request_changes hits the max_rework_cycles cap and escalates.
    fake_gh.pr_head_shas[456] = "sha-2"
    app.record_review(456, "request_changes", summary="fix B")

    result = app.merge_ready(456)
    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["review_decision"]["decision"] == "request_changes"
    assert result.data["review_decision"]["escalated"] is True


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
    app.gh.prs[0]["state"] = "CLOSED"
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
    app.gh.prs[0]["state"] = "CLOSED"
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
    app.gh.prs[0]["state"] = "CLOSED"
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
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
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
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
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

    result = app.loop(limit=0)

    # The launch-failure sidecar is reaped and reported
    reaped = result.data.get("reaped", [])
    matching = [entry for entry in reaped if entry["issue_number"] == 42]
    assert len(matching) == 1
    assert not sidecar_path.exists()

    # The throttle window from the classifier is persisted, same as the
    # dead-session lane
    state = load_state(paths.state_file)
    assert state.get("throttled_until") is not None
    throttle_time = datetime.fromisoformat(state["throttled_until"].replace("Z", "+00:00"))
    expected_time = datetime.now(UTC) + timedelta(minutes=10)
    assert abs((throttle_time - expected_time).total_seconds()) < 5


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
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
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
    app.gh.prs[0]["state"] = "CLOSED"
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


def test_classify_dead_rework_session_returns_to_rework_requested(
    tmp_path: Path,
) -> None:
    """Issue #295: a dead/launch-failed rework session with an open PR and a
    LIVE request_changes verdict (still matching the PR's current head) must
    be restored to rework_requested so the next dispatch_rework can re-select
    it.

    Issue #315 review rework: a bare rework-prompt.md on disk is no longer
    sufficient by itself (see the stale-prompt regression test below) — the
    prompt file is never deleted, so has_request_changes is now the single
    signal that gates the restore. This test records a live request_changes
    decision (matching production: _write_rework_prompt and the
    decision/reviewed_head_sha state write happen in the same review() call)
    so it keeps exercising the restore path under the corrected semantics.
    It also exercises finding 2's window-filtered redispatch_at bookkeeping
    (previously this lane preserved redispatch_at unchanged and never grew
    it, which is exactly why the escalation cap could never trip).

    Mutation gate: dropping the request_changes check or the rework_requested
    status rollback from _reap_restore_rework_requested fails this test.
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

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
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    # Issue is stuck in the dispatched state with the rework worker label.
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]

    # PR state records a LIVE request_changes decision matching the PR's
    # current head (fake_gh.prs[0]["headRefOid"] == "sha-abc123" by default) —
    # the only signal _reap_restore_rework_requested now honors (issue #315
    # finding 1). The on-disk rework-prompt.md below is still written (it's
    # what a real request_changes cycle produces) but is a diagnostic
    # supplement only, not required for the restore.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
            "prompt_path": str(paths.prs / "pr-456" / "rework-prompt.md"),
            "redispatch_at": ["2020-01-01T00:00:00Z"],
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    # Create the rework prompt on disk (the rework brief).
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    # Create a sessions directory with a launch-failure sidecar (rate-limit signature).
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text(
        "Reached overall message rate limit. Your limit will reset in 0 minutes.\n",
        encoding="utf-8",
    )

    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(rework_prompt),
        command=("devin", "--prompt-file", str(rework_prompt)),
        pid=None,  # launch-failure sidecar
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="devin launch failed: rate limit",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Run the reap pass.
    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    # Verify state was restored to rework_requested for the owning lane.
    state = load_state(paths.state_file)
    entry = state["issues"].get("123")
    assert entry is not None
    assert entry.get("status") == "rework_requested"
    assert entry.get("dispatched_at") is None
    # Issue #315 finding 2: the old 2020 entry is outside the redispatch
    # window (default 240 minutes) and is dropped; a fresh entry is appended
    # in its place — proof the cap bookkeeping this lane previously skipped
    # now actually runs, while staying under config.watchdog.max_auto_redispatch
    # (default 3) so the restore (not escalation) path is taken.
    redispatch_at = entry.get("redispatch_at")
    assert redispatch_at is not None
    assert len(redispatch_at) == 1
    assert redispatch_at[0] != "2020-01-01T00:00:00Z"
    # Liveness fingerprint preserved for recovery path (issue #282)
    assert entry.get("worker_pid") == 99999
    assert entry.get("worker_process_start_time") == 1234567890.0
    # Label transitioned from in_progress to needs_rework
    assert (123, config.labels.in_progress) in fake_gh.labels_removed
    assert (123, config.labels.needs_rework) in fake_gh.labels_added

    # Next dispatch_rework should re-select the issue.
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch_rework()
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["sessions"][0]["issue_number"] == 123
    assert str(result.data["sessions"][0]["prompt_path"]).endswith("rework-prompt.md")


def test_classify_dead_rework_session_stale_prompt_does_not_reopen_approved_head(
    tmp_path: Path,
) -> None:
    """Issue #315 review finding 1: a stale rework-prompt.md left over from an
    earlier cycle must NOT roll a PR whose CURRENT head is already approved
    back to rework_requested. The prompt file is written once per PR
    (workflow._write_rework_prompt) and is never deleted, so its mere
    existence cannot distinguish "still awaiting this cycle's rework" from
    "leftover from a cycle that has since been approved" the way
    has_request_changes can (it re-derives from the PR's live review record
    on every call).

    Mutation gate: reverting _reap_restore_rework_requested's gate from
    `if not has_request_changes: return` back to
    `if not has_request_changes and not has_rework_prompt: return` makes this
    test fail — the stale prompt alone would incorrectly trigger the restore.
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

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
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
    # The PR's CURRENT head is approved (a fresh review cycle already ran and
    # passed) -- not request_changes.
    fake_gh.prs[0]["headRefOid"] = "sha-approved-head"

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
            "redispatch_at": [],
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "approved",
            "reviewed_head_sha": "sha-approved-head",
        }
        save_state(paths.state_file, state)

    # Stale rework-prompt.md left over from an EARLIER cycle (never deleted).
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the old issues", encoding="utf-8")

    # Dead worker that exited normally (no launch error). The worktree is
    # never created (is_completed=False), isolating this test to finding 1's
    # has_request_changes fix rather than finding 1's is_completed guard
    # (covered by test_classify_dead_rework_session_completed_worktree_not_rolled_back).
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),  # never created
        prompt_path=str(rework_prompt),
        command=("devin", "--prompt-file", str(rework_prompt)),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(sessions_dir / "issue-123.log"),
        error=None,  # exited normally
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    # Must NOT be rolled back -- the approved head is live, the prompt is stale.
    assert entry["status"] != "rework_requested"
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


def test_classify_dead_rework_session_escalates_at_redispatch_cap(
    tmp_path: Path,
) -> None:
    """Issue #315 review finding 2a: a dead rework worker must be escalated
    (not restored to rework_requested) once its redispatch_at history hits
    config.watchdog.max_auto_redispatch, matching the cap semantics the
    sibling lanes already enforce (workflow.py's no-open-PR dead-session lane
    and OrchestratorApp.dispatch_rework's success path both escalate when
    `len(redispatch_at) > max_auto_redispatch`).

    Mutation gate: dropping the
    `len(redispatch_at) > config.watchdog.max_auto_redispatch` half of
    _reap_restore_rework_requested's `should_escalate` check makes this test
    fail (the issue would be restored to rework_requested indefinitely
    instead of escalating).
    """
    import json
    from datetime import UTC, datetime, timedelta

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

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
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
    # fake_gh.prs[0]["headRefOid"] defaults to "sha-abc123".

    now = datetime.now(UTC)
    # Three recent redispatches, all inside the default 240-minute window --
    # max_auto_redispatch defaults to 3, so a fourth entry trips the cap.
    recent_redispatches = [
        (now - timedelta(minutes=m)).isoformat().replace("+00:00", "Z") for m in (6, 4, 2)
    ]

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
            "redispatch_at": recent_redispatches,
        }
        # Live request_changes decision matching the current head, so this
        # test isolates the cap check rather than finding 1's gate.
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    # Launch-failure sidecar with a non-deterministic failure signature (rate
    # limit) -- isolates the cap check from finding 2b's deterministic-kind
    # guard (covered by the worktree_unsafe test below).
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text(
        "Reached overall message rate limit. Your limit will reset in 0 minutes.\n",
        encoding="utf-8",
    )
    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(paths.prs / "pr-456" / "rework-prompt.md"),
        command=("devin", "--prompt-file", "rework-prompt.md"),
        pid=None,  # launch-failure sidecar
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="devin launch failed: rate limit",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "redispatch_cap_exceeded"
    assert len(entry["redispatch_at"]) == 4
    assert (123, config.labels.human_needed) in fake_gh.labels_added
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 123]
    assert "session_failed_escalated" in event_kinds
    assert "rework_requeued" not in event_kinds


def test_classify_dead_rework_session_deterministic_failure_kind_escalates_immediately(
    tmp_path: Path,
) -> None:
    """Issue #315 review finding 2b: a dead rework worker whose failure_kind is
    confirmed-deterministic (config.DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    e.g. worktree_unsafe) must escalate immediately, bypassing the redispatch
    cap entirely -- identical to the no-open-PR lane's immediate-escalation
    block (workflow.py ~line 936). That block is gated on
    `w.issue_number not in open_prs_by_issue`, so a rework worker (which
    always has an open PR) bypasses it entirely and would otherwise fall
    through to the ordinary cap-based path in _reap_restore_rework_requested.

    Mutation gate: dropping the `terminal_failure or` half of
    _reap_restore_rework_requested's `should_escalate` check makes this test
    fail (the issue would be restored to rework_requested since the
    redispatch history is empty and well under the cap).
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
    # fake_gh.prs[0]["headRefOid"] defaults to "sha-abc123".

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
            "redispatch_at": [],  # nowhere near the cap
        }
        # Live request_changes decision matching the current head, so this
        # test isolates the deterministic-kind guard rather than finding 1's
        # gate.
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text("worktree contains local work, cannot reset\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(paths.prs / "pr-456" / "rework-prompt.md"),
        command=("devin", "--prompt-file", "rework-prompt.md"),
        pid=None,  # Launch failure -- process never started
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="worktree creation failed: worktree contains local work",
        failure_kind="worktree_unsafe",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worktree_unsafe"
    assert (123, config.labels.human_needed) in fake_gh.labels_added
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 123]
    assert "session_failed_escalated" in event_kinds
    assert "rework_requeued" not in event_kinds


def test_classify_dead_rework_session_completed_worktree_not_rolled_back(
    tmp_path: Path,
) -> None:
    """LOW (issue #315 review): a rework worker that actually finished its
    work (worktree ahead of base and clean -- is_completed=True) must never be
    rolled back to rework_requested, even if this reap pass's PR-list
    snapshot (fetched once, at the top of
    _classify_dead_sessions_and_update_throttle_state) hasn't caught up to a
    fresh push yet and still shows the pre-rework head that matches a live
    request_changes decision. has_request_changes alone cannot catch this --
    it would look identical to a genuine, never-pushed rework -- so the
    open-PR branch must ALSO consult is_completed (issue #315 finding 1's
    second half).

    Mutation gate: removing the `if not is_completed:` guard around the
    _reap_restore_rework_requested call in the open-PR dead-session branch
    makes this test fail (the stale PR-list snapshot would incorrectly
    trigger a rollback to rework_requested).
    """
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 315)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 315, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 315,
            "title": "Test issue",
            "url": "https://example.test/issues/315",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    # Open PR whose PR-list snapshot still shows the STALE pre-rework head --
    # simulating the race where the worker's fresh push hasn't been reflected
    # in the PR-list fetched at the top of this reap pass yet.
    gh.prs = [
        {
            "number": 900,
            "title": "Fix #315",
            "headRefName": branch,
            "headRefOid": "sha-stale-snapshot",
            "isCrossRepository": False,
            "body": "Closes #315",
            "labels": [],
            "state": "OPEN",
        }
    ]

    with state_lock(state_file):
        state = load_state(state_file)
        state["issues"]["315"] = {
            "number": 315,
            "status": "dispatched",
            "branch_name": branch,
            "worker_pid": 12345,
            "worker_process_start_time": 1111111111.0,
        }
        # This decision is still "live" against the STALE snapshot head --
        # exactly what would make has_request_changes incorrectly True if
        # is_completed weren't consulted.
        state["prs"]["900"] = {
            "number": 900,
            "issue_number": 315,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-stale-snapshot",
        }
        save_state(state_file, state)

    _classify_dead_sessions_and_update_throttle_state(sessions_dir, state_file, gh, config)

    state = load_state(state_file)
    entry = state["issues"]["315"]
    assert entry["status"] != "rework_requested"
    assert (315, config.labels.needs_rework) not in gh.labels_added


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
    now = datetime.now(UTC)
    worktree_path = str(tmp_path / "worktree")

    db_path = tmp_path / "sessions.db"
    make_sessions_db(
        db_path,
        session_id="sess-1",
        working_directory=worktree_path,
        created_at=now.isoformat(),
        rows=[
            {
                "role": "tool",
                "content": (
                    'Tool blocked: {"decision": "block", "reason": "push-gate hook rejected"}'
                ),
                "created_at": now.isoformat(),
            }
        ],
    )

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
        failure_kind="worktree_unsafe",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    # No hot relabel-to-ready.
    assert (42, config.labels.ready) not in fake_gh.labels_added
    # Escalation transition added human_needed.
    assert (42, config.labels.human_needed) in fake_gh.labels_added
    # The launch never succeeded, so the issue should not be marked in_progress.
    assert (42, config.labels.in_progress) not in fake_gh.labels_added

    state = load_state(paths.state_file)
    issue_entry = state["issues"]["42"]
    assert issue_entry["status"] == "escalated"
    assert issue_entry["escalation_reason"] == "worktree_unsafe"

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 42]
    assert "session_failed_relabeled" not in event_kinds
    assert "session_failed_escalated" in event_kinds

    fake_gh.issues[0]["labels"].append({"name": config.labels.human_needed})
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
        sessions_dir, paths.state_file, fake_gh, config
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


@pytest.mark.real_activity_probe_live
def test_classify_dead_sessions_retains_sidecar_on_inconclusive_probe(
    tmp_path: Path,
) -> None:
    """Issue #343: a not-alive-looking pid with an inconclusive real-activity
    probe must have its sidecar RETAINED (deferred), not reaped, on this pass.

    Before this fix, ``_classify_dead_sessions_and_update_throttle_state``
    treated ``not w.is_alive()`` as sufficient grounds to relabel the issue
    and delete the sidecar unconditionally -- bypassing the same
    corroboration + inconclusive-probe deferral cap that
    ``classify_worker_health`` already enforces for the sibling stall/kill
    lane. That let a fail-open reap remove the sidecar of a worker whose
    liveness signal was merely ambiguous, leaving the underlying process
    invisible to the concurrency governor (issue #343's concrete production
    instance: pid 23440 verified alive via ``Get-Process``, yet its sidecar
    was removed after a ``matched: false`` post-mortem).

    Marked ``real_activity_probe_live`` so the autouse
    ``_stub_real_activity_probe_for_stalled_tests`` fixture (issue #307)
    leaves ``real_activity_probe_for`` unstubbed -- that stub always returns
    a 30-minute-stale-but-non-erroring probe, which is never "inconclusive"
    (``_real_activity_is_inconclusive`` requires every source to error), so
    it would defeat this test's premise. With the real probe, pointing
    ``post_mortem.db_path`` at a nonexistent path makes every source error
    deterministically regardless of what happens to be on the test host.

    MUTATION GATE: reverting the ``if health is not WorkerHealth.DEAD:
    continue`` gate in ``_classify_dead_sessions_and_update_throttle_state``
    (src/charlie_work/workflow.py) makes this test fail -- the sidecar would
    be reaped and the issue relabeled on this single pass.
    """
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="manual"),
        # Point post-mortem's sessions.db at a path that can never exist, so
        # the real-activity probe is deterministically inconclusive (every
        # source errors) regardless of what happens to be on the test host.
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "no-such-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 343,
            "title": "Ghost sidecar issue",
            "url": "https://example.test/issues/343",
            "body": "x",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    fake_gh.prs = []

    sessions_dir = paths.root / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-343.log"
    log_path.write_text("Working...\n", encoding="utf-8")

    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path

    sidecar_path = devin_sidecar_path(sessions_dir, 343)
    record = SessionRecord(
        issue_number=343,
        branch="agent/issue-343-x",
        worktree_path=str(tmp_path / "worktree-343"),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=54321,  # A pid our liveness check will report as gone (mocked below)
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        process_start_time=1_700_000_000.0,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    with patch("charlie_work.worker.is_session_alive", return_value=False):
        _classify_dead_sessions_and_update_throttle_state(
            sessions_dir, paths.state_file, fake_gh, config
        )

    assert sidecar_path.exists(), "sidecar must be RETAINED when the probe is inconclusive"
    assert (343, config.labels.in_progress) not in fake_gh.labels_removed
    assert (343, config.labels.ready) not in fake_gh.labels_added

    # The Signal-1 deferral counter must have advanced so the escalation cap
    # (max_inconclusive_probe_deferrals) is still reachable over later passes.
    persisted = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert persisted.get("inconclusive_probe_deferred_count") == 1


def test_classify_dead_sessions_reaps_sidecar_when_probe_conclusively_stale(
    tmp_path: Path,
) -> None:
    """Regression pin: a genuinely dead pid whose corroboration probe is
    conclusively stale (not fresh, not inconclusive) is still reaped
    immediately -- the issue #343 fix must not invert into "never reap".
    """
    from datetime import timedelta

    from charlie_work.devin_shell import SessionRecord
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(devin=DevinConfig(adapter="manual"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 344,
            "title": "Genuinely dead worker",
            "url": "https://example.test/issues/344",
            "body": "x",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    fake_gh.prs = []

    sessions_dir = paths.root / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-344.log"
    log_path.write_text("Working...\n", encoding="utf-8")

    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path

    sidecar_path = devin_sidecar_path(sessions_dir, 344)
    record = SessionRecord(
        issue_number=344,
        branch="agent/issue-344-x",
        worktree_path=str(tmp_path / "worktree-344"),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=54322,
        started_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        process_start_time=1_700_000_000.0,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.post_mortem import ActivitySource, RealActivityProbe

    stale_probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=datetime.now(UTC) - timedelta(hours=2),
                staleness_seconds=7200.0,
                error=None,
            ),
        )
    )

    with (
        patch("charlie_work.worker.is_session_alive", return_value=False),
        patch("charlie_work.worker.real_activity_probe_for", return_value=stale_probe),
    ):
        _classify_dead_sessions_and_update_throttle_state(
            sessions_dir, paths.state_file, fake_gh, config
        )

    assert not sidecar_path.exists(), "a conclusively-stale probe must still allow reaping"
    assert (344, config.labels.in_progress) in fake_gh.labels_removed
    assert (344, config.labels.ready) in fake_gh.labels_added


@pytest.mark.real_activity_probe_live
def test_stall_and_dead_lane_increment_deferral_counter_at_most_once_per_pass(
    tmp_path: Path,
) -> None:
    """Issue #343 Finding 2: the stall lane (_detect_and_handle_stalled_sessions)
    and the dead lane (_classify_dead_sessions_and_update_throttle_state) both
    corroborate a not-alive, pid-bearing, error-free worker against the same
    real-activity probe, and both used to unconditionally persist Signal-1's
    inconclusive-probe deferral counter. Within a single ``loop()`` pass that
    double-incremented the counter (0->1 in the stall lane, then re-read and
    ->2 in the dead lane) -- halving the effective deferral grace period, and
    the very mechanism that opens Finding 1's pass-2 phantom-sidecar window.

    ``loop()`` (workflow.py, ~4100/~4119) always runs the stall lane
    immediately before the dead lane and passes the dead lane
    ``persist_inconclusive_probe_counter=False`` for exactly this reason --
    this test drives both lanes once in that same order, with that same
    argument, and pins the counter at exactly 1 after the pass. Every other
    caller (every existing standalone unit test, plus dispatch()/
    dispatch_rework(), which never call the dead lane at all) leaves the
    dead lane's default (True) alone, so the stall lane remains the correct
    sole writer when the dead lane doesn't run in the same pass -- see
    ``test_detect_and_handle_stalled_sessions_inconclusive_probe_deferred_
    then_escalated`` in test_worker.py, which pins that standalone case.

    MUTATION GATE: dropping the ``persist_inconclusive_probe_counter=False``
    argument from this call (i.e. reverting to the unconditional write) makes
    this test fail -- the counter would read 2, not 1.
    """
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe
    from charlie_work.workflow import (
        _classify_dead_sessions_and_update_throttle_state,
        _detect_and_handle_stalled_sessions,
    )

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="manual"),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "no-such-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 343,
            "title": "Double-increment guard",
            "url": "https://example.test/issues/343",
            "body": "x",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    fake_gh.prs = []

    sessions_dir = paths.root / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-343.log"
    log_path.write_text("Working...\n", encoding="utf-8")

    sidecar_path = devin_sidecar_path(sessions_dir, 343)
    record = SessionRecord(
        issue_number=343,
        branch="agent/issue-343-x",
        worktree_path=str(tmp_path / "worktree-343"),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=54321,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        process_start_time=1_700_000_000.0,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    inconclusive_probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error="sessions.db unavailable",
            ),
        )
    )

    with (
        patch("charlie_work.worker.is_session_alive", return_value=False),
        patch("charlie_work.worker.real_activity_probe_for", return_value=inconclusive_probe),
    ):
        # loop() order and arguments: stall lane runs before the dead lane,
        # which is told not to persist the counter itself this pass.
        _detect_and_handle_stalled_sessions(sessions_dir, paths.state_file, config)
        _classify_dead_sessions_and_update_throttle_state(
            sessions_dir,
            paths.state_file,
            fake_gh,
            config,
            persist_inconclusive_probe_counter=False,
        )

    assert sidecar_path.exists(), "sidecar must be RETAINED when the probe is inconclusive"
    persisted = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert persisted.get("inconclusive_probe_deferred_count") == 1, (
        "counter must increment at most once per worker per pass, not twice"
    )


@pytest.mark.real_activity_probe_live
def test_stall_then_dead_lane_composition_survives_phantom_post_mortem_sidecar(
    tmp_path: Path,
) -> None:
    """Issue #343 Finding 1 (composition gap): every pre-existing test drives
    ``_classify_dead_sessions_and_update_throttle_state`` in isolation. In
    production, ``loop()`` always runs the stall lane
    (``_detect_and_handle_stalled_sessions``) immediately before the dead
    lane, and the stall lane can itself reach a DEAD verdict and write a
    ``issue-N.post-mortem.json`` sidecar (via ``classify_and_record``)
    without reaping the session sidecar (it never reaps -- only the dead lane
    does). Before the ``read_session_records`` fix, that leftover post-mortem
    file was misread by the devin glob (``issue-*.json``) as a bogus phantom
    ``SessionRecord(pid=None, log_path="")``. The phantom's ``pid is None``
    skips corroboration entirely and reaches ``reap_sidecar``, which resolves
    to the SAME path as the REAL ``issue-343.json`` sidecar and deletes it --
    even when the real worker's own corroboration correctly defers it as not
    (yet) provably dead.

    Setup: forces the stall lane to reach a DEAD verdict via an unconditional
    terminal-error-marker log line (Signal 2, which "bypasses corroboration
    and still returns DEAD immediately" per classify_worker_health's
    docstring) so it writes the post-mortem sidecar without needing to
    fabricate a diverging probe. The marker line is then removed so the two
    "real" passes that follow are driven by one ordinary inconclusive probe
    throughout, exactly like the simpler double-increment test above.

    Drives the stall lane then the dead lane, in ``loop()``'s own order and
    with its own ``persist_inconclusive_probe_counter=False`` argument,
    across two passes with the stale post-mortem sidecar from setup still on
    disk throughout. Asserts the REAL sidecar survives both passes (still
    deferred) and the deferral counter advances by exactly 1 per pass
    (0 -> 1 -> 2), pinning both Finding 1 (the phantom must never be read
    back as a session) and Finding 2 (at most one increment per worker per
    pass) together.

    MUTATION GATE: reverting either the ``read_session_records`` stem
    exclusion (Finding 1) or dropping
    ``persist_inconclusive_probe_counter=False`` from the dead lane call
    (Finding 2) makes this test fail -- the sidecar is deleted mid-pass, or
    the counter overshoots to 2/4 instead of 1/2.
    """
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe
    from charlie_work.workflow import (
        _classify_dead_sessions_and_update_throttle_state,
        _detect_and_handle_stalled_sessions,
    )

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="manual"),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "no-such-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 343,
            "title": "Phantom post-mortem sidecar",
            "url": "https://example.test/issues/343",
            "body": "x",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    fake_gh.prs = []

    sessions_dir = paths.root / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-343.log"

    sidecar_path = devin_sidecar_path(sessions_dir, 343)
    record = SessionRecord(
        issue_number=343,
        branch="agent/issue-343-x",
        worktree_path=str(tmp_path / "worktree-343"),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=54321,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        process_start_time=1_700_000_000.0,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    inconclusive_probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error="sessions.db unavailable",
            ),
        )
    )

    def _run_stall_lane() -> None:
        with (
            patch("charlie_work.worker.is_session_alive", return_value=False),
            patch(
                "charlie_work.worker.real_activity_probe_for",
                return_value=inconclusive_probe,
            ),
        ):
            _detect_and_handle_stalled_sessions(sessions_dir, paths.state_file, config)

    def _run_dead_lane() -> None:
        with (
            patch("charlie_work.worker.is_session_alive", return_value=False),
            patch(
                "charlie_work.worker.real_activity_probe_for",
                return_value=inconclusive_probe,
            ),
        ):
            _classify_dead_sessions_and_update_throttle_state(
                sessions_dir,
                paths.state_file,
                fake_gh,
                config,
                persist_inconclusive_probe_counter=False,
            )

    def _run_pass() -> None:
        # loop() order and arguments: stall lane runs before the dead lane,
        # which is told not to persist the counter itself.
        _run_stall_lane()
        _run_dead_lane()

    # Setup (not one of the two counted passes): a terminal-error-marker log
    # line makes the stall lane's classify_worker_health call return DEAD
    # unconditionally (Signal 2 bypasses corroboration), so it writes the
    # post-mortem sidecar without reaping (the stall lane never reaps). The
    # marker is cleared BEFORE the dead lane runs -- matching loop()'s own
    # sequential order, where nothing else touches the log between the two
    # calls -- so the dead lane's own corroboration this pass is driven by
    # the inconclusive probe alone; otherwise Signal 2 would ALSO fire there
    # and reap the sidecar during setup.
    log_path.write_text("Error: Agent error: fatal\n", encoding="utf-8")
    _run_stall_lane()
    log_path.write_text("Working...\n", encoding="utf-8")
    _run_dead_lane()

    post_mortem_path = sessions_dir / "issue-343.post-mortem.json"
    assert post_mortem_path.exists(), "setup precondition: stall lane must have written it"
    assert sidecar_path.exists(), "setup must not reap -- dead lane saw a clean log, deferred"

    # Pass 1: ordinary inconclusive probe: both lanes defer.
    _run_pass()

    assert sidecar_path.exists(), "real sidecar must survive pass 1 (deferred, not DEAD)"
    persisted = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert persisted.get("inconclusive_probe_deferred_count") == 1

    # Pass 2: inconclusive again; the stale post-mortem sidecar from setup
    # is still on disk.
    _run_pass()

    assert post_mortem_path.exists(), "post-mortem sidecars are never reaped by this code path"
    assert sidecar_path.exists(), "real sidecar must survive pass 2 (deferred, not DEAD)"
    persisted = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert persisted.get("inconclusive_probe_deferred_count") == 2, (
        "counter must advance by exactly 1 per pass across both passes"
    )


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


def test_merge_ready_stale_base_deferred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #316: a PR whose merge-base is not the current base tip is deferred."""
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
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    # Create review decision files for both PRs (approved state)
    for pr_number, head_sha in [(456, "sha-abc123"), (789, "sha-def456")]:
        decision_dir = paths.prs / f"pr-{pr_number}"
        decision_dir.mkdir(parents=True, exist_ok=True)
        (decision_dir / "review-decision.json").write_text(
            json.dumps(
                {"decision": "approved", "reviewed_head_sha": head_sha},
                indent=2,
            ),
            encoding="utf-8",
        )

    # Merging PR 456 advances the fake base tip. PR 456 is then up-to-date with
    # the new base tip, but PR 789's merge-base is still the old base tip, so
    # the merge-base freshness gate defers it organically.

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Simulate a base-sync that reports success but does not advance the head, so
    # the merge-base freshness gate still defers the PR after the first PR merges.
    monkeypatch.setattr(fake_gh, "pr_update_branch", lambda pr_number: True)

    result_456 = app.merge_ready(456, merge=True)
    assert result_456.ok is True
    assert result_456.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]

    result_789 = app.merge_ready(789, merge=True, merge_train_head=789)
    assert result_789.ok is True
    assert result_789.data["can_merge"] is False
    assert result_789.data["merged"] is False
    assert result_789.data.get("stale_base") is True
    assert fake_gh.merged == [(456, "squash")]

    state = json.loads(paths.state_file.read_text())
    stale_events = [
        event for event in state["events"] if event["kind"] == "merge_deferred_stale_base"
    ]
    assert len(stale_events) == 1
    assert stale_events[0]["payload"]["pr_number"] == 789
    assert stale_events[0]["payload"]["reason"] == "base_stale"


def test_merge_ready_require_current_base_false_allows_stale_base(tmp_path: Path) -> None:
    """Operators may opt out of the merge-base freshness gate."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="off",
            require_current_base=False,
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
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    for pr_number, head_sha in [(456, "sha-abc123"), (789, "sha-def456")]:
        decision_dir = paths.prs / f"pr-{pr_number}"
        decision_dir.mkdir(parents=True, exist_ok=True)
        (decision_dir / "review-decision.json").write_text(
            json.dumps(
                {"decision": "approved", "reviewed_head_sha": head_sha},
                indent=2,
            ),
            encoding="utf-8",
        )

    # With require_current_base=False the gate is disabled, so the second PR
    # ships even though merge_pr(456) has advanced the fake base tip.

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result_456 = app.merge_ready(456, merge=True)
    assert result_456.data["merged"] is True

    result_789 = app.merge_ready(789, merge=True)
    assert result_789.data["can_merge"] is True
    assert result_789.data["merged"] is True
    assert result_789.data.get("stale_base") is not True
    assert fake_gh.merged == [(456, "squash"), (789, "squash")]

    state = json.loads(paths.state_file.read_text())
    assert not any(e["kind"] == "merge_deferred_stale_base" for e in state["events"])


def test_merge_ready_next_mode_syncs_head_before_merge(tmp_path: Path) -> None:
    """Merge-train head with a BEHIND mergeStateStatus is base-synced before merge."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["mergeStateStatus"] = "BEHIND"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    # The branch was synced before the merge
    assert fake_gh.prs[0]["headRefOid"] == "sha-abc123-updated"
    # The approved head was updated to match the new base
    decision = json.loads((paths.prs / "pr-456" / "review-decision.json").read_text())
    assert decision["reviewed_head_sha"] == "sha-abc123-updated"


def test_merge_ready_next_mode_skips_non_head(tmp_path: Path) -> None:
    """In merge-train mode, a non-head approved PR cannot be merged."""
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
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    app.record_review(789, "approved", summary="lgtm")
    # Ensure 456 is the head of the queue regardless of when approvals occurred.
    for idx, pr_number in enumerate((456, 789)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    result = app.merge_ready(789, merge=True)

    assert result.ok is True
    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert "not the head" in result.message
    assert fake_gh.merged == []


def test_merge_ready_clean_stale_base_syncs_and_merges(tmp_path: Path) -> None:
    """Issue #334: approved PR with mergeStateStatus CLEAN but stale merge-base is synced and merged."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # mergeStateStatus is CLEAN, but the compare API says the merge-base is stale.
    fake_gh.compare_overrides[("main", "sha-abc123")] = {
        "base_commit": {"sha": "base-sha"},
        "merge_base_commit": {"sha": "base-sha-old"},
    }
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    # The branch was synced despite mergeStateStatus CLEAN because the base was stale.
    assert fake_gh.prs[0]["headRefOid"] == "sha-abc123-updated"
    # The approved head was updated to match the new base.
    decision = json.loads((paths.prs / "pr-456" / "review-decision.json").read_text())
    assert decision["reviewed_head_sha"] == "sha-abc123-updated"


def test_merge_ready_current_base_no_sync(tmp_path: Path) -> None:
    """Issue #334 negative control: an already-current approved PR should not be synced."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # mergeStateStatus CLEAN and the compare API agrees the base is current.
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    # No update-branch should have been attempted.
    assert fake_gh.prs[0]["headRefOid"] == "sha-abc123"
    decision = json.loads((paths.prs / "pr-456" / "review-decision.json").read_text())
    assert decision["reviewed_head_sha"] == "sha-abc123"


def test_merge_ready_merge_conflict_routes_to_rework(tmp_path: Path) -> None:
    """Issue #371: an approved PR with a genuine merge conflict is routed to rework.

    A conflict is detected from ``mergeable=CONFLICTING`` and is not retried
    with ``gh pr update-branch``. Instead, the linked issue moves to
    ``rework_requested`` and dispatch_rework can select it.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        ),
        devin=DevinConfig(adapter="command", dispatch_command="exit 0"),
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
            "mergeStateStatus": "BEHIND",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is False
    assert result.data["merge_attempt_warning"] is None
    # The sync step is bypassed: the PR head should not be advanced.
    assert fake_gh.prs[0]["headRefOid"] == "sha-abc123"

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["status"] == "rework_requested"
    assert state["events"][-1]["kind"] == "merge_ready"
    conflict_events = [
        e for e in state["events"] if e["kind"] == "merge_conflict_rework_requested"
    ]
    assert len(conflict_events) == 1
    assert conflict_events[0]["payload"]["pr_number"] == 456
    assert conflict_events[0]["payload"]["issue_number"] == 123

    # The rework prompt was written and the issue was labeled for rework.
    prompt_path = paths.prs / "pr-456" / "rework-prompt.md"
    assert prompt_path.exists()
    assert (123, config.labels.needs_rework) in fake_gh.labels_added

    # dispatch_rework can pick the issue up and launch a worker.
    dispatch = app.dispatch_rework()
    assert dispatch.data["selected_count"] == 1
    assert dispatch.data["sessions"][0]["issue_number"] == 123
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"


def _init_cross_pr_revert_repo(repo_root: Path) -> tuple[str, str, str]:
    """Set up a git repo where main has C and an agent branch merges C then reverts C.

    Returns ``(base_sha, feature_sha, agent_sha)`` where ``feature_sha`` is the
    commit on main that the agent branch silently reverts.
    """
    remote = repo_root / "remote"
    remote.mkdir()
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main"],
        cwd=remote,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(remote)],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Base commit (before the feature that will be reverted)
    (repo_root / "base.txt").write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "base.txt"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # Feature commit C on main
    (repo_root / "feature.txt").write_text("feature", encoding="utf-8")
    subprocess.run(["git", "add", "feature.txt"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feature C"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    feature_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Agent branch merges the feature commit and then reverts it
    subprocess.run(
        ["git", "checkout", "-b", "agent/issue-123-revert", base_sha],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "merge", "--no-ff", "main", "-m", "Merge main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "revert", "--no-edit", feature_sha],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    agent_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "push", "-u", "origin", "agent/issue-123-revert"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Leave main checked out in the orchestrator repo
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    return base_sha, feature_sha, agent_sha


def test_merge_ready_silent_cross_pr_revert_blocks_and_routes_to_rework(
    tmp_path: Path,
) -> None:
    """Issue #390: a branch that merges a base commit and reverts it must not merge."""
    from charlie_work.config import AutoMergeConfig, DevinConfig

    _base_sha, _feature_sha, agent_sha = _init_cross_pr_revert_repo(tmp_path)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        ),
        devin=DevinConfig(adapter="command", dispatch_command="exit 0"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: revert cross-pr",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-revert",
            "baseRefName": "main",
            "headRefOid": agent_sha,
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["cross_pr_revert_detected"] is True
    assert result.data["cross_pr_revert_routed"] is True
    assert result.data["merge_conflict"] is False
    assert "feature C" in result.data.get("cross_pr_revert_reason", "")

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["status"] == "rework_requested"
    assert any(e["kind"] == "cross_pr_revert_rework_requested" for e in state["events"])
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])

    prompt_path = paths.prs / "pr-456" / "rework-prompt.md"
    assert prompt_path.exists()
    assert (123, config.labels.needs_rework) in fake_gh.labels_added


def test_merge_ready_silent_cross_pr_revert_allows_explicit_marker(
    tmp_path: Path,
) -> None:
    """Issue #390: an explicit 'allow-revert:' marker line in the PR body suppresses the block."""
    from charlie_work.config import AutoMergeConfig, DevinConfig

    _base_sha, _feature_sha, agent_sha = _init_cross_pr_revert_repo(tmp_path)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        ),
        devin=DevinConfig(adapter="command", dispatch_command="exit 0"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: intentional revert",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-revert",
            "baseRefName": "main",
            "headRefOid": agent_sha,
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nallow-revert: intentional revert of feature C",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["can_merge"] is True
    assert result.data["cross_pr_revert_detected"] is False
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]


def test_merge_ready_silent_cross_pr_revert_prompt_echo_does_not_bypass(
    tmp_path: Path,
) -> None:
    """Issue #390: a bare 'allow-revert' word (e.g. quoting the rework prompt) must not bypass the gate."""
    from charlie_work.config import AutoMergeConfig, DevinConfig

    _base_sha, _feature_sha, agent_sha = _init_cross_pr_revert_repo(tmp_path)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        ),
        devin=DevinConfig(adapter="command", dispatch_command="exit 0"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: revert cross-pr",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-revert",
            "baseRefName": "main",
            "headRefOid": agent_sha,
            "mergeStateStatus": "CLEAN",
            "body": (
                "Closes #123\n\n"
                "...or add an explicit 'allow-revert' marker to the PR body if the revert "
                "is intentional. Then push the corrected branch and re-request review."
            ),
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["cross_pr_revert_detected"] is True
    assert result.data["cross_pr_revert_routed"] is True
    assert result.data["merge_conflict"] is False
    assert "feature C" in result.data.get("cross_pr_revert_reason", "")


def test_merge_ready_stale_base_not_routed_to_rework(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #371: a stale but fast-forwardable base is deferred, not sent to rework."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # The default base_head_sha is base-sha, which is already the merge-base of
    # sha-abc123. Advance it to a post-merge tip whose merge-base with sha-abc123
    # is still base-sha, so the freshness gate sees a stale base.
    post_merge_base = "main-merged-sha-abc123"
    fake_gh.base_head_sha = post_merge_base
    fake_gh.commits[post_merge_base] = {"parents": [{"sha": "base-sha"}, {"sha": "sha-abc123"}]}
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "BEHIND",
            "mergeable": "MERGEABLE",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    # Simulate a base-sync that reports success but does not advance the head, so
    # the merge-base freshness gate still defers the PR.
    monkeypatch.setattr(fake_gh, "pr_update_branch", lambda pr_number: True)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is False
    assert result.data.get("stale_base") is True
    assert result.data["merge_attempt_alarm"] is False
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "approved"
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])
    assert (paths.prs / "pr-456" / "rework-prompt.md").exists() is False
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


def test_merge_ready_conflict_alarm_message_is_honest(tmp_path: Path) -> None:
    """Issue #371: a persistent merge conflict deferral produces an honest alarm."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=3,
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
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")

    result1 = app.merge_ready(456, merge=False)
    assert result1.data["merge_conflict"] is True
    assert result1.data["merge_attempt_alarm"] is False
    assert result1.data["merge_attempt_warning"] is None

    result2 = app.merge_ready(456, merge=False)
    assert result2.data["merge_attempt_alarm"] is False

    result3 = app.merge_ready(456, merge=False)
    assert result3.data["merge_attempt_alarm"] is True
    warning = result3.data["merge_attempt_warning"]
    assert warning is not None
    assert "PR #456 approved but unmergeable for 3 passes" in warning
    assert "merge conflict" in warning.lower()

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    conflict_events = [
        e for e in state["events"] if e["kind"] == "merge_conflict_rework_requested"
    ]
    assert len(conflict_events) == 1
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert len(alarm_events) == 1
    assert "merge conflict" in alarm_events[0]["payload"]["message"].lower()


def test_merge_ready_conflict_no_linked_issue_alarm_is_honest(tmp_path: Path) -> None:
    """Issue #379: an approved conflicting PR with no linked issue cannot be routed.

    The alarm must be honest about the inability to dispatch rework, not claim
    a rework worker was dispatched.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Cross-repo fix",
            "url": "https://example.test/pull/456",
            "headRefName": "fork/fix",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Tests: regression coverage added.",
            "labels": [],
            "isCrossRepository": True,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is True
    warning = result.data["merge_attempt_warning"]
    assert warning is not None
    assert "PR #456 approved but unmergeable for 1 pass" in warning
    assert "merge conflict" in warning.lower()
    assert "no linked issue, cannot route to rework" in warning
    assert result.data["issue"] is None
    assert result.data["label_error"] is None

    state = load_state(paths.state_file)
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])
    assert (paths.prs / "pr-456" / "rework-prompt.md").exists() is False
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


def test_merge_ready_conflict_label_failure_is_recorded(tmp_path: Path) -> None:
    """Issue #379: a merge-conflict rework routing label failure is not swallowed.

    The rework label error must be returned in data['label_error'] and reflected
    in the alarm message.
    """
    from charlie_work.config import AutoMergeConfig
    from charlie_work.labels import TransitionOutcome

    class ReworkLabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            return False

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = ReworkLabelFailGitHub()
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
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is True
    warning = result.data["merge_attempt_warning"]
    assert warning is not None
    assert "merge conflict" in warning.lower()
    assert "rework dispatch attempted" in warning
    assert "label update failed" in warning

    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "rework_requested"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert len(label_error["add_failures"]) > 0

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


def test_merge_ready_conflict_inflight_worker_returns_early(tmp_path: Path) -> None:
    """Issue #379: a merge conflict whose linked issue is already in-flight is not re-routed.

    The early return must not fire the alarm and must leave the worker state alone.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
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
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"]["status"] = "dispatched"
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is False
    assert result.data["merge_attempt_warning"] is None
    assert result.message == "PR #456 merge conflict is being resolved by a rework worker"

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])


@pytest.mark.parametrize("terminal_status", ["escalated", "blocked"])
def test_merge_ready_conflict_human_terminal_issue_not_rerouted(
    tmp_path: Path, terminal_status: str
) -> None:
    """Issue #379 rework: a merge conflict whose linked issue is escalated/blocked
    (human-terminal) must never be rerouted to rework_requested.

    transition() has no source-state validation, so rerouting would silently
    strip the human_needed label and hand the issue back to automation behind
    the human's back. The PR and issue must be left untouched.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
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
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    # Mark the linked issue as carrying the human_needed label, matching a
    # real escalated/blocked issue, so a stripped label would be observable.
    fake_gh.issues[0]["labels"] = [{"name": config.labels.human_needed}]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"]["status"] = terminal_status
        save_state(paths.state_file, state)

    labels_removed_before = list(fake_gh.labels_removed)
    labels_added_before = list(fake_gh.labels_added)

    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is False
    assert result.data["merge_attempt_warning"] is None

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == terminal_status
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])
    # No label mutation must have been issued for the linked issue —
    # human_needed must stay in place.
    assert fake_gh.labels_removed == labels_removed_before
    assert fake_gh.labels_added == labels_added_before


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

    app.record_review(456, "approved", summary="lgtm")
    app.record_review(789, "approved", summary="lgtm")
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
        app.record_review(pr_number, "approved", summary="lgtm")
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

    app.record_review(456, "approved", summary="lgtm")
    app.record_review(789, "approved", summary="lgtm")
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

    app.record_review(456, "approved", summary="lgtm")
    app.record_review(789, "approved", summary="lgtm")
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
        app.record_review(pr_number, "approved", summary="lgtm")
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
        app.record_review(pr_number, "approved", summary="lgtm")
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

    app.record_review(456, "approved", summary="lgtm")
    app.record_review(789, "request_changes", summary="needs work")
    app.record_review(101, "approved", summary="lgtm")
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


def test_merge_ready_compare_unavailable_fail_closed(tmp_path: Path) -> None:
    """Issue #333: a failed compare API returns ``None`` and merge_ready fails closed.

    Mutating the gate to fail-open (treating ``base_current is None`` as current)
    causes this test to fail because the PR is merged instead of deferred.
    """
    from charlie_work.config import AutoMergeConfig

    class FakeGitHubCompareUnavailable(FakeGitHub):
        """compare() returns None, simulating an unavailable GitHub compare API."""

        def compare(self, base: str, head: str) -> dict[str, Any] | None:
            return None

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubCompareUnavailable()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merged"] is False
    assert result.data.get("stale_base") is True
    assert fake_gh.merged == []

    state = json.loads(paths.state_file.read_text())
    stale_events = [
        event for event in state["events"] if event["kind"] == "merge_deferred_stale_base"
    ]
    assert len(stale_events) == 1
    assert stale_events[0]["payload"]["pr_number"] == 456
    assert stale_events[0]["payload"]["reason"] == "compare_unavailable"


def test_merge_ready_merge_train_post_sync_head_race_rejected(tmp_path: Path) -> None:
    """If pr_view returns a non-qualifying head after update-branch, do not merge.

    Regression test for the TOCTOU described in issue #258: a racing push to the
    PR branch in the update-window must not be blessed as the approved head.
    """
    from charlie_work.config import AutoMergeConfig

    class FakeGitHubRacingUpdate(FakeGitHub):
        def pr_update_branch(self, pr_number: int) -> bool:
            ok = super().pr_update_branch(pr_number)
            for pr in self.prs:
                if pr["number"] == pr_number:
                    racing = "racing-sha"
                    self.pr_head_shas[pr_number] = racing
                    self.commits[racing] = {
                        "parents": [{"sha": "other-sha"}],
                        "committer": {"login": "not-web-flow"},
                        "commit": {"committer": {"name": "Not GitHub"}},
                    }
            return ok

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubRacingUpdate()
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
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert fake_gh.merged == []
    # The approved head must not be migrated to the racing SHA.
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-abc123"


def _make_racing_merge_ready_app(
    tmp_path: Path, racing_commit: dict[str, Any]
) -> tuple[Any, Any, Any]:
    """Build an app whose update-branch races in a crafted merge commit.

    The racing commit's parents deliberately satisfy the structural checks
    (two parents, old head included) so only the committer-identity predicate
    is under test.
    """
    from charlie_work.config import AutoMergeConfig

    class FakeGitHubRacingUpdate(FakeGitHub):
        def pr_update_branch(self, pr_number: int) -> bool:
            ok = super().pr_update_branch(pr_number)
            racing = "racing-sha"
            self.pr_head_shas[pr_number] = racing
            self.commits[racing] = racing_commit
            return ok

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubRacingUpdate()
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
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, fake_gh, paths


def test_merge_ready_race_with_spoofed_committer_name_rejected(tmp_path: Path) -> None:
    """A racing push whose git metadata claims name 'GitHub' must still be rejected.

    The commit.committer.name field is settable by any pusher; only the
    web-flow account login together with the GitHub name identifies a real
    base-sync merge. Structural parent checks are satisfied on purpose.
    """
    app, fake_gh, paths = _make_racing_merge_ready_app(
        tmp_path,
        {
            "parents": [{"sha": "sha-abc123"}, {"sha": "main-tip-sha"}],
            "committer": {"login": "attacker"},
            "commit": {"committer": {"name": "GitHub"}},
        },
    )

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=True)

    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert fake_gh.merged == []
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-abc123"


def test_merge_ready_race_with_spoofed_webflow_login_rejected(tmp_path: Path) -> None:
    """A racing push attributed to web-flow but with a non-GitHub name is rejected.

    Login attribution follows the committer email, which a pusher can set to
    noreply@github.com; the git metadata name must corroborate it.
    """
    app, fake_gh, paths = _make_racing_merge_ready_app(
        tmp_path,
        {
            "parents": [{"sha": "sha-abc123"}, {"sha": "main-tip-sha"}],
            "committer": {"login": "web-flow"},
            "commit": {"committer": {"name": "Devin Worker"}},
        },
    )

    app.record_review(456, "approved", summary="lgtm")
    result = app.merge_ready(456, merge=True)

    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert fake_gh.merged == []
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-abc123"


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

    app.record_review(456, "approved", summary="lgtm")
    app.record_review(789, "approved", summary="lgtm")
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


def test_concurrency_governor_unlimited_when_unset(tmp_path: Path) -> None:
    """When max_concurrent_sessions is 0 (default), dispatch should behave as before (unlimited)."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=0),
        devin=DevinConfig(adapter="manual"),
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
        devin=DevinConfig(adapter="manual"),
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
    def mock_count_live(sessions_dir, state_file=None):
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

    app.gh.prs[0]["state"] = "CLOSED"
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
    app = OrchestratorApp(
        tmp_path, paths, config, fake_gh, fleet_dir_override=str(tmp_path / "fleet")
    )

    app.gh.prs[0]["state"] = "CLOSED"
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
    def mock_count_live(sessions_dir, state_file=None):
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
    app = OrchestratorApp(
        tmp_path, paths, config, fake_gh, fleet_dir_override=str(tmp_path / "fleet")
    )

    app.gh.prs[0]["state"] = "CLOSED"
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
    def mock_count_live(sessions_dir, state_file=None):
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
    app = OrchestratorApp(
        tmp_path, paths, config, fake_gh, fleet_dir_override=str(tmp_path / "fleet")
    )

    app.gh.prs[0]["state"] = "CLOSED"
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

    def mock_count_live(sessions_dir, state_file=None):
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

    def mock_count_live(sessions_dir, state_file=None):
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


def _dispatch_rework_config() -> OrchestratorConfig:
    return OrchestratorConfig(
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


def test_dispatch_rework_routes_to_review_instead_of_relaunch_when_head_moved(
    tmp_path: Path,
) -> None:
    """Issue #339: a rework worker relaunched onto a PR whose rework was
    already pushed (head moved past the last request_changes verdict) finds
    nothing to do, idles, and gets watchdog-reaped, burning a session and a
    concurrency slot. dispatch_rework must detect the head-moved-with-real-
    content-change case and route the issue to the review lane instead of
    launching a redundant worker (acceptance criterion 1).
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record a request_changes decision: this puts the issue into
    # rework_requested and pins reviewed_head_sha/reviewed_patch_id to the
    # current (pre-rework) head/diff.
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+first"
    )
    app.record_review(456, "request_changes", summary="fix A")

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        assert state["issues"]["123"]["status"] == "rework_requested"
        assert state["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"

    # Simulate the rework already having been pushed: head advances AND the
    # diff content genuinely changes (not just a sync-merge).
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+second"
    )

    # A rework prompt exists — absent the fix, this is exactly what lets
    # dispatch proceed and launch a redundant worker.
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["routed_to_review"] == [123]
    assert result.data["skipped_head_indeterminate"] == []
    # No rework worker was launched for issue 123.
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    # Routed to the review lane instead: needs_rework cleared, reviewing/pr_open added.
    assert (123, "agent:needs-rework") in fake_gh.labels_removed
    assert (123, "agent:reviewing") in fake_gh.labels_added

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "reviewing"
    assert any(e["kind"] == "rework_already_pushed" for e in state["events"])


def test_dispatch_rework_launches_when_head_matches_reviewed_sha(tmp_path: Path) -> None:
    """Regression pin (issue #339 acceptance criterion 2): dispatch_rework
    must still launch exactly as before when the PR head is unchanged since
    the request_changes verdict — the rework is genuinely outstanding.
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(456, "request_changes", summary="fix A")

    # Head is unchanged (still the default "sha-abc123") — genuinely outstanding.
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["routed_to_review"] == []
    assert result.data["sessions"][0]["issue_number"] == 123
    assert (123, "agent:in-progress") in fake_gh.labels_added


def test_dispatch_rework_skips_without_stranding_when_head_indeterminate(
    tmp_path: Path,
) -> None:
    """Issue #339 fail-safe direction: if content identity can't be
    established after a head change (diff fetch fails), dispatch_rework must
    not launch a redundant worker, but must also not strand the issue — it
    stays rework_requested so the next pass retries (acceptance: fail-closed
    without permanent stranding).
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(456, "request_changes", summary="fix A")

    # Head moves, but the diff fetch now fails — GitHub.pr_diff's real
    # allow_failure=True contract returns "" on failure, so an empty diff is
    # the correct fake-adapter stand-in for "gh pr diff failed".
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = ""

    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["routed_to_review"] == []
    assert result.data["skipped_head_indeterminate"] == [123]
    # Not stranded: still rework_requested so the next pass retries.
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    # No launch, and no review-lane relabeling either — genuinely indeterminate.
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    assert (123, "agent:reviewing") not in fake_gh.labels_added


def test_dispatch_rework_launches_when_head_moved_by_sync_merge_only(tmp_path: Path) -> None:
    """Issue #339 acceptance: a sync-merge-only head advance (base merged into
    the PR branch moves headRefOid, but the PR's own patch content is
    unchanged) must NOT be treated as "already reworked" — the same patch
    still needs a genuine rework cycle, so dispatch_rework must still launch.

    Regression coverage for a reviewer-caught gap: mutating away the
    same-patch-id carve-out (routing to review on ANY head mismatch,
    regardless of patch-id) left all pre-existing dispatch_rework tests
    green, because none of them exercised a moved-head/same-patch-id PR —
    exactly the common case on a fleet where every open PR gets synced with
    main constantly.
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    diff_text = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    fake_gh.diffs[456] = diff_text
    app.record_review(456, "request_changes", summary="fix A")

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        assert state["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"
        assert state["prs"]["456"]["reviewed_patch_id"]

    # Simulate a sync-merge: base merged into the branch moves the head SHA
    # (FakeGitHub.pr_update_branch models this exactly), but the PR's diff
    # content is unchanged — a real sync merge does not touch the patch.
    fake_gh.pr_update_branch(456)
    new_head = fake_gh.prs[0]["headRefOid"]
    assert new_head != "sha-abc123"
    fake_gh.pr_head_shas[456] = new_head
    fake_gh.diffs[456] = diff_text

    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    fake_gh.labels_added.clear()

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["routed_to_review"] == []
    assert result.data["review_blocked_retry"] == []
    assert result.data["sessions"][0]["issue_number"] == 123
    assert (123, "agent:in-progress") in fake_gh.labels_added


def test_dispatch_rework_head_moved_but_review_blocked_by_janitor_retries_next_pass(
    tmp_path: Path,
) -> None:
    """Issue #339 finding 1 (reviewer repro): a candidate whose PR head moved
    with a real content change gets routed to review() — but if the PR is
    CONFLICTING, review()'s deterministic janitor gate returns ok=False
    *before* writing a packet or firing the review_started transition, and
    without touching reviewed_head_sha. The routing helper must not
    force-flip the issue's status to "reviewing" in that case: doing so would
    desync state.json from GitHub reality (labels still say needs-rework, no
    packet exists) and strand the issue outside dispatch_rework's own
    candidate pool forever, with no automated recovery path. The issue must
    stay rework_requested so the next pass retries (the janitor block is
    often transient, e.g. a merge-train branch sync resolving the conflict).
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+first"
    )
    app.record_review(456, "request_changes", summary="fix A")

    # Head advances with a real content change (routes to review) AND the PR
    # is now conflicting (janitor blocks review() before any packet/label write).
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.prs[0]["mergeable"] = "CONFLICTING"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+second"
    )

    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    # Not reported as successfully routed — review() never produced a packet.
    assert result.data["routed_to_review"] == []
    assert result.data["review_blocked_retry"] == [123]

    # No label churn at all: neither a launch nor a review_started transition.
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    assert (123, "agent:reviewing") not in fake_gh.labels_added
    assert (123, "agent:needs-rework") not in fake_gh.labels_removed

    # Not stranded: still rework_requested so the next pass retries.
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    # PR record shows the janitor block, not a flipped "reviewing" state.
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert any(
        e["kind"] == "rework_already_pushed" and e["payload"].get("routed") is False
        for e in state["events"]
    )


def test_dispatch_rework_skips_when_live_head_ref_oid_missing(tmp_path: Path) -> None:
    """Non-blocking coverage: live_head_sha itself (not just a diff-fetch
    failure) can be unavailable — pr_list() returning a record with no
    headRefOid. Must fail closed the same as any other indeterminate case.
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(456, "request_changes", summary="fix A")

    # Live head is unavailable from the PR list response.
    fake_gh.prs[0]["headRefOid"] = None

    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["skipped_head_indeterminate"] == [123]
    assert result.data["routed_to_review"] == []
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"


def test_dispatch_rework_skips_when_reviewed_patch_id_missing(tmp_path: Path) -> None:
    """Non-blocking coverage: an older/malformed pr_state that recorded
    reviewed_head_sha without reviewed_patch_id must also fail closed on a
    head mismatch rather than guessing at content identity.
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
            # reviewed_patch_id deliberately absent (older/malformed record).
        }
        save_state(paths.state_file, state)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    # Head has moved relative to the recorded reviewed_head_sha.
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["skipped_head_indeterminate"] == [123]
    assert result.data["routed_to_review"] == []
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"


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


def test_review_does_not_clobber_escalated_label_on_head_advance(tmp_path: Path) -> None:
    """Issue #384: an escalated issue must stay terminal on re-review.

    After record_review escalates an issue to agent:human-needed, a later
    review() pass (e.g., from loop()) that sees a newly-advanced head must not
    regenerate a packet or fire review_started, which would strip the human-needed
    label and put the PR back into an active-automation state.
    """
    config = OrchestratorConfig()  # max_rework_cycles = 2
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # First request_changes (count = 1, head = "sha-1")
    fake_gh.pr_head_shas[456] = "sha-1"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 1"
    app.record_review(456, "request_changes", summary="fix A")

    # Second request_changes (count = 2, head = "sha-2")
    fake_gh.pr_head_shas[456] = "sha-2"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 2"
    app.record_review(456, "request_changes", summary="fix B")

    # Third request_changes (escalated, head = "sha-3")
    fake_gh.pr_head_shas[456] = "sha-3"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 3"
    app.record_review(456, "request_changes", summary="fix C")

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["status"] == "escalated"
    assert (123, config.labels.human_needed) in fake_gh.labels_added

    # Clear label tracking to isolate the review() call
    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    # Simulate a worker pushing after escalation: new head and new diff
    fake_gh.pr_head_shas[456] = "sha-new"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change new"

    result = app.review(456)

    # review() must short-circuit and must not touch the human-needed label
    assert result.ok is True
    assert (123, config.labels.human_needed) not in fake_gh.labels_removed
    assert (123, config.labels.pr_open) not in fake_gh.labels_added
    assert (123, config.labels.reviewing) not in fake_gh.labels_added

    # State must stay escalated and not be overwritten back to "reviewing"
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["status"] == "escalated"


def test_review_short_circuits_escalated_issue_less_pr(tmp_path: Path) -> None:
    """Issue #384: PR-level escalation is terminal even without a linked issue.

    Cross-repo PRs (or same-repo branches that don't match the configured
    prefix) fail closed: ``linked_issue_number`` returns ``None``.
    ``record_review`` still sets the PR's own state status to ``"escalated"``
    after ``max_rework_cycles``. A later ``review()`` pass must short-circuit on
    that PR-level status and must not fall through to the janitor gate, which
    would overwrite ``status`` with ``"janitor_blocked"``.
    """
    config = OrchestratorConfig()  # max_rework_cycles = 2, require_issue_link = True
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Cross-repo PRs never resolve to a linked issue for lifecycle purposes.
    fake_gh.prs[0]["isCrossRepository"] = True

    # First request_changes (count = 1, head = "sha-1")
    fake_gh.pr_head_shas[456] = "sha-1"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 1"
    app.record_review(456, "request_changes", summary="fix A")

    # Second request_changes (count = 2, head = "sha-2")
    fake_gh.pr_head_shas[456] = "sha-2"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 2"
    app.record_review(456, "request_changes", summary="fix B")

    # Third request_changes (escalated, head = "sha-3")
    fake_gh.pr_head_shas[456] = "sha-3"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+change 3"
    result = app.record_review(456, "request_changes", summary="fix C")
    assert result.data["escalated"] is True

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"
    # No issue entry should exist for an issue-less PR.
    assert "123" not in state.get("issues", {})
    # Label transitions are gated on issue_number, so no labels should fire.
    assert not fake_gh.labels_added
    assert not fake_gh.labels_removed

    # Clear label tracking to isolate the review() call.
    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    # Simulate a later loop/review pass with a new head and new diff.
    fake_gh.pr_head_shas[456] = "sha-new"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+new change"
    review_result = app.review(456)

    # review() must short-circuit and must not touch anything.
    assert review_result.ok is True
    assert review_result.data.get("skipped") is True
    assert not fake_gh.labels_added
    assert not fake_gh.labels_removed

    # No review packet/decision should be (re)written on the short-circuited call.
    pr_dir = paths.prs / "pr-456"
    assert not (pr_dir / "review-prompt.md").exists()

    # The PR state must remain escalated, not be clobbered to janitor_blocked.
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"


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

    app.gh.prs[0]["state"] = "CLOSED"
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

    app.gh.prs[0]["state"] = "CLOSED"
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

    app.gh.prs[0]["state"] = "CLOSED"
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

    app.gh.prs[0]["state"] = "CLOSED"
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

    app.gh.prs[0]["state"] = "CLOSED"
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
        app.gh.prs[0]["state"] = "CLOSED"
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

    config = OrchestratorConfig(devin=DevinConfig(adapter="manual"))
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
        devin=DevinConfig(adapter="manual"),
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
        app.gh.prs[0]["state"] = "CLOSED"
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
    def fake_github(
        repo_root: Path, dry_run: bool = False, runtime: object | None = None
    ) -> GitHub:
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
    def mock_count_live(sessions_dir, state_file=None):
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
    app.gh.prs[0]["state"] = "CLOSED"
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

    # Test fleet review-queue parsing
    args_review_queue = parser.parse_args(["fleet", "review-queue"])
    assert args_review_queue.command == "fleet"
    assert args_review_queue.fleet_command == "review-queue"

    # Test single-repo review-queue parsing
    args_single = parser.parse_args(["review-queue"])
    assert args_single.command == "review-queue"


def test_fleet_review_queue_aggregates_and_isolates_errors(tmp_path: Path, monkeypatch) -> None:
    """Issue #369: fleet review-queue aggregates per repo and isolates errors."""
    fleet_override = str(tmp_path / "fleet")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", fleet_override)

    repo_ok = tmp_path / "repo_ok"
    repo_ok.mkdir()
    config_ok = repo_ok / "orchestrator.config.yaml"
    config_ok.write_text(
        "labels:\n  ready: automated-ready\n  queued: agent:queued\n  in_progress: agent:in-progress\nruntime:\n  state_dir: .var/charlie-work\n"
    )
    (repo_ok / ".var" / "charlie-work").mkdir(parents=True)

    # Good repo has one PR with a current packet and no decision
    prs_dir = repo_ok / ".var" / "charlie-work" / "prs" / "pr-7"
    prs_dir.mkdir(parents=True)
    (prs_dir / "pr.json").write_text(
        json.dumps({"number": 7, "headRefOid": "sha-7"}), encoding="utf-8"
    )
    (prs_dir / "review-prompt.md").write_text("packet for PR 7", encoding="utf-8")

    # Create a valid state.json so load_state doesn't fail
    (repo_ok / ".var" / "charlie-work" / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )

    fleet_json_path = Path(fleet_override) / "fleet.json"
    fleet_json_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = {
        "version": 1,
        "repos": {
            "owner/repo_ok": {
                "repo_root": str(repo_ok),
                "name_with_owner": "owner/repo_ok",
                "config_path": str(config_ok),
                "state_dir": str(repo_ok / ".var" / "charlie-work"),
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
    fleet_json_path.write_text(json.dumps(registry_data, indent=2))

    from charlie_work.github import GitHub

    def mock_pr_list(self):
        return [
            {
                "number": 7,
                "title": "Fix #7: thing",
                "url": "https://example.test/pull/7",
                "headRefName": "agent/issue-7-fix",
                "baseRefName": "main",
                "headRefOid": "sha-7",
                "mergeStateStatus": "CLEAN",
                "body": "Closes #7",
                "labels": [],
                "isCrossRepository": False,
                "state": "OPEN",
            }
        ]

    monkeypatch.setattr(GitHub, "pr_list", mock_pr_list)

    args = cli.build_parser().parse_args(["fleet", "review-queue"])
    result = cli.run_fleet_review_queue(args)

    assert result.ok is False
    assert "1 repo(s), 1 error(s)" in result.message
    assert result.data["repos"]["owner/repo_ok"]["queue"] == [
        {
            "pr": 7,
            "issue": 7,
            "packet_head_sha": "sha-7",
            "decision": "missing",
            "reviewed_head_sha": None,
        }
    ]
    assert len(result.data["errors"]) == 1
    assert result.data["errors"][0]["repo_key"] == "owner/repo_broken"
    assert "does not exist" in result.data["errors"][0]["error"]


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
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(
            enabled=True, stall_minutes=20, max_inconclusive_probe_deferrals=10
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = []
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=False)

    sessions_dir = app._resolve(config.devin.sessions_dir)
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
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(
            enabled=True, stall_minutes=20, max_inconclusive_probe_deferrals=10
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = []
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=False)

    sessions_dir = app._resolve(config.devin.sessions_dir)
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
    """Test that redispatch_at is only written by the three known call sites (issue #165)."""
    # This test verifies by code inspection that redispatch_at is only written in:
    # 1. dispatch_rework (workflow.py:2440-2472)
    # 2. _classify_dead_sessions_and_update_throttle_state (workflow.py:468-504)
    # 3. _reap_restore_rework_requested (issue #315 review finding 2: the
    #    rework lane must consult the same redispatch cap the other two
    #    sites do, instead of preserving redispatch_at unchanged forever).
    # No other code paths write to redispatch_at.

    # Verify the two call sites exist in the code
    import charlie_work.workflow as workflow_module
    import inspect

    workflow_source = inspect.getsource(workflow_module)

    # Count occurrences of redispatch_at assignments to entry
    # We have 2 assignments in dispatch_rework (normal + escalation), 3 in
    # _classify_dead_sessions_and_update_throttle_state (launch-failure
    # escalation + dead-session normal + dead-session escalation), and 2 in
    # _reap_restore_rework_requested (rework escalation + rework restore,
    # issue #315).
    # Total of 7 assignments is correct.
    redispatch_assignments = workflow_source.count('entry["redispatch_at"]')
    assert redispatch_assignments == 7, (
        f"Expected 7 redispatch_at assignments, found {redispatch_assignments}"
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

    # Worker PID should be preserved for recovery-path verification (issue #282)
    assert entry["worker_pid"] == 99999
    assert entry["worker_process_start_time"] == 1234567890.0

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

    # Worker PID should be preserved for recovery-path verification (issue #282)
    assert entry["worker_pid"] == 99999
    assert entry["worker_process_start_time"] == 1234567890.0

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

    # Worker PID should be preserved for recovery-path verification (issue #282)
    assert entry["worker_pid"] == 99999
    assert entry["worker_process_start_time"] == 1234567890.0

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

    # Worker PID should be preserved for recovery-path verification (issue #282)
    assert entry["worker_pid"] == 99999
    assert entry["worker_process_start_time"] == 1234567890.0

    # Verify drift event was logged (not recovered)
    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["issue_number"] == 207
    assert drift_events[0]["payload"]["reason"] == "dead_worker_no_open_pr"

    # Verify NO recovered event
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 0

    # Issue #259: the entry should be marked so it is not re-flagged every pass.
    assert "orphan_flagged_at" in entry


def test_orphaned_worker_detection_no_open_pr_emits_once(tmp_path: Path) -> None:
    """Issue #259: sweep must emit only one drift event per zombie across N passes."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["259"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubForOrphan()

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(3):
            _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    state = load_state(paths.state_file)
    drift_events = [e for e in state.get("events", []) if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1, (
        f"Expected exactly one orphaned_worker_drift event, got {len(drift_events)}"
    )
    assert drift_events[0]["payload"]["issue_number"] == 259
    assert drift_events[0]["payload"]["reason"] == "dead_worker_no_open_pr"

    entry = state["issues"]["259"]
    assert entry.get("status") == "dispatched"
    assert "orphan_flagged_at" in entry


def test_orphaned_worker_with_flag_and_open_pr_request_changes_recovered(tmp_path: Path) -> None:
    """Issue #259 review: orphan suppression must not block open-PR recovery paths."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "orphan_flagged_at": "2024-01-01T00:00:00Z",
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

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # With an open PR and request_changes with unchanged head, recovery should
    # run regardless of the orphan_flagged_at suppression.
    assert entry.get("status") == "rework_requested"
    assert entry.get("dispatched_at") is None
    # Worker PID should be preserved for recovery-path verification (issue #282)
    assert entry["worker_pid"] == 99999
    assert entry["worker_process_start_time"] == 1234567890.0

    events = state.get("events", [])
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered_events) == 1
    assert recovered_events[0]["payload"]["reason"] == "dead_worker_with_request_changes"


def test_orphaned_worker_detection_bulk_sweep_excludes_pre_flagged(tmp_path: Path) -> None:
    """Issue #275 review: a sweep must aggregate only newly-flagged orphans.

    Pre-flagged entries (from #290's orphan_flagged_at guard) are suppressed
    before aggregation. A fresh bulk sweep of the remaining orphans is emitted
    as a single aggregated event.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    pre_flagged = {1, 2, 3}
    fresh = {4, 5, 6}
    for issue_number in pre_flagged | fresh:
        state["issues"][str(issue_number)] = {
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "dispatched_at": "2024-01-01T00:00:00Z",
        }
    for issue_number in pre_flagged:
        state["issues"][str(issue_number)]["orphan_flagged_at"] = "2024-01-01T00:00:00Z"
    save_state(paths.state_file, state)

    class FakeGitHubNoOrphanPrs(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubNoOrphanPrs()

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    state = load_state(paths.state_file)
    events = state.get("events", [])

    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 0, "pre-flagged orphans must not emit individual drift events"

    sweep_events = [e for e in events if e.get("kind") == "orphaned_worker_drift_sweep"]
    assert len(sweep_events) == 1
    assert sweep_events[0]["payload"]["count"] == len(fresh)
    assert set(sweep_events[0]["payload"]["issue_numbers"]) == fresh

    for issue_number in pre_flagged | fresh:
        entry = state["issues"][str(issue_number)]
        assert entry.get("status") == "dispatched"
        assert "orphan_flagged_at" in entry


def test_orphaned_worker_detection_bulk_sweep_does_not_flood_event_buffer(tmp_path: Path) -> None:
    """Regression test for issue #275: a single bulk reap sweep must not evict unrelated diagnostic events.

    A 500-issue orphan sweep would previously emit 500 ``orphaned_worker_drift``
    events and overrun the 200-entry event buffer. The sweep now aggregates
    same-kind events into one summary event, so prior diagnostic events survive.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    diagnostic_count = 199
    for seq in range(diagnostic_count):
        state = append_event(state, "diagnostic_event", {"seq": seq})
    for issue_number in range(1, 501):
        state["issues"][str(issue_number)] = {
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "dispatched_at": "2024-01-01T00:00:00Z",
        }
    save_state(paths.state_file, state)

    class FakeGitHubNoPrs(FakeGitHub):
        def __init__(self):
            super().__init__()
            self.prs = []

    fake_gh = FakeGitHubNoPrs()

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    state = load_state(paths.state_file)
    events = state["events"]

    # The 200-entry cap must not be exceeded and the buffer must not be flooded
    assert len(events) <= 200
    diagnostic_events = [e for e in events if e.get("kind") == "diagnostic_event"]
    assert len(diagnostic_events) == diagnostic_count

    # A single sweep summary should represent all 500 orphan drifts
    sweep_events = [e for e in events if e.get("kind") == "orphaned_worker_drift_sweep"]
    assert len(sweep_events) == 1
    assert sweep_events[0]["payload"]["count"] == 500
    assert set(sweep_events[0]["payload"]["issue_numbers"]) == set(range(1, 501))


# ---------------------------------------------------------------------------
# Issue #417: dead-session reclaim must be idempotent and resumable, not a
# one-shot handoff that permanently strands an issue if interrupted mid-way.
# ---------------------------------------------------------------------------


def test_orphaned_worker_no_open_pr_completes_interrupted_reclaim(tmp_path: Path) -> None:
    """Issue #417: a reclaim interrupted between the redispatch_at state.json
    write and the GitHub label swap (e.g. by a crash/reboot) must self-heal on
    the very next orphaned-worker sweep -- reproducing job-cannon #1172/#1176's
    exact fingerprint: status still "dispatched", worker_pid dead,
    redispatch_at already has one entry, and the GitHub issue still carries
    the stale active label alongside the ready label that was never removed
    from the original dispatch.

    Also covers the non-blocking follow-up: `status` deliberately never
    advances away from "dispatched" (matching the sidecar-based lane's own
    issue #282 fingerprint preservation), so a second pass would otherwise
    re-discover this same entry and emit a spurious orphaned_worker_drift for
    an issue that is already fully fixed. It must not.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1172"] = {
        "status": "dispatched",
        "dispatched_at": "2026-07-14T16:14:40Z",
        "redispatch_at": ["2026-07-14T16:17:20.606175Z"],
        "worker_pid": 40680,
        "worker_process_start_time": 1784045680.2843266,
    }
    save_state(paths.state_file, state)

    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 1172,
            "title": "some bug",
            "url": "https://example.test/issues/1172",
            "body": "",
            "labels": [
                {"name": config.labels.in_progress},
                {"name": config.labels.ready},
            ],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []  # dead worker never opened a PR

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    # The stale active label must have been removed -- this is exactly what
    # _is_dispatchable requires (ready present, no active label) for the
    # issue to become dispatchable again.
    assert (1172, config.labels.in_progress) in fake_gh.labels_removed
    # ready was already present, so it must NOT be redundantly re-added.
    assert (1172, config.labels.ready) not in fake_gh.labels_added

    state = load_state(paths.state_file)
    entry = state["issues"]["1172"]
    # This lane must never touch the sidecar-based lane's own bookkeeping --
    # a retry here must not inflate the escalation-cap counter.
    assert entry["redispatch_at"] == ["2026-07-14T16:17:20.606175Z"]
    assert entry["worker_pid"] == 40680

    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 1
    assert events[0]["payload"]["issue_number"] == 1172
    assert events[0]["payload"]["label_write_ok"] is True
    assert events[0]["payload"]["removed_labels"] == [config.labels.in_progress]

    # A once-stranded issue must not ALSO be flagged as unresolved drift now
    # that the reclaim fully succeeded.
    drift_events = [e for e in state["events"] if e["kind"] == "orphaned_worker_drift"]
    assert drift_events == []

    # Second pass: status.json still shows status="dispatched" with the same
    # dead worker_pid (nothing advanced it). FakeGitHub's remove/add_issue_label
    # only record calls -- unlike real GitHub, they don't mutate self.issues --
    # so simulate pass 1's successful label swap actually landing: in_progress
    # is gone, ready (already present) is unchanged. With labels now fully
    # correct (no active label, ready present) this must be a quiet no-op, not
    # a second "session_failed_relabeled" event or a fresh "orphaned_worker_drift"
    # for an issue that no longer needs anything.
    fake_gh.issues[0]["labels"] = [{"name": config.labels.ready}]
    fake_gh.labels_added = []
    fake_gh.labels_removed = []
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []
    state = load_state(paths.state_file)
    assert len([e for e in state["events"] if e["kind"] == "session_failed_relabeled"]) == 1
    assert [e for e in state["events"] if e["kind"] == "orphaned_worker_drift"] == []


def test_orphaned_worker_no_open_pr_reclaim_survives_label_api_failure(tmp_path: Path) -> None:
    """Issue #417: if the label swap itself fails (gh API error), the reclaim
    must not lose state.json bookkeeping or the sidecar-independent tracking,
    and a later pass -- once the API recovers -- must complete the reclaim.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1176"] = {
        "status": "dispatched",
        "dispatched_at": "2026-07-14T17:24:55Z",
        "redispatch_at": ["2026-07-14T17:29:56.087825Z"],
        "worker_pid": 29236,
        "worker_process_start_time": 1784049895.281971,
    }
    save_state(paths.state_file, state)

    class FlakyLabelGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.fail_remove = True

        def remove_issue_label(self, number: int, label: str) -> bool:
            self.labels_removed.append((number, label))
            return not self.fail_remove

    fake_gh = FlakyLabelGitHub()
    fake_gh.issues = [
        {
            "number": 1176,
            "title": "some other bug",
            "url": "https://example.test/issues/1176",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        # Pass 1: the gh API call fails.
        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["1176"]
    # Nothing lost: bookkeeping and the liveness fingerprint survive intact.
    assert entry["redispatch_at"] == ["2026-07-14T17:29:56.087825Z"]
    assert entry["worker_pid"] == 29236

    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 1
    assert events[0]["payload"]["label_write_ok"] is False
    assert (1176, config.labels.in_progress) in fake_gh.labels_removed

    # Pass 2: the API recovers.
    fake_gh.fail_remove = False
    fake_gh.labels_removed = []

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    assert (1176, config.labels.in_progress) in fake_gh.labels_removed
    assert (1176, config.labels.ready) in fake_gh.labels_added

    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 2
    assert events[1]["payload"]["label_write_ok"] is True
    # The redispatch_at bookkeeping must never have been touched by retries of
    # this sidecar-independent lane -- only the sidecar-based reap lane
    # (_classify_dead_sessions_and_update_throttle_state) owns that counter.
    assert state["issues"]["1176"]["redispatch_at"] == ["2026-07-14T17:29:56.087825Z"]


def test_classify_dead_sessions_no_open_pr_happy_path_reclaims_in_one_pass(
    tmp_path: Path,
) -> None:
    """Issue #417: the fully-clean happy path (no interruption, no API
    failure) must still fully reclaim a dead session -- with no open PR -- in
    a single pass of the sidecar-based reap lane, and now records
    label_write_ok=True so a genuine future failure is distinguishable from
    success.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig as DevinCfg
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinCfg(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 99,
            "title": "Fix thing",
            "url": "https://example.test/issues/99",
            "body": "Broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-99.log"
    log_path.write_text("Some work done, then the process died.\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-99.json"
    record = SessionRecord(
        issue_number=99,
        branch="agent/issue-99-x",
        worktree_path="/tmp/worktree-99",
        prompt_path="/tmp/prompt-99.md",
        command=("devin", "--prompt-file", "/tmp/prompt-99.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    assert (99, config.labels.in_progress) in fake_gh.labels_removed
    assert (99, config.labels.ready) in fake_gh.labels_added

    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 1
    assert events[0]["payload"]["label_write_ok"] is True
    assert events[0]["payload"]["added_ready"] is True
    assert (
        state["issues"]["99"]["redispatch_at"] and len(state["issues"]["99"]["redispatch_at"]) == 1
    )

    # The sidecar must be reaped once the reclaim fully succeeds.
    assert not sidecar_path.exists()


def test_orphaned_worker_no_open_pr_terminal_label_only_is_left_alone(tmp_path: Path) -> None:
    """Issue #417 regression: an issue in a legitimate terminal state (only
    agent:human-needed -- no active label, no ready) that ALSO happens to
    have a stale dispatched/dead-worker/no-PR state.json entry must be LEFT
    ALONE by the ground-truth label reclaim. A prior revision's early-exit
    gate (`if not active_labels and not needs_ready: continue`) proceeded
    whenever EITHER half was false, so a terminal-only issue (active_labels
    empty, needs_ready true) wrongly got `automated-ready` added back --
    producing a contradictory human-needed + automated-ready label pair and
    polluting the audit trail with a spurious `added_ready: True`. This test
    must fail against a head that regresses to that gate.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["500"] = {
        "status": "dispatched",
        "dispatched_at": "2026-07-01T00:00:00Z",
        "worker_pid": 12345,
        "worker_process_start_time": 1700000000.0,
    }
    save_state(paths.state_file, state)

    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 500,
            "title": "needs a human",
            "url": "https://example.test/issues/500",
            "body": "",
            "labels": [{"name": config.labels.human_needed}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    # No GitHub label call at all -- not even a redundant re-add of a label
    # that was already there.
    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []

    state = load_state(paths.state_file)
    assert [e for e in state["events"] if e["kind"] == "session_failed_relabeled"] == []
    # state.json bookkeeping for this issue must be untouched by the reclaim
    # (the pre-existing orphaned_worker_drift diagnostic fallback may still
    # flag it -- that part of the behavior predates issue #417 and is not
    # this test's concern).
    entry = state["issues"]["500"]
    assert entry.get("worker_pid") == 12345
    assert "redispatch_at" not in entry


def test_classify_dead_sessions_terminal_label_only_is_left_alone(tmp_path: Path) -> None:
    """Issue #417 regression: same bug as the orphaned-worker sweep's, but
    for the sidecar-based reap lane -- a dead session whose issue carries
    ONLY a terminal label must be left alone: no labels touched, and no
    redispatch_at bump (which would otherwise spend down the
    max_auto_redispatch escalation cap for an issue that needs no automatic
    recovery at all). This test must fail against a head that regresses to
    the `if not active_labels and not needs_ready: continue` gate.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig as DevinCfg
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinCfg(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 501,
            "title": "needs a human too",
            "url": "https://example.test/issues/501",
            "body": "",
            "labels": [{"name": config.labels.human_needed}],
        }
    ]

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-501.log"
    log_path.write_text("Some work done, then the process died.\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-501.json"
    record = SessionRecord(
        issue_number=501,
        branch="agent/issue-501-x",
        worktree_path="/tmp/worktree-501",
        prompt_path="/tmp/prompt-501.md",
        command=("devin", "--prompt-file", "/tmp/prompt-501.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []

    state = load_state(paths.state_file)
    assert [e for e in state["events"] if e["kind"] == "session_failed_relabeled"] == []
    # No redispatch_at bookkeeping should have been written at all for this
    # issue -- the escalation-cap counter must not spend down on an issue
    # that needed no automatic recovery.
    entry = state["issues"].get("501", {})
    assert entry.get("redispatch_at") is None

    # The sidecar is still reaped -- issue #113 phantom-session protection is
    # unrelated to whether there was anything to relabel.
    assert not sidecar_path.exists()


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
    # Ensure PRs passed into loop tests are treated as open even if callers omit state.
    for pr in prs:
        pr.setdefault("state", "OPEN")
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


def test_fleet_lock_serializes_cross_repo_dispatch(tmp_path: Path, monkeypatch) -> None:
    """Independent dispatch() calls across two repos sharing a fleet cap cannot
    over-dispatch. Without the fleet lock, both repos could read a stale live
    count of 0 and dispatch up to the cap each, oversubscribing the fleet.
    """
    from charlie_work.adapters import SessionDispatchResult
    from charlie_work.devin_shell import _sidecar_path

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    shared_fleet_dir = str(fleet_dir)

    # Fake a non-blocking devin-shell launch that writes a sidecar so the
    # fleet-wide live count is visible to subsequent dispatchers.
    def fake_run_devin_shell(
        repo_root: Path,
        request: Any,
        sessions_dir: Path,
        settings: Any,
    ) -> SessionDispatchResult:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        record = SessionRecord(
            issue_number=request.issue_number,
            branch=request.branch_name,
            worktree_path=str(repo_root),
            prompt_path=str(request.prompt_path),
            command=("devin",),
            pid=9999,
            started_at="2024-01-01T00:00:00Z",
            log_path=str(sessions_dir / f"issue-{request.issue_number}.log"),
            error=None,
            process_start_time=1.0,
        )
        _sidecar_path(sessions_dir, request.issue_number).write_text(
            json.dumps(record.to_dict()), encoding="utf-8"
        )
        return SessionDispatchResult(
            issue_number=request.issue_number,
            issue_title=request.issue_title,
            prompt_path=str(request.prompt_path),
            branch_name=request.branch_name,
            adapter="devin-shell",
            ok=True,
            command=list(record.command),
            pid=record.pid,
            process_start_time=record.process_start_time,
        )

    monkeypatch.setattr("charlie_work.adapters._run_devin_shell_adapter", fake_run_devin_shell)
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda _record: True)

    # Build two independent repos, each sharing the same fleet directory.
    apps: list[OrchestratorApp] = []
    repo_entries: dict[str, dict[str, str]] = {}
    for repo_name in ("owner/repo-a", "owner/repo-b"):
        repo_root = tmp_path / repo_name.replace("/", "--")
        repo_root.mkdir()
        (repo_root / ".git").mkdir()
        paths = runtime_paths(repo_root, ".var/charlie-work")
        repo_entries[repo_name] = {
            "repo_root": str(repo_root),
            "state_dir": str(paths.root),
        }

        config = OrchestratorConfig(
            devin=DevinConfig(adapter="devin-shell"),
            dispatch=DispatchConfig(
                default_limit=3,
                launch_stagger_seconds=0,
            ),
            fleet=FleetConfig(global_max_concurrent_sessions=2),
        )
        fake_gh = FakeGitHub()
        fake_gh.prs = []
        fake_gh.issues = [
            {
                "number": 100 + i,
                "title": f"Issue {i}",
                "url": f"https://example.test/issues/{100 + i}",
                "body": "",
                "labels": [{"name": "automated-ready"}],
                "state": "OPEN",
            }
            for i in range(3)
        ]
        app = OrchestratorApp(
            repo_root,
            paths,
            config,
            fake_gh,
            fleet_dir_override=shared_fleet_dir,
        )
        apps.append(app)

    # Seed the fleet registry so both repos are visible to count_fleet_live_sessions.
    save_state(fleet_dir / "fleet.json", {"repos": repo_entries})

    # Launch both dispatch() calls concurrently from the same barrier.
    barrier = threading.Barrier(2)
    results: list[CommandResult] = []

    def worker(app: OrchestratorApp) -> None:
        barrier.wait(timeout=5)
        results.append(app.dispatch())

    threads = [threading.Thread(target=worker, args=(app,)) for app in apps]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    total_dispatched = sum(r.data.get("selected_count", 0) for r in results)
    # With a fleet cap of 2 and 3 ready issues in each repo, the combined
    # dispatch across both repos must never exceed the fleet-wide cap.
    assert total_dispatched <= 2, (
        f"fleet-wide dispatch over-subscribed: {total_dispatched} workers launched"
    )
    # Exactly one path should have succeeded; the other either clamped to 0 or
    # deferred because the fleet lock was held.
    assert any(r.data.get("fleet_live_session_count") is not None for r in results), (
        "fleet live count should be reported in dispatch results"
    )


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

    config = OrchestratorConfig(devin=DevinConfig(adapter="devin-shell"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)

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
    assert result.data.get("skipped") is True or result.data.get("state_lock_busy") is True
    assert state_path.stat().st_mtime == initial_mtime
    assert state_path.read_text(encoding="utf-8") == initial_content


def test_spec_review_state_lock_guard_returns_skip_when_lock_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #398: spec_review is also guarded by the state-lock skip pattern."""
    from charlie_work.cross_family import CrossFamilyResult

    monkeypatch.setattr(state_module, "_LOCK_TIMEOUT_SECONDS", 0.05)

    config = OrchestratorConfig(devin=DevinConfig(adapter="devin-shell"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)

    spec_path = tmp_path / "spec.md"
    spec_path.write_text("# spec\n", encoding="utf-8")

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

    def fake_run_cross_family_review(*args: object, **kwargs: object) -> CrossFamilyResult:
        return CrossFamilyResult(ok=True, report_path=str(tmp_path / "report.md"), model="test")

    monkeypatch.setattr(
        "charlie_work.cross_family.run_cross_family_review",
        fake_run_cross_family_review,
    )

    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    with _hold_state_lock(lock_path):
        result = app.spec_review(spec_path)

    assert result.ok is True
    reason = result.data.get("reason") or result.data.get("deferred_reason")
    assert reason in {"state_lock_busy", "supervisor_lock_held", "graphql_rate_limit"}
    assert result.data.get("skipped") is True or result.data.get("state_lock_busy") is True
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


def test_orphaned_worker_routes_merge_conflict_to_rework(tmp_path: Path) -> None:
    """Issue #439: a dead worker with a CONFLICTING open PR is routed to rework."""
    from datetime import UTC, datetime

    from charlie_work.config import AutoMergeConfig
    from charlie_work.state import load_state, save_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(required_checks=("Tests passed", "Lint & Format", "Pre-commit"))
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub(repo_root=tmp_path)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    fake_gh.prs = [
        {
            "number": 1,
            "title": "Fix #42: search",
            "url": "https://example.test/pull/1",
            "headRefName": "agent/issue-42-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeable": "CONFLICTING",
            "mergeStateStatus": "DIRTY",
            "body": "Closes #42",
            "state": "OPEN",
            "labels": [],
            "isCrossRepository": False,
            "updatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "statusCheckRollup": [],
        }
    ]

    state = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "issues": {
            "42": {
                "number": 42,
                "status": "dispatched",
                "worker_pid": 9999999,
                "worker_process_start_time": 1234567890.0,
                "redispatch_at": [],
            }
        },
        "prs": {},
        "events": [],
    }
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    state = load_state(paths.state_file)
    assert state["issues"]["42"]["status"] == "rework_requested"
    assert state["issues"]["42"]["pre_review_rework_reason"] == "merge_conflict"
    assert state["issues"]["42"]["worker_pid"] == 9999999
    assert state["issues"]["42"]["worker_process_start_time"] == 1234567890.0
    assert state["prs"]["1"]["status"] == "rework_requested"
    assert (42, config.labels.needs_rework) in fake_gh.labels_added
    assert (42, config.labels.in_progress) in fake_gh.labels_removed

    prompt_path = paths.prs / "pr-1" / "rework-prompt.md"
    assert prompt_path.exists()
    assert "merge conflict" in prompt_path.read_text(encoding="utf-8").lower()


def test_orphaned_worker_routes_stale_empty_checks_to_rework(tmp_path: Path) -> None:
    """Issue #439: a dead worker with an old PR and empty statusCheckRollup is routed to rework."""
    from datetime import UTC, datetime, timedelta

    from charlie_work.config import AutoMergeConfig
    from charlie_work.state import load_state, save_state
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(required_checks=("Tests passed", "Lint & Format", "Pre-commit"))
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub(repo_root=tmp_path)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    old_updated = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    fake_gh.prs = [
        {
            "number": 1,
            "title": "Fix #42: search",
            "url": "https://example.test/pull/1",
            "headRefName": "agent/issue-42-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #42",
            "state": "OPEN",
            "labels": [],
            "isCrossRepository": False,
            "updatedAt": old_updated,
            "statusCheckRollup": [],
        }
    ]

    state = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "issues": {
            "42": {
                "number": 42,
                "status": "dispatched",
                "worker_pid": 9999999,
                "worker_process_start_time": 1234567890.0,
                "redispatch_at": [],
            }
        },
        "prs": {},
        "events": [],
    }
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    _detect_and_handle_orphaned_workers(sessions_dir, paths.state_file, config, fake_gh)

    state = load_state(paths.state_file)
    assert state["issues"]["42"]["status"] == "rework_requested"
    assert state["issues"]["42"]["pre_review_rework_reason"] == "stale_empty_checks"
    assert state["issues"]["42"]["worker_pid"] == 9999999
    assert state["issues"]["42"]["worker_process_start_time"] == 1234567890.0
    assert state["prs"]["1"]["status"] == "rework_requested"
    assert (42, config.labels.needs_rework) in fake_gh.labels_added
    assert (42, config.labels.in_progress) in fake_gh.labels_removed

    prompt_path = paths.prs / "pr-1" / "rework-prompt.md"
    assert prompt_path.exists()
    prompt_text = prompt_path.read_text(encoding="utf-8").lower()
    assert "no ci checks" in prompt_text


def test_dispatch_label_error_reason_in_event_payload(tmp_path: Path) -> None:
    """Issue #453: dispatch label transition failures must carry a reason in the failures map."""
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

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            return False

    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.gh.prs[0]["state"] = "CLOSED"

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert 123 in result.data["label_errors"]
    assert 123 in result.data["failures"]
    reason = result.data["failures"][123]
    assert "label transition" in reason
    assert "dispatched" in reason
    assert "partial_failure" in reason

    state = load_state(paths.state_file)
    dispatch_events = [e for e in state["events"] if e["kind"] == "dispatch"]
    assert dispatch_events
    payload = dispatch_events[-1]["payload"]
    assert "123" in payload["failures"]
    assert payload["failures"]["123"] == reason


def test_dispatch_rework_label_error_reason_in_event_payload(tmp_path: Path) -> None:
    """Issue #453: rework dispatch label transition failures must carry a reason in the failures map."""
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


def test_dispatch_failed_retries_are_capped_and_escalate(tmp_path: Path) -> None:
    """Issue #461: repeated dispatch failures are capped and then escalated."""
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; sys.exit(7)"),
        ),
        watchdog=WatchdogConfig(max_auto_redispatch=1),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Avoid the open-PR exclusion by closing the default fixture PR.
    fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # First dispatch failure is recorded normally.
    result1 = app.dispatch(limit=1)
    assert result1.ok is False
    assert result1.data["failed_count"] == 1
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatch_failed"
    assert len(state["issues"]["123"]["dispatch_failed_at"]) == 1

    # Second failure exceeds the cap and escalates the issue.
    result2 = app.dispatch(limit=1)
    assert result2.ok is False
    assert result2.data["failed_count"] == 1
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "dispatch_failed_cap_exceeded"
    assert len(state["issues"]["123"]["dispatch_failed_at"]) == 2
    assert (123, "agent:human-needed") in fake_gh.labels_added

    # Third dispatch no longer selects the escalated issue.
    result3 = app.dispatch(limit=1)
    assert result3.ok is True
    assert result3.data["selected_count"] == 0
