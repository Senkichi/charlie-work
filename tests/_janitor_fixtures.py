"""Shared helpers for the ``test_janitor_*`` sibling modules.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1): the config /
PR-dict / check-summary builders and the temp-git-repo initializer shared by
the janitor test siblings. A dedicated fixture module, not ``conftest.py``
(graft H).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from charlie_work.checks import CheckSummary
from charlie_work.config import (
    AutoMergeConfig,
    OrchestratorConfig,
    ReviewConfig,
    TestAdequacyConfig,
)


REQUIRED_CHECKS = ("Tests passed", "Lint & Format")


_STALE_REQUIRED = ("Tests passed", "Pre-commit")


def _init_repo(repo_root: Path) -> None:
    """Initialize a git repo with a single commit."""
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main"], cwd=repo_root, check=True, capture_output=True
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


def _config(**overrides) -> OrchestratorConfig:
    review = ReviewConfig(
        require_tests_or_rationale=overrides.pop("require_tests_or_rationale", True),
        require_issue_link=overrides.pop("require_issue_link", True),
    )
    auto_merge = AutoMergeConfig(required_checks=overrides.pop("required_checks", REQUIRED_CHECKS))
    assert not overrides, f"unused overrides: {overrides}"
    return OrchestratorConfig(review=review, auto_merge=auto_merge)


def _green_pr(**overrides) -> dict:
    base = {
        "number": 456,
        "title": "fix: search is broken",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": "abc123",
        "baseRefName": "main",
        "body": "Closes #123.\n\nTests: added unit tests for the search path.",
        "isDraft": False,
        "mergeable": "MERGEABLE",
        "additions": 10,
        "deletions": 5,
        "isCrossRepository": False,
    }
    base.update(overrides)
    return base


def _green_checks() -> list[dict]:
    return [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]


def _test_adequacy_config(**overrides) -> TestAdequacyConfig:
    """Build a TestAdequacyConfig with overrides for testing."""
    return TestAdequacyConfig(**overrides)


def _test_pr(**overrides) -> dict:
    """Build a minimal PR dict for test-adequacy testing."""
    base = {
        "number": 123,
        "body": "Closes #123.",
    }
    base.update(overrides)
    return base


def _stale_decision(required_changes: list) -> dict:
    return {
        "decision": "request_changes",
        "escalated": False,
        "required_changes": required_changes,
    }


def _all_green_summary(required: tuple = _STALE_REQUIRED) -> CheckSummary:
    return CheckSummary(
        required=required,
        passed=required,
        pending=(),
        failed=(),
        missing=(),
        infra_failed=(),
        unavailable=(),
    )
