"""GitHub CI-check reads: pr_checks, check-run annotations, field-list validation.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
import pytest
from charlie_work import github as github_module
from charlie_work.config import (
    ConfigError,
    RuntimeConfig,
)


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
            "mergedAt",
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
