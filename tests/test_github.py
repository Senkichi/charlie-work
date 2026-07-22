"""Regression tests for GitHub.run() bounded retry of transient API failures."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from charlie_work import github as github_module
from charlie_work.config import RuntimeConfig


def _issue_list_args() -> list[str]:
    return [
        "issue",
        "list",
        "--state",
        "open",
        "--limit",
        "10",
        "--json",
        github_module.ISSUE_LIST_FIELDS,
    ]


def test_run_retries_transient_read_failure_then_succeeds(monkeypatch, tmp_path: Path) -> None:
    """A read command that fails twice with a TLS handshake timeout then succeeds
    is retried transparently and returns the parsed JSON value."""
    call_count = 0
    sleeps: list[float] = []

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr='Post "https://api.github.com/graphql": net/http: TLS handshake timeout',
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='[{"number": 1}]',
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    gh = github_module.GitHub(
        tmp_path,
        runtime=RuntimeConfig(gh_max_retries=3, gh_retry_base_seconds=1.0),
    )
    result = gh.run(_issue_list_args(), json_output=True)

    assert result == [{"number": 1}]
    assert call_count == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize(
    "stderr",
    [
        "Bad credentials",
        "Could not resolve to a Issue",
        "HTTP 422: Validation Failed",
    ],
    ids=["bad_credentials", "not_found", "validation"],
)
def test_run_terminal_error_fails_fast_no_retry(stderr: str, monkeypatch, tmp_path: Path) -> None:
    """Terminal errors raise GitHubError immediately and are never retried."""
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr=stderr)

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    with pytest.raises(github_module.GitHubError):
        gh.run(["issue", "view", "123"], json_output=True)

    assert call_count == 1


@pytest.mark.parametrize(
    "stderr",
    [
        'Post "https://api.github.com/graphql": i/o timeout',
        "HTTP 502: Bad Gateway",
    ],
    ids=["io_timeout", "gateway_502"],
)
def test_run_mutating_post_send_timeout_not_retried(
    stderr: str, monkeypatch, tmp_path: Path
) -> None:
    """Mutating commands are not retried on post-send ambiguous failures,
    preserving at-most-once semantics for merges/label writes."""
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr=stderr)

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path, runtime=RuntimeConfig(gh_max_retries=3))
    with pytest.raises(github_module.GitHubError):
        gh.run(["pr", "merge", "123", "--squash"])

    assert call_count == 1


def test_run_mutating_pre_connection_failure_retried(monkeypatch, tmp_path: Path) -> None:
    """Mutating commands are retried on provable pre-connection failures."""
    call_count = 0
    sleeps: list[float] = []

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="dial tcp: connect connection refused",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="merged #123", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    gh = github_module.GitHub(tmp_path, runtime=RuntimeConfig(gh_max_retries=3))
    result = gh.run(["pr", "merge", "123", "--squash"])

    assert result == "merged #123"
    assert call_count == 3
    assert len(sleeps) == 2


def test_run_retry_backoff_is_bounded_and_grows(monkeypatch, tmp_path: Path) -> None:
    """After gh_max_retries transient failures the error surfaces, and the
    injected sleep intervals grow exponentially."""
    call_count = 0
    sleeps: list[float] = []

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr='Post "https://api.github.com/graphql": net/http: TLS handshake timeout',
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(github_module.random, "uniform", lambda a, b: 0.0)

    gh = github_module.GitHub(
        tmp_path,
        runtime=RuntimeConfig(gh_max_retries=2, gh_retry_base_seconds=1.0),
    )
    with pytest.raises(github_module.GitHubError):
        gh.run(_issue_list_args(), json_output=True)

    assert call_count == 3
    assert sleeps == [1.0, 2.0]


def test_run_allow_failure_retries_then_returns_error_result(monkeypatch, tmp_path: Path) -> None:
    """allow_failure=True still retries transient errors and returns a structured
    error result once retries are exhausted."""
    call_count = 0
    sleeps: list[float] = []

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr='Post "https://api.github.com/graphql": net/http: TLS handshake timeout',
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    gh = github_module.GitHub(
        tmp_path,
        runtime=RuntimeConfig(gh_max_retries=1, gh_retry_base_seconds=1.0),
    )
    result = gh.run(_issue_list_args(), json_output=True, allow_failure=True)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.returncode == 1
    assert "TLS handshake timeout" in (result.error or "")
    assert call_count == 2
    assert len(sleeps) == 1


def test_run_add_issue_label_retries_pre_connection_then_succeeds(
    monkeypatch, tmp_path: Path
) -> None:
    """Label edits (mutating) are retried on pre-connection failures and return
    boolean success once the connection succeeds."""
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="error connecting to api.github.com: connection refused",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = github_module.GitHub(tmp_path, runtime=RuntimeConfig(gh_max_retries=3))
    assert gh.add_issue_label(123, "agent:in-progress") is True
    assert call_count == 2


def test_pr_checks_injects_run_id(monkeypatch, tmp_path: Path) -> None:
    """Issue #391: pr_checks derives the GitHub Actions workflow run id from link."""

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='[{"name":"Tests passed","state":"FAILURE","link":"https://github.com/owner/repo/actions/runs/29525590823/job/87713099471"}]',
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    checks = gh.pr_checks(456)

    assert checks == [
        {
            "name": "Tests passed",
            "state": "FAILURE",
            "link": "https://github.com/owner/repo/actions/runs/29525590823/job/87713099471",
            "databaseId": 87713099471,
            "runId": 29525590823,
        }
    ]


_FIXTURES = Path(__file__).parent / "fixtures"


def _read_fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def test_check_graphql_rate_limit_parses_live_payload(monkeypatch, tmp_path: Path) -> None:
    """Issue #398: the rate-limit guard must parse a live ``gh api rate_limit`` payload."""
    rate_limit_json = _read_fixture("gh_rate_limit.json")

    def fake_run(cmd, *args, **kwargs):
        assert cmd[:3] == ["gh", "api", "rate_limit"]
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=rate_limit_json, stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    sufficient, remaining, reset_at = gh.check_graphql_rate_limit(threshold=1500)

    # Fixture has graphql.remaining == 4114, reset is a unix timestamp.
    assert sufficient is True
    assert remaining == 4114
    assert isinstance(reset_at, int)


def test_check_graphql_rate_limit_below_threshold_returns_insufficient(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #398: when remaining points are below the threshold the guard reports insufficient."""
    rate_limit_json = _read_fixture("gh_rate_limit.json")

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=rate_limit_json, stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    sufficient, remaining, reset_at = gh.check_graphql_rate_limit(threshold=5000)

    assert sufficient is False
    assert remaining == 4114
    assert isinstance(reset_at, int)


def test_merged_pr_list_uses_rest_pagination_and_filters_merged(
    monkeypatch, tmp_path: Path
) -> None:
    """merged_pr_list() now paginates through the REST pulls endpoint and
    filters to merged PRs, avoiding the GraphQL query entirely.
    """
    page1 = [
        {
            "number": 1,
            "title": "x",
            "body": "",
            "head": {"ref": "agent/issue-1-x", "repo": {"full_name": "owner/repo"}},
            "base": {"repo": {"full_name": "owner/repo"}},
            "merged_at": "2026-07-21T20:00:00Z",
            "state": "closed",
        },
        {
            "number": 2,
            "title": "closed not merged",
            "body": "",
            "head": {"ref": "other", "repo": {"full_name": "owner/repo"}},
            "base": {"repo": {"full_name": "owner/repo"}},
            "merged_at": None,
            "state": "closed",
        },
    ]
    call_log: list[list[str]] = []
    pull_call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal pull_call_count
        call_log.append(cmd)
        if cmd[:2] == ["gh", "api"] and "pulls" in cmd[2]:
            pull_call_count += 1
            if pull_call_count == 1:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=json.dumps(page1), stderr=""
                )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.merged_pr_list()

    assert result == [
        {
            "number": 1,
            "title": "x",
            "body": "",
            "headRefName": "agent/issue-1-x",
            "isCrossRepository": False,
            "state": "MERGED",
        }
    ]
    assert pull_call_count >= 1
    assert any("pulls?state=closed" in c[2] for c in call_log)
    assert not any(c[:2] == ["gh", "pr"] for c in call_log)


def test_merged_pr_list_raises_on_rest_pagination_error(monkeypatch, tmp_path: Path) -> None:
    """A terminal REST failure during pagination raises GitHubError."""
    merged_pr = {
        "number": 1,
        "title": "x",
        "body": "",
        "head": {"ref": "agent/issue-1-x", "repo": {"full_name": "owner/repo"}},
        "base": {"repo": {"full_name": "owner/repo"}},
        "merged_at": "2026-07-21T20:00:00Z",
        "state": "closed",
    }
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps([merged_pr]), stderr=""
            )
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="HTTP 401: Bad credentials"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    with pytest.raises(github_module.GitHubError):
        gh.merged_pr_list()

    assert call_count == 2


def test_merged_prs_for_issue_returns_bound_pr_without_graphql_budget_check(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #433: per-issue merged-PR lookup bypasses the 500-window cap and does
    not consume the GraphQL budget check used by merged_pr_list().
    """
    search_json = _read_fixture("gh_pr_list_search_merged.json")
    call_log: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        call_log.append(cmd)
        if cmd[:3] == ["gh", "api", "rate_limit"]:
            # Should not be called; this method is intentionally budget-agnostic.
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="{}", stderr="")
        if cmd[:2] == ["gh", "pr"] and "--search" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=search_json, stderr=""
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.merged_prs_for_issue(326, branch_prefix="agent/issue")

    assert len(result) == 1
    assert result[0]["number"] == 335
    assert result[0]["state"] == "MERGED"
    search_calls = [c for c in call_log if c[:2] == ["gh", "pr"] and "--search" in c]
    assert len(search_calls) == 1
    assert search_calls[0] == [
        "gh",
        "pr",
        "list",
        "--state",
        "merged",
        "--search",
        '"#326"',
        "--limit",
        "20",
        "--json",
        github_module.MERGED_PR_LIST_FIELDS,
    ]
    assert not any(c[:3] == ["gh", "api", "rate_limit"] for c in call_log)


def test_merged_prs_for_issue_returns_empty_when_pr_is_not_bound(
    monkeypatch, tmp_path: Path
) -> None:
    """A merged PR that only mentions the issue (no branch prefix / closing keyword)
    must not be returned, even if the issue number appears in its title/body.
    """
    prs = [
        {
            "number": 1,
            "title": "chore: unrelated",
            "body": "While in the area, this also happens to fix issue #326.",
            "headRefName": "unrelated-cleanup",
            "isCrossRepository": False,
            "state": "MERGED",
        }
    ]

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["gh", "pr"] and "--search" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps(prs), stderr=""
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.merged_prs_for_issue(326, branch_prefix="agent/issue")

    assert result == []


def test_merged_prs_for_issue_returns_empty_on_gh_failure(monkeypatch, tmp_path: Path) -> None:
    """Per-issue lookup is best-effort; a non-zero gh exit returns an empty list."""

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["gh", "pr"] and "--search" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="HTTP 502: Bad gateway",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.merged_prs_for_issue(326, branch_prefix="agent/issue")

    assert result == []
    assert result.ok is False


def test_run_raises_not_found_error_for_graphql_could_not_resolve(
    monkeypatch, tmp_path: Path
) -> None:
    """A GraphQL could-not-resolve terminal error raises GitHubNotFoundError, a
    GitHubError subclass, so existing `except GitHubError` callers still catch it."""
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr=(
                "GraphQL: Could not resolve to an issue or pull request with the "
                "number of 1337. (repository.issue)"
            ),
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    with pytest.raises(github_module.GitHubNotFoundError) as exc_info:
        gh.run(["issue", "view", "1337"], json_output=True)

    assert isinstance(exc_info.value, github_module.GitHubError)
    assert call_count == 1


def test_run_raises_plain_github_error_for_unrelated_terminal_error(
    monkeypatch, tmp_path: Path
) -> None:
    """An unrelated terminal error raises plain GitHubError, not the not-found
    subclass — callers that only special-case not-found must not misclassify it."""

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="some fatal thing"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    with pytest.raises(github_module.GitHubError) as exc_info:
        gh.run(["issue", "view", "1"], json_output=True)

    assert not isinstance(exc_info.value, github_module.GitHubNotFoundError)


@pytest.mark.parametrize(
    "error, expected",
    [
        (
            "GraphQL: Could not resolve to an issue or pull request with the "
            "number of 1337. (repository.issue)",
            True,
        ),
        ("Not Found (HTTP 404)", True),
        ("NOT_FOUND", True),
        ("TLS handshake timeout", False),
        ("HTTP 403: Forbidden", False),
    ],
    ids=[
        "graphql_could_not_resolve",
        "rest_404",
        "not_found_token",
        "tls_timeout",
        "http_403",
    ],
)
def test_is_not_found_gh_error_classifies_correctly(error: str, expected: bool) -> None:
    assert github_module._is_not_found_gh_error(error) is expected
