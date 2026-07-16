"""Regression tests for GitHub.run() bounded retry of transient API failures."""

from __future__ import annotations

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
