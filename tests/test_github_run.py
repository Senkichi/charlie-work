"""``GitHub.run()`` contract tests: retry, timeout, empty-stdout, and error
classification -- plus the thin-wrapper propagation proofs that depend on
run() raising.

Split out of ``tests/test_github.py`` (issue #1572, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_github_fixtures.py``.
"""

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


def _empty_stdout_success(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")


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


def test_run_read_command_timeout_retries_then_succeeds(monkeypatch, tmp_path: Path) -> None:
    """A read command that times out once is retried transparently, the same
    as any other transient failure."""
    call_count = 0
    sleeps: list[float] = []

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout='[{"number": 1}]', stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    gh = github_module.GitHub(
        tmp_path,
        runtime=RuntimeConfig(gh_max_retries=3, gh_retry_base_seconds=1.0),
    )
    result = gh.run(_issue_list_args(), json_output=True)

    assert result == [{"number": 1}]
    assert call_count == 2
    assert len(sleeps) == 1


def test_run_read_command_timeout_exhausts_retries_raises(monkeypatch, tmp_path: Path) -> None:
    """A read command that always times out retries up to gh_max_retries and
    then raises GitHubError."""
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = github_module.GitHub(
        tmp_path,
        runtime=RuntimeConfig(gh_max_retries=2, gh_retry_base_seconds=1.0),
    )
    with pytest.raises(github_module.GitHubError, match="timed out"):
        gh.run(_issue_list_args(), json_output=True)

    assert call_count == 3


def test_run_mutating_command_timeout_not_retried(monkeypatch, tmp_path: Path) -> None:
    """A mutating command that times out is NOT retried, even though retries
    remain — retrying risks double-applying a mutation (double merge, double
    label write) because a timeout is not evidence the request never reached
    GitHub. Exactly one attempt is made before GitHubError is raised."""
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = github_module.GitHub(tmp_path, runtime=RuntimeConfig(gh_max_retries=3))
    with pytest.raises(github_module.GitHubError, match="timed out"):
        gh.run(["pr", "merge", "123", "--squash"])

    assert call_count == 1


def test_run_mutating_command_timeout_allow_failure_returns_error_result(
    monkeypatch, tmp_path: Path
) -> None:
    """allow_failure=True on a timed-out mutating command returns a structured
    error result — ok=False, returncode=124 (never 0, which callers read as
    success) — after exactly one attempt, no retry."""
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = github_module.GitHub(tmp_path, runtime=RuntimeConfig(gh_max_retries=3))
    result = gh.run(["pr", "merge", "123", "--squash"], allow_failure=True)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.returncode == 124
    assert "timed out" in (result.error or "")
    assert call_count == 1


def test_run_read_command_timeout_allow_failure_terminal_returns_124(
    monkeypatch, tmp_path: Path
) -> None:
    """allow_failure=True on a read command that always times out returns a
    terminal error result once retries are exhausted, with returncode=124 —
    not 0, which callers would misread as success."""
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = github_module.GitHub(
        tmp_path,
        runtime=RuntimeConfig(gh_max_retries=1, gh_retry_base_seconds=1.0),
    )
    result = gh.run(_issue_list_args(), json_output=True, allow_failure=True)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.returncode == 124
    assert result.returncode != 0
    assert "timed out" in (result.error or "")
    assert call_count == 2


def test_run_file_not_found_raises_github_error(monkeypatch, tmp_path: Path) -> None:
    """Pre-existing behavior unchanged by the timeout fix: a missing `gh`
    binary raises GitHubError, not GitHubError-via-timeout-path."""

    def fake_run(cmd, *args, **kwargs):
        raise FileNotFoundError("gh not found")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    with pytest.raises(github_module.GitHubError, match="not installed"):
        gh.run(_issue_list_args(), json_output=True)


def test_run_file_not_found_allow_failure_returns_error_result(
    monkeypatch, tmp_path: Path
) -> None:
    """Pre-existing behavior unchanged: allow_failure=True on a missing `gh`
    binary returns a structured error result with returncode=0 (distinct from
    the timeout path's returncode=124)."""

    def fake_run(cmd, *args, **kwargs):
        raise FileNotFoundError("gh not found")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.run(_issue_list_args(), json_output=True, allow_failure=True)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.returncode == 0
    assert "not installed" in (result.error or "")


def test_run_passes_configured_gh_timeout_seconds_to_subprocess_run(
    monkeypatch, tmp_path: Path
) -> None:
    """The configured gh_timeout_seconds reaches subprocess.run as the
    `timeout=` kwarg — not the module default."""
    captured_timeouts: list[float] = []

    def fake_run(cmd, *args, **kwargs):
        captured_timeouts.append(kwargs.get("timeout"))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path, runtime=RuntimeConfig(gh_timeout_seconds=7.5))
    gh.run(_issue_list_args(), json_output=True)

    assert captured_timeouts == [7.5]


def test_run_raises_on_empty_stdout_success_not_none(monkeypatch, tmp_path: Path) -> None:
    """Issue #756: gh exiting 0 with empty stdout under json_output=True and
    allow_failure=False (the default) must raise GitHubError, not return None.

    Callers throughout the codebase coerce a non-list/non-dict result with
    ``result if isinstance(result, X) else DEFAULT`` -- a bare ``None`` return
    silently reads as "the response was empty" (DEFAULT) rather than "the
    response was unreadable". This is the boundary-level fix: GitHub.run()
    itself no longer returns None for this ambiguous case.
    """

    def fake_run(cmd, *args, **kwargs):
        return _empty_stdout_success(cmd)

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    with pytest.raises(github_module.GitHubError):
        gh.run(["issue", "list", "--json", "number"], json_output=True)


def test_run_returns_empty_list_for_genuine_empty_json_array(monkeypatch, tmp_path: Path) -> None:
    """Positive control for test_run_raises_on_empty_stdout_success_not_none:
    a genuinely empty result (stdout is the JSON array "[]", not empty stdout)
    must still parse cleanly and must NOT raise."""

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.run(["issue", "list", "--json", "number"], json_output=True)

    assert result == []


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("issue_list", ()),
        ("pr_list", ()),
        pytest.param("issue_view", (123,), id="issue_view"),
        pytest.param("pr_view", (123,), id="pr_view"),
        ("label_list", ()),
    ],
)
def test_wrapper_method_raises_on_unreadable_empty_stdout(
    monkeypatch, tmp_path: Path, method_name: str, args: tuple
) -> None:
    """Issue #756: issue_list/pr_list/issue_view/pr_view/label_list must not
    silently coerce an unreadable (empty-stdout, gh exit 0) response to []/{}
    -- that reads "I could not read GitHub" as "GitHub has zero items", which
    is the exact bug this issue fixes. These wrapper methods rely on
    GitHub.run() raising GitHubError for this case; this test proves the
    raise actually propagates all the way out to the caller.
    """

    def fake_run(cmd, *a, **kwargs):
        return _empty_stdout_success(cmd)

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    method = getattr(gh, method_name)
    with pytest.raises(github_module.GitHubError):
        method(*args)


@pytest.mark.parametrize(
    ("method_name", "args", "empty_json_stdout", "expected"),
    [
        ("issue_list", (), "[]", []),
        ("pr_list", (), "[]", []),
        pytest.param("issue_view", (123,), "{}", {}, id="issue_view"),
        pytest.param("pr_view", (123,), "{}", {}, id="pr_view"),
        ("label_list", (), "[]", []),
    ],
)
def test_wrapper_method_returns_empty_for_genuine_empty_json(
    monkeypatch,
    tmp_path: Path,
    method_name: str,
    args: tuple,
    empty_json_stdout: str,
    expected: object,
) -> None:
    """Positive control for test_wrapper_method_raises_on_unreadable_empty_stdout:
    a genuinely empty JSON response ("[]" or "{}", not empty stdout) must still
    return the empty container cleanly, without raising."""

    def fake_run(cmd, *a, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=empty_json_stdout, stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    method = getattr(gh, method_name)
    result = method(*args)

    assert result == expected


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
