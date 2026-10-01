"""``GitHub.run()`` contract tests: retry, timeout, empty-stdout, and error
classification -- plus the thin-wrapper propagation proofs that depend on
run() raising.

Split out of ``tests/test_github.py`` (issue #1572, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_github_fixtures.py``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from _fake_transport import (
    FakeAdapter,
    failure,
    gh_kill_switch_runtime,
    graphql_ok,
    make_github,
    ok,
)
from charlie_work import github as github_module
from charlie_work.config import RuntimeConfig
from charlie_work.github_transport import FailureKind, GraphQLRequest
from charlie_work.github_transport.gh_adapter import GhAdapter


_READ = ["api", "repos/{owner}/{repo}/issues"]
_WRITE = ["run", "rerun", "123"]  # a REST POST: the modelled mutation under gh
_TLS = 'Post "https://api.github.com/graphql": net/http: TLS handshake timeout'


def _proc(cmd, *, rc: int = 0, out: str = "", err: str = "") -> subprocess.CompletedProcess:
    """A ``gh api --include`` reply: successes carry a status line, as gh prints one."""
    prefix = "HTTP/2.0 200 OK\r\n\r\n" if rc == 0 and "--include" in cmd else ""
    return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout=prefix + out, stderr=err)


class _Spawn:
    """Scripted gh subprocess: ``replies`` in order (last repeats); an exception is raised."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, argv, stdin, cwd, timeout):
        self.calls.append((list(argv), timeout))
        item = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(item, BaseException):
            raise item
        return item(argv) if callable(item) else item


def _gh(tmp_path: Path, spawn: _Spawn, **runtime: object):
    """A real ``GitHub`` under the kill switch whose gh subprocess is *spawn*."""
    github, _http, _gh_adapter = make_github(
        tmp_path,
        gh=GhAdapter(tmp_path, spawn=spawn),
        runtime=gh_kill_switch_runtime(**runtime),
    )
    return github


def _fail(err: str, rc: int = 1):
    return lambda cmd: _proc(cmd, rc=rc, err=err)


def _timeout(cmd=None):
    return subprocess.TimeoutExpired(cmd=cmd or "gh", timeout=1.0)


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
    sleeps: list[float] = []
    spawn = _Spawn(
        _fail(_TLS),
        _fail(_TLS),
        lambda cmd: _proc(cmd, out='[{"number": 1}]'),
    )
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    gh = _gh(tmp_path, spawn, gh_max_retries=3, gh_retry_base_seconds=1.0)
    result = gh.run(_READ, json_output=True)

    assert result == [{"number": 1}]
    assert len(spawn.calls) == 3
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
    spawn = _Spawn(_fail(stderr))

    gh = _gh(tmp_path, spawn)
    with pytest.raises(github_module.GitHubError):
        gh.run(_READ, json_output=True)

    assert len(spawn.calls) == 1


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
    spawn = _Spawn(_fail(stderr))
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = _gh(tmp_path, spawn, gh_max_retries=3)
    with pytest.raises(github_module.GitHubError):
        gh.run(_WRITE)

    assert len(spawn.calls) == 1


def test_run_mutating_pre_connection_failure_retried(monkeypatch, tmp_path: Path) -> None:
    """Mutating commands are retried on provable pre-connection failures."""
    sleeps: list[float] = []
    refused = _fail("dial tcp: connect connection refused")
    spawn = _Spawn(refused, refused, lambda cmd: _proc(cmd, out="merged #123"))
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    gh = _gh(tmp_path, spawn, gh_max_retries=3)
    result = gh.run(_WRITE)

    assert result == "merged #123"
    assert len(spawn.calls) == 3
    assert len(sleeps) == 2


def test_run_retry_backoff_is_bounded_and_grows(monkeypatch, tmp_path: Path) -> None:
    """After gh_max_retries transient failures the error surfaces, and the
    injected sleep intervals grow exponentially."""
    sleeps: list[float] = []
    spawn = _Spawn(_fail(_TLS))
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(github_module.random, "uniform", lambda a, b: 0.0)

    gh = _gh(tmp_path, spawn, gh_max_retries=2, gh_retry_base_seconds=1.0)
    with pytest.raises(github_module.GitHubError):
        gh.run(_READ, json_output=True)

    assert len(spawn.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_run_allow_failure_retries_then_returns_error_result(monkeypatch, tmp_path: Path) -> None:
    """allow_failure=True still retries transient errors and returns a structured
    error result once retries are exhausted."""
    sleeps: list[float] = []
    spawn = _Spawn(_fail(_TLS))
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    gh = _gh(tmp_path, spawn, gh_max_retries=1, gh_retry_base_seconds=1.0)
    result = gh.run(_READ, json_output=True, allow_failure=True)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.returncode == 1
    assert "TLS handshake timeout" in (result.error or "")
    assert len(spawn.calls) == 2
    assert len(sleeps) == 1


def test_run_add_issue_label_retries_pre_connection_then_succeeds(
    monkeypatch, tmp_path: Path
) -> None:
    """Label edits (mutating) are retried on pre-connection failures and return
    boolean success once the connection succeeds."""
    http = FakeAdapter(
        "http", [failure(FailureKind.CONNECT, "connection refused"), ok([{"name": "x"}])]
    )
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    # A CONNECT failure falls back to gh for the same attempt; gh fails the same
    # way, so the guard's pre-connection retry (safe for mutations) kicks in.
    gh_adapter = FakeAdapter(
        "gh", [failure(FailureKind.CONNECT, "connection refused")], token="tok-1"
    )
    gh, http, _ = make_github(
        tmp_path, http=http, gh=gh_adapter, runtime=RuntimeConfig(gh_max_retries=3)
    )
    assert gh.add_issue_label(123, "agent:in-progress") is True
    assert len(http.calls) == 2
    assert len(gh_adapter.api_requests) == 1


def test_run_read_command_timeout_retries_then_succeeds(monkeypatch, tmp_path: Path) -> None:
    """A read command that times out once is retried transparently, the same
    as any other transient failure."""
    sleeps: list[float] = []
    spawn = _Spawn(_timeout(), lambda cmd: _proc(cmd, out='[{"number": 1}]'))
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    gh = _gh(tmp_path, spawn, gh_max_retries=3, gh_retry_base_seconds=1.0)
    result = gh.run(_READ, json_output=True)

    assert result == [{"number": 1}]
    assert len(spawn.calls) == 2
    assert len(sleeps) == 1


def test_run_read_command_timeout_exhausts_retries_raises(monkeypatch, tmp_path: Path) -> None:
    """A read command that always times out retries up to gh_max_retries and
    then raises GitHubError."""
    spawn = _Spawn(_timeout())
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = _gh(tmp_path, spawn, gh_max_retries=2, gh_retry_base_seconds=1.0)
    with pytest.raises(github_module.GitHubError, match="timed out"):
        gh.run(_READ, json_output=True)

    assert len(spawn.calls) == 3


def test_run_mutating_command_timeout_not_retried(monkeypatch, tmp_path: Path) -> None:
    """A mutating command that times out is NOT retried, even though retries
    remain -- retrying risks double-applying a mutation (double merge, double
    label write) because a timeout is not evidence the request never reached
    GitHub. Exactly one attempt is made before GitHubError is raised."""
    spawn = _Spawn(_timeout())
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = _gh(tmp_path, spawn, gh_max_retries=3)
    with pytest.raises(github_module.GitHubError, match="timed out"):
        gh.run(_WRITE)

    assert len(spawn.calls) == 1


def test_run_mutating_command_timeout_allow_failure_returns_error_result(
    monkeypatch, tmp_path: Path
) -> None:
    """allow_failure=True on a timed-out mutating command returns a structured
    error result -- ok=False, returncode=124 (never 0, which callers read as
    success) -- after exactly one attempt, no retry."""
    spawn = _Spawn(_timeout())
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = _gh(tmp_path, spawn, gh_max_retries=3)
    result = gh.run(_WRITE, allow_failure=True)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.returncode == 124
    assert "timed out" in (result.error or "")
    assert len(spawn.calls) == 1


def test_run_read_command_timeout_allow_failure_terminal_returns_124(
    monkeypatch, tmp_path: Path
) -> None:
    """allow_failure=True on a read command that always times out returns a
    terminal error result once retries are exhausted, with returncode=124 --
    not 0, which callers would misread as success."""
    spawn = _Spawn(_timeout())
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = _gh(tmp_path, spawn, gh_max_retries=1, gh_retry_base_seconds=1.0)
    result = gh.run(_READ, json_output=True, allow_failure=True)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.returncode == 124
    assert result.returncode != 0
    assert "timed out" in (result.error or "")
    assert len(spawn.calls) == 2


def test_run_file_not_found_raises_github_error(monkeypatch, tmp_path: Path) -> None:
    """Pre-existing behavior unchanged by the timeout fix: a missing `gh`
    binary raises GitHubError, not GitHubError-via-timeout-path."""
    gh = _gh(tmp_path, _Spawn(FileNotFoundError("gh not found")))
    with pytest.raises(github_module.GitHubError, match="not installed"):
        gh.run(_READ, json_output=True)


def test_run_file_not_found_allow_failure_returns_error_result(
    monkeypatch, tmp_path: Path
) -> None:
    """Pre-existing behavior unchanged: allow_failure=True on a missing `gh`
    binary returns a structured error result with returncode=0 (distinct from
    the timeout path's returncode=124)."""
    gh = _gh(tmp_path, _Spawn(FileNotFoundError("gh not found")))
    result = gh.run(_READ, json_output=True, allow_failure=True)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.returncode == 0
    assert "not installed" in (result.error or "")


def test_run_passes_configured_gh_timeout_seconds_to_subprocess_run(
    monkeypatch, tmp_path: Path
) -> None:
    """The configured gh_timeout_seconds reaches the gh subprocess as its
    timeout -- not the module default."""
    spawn = _Spawn(lambda cmd: _proc(cmd, out="[]"))

    gh = _gh(tmp_path, spawn, gh_timeout_seconds=7.5)
    gh.run(_READ, json_output=True)

    assert [timeout for _argv, timeout in spawn.calls] == [7.5]


def test_run_raises_on_empty_stdout_success_not_none(monkeypatch, tmp_path: Path) -> None:
    """Issue #756: gh exiting 0 with empty stdout under json_output=True and
    allow_failure=False (the default) must raise GitHubError, not return None.

    Callers throughout the codebase coerce a non-list/non-dict result with
    ``result if isinstance(result, X) else DEFAULT`` -- a bare ``None`` return
    silently reads as "the response was empty" (DEFAULT) rather than "the
    response was unreadable". This is the boundary-level fix: GitHub.run()
    itself no longer returns None for this ambiguous case.
    """
    gh = _gh(tmp_path, _Spawn(lambda cmd: _proc(cmd, out="")))
    with pytest.raises(github_module.GitHubError):
        gh.run(_READ, json_output=True)


def test_run_returns_empty_list_for_genuine_empty_json_array(monkeypatch, tmp_path: Path) -> None:
    """Positive control for test_run_raises_on_empty_stdout_success_not_none:
    a genuinely empty result (stdout is the JSON array "[]", not empty stdout)
    must still parse cleanly and must NOT raise."""
    gh = _gh(tmp_path, _Spawn(lambda cmd: _proc(cmd, out="[]")))

    assert gh.run(_READ, json_output=True) == []


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

    # Every request (REST or GraphQL) is answered 200 with an empty body.
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", handler=lambda req: ok("")))
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

    page = {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}}
    # The --json reads are GraphQL now: a genuinely empty result is an empty
    # connection (lists) or a node with no populated fields (views).
    graphql = {
        "issue_list": {"repository": {"issues": page}},
        "pr_list": {"repository": {"pullRequests": page}},
        "issue_view": {"repository": {"issue": {}}},
        "pr_view": {"repository": {"pullRequest": {}}},
    }

    def handler(request):
        if isinstance(request, GraphQLRequest):
            return graphql_ok(graphql[method_name])
        return ok(json.loads(empty_json_stdout))

    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", handler=handler))
    method = getattr(gh, method_name)
    result = method(*args)

    if isinstance(result, dict):  # a view of an empty node: every field unpopulated
        result = {k: v for k, v in result.items() if v not in (None, [], "")}
    assert result == expected


def test_run_raises_not_found_error_for_graphql_could_not_resolve(
    monkeypatch, tmp_path: Path
) -> None:
    """A GraphQL could-not-resolve terminal error raises GitHubNotFoundError, a
    GitHubError subclass, so existing `except GitHubError` callers still catch it."""
    not_found = (
        "GraphQL: Could not resolve to an issue or pull request with the "
        "number of 1337. (repository.issue)"
    )
    spawn = _Spawn(_fail(not_found))

    gh = _gh(tmp_path, spawn)
    with pytest.raises(github_module.GitHubNotFoundError) as exc_info:
        gh.run(_READ, json_output=True)

    assert isinstance(exc_info.value, github_module.GitHubError)
    assert len(spawn.calls) == 1


def test_run_raises_plain_github_error_for_unrelated_terminal_error(
    monkeypatch, tmp_path: Path
) -> None:
    """An unrelated terminal error raises plain GitHubError, not the not-found
    subclass -- callers that only special-case not-found must not misclassify it."""
    gh = _gh(tmp_path, _Spawn(_fail("some fatal thing")))
    with pytest.raises(github_module.GitHubError) as exc_info:
        gh.run(_READ, json_output=True)

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
