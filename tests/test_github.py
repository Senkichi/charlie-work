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


def test_pr_checks_zero_checks_returns_empty_list_not_none(monkeypatch, tmp_path: Path) -> None:
    """Issue #846: a PR with zero checks must return [], not None.

    `gh pr checks` exits non-zero with "no checks reported" and no JSON when a
    PR has no checks yet -- a shape indistinguishable from a genuine command
    failure using result.ok/result.value alone (measured against PR #700 in
    this repo). pr_checks must disambiguate via the statusCheckRollup fallback
    and return [] here, not treat it as an infrastructure failure.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        if cmd[1:3] == ["pr", "checks"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="no checks reported on the 'agent/issue-627-fix' branch",
            )
        assert cmd[1:3] == ["pr", "view"], cmd
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='{"statusCheckRollup":[]}',
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    checks = gh.pr_checks(700)

    assert checks == []
    # Exactly one disambiguating fallback call, not a retry storm.
    assert len(calls) == 2


def test_pr_checks_genuine_failure_returns_none(monkeypatch, tmp_path: Path) -> None:
    """Issue #846: pr_checks must still return None for a real outage.

    When `gh pr checks` fails AND the statusCheckRollup disambiguation
    fallback also fails, pr_checks must preserve its existing "unavailable"
    contract (None) rather than silently mask the outage as "no checks".
    """

    def fake_run(cmd, *args, **kwargs):
        if cmd[1:3] == ["pr", "checks"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr='Unknown JSON field: "bucket"',
            )
        assert cmd[1:3] == ["pr", "view"], cmd
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="GraphQL: Could not resolve to a PullRequest with the number of 700.",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    checks = gh.pr_checks(700)

    assert checks is None


def test_pr_checks_fallback_maps_transient_glitch_rollup(monkeypatch, tmp_path: Path) -> None:
    """Issue #846: a transient `gh pr checks` glitch must not lose real checks.

    If `gh pr checks` fails but the PR genuinely has checks, the
    statusCheckRollup fallback maps them into the same shape pr_checks
    normally returns -- including the databaseId/runId injection every
    consumer (checks.py, janitor.py, workflow.py) relies on -- rather than
    collapsing them to None.
    """

    def fake_run(cmd, *args, **kwargs):
        if cmd[1:3] == ["pr", "checks"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="unexpected end of JSON input",
            )
        assert cmd[1:3] == ["pr", "view"], cmd
        rollup = {
            "statusCheckRollup": [
                {
                    "__typename": "CheckRun",
                    "name": "Tests",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                    "detailsUrl": "https://github.com/owner/repo/actions/runs/111/job/222",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Lint",
                    "status": "IN_PROGRESS",
                    "conclusion": "",
                    "detailsUrl": "https://github.com/owner/repo/actions/runs/111/job/333",
                },
            ]
        }
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps(rollup), stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    checks = gh.pr_checks(679)

    assert checks == [
        {
            "name": "Tests",
            "state": "SUCCESS",
            "link": "https://github.com/owner/repo/actions/runs/111/job/222",
            "databaseId": 222,
            "runId": 111,
        },
        {
            "name": "Lint",
            "state": "IN_PROGRESS",
            "link": "https://github.com/owner/repo/actions/runs/111/job/333",
            "databaseId": 333,
            "runId": 111,
        },
    ]


def test_pr_checks_fallback_declines_non_checkrun_rollup_entry(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #846: an unrecognized rollup entry shape must not be guessed at.

    A `statusCheckRollup` entry that isn't a GitHub Actions `CheckRun` (e.g. an
    external `StatusContext`) has no field mapping verified against this repo.
    Rather than fabricate a lossy mapping, pr_checks declines and returns None
    -- the same "unavailable" outcome as before issue #846's fix, never worse.
    """

    def fake_run(cmd, *args, **kwargs):
        if cmd[1:3] == ["pr", "checks"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="no checks reported on the 'x' branch",
            )
        assert cmd[1:3] == ["pr", "view"], cmd
        rollup = {"statusCheckRollup": [{"__typename": "StatusContext", "state": "SUCCESS"}]}
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps(rollup), stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    checks = gh.pr_checks(700)

    assert checks is None


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
            "head": {
                "ref": "agent/issue-1-x",
                "sha": "aaaa1111",
                "repo": {"full_name": "owner/repo"},
            },
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
            "headRefOid": "aaaa1111",
            "mergeCommitOid": None,
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


def test_merged_pr_list_raises_on_empty_stdout_not_silent_empty(
    monkeypatch, tmp_path: Path
) -> None:
    """gh exiting 0 with empty stdout is an unusable response, not an empty page.

    As of issue #756, run() itself raises GitHubError for that case (the
    ambiguous success-with-empty-stdout path). Before #756 it returned None,
    which merged_pr_list's own isinstance check also raised on — this test
    is unaffected by where the raise originates. A genuine empty page comes
    back as the JSON array ``[]`` (a list). The previous idiom
    ``result if isinstance(result, list) else []`` coerced None to [] and
    silently broke the pagination loop, returning [] as though the repository
    had no merged PRs — indistinguishable from a successful empty fetch. This
    is the silent-empty path that would arm the #502 post-merge tripwire with
    an empty baseline and leave it permanently blind (issue #633).
    """
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        # gh exits 0 with empty stdout — run() now raises GitHubError directly.
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    with pytest.raises(github_module.GitHubError):
        gh.merged_pr_list()

    # The first page is where the unusable response is detected.
    assert call_count == 1


def test_merged_pr_list_empty_page_terminates_cleanly(monkeypatch, tmp_path: Path) -> None:
    """A genuine empty page (``[]``) terminates pagination without raising.

    This is the positive counterpart to test_merged_pr_list_raises_on_empty_stdout:
    a real empty page is a list (``[]``) and must keep being treated as "no more
    results", not as an unusable response.
    """
    responses = ["[]"]

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=responses.pop(0), stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.merged_pr_list()

    assert result == []


def _empty_stdout_success(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")


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


def test_compare_diff_hits_three_dot_compare_with_diff_media_type(
    monkeypatch, tmp_path: Path
) -> None:
    """compare_diff must call the three-dot compare endpoint with the diff
    media type Accept header (not the default JSON compare metadata), and
    return the raw response body unwrapped from GitHubRunResult."""
    seen_cmd: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        seen_cmd.extend(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new\n",
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.compare_diff("sha-old", "sha-new")

    assert seen_cmd[:2] == ["gh", "api"]
    assert seen_cmd[2] == "repos/{owner}/{repo}/compare/sha-old...sha-new"
    assert "-H" in seen_cmd
    h_idx = seen_cmd.index("-H")
    assert seen_cmd[h_idx + 1] == "Accept: application/vnd.github.v3.diff"
    assert result == "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new"


def test_compare_diff_returns_none_on_failure(monkeypatch, tmp_path: Path) -> None:
    """A failed compare (404, GC'd SHA, API error) must return None, never
    raise — errors are returned as values, per the GitHub wrapper's pattern."""

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="HTTP 404: Not Found",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path, runtime=RuntimeConfig(gh_max_retries=0))
    result = gh.compare_diff("sha-old", "sha-new")

    assert result is None


# --- merged-PR field contract: the REST normalizer must reproduce exactly the
# key set that merged_prs_for_issue() gets from `gh pr list --json`. These are
# two independent producers of the same value shape; when they drift, consumers
# reading a field the normalizer forgot silently see None on the REST path
# (which is the only path merged_pr_list() uses) while every FakeGitHub-based
# test keeps passing, because those fixtures hand-write the richer shape.


def test_normalize_rest_pr_satisfies_merged_pr_list_field_contract() -> None:
    """_normalize_rest_pr() must emit exactly MERGED_PR_LIST_FIELDS plus the
    declared REST-only extras.

    MERGED_PR_LIST_FIELDS is the single source of truth for the shared shape;
    MERGED_PR_REST_ONLY_FIELDS declares the fields gh's `--json` list cannot
    express (see the comment at its definition). This asserts the REST path
    honors both rather than restating the field lists, so adding a field to
    either contract forces the normalizer to supply it — and an undeclared
    extra still fails, keeping the two producers' drift visible.
    """
    expected: set[str] = set(github_module.MERGED_PR_LIST_FIELDS.split(",")) | set(
        github_module.MERGED_PR_REST_ONLY_FIELDS
    )

    gh = github_module.GitHub(Path("."))
    normalized = gh._normalize_rest_pr(
        {
            "number": 501,
            "title": "fix: something",
            "body": "Closes #494",
            "merged_at": "2026-07-20T20:19:07Z",
            "head": {
                "ref": "agent/issue-494-fix-something",
                "sha": "27a20fbdc0ffee0123456789abcdef0123456789",
                "repo": {"full_name": "Senkichi/charlie-work"},
            },
            "base": {"repo": {"full_name": "Senkichi/charlie-work"}},
        }
    )

    assert set(normalized) == expected, (
        "REST normalizer drifted from MERGED_PR_LIST_FIELDS: "
        f"missing={sorted(expected - set(normalized))} "
        f"extra={sorted(set(normalized) - expected)}"
    )


def test_normalize_rest_pr_maps_head_sha_to_head_ref_oid() -> None:
    """REST spells the merged head OID `head.sha`; consumers read gh's GraphQL
    name `headRefOid`. Post-merge audits use it to prove *which* commit was
    merged, so a None here silently defeats any approved-SHA comparison."""
    gh = github_module.GitHub(Path("."))

    normalized = gh._normalize_rest_pr(
        {
            "number": 1,
            "head": {"ref": "topic", "sha": "deadbeef", "repo": {"full_name": "o/r"}},
            "base": {"repo": {"full_name": "o/r"}},
        }
    )

    assert normalized["headRefOid"] == "deadbeef"


def test_normalize_rest_pr_maps_merge_commit_sha_to_merge_commit_oid() -> None:
    """Issue #1194: REST spells the landing merge commit `merge_commit_sha`;
    the #502 tripwire's queue-sync-merge recognition anchors its reachability
    check at this commit's first parent, and reads it as `mergeCommitOid`
    (gh's GraphQL-style naming, matching headRefOid's convention). A None
    here silently defeats condition 3 of `_queue_sync_merge_covered` for
    every REST-sourced merged PR."""
    gh = github_module.GitHub(Path("."))

    normalized = gh._normalize_rest_pr(
        {
            "number": 1,
            "head": {"ref": "topic", "sha": "deadbeef", "repo": {"full_name": "o/r"}},
            "base": {"repo": {"full_name": "o/r"}},
            "merge_commit_sha": "c0ffee",
        }
    )

    assert normalized["mergeCommitOid"] == "c0ffee"


def test_normalize_rest_pr_merge_commit_oid_is_none_when_absent() -> None:
    """A REST payload with no `merge_commit_sha` (should not happen for a
    genuinely merged PR, but the field is attacker/API-controlled input) must
    map to None rather than KeyError, so `_queue_sync_merge_covered` sees its
    documented fail-closed `not merge_commit_sha` branch instead of crashing
    the tripwire pass (issue #1194)."""
    gh = github_module.GitHub(Path("."))

    normalized = gh._normalize_rest_pr(
        {
            "number": 1,
            "head": {"ref": "topic", "sha": "deadbeef", "repo": {"full_name": "o/r"}},
            "base": {"repo": {"full_name": "o/r"}},
        }
    )

    assert normalized["mergeCommitOid"] is None


def test_merged_pr_list_exposes_head_ref_oid_end_to_end(monkeypatch, tmp_path: Path) -> None:
    """The full REST path — not just the normalizer — must surface headRefOid,
    since merged_pr_list() is REST-only by construction (issue #361)."""
    page = [
        {
            "number": 501,
            "title": "fix: something",
            "body": "",
            "merged_at": "2026-07-20T20:19:07Z",
            "head": {"ref": "agent/issue-494", "sha": "27a20fbd", "repo": {"full_name": "o/r"}},
            "base": {"repo": {"full_name": "o/r"}},
        }
    ]
    responses = [json.dumps(page), "[]"]

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=responses.pop(0), stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    merged = gh.merged_pr_list()

    assert len(merged) == 1
    assert merged[0]["headRefOid"] == "27a20fbd"


def test_commit_check_runs_wraps_rest_endpoint(monkeypatch, tmp_path: Path) -> None:
    """reconcile.py's aviator_stale_blocked detection needs output.summary,
    which gh pr checks --json cannot surface (its description field is always
    empty for App-created Check Runs) -- this is the only path that can."""
    payload = {
        "check_runs": [
            {
                "id": 90085390042,
                "name": "aviator/checks",
                "status": "completed",
                "conclusion": "failure",
                "output": {
                    "title": "Aviator checks - blocked",
                    "summary": (
                        "This PR is not ready to merge (currently in state blocked): "
                        "PR has a blocked label, remove to re-queue."
                    ),
                },
            }
        ]
    }

    def fake_run(cmd, *args, **kwargs):
        assert cmd[:2] == ["gh", "api"]
        assert cmd[2] == "repos/{owner}/{repo}/commits/abc123/check-runs"
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=json.dumps(payload), stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    check_runs = gh.commit_check_runs("abc123")

    assert check_runs is not None
    assert check_runs[0]["name"] == "aviator/checks"
    assert check_runs[0]["conclusion"] == "failure"
    assert "blocked label" in check_runs[0]["output"]["summary"]


def test_commit_check_runs_returns_none_on_failure(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="not found")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    assert gh.commit_check_runs("missing-sha") is None


def test_remove_pr_label_invokes_gh_pr_edit(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    ok = gh.remove_pr_label(1400, "blocked")

    assert ok is True
    assert calls[-1][:5] == ["gh", "pr", "edit", "1400", "--remove-label"]
    assert calls[-1][5] == "blocked"


# --- pr_close / pr_reopen / push_empty_commit (issue #1274, W17) -----------
#
# These three exercise only the GitHub-client-surface contract in isolation
# (dry-run synthetic-ok, allow_failure propagation, never-raises) -- nothing
# in review()'s janitor-gate path calls them yet. That wiring, and its own
# fixture-level tests (AC3-AC8/AC10/AC11), land in a later step of this
# item.


def test_pr_close_dry_run_returns_synthetic_ok_without_subprocess_call(
    monkeypatch, tmp_path: Path
) -> None:
    """Dry-run must short-circuit before any `gh` call, exactly like pr_ready."""

    def fake_run(cmd, *args, **kwargs):
        raise AssertionError(f"dry-run pr_close must not invoke subprocess: {cmd}")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path, dry_run=True)
    result = gh.pr_close(42)

    assert result == github_module.GitHubRunResult(
        ok=True, returncode=0, stdout="", stderr="", value=None, error=None
    )


def test_pr_close_success_returns_ok_result(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.pr_close(42)

    assert calls == [["gh", "pr", "close", "42"]]
    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True


def test_pr_close_failure_never_raises_returns_error_result(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="gh: pull request #42 not found"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.pr_close(42)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert "not found" in (result.error or "")


def test_pr_reopen_dry_run_returns_synthetic_ok_without_subprocess_call(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_run(cmd, *args, **kwargs):
        raise AssertionError(f"dry-run pr_reopen must not invoke subprocess: {cmd}")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path, dry_run=True)
    result = gh.pr_reopen(42)

    assert result == github_module.GitHubRunResult(
        ok=True, returncode=0, stdout="", stderr="", value=None, error=None
    )


def test_pr_reopen_success_returns_ok_result(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.pr_reopen(42)

    assert calls == [["gh", "pr", "reopen", "42"]]
    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True


def test_pr_reopen_failure_never_raises_returns_error_result(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="gh: could not reopen"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.pr_reopen(42)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert "could not reopen" in (result.error or "")


def _make_fake_push_empty_commit_run(
    *,
    tip_sha: str = "tip-sha-abc",
    tree_sha: str = "tree-sha-def",
    new_sha: str = "new-sha-ghi",
    fail_at_step: int | None = None,
):
    """Build a fake ``subprocess.run`` covering ``push_empty_commit``'s four
    ordered `gh api` calls: GET ref -> GET commit -> POST commit -> PATCH
    ref. ``fail_at_step`` (1-indexed) makes that call return a nonzero exit
    so callers can assert the method stops there rather than proceeding
    against inconsistent state.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        step = len(calls)
        if fail_at_step is not None and step == fail_at_step:
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr=f"boom at step {step}"
            )
        joined = " ".join(cmd)
        if "-X" not in cmd and "refs/heads/" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps({"object": {"sha": tip_sha}}),
                stderr="",
            )
        if "-X" not in cmd and "git/commits/" in joined:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps({"tree": {"sha": tree_sha}}),
                stderr="",
            )
        if "-X" in cmd and "POST" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps({"sha": new_sha}), stderr=""
            )
        if "-X" in cmd and "PATCH" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps({"object": {"sha": new_sha}}),
                stderr="",
            )
        raise AssertionError(f"unexpected gh api call in push_empty_commit test: {cmd}")

    return fake_run, calls


def test_push_empty_commit_dry_run_returns_synthetic_ok_without_subprocess_call(
    monkeypatch, tmp_path: Path
) -> None:
    """Dry-run must short-circuit before ANY `gh` call -- including the
    read-only ref/commit lookups -- because the operation as a whole is
    unconditionally mutating.
    """

    def fake_run(cmd, *args, **kwargs):
        raise AssertionError(f"dry-run push_empty_commit must not invoke subprocess: {cmd}")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path, dry_run=True)
    result = gh.push_empty_commit("agent/issue-123-fix")

    assert result == github_module.GitHubRunResult(
        ok=True, returncode=0, stdout="", stderr="", value=None, error=None
    )


def test_push_empty_commit_success_walks_all_four_steps(monkeypatch, tmp_path: Path) -> None:
    fake_run, calls = _make_fake_push_empty_commit_run()
    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.push_empty_commit("agent/issue-123-fix")

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True
    assert len(calls) == 4
    # GET ref, GET commit -> read-only, no -X
    assert "-X" not in calls[0]
    assert "-X" not in calls[1]
    # POST new commit object, PATCH ref -> mutating
    assert "POST" in calls[2]
    assert "PATCH" in calls[3]


@pytest.mark.parametrize("fail_at_step", [1, 2, 3, 4])
def test_push_empty_commit_never_raises_stops_at_first_failure(
    fail_at_step: int, monkeypatch, tmp_path: Path
) -> None:
    """A failure at any of the four steps returns ok=False and does not
    attempt any later step against now-inconsistent state.
    """
    fake_run, calls = _make_fake_push_empty_commit_run(fail_at_step=fail_at_step)
    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.push_empty_commit("agent/issue-123-fix")

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.error
    assert len(calls) == fail_at_step
