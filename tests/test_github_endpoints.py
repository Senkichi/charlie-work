"""Single-endpoint wrapper tests: ``check_graphql_rate_limit``,
``compare_diff``, ``commit_check_runs``, and ``remove_pr_label`` -- thin
``gh``/``gh api`` calls and their return-shape contracts.

Split out of ``tests/test_github.py`` (issue #1572, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_github_fixtures.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _fake_transport import FakeAdapter, make_github, ok, sent
from charlie_work import github as github_module
from charlie_work.config import RuntimeConfig
from charlie_work.github_transport.request import GraphQLRequest
from _github_fixtures import _read_fixture


def test_check_graphql_rate_limit_parses_live_payload(tmp_path: Path) -> None:
    """Issue #398: the rate-limit guard must parse a live ``rate_limit`` payload."""
    rate_limit_json = _read_fixture("gh_rate_limit.json")
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok(rate_limit_json)]))

    sufficient, remaining, reset_at = gh.check_graphql_rate_limit(threshold=1500)

    # Fixture has graphql.remaining == 4114, reset is a unix timestamp.
    assert sufficient is True
    assert remaining == 4114
    assert isinstance(reset_at, int)
    assert sent(http) == [("GET", "rate_limit", None)]


_FAR_FUTURE_RESET = "4102444800"  # 2100-01-01: never "already reset" on any host clock


def _graphql_reply_with_window(remaining: int, resource: str = "graphql"):
    reply = ok(
        {"data": {"viewer": {"login": "octo"}}},
        headers={
            "x-ratelimit-limit": "5000",
            "x-ratelimit-remaining": str(remaining),
            "x-ratelimit-reset": _FAR_FUTURE_RESET,
            "x-ratelimit-resource": resource,
        },
    )
    return reply


def test_check_graphql_rate_limit_answers_from_observed_headers(tmp_path: Path) -> None:
    """B15: a GraphQL window seen in a recent response answers the budget check
    without a ``rate_limit`` call."""
    http = FakeAdapter("http", [_graphql_reply_with_window(4000)])
    gh, http, _ = make_github(tmp_path, http=http)
    gh._transport_v2.send(GraphQLRequest.of("query { viewer { login } }"))

    result = gh.check_graphql_rate_limit(threshold=1500)

    assert result == (True, 4000, int(_FAR_FUTURE_RESET))
    assert len(http.api_requests) == 1  # the GraphQL read; no rate_limit request
    assert gh.check_graphql_rate_limit(threshold=4001) == (False, 4000, int(_FAR_FUTURE_RESET))


def test_check_graphql_rate_limit_ignores_a_window_for_another_resource(tmp_path: Path) -> None:
    """B15: only the ``graphql`` window answers; a core window does not."""
    replies = [
        _graphql_reply_with_window(10, resource="core"),
        ok(_read_fixture("gh_rate_limit.json")),
    ]
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", replies))
    gh._transport_v2.send(GraphQLRequest.of("query { viewer { login } }"))

    sufficient, remaining, _reset = gh.check_graphql_rate_limit(threshold=1500)

    assert (sufficient, remaining) == (True, 4114)
    assert sent(http)[-1] == ("GET", "rate_limit", None)


def test_check_graphql_rate_limit_below_threshold_returns_insufficient(tmp_path: Path) -> None:
    """Issue #398: when remaining points are below the threshold the guard reports insufficient."""
    rate_limit_json = _read_fixture("gh_rate_limit.json")
    gh, _, _ = make_github(tmp_path, http=FakeAdapter("http", [ok(rate_limit_json)]))

    sufficient, remaining, reset_at = gh.check_graphql_rate_limit(threshold=5000)

    assert sufficient is False
    assert remaining == 4114
    assert isinstance(reset_at, int)


def test_compare_diff_hits_three_dot_compare_with_diff_media_type(tmp_path: Path) -> None:
    """compare_diff must call the three-dot compare endpoint with the diff
    media type Accept header (not the default JSON compare metadata), and
    return the raw response body unwrapped from GitHubRunResult."""
    diff = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new\n"
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok(diff)]))

    result = gh.compare_diff("sha-old", "sha-new")

    assert sent(http) == [("GET", "repos/{owner}/{repo}/compare/sha-old...sha-new", None)]
    assert http.api_requests[0].accept == "application/vnd.github.v3.diff"  # type: ignore[union-attr]
    assert result == diff.strip()


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

    # gh_transport="gh": this test exercises the gh-subprocess failure path
    # directly via `fake_run`. `compare_diff`'s REST-GET-with-header shape is
    # an HTTP-transport candidate (issue #1834), and RuntimeConfig's
    # gh_transport field defaults to "http" -- pinning "gh" here keeps this
    # test exercising what it was written to test rather than the (already
    # separately covered, see tests/test_http_transport.py) HTTP path.
    gh = github_module.GitHub(tmp_path, runtime=RuntimeConfig(gh_max_retries=0, gh_transport="gh"))
    result = gh.compare_diff("sha-old", "sha-new")

    assert result is None


def test_commit_check_runs_wraps_rest_endpoint(tmp_path: Path) -> None:
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
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok(payload)]))

    check_runs = gh.commit_check_runs("abc123")

    assert sent(http) == [("GET", "repos/{owner}/{repo}/commits/abc123/check-runs", None)]
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


def test_remove_pr_label_invokes_gh_pr_edit(tmp_path: Path) -> None:
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok([])]))
    done = gh.remove_pr_label(1400, "blocked")

    assert done is True
    assert sent(http) == [("DELETE", "repos/{owner}/{repo}/issues/1400/labels/blocked", None)]
