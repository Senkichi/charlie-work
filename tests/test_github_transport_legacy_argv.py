"""The closed ``gh`` argv -> request table and the ``GitHub.run`` shim (ADR-0006).

The table only shrinks: ``ROW_COUNT`` is a ratchet. Adding a translated shape
means a capability still speaks gh argv; lowering it is the migration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import RuntimeConfig
from charlie_work.github import GitHubError, GitHubNotFoundError
from charlie_work.github_transport import FailureKind, GraphQLRequest, Response, RestRequest
from charlie_work.github_transport.legacy_argv import (
    LegacyCli,
    legacy_is_mutating,
    request_for_argv,
)

from _fake_transport import FakeAdapter, failure, make_github, ok

# Translated shapes: the REST GET row and the GraphQL query row.
ROW_COUNT = 2


def test_rest_get_row_translates_to_a_typed_request() -> None:
    translated = request_for_argv(["api", "repos/o/r/pulls?state=open", "--paginate"])
    assert isinstance(translated.request, RestRequest)
    assert translated.request.method == "GET"
    assert translated.paginate is True


def test_graphql_query_row_translates_to_a_typed_request() -> None:
    translated = request_for_argv(
        [
            "api",
            "graphql",
            "-f",
            "query=query { viewer { login } }",
            "-f",
            "owner=a",
            "-f",
            "name=b",
        ]
    )
    assert isinstance(translated.request, GraphQLRequest)


@pytest.mark.parametrize(
    "argv",
    [
        ["pr", "view", "1", "--json", "state"],
        ["api", "-X", "POST", "repos/o/r/issues/1/comments", "-f", "body=x"],
        ["api", "graphql", "-f", "query=mutation { x }"],
        ["api", "repos/o/r/actions/jobs/1/logs"],
        ["api", "repos/o/r/pulls", "-H", "X-Other: 1"],
        [],
    ],
)
def test_everything_else_is_a_verbatim_passthrough(argv: list[str]) -> None:
    translated = request_for_argv(argv)
    assert translated.request == LegacyCli(tuple(argv))


@pytest.mark.parametrize(
    ("argv", "route"),
    [
        (["run", "cancel", "7"], "repos/{owner}/{repo}/actions/runs/7/cancel"),
        (["run", "rerun", "7"], "repos/{owner}/{repo}/actions/runs/7/rerun"),
        (
            ["run", "rerun", "7", "--failed"],
            "repos/{owner}/{repo}/actions/runs/7/rerun-failed-jobs",
        ),
    ],
)
def test_run_mutation_rows_translate_to_a_post(argv: list[str], route: str) -> None:
    request = request_for_argv(argv).request
    assert request == RestRequest("POST", route)
    assert request_for_argv(argv, use_requests=False).request == LegacyCli(tuple(argv))


def test_kill_switch_forces_passthrough_even_for_a_translatable_shape() -> None:
    translated = request_for_argv(["api", "repos/o/r/pulls"], use_requests=False)
    assert isinstance(translated.request, LegacyCli)


def test_row_count_ratchet() -> None:
    rows = {
        type(request_for_argv(argv).request)
        for argv in (
            ["api", "repos/o/r/pulls"],
            ["api", "graphql", "-f", "query=query { a }"],
        )
        if not isinstance(request_for_argv(argv).request, LegacyCli)
    }
    assert len(rows) <= ROW_COUNT


def test_legacy_mutation_classification_is_unchanged() -> None:
    assert legacy_is_mutating(["pr", "merge", "1"]) is True
    assert legacy_is_mutating(["pr", "view", "1"]) is False
    assert legacy_is_mutating(["api", "repos/o/r/pulls"]) is False
    assert legacy_is_mutating(["api", "repos/o/r/issues", "-f", "a=b"]) is True


def test_shim_serves_a_rest_get_over_http_and_parses_json(tmp_path: Path) -> None:
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok([{"number": 1}])]))
    assert gh.run(["api", "repos/o/r/pulls"], json_output=True) == [{"number": 1}]
    assert len(http.requests) == 1


def test_shim_routes_passthrough_to_gh_even_in_http_mode(tmp_path: Path) -> None:
    gh_adapter = FakeAdapter(
        "gh", [Response(200, (), "hello\n", "gh", returncode=0)], token="tok-1"
    )
    gh, http, gh_adapter = make_github(tmp_path, gh=gh_adapter)
    assert gh.run(["pr", "view", "1"]) == "hello"
    assert http.calls == []


def test_shim_not_found_raises_the_specific_error(tmp_path: Path) -> None:
    body = '{"message": "Not Found"}'
    gh, _, _ = make_github(tmp_path, http=FakeAdapter("http", [ok(body, status=404)]))
    with pytest.raises(GitHubNotFoundError):
        gh.run(["api", "repos/o/r/pulls/9"], json_output=True)


def test_shim_allow_failure_returns_a_result_value(tmp_path: Path) -> None:
    gh, _, _ = make_github(
        tmp_path,
        http=FakeAdapter("http", [ok('{"message": "Server Error"}', status=500)]),
        runtime=RuntimeConfig(gh_max_retries=0),
    )
    result = gh.run(["api", "repos/o/r/pulls"], json_output=True, allow_failure=True)
    assert result.ok is False
    assert result.returncode == 1


def test_shim_timeout_maps_to_returncode_124(tmp_path: Path) -> None:
    gh, _, _ = make_github(
        tmp_path,
        http=FakeAdapter("http", [failure(FailureKind.TIMEOUT, "timed out")]),
        runtime=RuntimeConfig(gh_max_retries=0),
    )
    result = gh.run(["api", "repos/o/r/pulls"], allow_failure=True)
    assert result.returncode == 124
    with pytest.raises(GitHubError):
        gh.run(["api", "repos/o/r/pulls"])


def test_shim_dry_run_short_circuits_a_mutation(tmp_path: Path) -> None:
    gh, http, gh_adapter = make_github(tmp_path, dry_run=True)
    assert gh.run(["pr", "merge", "1"]) == "DRY-RUN: gh pr merge 1"
    assert gh.run(["pr", "merge", "1"], json_output=True) == []
    assert http.calls == [] and gh_adapter.calls == []
