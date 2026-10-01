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
from charlie_work.github_transport.json_read import JsonRead, RunListRead
from charlie_work.github_transport.legacy_argv import (
    LegacyCli,
    legacy_is_mutating,
    request_for_argv,
)

from _fake_transport import FakeAdapter, failure, make_github, ok

# Translated shapes: the REST GET row, the GraphQL query row, the ``--json``
# issue/pr dialect row and the ``run list --json`` row (the run mutations are
# a REST POST, the same request type as the GET row).
ROW_COUNT = 4


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


def test_job_log_route_translates_with_redirect_following() -> None:
    translated = request_for_argv(["api", "repos/{owner}/{repo}/actions/jobs/9/logs"])
    assert isinstance(translated.request, RestRequest)
    assert translated.request.follow_redirect is True
    plain = request_for_argv(["api", "repos/{owner}/{repo}/pulls/9"])
    assert isinstance(plain.request, RestRequest)
    assert plain.request.follow_redirect is False


@pytest.mark.parametrize(
    "argv",
    [
        ["pr", "view", "1", "--web"],
        ["api", "-X", "POST", "repos/o/r/issues/1/comments", "-f", "body=x"],
        ["api", "graphql", "-f", "query=mutation { x }"],
        ["issue", "list", "--json", "number", "--assignee", "me"],
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
            ["pr", "view", "1", "--json", "state"],
            ["run", "list", "--json", "databaseId"],
        )
        if not isinstance(request_for_argv(argv).request, LegacyCli)
    }
    assert len(rows) <= ROW_COUNT


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["pr", "view", "7", "--json", "state"],
            JsonRead("pr", "view", "state", number=7),
        ),
        (
            ["issue", "view", "7", "--json", "title"],
            JsonRead("issue", "view", "title", number=7),
        ),
        (
            ["pr", "checks", "7", "--json", "name"],
            JsonRead("pr", "checks", "name", number=7),
        ),
        (
            ["issue", "list", "--limit", "5", "--state", "closed", "--label", "a", "--label", "b",
             "--json", "number"],
            JsonRead("issue", "list", "number", state="closed", labels=("a", "b"), limit=5),
        ),
        (
            ["pr", "list", "--head", "agent/x", "--state", "all", "--json", "number"],
            JsonRead("pr", "list", "number", state="all", head="agent/x"),
        ),
        (
            ["pr", "list", "--state", "merged", "--search", '"#9"', "--limit", "20",
             "--json", "number"],
            JsonRead("pr", "search", "number", state="merged", search='"#9"', limit=20),
        ),
        (
            ["run", "list", "--workflow", "CI", "--branch", "main", "--status", "queued",
             "--limit", "100", "--json", "databaseId"],
            RunListRead("databaseId", workflow="CI", branch="main", status="queued", limit=100),
        ),
    ],
)  # fmt: skip
def test_json_rows_translate_to_dialect_reads(argv: list[str], expected: object) -> None:
    assert request_for_argv(argv).request == expected
    assert request_for_argv(argv, use_requests=False).request == LegacyCli(tuple(argv))


@pytest.mark.parametrize(
    "argv",
    [
        ["pr", "view", "x", "--json", "state"],
        ["issue", "checks", "1", "--json", "name"],
        ["issue", "list", "--search", "x", "--json", "number"],
        ["pr", "list", "--state", "bogus", "--json", "number"],
        ["pr", "list", "--limit", "0", "--json", "number"],
        ["run", "list", "--user", "me", "--json", "databaseId"],
    ],
)
def test_json_rows_fail_closed_to_passthrough(argv: list[str]) -> None:
    assert request_for_argv(argv).request == LegacyCli(tuple(argv))


def test_json_reads_are_never_mutations() -> None:
    assert JsonRead("pr", "view", "state", number=1).is_mutation is False
    assert RunListRead("databaseId").is_mutation is False


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


def test_shim_dry_run_does_not_suppress_a_dialect_read(tmp_path: Path) -> None:
    """B1: mutation-ness comes from the translated request, so a ``--json``
    read runs under --dry-run instead of returning the DRY-RUN stub."""
    gh, http, _ = make_github(tmp_path, dry_run=True, http=FakeAdapter("http", [ok({"data": {
        "repository": {"pullRequest": {"state": "OPEN"}}}})]))  # fmt: skip

    assert gh.run(["pr", "view", "7", "--json", "state"], json_output=True) == {"state": "OPEN"}
    assert len(http.api_requests) == 1


def test_shim_serves_run_list_over_rest(tmp_path: Path) -> None:
    runs = {"workflow_runs": [
        {"id": 1, "name": "CI", "status": "queued", "created_at": "t1", "head_branch": "main"},
        {"id": 2, "name": "Other", "status": "queued", "created_at": "t2", "head_branch": "main"},
    ]}  # fmt: skip
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok(runs)]))

    result = gh.run(
        ["run", "list", "--workflow", "CI", "--branch", "main", "--status", "queued",
         "--limit", "100", "--json", "databaseId,status,createdAt,headBranch"],
        json_output=True,
    )  # fmt: skip

    assert result == [
        {"databaseId": 1, "status": "queued", "createdAt": "t1", "headBranch": "main"}
    ]
    request = http.api_requests[0]
    assert request.route == "repos/octo/hello/actions/runs"
    assert dict(request.query) == {"per_page": "100", "branch": "main", "status": "queued"}


def test_shim_dry_run_short_circuits_a_mutation(tmp_path: Path) -> None:
    gh, http, gh_adapter = make_github(tmp_path, dry_run=True)
    assert gh.run(["pr", "merge", "1"]) == "DRY-RUN: gh pr merge 1"
    assert gh.run(["pr", "merge", "1"], json_output=True) == []
    assert http.calls == [] and gh_adapter.calls == []
