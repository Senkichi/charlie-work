"""The ``gh --json`` dialect over GraphQL (ADR-0006, G4; flagged changes B1, B7, B12).

Contract: for every recorded object, normalizing the GraphQL response gives
exactly the JSON ``gh`` printed (``tests/fixtures/gh_json/``, recorded by
``scripts/record_gh_json_fixtures.py``). Executor cases drive a real
``GuardedTransport`` over scripted adapters; nothing touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _fake_transport import FakeAdapter, build_guard, ok
from charlie_work.checks import summarize_checks
from charlie_work.github_capabilities import checks, issues, pull_requests, transport
from charlie_work.github_transport import GraphQLError, Response
from charlie_work.github_transport import gh_json_fields as g
from charlie_work.github_transport.json_read import JsonRead

FIXTURES = Path(__file__).parent / "fixtures" / "gh_json"
CASES = sorted(FIXTURES.glob("*.json"))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(case: dict[str, Any]) -> Any:
    repo = case["graphql_data"]["repository"]
    if case["resource"] == "checks":
        return g.normalize_checks(g.checks_contexts(repo["pullRequest"]), case["fields"])
    node = repo["issue"] if case["resource"] == "issue" else repo["pullRequest"]
    return g.normalize_node(case["resource"], node, case["fields"])


def test_fixtures_exist() -> None:
    assert len(CASES) >= 8


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_graphql_normalizes_to_what_gh_printed(path: Path) -> None:
    case = _load(path)

    assert _normalize(case) == case["gh_json"]


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_fixtures_are_not_vacuous(path: Path) -> None:
    gh_json = _load(path)["gh_json"]

    assert gh_json, "an empty fixture proves nothing"


def test_fixtures_hold_a_pass_a_fail_and_a_pending_check() -> None:
    buckets = {
        entry["bucket"]
        for name in ("pr_checks_merged", "pr_checks_open")
        for entry in _load(FIXTURES / f"{name}.json")["gh_json"]
    }

    assert {"pass", "fail", "pending"} <= buckets


# -- field registry ---------------------------------------------------------

_ISSUE_CONSTANTS = [
    issues.ISSUE_LIST_FIELDS,
    issues.ISSUE_VIEW_FIELDS,
    transport.RECONCILE_ISSUE_FIELDS,
]
_PR_CONSTANTS = [
    pull_requests.PR_LIST_FIELDS,
    pull_requests.PR_VIEW_FIELDS,
    pull_requests.MERGED_PR_LIST_FIELDS,
    transport.RECONCILE_PR_FIELDS,
    transport.PR_STATUS_CHECK_ROLLUP_FIELDS,
]


@pytest.mark.parametrize("fields", _ISSUE_CONSTANTS)
def test_every_issue_field_constant_builds_a_document(fields: str) -> None:
    for shape in ("list", "view"):
        assert "query(" in g.document_for("issue", fields, shape)  # type: ignore[arg-type]


@pytest.mark.parametrize("fields", _PR_CONSTANTS)
def test_every_pr_field_constant_builds_a_document(fields: str) -> None:
    for shape in ("list", "view", "search"):
        assert "query(" in g.document_for("pr", fields, shape)  # type: ignore[arg-type]


def test_every_pr_checks_field_constant_is_known() -> None:
    rows = g.normalize_checks([], checks.PR_CHECKS_FIELDS)

    assert rows == []


def test_unknown_field_raises_when_the_document_is_built() -> None:
    with pytest.raises(g.UnknownFieldError, match="nope"):
        g.document_for("pr", "number,nope", "view")


def test_unknown_check_field_raises() -> None:
    with pytest.raises(g.UnknownFieldError, match="nope"):
        g.normalize_checks([], "name,nope")


def test_search_documents_exist_for_pull_requests_only() -> None:
    with pytest.raises(g.UnknownFieldError):
        g.document_for("issue", "number", "search")


@pytest.mark.parametrize(
    ("state", "bucket"),
    [
        ("SUCCESS", "pass"),
        ("SKIPPED", "skipping"),
        ("NEUTRAL", "skipping"),
        ("FAILURE", "fail"),
        ("ERROR", "fail"),
        ("TIMED_OUT", "fail"),
        ("ACTION_REQUIRED", "fail"),
        ("CANCELLED", "cancel"),
        ("IN_PROGRESS", "pending"),
        ("QUEUED", "pending"),
        ("PENDING", "pending"),
    ],
)
def test_bucket_port(state: str, bucket: str) -> None:
    assert g.bucket_for(state) == bucket


def test_states_for() -> None:
    assert g.states_for("pr", "open") == ["OPEN"]
    assert g.states_for("pr", "merged") == ["MERGED"]
    assert g.states_for("issue", "all") is None
    with pytest.raises(g.UnknownFieldError):
        g.states_for("issue", "merged")


def test_bot_author_gets_the_app_prefix() -> None:
    node = {"author": {"__typename": "Bot", "login": "dependabot", "id": "B1"}}

    assert g.normalize_node("pr", node, "author") == {
        "author": {"id": "B1", "is_bot": True, "login": "app/dependabot", "name": ""}
    }


def test_assignees_are_flattened() -> None:
    node = {"assignees": {"nodes": [{"id": "U1", "login": "me", "name": None, "databaseId": 5}]}}

    assert g.normalize_node("issue", node, "assignees") == {
        "assignees": [{"id": "U1", "login": "me", "name": "", "databaseId": 5}]
    }


def test_null_review_decision_is_an_empty_string_like_gh() -> None:
    assert g.normalize_node("pr", {"reviewDecision": None}, "reviewDecision") == {
        "reviewDecision": ""
    }


def test_checks_keep_the_newest_entry_per_name() -> None:
    contexts = [
        {"__typename": "CheckRun", "name": "Tests", "status": "COMPLETED",
         "conclusion": "FAILURE", "startedAt": "2026-01-01T00:00:00Z"},
        {"__typename": "CheckRun", "name": "Tests", "status": "COMPLETED",
         "conclusion": "SUCCESS", "startedAt": "2026-01-02T00:00:00Z"},
    ]  # fmt: skip

    assert g.normalize_checks(contexts, "name,bucket") == [{"name": "Tests", "bucket": "pass"}]


def _run(name: str, workflow: str, event: str, conclusion: str, started: str) -> dict[str, Any]:
    return {
        "__typename": "CheckRun",
        "name": name,
        "status": "COMPLETED",
        "conclusion": conclusion,
        "startedAt": started,
        "checkSuite": {"workflowRun": {"event": event, "workflow": {"name": workflow}}},
    }


def test_same_named_runs_of_different_workflows_both_survive() -> None:
    """A newer passing run must not hide an older failing run of another workflow."""
    contexts = [
        _run("test", "CI", "pull_request", "FAILURE", "2026-09-30T10:00:00Z"),
        _run("test", "CI", "push", "SUCCESS", "2026-09-30T10:00:05Z"),
        _run("test", "Nightly", "pull_request", "SUCCESS", "2026-09-30T10:00:09Z"),
    ]

    rows = g.normalize_checks(contexts, "name,workflow,event,bucket")

    assert sorted((r["workflow"], r["event"], r["bucket"]) for r in rows) == [
        ("CI", "pull_request", "fail"),
        ("CI", "push", "pass"),
        ("Nightly", "pull_request", "pass"),
    ]
    assert summarize_checks(rows, ("test",)).failed == ("test",)


def test_status_context_and_check_run_of_one_name_do_not_collide() -> None:
    contexts = [
        {"__typename": "StatusContext", "context": "build", "state": "FAILURE",
         "createdAt": "2026-09-30T09:00:00Z"},
        _run("build", "CI", "push", "SUCCESS", "2026-09-30T10:00:00Z"),
    ]  # fmt: skip

    rows = g.normalize_checks(contexts, "name,bucket")

    assert sorted(r["bucket"] for r in rows) == ["fail", "pass"]


def test_a_rerun_of_the_same_workflow_and_event_still_dedupes_to_the_newest() -> None:
    contexts = [
        _run("test", "CI", "push", "FAILURE", "2026-09-30T10:00:00Z"),
        _run("test", "CI", "push", "SUCCESS", "2026-09-30T10:05:00Z"),
    ]

    assert g.normalize_checks(contexts, "name,bucket") == [{"name": "test", "bucket": "pass"}]


def test_status_context_is_a_check_named_by_its_context() -> None:
    contexts = [{"__typename": "StatusContext", "context": "ci/x", "state": "PENDING"}]

    assert g.normalize_checks(contexts, "name,state,bucket") == [
        {"name": "ci/x", "state": "PENDING", "bucket": "pending"}
    ]


def test_status_context_selection_asks_for_description() -> None:
    """gh selects ``description`` on StatusContext; without it ``pr checks
    --json description`` normalizes every status context to ""."""
    fragment = "... on StatusContext{context state targetUrl createdAt description}"

    assert fragment in g.checks_document()
    assert fragment in g.document_for("pr", "number,statusCheckRollup", "view")


def test_status_context_description_is_emitted() -> None:
    contexts = [
        {
            "__typename": "StatusContext",
            "context": "ci/x",
            "state": "SUCCESS",
            "description": "build finished",
        }
    ]

    assert g.normalize_checks(contexts, "name,description") == [
        {"name": "ci/x", "description": "build finished"}
    ]


def test_checks_next_cursor_raises_on_a_next_page_without_an_end_cursor() -> None:
    """``hasNextPage`` without ``endCursor`` is a contract violation, not the
    end of the walk."""
    node = {
        "statusCheckRollup": {
            "contexts": {"nodes": [], "pageInfo": {"hasNextPage": True, "endCursor": None}}
        }
    }

    with pytest.raises(g.IncompletePageError):
        g.checks_next_cursor(node)


def test_checks_next_cursor_returns_none_on_the_last_page() -> None:
    assert g.checks_next_cursor({"statusCheckRollup": None}) is None
    info = {"hasNextPage": False, "endCursor": None}
    node = {"statusCheckRollup": {"contexts": {"nodes": [], "pageInfo": info}}}

    assert g.checks_next_cursor(node) is None


# -- executor ---------------------------------------------------------------


def _gql(data: object) -> Any:
    return ok({"data": data})


def _guard(*replies: Any) -> tuple[Any, FakeAdapter]:
    http = FakeAdapter("http", list(replies))
    guard, http, _gh, _ = build_guard(http=http)
    return guard, http


def _body(outcome: Any) -> Any:
    return json.loads(outcome.body)


def test_view_returns_gh_shaped_json() -> None:
    case = _load(FIXTURES / "issue_view_closed.json")
    # ``issue view`` reads ``issueOrPullRequest`` (gh accepts a PR number too).
    data = {"repository": {"issueOrPullRequest": case["graphql_data"]["repository"]["issue"]}}
    guard, http = _guard(_gql(data))

    outcome = JsonRead("issue", "view", case["fields"], number=2090).execute(
        guard, "octo", "hello"
    )

    assert _body(outcome) == case["gh_json"]
    request = http.api_requests[0]
    assert json.loads(request.variables) == {"owner": "octo", "name": "hello", "number": 2090}


def test_list_paginates_to_the_limit_and_sends_filters() -> None:
    def page(numbers: list[int], more: bool) -> Any:
        nodes = [{"number": n} for n in numbers]
        info = {"hasNextPage": more, "endCursor": "c1" if more else None}
        return _gql({"repository": {"issues": {"nodes": nodes, "pageInfo": info}}})

    guard, http = _guard(page(list(range(100)), True), page(list(range(100, 130)), False))

    outcome = JsonRead("issue", "list", "number", state="open", labels=("a",), limit=120).execute(
        guard, "octo", "hello"
    )

    assert [row["number"] for row in _body(outcome)] == list(range(120))
    first = json.loads(http.api_requests[0].variables)
    assert first["states"] == ["OPEN"] and first["labels"] == ["a"] and first["first"] == 100
    assert json.loads(http.api_requests[1].variables)["after"] == "c1"


def test_head_lookup_filters_by_branch_and_state_all() -> None:
    payload = {
        "repository": {
            "pullRequests": {
                "nodes": [{"number": 7}],
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
        }
    }
    guard, http = _guard(_gql(payload))

    outcome = JsonRead("pr", "list", "number", state="all", head="agent/x").execute(
        guard, "octo", "hello"
    )

    assert _body(outcome) == [{"number": 7}]
    variables = json.loads(http.api_requests[0].variables)
    assert variables["head"] == "agent/x" and variables["states"] is None


def test_search_builds_the_repo_scoped_query_and_drops_non_prs() -> None:
    payload = {"search": {"nodes": [{"number": 3}, {}]}}
    guard, http = _guard(_gql(payload))

    outcome = JsonRead("pr", "search", "number", state="merged", search='"#9"', limit=20).execute(
        guard, "octo", "hello"
    )

    assert _body(outcome) == [{"number": 3}]
    assert (
        json.loads(http.api_requests[0].variables)["q"] == 'repo:octo/hello is:pr is:merged "#9"'
    )


def test_checks_executor_returns_the_bucketed_list() -> None:
    case = _load(FIXTURES / "pr_checks_open.json")
    guard, _http = _guard(_gql(case["graphql_data"]))

    outcome = JsonRead("pr", "checks", case["fields"], number=2130).execute(guard, "octo", "hello")

    assert _body(outcome) == case["gh_json"]


def test_checks_with_no_contexts_is_an_empty_list_not_an_error() -> None:
    payload = {"repository": {"pullRequest": {"statusCheckRollup": None}}}
    guard, _http = _guard(_gql(payload))

    outcome = JsonRead("pr", "checks", "name,bucket", number=1).execute(guard, "octo", "hello")

    assert _body(outcome) == []


def test_missing_object_is_an_adapter_defect_value() -> None:
    guard, _http = _guard(_gql({"repository": {"pullRequest": None}}))

    outcome = JsonRead("pr", "view", "number", number=9).execute(guard, "octo", "hello")

    assert "no pr #9" in outcome.detail


def test_graphql_errors_pass_through_as_a_failed_response() -> None:
    reply = Response(
        200, (), '{"data": null}', "http", graphql_errors=(GraphQLError("gone", "NOT_FOUND"),)
    )
    guard, _http = _guard(reply)

    outcome = JsonRead("pr", "view", "number", number=9).execute(guard, "octo", "hello")

    assert not outcome.ok
    assert outcome.graphql_errors[0].message == "gone"


def test_unknown_field_is_a_value_not_a_raise() -> None:
    guard, http = _guard(_gql({}))

    outcome = JsonRead("pr", "view", "bogus", number=9).execute(guard, "octo", "hello")

    assert "bogus" in outcome.detail
    assert http.calls == []
