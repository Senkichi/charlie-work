"""Nested connections gh pages to the end, and the two read fixes beside them.

ADR-0006 review r3: F-F (``comments``, ``closingIssuesReferences`` and
``statusCheckRollup`` contexts were cut at ``first:100``), F-C (``issue view``
of a PR number), F-E (``--state all`` search). Scripted adapters only; nothing
touches the network.
"""

from __future__ import annotations

import json
from typing import Any

from _fake_transport import FakeAdapter, build_guard, ok
from charlie_work.github_transport import gh_json_fields as g
from charlie_work.github_transport import gh_json_pages as pages
from charlie_work.github_transport.json_read import JsonRead
from charlie_work.github_transport.outcome import FailureKind, TransportFailure


def _gql(data: object) -> Any:
    return ok({"data": data})


def _guard(*replies: Any) -> tuple[Any, FakeAdapter]:
    http = FakeAdapter("http", list(replies))
    guard, http, _gh, _ = build_guard(http=http)
    return guard, http


def _body(outcome: Any) -> Any:
    return json.loads(outcome.body)


def _info(more: bool, cursor: str | None = None) -> dict[str, Any]:
    return {"hasNextPage": more, "endCursor": cursor if more else "end"}


def _run(i: int) -> dict[str, Any]:
    return {
        "__typename": "CheckRun",
        "name": f"job-{i}",
        "status": "COMPLETED",
        "conclusion": "SUCCESS",
        "completedAt": "2026-10-01T00:00:00Z",
        "startedAt": "2026-10-01T00:00:00Z",
        "detailsUrl": f"https://example.invalid/{i}",
        "checkSuite": {"workflowRun": {"event": "pull_request", "workflow": {"name": "CI"}}},
    }


def _rollup(runs: range, more: bool, cursor: str = "c1") -> dict[str, Any]:
    return {"contexts": {"nodes": [_run(i) for i in runs], "pageInfo": _info(more, cursor)}}


def _pr(rollup: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
    return {"id": "PR_1", "number": 5, "statusCheckRollup": rollup, **extra}


def _names(body: Any) -> list[str]:
    return [row["name"] for row in body["statusCheckRollup"]]


def _view(fields: str, number: int = 5, resource: str = "pr") -> JsonRead:
    return JsonRead(resource, "view", fields, number=number)  # type: ignore[arg-type]


# -- F-F: view ----------------------------------------------------------------


def test_pr_view_follows_a_rollup_with_more_than_a_page_of_contexts() -> None:
    first = _gql({"repository": {"pullRequest": _pr(_rollup(range(100), True))}})
    second = _gql({"node": {"statusCheckRollup": _rollup(range(100, 130), False)}})
    guard, http = _guard(first, second)

    outcome = _view("number,statusCheckRollup").execute(guard, "octo", "hello")

    assert _names(_body(outcome)) == [f"job-{i}" for i in range(130)]
    follow = json.loads(http.api_requests[1].variables)
    assert follow == {"id": "PR_1", "after": "c1"}


def test_pr_view_follows_closing_references_past_a_page() -> None:
    repo = {"id": "R", "name": "hello", "owner": {"id": "O", "login": "octo"}}

    def ref(n: int) -> dict[str, Any]:
        return {"id": f"I_{n}", "number": n, "url": "u", "repository": repo}

    node = _pr(None, closingIssuesReferences={"nodes": [ref(1)], "pageInfo": _info(True, "k")})
    more = {"node": {"closingIssuesReferences": {"nodes": [ref(2)], "pageInfo": _info(False)}}}
    guard, _http = _guard(_gql({"repository": {"pullRequest": node}}), _gql(more))

    outcome = _view("number,closingIssuesReferences").execute(guard, "octo", "hello")

    assert [r["number"] for r in _body(outcome)["closingIssuesReferences"]] == [1, 2]


def test_issue_view_follows_comments_past_a_page() -> None:
    def comment(n: int) -> dict[str, Any]:
        return {"id": f"C_{n}", "author": {"login": "a"}, "body": f"b{n}", "url": "u"}

    node = {
        "id": "I_1",
        "number": 9,
        "comments": {"nodes": [comment(1)], "pageInfo": _info(True, "k")},
    }
    more = {"node": {"comments": {"nodes": [comment(2), comment(3)], "pageInfo": _info(False)}}}
    guard, http = _guard(_gql({"repository": {"issueOrPullRequest": node}}), _gql(more))

    outcome = _view("number,comments", 9, "issue").execute(guard, "octo", "hello")

    assert [c["body"] for c in _body(outcome)["comments"]] == ["b1", "b2", "b3"]
    assert "... on Issue" in http.api_requests[1].document


def test_a_connection_that_never_ends_is_a_defect_not_a_truncation() -> None:
    forever = _gql({"node": {"statusCheckRollup": _rollup(range(1), True, "again")}})
    guard, _http = _guard(
        _gql({"repository": {"pullRequest": _pr(_rollup(range(100), True))}}),
        *([forever] * 60),
    )

    outcome = _view("number,statusCheckRollup").execute(guard, "octo", "hello")

    assert isinstance(outcome, TransportFailure)
    assert outcome.kind is FailureKind.ADAPTER_DEFECT


def test_a_next_page_without_an_id_is_a_defect() -> None:
    node = {"number": 5, "statusCheckRollup": _rollup(range(100), True)}
    guard, _http = _guard(_gql({"repository": {"pullRequest": node}}))

    outcome = _view("number,statusCheckRollup").execute(guard, "octo", "hello")

    assert isinstance(outcome, TransportFailure)
    assert outcome.kind is FailureKind.ADAPTER_DEFECT


def test_a_next_page_without_an_end_cursor_is_a_defect() -> None:
    """``hasNextPage`` with no usable ``endCursor`` is a GitHub contract
    violation -- an ADAPTER_DEFECT (gh fallback), not a silently short list."""
    node = _pr(_rollup(range(1), more=True, cursor=None))
    guard, _http = _guard(_gql({"repository": {"pullRequest": node}}))

    outcome = _view("number,statusCheckRollup").execute(guard, "octo", "hello")

    assert isinstance(outcome, TransportFailure)
    assert outcome.kind is FailureKind.ADAPTER_DEFECT


def test_checks_paging_without_an_end_cursor_is_a_defect() -> None:
    """The ``gh pr checks`` walk has the same contract: a reported next page
    it cannot follow must not normalize as a short list."""
    node = _pr(_rollup(range(1), more=True, cursor=None))
    guard, _guard_http = _guard(_gql({"repository": {"pullRequest": node}}))

    outcome = JsonRead("pr", "checks", "name,bucket", number=5).execute(guard, "octo", "hello")

    assert isinstance(outcome, TransportFailure)
    assert outcome.kind is FailureKind.ADAPTER_DEFECT


def test_a_failed_follow_up_page_is_returned_not_swallowed() -> None:
    guard, _http = _guard(
        _gql({"repository": {"pullRequest": _pr(_rollup(range(100), True))}}),
        _gql(None),
    )

    outcome = _view("number,statusCheckRollup").execute(guard, "octo", "hello")

    assert isinstance(outcome, TransportFailure)


def test_a_complete_connection_costs_no_extra_request() -> None:
    guard, http = _guard(_gql({"repository": {"pullRequest": _pr(_rollup(range(3), False))}}))

    outcome = _view("number,statusCheckRollup").execute(guard, "octo", "hello")

    assert len(_names(_body(outcome))) == 3
    assert len(http.api_requests) == 1


# -- F-F: list shapes ---------------------------------------------------------


def test_pr_list_rollup_is_completed_per_row() -> None:
    page = {
        "repository": {
            "pullRequests": {
                "nodes": [_pr(_rollup(range(100), True))],
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
        }
    }
    guard, _http = _guard(
        _gql(page),
        _gql({"node": {"statusCheckRollup": _rollup(range(100, 101), False)}}),
    )

    outcome = JsonRead("pr", "list", "number,statusCheckRollup", state="open").execute(
        guard, "octo", "hello"
    )

    assert len(_names(_body(outcome)[0])) == 101


def test_the_documents_ask_for_page_info_and_ids() -> None:
    view = g.document_for("pr", "number,statusCheckRollup,closingIssuesReferences", "view")

    assert view.count("pageInfo{hasNextPage endCursor}") >= 2
    assert "{id " in view
    assert "contexts(first:100,after:$after)" in g.checks_document()


def test_pending_pages_ignores_nodes_without_page_info() -> None:
    node = {"statusCheckRollup": {"contexts": {"nodes": []}}}

    assert pages.pending_pages("pr", "statusCheckRollup", node) == []


# -- F-C: issue view of a PR number ---------------------------------------------


def test_issue_view_document_reads_issue_or_pull_request() -> None:
    document = g.document_for("issue", "number,state", "view")

    assert "issueOrPullRequest(number:$number)" in document
    assert "... on Issue" in document and "... on PullRequest" in document


def test_issue_view_of_a_pr_number_returns_its_state() -> None:
    node = {"number": 2105, "state": "MERGED"}
    guard, _http = _guard(_gql({"repository": {"issueOrPullRequest": node}}))

    outcome = _view("number,state", 2105, "issue").execute(guard, "octo", "hello")

    assert _body(outcome) == {"number": 2105, "state": "MERGED"}


# -- F-E: --state all ---------------------------------------------------------------


def test_search_with_state_all_adds_no_state_qualifier() -> None:
    guard, http = _guard(_gql({"search": {"nodes": [{"number": 3}]}}))

    JsonRead("pr", "search", "number", state="all", search='"#9"', limit=20).execute(
        guard, "octo", "hello"
    )

    assert json.loads(http.api_requests[0].variables)["q"] == 'repo:octo/hello is:pr "#9"'


def test_search_with_state_all_and_no_term_has_no_trailing_space() -> None:
    guard, http = _guard(_gql({"search": {"nodes": []}}))

    JsonRead("pr", "search", "number", state="all", limit=20).execute(guard, "octo", "hello")

    assert json.loads(http.api_requests[0].variables)["q"] == "repo:octo/hello is:pr"
