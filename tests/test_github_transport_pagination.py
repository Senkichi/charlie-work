"""Pagination above the transport (ADR-0006), over a recording FakeTransport."""

from __future__ import annotations

import json

from _fake_transport import FakeTransport, failure, ok

from charlie_work.github_transport import (
    FailureKind,
    GraphQLError,
    GraphQLRequest,
    Response,
    RestRequest,
    TransportFailure,
    paginate_graphql,
    paginate_rest,
)
from charlie_work.github_transport.pagination import MAX_PAGES, next_link, path_from_url

LIST = RestRequest.of("GET", "repos/o/r/pulls", query={"state": "open", "per_page": 2})


def _link(url: str) -> dict[str, str]:
    return {"Link": f'<{url}>; rel="next", <https://api.github.com/x?page=9>; rel="last"'}


def test_next_link_and_path_helpers() -> None:
    assert next_link(_link("https://api.github.com/a?page=2")) == "https://api.github.com/a?page=2"
    assert next_link({"link": '<u>; rel="prev"'}) is None
    assert next_link({}) is None
    assert path_from_url("https://api.github.com/a/b?c=1") == "/a/b?c=1"


def test_rest_pages_are_concatenated_following_next_links() -> None:
    pages = {
        "repos/o/r/pulls": ok(
            [1, 2], headers=_link("https://api.github.com/repos/o/r/pulls?page=2")
        ),
        "repos/o/r/pulls?page=2": ok([3]),
    }

    def handler(request):  # noqa: ANN001
        return pages[request.route + ("?page=2" if ("page", "2") in request.query else "")]

    transport = FakeTransport(handler)
    out = paginate_rest(transport, LIST)
    assert isinstance(out, Response) and json.loads(out.body) == [1, 2, 3]
    assert len(transport.requests) == 2
    assert ("page", "2") in transport.requests[1].query


def test_items_key_unwraps_object_bodies() -> None:
    transport = FakeTransport(
        lambda r: ok({"total_count": 2, "workflow_runs": [{"id": 1}, {"id": 2}]})
    )
    out = paginate_rest(transport, LIST, items_key="workflow_runs")
    assert isinstance(out, Response) and json.loads(out.body) == [{"id": 1}, {"id": 2}]


def test_the_first_failing_page_is_the_outcome_not_a_silent_partial_list() -> None:
    calls = iter(
        [
            ok([1], headers=_link("https://api.github.com/repos/o/r/pulls?page=2")),
            ok({"message": "boom"}, status=502),
        ]
    )
    out = paginate_rest(FakeTransport(lambda r: next(calls)), LIST)
    assert isinstance(out, Response) and out.status == 502


def test_a_transport_failure_on_a_later_page_propagates() -> None:
    calls = iter(
        [
            ok([1], headers=_link("https://api.github.com/repos/o/r/pulls?page=2")),
            failure(FailureKind.TIMEOUT),
        ]
    )
    out = paginate_rest(FakeTransport(lambda r: next(calls)), LIST)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.TIMEOUT


def test_exceeding_the_page_cap_is_a_defect_not_a_truncation() -> None:
    transport = FakeTransport(
        lambda r: ok([1], headers=_link("https://api.github.com/repos/o/r/pulls?page=2"))
    )
    out = paginate_rest(transport, LIST, max_pages=3)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.ADAPTER_DEFECT
    assert len(transport.requests) == 3
    assert MAX_PAGES == 50


def test_off_host_next_links_are_refused() -> None:
    transport = FakeTransport(
        lambda r: ok([1], headers=_link("https://evil.example.net/x?page=2"))
    )
    out = paginate_rest(transport, LIST)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.ADAPTER_DEFECT
    assert len(transport.requests) == 1


def test_a_non_list_body_is_a_defect() -> None:
    out = paginate_rest(FakeTransport(lambda r: ok({"a": 1})), LIST)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.ADAPTER_DEFECT


# -- GraphQL ------------------------------------------------------------------

DOC = "query Q($after: String) { repository { pullRequests(first: 2, after: $after) { nodes { n } pageInfo { hasNextPage endCursor } } } }"
GQL = GraphQLRequest.of(DOC, {"owner": "o"})
PATH = ("repository", "pullRequests")


def _gpage(
    nodes: list, *, has_next: bool, cursor: str | None, errors: list | None = None
) -> Response:
    body: dict = {
        "data": {
            "repository": {
                "pullRequests": {
                    "nodes": nodes,
                    "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                }
            }
        }
    }
    errs = ()
    if errors:
        body["errors"] = errors
        errs = tuple(GraphQLError(e["message"], e.get("type")) for e in errors)
    return Response(200, (), json.dumps(body), "http", errs)


def test_graphql_connection_is_walked_with_the_cursor_and_merged() -> None:
    pages = iter(
        [
            _gpage([{"n": 1}, {"n": 2}], has_next=True, cursor="c1"),
            _gpage([{"n": 3}], has_next=False, cursor=None),
        ]
    )
    transport = FakeTransport(lambda r: next(pages))
    out = paginate_graphql(transport, GQL, connection_path=PATH, limit=10)
    assert isinstance(out, Response)
    nodes = json.loads(out.body)["data"]["repository"]["pullRequests"]["nodes"]
    assert nodes == [{"n": 1}, {"n": 2}, {"n": 3}]
    first, second = (json.loads(r.variables) for r in transport.requests)
    assert first["after"] is None and second["after"] == "c1" and second["owner"] == "o"


def test_graphql_limit_stops_early_and_truncates() -> None:
    transport = FakeTransport(lambda r: _gpage([{"n": 1}, {"n": 2}], has_next=True, cursor="c"))
    out = paginate_graphql(transport, GQL, connection_path=PATH, limit=3)
    assert isinstance(out, Response)
    nodes = json.loads(out.body)["data"]["repository"]["pullRequests"]["nodes"]
    assert len(nodes) == 3 and len(transport.requests) == 2


def test_graphql_errors_abort_unless_partial_is_ok() -> None:
    errors = [{"message": "partial", "type": "NOT_FOUND"}]
    page = _gpage([{"n": 1}], has_next=False, cursor=None, errors=errors)
    strict = paginate_graphql(FakeTransport(lambda r: page), GQL, connection_path=PATH, limit=5)
    assert (
        isinstance(strict, Response)
        and strict.graphql_errors
        and len(json.loads(strict.body)["data"]["repository"]["pullRequests"]["nodes"]) == 1
    )

    lenient_request = GraphQLRequest.of(DOC, {"owner": "o"}, partial_ok=True)
    lenient = paginate_graphql(
        FakeTransport(lambda r: page), lenient_request, connection_path=PATH, limit=5
    )
    assert isinstance(lenient, Response) and lenient.graphql_errors[0].message == "partial"


def test_a_missing_connection_is_a_defect() -> None:
    bad = Response(200, (), json.dumps({"data": {"repository": None}}), "http")
    out = paginate_graphql(FakeTransport(lambda r: bad), GQL, connection_path=PATH, limit=5)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.ADAPTER_DEFECT
