"""``RunListRead`` reads the workflow's own runs route, never an offset-paged repo list (r5).

``gh run list --workflow W`` resolves W and reads
``actions/workflows/{id}/runs`` with ``branch``/``status``/``event`` filtered
server side. The repo-wide ``actions/runs`` list shifts between page reads (a
run queued between two reads repeats the boundary entry), which made
``cancel_superseded_runs`` cancel the run it keeps.
"""

from __future__ import annotations

import json

import pytest
from _fake_transport import FakeTransport, ok

from charlie_work.github_transport import FailureKind, Response, TransportFailure
from charlie_work.github_transport.json_read import JsonRead, RunListRead
from charlie_work.github_transport.pagination import MAX_PAGES

WORKFLOWS = "repos/{owner}/{repo}/actions/workflows"


def _run(i: int, ts: str = "2026-09-30T10:00:00Z") -> dict:
    return {"id": i, "name": "CI", "status": "queued", "created_at": ts, "conclusion": None}


def _query(request) -> dict[str, str]:  # noqa: ANN001
    return dict(request.query)


def _ids(out) -> list[int]:  # noqa: ANN001
    assert isinstance(out, Response), out
    return [r["databaseId"] for r in json.loads(out.body)]


def test_a_run_queued_between_pages_does_not_repeat_the_boundary_run() -> None:
    """The r5 probe: the newest run sits at the last slot of page 1, a new run
    shifts it onto page 2, and the keep-newest consumer must still see it once."""
    pages = {
        1: [_run(1000 + k) for k in range(99)] + [_run(500, "2026-09-30T11:00:00Z")],
        2: [_run(500, "2026-09-30T11:00:00Z"), _run(400, "2026-09-30T10:00:00Z")],
    }
    transport = FakeTransport(
        lambda r: ok({"workflow_runs": pages[int(_query(r).get("page", 1))]})
    )
    out = RunListRead("databaseId,createdAt", branch="main", limit=101).execute(
        transport, "o", "r"
    )
    ids = _ids(out)
    assert ids.count(500) == 1 and len(ids) == len(set(ids)) == 101


def test_a_workflow_file_reads_its_own_runs_route_with_server_side_filters() -> None:
    transport = FakeTransport(lambda r: ok({"workflow_runs": [_run(1), _run(2)]}))
    read = RunListRead(
        "databaseId", workflow="ci.yml", branch="main", status="queued", event="push", limit=100
    )
    out = read.execute(transport, "o", "r")
    assert _ids(out) == [1, 2]
    (request,) = transport.requests  # one request, no workflow lookup for a file name
    assert request.route == "repos/{owner}/{repo}/actions/workflows/ci.yml/runs"
    assert _query(request) == {
        "per_page": "100",
        "branch": "main",
        "status": "queued",
        "event": "push",
    }


def test_a_workflow_name_is_resolved_then_its_runs_are_read() -> None:
    def respond(request):  # noqa: ANN001
        if request.route == WORKFLOWS:
            return ok(
                {
                    "workflows": [
                        {"id": 7, "name": "Other"},
                        {"id": 42, "name": "CI", "state": "active"},
                    ]
                }
            )
        assert request.route == f"{WORKFLOWS}/42/runs"
        return ok({"workflow_runs": [_run(9)]})

    transport = FakeTransport(respond)
    out = RunListRead("databaseId", workflow="CI", branch="main", limit=5).execute(
        transport, "o", "r"
    )
    assert _ids(out) == [9]
    assert [r.route for r in transport.requests] == [WORKFLOWS, f"{WORKFLOWS}/42/runs"]
    assert _query(transport.requests[-1])["per_page"] == "5"
    assert "page" not in _query(transport.requests[-1])


@pytest.mark.parametrize(
    "matches",
    [
        [],
        [{"id": 1, "name": "CI", "state": "active"}, {"id": 2, "name": "CI", "state": "active"}],
    ],
)
def test_an_unknown_or_ambiguous_workflow_name_is_a_failed_read_not_an_empty_list(
    matches: list,
) -> None:
    transport = FakeTransport(lambda r: ok({"workflows": matches}))
    out = RunListRead("databaseId", workflow="CI").execute(transport, "o", "r")
    assert isinstance(out, TransportFailure)
    assert out.kind is FailureKind.ADAPTER_DEFECT
    assert len(transport.requests) == 1  # no runs read


def _resolved_route(workflow: str, workflows: list[dict]) -> str | TransportFailure:
    def respond(request):  # noqa: ANN001
        if request.route == WORKFLOWS:
            return ok({"workflows": workflows})
        return ok({"workflow_runs": [_run(9)]})

    transport = FakeTransport(respond)
    out = RunListRead("databaseId", workflow=workflow).execute(transport, "o", "r")
    if isinstance(out, TransportFailure):
        return out
    return transport.requests[-1].route


def test_a_workflow_name_matches_case_insensitively_like_gh() -> None:
    route = _resolved_route("ci", [{"id": 42, "name": "CI", "state": "active"}])
    assert route == f"{WORKFLOWS}/42/runs"


def test_a_disabled_same_named_workflow_is_ignored_like_gh() -> None:
    route = _resolved_route(
        "CI",
        [
            {"id": 1, "name": "CI", "state": "disabled_manually"},
            {"id": 2, "name": "CI", "state": "active"},
        ],
    )
    assert route == f"{WORKFLOWS}/2/runs"


def test_an_uppercase_file_suffix_is_a_file_not_a_name() -> None:
    transport = FakeTransport(lambda r: ok({"workflow_runs": [_run(1)]}))
    RunListRead("databaseId", workflow="CI.YML").execute(transport, "o", "r")
    (request,) = transport.requests  # no workflow lookup
    assert request.route == f"{WORKFLOWS}/CI.YML/runs"


def test_a_limit_over_one_page_pages_the_workflow_route_and_dedupes_by_run_id() -> None:
    pages = {
        1: [_run(i) for i in range(100)],
        2: [_run(99), _run(100), _run(101)],  # 99 repeats: the list shifted between reads
    }
    transport = FakeTransport(
        lambda r: ok({"workflow_runs": pages[int(_query(r).get("page", 1))]})
    )
    out = RunListRead("databaseId", workflow="ci.yml", limit=150).execute(transport, "o", "r")
    assert _ids(out) == list(range(102))
    assert [r.route for r in transport.requests] == [
        "repos/{owner}/{repo}/actions/workflows/ci.yml/runs"
    ] * 2
    assert [_query(r).get("page") for r in transport.requests] == [None, "2"]


def test_paging_stops_once_limit_runs_are_collected() -> None:
    transport = FakeTransport(lambda r: ok({"workflow_runs": [_run(i) for i in range(5)]}))
    out = RunListRead("databaseId", limit=5).execute(transport, "o", "r")
    assert _ids(out) == [0, 1, 2, 3, 4]
    assert len(transport.requests) == 1


def test_an_unbounded_listing_is_an_adapter_defect_not_a_truncated_list() -> None:
    counter = iter(range(10**6))
    transport = FakeTransport(
        lambda r: ok({"workflow_runs": [_run(next(counter)) for _ in range(100)]})
    )
    out = RunListRead("databaseId", workflow="ci.yml", limit=10**6).execute(transport, "o", "r")
    assert isinstance(out, TransportFailure)
    assert out.kind is FailureKind.ADAPTER_DEFECT
    assert len(transport.requests) == MAX_PAGES


def _issues_page(nodes: list, more: bool) -> dict:
    return {
        "data": {
            "repository": {
                "issues": {
                    "nodes": nodes,
                    "pageInfo": {"hasNextPage": more, "endCursor": "c" if more else None},
                }
            }
        }
    }


def test_a_repeated_node_across_graphql_pages_is_dropped() -> None:
    """Defensive dedupe on the GraphQL list path."""
    replies = iter(
        [
            _issues_page([{"id": "A", "number": 1}, {"id": "B", "number": 2}], True),
            _issues_page([{"id": "B", "number": 2}, {"id": "C", "number": 3}], False),
        ]
    )
    transport = FakeTransport(lambda r: ok(next(replies)))
    out = JsonRead("issue", "list", "number", limit=10).execute(transport, "o", "r")
    assert isinstance(out, Response), out
    assert [n["number"] for n in json.loads(out.body)] == [1, 2, 3]
