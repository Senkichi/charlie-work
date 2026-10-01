"""``RunListRead`` pages until ``limit`` runs match the workflow filter (r4).

``gh run list --workflow W --limit N`` filters before it limits, so a backlog
of other workflows' runs on the same branch must not hide W's runs.
"""

from __future__ import annotations

import json

from _fake_transport import FakeTransport, ok

from charlie_work.github_transport import FailureKind, Response, TransportFailure
from charlie_work.github_transport.json_read import RunListRead
from charlie_work.github_transport.pagination import MAX_PAGES


def _run(i: int, name: str) -> dict:
    return {
        "id": i,
        "name": name,
        "path": f".github/workflows/{name.lower()}.yml",
        "status": "queued",
    }


def _page_of(request) -> int:  # noqa: ANN001
    return int(dict(request.query).get("page", 1))


def test_workflow_filter_pages_past_other_workflows_runs() -> None:
    pages = {
        1: [_run(i, "Other") for i in range(100)],
        2: [_run(1000 + i, "Other") for i in range(100)],
        3: [_run(2000, "CI"), _run(2001, "CI")],
    }
    transport = FakeTransport(lambda r: ok({"workflow_runs": pages[_page_of(r)]}))
    out = RunListRead("databaseId", workflow="CI", limit=100).execute(transport, "o", "r")
    assert isinstance(out, Response)
    assert [r["databaseId"] for r in json.loads(out.body)] == [2000, 2001]
    assert [_page_of(r) for r in transport.requests] == [1, 2, 3]


def test_paging_stops_once_limit_matches_are_collected() -> None:
    transport = FakeTransport(lambda r: ok({"workflow_runs": [_run(i, "CI") for i in range(100)]}))
    out = RunListRead("databaseId", workflow="CI", limit=5).execute(transport, "o", "r")
    assert isinstance(out, Response) and len(json.loads(out.body)) == 5
    assert len(transport.requests) == 1


def test_an_unbounded_backlog_is_an_adapter_defect_not_a_truncated_list() -> None:
    transport = FakeTransport(lambda r: ok({"workflow_runs": [_run(1, "Other")] * 100}))
    out = RunListRead("databaseId", workflow="CI", limit=10).execute(transport, "o", "r")
    assert isinstance(out, TransportFailure)
    assert out.kind is FailureKind.ADAPTER_DEFECT
    assert len(transport.requests) == MAX_PAGES
