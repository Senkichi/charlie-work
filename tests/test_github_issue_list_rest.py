"""One REST/ETag open-issues read per pass (issue #2443).

Parity: REST-mapped issues equal what gh's GraphQL dialect printed for the same
issues. Call counting: ``issue_list`` makes zero GraphQL calls, however many
label queries a pass runs, and an unchanged repo's second pass costs only 304s
(real ``HttpAdapter`` + ``FileEtagCache`` over a scripted server). Failure and
kill switch both restore the GraphQL path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from _fake_transport import FakeAdapter, FakeRaw, graphql_ok, make_github, ok
from charlie_work.github_capabilities import http_cache
from charlie_work.github_capabilities.issue_list_rest import (
    KILL_SWITCH_ENV,
    filter_by_labels,
    map_rest_issue,
)
from charlie_work.github_capabilities.issues import ISSUE_LIST_FIELDS
from charlie_work.github_transport import GraphQLRequest, RestRequest
from charlie_work.github_transport import gh_json_fields as g
from charlie_work.github_transport.http_adapter import HttpAdapter
from charlie_work.instrumentation import query_events
from charlie_work.layout import state_file_path


def _rest_issue(number: int, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "number": number,
        "title": f"issue {number}",
        "html_url": f"https://github.com/octo/hello/issues/{number}",
        "body": f"body {number}",
        "state": "open",
        "labels": [],
        "user": {"login": "alice", "node_id": "U_alice", "type": "User"},
        "assignees": [],
        "milestone": None,
        "created_at": "2026-10-01T00:00:00Z",
        "updated_at": "2026-10-02T00:00:00Z",
    }
    base.update(over)
    return base


def _graphql_node(number: int, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "number": number,
        "title": f"issue {number}",
        "url": f"https://github.com/octo/hello/issues/{number}",
        "body": f"body {number}",
        "state": "OPEN",
        "labels": {"nodes": []},
        "author": {"__typename": "User", "login": "alice", "id": "U_alice", "name": None},
        "createdAt": "2026-10-01T00:00:00Z",
        "updatedAt": "2026-10-02T00:00:00Z",
    }
    base.update(over)
    return base


def _rest_label(name: str, desc: str | None = None) -> dict[str, Any]:
    return {"id": 1, "node_id": f"L_{name}", "name": name, "description": desc, "color": "ededed"}


def _gql_label(name: str, desc: str | None = None) -> dict[str, Any]:
    return {"id": f"L_{name}", "name": name, "description": desc, "color": "ededed"}


def test_parity_with_the_graphql_dialect() -> None:
    pairs = [
        (_rest_issue(1), _graphql_node(1)),
        (
            _rest_issue(
                2,
                labels=[_rest_label("agent:queued", "ready"), _rest_label("bug")],
                assignees=[{"login": "bob", "node_id": "U_bob"}],
                milestone={"title": "m1"},
            ),
            _graphql_node(
                2, labels={"nodes": [_gql_label("agent:queued", "ready"), _gql_label("bug")]}
            ),
        ),
        # body None: GraphQL's body is "" where REST sends null.
        (_rest_issue(3, body=None), _graphql_node(3, body="")),
        (
            _rest_issue(4, user={"login": "dependabot[bot]", "node_id": "B_dep", "type": "Bot"}),
            _graphql_node(4, author={"__typename": "Bot", "login": "dependabot", "id": "B_dep"}),
        ),
        (_rest_issue(5, state="closed"), _graphql_node(5, state="CLOSED")),
    ]
    for rest, node in pairs:
        assert map_rest_issue(rest) == g.normalize_node("issue", node, ISSUE_LIST_FIELDS)


def test_label_filter_is_and_and_case_insensitive() -> None:
    issues = [
        map_rest_issue(_rest_issue(1, labels=[_rest_label("A"), _rest_label("b")])),
        map_rest_issue(_rest_issue(2, labels=[_rest_label("a")])),
        map_rest_issue(_rest_issue(3)),
    ]

    assert [i["number"] for i in filter_by_labels(issues, ("a", "B"))] == [1]
    assert [i["number"] for i in filter_by_labels(issues, ("a",))] == [1, 2]
    assert [i["number"] for i in filter_by_labels(issues, ())] == [1, 2, 3]


class _Server:
    """Scripted REST server: paged ``issues`` with ETags, 304 on a matching
    ``If-None-Match``, and a counter for every request class."""

    def __init__(self, items: list[dict[str, Any]], per_page: int = 100) -> None:
        self.items = items
        self.per_page = per_page
        self.graphql = 0
        self.full = 0
        self.not_modified = 0
        self.sock = None
        self.timeout = 0.0
        self._pending: tuple[str, dict[str, str]] | None = None

    def connect(self) -> None:  # pragma: no cover - never used (factory returns self)
        pass

    def close(self) -> None:
        pass

    def request(self, method: str, path: str, body=None, headers=None) -> None:
        self._pending = (path, dict(headers or {}))
        if method == "POST":
            self.graphql += 1

    def getresponse(self) -> FakeRaw:
        assert self._pending is not None
        path, headers = self._pending
        if "graphql" in path:
            return FakeRaw(200, body=json.dumps({"data": {}}).encode())
        parts = urlsplit(path)
        query = parse_qs(parts.query)
        page = int(query.get("page", ["1"])[0])
        start = (page - 1) * self.per_page
        chunk = self.items[start : start + self.per_page]
        body = json.dumps(chunk).encode()
        etag = '"' + hashlib.sha1(body).hexdigest() + '"'
        resp_headers = {"ETag": etag}
        if start + self.per_page < len(self.items):
            nxt = dict((k, v[0]) for k, v in query.items())
            nxt["page"] = str(page + 1)
            qs = "&".join(f"{k}={v}" for k, v in nxt.items())
            resp_headers["Link"] = f'<https://api.github.com{parts.path}?{qs}>; rel="next"'
        if headers.get("If-None-Match") == etag:
            self.not_modified += 1
            return FakeRaw(304, resp_headers)
        self.full += 1
        return FakeRaw(200, resp_headers, body)


def _github_over(server: _Server, tmp_path: Path):
    cache = http_cache.FileEtagCache(tmp_path / "etag.json")
    http = HttpAdapter(cache=cache, connection_factory=lambda host, timeout: server)
    gh_adapter = FakeAdapter("gh", token="tok-1")
    gh, _http, _gh = make_github(tmp_path, http=http, gh=gh_adapter)  # type: ignore[arg-type]
    return gh


def _items() -> list[dict[str, Any]]:
    items = [_rest_issue(n, labels=[_rest_label("agent:queued")]) for n in range(1, 151)]
    # PR rows share the endpoint; they must be dropped.
    items.insert(3, _rest_issue(900, pull_request={"url": "x"}))
    items.insert(120, _rest_issue(901, pull_request={"url": "x"}))
    items.append(_rest_issue(902, labels=[_rest_label("other")]))
    return items


def test_pass_makes_zero_graphql_issue_lists_and_paginates(tmp_path: Path) -> None:
    server = _Server(_items())
    gh = _github_over(server, tmp_path)

    queued = gh.issue_list(["agent:queued"])
    everything = gh.issue_list(state="open")
    other = gh.issue_list(labels=["other"])
    again = gh.issue_list(["agent:queued"])

    assert [i["number"] for i in queued] == list(range(1, 151))
    assert len(everything) == 151  # 150 + #902, PRs dropped
    assert [i["number"] for i in other] == [902]
    assert again is queued  # per-pass cache still answers repeats
    assert server.graphql == 0
    assert server.full == 2  # 153 rows -> two pages, fetched once for the whole pass


def test_unchanged_repo_second_pass_costs_only_304s(tmp_path: Path) -> None:
    server = _Server(_items())
    gh = _github_over(server, tmp_path)
    first = gh.issue_list(["agent:queued"])

    gh.invalidate_list_cache()  # next pass
    second = gh.issue_list(["agent:queued"])
    gh.issue_list(["other"])

    assert second == first
    assert server.full == 2
    assert server.not_modified == 2
    assert server.graphql == 0


def test_rest_failure_falls_back_to_graphql_with_an_event(tmp_path: Path) -> None:
    node = _graphql_node(7, labels={"nodes": [_gql_label("agent:queued")]})

    def handler(request: Any) -> Any:
        if isinstance(request, RestRequest):
            return ok({"message": "boom"}, status=500)
        assert isinstance(request, GraphQLRequest)
        return graphql_ok({"repository": {"issues": {"nodes": [node], "pageInfo": {}}}})

    http = FakeAdapter("http", handler=handler)
    gh, _http, _ = make_github(tmp_path, http=http)

    result = gh.issue_list(["agent:queued"])

    assert [i["number"] for i in result] == [7]
    events = query_events(
        state_file_path(tmp_path / ".var" / "charlie-work"), kind="github_transport_fallback"
    )
    assert any(e["payload"]["reason"] == "rest_issue_list_failed" for e in events)


def test_kill_switch_restores_graphql(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KILL_SWITCH_ENV, "off")
    node = _graphql_node(8, labels={"nodes": [_gql_label("agent:queued")]})
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        return graphql_ok({"repository": {"issues": {"nodes": [node], "pageInfo": {}}}})

    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", handler=handler))

    result = gh.issue_list(["agent:queued"])

    assert [i["number"] for i in result] == [8]
    assert seen and all(isinstance(r, GraphQLRequest) for r in seen)


def test_closed_and_all_states_stay_on_graphql(tmp_path: Path) -> None:
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        return graphql_ok({"repository": {"issues": {"nodes": [], "pageInfo": {}}}})

    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", handler=handler))

    gh.issue_list(state="all")

    assert seen and all(isinstance(r, GraphQLRequest) for r in seen)


def test_pr_list_drops_the_status_check_rollup_but_with_checks_keeps_it() -> None:
    from charlie_work.github_capabilities.pull_requests import (
        PR_LIST_FIELDS,
        PR_LIST_WITH_CHECKS_FIELDS,
    )

    assert "statusCheckRollup" not in PR_LIST_FIELDS.split(",")
    with_checks = PR_LIST_WITH_CHECKS_FIELDS.split(",")
    assert "statusCheckRollup" in with_checks
    assert [f for f in with_checks if f != "statusCheckRollup"] == PR_LIST_FIELDS.split(",")


def test_absent_rollup_is_unknown_not_stale_empty_checks() -> None:
    from datetime import UTC, datetime

    from charlie_work.dead_worker_sweep.decide_common import is_pre_review_rework_candidate

    now = datetime(2026, 10, 6, tzinfo=UTC)
    old = {"updatedAt": "2026-01-01T00:00:00Z"}

    assert is_pre_review_rework_candidate(old, 30, now) == (False, "")
    assert is_pre_review_rework_candidate({**old, "statusCheckRollup": []}, 30, now) == (
        True,
        "stale_empty_checks",
    )
