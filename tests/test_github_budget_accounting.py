"""Per-capability GitHub spend accounting from ``x-ratelimit-used`` deltas (#2439).

Driven at the ``Adapter`` seam with scripted fakes: the clock, sleep and jitter
are injected and every guard gets its own ``UsedCursor`` so nothing leaks
between tests through the process-wide default.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.api_budget import GitHubSpendEntry
from charlie_work.github_capabilities._send import send as capability_send
from charlie_work.github_transport import (
    Adapters,
    GraphQLRequest,
    GuardedTransport,
    RateBudgetHolder,
    RestRequest,
    capability_name,
    capability_scope,
    emit_github_budget_pass,
)
from charlie_work.github_transport.guarded import UsedCursor
from charlie_work.instrumentation import close_db, query_events

from _fake_transport import FakeAdapter, Runtime, Sleeps, graphql_ok, ok

GET_PR = RestRequest.of("GET", "repos/{owner}/{repo}/pulls/7")
LIST_ISSUES = RestRequest.of("GET", "repos/{owner}/{repo}/issues")
QUERY = GraphQLRequest.of("query Q { viewer { login } }")


def _h(used: int, *, resource: str = "core", reset: int = 9000, remaining: int | None = None):
    return {
        "X-RateLimit-Limit": "5000",
        "X-RateLimit-Used": str(used),
        "X-RateLimit-Remaining": str(5000 - used if remaining is None else remaining),
        "X-RateLimit-Reset": str(reset),
        "X-RateLimit-Resource": resource,
    }


def _guard(*script, state_path: Path | None = None, cursor: UsedCursor | None = None):
    http = FakeAdapter("http", list(script))
    gh = FakeAdapter("gh", token="tok-1")
    guard = GuardedTransport(
        Adapters(http=http, gh=gh),
        runtime=Runtime(gh_max_retries=2),
        state_path=state_path,
        resolve_owner_repo=lambda: ("octo", "hello"),
        budget=RateBudgetHolder(cursor if cursor is not None else UsedCursor()),
        sleep=Sleeps(),
        jitter=lambda lo, hi: 0.0,
        now=lambda: 1000.0,
    )
    return guard


def _by_cap(spend) -> dict[tuple[str, str], tuple[int, int]]:
    return {(e.capability, e.resource): (e.requests, e.points) for e in spend.entries}


def test_points_are_header_deltas_per_capability_and_resource() -> None:
    guard = _guard(
        ok({}, headers=_h(100)),  # baseline: nothing is attributed to it
        ok({}, headers=_h(103)),
        ok({}, headers=_h(104)),
        ok({}, headers=_h(500, resource="graphql", remaining=4500)),
        ok({}, headers=_h(520, resource="graphql", remaining=4480)),
    )
    with capability_scope("issues"):
        guard.send(GET_PR)  # baseline
        guard.send(GET_PR)  # +3
    with capability_scope("checks"):
        guard.send(GET_PR)  # +1
    with capability_scope("pull_requests"):
        guard.send(QUERY)  # graphql baseline
        guard.send(QUERY)  # +20
    assert _by_cap(guard.take_spend()) == {
        ("issues", "core"): (2, 3),
        ("checks", "core"): (1, 1),
        ("pull_requests", "graphql"): (2, 20),
    }


def test_a_response_without_headers_counts_a_request_but_no_points() -> None:
    guard = _guard(ok({}))
    guard.send(GET_PR)
    (entry,) = guard.take_spend().entries
    assert entry == GitHubSpendEntry("rest.pulls", "core", 1, 0, 1)


def test_unscoped_requests_are_named_from_the_request() -> None:
    guard = _guard(
        ok({}, headers=_h(1)),
        ok({}, headers=_h(2)),
        graphql_ok({}),
    )
    guard.send(GET_PR)
    guard.send(LIST_ISSUES)
    guard.send(QUERY)
    caps = {e.capability for e in guard.take_spend().entries}
    assert caps == {"rest.pulls", "rest.issues", "graphql.query"}


def test_a_retried_call_counts_every_attempt() -> None:
    guard = _guard(
        ok({}, headers=_h(10), status=503),
        ok({}, headers=_h(11)),
    )
    guard.send(GET_PR)
    assert _by_cap(guard.take_spend()) == {("rest.pulls", "core"): (2, 1)}


def test_guards_sharing_a_cursor_do_not_double_count_concurrent_spend() -> None:
    cursor = UsedCursor()
    a = _guard(ok({}, headers=_h(10)), ok({}, headers=_h(14)), cursor=cursor)
    b = _guard(ok({}, headers=_h(12)), cursor=cursor)
    a.send(GET_PR)  # baseline 10
    b.send(GET_PR)  # +2 (other lanes' spend, attributed once)
    a.send(GET_PR)  # +2 (12 -> 14)
    total = a.take_spend().points + b.take_spend().points
    assert total == 4  # == 14 - 10, the observed counter delta


def test_a_stale_or_out_of_order_response_counts_nothing() -> None:
    guard = _guard(
        ok({}, headers=_h(50)),
        ok({}, headers=_h(55)),
        ok({}, headers=_h(52)),  # late arrival of an older response
        ok({}, headers=_h(60, reset=8000)),  # an older window
    )
    for _ in range(4):
        guard.send(GET_PR)
    assert guard.take_spend().points == 5


def test_a_new_window_rebaselines() -> None:
    guard = _guard(
        ok({}, headers=_h(4000)),
        ok({}, headers=_h(3, reset=12000)),
        ok({}, headers=_h(5, reset=12000)),
    )
    for _ in range(3):
        guard.send(GET_PR)
    assert guard.take_spend().points == 2


def test_take_spend_resets_so_each_pass_reads_its_own_delta() -> None:
    guard = _guard(ok({}, headers=_h(1)), ok({}, headers=_h(4)), ok({}, headers=_h(9)))
    guard.send(GET_PR)
    guard.send(GET_PR)
    assert guard.take_spend().points == 3
    guard.send(GET_PR)
    assert guard.take_spend().points == 5
    assert guard.take_spend().entries == ()


def test_capability_name_is_snake_cased_from_the_collaborator_class() -> None:
    class PullRequests:
        pass

    assert capability_name(PullRequests()) == "pull_requests"


def test_the_send_helper_scopes_the_collaborators_capability() -> None:
    guard = _guard(ok({}, headers=_h(1)), ok({}, headers=_h(3)))

    class Checks:
        _transport_v2 = guard

    collab = Checks()
    capability_send(collab, GET_PR)
    capability_send(collab, GET_PR)
    assert _by_cap(guard.take_spend()) == {("checks", "core"): (2, 2)}


# -- the github_budget_pass event ---------------------------------------------


class _Client:
    def __init__(self, guard: GuardedTransport) -> None:
        self._transport_v2 = guard


def test_budget_pass_event_payload_shape(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    guard = _guard(
        ok({}, headers=_h(100)),
        ok({}, headers=_h(105)),
        ok({}, headers=_h(106)),
        ok({}, headers=_h(900, resource="graphql", remaining=4100)),
        ok({}, headers=_h(902, resource="graphql", remaining=4098)),
    )
    with capability_scope("issues"):
        guard.send(GET_PR)
        guard.send(GET_PR)
    with capability_scope("checks"):
        guard.send(GET_PR)
        guard.send(QUERY)
        guard.send(QUERY)
    try:
        assert emit_github_budget_pass(_Client(guard), state, pass_kind="lane", repo_key="o/r")
        (event,) = query_events(state, kind="github_budget_pass")
    finally:
        close_db(state)
    payload = event["payload"]
    assert payload["pass_kind"] == "lane" and payload["repo_key"] == "o/r"
    assert (payload["requests"], payload["points"]) == (5, 8)
    assert payload["points_by_resource"] == {"core": 6, "graphql": 2}
    # heaviest first, every row carries both counters
    assert [
        (c["capability"], c["resource"], c["requests"], c["points"])
        for c in payload["capabilities"]
    ] == [
        ("issues", "core", 2, 5),
        ("checks", "graphql", 2, 2),
        ("checks", "core", 1, 1),
    ]
    assert payload["observed"]["graphql"] == {"remaining": 4098, "limit": 5000, "reset": 9000}
    assert payload["observed"]["core"]["remaining"] == 4894
    # the emit drained the spend: the next pass starts from zero
    assert guard.take_spend().entries == ()


def test_budget_pass_is_skipped_for_a_client_without_a_guarded_transport(tmp_path: Path) -> None:
    assert emit_github_budget_pass(object(), tmp_path / "state.json", pass_kind="reap") is False


# -- primary rate limit -> event ----------------------------------------------


def test_primary_rate_limit_emits_one_event_across_retries(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    limited = ok(
        {"message": "API rate limit exceeded"},
        headers=_h(5000, remaining=0, reset=9500),
        status=403,
    )
    guard = _guard(limited, limited, limited, state_path=state)
    with capability_scope("issues"):
        guard.send(GET_PR)  # 3 attempts, one exhausted window
    try:
        (event,) = query_events(state, kind="github_rate_limited")
    finally:
        close_db(state)
    assert event["payload"] == {
        "resource": "core",
        "capability": "issues",
        "request": "GET repos/octo/hello/pulls/7",
        "status": 403,
        "limit": 5000,
        "remaining": 0,
        "reset": 9500,
    }


def test_a_plain_403_is_not_a_rate_limit_event(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    guard = _guard(ok({"message": "Forbidden"}, status=403), state_path=state)
    guard.send(GET_PR)
    try:
        assert query_events(state, kind="github_rate_limited") == []
    finally:
        close_db(state)


def test_a_new_window_emits_again(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    first = ok({}, headers=_h(5000, remaining=0, reset=9500), status=403)
    second = ok({}, headers=_h(5000, remaining=0, reset=13100), status=403)
    guard = _guard(first, second, state_path=state)
    guard._runtime = Runtime(gh_max_retries=0)  # type: ignore[assignment]
    guard.send(GET_PR)
    guard.send(GET_PR)
    try:
        assert len(query_events(state, kind="github_rate_limited")) == 2
    finally:
        close_db(state)


# -- fleet wiring -------------------------------------------------------------


def test_a_fleet_lane_emits_one_budget_pass_event_even_when_the_lane_raises(
    tmp_path: Path,
) -> None:
    from unittest.mock import MagicMock

    from charlie_work.config import OrchestratorConfig
    from charlie_work.fleet_lanes import _run_fleet_repo_lane
    from _fake_transport import make_github

    state = tmp_path / "fleet" / "state.json"
    state.parent.mkdir()
    gh, _, _ = make_github(
        tmp_path,
        http=FakeAdapter("http", [ok([], headers=_h(7)), ok([], headers=_h(9))]),
    )
    gh._transport_v2.budget._cursor = UsedCursor()  # isolate from the process cursor

    def loop(*args, **kwargs):
        gh.run(["api", "repos/{owner}/{repo}/issues"])
        gh.run(["api", "repos/{owner}/{repo}/issues"])
        raise RuntimeError("lane blew up")

    app = MagicMock()
    app.gh = gh
    app.loop.side_effect = loop
    lock = MagicMock()
    try:
        try:
            _run_fleet_repo_lane(
                "octo/hello",
                app,
                OrchestratorConfig(),
                lock,
                work_only=False,
                drain=False,
                limit=1,
                merge=True,
                ensure_labels=False,
                fleet_state_path=state,
            )
        except RuntimeError:
            pass
        (event,) = query_events(state, kind="github_budget_pass")
    finally:
        close_db(state)
    payload = event["payload"]
    assert (payload["pass_kind"], payload["repo_key"]) == ("lane", "octo/hello")
    assert (payload["requests"], payload["points"]) == (2, 2)
    assert payload["capabilities"][0]["capability"] == "rest.issues"
    lock.release.assert_called_once()
