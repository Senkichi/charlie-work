"""Per-capability GitHub spend accounting from ``x-ratelimit-used`` deltas (#2439).

Driven at the ``Adapter`` seam with scripted fakes: the clock, sleep and jitter
are injected and every guard gets its own ``UsedCursor`` so nothing leaks
between tests through the process-wide default.
"""

from __future__ import annotations

import threading
from pathlib import Path

from charlie_work.api_budget import GitHubSample, GitHubSpendEntry
from charlie_work.github_capabilities._send import send as capability_send
from charlie_work.github_transport import (
    Adapters,
    CliCommand,
    CliRequest,
    GraphQLError,
    GraphQLRequest,
    GuardedTransport,
    RateBudgetHolder,
    Response,
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


def _guard(
    *script,
    state_path: Path | None = None,
    cursor: UsedCursor | None = None,
    retries: int = 2,
    gh: FakeAdapter | None = None,
):
    http = FakeAdapter("http", list(script))
    gh = gh if gh is not None else FakeAdapter("gh", token="tok-1")
    guard = GuardedTransport(
        Adapters(http=http, gh=gh),
        runtime=Runtime(gh_max_retries=retries),
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
    """Two lanes both baseline, then both see the same counter move (10 -> 14).

    With a shared cursor the first sighting claims the +4 and the second sees
    nothing new (total 4 == the observed delta). Per-holder cursors would each
    claim +4 (total 8). The interleave is explicit, so it is deterministic.
    """
    cursor = UsedCursor()
    a = _guard(ok({}, headers=_h(10)), ok({}, headers=_h(14)), cursor=cursor)
    b = _guard(ok({}, headers=_h(10)), ok({}, headers=_h(14)), cursor=cursor)
    a.send(GET_PR)  # baseline 10
    b.send(GET_PR)  # same window, same counter: nothing to attribute
    a.send(GET_PR)  # 10 -> 14
    b.send(GET_PR)  # 14 again: already attributed to a
    assert a.take_spend().points + b.take_spend().points == 4


def test_guards_sharing_a_cursor_stay_exact_under_real_threads() -> None:
    cursor = UsedCursor()
    guards = [
        _guard(*(ok({}, headers=_h(n)) for n in range(100, 140)), cursor=cursor) for _ in range(4)
    ]
    barrier = threading.Barrier(len(guards))

    def run(guard: GuardedTransport) -> None:
        barrier.wait()
        for _ in range(40):
            guard.send(GET_PR)

    threads = [threading.Thread(target=run, args=(g,)) for g in guards]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # The cursor only advances, so what is claimed in total is last - first,
    # however the four replays of 100..139 interleave.
    assert sum(g.take_spend().points for g in guards) == 39


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


def test_alternating_windows_each_attribute_their_own_deltas() -> None:
    """GitHub can answer one resource from two live windows at once (#2488).

    The cursor keys its mark by (resource, reset), so a response naming the
    older window still deltas against that window's baseline instead of being
    dropped -- otherwise every older-window sample after the first
    newer-window sighting attributes nothing and ``points`` undercounts."""
    guard = _guard(
        ok({}, headers=_h(10, reset=9000)),  # window A baseline
        ok({}, headers=_h(50, reset=9500)),  # window B baseline
        ok({}, headers=_h(12, reset=9000)),  # A: +2
        ok({}, headers=_h(55, reset=9500)),  # B: +5
        ok({}, headers=_h(13, reset=9000)),  # A: +1
        ok({}, headers=_h(57, reset=9500)),  # B: +2
    )
    for _ in range(6):
        guard.send(GET_PR)
    assert _by_cap(guard.take_spend()) == {("rest.pulls", "core"): (6, 10)}


def _sample(used: int, *, reset: int, resource: str = "core") -> GitHubSample:
    return GitHubSample(resource, used, reset, None, None)


def test_successive_windows_leave_the_cursor_bounded() -> None:
    """Hourly windows marching forward: dead-window marks are pruned, so the
    process-wide cursor never holds more than a resource's two live windows
    (#2488 rework: the (resource, reset) keying removed the old
    one-mark-per-resource bound)."""
    cursor = UsedCursor()
    for i in range(50):
        cursor.advance(_sample(10 + i, reset=9000 + 3600 * i))
        assert len(cursor._value.marks) <= 2
    assert [m.reset_epoch for m in cursor._value.marks] == [9000 + 3600 * 48, 9000 + 3600 * 49]


def test_a_late_sample_for_a_pruned_window_rebaselines_at_zero() -> None:
    """A response naming a window more than 3600s behind the newest reset
    counts 0 -- its ``used`` includes spend this process never saw -- and is
    not retained, so a second late sighting baselines at 0 again instead of
    summing a dead window's counter."""
    cursor = UsedCursor()
    cursor.advance(_sample(10, reset=9000))
    cursor.advance(_sample(3, reset=12600))
    cursor.advance(_sample(7, reset=16200))  # new frontier: 9000 is now prunable
    assert cursor.advance(_sample(80, reset=9000)) == 0
    assert cursor.advance(_sample(85, reset=9000)) == 0  # still dead: re-baseline
    # the window exactly one behind the frontier stays live and attributes
    assert cursor.advance(_sample(5, reset=12600)) == 2
    assert [m.reset_epoch for m in cursor._value.marks] == [16200, 12600]


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
    guard = _guard(limited, limited, limited, state_path=state, cursor=UsedCursor())
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
    guard = _guard(first, second, state_path=state, retries=0, cursor=UsedCursor())
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


def _isolated_gh(tmp_path: Path):
    """A real ``GitHub`` over scripted fakes with its own used-cursor."""
    from _fake_transport import make_github

    gh, _, _ = make_github(
        tmp_path,
        http=FakeAdapter("http", [ok([], headers=_h(7)), ok([], headers=_h(9))]),
    )
    gh._transport_v2.budget._cursor = UsedCursor()
    return gh


def _two_requests(gh) -> None:
    gh.run(["api", "repos/{owner}/{repo}/issues"])
    gh.run(["api", "repos/{owner}/{repo}/issues"])


def _reap_once(tmp_path: Path, monkeypatch, sweep) -> Path:
    """Run ``_run_fleet_reap_sweep`` over one real-registry repo whose app's
    ``_run_review_reap_sweeps`` is ``sweep(gh)``; return the fleet state path."""
    import json
    from unittest.mock import MagicMock

    from charlie_work import fleet_lanes, layout
    from charlie_work.config import OrchestratorConfig

    fleet = tmp_path / "fleet"
    fleet.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    registry = {"repos": {"octo/hello": {"repo_root": str(repo_root)}}}
    layout.fleet_registry_path(override=str(fleet)).write_text(
        json.dumps(registry), encoding="utf-8"
    )
    gh = _isolated_gh(tmp_path)
    app = MagicMock()
    app._run_review_reap_sweeps.side_effect = lambda now: sweep(gh)
    monkeypatch.setattr(fleet_lanes, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    monkeypatch.setattr(fleet_lanes, "runtime_paths", lambda *a, **k: MagicMock())
    monkeypatch.setattr(fleet_lanes, "github_client_for", lambda *a, **k: gh)
    monkeypatch.setattr(fleet_lanes, "OrchestratorApp", lambda *a, **k: app)
    fleet_lanes._run_fleet_reap_sweep(fleet_dir_override=str(fleet))
    return layout.state_file_path(fleet)


def test_reap_sweep_emits_one_budget_pass_per_repo(tmp_path: Path, monkeypatch) -> None:
    def sweep(gh) -> dict:
        _two_requests(gh)
        return {}

    state = _reap_once(tmp_path, monkeypatch, sweep)
    try:
        (event,) = query_events(state, kind="github_budget_pass")
    finally:
        close_db(state)
    payload = event["payload"]
    assert (payload["pass_kind"], payload["repo_key"]) == ("reap", "octo/hello")
    assert (payload["requests"], payload["points"]) == (2, 2)


def test_reap_sweep_emits_budget_pass_even_when_the_sweep_raises(
    tmp_path: Path, monkeypatch
) -> None:
    """Consistent with the lane pass: spend before a failure still counts."""

    def sweep(gh) -> dict:
        _two_requests(gh)
        raise RuntimeError("sweep blew up")

    state = _reap_once(tmp_path, monkeypatch, sweep)
    try:
        (event,) = query_events(state, kind="github_budget_pass")
    finally:
        close_db(state)
    assert event["payload"]["requests"] == 2


# -- exhausted-window dedupe is process-wide ------------------------------------


def test_two_clients_observing_the_same_exhausted_window_emit_once(tmp_path: Path) -> None:
    """The fleet builds a fresh client per lane pass: dedupe is per process."""
    state = tmp_path / "state.json"
    cursor = UsedCursor()

    def limited(*, reset: int, resource: str = "core") -> Response:
        return ok({}, headers=_h(5000, remaining=0, reset=reset, resource=resource), status=403)

    def fire(response: Response, request) -> None:
        _guard(response, state_path=state, cursor=cursor, retries=0).send(request)

    try:
        fire(limited(reset=9500), GET_PR)
        fire(limited(reset=9500), GET_PR)
        assert len(query_events(state, kind="github_rate_limited")) == 1
        fire(limited(reset=9500, resource="graphql"), QUERY)  # other resource
        assert len(query_events(state, kind="github_rate_limited")) == 2
        fire(limited(reset=13100), GET_PR)  # new window re-arms
        assert len(query_events(state, kind="github_rate_limited")) == 3
    finally:
        close_db(state)


def test_rate_limit_dedupe_is_thread_safe() -> None:
    cursor = UsedCursor()
    barrier = threading.Barrier(8)
    wins: list[bool] = []

    def run() -> None:
        barrier.wait()
        wins.append(cursor.first_exhaustion("core", 9500, 1000.0))

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert wins.count(True) == 1


def test_a_missing_reset_header_dedupes_per_hour_not_forever() -> None:
    cursor = UsedCursor()
    assert cursor.first_exhaustion("core", None, 1000.0)
    assert not cursor.first_exhaustion("core", None, 1500.0)  # same hour
    assert cursor.first_exhaustion("core", None, 1000.0 + 3600)  # next hour
    assert cursor.first_exhaustion("core", 0, 1000.0)  # never collides with a real epoch


# -- other rate-limit shapes ----------------------------------------------------


def test_graphql_rate_limited_error_emits_event(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    limited = Response(
        200,
        (),
        '{"data": null}',
        "http",
        graphql_errors=(GraphQLError("API rate limit exceeded", "RATE_LIMITED"),),
    )
    guard = _guard(limited, state_path=state, cursor=UsedCursor(), retries=0)
    with capability_scope("pull_requests"):
        guard.send(QUERY)
    try:
        (event,) = query_events(state, kind="github_rate_limited")
    finally:
        close_db(state)
    assert event["payload"]["capability"] == "pull_requests"
    assert event["payload"]["status"] == 200


def test_cli_requests_are_neither_spend_nor_rate_limit(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    gh = FakeAdapter(
        "gh",
        [ok({}, headers=_h(5000, remaining=0, reset=9500), status=403)],
        token="tok-1",
    )
    guard = _guard(state_path=state, cursor=UsedCursor(), retries=0, gh=gh)
    guard.send(CliRequest(CliCommand.AUTH_STATUS))
    try:
        assert query_events(state, kind="github_rate_limited") == []
    finally:
        close_db(state)
    assert guard.take_spend().entries == ()


def test_legacy_gh_run_names_the_capability_from_the_argv(tmp_path: Path) -> None:
    gh = _isolated_gh(tmp_path)
    gh.run(["pr", "view", "7", "--json", "number"], allow_failure=True)
    gh.run(["api", "repos/{owner}/{repo}/issues"], allow_failure=True)
    caps = {e.capability for e in gh._transport_v2.take_spend().entries}
    assert caps == {"gh.pr_view", "rest.issues"}
