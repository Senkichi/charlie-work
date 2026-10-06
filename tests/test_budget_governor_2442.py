"""The fleet-wide GraphQL budget governor (issue #2442).

Covers the four parts: tiered lane deferral, the shared per-token budget file,
no retries at a primary rate limit, and the one event per window. Driven at the
``Adapter`` seam with scripted fakes and an injected clock.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import charlie_work.workflow  # noqa: F401  (import before orchestration.* submodules)
from charlie_work.api_budget import GitHubRateBudget, GitHubRateWindow
from charlie_work.command_result import CommandResult
from charlie_work.github_transport import (
    Adapters,
    GraphQLRequest,
    GuardedTransport,
    RateBudgetHolder,
    Response,
    RestRequest,
)
from charlie_work.github_transport.governor import (
    LANE_TIERS,
    BudgetGovernor,
    BudgetReserves,
    Tier,
    decide,
    tier_of,
)
from charlie_work.github_transport.guarded import UsedCursor
from charlie_work.github_transport.shared_budget import (
    SharedBudgetFile,
    merge_github_rate,
    token_fingerprint,
)
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.instrumentation import close_db, query_events
from charlie_work.budget_gates import budget_deferral, budget_gate_for
from charlie_work.orchestration.misc_review_verdicts import _run_review_reap_sweeps
from charlie_work.pass_deadline import PassDeadline

from _fake_transport import FakeAdapter, Runtime, Sleeps, ok

SRC = Path(__file__).resolve().parents[1] / "src" / "charlie_work"
NOW = 1000.0
RESET = 4600
GET_PR = RestRequest.of("GET", "repos/{owner}/{repo}/pulls/7")
QUERY = GraphQLRequest.of("query Q { viewer { login } }")
RESERVES = BudgetReserves(1500, 800, 300)


def _window(remaining: int, *, reset: int = RESET, resource: str = "graphql"):
    return GitHubRateWindow(resource, 5000, remaining, reset, NOW)


def _h(remaining: int, *, reset: int = RESET, resource: str = "graphql", used: int | None = None):
    return {
        "X-RateLimit-Limit": "5000",
        "X-RateLimit-Used": str(5000 - remaining if used is None else used),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(reset),
        "X-RateLimit-Resource": resource,
    }


@dataclass(frozen=True)
class GovRuntime(Runtime):
    github_budget_governor: bool = True
    state_dir: str = ".var/charlie-work"
    gh_primary_limit_max_wait_seconds: float = 30.0


def _guard(
    *script,
    runtime: Runtime | None = None,
    state_path: Path | None = None,
    shared: SharedBudgetFile | None = None,
    cursor: UsedCursor | None = None,
    now: float = NOW,
    sleeps: Sleeps | None = None,
    token: str = "tok-1",
):
    http = FakeAdapter("http", list(script))
    guard = GuardedTransport(
        Adapters(http=http, gh=FakeAdapter("gh", token=token)),
        runtime=runtime or GovRuntime(gh_max_retries=3),
        state_path=state_path,
        resolve_owner_repo=lambda: ("octo", "hello"),
        budget=RateBudgetHolder(cursor or UsedCursor(), shared=shared),
        sleep=sleeps if sleeps is not None else Sleeps(),
        jitter=lambda lo, hi: 0.0,
        now=lambda: now,
    )
    return guard, http


# -- tiers ---------------------------------------------------------------------


def _deferring(remaining: int | None) -> set[str]:
    window = None if remaining is None else _window(remaining)
    return {lane for lane in LANE_TIERS if decide(lane, RESERVES, window).defer}


RESERVE_LANES = {"reconcile", "drift", "review_reap", "ci_reclaim"}
SCAN_LANES = {"intake", "dispatch_rework", "dispatch", "dispatch_reviews", "quota_probe"}
FLOOR_LANES = {"deescalate"}
EXEMPT_LANES = {"local_lane"}
ESSENTIAL_LANES = {"merge", "label_write", "unauthorized_merge_tripwire"}


def test_the_mapping_covers_exactly_the_documented_lanes() -> None:
    assert set(LANE_TIERS) == (
        RESERVE_LANES | SCAN_LANES | FLOOR_LANES | ESSENTIAL_LANES | EXEMPT_LANES
    )
    assert {n for n, t in LANE_TIERS.items() if t is Tier.ESSENTIAL} == ESSENTIAL_LANES


def test_tier_1500_reconcile_drift_and_review_reaps_defer_first() -> None:
    assert _deferring(1500) == set()
    assert _deferring(1499) == RESERVE_LANES


def test_tier_800_dispatch_scans_and_probes_defer_next() -> None:
    assert _deferring(800) == RESERVE_LANES
    assert _deferring(799) == RESERVE_LANES | SCAN_LANES


def test_tier_300_everything_but_merge_label_writes_and_the_tripwire_defers() -> None:
    assert _deferring(300) == RESERVE_LANES | SCAN_LANES
    assert _deferring(299) == RESERVE_LANES | SCAN_LANES | FLOOR_LANES
    assert _deferring(0) == RESERVE_LANES | SCAN_LANES | FLOOR_LANES  # essential still go


def test_an_unmapped_lane_gives_way_last_of_the_optional_work() -> None:
    assert tier_of("some_new_gate") is Tier.FLOOR
    assert not decide("some_new_gate", RESERVES, _window(300)).defer
    assert decide("some_new_gate", RESERVES, _window(299)).defer


def test_an_unobserved_budget_fails_open() -> None:
    assert _deferring(None) == set()


def test_reserves_are_monotone_so_a_misordered_config_cannot_invert_the_tiers() -> None:
    rt = SimpleNamespace(
        graphql_rate_limit_threshold=500, graphql_scan_reserve=900, graphql_floor_reserve=700
    )
    assert BudgetReserves.from_runtime(rt) == BudgetReserves(500, 500, 500)
    off = SimpleNamespace(graphql_rate_limit_threshold=0)
    assert BudgetReserves.from_runtime(off) == BudgetReserves(0, 0, 0)  # 0 disables the guard


def test_a_disabled_governor_never_defers() -> None:
    governor = BudgetGovernor(lambda: _window(0), RESERVES, enabled=False)
    assert not governor.check("review_reap").defer


def _lane_names_in_source() -> set[str]:
    """Every lane name a source file hands the gate, derived from the AST."""
    names: set[str] = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "phase" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.add(first.value)
            for kw in node.keywords:
                if kw.arg == "lane" and isinstance(kw.value, ast.Constant):
                    names.add(str(kw.value.value))
            if (
                isinstance(fn, ast.Name)
                and fn.id == "budget_deferral"
                and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
            ):
                names.add(str(node.args[1].value))
    return names


def test_every_gated_lane_in_the_source_is_mapped_and_none_is_essential() -> None:
    gated = _lane_names_in_source()
    assert {"intake", "dispatch", "review_reap", "quota_probe", "ci_reclaim"} <= gated
    assert gated <= set(LANE_TIERS), gated - set(LANE_TIERS)
    assert not {n for n in gated if tier_of(n) is Tier.ESSENTIAL}


def test_the_loop_body_arms_the_gate_on_its_pass_deadline() -> None:
    source = (SRC / "orchestration" / "reap_loop.py").read_text(encoding="utf-8")
    assert "budget_gate=budget_gate_for(self)" in source


# -- the app-side seam -----------------------------------------------------------


class _Gate:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_event(self, kind: str, payload: dict, **_: object) -> None:
        self.events.append((kind, payload))


def _app(tmp_path: Path, remaining: int | None, *, governor: bool = True, reset: int = RESET):
    shared = None
    headers = _h(remaining, reset=reset) if remaining is not None else {}
    guard, _ = _guard(
        ok({}, headers=headers), runtime=GovRuntime(github_budget_governor=governor), now=NOW
    )
    guard.send(GET_PR)
    return SimpleNamespace(
        gh=SimpleNamespace(_transport_v2=guard),
        config=SimpleNamespace(runtime=SimpleNamespace()),
        paths=SimpleNamespace(state_file=tmp_path / f"state-{remaining}-{reset}.json"),
        write_gate=_Gate(),
        dry_run=False,
        shared=shared,
    )


def test_budget_deferral_records_one_event_per_lane_per_window(tmp_path: Path) -> None:
    app = _app(tmp_path, 700)
    for _ in range(3):
        decision = budget_deferral(app, "dispatch")
    assert decision is not None and decision.threshold == 800
    assert budget_deferral(app, "review_reap") is not None
    assert [(k, p["phase"]) for k, p in app.write_gate.events] == [
        ("graphql_rate_limit_deferred", "dispatch"),
        ("graphql_rate_limit_deferred", "review_reap"),
    ]
    assert app.write_gate.events[0][1] == {
        "remaining": 700,
        "reset": RESET,
        "threshold": 800,
        "phase": "dispatch",
        "tier": "scan",
    }


def test_budget_deferral_ignores_unguarded_clients_and_the_kill_switch(tmp_path: Path) -> None:
    app = _app(tmp_path, 10, governor=False)
    assert budget_deferral(app, "review_reap") is None
    fake = SimpleNamespace(gh=object(), config=app.config, paths=app.paths, write_gate=_Gate())
    assert budget_deferral(fake, "review_reap") is None


def test_pass_deadline_phases_defer_in_tier_order_and_merge_proceeds_below_300(
    tmp_path: Path,
) -> None:
    ran: list[str] = []

    def run(remaining: int) -> list[str]:
        ran.clear()
        app = _app(tmp_path, remaining)
        deadline = PassDeadline(None, CommandResult, budget_gate=budget_gate_for(app))
        for lane in (
            "intake",
            "dispatch",
            "dispatch_reviews",
            "local_lane",
            "merge",
            "label_write",
        ):
            result = deadline.phase(
                lane, lambda lane=lane: ran.append(lane) or CommandResult(True, "", {})
            )
            assert result.ok
        deadline.call(lambda: ran.append("ci_reclaim"), None, lane="ci_reclaim")
        deadline.call(
            lambda: ran.append("unauthorized_merge_tripwire"),
            None,
            lane="unauthorized_merge_tripwire",
        )
        return list(ran)

    everything = {
        "intake", "dispatch", "dispatch_reviews", "local_lane", "merge", "label_write",
        "ci_reclaim", "unauthorized_merge_tripwire",
    }  # fmt: skip
    assert set(run(5000)) == everything
    assert set(run(1400)) == everything - {"ci_reclaim"}
    assert set(run(790)) == everything - {"ci_reclaim", "intake", "dispatch", "dispatch_reviews"}
    survivors = {"merge", "label_write", "unauthorized_merge_tripwire", "local_lane"}
    assert set(run(250)) == survivors  # the local lane spends no budget
    assert set(run(0)) == survivors


def test_a_deferred_phase_reports_budget_deferred_not_deadline_deferred(tmp_path: Path) -> None:
    app = _app(tmp_path, 100)
    deadline = PassDeadline(None, CommandResult, budget_gate=budget_gate_for(app))
    result = deadline.phase("dispatch", lambda: pytest.fail("must not run"))
    assert result.ok and result.data == {"budget_deferred": True}
    assert not deadline.hit


def test_review_reaps_defer_below_1500_and_run_above(tmp_path: Path) -> None:
    low = _app(tmp_path, 1400)
    low.dry_run = False
    summary = _run_review_reap_sweeps(low, None)
    assert summary["stalled"] == [] and summary["verdict_result"] == {"recorded": [], "missed": []}
    assert [p["phase"] for _, p in low.write_gate.events] == ["review_reap"]
    ok_app = _app(tmp_path, 1500)
    ok_app._layout = None  # the sweep proceeds and touches it: proof it did not defer
    with pytest.raises(AttributeError):
        _run_review_reap_sweeps(ok_app, None)


def test_the_local_lane_spends_no_budget_and_proceeds_at_zero_remaining() -> None:
    assert tier_of("local_lane") is Tier.EXEMPT
    assert not decide("local_lane", RESERVES, _window(0)).defer


def test_a_local_repo_app_never_defers_any_lane_at_zero_remaining(tmp_path: Path) -> None:
    """Derived exemption: a client that publishes no PRs spends no GitHub budget."""
    app = _app(tmp_path, 0)
    assert budget_deferral(app, "dispatch") is not None  # a remote app at 0 does defer
    app.write_gate.events.clear()
    assert LocalFileGitHub.publishes_pull_requests is False
    app.gh = SimpleNamespace(_transport_v2=app.gh._transport_v2, publishes_pull_requests=False)
    for lane in (*LANE_TIERS, "some_new_gate"):
        assert budget_deferral(app, lane) is None
    assert app.write_gate.events == []


# -- the shared budget file ------------------------------------------------------


def test_two_client_instances_share_one_observed_budget_through_the_file(tmp_path: Path) -> None:
    path = tmp_path / "fleet" / "github-budget.json"
    first, _ = _guard(
        ok({}, headers=_h(900)), shared=SharedBudgetFile(path, now=lambda: NOW, publish_interval=0)
    )
    second, _ = _guard(shared=SharedBudgetFile(path, now=lambda: NOW, read_ttl=0))  # never sends
    assert second.rate_window("graphql") is None
    first.send(GET_PR)
    window = second.rate_window("graphql")
    assert window is not None and window.remaining == 900 and window.reset_epoch == RESET
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert list(on_disk["tokens"]) == [token_fingerprint("tok-1")]
    assert "tok-1" not in path.read_text(encoding="utf-8")


def test_a_fresh_client_reads_its_tokens_snapshot_before_sending(tmp_path: Path) -> None:
    path = tmp_path / "github-budget.json"
    writer, _ = _guard(
        ok({}, headers=_h(1200)),
        shared=SharedBudgetFile(path, now=lambda: NOW, publish_interval=0),
    )
    writer.send(GET_PR)
    reader, http = _guard(shared=SharedBudgetFile(path, now=lambda: NOW))
    assert reader.fresh_rate_window("graphql", 60.0).remaining == 1200  # type: ignore[union-attr]
    assert http.calls == []


def test_a_different_token_does_not_see_the_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "github-budget.json"
    one, _ = _guard(
        ok({}, headers=_h(50)), shared=SharedBudgetFile(path, now=lambda: NOW, publish_interval=0)
    )
    one.send(GET_PR)
    other, _ = _guard(shared=SharedBudgetFile(path, now=lambda: NOW), token="tok-2")
    assert other.rate_window("graphql") is None


def test_a_snapshot_from_a_past_window_is_ignored_and_pruned(tmp_path: Path) -> None:
    path = tmp_path / "github-budget.json"
    clock = {"t": NOW}
    shared = SharedBudgetFile(path, now=lambda: clock["t"], read_ttl=0, publish_interval=0)
    key = token_fingerprint("tok-1")
    shared.publish(key, GitHubRateBudget((_window(10),)))
    assert shared.read(key).windows[0].remaining == 10
    clock["t"] = RESET + 1  # the window has reset
    assert shared.read(key).windows == ()
    shared.publish(key, GitHubRateBudget((_window(4000, reset=RESET + 3600),)))
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert [w["reset"] for w in on_disk["tokens"][key]["windows"]] == [RESET + 3600]


def test_publishing_never_moves_a_window_backwards(tmp_path: Path) -> None:
    shared = SharedBudgetFile(tmp_path / "b.json", now=lambda: NOW, read_ttl=0, publish_interval=0)
    shared.publish("k", GitHubRateBudget((_window(500),)))
    shared.publish("k", GitHubRateBudget((_window(900),)))  # an older, out-of-order observation
    assert shared.read("k").windows[0].remaining == 500
    shared.publish("k", GitHubRateBudget((_window(4900, reset=RESET + 3600),)))
    assert shared.read("k").windows[0].reset_epoch == RESET + 3600


def test_a_corrupt_or_foreign_file_degrades_to_no_shared_knowledge(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    shared = SharedBudgetFile(path, now=lambda: NOW, read_ttl=0, publish_interval=0)
    for text in ("{not json", '{"version": 99, "tokens": {}}', "[]", '{"version":1,"tokens":5}'):
        path.write_text(text, encoding="utf-8")
        assert shared.read("k").windows == ()
    assert shared.publish("k", GitHubRateBudget((_window(7),)))  # and it heals
    assert shared.read("k").windows[0].remaining == 7


def test_publish_is_throttled_but_a_new_window_or_exhaustion_always_goes_out(
    tmp_path: Path,
) -> None:
    clock = {"t": NOW}
    shared = SharedBudgetFile(
        tmp_path / "b.json", now=lambda: clock["t"], read_ttl=0, publish_interval=5.0
    )
    assert shared.publish("k", GitHubRateBudget((_window(900),)))
    assert not shared.publish("k", GitHubRateBudget((_window(899),)))  # same window, too soon
    assert shared.publish("k", GitHubRateBudget((_window(0),)))  # exhausted: always
    assert shared.publish(
        "k", GitHubRateBudget((_window(4000, reset=RESET + 3600),))
    )  # new window
    clock["t"] += 6
    assert shared.publish("k", GitHubRateBudget((_window(3990, reset=RESET + 3600),)))


def test_merge_keeps_the_newest_window_per_resource() -> None:
    a = GitHubRateBudget((_window(100), _window(10, resource="core")))
    b = GitHubRateBudget((_window(50), _window(20, resource="core", reset=RESET - 1)))
    merged = {w.resource: w.remaining for w in merge_github_rate(a, b).windows}
    assert merged == {"graphql": 50, "core": 10}


def test_the_kill_switch_runs_without_a_shared_file(tmp_path: Path) -> None:
    from charlie_work.github_capabilities.transport_wiring import build_guarded_transport

    gh = SimpleNamespace(
        runtime=GovRuntime(github_budget_governor=False),
        adapters=Adapters(http=FakeAdapter("http"), gh=FakeAdapter("gh")),
        repo_root=tmp_path,
        dry_run=False,
        _pass_deadline_exceeded=None,
        _circuit_breaker_state=None,
    )
    assert build_guarded_transport(gh).budget.shared_enabled is False  # type: ignore[arg-type]
    gh.runtime = GovRuntime(github_budget_governor=True)
    assert build_guarded_transport(gh).budget.shared_enabled is True  # type: ignore[arg-type]


# -- primary rate limit: no retries ---------------------------------------------------


def _limited(*, reset: int = RESET, resource: str = "graphql") -> Response:
    return ok(
        {"message": "API rate limit exceeded"},
        headers=_h(0, reset=reset, resource=resource),
        status=403,
    )


def test_at_zero_remaining_no_retries_are_sent(tmp_path: Path) -> None:
    sleeps = Sleeps()
    guard, http = _guard(_limited(), sleeps=sleeps)  # reset is 3600s away: defer
    out = guard.send(QUERY)
    assert isinstance(out, Response) and out.status == 403
    assert len(http.api_requests) == 1
    assert sleeps.delays == []


def test_a_nearby_reset_waits_for_it_then_retries_once(tmp_path: Path) -> None:
    sleeps = Sleeps()
    guard, http = _guard(
        _limited(reset=int(NOW) + 10), _limited(reset=int(NOW) + 10), ok({}), sleeps=sleeps
    )
    out = guard.send(QUERY)
    # One wait until the reset (+1s margin); a second limited reply is returned, not re-waited.
    assert sleeps.delays == [11.0]
    assert len(http.api_requests) == 2
    assert isinstance(out, Response) and out.status == 403


def test_a_nearby_reset_that_clears_succeeds_after_one_wait() -> None:
    sleeps = Sleeps()
    guard, http = _guard(_limited(reset=int(NOW) + 5), ok({}), sleeps=sleeps)
    out = guard.send(QUERY)
    assert isinstance(out, Response) and out.ok
    assert sleeps.delays == [6.0] and len(http.api_requests) == 2


def test_the_wait_bound_is_a_config_knob() -> None:
    rt = GovRuntime(gh_max_retries=3, gh_primary_limit_max_wait_seconds=5.0)
    sleeps = Sleeps()
    guard, http = _guard(_limited(reset=int(NOW) + 10), runtime=rt, sleeps=sleeps)
    guard.send(QUERY)
    assert len(http.api_requests) == 1 and sleeps.delays == []


def test_the_kill_switch_restores_the_plain_retries() -> None:
    sleeps = Sleeps()
    rt = GovRuntime(gh_max_retries=3, github_budget_governor=False)
    guard, http = _guard(_limited(), runtime=rt, sleeps=sleeps)
    guard.send(QUERY)
    assert len(http.api_requests) == 4
    assert sleeps.delays == [1.0, 2.0, 4.0]


def test_a_secondary_rate_limit_keeps_the_backoff() -> None:
    sleeps = Sleeps()
    secondary = ok({"message": "secondary"}, headers={"Retry-After": "1"}, status=429)
    guard, http = _guard(secondary, secondary, ok({}), sleeps=sleeps)
    out = guard.send(QUERY)
    assert isinstance(out, Response) and out.ok
    assert sleeps.delays == [1.0, 2.0] and len(http.api_requests) == 3


def test_a_limit_without_a_reset_time_keeps_the_backoff() -> None:
    sleeps = Sleeps()
    bare = ok({}, headers={"X-RateLimit-Remaining": "0"}, status=403)
    guard, http = _guard(bare, ok({}), sleeps=sleeps)
    assert guard.send(QUERY).ok  # type: ignore[union-attr]
    assert sleeps.delays == [1.0]


# -- one event per window ----------------------------------------------------------------


def test_exactly_one_rate_limited_event_per_window_across_clients(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    cursor = UsedCursor()
    try:
        for _ in range(3):  # three fresh clients, one exhausted window
            guard, _ = _guard(_limited(), state_path=state, cursor=cursor)
            guard.send(QUERY)
        assert len(query_events(state, kind="github_rate_limited")) == 1
        guard, _ = _guard(_limited(reset=RESET + 3600), state_path=state, cursor=cursor)
        guard.send(QUERY)  # the next window re-arms
        assert len(query_events(state, kind="github_rate_limited")) == 2
    finally:
        close_db(state)


# -- the work-only lane path --------------------------------------------------------------


def test_the_work_only_lane_defers_dispatch_below_800(tmp_path: Path) -> None:
    from charlie_work.fleet_lanes import _run_fleet_repo_lane

    app = _app(tmp_path, 600)
    app.repo_root = tmp_path
    app.dispatch = lambda limit: pytest.fail("dispatch must be deferred")
    released: list[bool] = []
    lock = SimpleNamespace(release=lambda: released.append(True))
    result = _run_fleet_repo_lane(
        "r",
        app,  # type: ignore[arg-type]
        SimpleNamespace(review_dispatch=SimpleNamespace(enabled=True)),  # type: ignore[arg-type]
        lock,
        work_only=True,
        drain=False,
        limit=None,
        merge=None,
        ensure_labels=False,
        fleet_state_path=tmp_path / "fleet-state.json",
    )
    try:
        assert result.ok and result.data == {"budget_deferred": True}
        assert released == [True]
        assert [p["phase"] for _, p in app.write_gate.events] == ["dispatch"]
    finally:
        close_db(tmp_path / "fleet-state.json")
