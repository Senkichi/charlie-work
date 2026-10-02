"""Per-kind coverage, blackout buckets, backfill and category bounds (review-metrics H1/H3/M1/M2).

Every expectation is a hand-written literal worked out from the events each test emits; the
fleets are built through the real writers (``Fleet.emit`` -> ``log_event``) and rolled up.
"""

from __future__ import annotations

import pytest
from _dashboard_metrics_fixtures import CURRENT, NOW, PRIOR, day, merged
from _dashboard_rollup_fixtures import ALPHA, BETA, Fleet

from charlie_work.dashboard import metrics_flow as flow
from charlie_work.dashboard import metrics_quality as quality
from charlie_work.dashboard import rollup
from charlie_work.dashboard.metrics_base import MetricQuery, open_dashboard_ro
from charlie_work.dashboard.takeaways import takeaway


def pts(*pairs: tuple[int, float]) -> tuple[tuple[str, float], ...]:
    return tuple((day(d), v) for d, v in pairs)


@pytest.fixture
def fleet_env(tmp_path, monkeypatch):
    """``(fleet, open_db)``: emit events through ``fleet``, then ``open_db()`` rolls up + opens."""
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    f = Fleet(tmp_path, monkeypatch)
    opened = []

    def open_db():
        errors = rollup.run_rollup(f.sources(), NOW).errors
        assert all(": missing: " in e for e in errors), errors  # sources that never wrote
        conn, err = open_dashboard_ro(f.sources().db_path)
        assert conn is not None, err
        opened.append(conn)
        return conn

    yield f, open_db
    for conn in opened:
        conn.close()
    f.close()


def beat(f: Fleet, state, days) -> None:
    for d in days:  # a heartbeat per day: the source is alive, so an empty bucket is a real 0
        f.emit(state, day(d, 3), "supervisor_started", {})


def test_a_repo_that_joined_late_has_no_leading_zeros(fleet_env) -> None:
    f, open_db = fleet_env
    beat(f, f.alpha, range(1, 8))
    merged(f, f.alpha, day(1, 5), 1)
    beat(f, f.beta, range(4, 8))  # beta joins on 10-04
    merged(f, f.beta, day(4, 5), 2)
    s = flow.merges_per_day(open_db(), CURRENT)
    assert s.per_repo[BETA] == pts((4, 1.0), (5, 0.0), (6, 0.0), (7, 0.0))
    assert s.per_repo[ALPHA] == pts(
        (1, 1.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0)
    )
    assert s.points == pts((1, 1.0), (2, 0.0), (3, 0.0), (4, 1.0), (5, 0.0), (6, 0.0), (7, 0.0))


def test_blackout_buckets_are_absent_not_zero(fleet_env) -> None:
    f, open_db = fleet_env
    merged(f, f.alpha, day(2, 3), 1)  # first evidence
    beat(f, f.alpha, (4, 6))  # alive on 10-04 and 10-06 only: 10-03/05 are a blackout
    s = flow.merges_per_day(open_db(), CURRENT)
    assert s.points == pts((2, 1.0), (4, 0.0), (6, 0.0))


def test_reconcile_backfill_burst_is_not_counted_as_merges(fleet_env) -> None:
    f, open_db = fleet_env
    merged(f, f.alpha, day(2, 1), 1)
    merged(f, f.alpha, day(2, 2), 2)
    for i in range(20):  # one catch-up pass: 20 reconcile detections in a single hour
        merged(f, f.alpha, day(3, 5), 100 + i)
    for i in range(19):  # one fewer than the burst threshold: genuine flow
        merged(f, f.alpha, day(4, 5), 200 + i)
    s = flow.merges_per_day(open_db(), CURRENT)
    assert s.points == pts((2, 2.0), (3, 0.0), (4, 19.0))
    assert s.approx is True  # reconcile stamps the detection time, not the merge time


def test_salvage_share_is_claimed_only_from_when_both_kinds_exist(fleet_env) -> None:
    f, open_db = fleet_env
    for d in range(-5, 8):  # orphan PRs all along; handoff PRs only appear on 10-02
        f.emit(
            f.alpha,
            day(d, 4),
            "orphaned_worker_opened_pr",
            {"issue_number": d + 50, "pr_number": 9},
        )
    f.emit(f.alpha, day(2, 5), "worker_handoff_pr_opened", {"issue_number": 1, "pr_number": 2})
    db = open_db()
    prior, cur = quality.salvage_share(db, PRIOR), quality.salvage_share(db, CURRENT)
    assert prior.points == ()  # not the fake 100% an orphan-only history would give
    assert cur.points == pts((2, 0.5), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0))
    assert cur.coverage_start == day(2, 5)
    assert takeaway(cur, prior) == "not comparable: salvage_share starts 2026-10-02"


def test_capped_demand_is_not_comparable_across_a_new_kind(fleet_env) -> None:
    f, open_db = fleet_env
    beat(f, f.alpha, range(-6, 8))
    for d in (-5, -3, 2, 3, 4):
        f.emit(
            f.alpha,
            day(d, 2),
            "dispatch",
            {"concurrency_governor": {}, "backlog_reachability": {}},
        )
    f.emit(f.alpha, day(3, 6), "dispatch_backpressure", {"clamped_by": "host_load"})
    db = open_db()
    cur, prior = (flow_cap(db, q) for q in (CURRENT, PRIOR))
    assert cur.coverage_start == day(3, 6)
    assert takeaway(cur, prior) == "not comparable: capped_demand starts 2026-10-03"


def flow_cap(db, q):
    from charlie_work.dashboard import metrics_capacity as cap

    return cap.capped_demand(db, q)


def test_escalation_categories_are_bounded_to_top_n_plus_other(fleet_env) -> None:
    f, open_db = fleet_env
    n = 0
    for k in range(1, 11):  # reason r01 x10, r02 x9, ... r10 x1
        for _ in range(11 - k):
            n += 1
            f.emit(
                f.alpha,
                day(2, 1),
                "session_failed_escalated",
                {"issue_number": n, "reason": f"r{k:02d}"},
            )
    series = {s.name: sum(v for _, v in s.points) for s in quality.escalations(open_db(), CURRENT)}
    assert sorted(series) == sorted(
        ["escalations", "escalations.other"] + [f"escalations.r{k:02d}" for k in range(1, 9)]
    )
    assert series["escalations"] == 55.0
    assert (
        series["escalations.r08"] == 3.0 and series["escalations.other"] == 3.0
    )  # r09 (2) + r10 (1)


def test_zero_baseline_headline_states_the_absolute_change(fleet_env) -> None:
    f, open_db = fleet_env
    merged(f, f.alpha, day(-5, 3), 1)  # evidence before both windows
    beat(f, f.alpha, range(1, 8))
    for i in range(3):
        merged(f, f.alpha, day(5, 8 + i), 10 + i)
    q = MetricQuery(day_dt(4), day_dt(7), CURRENT.bucket)
    db = open_db()
    got = takeaway(flow.merges_per_day(db, q), flow.merges_per_day(db, q.prior()))
    assert got == "Merges/day up from 0 to 1 vs prior 3d (approx.)"


def day_dt(d: int):
    from charlie_work.dashboard.metrics_base import parse_ts

    return parse_ts(day(d))
