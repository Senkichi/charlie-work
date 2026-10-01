# ruff: noqa: F811  (the imported ``ro_db`` fixture is re-bound as a test parameter)
"""Dashboard History metrics (``dashboard/metrics*.py``): exact values from a real-writer fleet."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _dashboard_metrics_fixtures import (  # noqa: F401  (ro_db is a pytest fixture)
    CURRENT,
    NOW,
    PRIOR,
    day,
    dispatch,
    merged,
    ro_db,
    values,
)
from _dashboard_rollup_fixtures import ALPHA, BETA, Fleet

from charlie_work.dashboard import metrics, rollup
from charlie_work.dashboard import metrics_capacity as cap
from charlie_work.dashboard import metrics_flow as flow
from charlie_work.dashboard import metrics_quality as quality
from charlie_work.dashboard import metrics_reliability as rel
from charlie_work.dashboard.metrics_base import MetricQuery, open_dashboard_ro
from charlie_work.dashboard.takeaways import takeaway

ZEROS = [0.0] * 7


def pts(*pairs: tuple[int, float]) -> tuple[tuple[str, float], ...]:
    return tuple((day(d), v) for d, v in pairs)


def by_name(series) -> dict[str, float]:
    return {s.name: sum(values(s)) for s in series}


def test_query_validation_and_prior() -> None:
    assert PRIOR.start == datetime(2026, 9, 24, tzinfo=UTC) and PRIOR.end == CURRENT.start
    assert CURRENT.n_buckets == 7
    with pytest.raises(ValueError):
        MetricQuery(NOW, NOW, timedelta(days=1))
    with pytest.raises(ValueError):
        MetricQuery(datetime(2026, 1, 1), NOW, timedelta(days=1))  # naive


def test_open_missing_dashboard_db_is_a_value(tmp_path: Path) -> None:
    conn, err = open_dashboard_ro(tmp_path / "nope.db")
    assert conn is None and err is not None and err.startswith("missing:")


def test_merges_per_day_and_headline(ro_db) -> None:
    cur, prior = flow.merges_per_day(ro_db, CURRENT), flow.merges_per_day(ro_db, PRIOR)
    assert values(cur) == [0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 0.0]  # duplicate report of 201 once
    assert values(prior) == [0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0]
    assert cur.per_repo[ALPHA][1][1] == 1.0 and cur.per_repo[BETA][1][1] == 1.0
    assert (cur.n, prior.n, cur.approx, cur.not_instrumented) == (5, 8, False, False)
    assert takeaway(cur, prior) == f"Merges/day ↓38% vs prior 7d, driven by {ALPHA}"


def test_lead_time_approx_from_milestones(ro_db) -> None:
    s = flow.lead_time(ro_db, CURRENT)
    assert s.points == pts((2, 7.5), (3, 36.0), (5, 12.5))
    assert s.per_repo[ALPHA] == pts((2, 10.0), (3, 36.0))
    assert s.per_repo[BETA] == pts((2, 5.0), (5, 12.5))
    assert (s.approx, s.exact_from, s.unit, s.n) == (True, None, "hours", 5)
    # no dispatched->pr_opened pair exists in the fixture, so stage time is empty, not zero
    assert flow.stage_time(ro_db, CURRENT, "in_progress").points == ()
    with pytest.raises(ValueError):
        flow.stage_time(ro_db, CURRENT, "nope")


def test_wip_queue_depth_and_capped_demand(ro_db) -> None:
    wip = flow.work_in_progress(ro_db, CURRENT)
    assert wip.points == pts((2, 11 / 3), (5, 3.0), (7, 3.0))
    assert wip.per_repo[ALPHA] == pts((2, 1.5), (7, 1.0))
    assert wip.per_repo[BETA] == pts((2, 1.0), (5, 1.0), (7, 1.0))
    depth = flow.queue_depth(ro_db, CURRENT)
    assert depth.points == pts((2, 11.0), (5, 5.0), (7, 0.0))  # sum of per-repo means
    capped = cap.capped_demand(ro_db, CURRENT)
    assert values(capped) == [0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0]  # 2 passes + 1 backpressure
    assert not wip.not_instrumented and not wip.approx


def test_quality_metrics(ro_db) -> None:
    mix = {s.name: values(s) for s in quality.verdict_mix(ro_db, CURRENT)}
    assert mix["verdict_mix.approved"] == [0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert mix["verdict_mix.request_changes"] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert mix["verdict_mix.blocked"] == ZEROS
    assert mix["verdict_mix"] == [0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]

    rework = quality.rework_rate(ro_db, CURRENT)
    assert rework.points == pts((2, 0.0), (5, 1 / 3))
    assert rework.n == 6

    esc = {s.name: values(s) for s in quality.escalations(ro_db, CURRENT)}
    assert esc["escalations"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # unescalate excluded
    assert esc["escalations.capped"] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert esc["escalations.review_verdict_escalated"] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    assert by_name(quality.worker_fate(ro_db, CURRENT)) == {
        "worker_fate": 8.0,
        "worker_fate.handoff_pr": 3.0,
        "worker_fate.orphan_pr": 1.0,
        "worker_fate.rate_limited": 1.0,
        "worker_fate.stalled": 2.0,
        "worker_fate.unclassified": 1.0,
    }
    share = quality.salvage_share(ro_db, CURRENT)
    assert share.points == pts((3, 0.25)) and share.per_repo[ALPHA] == pts((3, 0.25))
    assert by_name(quality.verdicts_missed(ro_db, CURRENT)) == {
        "verdicts_missed": 3.0,
        "verdicts_missed.launch_failed": 1.0,
        "verdicts_missed.pr #": 2.0,
    }


def test_capacity_metrics(ro_db) -> None:
    live, limit = cap.reviewers_live(ro_db, CURRENT), cap.reviewers_cap(ro_db, CURRENT)
    assert live.points == pts((2, 2.0)) and limit.points == pts((2, 6.0))
    running, capacity = cap.runners_running(ro_db, CURRENT), cap.runners_capacity(ro_db, CURRENT)
    assert running.points == pts((3, 1.5)) and capacity.points == pts((3, 4.5))
    assert running.per_repo == {"o/x": pts((3, 0.5)), "o/y": pts((3, 2.0))}
    assert capacity.per_repo == {"o/x": pts((3, 2.0)), "o/y": pts((3, 5.0))}
    wait = cap.ci_queue_wait(ro_db, CURRENT)
    assert (wait.points, wait.per_repo, wait.n) == (pts((3, 20.0)), {}, 3)


def test_workers_cap_uses_fleet_limit_and_per_repo_limit(ro_db) -> None:
    s = cap.workers_cap(ro_db, CURRENT)
    assert s.points == pts((2, 4.0), (5, 4.0), (7, 4.0))  # GOV fleet_concurrency_limit
    assert s.per_repo[ALPHA] == pts((2, 5.0), (7, 5.0))  # GOV concurrency_limit


def test_reliability_metrics(ro_db) -> None:
    assert rel.loop_pass_duration(ro_db, CURRENT).points == pts((2, 150.0))
    assert values(rel.loop_pass_errors(ro_db, CURRENT)) == [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fail = rel.launch_failures(ro_db, CURRENT)
    assert values(fail) == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0] and fail.partial
    deploys = {s.name: values(s) for s in rel.self_deploys(ro_db, CURRENT)}
    assert deploys["self_deploys"] == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert deploys["self_deploys.succeeded"] == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert deploys["self_deploys.failed"] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    throttles = {s.name: values(s) for s in rel.throttles(ro_db, CURRENT)}
    assert throttles["throttles"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert throttles["throttles.review_quota_exhausted"][2] == 1.0
    assert throttles["throttles.worker_rate_limited"][3] == 1.0


def test_registry_covers_every_history_metric(ro_db) -> None:
    out = metrics.all_series(ro_db, CURRENT)
    assert {t: sorted(m) for t, m in out.items()} == {
        "Flow": sorted(
            ["merges_per_day", "lead_time", "wip", "queue_depth"]
            + [f"stage_time.{s}" for s in flow.STAGES]
        ),
        "Quality": sorted(
            ["verdict_mix", "rework_rate", "escalations", "worker_fate", "salvage_share"]
            + ["verdicts_missed"]
        ),
        "Capacity": sorted(
            ["workers_cap", "reviewers_live", "reviewers_cap", "runners_running"]
            + ["runners_capacity", "ci_queue_wait", "capped_demand"]
        ),
        "Reliability": sorted(
            ["loop_pass_duration", "loop_pass_errors", "launch_failures", "throttles"]
            + ["self_deploys"]
        ),
    }
    for per_tab in out.values():
        for series_tuple in per_tab.values():
            assert all(s.window_start == CURRENT.start_iso for s in series_tuple)


def test_bucket_size_changes_resolution(ro_db) -> None:
    week = MetricQuery(CURRENT.start, CURRENT.end, timedelta(days=7))
    s = flow.merges_per_day(ro_db, week)
    assert s.points == ((CURRENT.start_iso, 5.0),)


def test_not_instrumented_when_no_source_ever_emitted(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    f = Fleet(tmp_path, monkeypatch)
    f.emit(f.alpha, day(2, 6), "session_exited", {"failure_kind": "stalled", "issue_number": 5})
    rollup.run_rollup(f.sources(), NOW)
    db, err = open_dashboard_ro(f.sources().db_path)
    assert db is not None, err
    s = flow.work_in_progress(db, CURRENT)
    assert (s.not_instrumented, s.points) == (True, ())
    assert takeaway(s, flow.work_in_progress(db, PRIOR)) == (
        "Work in progress: not instrumented yet"
    )
    db.close()
    f.close()


def test_lifecycle_exact_path_takes_over_at_cutover(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    f = Fleet(tmp_path, monkeypatch)
    a = f.alpha
    # approx era: issue 800 dispatched 4h before its PR, merged 10h after dispatch
    dispatch(f, a, day(-3, 0), [800])
    f.emit(a, day(-3, 4), "worker_handoff_pr_opened", {"issue_number": 800, "pr_number": 801})
    merged(f, a, day(-3, 10), 800)
    # exact era: issue 900 walks the lifecycle, including one rework loop
    f.emit(a, day(2, 0), "lifecycle_transition", {"issue_number": 900, "to_state": "Ready"})
    for hh, state in (
        (2, "In progress"), (6, "PR open"), (8, "Reviewing"), (9, "Needs rework"),
        (10, "In progress"), (12, "PR open"), (20, "Done"),
    ):  # fmt: skip
        f.emit(a, day(2, hh), "lifecycle_transition", {"issue_number": 900, "to_state": state})
    dispatch(f, a, day(7, 23))
    rollup.run_rollup(f.sources(), NOW)
    db, err = open_dashboard_ro(f.sources().db_path)
    assert db is not None, err
    q = MetricQuery(datetime(2026, 9, 25, tzinfo=UTC), NOW, timedelta(days=1))

    def at(d: int) -> str:
        return day(d)

    cut = day(2, 0)
    lead = flow.lead_time(db, q)
    assert lead.points == ((at(-3), 10.0), (at(2), 20.0))
    assert (lead.approx, lead.exact_from) == (True, cut)  # window still holds the approx era
    assert flow.stage_time(db, q, "in_progress").points == ((at(-3), 4.0), (at(2), 6.0))
    assert flow.stage_time(db, q, "pr_open").points == ((at(2), 10.0),)  # 2h + 8h, two visits
    assert flow.stage_time(db, q, "needs_rework").points == ((at(2), 1.0),)
    assert flow.stage_time(db, q, "reviewing").points == ((at(2), 1.0),)
    assert [p for p in flow.merges_per_day(db, q).points if p[1]] == [(at(-3), 1.0), (at(2), 1.0)]

    after = MetricQuery(datetime(2026, 10, 2, tzinfo=UTC), NOW, timedelta(days=1))
    exact_only = flow.lead_time(db, after)
    assert exact_only.points == ((at(2), 20.0),) and exact_only.approx is False
    db.close()
    f.close()
