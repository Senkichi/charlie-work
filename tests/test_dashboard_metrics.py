# ruff: noqa: F811  (the imported ``ro_db`` fixture is re-bound as a test parameter)
"""Dashboard History metrics (``dashboard/metrics*.py``): exact values from a real-writer fleet."""

from __future__ import annotations

import sqlite3
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
q_day = timedelta(days=1)


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


def test_open_dashboard_ro_is_mode_ro_without_immutable(tmp_path, monkeypatch) -> None:
    """The rollup writes dashboard.db in-process: a reader must take part in WAL locking
    (``immutable=1`` takes no locks and can observe a checkpoint mid-write), so the open
    mode is pinned to plain ``mode=ro``."""
    path = tmp_path / "dashboard.db"
    seed = sqlite3.connect(path)
    seed.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    seed.commit()
    seed.close()
    uris: list[str] = []
    real_connect = sqlite3.connect

    def spy(*args, **kwargs):
        uris.append(str(args[0]))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy)
    conn, err = open_dashboard_ro(path)
    assert err is None and conn is not None
    conn.close()
    assert uris and "mode=ro" in uris[0] and "immutable" not in uris[0]


def test_merges_per_day_and_headline(ro_db) -> None:
    cur, prior = flow.merges_per_day(ro_db, CURRENT), flow.merges_per_day(ro_db, PRIOR)
    assert values(cur) == [0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 0.0]  # duplicate report of 201 once
    # the first merge evidence (a reconcile at 09-25T12) starts coverage: 09-24 is not claimed
    assert values(prior) == [1.0, 1.0, 1.0, 2.0, 1.0, 2.0]
    assert cur.per_repo[ALPHA][1][1] == 1.0 and cur.per_repo[BETA][1][1] == 1.0
    assert (cur.n, prior.n, cur.approx, cur.not_instrumented) == (
        5,
        8,
        True,
        False,
    )  # reconcile-detected: stamped at detection
    # the first merge evidence is 09-25T12, after the prior window began: no comparison
    assert takeaway(cur, prior) == "not comparable: merges_per_day starts 2026-09-25"
    # a shorter pair that both sit inside coverage compares: 2 merges (10-05) vs 3 (10-02/03)
    short = MetricQuery(
        datetime(2026, 10, 4, tzinfo=UTC), datetime(2026, 10, 7, tzinfo=UTC), q_day
    )
    got = takeaway(
        flow.merges_per_day(ro_db, short), flow.merges_per_day(ro_db, short.prior()), min_sample=1
    )
    assert got == f"Merges/day ↓33% vs prior 3d, driven by {ALPHA} (approx.)"


def test_lead_time_approx_from_milestones(ro_db) -> None:
    s = flow.lead_time(ro_db, CURRENT)
    assert s.points == pts((2, 7.5), (3, 36.0), (5, 12.5))
    assert s.per_repo[ALPHA] == pts((2, 10.0), (3, 36.0))
    assert s.per_repo[BETA] == pts((2, 5.0), (5, 12.5))
    assert (s.approx, s.exact_from, s.unit, s.n) == (True, None, "hours", 5)
    # no dispatched->pr_opened pair exists in the fixture, so each in_progress visit is
    # a dispatched->merged span -- the same interval lead time reports (issue #2473: any
    # other milestone, including the merge itself, ends an open visit)
    assert flow.stage_time(ro_db, CURRENT, "in_progress").points == s.points
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
    # coverage starts with the first backpressure (10-02); only days a dispatch pass ran are alive
    assert capped.points == pts((2, 3.0), (5, 2.0), (7, 0.0))  # 2 passes + 1 backpressure
    assert not wip.not_instrumented and not wip.approx


def test_quality_metrics(ro_db) -> None:
    mix = {s.name: values(s) for s in quality.verdict_mix(ro_db, CURRENT)}
    # record_review first appears 10-02: nothing is claimed for 10-01
    assert mix["verdict_mix.approved"] == [2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert mix["verdict_mix.request_changes"] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert mix["verdict_mix.blocked"] == [0.0] * 6
    assert mix["verdict_mix"] == [2.0, 2.0, 0.0, 0.0, 0.0, 0.0]

    rework = quality.rework_rate(ro_db, CURRENT)
    # alpha's coverage starts at its first dispatch_rework (10-05): its 10-02 dispatches drop out
    # for alpha only; beta has no rework event, so its own coverage starts with its dispatches
    assert rework.points == pts((2, 0.0), (5, 1 / 3))
    assert rework.n == 4

    esc = {s.name: values(s) for s in quality.escalations(ro_db, CURRENT)}
    # alternative routes: coverage starts with the first of them (record_review, 10-02)
    assert esc["escalations"] == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # unescalate excluded
    assert esc["escalations.capped"] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert esc["escalations.review_verdict_escalated"] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

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
    # a day with no loop pass is a blackout, not a quiet day: only 10-02 has a point
    assert rel.loop_pass_errors(ro_db, CURRENT).points == pts((2, 3.0))
    fail = rel.launch_failures(ro_db, CURRENT)
    assert values(fail) == [1.0, 0.0, 0.0, 0.0, 0.0] and fail.partial  # starts 10-03
    deploys = {s.name: values(s) for s in rel.self_deploys(ro_db, CURRENT)}
    assert deploys["self_deploys"] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # from the first, 10-02
    assert deploys["self_deploys.succeeded"] == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert deploys["self_deploys.failed"] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    throttles = {s.name: values(s) for s in rel.throttles(ro_db, CURRENT)}
    assert throttles["throttles"] == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # from 10-02 (session_exited)
    assert throttles["throttles.review_quota_exhausted"][1] == 1.0
    assert throttles["throttles.worker_rate_limited"][2] == 1.0


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


def test_stage_time_approx_excludes_parked_and_dead_time(tmp_path, monkeypatch) -> None:
    """Issue #2473: an approx ``in_progress`` visit ends at any exit, not only at PR open.

    The incident behind #2473 -- dispatched, escalated, parked ~10 days,
    unescalated, redispatched, PR opened 17 min later -- was recorded as one
    1534h visit because ``escalated`` never closed the open interval and a
    second ``dispatched`` did not restart it.
    """
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    f = Fleet(tmp_path, monkeypatch)
    a = f.alpha
    # The incident shape: a 1h first run, a long park, then a 17-minute winning run.
    dispatch(f, a, "2026-09-25T00:00:00Z", [1427])
    f.emit(
        a, "2026-09-25T01:00:00Z", "session_failed_escalated",
        {"issue_number": 1427, "reason": "worker_dead"},
    )  # fmt: skip
    f.emit(
        a, "2026-10-05T00:00:00Z", "unescalate",
        {"issue_number": 1427, "cleared_escalation_reason": "operator_reviewed"},
    )  # fmt: skip
    dispatch(f, a, "2026-10-05T12:00:00Z", [1427])
    f.emit(
        a, "2026-10-05T12:17:00Z", "worker_handoff_pr_opened",
        {"issue_number": 1427, "pr_number": 6427},
    )  # fmt: skip
    # A dead worker's relabel exits in_progress too: 2h dead run + 30min winning run.
    dispatch(f, a, "2026-10-06T00:00:00Z", [2700])
    f.emit(
        a, "2026-10-06T02:00:00Z", "session_failed_relabeled",
        {"issue_number": 2700, "reason": "dead_worker_no_open_pr_orphan_sweep"},
    )  # fmt: skip
    dispatch(f, a, "2026-10-06T10:00:00Z", [2700])
    f.emit(
        a, "2026-10-06T10:30:00Z", "worker_handoff_pr_opened",
        {"issue_number": 2700, "pr_number": 7700},
    )  # fmt: skip
    # A re-dispatch while the visit is still open restarts it: the last dispatch wins.
    dispatch(f, a, "2026-10-04T00:00:00Z", [2702])
    dispatch(f, a, "2026-10-04T06:00:00Z", [2702])
    f.emit(
        a, "2026-10-04T06:20:00Z", "worker_handoff_pr_opened",
        {"issue_number": 2702, "pr_number": 7702},
    )  # fmt: skip
    # The same review for pr_open: an escalation mid-review-wait ends the visit.
    f.emit(
        a, "2026-10-03T00:00:00Z", "worker_handoff_pr_opened",
        {"issue_number": 2704, "pr_number": 7704},
    )  # fmt: skip
    f.emit(
        a, "2026-10-03T02:00:00Z", "session_failed_escalated",
        {"issue_number": 2704, "pr_number": 7704, "reason": "review_dispatch_escalated"},
    )  # fmt: skip
    f.emit(
        a, "2026-10-03T10:00:00Z", "unescalate",
        {"issue_number": 2704, "cleared_escalation_reason": "operator_reviewed"},
    )  # fmt: skip
    f.emit(a, "2026-10-03T12:00:00Z", "review_dispatch_claim", {"pr_numbers": [7704]})
    f.emit(
        a, "2026-10-03T13:00:00Z", "record_review",
        {"decision": "approved", "issue_number": 2704, "pr_number": 7704},
    )  # fmt: skip
    rollup.run_rollup(f.sources(), NOW)
    db, err = open_dashboard_ro(f.sources().db_path)
    assert db is not None, err

    s = flow.stage_time(db, CURRENT, "in_progress")
    assert [p for p, _ in s.points] == [day(4), day(5), day(6)]
    assert [v for _, v in s.points] == pytest.approx([1 / 3, 77 / 60, 2.5])
    # pr_open for 2704 ends at its escalation; the unescalate->claim wait is not
    # re-opened without a new PR-open milestone. reviewing runs claim -> verdict.
    assert flow.stage_time(db, CURRENT, "pr_open").points == ((day(3), 2.0),)
    assert flow.stage_time(db, CURRENT, "reviewing").points == ((day(3), 1.0),)
    assert flow.stage_time(db, CURRENT, "needs_rework").points == ()
    db.close()
    f.close()


def test_relabeled_sweep_expands_to_milestones_that_close_in_progress(
    tmp_path, monkeypatch
) -> None:
    """Issue #2473: the sweep's batch form expands to per-issue milestone rows.

    ``_append_sweep_events`` folds a multi-issue relabel into one
    ``session_failed_relabeled_sweep`` event carrying only ``issue_numbers``;
    the rollup expands it back so each batched issue's open ``in_progress``
    visit ends at the relabel, not at a far-later close.
    """
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    f = Fleet(tmp_path, monkeypatch)
    a = f.alpha
    dispatch(f, a, day(4, 0), [2710, 2711])
    f.emit(
        a, day(4, 2), "session_failed_relabeled_sweep",
        {"count": 2, "issue_numbers": [2710, 2711]},
    )  # fmt: skip
    f.emit(
        a, day(6, 0), "worker_handoff_pr_opened",
        {"issue_number": 2710, "pr_number": 7710},
    )  # fmt: skip
    rollup.run_rollup(f.sources(), NOW)
    db, err = open_dashboard_ro(f.sources().db_path)
    assert db is not None, err
    # both batched issues got their own milestone row off the one sweep event
    assert db.execute(
        "SELECT issue, milestone, event_kind FROM issue_milestones"
        " WHERE event_kind = 'session_failed_relabeled_sweep' ORDER BY issue"
    ).fetchall() == [
        (2710, "session_failed_relabeled", "session_failed_relabeled_sweep"),
        (2711, "session_failed_relabeled", "session_failed_relabeled_sweep"),
    ]
    # each visit closed at the relabel (2h); without the milestone 2710's would
    # stretch to the PR open 46h later (the #2473 phantom shape)
    assert flow.stage_time(db, CURRENT, "in_progress").points == ((day(4), 2.0),)
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
    # approx-era pr_open: issue 800's PR sat open 6h before merging unclaimed (issue
    # #2473: the merge itself ends the visit); exact era: 2h + 8h over two visits
    assert flow.stage_time(db, q, "pr_open").points == ((at(-3), 6.0), (at(2), 10.0))
    assert flow.stage_time(db, q, "needs_rework").points == ((at(2), 1.0),)
    assert flow.stage_time(db, q, "reviewing").points == ((at(2), 1.0),)
    assert [p for p in flow.merges_per_day(db, q).points if p[1]] == [(at(-3), 1.0), (at(2), 1.0)]

    after = MetricQuery(datetime(2026, 10, 2, tzinfo=UTC), NOW, timedelta(days=1))
    exact_only = flow.lead_time(db, after)
    assert exact_only.points == ((at(2), 20.0),) and exact_only.approx is False
    db.close()
    f.close()
