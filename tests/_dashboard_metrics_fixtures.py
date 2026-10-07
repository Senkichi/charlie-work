"""Fixtures for the dashboard metrics tests: a two-week fleet built through the real writers.

Windows (1-day buckets): current = [2026-10-01, 2026-10-08), prior = [2026-09-24, 2026-10-01).
Both repos and the global DB have an event at 09-24T01:00 and 10-07T23:00, so coverage spans
both windows.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _dashboard_rollup_fixtures import GOV, Fleet

from charlie_work import instrumentation
from charlie_work.dashboard import rollup
from charlie_work.dashboard.metrics_base import MetricQuery

NOW = datetime(2026, 10, 8, 0, 0, 0, tzinfo=UTC)
CURRENT = MetricQuery(datetime(2026, 10, 1, tzinfo=UTC), NOW, timedelta(days=1))
PRIOR = CURRENT.prior()


def day(n: int, hh: int = 0) -> str:
    """ts on 2026-10-<n> (n may be <= 0 to reach September)."""
    base = datetime(2026, 10, 1, tzinfo=UTC) + timedelta(days=n - 1, hours=hh)
    return base.strftime("%Y-%m-%dT%H:%M:%SZ")


def dispatch(f: Fleet, state: Path, ts: str, issues=(), live=1, fleet=3, dispatchable=5, limit=1):
    gov = {**GOV, "live_session_count": live, "fleet_live_session_count": fleet}
    gov["dispatch_limit"] = limit
    backlog = {"open_total": 47, "dispatchable": dispatchable}
    payload = {"concurrency_governor": gov, "backlog_reachability": backlog}
    f.emit(state, ts, "dispatch", {**payload, "issue_numbers": list(issues)})


def merged(f: Fleet, state: Path, ts: str, issue: int) -> None:
    f.emit(
        state, ts, "reconcile",
        {"kind": "merged_outside_orchestrator", "issue_number": issue, "pr_number": issue + 5000},
    )  # fmt: skip


def job(job_id: str, queue_wait: float) -> dict:
    d = {"kind": "measured", "seconds": queue_wait}
    return {
        "job_id": job_id,
        "name": "Tests",
        "status": "completed",
        "durations": {"queue_wait": d, "execution": d, "wall": d},
    }


def build_base(f: Fleet) -> None:
    """Edges of coverage, merges (6+2 prior, 2+3 current) and the flow-pass samples."""
    a, b, g = f.alpha, f.beta, f.fleet_state
    for state in (a, b):  # uncapped edge passes (dispatchable 0)
        dispatch(f, state, day(-6, 1), dispatchable=0)
        dispatch(f, state, day(7, 23), dispatchable=0)
    for n in range(-6, 8):  # a daily heartbeat: every bucket's source is alive, so zeros are real
        for state in (a, b, g):
            f.emit(state, day(n, 3), "supervisor_started", {})
    f.emit(g, day(-6, 1), "supervisor_started", {})
    f.emit(g, day(7, 23), "supervisor_started", {})
    for n in range(6):  # prior window: alpha 6, beta 2
        merged(f, a, day(-5 + n, 12), 101 + n)
    merged(f, b, day(-2, 9), 401)
    merged(f, b, day(0, 9), 402)
    # current window: alpha 201 (lead 10h) + 202 (36h), beta 301 (5h), 302 (5h), 303 (20h)
    dispatch(f, a, day(2, 2), [201], live=1, fleet=3, dispatchable=5, limit=1)
    dispatch(f, a, day(2, 0), [202], live=2, fleet=5, dispatchable=7, limit=9)
    dispatch(f, b, day(2, 13), [301])
    dispatch(f, b, day(5, 4), [302])
    dispatch(f, b, day(5, 0), [303])
    merged(f, a, day(2, 12), 201)
    merged(f, a, day(3, 12), 202)
    merged(f, a, day(4, 0), 201)  # duplicate report of 201: still one merge
    merged(f, b, day(2, 18), 301)
    merged(f, b, day(5, 9), 302)
    merged(f, b, day(5, 20), 303)
    f.emit(
        a, day(2, 5), "dispatch_backpressure",
        {"clamped_by": "host_load", "clamped_limit": 0, "requested_limit": 3},
    )  # fmt: skip


def build_quality(f: Fleet) -> None:
    a, b = f.alpha, f.beta
    rr = {"escalated": False, "issue_number": 1, "pr_number": 2}
    f.emit(a, day(2, 10), "record_review", {**rr, "decision": "approved"})
    f.emit(a, day(3, 10), "record_review", {**rr, "decision": "approved"})
    f.emit(b, day(2, 11), "record_review", {**rr, "decision": "approved"})
    f.emit(
        a, day(3, 11), "record_review", {**rr, "decision": "request_changes", "escalated": True}
    )
    f.emit(a, day(5, 1), "dispatch_rework", {"issue_numbers": [250], "pr_number": 251})
    f.emit(a, day(4, 3), "session_failed_escalated", {"issue_number": 60, "reason": "capped"})
    f.emit(a, day(4, 4), "unescalate", {"issue_number": 61, "cleared_escalation_reason": "x"})
    for i, ts in enumerate((day(3, 1), day(3, 2), day(3, 3))):
        f.emit(a, ts, "worker_handoff_pr_opened", {"issue_number": 10 + i, "pr_number": 20 + i})
    f.emit(a, day(3, 4), "orphaned_worker_opened_pr", {"issue_number": 15, "pr_number": 25})
    for ts, kind in ((day(2, 6), "stalled"), (day(2, 7), "stalled"), (day(4, 8), "rate_limited")):
        f.emit(a, ts, "session_exited", {"failure_kind": kind, "issue_number": 5})
    f.emit(b, day(2, 8), "session_exited", {"failure_kind": None, "issue_number": 6})
    f.emit(a, day(3, 5), "review_verdict_missed", {"reason": "launch_failed", "pr_number": 3})
    # Issue #2476: token reason + human detail + a cause object; PR 3 collects
    # three more misses from redispatch retries (one on a later day), so
    # attempts (6) outnumber distinct PRs (3) and api_error_status beats
    # cause.cause for the cause label.
    for d in (3, 3, 4):
        f.emit(
            a,
            day(d, 6),
            "review_verdict_missed",
            {
                "reason": "died_mid_session",
                "detail": "reviewer exited before writing a verdict (API error 429)",
                "pr_number": 3,
                "cause": {"cause": "died_mid_session", "api_error_status": 429},
            },
        )
    # Pre-#2476 shape: free text in ``reason`` (still carrying pr_number, as the
    # emit site always did) -- the rollup moves the text to ``detail``.
    for pr in (4, 5):
        f.emit(
            a,
            day(3, 7),
            "review_verdict_missed",
            {"reason": f"PR #{pr} is MERGED on GitHub", "pr_number": pr},
        )


def build_capacity(f: Fleet) -> None:
    a, g = f.alpha, f.fleet_state
    review = {
        "failed": [], "fleet_available_review_slots": 4, "fleet_live_review_count": 2,
        "fleet_review_concurrency_limit": 6, "launched": [], "quota_hit": False,
    }  # fmt: skip
    f.emit(a, day(2, 9), "review_dispatch", review)
    alloc = {"budget": 8, "targets": [
        {"repo": "o/x", "capacity": 1, "demand": 0, "running": 1, "target": 1},
        {"repo": "o/y", "capacity": 5, "demand": 2, "running": 2, "target": 3},
    ]}  # fmt: skip
    f.emit(g, day(3, 1), "runner_allocation", alloc)
    alloc2 = {**alloc, "targets": [{**alloc["targets"][0], "running": 0, "capacity": 3}]}
    f.emit(g, day(3, 2), "runner_allocation", alloc2)
    f.emit(g, day(3, 3), "fleet_job_observations", {"jobs": [job("j10", 10.0), job("j11", 30.0)]})
    f.emit(g, day(3, 4), "fleet_job_observations", {"jobs": [job("j12", 20.0)]})


def build_reliability(f: Fleet) -> None:
    a = f.alpha / "state.json"
    for corr, (start, end, secs, errors) in enumerate(
        [(day(2, 1), day(2, 2), 100.0, 1), (day(2, 3), day(2, 4), 200.0, 2)]
    ):
        instrumentation.record_loop_pass(a, f"c{corr}", start)
        instrumentation.record_loop_pass(
            a, f"c{corr}", start, end, ok=True, elapsed_seconds=secs, error_count=errors
        )
    ok = {"changed": True, "error": None, "from_sha": "a", "to_sha": "b", "ok": True}
    f.emit(f.alpha, day(2, 12), "self_deploy_succeeded", ok)
    f.emit(f.alpha, day(3, 12), "self_deploy_succeeded", ok)
    f.emit(f.alpha, day(4, 12), "self_deploy_failed", {**ok, "ok": False, "error": "boom"})
    f.emit(f.alpha, day(3, 8), "review_quota_exhausted", {"throttled_until": day(3, 13)})


@pytest.fixture
def ro_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Rolled-up dashboard.db of the full scenario, opened read-only."""
    from charlie_work.dashboard.metrics_base import open_dashboard_ro

    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    f = Fleet(tmp_path, monkeypatch)
    for build in (build_base, build_quality, build_capacity, build_reliability):
        build(f)
    assert rollup.run_rollup(f.sources(), NOW).errors == ()
    conn, err = open_dashboard_ro(f.sources().db_path)
    assert conn is not None, err
    yield conn
    conn.close()
    f.close()


def values(series) -> list[float]:
    return [v for _, v in series.points]
