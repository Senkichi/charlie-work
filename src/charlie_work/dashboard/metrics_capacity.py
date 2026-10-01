"""Capacity metrics: workers/reviewers/runners against caps, CI queue wait, capped demand."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

from .metrics_base import MetricQuery, Sample, Series, SeriesSpec, gauge_samples, make_series
from .metrics_flow import pass_gauge
from .rollup_schema import FLEET_SOURCE

_WINDOW = "ts >= ? AND ts < ? AND source != ?"


def workers_cap(db: sqlite3.Connection, q: MetricQuery) -> Series:
    """Worker cap: per repo its own limit, all-repos the fleet-wide limit."""
    spec = SeriesSpec(
        "workers_cap", "Worker cap", "workers", "gauge", ("dispatch",), check_instrumented=True
    )
    return pass_gauge(db, q, spec, "concurrency_limit", "fleet_concurrency_limit")


def _review_gauge(
    db: sqlite3.Connection, q: MetricQuery, name: str, label: str, col: str
) -> Series:
    """Mean of a ``review_samples`` column (already fleet-wide, so one line serves both views)."""
    rows = db.execute(
        f"SELECT ts, source, {col} FROM review_samples WHERE {_WINDOW}",
        (q.start_iso, q.end_iso, FLEET_SOURCE),
    ).fetchall()
    samples = gauge_samples(rows)
    spec = SeriesSpec(
        name, label, "reviewers", "gauge", ("review_dispatch",), check_instrumented=True
    )
    return make_series(db, q, spec, samples, samples, how="mean")


def reviewers_live(db: sqlite3.Connection, q: MetricQuery) -> Series:
    return _review_gauge(db, q, "reviewers_live", "Reviewers live", "live_reviews")


def reviewers_cap(db: sqlite3.Connection, q: MetricQuery) -> Series:
    return _review_gauge(db, q, "reviewers_cap", "Reviewer cap", "review_limit")


def _runner_gauge(
    db: sqlite3.Connection, q: MetricQuery, name: str, label: str, col: str
) -> Series:
    """Per allocation sample: all-repos = sum over targets, per repo = that target."""
    rows = db.execute(
        f"SELECT ts, target_repo, {col}, src_id FROM runner_samples WHERE ts >= ? AND ts < ?",
        (q.start_iso, q.end_iso),
    ).fetchall()
    per_repo = gauge_samples([(ts, repo, v) for ts, repo, v, _ in rows])
    totals: dict[tuple[int, str], float] = {}
    for ts, _, v, src_id in rows:
        if v is not None:
            totals[(src_id, ts)] = totals.get((src_id, ts), 0.0) + float(v)
    combined: list[Sample] = [(ts, "*", v) for (_, ts), v in sorted(totals.items())]
    spec = SeriesSpec(
        name, label, "runners", "gauge", ("runner_allocation",), sources="fleet",
        check_instrumented=True,
    )  # fmt: skip
    return make_series(db, q, spec, combined, per_repo, how="mean")


def runners_running(db: sqlite3.Connection, q: MetricQuery) -> Series:
    return _runner_gauge(db, q, "runners_running", "Runners running", "running")


def runners_capacity(db: sqlite3.Connection, q: MetricQuery) -> Series:
    return _runner_gauge(db, q, "runners_capacity", "Runner capacity", "capacity")


def ci_queue_wait(db: sqlite3.Connection, q: MetricQuery) -> Series:
    """Median CI queue wait of jobs completing per bucket (jobs carry no repo)."""
    rows = db.execute(
        "SELECT ts, 'ci', queue_wait_seconds FROM job_observations"
        " WHERE ts >= ? AND ts < ? AND queue_wait_seconds IS NOT NULL",
        (q.start_iso, q.end_iso),
    ).fetchall()
    samples = gauge_samples(rows)
    spec = SeriesSpec(
        "ci_queue_wait", "CI queue wait", "seconds", "duration", ("fleet_job_observations",),
        sources="fleet",
    )  # fmt: skip
    series = make_series(db, q, spec, samples, samples, how="median")
    return replace(series, per_repo={})


def capped_demand(db: sqlite3.Connection, q: MetricQuery) -> Series:
    """Capped-demand moments per bucket.

    Counts dispatch passes where Dispatchable issues outnumbered the dispatch limit
    (``backlog_reachability.dispatchable > concurrency_governor.dispatch_limit``) plus the
    explicit ``dispatch_backpressure`` / ``dispatch_deferred`` / ``dispatch_starved`` /
    ``runner_capacity_starved`` events.
    """
    passes = db.execute(
        "SELECT ts, source FROM pass_samples WHERE ts >= ? AND ts < ?"
        " AND dispatchable > dispatch_limit",
        (q.start_iso, q.end_iso),
    ).fetchall()
    events = db.execute(
        "SELECT ts, repo FROM capped_demand WHERE ts >= ? AND ts < ?",
        (q.start_iso, q.end_iso),
    ).fetchall()
    samples: list[Sample] = [(ts, repo, 1.0) for ts, repo in (*passes, *events)]
    spec = SeriesSpec(
        "capped_demand", "Capped demand", "moments", "count",
        ("dispatch", "dispatch_backpressure", "dispatch_deferred", "dispatch_starved"),
        sources="all", pulse="dispatch",
    )  # fmt: skip
    return make_series(db, q, spec, samples, samples, how="sum")
