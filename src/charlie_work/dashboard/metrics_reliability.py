"""Reliability metrics: loop passes, launch failures, throttling, self-deploys."""

from __future__ import annotations

import sqlite3

from .metrics_base import (
    MetricQuery,
    Sample,
    Series,
    SeriesSpec,
    category_series,
    gauge_samples,
    make_series,
)

_SIGNAL_EXITS = ("rate_limited", "quota_exhausted")


def _passes(db: sqlite3.Connection, q: MetricQuery, col: str) -> list[Sample]:
    rows = db.execute(
        f"SELECT completed_at, source, {col} FROM loop_passes"
        " WHERE completed_at >= ? AND completed_at < ?",
        (q.start_iso, q.end_iso),
    ).fetchall()
    return gauge_samples(rows)


def loop_pass_duration(db: sqlite3.Connection, q: MetricQuery) -> Series:
    """Median wall time of loop passes completing per bucket."""
    samples = _passes(db, q, "elapsed_seconds")
    spec = SeriesSpec(
        "loop_pass_duration", "Loop pass duration", "seconds", "duration", ("*",),
        check_instrumented=True,
    )  # fmt: skip
    return make_series(db, q, spec, samples, samples, how="median")


def loop_pass_errors(db: sqlite3.Connection, q: MetricQuery) -> Series:
    """Errors reported by loop passes completing per bucket."""
    samples = _passes(db, q, "error_count")
    spec = SeriesSpec("loop_pass_errors", "Loop pass errors", "errors", "count", ("*",))
    return make_series(db, q, spec, samples, samples, how="sum")


def launch_failures(db: sqlite3.Connection, q: MetricQuery) -> Series:
    """Launch failures, PARTIAL: no event kind records them (follow-up filed).

    Only what ``review_verdict_missed`` covers (``reason`` = ``launch_failed``) is counted;
    ``dispatch.failures`` is not rolled up. The series is flagged ``partial``.
    """
    rows = db.execute(
        "SELECT ts, source FROM verdict_missed"
        " WHERE reason_group = 'launch_failed' AND ts >= ? AND ts < ?",
        (q.start_iso, q.end_iso),
    ).fetchall()
    samples: list[Sample] = [(ts, repo, 1.0) for ts, repo in rows]
    spec = SeriesSpec(
        "launch_failures", "Launch failures", "failures", "count", ("review_verdict_missed",)
    )
    return make_series(db, q, spec, samples, samples, how="sum", partial=True)


def throttles(db: sqlite3.Connection, q: MetricQuery) -> tuple[Series, ...]:
    """Rate limits and quota exhaustion: throttle events plus rate-limited worker exits."""
    events = db.execute(
        "SELECT ts, source, event_kind FROM throttles WHERE ts >= ? AND ts < ?",
        (q.start_iso, q.end_iso),
    ).fetchall()
    exits = db.execute(
        "SELECT ts, source, 'worker_' || failure_kind FROM worker_exits"
        f" WHERE failure_kind IN ({', '.join('?' for _ in _SIGNAL_EXITS)}) AND ts >= ? AND ts < ?",
        (*_SIGNAL_EXITS, q.start_iso, q.end_iso),
    ).fetchall()
    spec = SeriesSpec(
        "throttles", "Rate limits and quota", "events", "count",
        ("review_quota_exhausted", "session_rate_limit_deferred", "graphql_rate_limit_deferred",
         "session_exited"),
        sources="all",
    )  # fmt: skip
    return category_series(db, q, spec, [*events, *exits])


def self_deploys(db: sqlite3.Connection, q: MetricQuery) -> tuple[Series, ...]:
    """Self-deploys per bucket: total, then succeeded / failed."""
    rows = db.execute(
        "SELECT ts, source, CASE ok WHEN 1 THEN 'succeeded' ELSE 'failed' END FROM deploys"
        " WHERE ts >= ? AND ts < ?",
        (q.start_iso, q.end_iso),
    ).fetchall()
    spec = SeriesSpec(
        "self_deploys", "Self-deploys", "deploys", "count",
        ("self_deploy_succeeded", "self_deploy_failed"), sources="all",
    )  # fmt: skip
    return category_series(db, q, spec, rows, ("succeeded", "failed"))
