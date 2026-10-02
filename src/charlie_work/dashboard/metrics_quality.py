"""Quality metrics: verdict mix, rework rate, escalations, worker fate, salvage, verdicts missed."""

from __future__ import annotations

import sqlite3

from .metrics_base import (
    MetricQuery,
    Sample,
    Series,
    SeriesSpec,
    category_series,
    make_ratio_series,
)

_WINDOW = "ts >= ? AND ts < ?"
# ``escalations.reason`` is already the rollup's normalised category; still bound the series
# count so a new free-text head can never mint an unbounded catalogue.
MAX_ESCALATION_CATEGORIES = 8


def _rows(db: sqlite3.Connection, sql: str, q: MetricQuery, *extra) -> list[tuple]:
    return db.execute(sql, (q.start_iso, q.end_iso, *extra)).fetchall()


def _ones(rows: list[tuple]) -> list[Sample]:
    return [(ts, repo, 1.0) for ts, repo, *_ in rows]


def verdict_mix(db: sqlite3.Connection, q: MetricQuery) -> tuple[Series, ...]:
    """Review verdicts per bucket: total, then approved / request_changes / blocked."""
    rows = _rows(
        db,
        "SELECT ts, source, substr(milestone, 9) FROM issue_milestones"
        f" WHERE milestone LIKE 'verdict\\_%' ESCAPE '\\' AND {_WINDOW}",
        q,
    )
    spec = SeriesSpec("verdict_mix", "Review verdicts", "verdicts", "count", ("record_review",))
    return category_series(db, q, spec, rows, ("approved", "request_changes", "blocked"))


def rework_rate(db: sqlite3.Connection, q: MetricQuery) -> Series:
    """Rework dispatches / all dispatches (first-run plus rework)."""
    rows = _rows(
        db,
        "SELECT ts, source, milestone FROM issue_milestones"
        f" WHERE milestone IN ('dispatched', 'rework_dispatched') AND {_WINDOW}",
        q,
    )
    num = [(ts, repo, 1.0) for ts, repo, m in rows if m == "rework_dispatched"]
    spec = SeriesSpec(
        "rework_rate", "Rework rate", "ratio", "ratio", ("dispatch", "dispatch_rework")
    )
    return make_ratio_series(db, q, spec, num, _ones(rows))


def escalations(db: sqlite3.Connection, q: MetricQuery) -> tuple[Series, ...]:
    """Escalations (un-escalations excluded) per bucket: total, then by reason."""
    rows = _rows(
        db,
        "SELECT ts, source, COALESCE(reason, 'unknown') FROM escalations"
        f" WHERE event_kind != 'unescalate' AND {_WINDOW}",
        q,
    )
    spec = SeriesSpec(
        "escalations", "Escalations", "escalations", "count",
        ("session_failed_escalated", "review_dispatch_escalated", "record_review"),
        any_kind=True,  # alternative routes into the same escalated state
    )  # fmt: skip
    return category_series(db, q, spec, rows, top=MAX_ESCALATION_CATEGORIES)


def worker_fate(db: sqlite3.Connection, q: MetricQuery) -> tuple[Series, ...]:
    """Worker fate: exit failure kinds, plus handoff and orphan PR openings."""
    exits = _rows(
        db,
        f"SELECT ts, source, COALESCE(failure_kind, 'unclassified') FROM worker_exits WHERE {_WINDOW}",
        q,
    )
    prs = _rows(
        db,
        "SELECT ts, source, CASE event_kind WHEN 'worker_handoff_pr_opened' THEN 'handoff_pr'"
        " ELSE 'orphan_pr' END FROM issue_milestones WHERE event_kind IN"
        f" ('worker_handoff_pr_opened', 'orphaned_worker_opened_pr') AND {_WINDOW}",
        q,
    )
    spec = SeriesSpec(
        "worker_fate", "Worker fate", "workers", "count",
        ("session_exited", "worker_handoff_pr_opened", "orphaned_worker_opened_pr"),
        any_kind=True,  # alternative fates of one worker
    )  # fmt: skip
    return category_series(db, q, spec, exits + prs)


def salvage_share(db: sqlite3.Connection, q: MetricQuery) -> Series:
    """orphaned_worker_opened_pr / (orphaned + worker_handoff_pr_opened)."""
    rows = _rows(
        db,
        "SELECT ts, source, event_kind FROM issue_milestones WHERE event_kind IN"
        f" ('worker_handoff_pr_opened', 'orphaned_worker_opened_pr') AND {_WINDOW}",
        q,
    )
    num = [(ts, repo, 1.0) for ts, repo, k in rows if k == "orphaned_worker_opened_pr"]
    spec = SeriesSpec(
        "salvage_share", "Salvage share", "ratio", "ratio",
        ("worker_handoff_pr_opened", "orphaned_worker_opened_pr"),
    )  # fmt: skip
    return make_ratio_series(db, q, spec, num, _ones(rows))


def verdicts_missed(db: sqlite3.Connection, q: MetricQuery) -> tuple[Series, ...]:
    """``review_verdict_missed`` per bucket: total, then by grouped reason."""
    rows = _rows(
        db,
        "SELECT ts, source, COALESCE(reason_group, 'unknown') FROM verdict_missed"
        f" WHERE {_WINDOW}",
        q,
    )
    spec = SeriesSpec(
        "verdicts_missed",
        "Review verdicts missed",
        "verdicts",
        "count",
        ("review_verdict_missed",),
    )
    return category_series(db, q, spec, rows)
