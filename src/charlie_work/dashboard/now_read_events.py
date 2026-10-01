"""Event-database reads for the Now collector: runner busy, pass cadence, merges in 24h.

Everything opens its database read-only (``?mode=ro``) and degrades to ``None`` / empty
(unknown), never to a made-up zero.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

from .. import layout
from ..safe_path import contains
from . import now_cadence
from . import sources as src
from .metrics_base import MetricQuery, open_dashboard_ro
from .metrics_flow import merges_per_day

CADENCE_WINDOW = timedelta(hours=24)


def repo_slug_from_github_url(url: str) -> str | None:
    """``owner/name`` from a runner registration URL (last two path segments)."""
    segments = [s for s in url.strip().rstrip("/").split("://", 1)[-1].split("/")[1:] if s]
    if len(segments) < 2:
        return None
    owner, name = segments[-2], segments[-1].removesuffix(".git")
    return f"{owner}/{name}" if owner and name else None


def runner_repos(managed_root: str) -> dict[str, str]:
    """Runner ``agentName`` -> ``owner/name``, read from each ``<root>/<dir>/.runner``.

    Non-recursive and containment-checked on the *resolved* path (a junction under the root
    must not hand back an unrelated runner service); read-only, nothing is started or parked.
    ``{}`` when the root is unset or unreadable.
    """
    if not managed_root:
        return {}
    root = Path(managed_root)
    try:
        entries = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return {}
    out: dict[str, str] = {}
    for entry in entries:
        if not contains(root, entry):
            continue
        try:
            data = json.loads((entry / ".runner").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        slug = (
            repo_slug_from_github_url(str(data.get("gitHubUrl", "")))
            if isinstance(data, dict)
            else None
        )
        if slug:
            out[str(data.get("agentName") or entry.name)] = slug
    return out


def runner_busy(
    conn: sqlite3.Connection, names: Mapping[str, str], now: datetime, max_age_seconds: float
) -> dict[str, int]:
    """Busy listeners per repo from the newest ``runner_health`` totals.

    A listener is busy when it has a ``Runner.Worker`` descendant (``process_count > 0``).
    A repo appears only when at least one of its runners was observed; a stale or missing
    event yields ``{}`` so the model reports unknown rather than idle.
    """
    event = src.latest_event(conn, "runner_health")
    when = src._parse_utc(event.get("ts")) if event else None
    if event is None or when is None or (now - when).total_seconds() > max_age_seconds:
        return {}
    totals = event["payload"].get("totals") if isinstance(event["payload"], dict) else None
    busy: dict[str, int] = {}
    for total in totals if isinstance(totals, list) else []:
        repo = names.get(str(total.get("runner"))) if isinstance(total, dict) else None
        if repo is None:
            continue
        count = total.get("process_count")
        working = isinstance(count, int) and not isinstance(count, bool) and count > 0
        busy[repo] = busy.get(repo, 0) + (1 if working else 0)
    return busy


def pass_completion_gaps(conn: sqlite3.Connection, now: datetime) -> list[float]:
    """Seconds between consecutive completed loop passes in the last 24h."""
    try:
        rows = conn.execute(
            "SELECT completed_at FROM loop_passes WHERE completed_at IS NOT NULL"
            " AND completed_at >= ?",
            ((now - CADENCE_WINDOW).strftime("%Y-%m-%dT%H:%M:%S"),),
        ).fetchall()
    except sqlite3.Error:
        return []
    whens = [w for (raw,) in rows if (w := src._parse_utc(raw)) is not None]
    return now_cadence.completion_gaps(whens)


def merged_24h(
    now: datetime, fleet_dir_override: str | None, max_lag_seconds: float
) -> int | None:
    """Issues merged in the last 24h from ``dashboard.db`` (same dedup as History's Merges).

    None (unknown, never 0) when ``dashboard.db`` is missing, covers nothing, or its newest
    ingested event lags ``now`` by more than ``max_lag_seconds`` (a rollup that has not run
    would silently under-count).
    """
    db, _err = open_dashboard_ro(layout.dashboard_db_path(override=fleet_dir_override))
    if db is None:
        return None
    try:
        series = merges_per_day(db, MetricQuery(now - CADENCE_WINDOW, now, CADENCE_WINDOW))
    except sqlite3.Error:
        return None
    finally:
        db.close()
    end = src._parse_utc(series.coverage_end)
    if not series.points or end is None or (now - end).total_seconds() > max_lag_seconds:
        return None
    return int(series.points[0][1])
