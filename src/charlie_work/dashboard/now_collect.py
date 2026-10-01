"""One-shot local read of the fleet sources into ``SourcesRead`` (read-only).

The long-lived collector (a later step) will reuse this; the CLI ``dashboard now``
calls it once. Every input is read from where the fleet itself keeps it: caps from the
layered config loader, ages and live reviewers from ``state.json``, busy listeners from
the newest ``runner_health`` event, merges from ``dashboard.db``. An input that cannot be
read stays ``None`` / empty (the model treats that as "unknown", never as zero).
"""

from __future__ import annotations

from datetime import datetime

from .. import fleet_pause
from . import now_cadence
from . import sources as src
from .config import DashboardConfig
from .now_read_config import RepoConfigRead, read_repo_config
from .now_read_events import (
    merged_24h,
    pass_completion_gaps,
    runner_busy,
    runner_repos,
)
from .now_read_state import read_repo_state
from .now_types import RepoRead, SourcesRead


def _read_repo(
    r: src.RepoSource, now: datetime, cfg: RepoConfigRead | None
) -> tuple[RepoRead, list[float]]:
    state = read_repo_state(r.state_dir, cfg.reviews_dir if cfg else "")
    gaps: list[float] = []
    conn, _err = src.open_events_ro(r.events_db)
    if conn is not None:
        try:
            gaps = pass_completion_gaps(conn, now)
        finally:
            conn.close()
    return (
        RepoRead(
            key=r.key,
            repo_root=str(r.repo_root),
            snapshot=src.read_snapshot(r, now),
            reviewers_live=state.reviewers_live,
            worker_cap=cfg.worker_cap if cfg else None,
            review_cap=cfg.review_cap if cfg else None,
            escalated_since=state.escalated_since,
        ),
        gaps,
    )


def collect_sources_read(
    now: datetime,
    fleet_dir_override: str | None = None,
    collector_interval_seconds: float = float(DashboardConfig.collector_interval_seconds),
) -> SourcesRead:
    """Read every local Now-model input; unreadable pieces degrade to ``None``/error values."""
    fs = src.fleet_sources(fleet_dir_override)
    repo_sources = src.enumerate_repos(fleet_dir_override)
    configs = {r.key: read_repo_config(r.repo_root, fleet_dir_override) for r in repo_sources}
    read = [_read_repo(r, now, configs[r.key]) for r in repo_sources]
    repos = tuple(repo for repo, _ in read)
    loaded = [c for c in configs.values() if c is not None]
    heartbeat = src.read_json_file(fs.supervisor_heartbeat)
    gap_p90 = now_cadence.p90([g for _, gaps in read for g in gaps])
    threshold = now_cadence.stale_threshold_seconds(
        heartbeat.data, collector_interval_seconds, gap_p90
    )
    event = None
    busy: dict[str, int] = {}
    conn, _err = src.open_events_ro(fs.events_db)
    if conn is not None:
        try:
            event = src.latest_event(conn, "runner_allocation")
            managed_root = next((c.managed_root for c in loaded if c.managed_root), "")
            busy = runner_busy(conn, runner_repos(managed_root), now, threshold)
        finally:
            conn.close()
    return SourcesRead(
        repos=repos,
        # The fleet.* caps are host-wide: every repo's merged config carries the same global
        # layer, so the first readable one speaks for the fleet.
        global_worker_cap=loaded[0].global_worker_cap if loaded else None,
        global_review_cap=loaded[0].global_review_cap if loaded else None,
        runner_allocation=event,
        runner_busy=busy,
        supervisor_heartbeat=heartbeat,
        pause=fleet_pause.fleet_pause_status(fleet_dir_override),
        done_24h=merged_24h(now, fleet_dir_override, threshold),
        collector_interval_seconds=collector_interval_seconds,
        snapshot_gap_p90_seconds=gap_p90,
    )
