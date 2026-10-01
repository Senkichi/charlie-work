"""One-shot local read of the fleet sources into ``SourcesRead`` (read-only).

The long-lived collector (a later step) will reuse this; the CLI ``dashboard now``
calls it once. Per-repo caps and reviewer counts need the repo's config and
dispatch dirs, which a host-wide one-shot read does not load, so they stay
``None`` (the model treats ``None`` as "unknown", never as zero).
"""

from __future__ import annotations

from datetime import datetime

from .. import fleet_pause
from . import sources as src
from .now_types import RepoRead, SourcesRead


def collect_sources_read(now: datetime, fleet_dir_override: str | None = None) -> SourcesRead:
    """Read every local Now-model input; unreadable pieces degrade to ``None``/error values."""
    fs = src.fleet_sources(fleet_dir_override)
    repos = tuple(
        RepoRead(key=r.key, repo_root=str(r.repo_root), snapshot=src.read_snapshot(r, now))
        for r in src.enumerate_repos(fleet_dir_override)
    )
    event = None
    conn, _err = src.open_events_ro(fs.events_db)
    if conn is not None:
        try:
            event = src.latest_event(conn, "runner_allocation")
        finally:
            conn.close()
    return SourcesRead(
        repos=repos,
        runner_allocation=event,
        supervisor_heartbeat=src.read_json_file(fs.supervisor_heartbeat),
        pause=fleet_pause.fleet_pause_status(fleet_dir_override),
    )
