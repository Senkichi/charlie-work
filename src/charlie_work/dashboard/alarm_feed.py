"""Alarm findings the dashboard collector can evaluate from local, read-only data.

Wired here: ``loop-pass-freshness`` per registered repo (newest ``loop_started`` in
that repo's ``events.db``, opened read-only).

NOT wired yet (TODO, each needs a source the collector does not load today):
  - supervisor-heartbeat: already surfaced as a ``supervisor`` Needs-me row by
    ``now_needs_me`` from the same heartbeat file; wiring the leaf would duplicate it.
  - eval_error_events / eval_warning_events / eval_kind_events / eval_draft_pr_blocked /
    eval_infra_blocked / eval_ci_headroom_unavailable / eval_local_lane_stalled:
    need a per-repo events window with a baseline watermark (the heartbeat's cursor).
  - eval_log_freshness / eval_wedge_kill_loop / eval_notify_digest: need log mtimes,
    wedge-kill timestamps and the digest sidecar.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from .. import heartbeat_alarms_fleet as alarms
from . import sources as src
from .now_types import FindingLike


def loop_pass_findings(
    now: datetime, fleet_dir_override: str | None = None
) -> tuple[FindingLike, ...]:
    """One loop-pass-freshness finding per repo whose events.db is readable (never raises)."""
    found: list[FindingLike] = []
    for repo in src.enumerate_repos(fleet_dir_override):
        conn, _err = src.open_events_ro(repo.events_db)
        if conn is None:
            continue
        try:
            row = conn.execute("SELECT MAX(ts) FROM events WHERE kind = 'loop_started'").fetchone()
        except sqlite3.Error:  # table absent / locked: no verdict beats a false alarm
            continue
        finally:
            conn.close()
        newest = row[0] if row and isinstance(row[0], str) else None
        found.append(alarms.eval_loop_pass_freshness(repo.key, newest, now))
    return tuple(found)
