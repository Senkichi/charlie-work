"""Doctor check surfacing recent ``self_deploy_sync_starved`` events (issue #1855).

``run_doctor`` calls ``_check_sync_starvation`` alongside the other events.db
lookback checks (``_check_recent_lane_failures``,
``_check_git_network_retries``, ``_check_cross_repo_escalations``).

Lives outside ``doctor`` on purpose: ``doctor.py`` is over the 800-line
module cap and pinned by the file-size high-water-mark ratchet
(``tests/test_file_size_ratchet.py``), which never allows an over-cap file to
grow past its recorded mark -- new code lands in a domain module instead.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from .instrumentation import query_events
from .pending_sync import SELF_DEPLOY_SYNC_STARVED_KIND


_SYNC_STARVATION_LOOKBACK_HOURS = 24


def _check_sync_starvation(add: Any, self_deploy_state_path: Path) -> None:
    """Surface recent ``self_deploy_sync_starved`` events (issue #1855).

    ``self_deploy`` records the event against the orchestrator checkout's own
    state file (``supervise._self_deploy_state_path``), which is what the
    caller passes in -- a doctor run against another fleet repo still needs
    charlie-work's own events.db, exactly the dual-path shape
    ``_check_git_network_retries`` documents.

    Severity is a warning, not a hard error: starvation means the system
    detected the deferred-sync pileup and is already draining dispatch to
    resolve it -- this reports "the bound recently tripped", not a live
    health gate. Silent when the window is empty; ``query_events()`` never
    raises and returns ``[]`` on any query failure.
    """
    cutoff = (
        (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(hours=_SYNC_STARVATION_LOOKBACK_HOURS)
        )
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    events = query_events(self_deploy_state_path, kind=SELF_DEPLOY_SYNC_STARVED_KIND, since=cutoff)
    if not events:
        return
    latest = events[-1]
    payload = latest.get("payload") if isinstance(latest.get("payload"), dict) else {}
    add(
        "dependency sync starvation",
        False,
        f"{len(events)} self_deploy_sync_starved event(s) in the last "
        f"{_SYNC_STARVATION_LOOKBACK_HOURS}h, most recent at {latest.get('ts')}: "
        f"deferred uv sync pending {payload.get('pending_seconds')}s against a "
        f"{payload.get('starvation_seconds')}s bound",
        severity="warning",
    )
