"""Doctor check aggregating ``dispatch_cross_repo_escalated`` events by the
sibling repo each escalation pointed at (issue #1789).

``run_doctor`` calls ``_check_cross_repo_escalations`` alongside the other
events.db lookback checks (``_check_recent_lane_failures``,
``_check_git_network_retries``).

Lives outside ``doctor`` on purpose: ``doctor.py`` is over the 800-line
module cap and pinned by the file-size high-water-mark ratchet
(``tests/test_file_size_ratchet.py``), which never allows an over-cap file to
grow past its recorded mark -- new code lands in a domain module instead.
"""

from __future__ import annotations

import datetime
from typing import Any

from .instrumentation import query_events
from .paths import RuntimePaths


_CROSS_REPO_ESCALATION_LOOKBACK_HOURS = 24


def _check_cross_repo_escalations(add: Any, paths: RuntimePaths) -> None:
    """Surface recent ``dispatch_cross_repo_escalated`` events, grouped by
    the sibling repo each escalation pointed at (issue #1789).

    ``CrossRepoGateResult.found_in_repo`` — the managed fleet repo a missing
    candidate was positively matched under — is recorded on the event
    payload at emission (``orchestration/dispatch_state.py``) but before
    this check had no consumer beyond ad-hoc events.db queries: this repo's
    own "signal without a consumer" antipattern, the
    same gap ``_check_git_network_retries`` documents. An operator had to
    open the raw event payload to learn which sibling repo an escalation
    pointed at; aggregating here turns it into the at-a-glance "N
    escalations pointing at repo X" fleet signal — the shape that spots a
    mis-registered or unexpectedly-overlapping repo pair.

    Escalations with no ``found_in_repo`` — a ``cross_repo_scope``
    title-prefix escalation, a confirmed foreign absolute path outside the
    fleet registry, or a legacy payload from before the field existed — are
    counted as unattributed, bucketed by the ``reason`` prefix
    (``cross_repo_scope`` / ``cross_repo_target``) so a scope-gate burst is
    still distinguishable from a foreign-checkout one.

    Warning, not error, and silent when the window is empty: the escalation
    already happened and the parked issue is its own ``human-needed``
    marker — this reports "did this recently happen, and where did it
    point", mirroring ``_check_recent_lane_failures``. Read-only:
    ``query_events()`` never raises and returns ``[]`` on any failure.
    """
    cutoff = (
        (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(hours=_CROSS_REPO_ESCALATION_LOOKBACK_HOURS)
        )
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    events = query_events(paths.state_file, kind="dispatch_cross_repo_escalated", since=cutoff)
    if not events:
        return

    repo_counts: dict[str, int] = {}
    unattributed: dict[str, int] = {}
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        found_in_repo = payload.get("found_in_repo")
        if isinstance(found_in_repo, str) and found_in_repo:
            repo_counts[found_in_repo] = repo_counts.get(found_in_repo, 0) + 1
        else:
            reason = payload.get("reason")
            prefix = reason.split(":", 1)[0] if isinstance(reason, str) and reason else "unknown"
            unattributed[prefix] = unattributed.get(prefix, 0) + 1

    detail = (
        f"{len(events)} dispatch_cross_repo_escalated event(s) in the last "
        f"{_CROSS_REPO_ESCALATION_LOOKBACK_HOURS}h, most recent at {events[-1].get('ts')}"
    )
    if repo_counts:
        detail += "; pointing at: " + ", ".join(
            f"{repo} ({count})" for repo, count in sorted(repo_counts.items())
        )
    if unattributed:
        detail += "; unattributed: " + ", ".join(
            f"{prefix} ({count})" for prefix, count in sorted(unattributed.items())
        )
    add(
        "cross-repo escalations",
        False,
        detail,
        severity="warning",
    )
