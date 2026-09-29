"""The ``dead_dispatched_reap_minutes`` due-check shared by both sides of the
issue #1971 split.

The #654 timed backstop (``orphaned_worker_sweep.maybe_reap_dead_dispatched_
worker``) escalates a dead ``dispatched`` entry once ``orphan_drift_at`` has
been armed longer than ``dead_dispatched_reap_minutes``. Issue #1971 added a
pre-lock salvageable-park drain on no-PR backends
(``local_work_park.park_backstop_due_local_orphans``) that must select exactly
the same "backstop-due" entries the in-lock timer is about to escalate -- the
park itself takes ``state_lock`` and does git/label I/O, so it can never run
inside the locked classification where the timer lives. Housing the predicate
here (rather than in either consumer) keeps the two due-checks provably
identical and avoids an import cycle between the sweep and the park module.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .throttle_signatures import is_provider_throttle_failure


def dead_dispatched_reap_due(
    *,
    state: dict[str, Any],
    entry: dict[str, Any],
    dead_dispatched_reap_minutes: float,
    now: datetime,
) -> bool:
    """Whether the #654 timed backstop would escalate this entry this pass.

    The pure-predicate half of
    ``orphaned_worker_sweep.maybe_reap_dead_dispatched_worker``, extracted
    (issue #1971) so the sweep's pre-lock salvageable-park probe can select
    exactly the backstop-due entries without duplicating the drift-expiry or
    provider-throttle exemption logic.
    """
    # Deferred: workflow.py imports this module's consumers top-level, so a
    # top-level import here would cycle. Attribute access through the module
    # object also keeps suite patches on ``charlie_work.workflow.<name>``
    # live.
    import charlie_work.workflow as _wf

    orphan_drift_at = entry.get("orphan_drift_at")
    if orphan_drift_at is None or dead_dispatched_reap_minutes <= 0:
        return False
    if is_provider_throttle_failure(entry.get("dead_worker_failure_kind")):
        throttled_until_dt = _wf._parse_iso_timestamp(state.get("throttled_until"))
        if throttled_until_dt is not None and throttled_until_dt > now:
            return False
    drift_dt = _wf._parse_iso_timestamp(orphan_drift_at)
    return (
        drift_dt is not None
        and (now - drift_dt).total_seconds() / 60 >= dead_dispatched_reap_minutes
    )
