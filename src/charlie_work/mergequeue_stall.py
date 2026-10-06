"""``mergequeue_stalled``: one alarm per episode of a PR sitting in the merge queue too long (#2441).

A PR handed to the merge queue carries ``auto_merge.mergequeue_label`` until the
queue merges it. On 2026-10-06 several PRs held the label for hours and never
merged, with nothing to say so. The #1401 wedge watchdog (``reconcile``) acts at
24h by escalating; this is the early, non-destructive signal.

The dwell clock is ``prs[n].mergequeue_since`` (stamped by the merge path's
accounting stage and preserved while the head is unchanged; a head change
restarts it). An *episode* is one ``mergequeue_since`` value:
``prs[n].mergequeue_stalled_since`` records the episode already alarmed, so the
event fires once per episode and a re-queue (new ``since``) can fire again.

Reads only data the pass already holds (the per-pass PR snapshot and the state
entry); no GitHub call. The only write is the event plus the episode marker,
under the state lock through the write gate.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from typing import Any

from .github import label_names
from .host import current as _host_current
from .instrumentation import log_event
from .state import load_state, state_lock

logger = logging.getLogger(__name__)

STALL_AFTER = timedelta(hours=2)


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def stalled_since(
    *,
    mergequeue_label: str | None,
    pr_labels: Collection[str],
    pr_entry: dict[str, Any],
    now: datetime,
) -> datetime | None:
    """The un-alarmed episode's start when the PR is queued past ``STALL_AFTER``, else ``None``."""
    if not mergequeue_label or mergequeue_label not in pr_labels:
        return None
    if pr_entry.get("status") != "mergequeue":
        return None
    since_raw = pr_entry.get("mergequeue_since")
    since = _parse(since_raw)
    if since is None or now - since <= STALL_AFTER:
        return None
    if pr_entry.get("mergequeue_stalled_since") == since_raw:
        return None  # this episode was already alarmed
    return since


def alarm_if_stalled(app: Any, pr: dict[str, Any], pr_entry: dict[str, Any]) -> bool:
    """Emit ``mergequeue_stalled`` for ``pr`` if its episode is new and overdue."""
    label = app.config.auto_merge.mergequeue_label
    now = _host_current().clock.now()
    labels = label_names(pr)
    if (
        app.dry_run
        or stalled_since(mergequeue_label=label, pr_labels=labels, pr_entry=pr_entry, now=now)
        is None
    ):
        return False
    pr_number = int(pr["number"])
    state_file = app.paths.state_file
    with state_lock(state_file):
        state = load_state(state_file)
        entry = state["prs"].get(str(pr_number), {})
        # Re-judged on the locked state: a concurrent pass may have alarmed or dequeued it.
        since = stalled_since(mergequeue_label=label, pr_labels=labels, pr_entry=entry, now=now)
        if since is None:
            return False
        state = app._record_event(
            state,
            "mergequeue_stalled",  # event-consumer: scripts/heartbeat_event_alarms.py check_mergequeue_stalled
            {
                "pr_number": pr_number,
                "issue_number": entry.get("issue_number"),
                "mergequeue_label": label,
                "queued_since": entry.get("mergequeue_since"),
                "head_sha": entry.get("mergequeue_head_sha"),
                "dwell_hours": round((now - since).total_seconds() / 3600.0, 2),
                "threshold_hours": STALL_AFTER.total_seconds() / 3600.0,
            },
        )
        state["prs"][str(pr_number)] = {
            **state["prs"].get(str(pr_number), {}),
            "mergequeue_stalled_since": entry.get("mergequeue_since"),
        }
        app.write_gate.save_state(state)
    return True


def alarm_best_effort(app: Any, pr: dict[str, Any], pr_entry: dict[str, Any]) -> bool:
    """``alarm_if_stalled`` that can never abort the per-PR scan.

    The alarm is advisory, but it takes the state lock and writes state, so a
    lock timeout or a write error must not skip review/merge handling for the
    remaining PRs. A failure is logged and recorded as a lock-free
    ``mergequeue_stall_alarm_failed`` event (``log_event`` is itself best-effort),
    and the next pass retries because the episode marker was never written.
    """
    try:
        return alarm_if_stalled(app, pr, pr_entry)
    except Exception as exc:  # noqa: BLE001 - advisory path: nothing may escape
        logger.warning("mergequeue stall alarm failed for PR #%s: %s", pr.get("number"), exc)
        log_event(
            app.paths.state_file,
            "mergequeue_stall_alarm_failed",  # event-consumer: audit-only advisory alarm failure; the exception is also logged
            {"pr_number": pr.get("number"), "error": f"{type(exc).__name__}: {exc}"},
        )
        return False
