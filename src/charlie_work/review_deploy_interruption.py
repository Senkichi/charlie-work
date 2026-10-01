"""Deploy-interrupted reviewer detection (issue #2103).

A code-only ``self_deploy`` moves HEAD and the ``fleet supervise`` drift exit
replaces the daemon, so a reviewer in flight at that moment dies with no
verdict. That death says nothing about the PR under review, yet the reaper
used to record it as an ordinary ``review_verdict_missed`` and leave the
claim's attempt counted against ``max_review_dispatch_attempts``.

The only durable witness of the deploy is the ``self_deploy_succeeded`` row
``supervise.self_deploy`` writes to the orchestrator's own ``events.db``
(:func:`charlie_work.supervise._self_deploy_state_path`) -- not the consumer
repo's, which is where the reviewer's claim lives. This module reads that row.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from charlie_work.instrumentation import query_events

SELF_DEPLOY_SUCCEEDED = "self_deploy_succeeded"


def self_deploy_state_path() -> Path:
    """The ``state.json`` path whose sibling ``events.db`` records self-deploys."""
    # Lazy: ``supervise`` is a heavy module and this one is imported by the
    # reviewer-reap path that every loop pass reaches.
    from charlie_work.supervise import _self_deploy_state_path, orchestrator_root

    return _self_deploy_state_path(orchestrator_root())


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def deploy_interrupted_review(
    started_at: str | None, deploy_state_path: Path | None
) -> str | None:
    """Return the ts of a ``self_deploy_succeeded`` after ``started_at``, else ``None``.

    ``None`` (never raises) when the reviewer's start time is unparseable, the
    deploy store is absent or unreadable, or no deploy landed after the start:
    absence of evidence keeps the death classified as an ordinary miss.
    """
    started = _parse_ts(started_at)
    if started is None or deploy_state_path is None:
        return None
    if not deploy_state_path.with_name("events.db").exists():
        return None
    try:
        events = query_events(deploy_state_path, kind=SELF_DEPLOY_SUCCEEDED, limit=20)
    except Exception:  # best-effort evidence; never block the reap
        return None
    for event in reversed(events):
        ts = event.get("ts")
        deployed = _parse_ts(ts)
        if deployed is not None and deployed > started:
            return ts
    return None
