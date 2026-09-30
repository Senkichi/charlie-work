"""Event-ring queries for the worktree_unsafe stranded-commit salvage.

Kept out of ``orchestration/misc_worker_dispatch.py``: the delegate installer
attaches every top-level ``def`` of a delegate module to ``OrchestratorApp``,
and this helper takes no ``self``.
"""

from __future__ import annotations

from typing import Any


def repeats_last_salvage_failure(state: dict[str, Any], payload: dict[str, Any]) -> bool:
    """True when the newest ``worktree_unsafe_stranded_salvage_failed`` event
    for this issue already carries the same branch and skip reason. Derived
    from the event ring itself, so there is no marker to go stale."""
    for event in reversed(state.get("events", [])):
        if not isinstance(event, dict):
            continue
        prior = event.get("payload")
        if not isinstance(prior, dict) or prior.get("issue_number") != payload["issue_number"]:
            continue
        if event.get("kind") == "worktree_unsafe_stranded_salvaged":
            return False
        if event.get("kind") == "worktree_unsafe_stranded_salvage_failed":
            return prior.get("branch") == payload["branch"] and prior.get(
                "skip_reason"
            ) == payload.get("skip_reason")
    return False
