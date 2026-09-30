"""Periodic-pass scheduling helpers (file-size ratchet #1442; extracted from ``state.py``).

Extracted verbatim from ``state.py``: the three "next run at" cadence
sub-dicts and their due/arm helpers -- worktree reclamation (issue #636),
the in-loop reconcile pass (merge-lane-recovery section 6-B) and the
mechanical de-escalation sweep (issue #783).

``state.py`` re-exports every public symbol here via a facade import block
(the ``dispatch_cadence.py`` pattern), so existing import paths
(``charlie_work.state.arm_reconcile_pass`` and friends) and the
``workflow.py`` facade keep working unchanged. This module imports nothing
from the package, so it cannot create an import cycle.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _worktree_reclamation(data: dict[str, Any]) -> dict[str, Any]:
    """Return the worktree-reclamation scheduling sub-dict from ``data``.

    Tracks the ``next_run_at`` timestamp that gates the cadence-gated
    ``clean_worktrees`` sweep fired from the fleet pass (issue #636). Returns a
    mutable copy so callers can build new state without mutating ``data``.
    """
    sched = data.get("worktree_reclamation")
    if not isinstance(sched, dict):
        return {}
    return dict(sched)


def is_worktree_reclamation_due(data: dict[str, Any]) -> bool:
    """True when the reclamation sweep's interval has elapsed.

    An absent schedule means "never run yet", which is treated as due so the
    first fleet pass after startup clears the existing backlog of merged-PR
    worktrees (the exact accumulation issue #636 exists to fix). Malformed
    timestamps are also treated as due so a corrupt value cannot wedge
    reclamation off forever.
    """
    next_at = _worktree_reclamation(data).get("next_run_at")
    if not next_at:
        return True
    try:
        next_time = datetime.fromisoformat(next_at.replace("Z", "+00:00"))
        return datetime.now(UTC) >= next_time
    except (ValueError, TypeError):
        return True


def schedule_worktree_reclamation(data: dict[str, Any], next_run_at: str) -> dict[str, Any]:
    """Set the next reclamation sweep timestamp.

    Called after a sweep runs (or is skipped as not-due-armed) so the next
    sweep fires roughly ``interval_minutes`` later rather than on the very next
    pass. Returns a new state dict; does not mutate ``data``.
    """
    sched = _worktree_reclamation(data)
    sched["next_run_at"] = next_run_at
    return {**data, "worktree_reclamation": sched}


def _reconcile_pass(data: dict[str, Any]) -> dict[str, Any]:
    """Return the periodic in-loop reconcile scheduling sub-dict from ``data``.

    Distinct from ``quota_probe``: tracks the merge-lane-recovery §6-B
    cadence (``OrchestratorApp._maybe_reconcile_drift``), not the quota
    probe. Ensures a mutable copy so callers can build new state without
    mutating the original ``data``.
    """
    section = data.get("reconcile_pass")
    if not isinstance(section, dict):
        return {}
    return dict(section)


def is_reconcile_due(data: dict[str, Any]) -> bool:
    """True when the periodic in-loop reconcile pass should run.

    Unlike the quota probe (which only arms once a throttle indicator is
    observed, deliberately delaying the first real probe), reconcile has no
    "is something wrong" precondition to wait on -- it is a plain periodic
    cadence, so an absent schedule (never run before, e.g. right after a
    fresh deploy) is treated as due immediately rather than requiring one
    full interval to elapse first. This matters for G1: the divergence class
    this closes has already been sitting unrepaired indefinitely, so the
    first pass after this lands should not wait `interval_minutes` before
    doing anything. Malformed timestamps are treated as due, mirroring
    ``is_quota_probe_due``, so a corrupt value cannot wedge reconcile off
    forever.
    """
    next_at = _reconcile_pass(data).get("next_reconcile_at")
    if not next_at:
        return True
    try:
        next_time = datetime.fromisoformat(next_at.replace("Z", "+00:00"))
        return datetime.now(UTC) >= next_time
    except (ValueError, TypeError):
        return True


def arm_reconcile_pass(data: dict[str, Any], next_reconcile_at: str) -> dict[str, Any]:
    """Schedule the next periodic in-loop reconcile attempt.

    Returns a new state dict; does not mutate ``data``.
    """
    section = _reconcile_pass(data)
    section["next_reconcile_at"] = next_reconcile_at
    return {**data, "reconcile_pass": section}


def _deescalation_pass(data: dict[str, Any]) -> dict[str, Any]:
    """Return the periodic de-escalation sweep scheduling sub-dict from ``data``.

    Mirrors ``_reconcile_pass``: tracks issue #783's
    ``OrchestratorApp._maybe_deescalate_mechanical`` cadence. Ensures a
    mutable copy so callers can build new state without mutating ``data``.
    """
    section = data.get("deescalation_pass")
    if not isinstance(section, dict):
        return {}
    return dict(section)


def is_deescalation_due(data: dict[str, Any]) -> bool:
    """True when the periodic de-escalation sweep should run.

    Same "absent schedule is due immediately" semantics as
    ``is_reconcile_due`` -- a fresh deploy should not wait a full interval
    before its first pass, and a malformed timestamp is treated as due
    rather than wedging the sweep off forever.
    """
    next_at = _deescalation_pass(data).get("next_deescalation_at")
    if not next_at:
        return True
    try:
        next_time = datetime.fromisoformat(next_at.replace("Z", "+00:00"))
        return datetime.now(UTC) >= next_time
    except (ValueError, TypeError):
        return True


def arm_deescalation_pass(data: dict[str, Any], next_deescalation_at: str) -> dict[str, Any]:
    """Schedule the next periodic de-escalation sweep attempt.

    Returns a new state dict; does not mutate ``data``.
    """
    section = _deescalation_pass(data)
    section["next_deescalation_at"] = next_deescalation_at
    return {**data, "deescalation_pass": section}
