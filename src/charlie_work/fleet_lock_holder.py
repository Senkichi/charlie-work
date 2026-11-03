"""Fleet-launch-lock holder diagnostics (issue #2055).

``worker_launch_gate.FleetLaunchLock`` writes a holder sidecar while it holds
the fleet lock; these helpers read it back so a starved waiter's
``fleet_lock_held`` deferral can say who held the lock and for how long.

Relocated verbatim from ``worker_launch_gate`` (issue #2766): that module sat at
the 800-line cap (``tests/test_file_size_ratchet.py``) and the pause gate needed
room. Nothing here depends on the gate; the sidecar's *writer* stays with the
lock handle.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from charlie_work import layout

if TYPE_CHECKING:
    from charlie_work.workflow import OrchestratorApp


def read_fleet_lock_holder(fleet_dir_override: str | None) -> dict[str, Any]:
    """Best-effort read of the fleet-launch-lock holder sidecar (issue #2055).

    Returns ``{}`` when absent or unreadable -- a stale sidecar (holder died
    mid-hold) is still useful diagnostic data for a ``fleet_lock_held``
    deferral, so no liveness check is done here.
    """
    try:
        raw = json.loads(
            layout.fleet_lock_holder_path(override=fleet_dir_override).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _holder_held_seconds(acquired_at: Any) -> float | None:
    """Seconds since the holder sidecar's ``acquired_at``; ``None`` if unparseable."""
    if not isinstance(acquired_at, str) or not acquired_at:
        return None
    try:
        acquired = datetime.fromisoformat(acquired_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(UTC) - acquired).total_seconds())


def fleet_lock_held_extra(app: OrchestratorApp, wait_seconds: float) -> dict[str, Any]:
    """Payload fields for a ``fleet_lock_held`` deferral (shared by every lane).

    Carries the exhausted wait budget and, best-effort, who holds the lock.
    """
    holder = read_fleet_lock_holder(app.fleet_dir_override)
    extra: dict[str, Any] = {"lock_wait_seconds": wait_seconds}
    if isinstance(holder.get("repo"), str):
        extra["lock_holder_repo"] = holder["repo"]
    if isinstance(holder.get("pid"), int):
        extra["lock_holder_pid"] = holder["pid"]
    held_seconds = _holder_held_seconds(holder.get("acquired_at"))
    if held_seconds is not None:
        extra["lock_held_seconds"] = held_seconds
    return extra
