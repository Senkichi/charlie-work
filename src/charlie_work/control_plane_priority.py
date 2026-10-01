"""Raise the fleet control plane to NORMAL CPU priority at startup (issue #2140).

Task Scheduler launches ``fleet supervise-loop`` at BELOW_NORMAL and the class is
inherited by the supervisor and every short ``gh``/``git`` command it runs, so
any Normal-priority work on the host (interactive pytest-xdist) starves the
orchestrator's state transitions. Agent sessions pick an explicit class through
``popen_worker(priority=...)`` (#2122), so raising the control plane does not
raise agent work.

Windows only; POSIX has no equivalent inheritable class here, so it is a no-op.
Failure is non-fatal: it logs a warning and emits a ``control_plane_priority_failed``
event.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import psutil

from . import layout
from .instrumentation import log_event
from .supervise import supervisor_runtime_paths

logger = logging.getLogger(__name__)

EVENT_FAILED = "control_plane_priority_failed"


@dataclass(frozen=True)
class PriorityRaiseResult:
    """Outcome of a raise attempt; ``before``/``after`` are Windows priority class ints."""

    ok: bool
    before: int | None
    after: int | None
    error: str | None = None


def raise_to_normal_priority(state_path: Path | None = None) -> PriorityRaiseResult:
    """Set this process's priority class to NORMAL, logging before and after.

    Never raises. ``state_path`` locates the events.db for the warning event on
    failure; omit it to log only.
    """
    if os.name != "nt":
        return PriorityRaiseResult(ok=True, before=None, after=None)

    before: int | None = None
    try:
        proc = psutil.Process()
        before = int(proc.nice())
        proc.nice(psutil.NORMAL_PRIORITY_CLASS)
        after = int(proc.nice())
    except Exception as exc:  # noqa: BLE001 -- contract: never raise; main() dispatch must still start
        logger.warning("could not raise control-plane priority to NORMAL: %s", exc)
        if state_path is not None:
            log_event(
                state_path,
                # event-consumer: audit-only -- the warning row is the audit record for a
                # best-effort startup step; the ``logger.warning`` above is the alert
                EVENT_FAILED,
                {"before": before, "error": str(exc)},
                level="warning",
            )
        return PriorityRaiseResult(ok=False, before=before, after=None, error=str(exc))

    logger.info("control-plane priority class: %s -> %s", before, after)
    return PriorityRaiseResult(ok=True, before=before, after=after)


def raise_supervisor_to_normal() -> PriorityRaiseResult:
    """Startup hook for the ``fleet supervise`` / ``supervise-loop`` processes.

    Resolves the supervisor's ``state.json`` path itself so callers need only this call.
    """
    return raise_to_normal_priority(supervisor_runtime_paths(layout.DEFAULT_STATE_DIR).state_file)
