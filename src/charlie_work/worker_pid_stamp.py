"""Stamp a freshly launched worker's pid onto its issue entry.

``worker_pid``/``worker_process_start_time`` are the state-based liveness
signal (issue #207) that the dead-worker reap and the superseded-worker reap
consult. They describe ONE dispatch epoch: every successful launch must
replace them, and a launch that reports no pid must drop the previous
epoch's values rather than leave them standing.

A stale pair is not inert. The previous worker's pid reads as dead, so the
reap treats the NEW worker as dead on the next pass and parks or salvages its
branch mid-session. On the local lane that dispatched a reviewer against the
pre-rework head while the rework worker was still committing, and it also
blinded the superseded-worker reap, so a second writer could launch onto the
same branch (mdls #25/#7, 2026-09-29). All three launch sites (fresh
dispatch, remote rework, local rework) route through this one function, so a
new launch site cannot record the epoch half-way.
"""

from __future__ import annotations

from typing import Any


def stamp_worker_process(entry: dict[str, Any], result: Any | None) -> None:
    """Record ``result``'s pid on ``entry``, or clear a previous epoch's pid.

    ``result`` is the launch's ``SessionDispatchResult`` (or ``None`` when
    the caller could not find one). Call only for a launch that succeeded.
    """
    pid = getattr(result, "pid", None)
    if pid is None:
        entry.pop("worker_pid", None)
        entry.pop("worker_process_start_time", None)
        return
    entry["worker_pid"] = pid
    entry["worker_process_start_time"] = getattr(result, "process_start_time", None)
