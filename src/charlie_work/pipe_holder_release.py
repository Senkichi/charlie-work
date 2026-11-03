"""Release a worker's tool call that is blocked on an orphaned pipe holder (issue #2794).

An ``opencode run`` worker runs each ``bash`` tool call with its stdout/stderr on
a pipe it owns, and does not complete the call until that pipe reaches EOF. A
command that leaves a background process behind (``server &``, an orphaned
test-runner worker) hands that process the pipe's write end; when the command's
shell exits, the process outlives it. The tool's own timeout kills only the
shell -- it does not release a call whose pipe is still held -- so the call, and
with it the whole ``opencode run`` session, blocks silently: no event, no
provider error, a frozen worktree mtime, a live pid.

Reproduced against opencode 1.18.35 with a fake provider: ``sleep 300 &`` in a
tool call that carried ``timeout: 15000`` left the session silent for the full
300s; the same call with the child's stdio redirected returned in 0.2s, and a
foreground ``sleep 40`` under the same timeout returned a ``terminated command
after exceeding timeout`` result at +40s -- when the orphaned ``sleep`` exited and
closed the pipe, not when the timeout fired.

The holder cannot be found by ancestry. The shell exits right after backgrounding
its child, and Windows never reparents, so the holder's ppid names a dead process
and it is not a descendant of the worker in any process table. What every holder
shares is its working directory: it inherited the worktree as its cwd from the
tool call. So the holders are the processes that

* run with a cwd inside the worker's own worktree (exclusive to this worker),
* started after the worker did (an operator shell opened earlier is spared), and
* are not the worker, one of its live descendants (its own LSP/MCP helpers -- a
  holder is an orphan by construction, so a live parent chain means legitimate),
  or this process or one of its ancestors.

Killing them closes the pipe; the blocked call completes with its normal timeout
result and the model carries on, exiting cleanly at the end of its turn. A holder
that moved to a cwd outside the worktree is not found; the stalled lane then
falls through to its ordinary reap.

Containment is relative to its anchor, so the anchor is asserted first: the
worker's *own* cwd must be inside the worktree, and the worker's identity is
pinned by start time (a recycled pid is refused). Without that, a wrong or empty
``worktree_path`` (``Path("")`` is ``Path(".")``, this process's cwd) would select
unrelated processes; a relative path is refused outright for the same reason.
"""

from __future__ import annotations

import logging
from pathlib import Path

import psutil

from . import safe_path
from .process_utils import _ancestor_guard_exempt_pids, get_process_start_time

logger = logging.getLogger(__name__)

# Start times come from two sources (``get_process_start_time`` and psutil); the
# same 1s tolerance ``kill_process_tree`` allows between sidecar and live reads.
_START_TIME_TOLERANCE_SECONDS = 1.0
# TerminateProcess/SIGKILL teardown is prompt; the bound only absorbs
# propagation latency, and a still-live result is reported, not retried.
_CONFIRM_SECONDS = 2.0


def _cwd_inside(worktree: Path, proc: psutil.Process) -> bool:
    """True when ``proc``'s cwd is inside ``worktree``; unreadable is False."""
    try:
        cwd = proc.cwd()
    except (psutil.Error, OSError):
        return False
    return bool(cwd) and safe_path.contains(worktree, Path(cwd))


def _worker_anchored(worktree: Path, worker_pid: int, worker_start_time: float) -> bool:
    """The worker is the process the sidecar recorded, and runs inside ``worktree``."""
    current = get_process_start_time(worker_pid)
    if current is None or abs(current - worker_start_time) > _START_TIME_TOLERANCE_SECONDS:
        return False  # gone, or the pid was recycled onto another process
    try:
        return _cwd_inside(worktree, psutil.Process(worker_pid))
    except psutil.Error:
        return False


def _live_descendants(worker_pid: int) -> frozenset[int]:
    """The worker's live descendants by parent chain; empty when unreadable."""
    try:
        return frozenset(c.pid for c in psutil.Process(worker_pid).children(recursive=True))
    except psutil.Error:
        return frozenset()


def _holders(
    worktree: Path, worker_pid: int, worker_start_time: float, exempt: frozenset[int]
) -> list[psutil.Process]:
    spared = exempt | {worker_pid} | _live_descendants(worker_pid)
    holders: list[psutil.Process] = []
    for proc in psutil.process_iter():
        if proc.pid in spared:
            continue
        try:
            started_before_worker = (
                proc.create_time() < worker_start_time - _START_TIME_TOLERANCE_SECONDS
            )
        except (psutil.Error, OSError):
            continue
        if started_before_worker or not _cwd_inside(worktree, proc):
            continue
        holders.append(proc)
    return holders


def release_pipe_holders(
    worktree_path: Path, worker_pid: int | None, worker_start_time: float | None
) -> tuple[int, ...]:
    """Kill the processes a worker's tool call may be blocked on; return those verified gone.

    Spares the worker itself. Returns ``()`` -- touching nothing -- when the
    worker cannot be anchored (no pid or start time, gone, recycled, or its cwd
    is not inside ``worktree_path``) or when no holder exists. Never raises.
    """
    if not worker_pid or worker_pid <= 0 or worker_start_time is None:
        return ()
    try:
        # ``Path("")`` is ``Path(".")``: only an absolute path names a worktree, so
        # an unset ``WorkerView.worktree_path`` cannot anchor on this process's cwd.
        if not worktree_path.is_absolute() or not worktree_path.is_dir():
            return ()
        if not _worker_anchored(worktree_path, worker_pid, worker_start_time):
            return ()
        victims = _holders(
            worktree_path, worker_pid, worker_start_time, _ancestor_guard_exempt_pids()
        )
        for proc in victims:
            try:
                proc.kill()  # psutil pins the signal to the (pid, create_time) it enumerated
            except psutil.NoSuchProcess:
                continue
            except (psutil.Error, OSError):
                logger.warning(
                    "pipe-holder release: could not kill pid %d (issue #2794)",
                    proc.pid,
                    exc_info=True,
                )
        gone, alive = psutil.wait_procs(victims, timeout=_CONFIRM_SECONDS)
        for proc in alive:
            logger.warning("pipe-holder release: pid %d survived the kill", proc.pid)
        return tuple(sorted(proc.pid for proc in gone))
    except Exception:  # best-effort kill path: never raise into the sweep
        logger.warning("pipe-holder release raised; nothing reported", exc_info=True)
        return ()
