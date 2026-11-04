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
* are true orphans: the recorded parent is missing, or its pid was recycled onto a
  process created after the child. A process with a live parent -- the worker's
  own LSP/MCP helpers, an editor or shell an operator opened in the worktree
  after the dispatch, a build tool another process started there -- is
  legitimate by construction and is spared, as are this process and its
  ancestors. (A chain is only as orphaned as its first link: a live parent means
  the chain passes through a live process, so only the direct parent is read.)

Killing them closes the pipe; the blocked call completes with its normal timeout
result and the model carries on, exiting cleanly at the end of its turn. A holder
that moved to a cwd outside the worktree, or a POSIX holder the kernel reparented
to init (a live parent, so never selected), is not found; the stalled lane then
falls through to its ordinary reap.

Residual risk, stated plainly: the cwd rule is *new* kill reach.
``orphan_sweep.sweep_orphan_processes`` matches the worktree path in a process's
CommandLine, not its cwd, so it does not select a process merely for sitting in
the worktree. The orphan test narrows the new rule to leftovers, but an
orphaned process that started after the worker with its cwd in the worktree is
killed whatever it was for -- including a helper the worker itself detached from
its launcher. That is bounded: it runs once per dispatch, only for a worker
already STALLED past ``stall_minutes`` with no rate-limit explanation, and a
worker that does not resume within ``RELEASE_RESUME_GRACE`` takes the ordinary
reap it would have taken anyway.

Containment is relative to its anchor, so the anchor is asserted first: the
worker's *own* cwd must be inside the worktree, and the worker's identity is
pinned by start time (a recycled pid is refused). Without that, a wrong or empty
``worktree_path`` (``Path("")`` is ``Path(".")``, this process's cwd) would select
unrelated processes; a relative path is refused outright for the same reason.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
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


def _is_orphan(proc: psutil.Process) -> bool:
    """True only when ``proc``'s parent is provably gone; anything unreadable is False.

    The parent is gone when its pid names no process, or names one created after
    ``proc`` -- a user-mode parent always predates its child, so a younger
    "parent" is a recycled pid, not an ancestor (the rule ``psutil.Process.parent()``
    and ``process_chain.ancestor_chain_pids`` apply). Windows never reparents, so
    an orphan keeps its dead parent's number. Fail-safe: a process whose parent
    cannot be read is spared, never killed.
    """
    try:
        ppid = proc.ppid()
        if ppid <= 0:
            return False  # no recorded parent is not the same as a dead one
        created = proc.create_time()
        try:
            parent_created = psutil.Process(ppid).create_time()
        except psutil.NoSuchProcess:
            return True
        return parent_created > created
    except (psutil.Error, OSError):
        return False


def _fresh_processes() -> Iterator[psutil.Process]:
    """Every live process as a ``Process`` built now -- never one ``process_iter`` cached.

    ``process_iter`` keeps one ``Process`` per pid in a module-global cache and evicts
    a pid only when a sweep finds it absent. A pid freed and reborn between two
    sweeps (Windows recycles them freely) is served as the old instance, whose start
    time was snapshotted at construction. Read through it, a holder that started
    after the worker carries its predecessor's older start time, is judged to have
    predated the worker, and is spared: the blocked tool call stays blocked. A
    ``Process`` constructed here takes its identity from the process that holds the
    pid *now*, and ``kill`` is still pinned to that identity.

    The cache is seeded by any direct ``process_iter`` caller in the same process.
    Before this function existed that included ``_holders`` itself, so every earlier
    ``release_pipe_holders`` call left the next one a cache to read through (issue
    #3028). ``process_chain`` is not a seeder on Windows: it reaches ``process_iter``
    only in fallback branches, taken when psutil lacks ``ppid_map`` or
    ``cext.proc_times``, and Windows ships both. The cache is therefore outside this
    module's control, which is why selection avoids it rather than relying on nobody
    having populated it.

    A pid that exits between ``psutil.pids()`` and the ``Process`` constructor
    (``NoSuchProcess``) or whose identity cannot be read (``AccessDenied``, ``OSError``)
    is skipped, not raised: it cannot be a holder this sweep, and an exception escaping
    this generator would be swallowed by ``release_pipe_holders`` and turn the whole
    release into a silent no-op.
    """
    for pid in psutil.pids():
        try:
            yield psutil.Process(pid)
        except (psutil.Error, OSError):
            continue  # gone before it could be read, or unreadable: not a holder


def _holders(
    worktree: Path, worker_pid: int, worker_start_time: float, exempt: frozenset[int]
) -> list[psutil.Process]:
    spared = exempt | {worker_pid}
    holders: list[psutil.Process] = []
    for proc in _fresh_processes():
        if proc.pid in spared:
            continue
        try:
            started_before_worker = (
                proc.create_time() < worker_start_time - _START_TIME_TOLERANCE_SECONDS
            )
        except (psutil.Error, OSError):
            continue
        if started_before_worker or not _cwd_inside(worktree, proc) or not _is_orphan(proc):
            continue
        holders.append(proc)
    return holders


def release_pipe_holders(
    worktree_path: Path, worker_pid: int | None, worker_start_time: float | None
) -> tuple[int, ...]:
    """Kill the processes a worker's tool call may be blocked on; return those verified gone.

    Spares the worker itself and every process with a live parent. Returns ``()``
    -- touching nothing -- when the worker cannot be anchored (no pid or start time, gone, recycled, or its cwd
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
