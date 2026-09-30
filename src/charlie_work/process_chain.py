"""Shared process-ancestry machinery: ``(ppid, created)`` rows and the stale-link-safe walk.

One home for the ``pid -> ProcRow`` snapshot shape and the ancestor walk that
three consumers share (issue #2058):

* ``orphan_sweep._self_ancestor_pids`` -- the kill-guard ancestry set (issue
  #2057, where the row shape and the recycled-parent rule originated).
* ``quiesce.self_process_chain`` -- the quiescence gate's self/ancestor
  exclusion set.
* ``host_load._self_tree`` -- the load probe's own-suite exclusion.

Windows never reparents: a child's recorded ``ParentProcessId`` keeps naming
its parent after the parent exits, and the freed pid can then be recycled by
an unrelated, *younger* process. A walk that trusts the bare ppid follows the
recycled pid into a stranger -- pulling it into an exclusion set (quiesce,
host_load) or refusing to kill it (orphan_sweep). ``ancestor_chain_pids``
therefore ends the walk at a parent created after its child: a user-mode
parent always predates its child, so a younger "parent" is a recycled pid,
not an ancestor (the same check ``psutil.Process.parent()`` applies).

Accepted limits of the parent-older-than-child premise: (1) it holds for
user-mode processes only -- children of the kernel ``System`` process (pid 4)
can appear older than it, which only ever ends the walk early at a kernel
boundary no caller descends from; (2) a system clock stepped backwards
between the parent's and the child's creation can invert their timestamps,
which ``psutil.Process.parent()`` accepts as well.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NamedTuple, Protocol

import psutil

logger = logging.getLogger(__name__)

# Hard backstop on the ancestor walk. A cycle already terminates the walk on
# its own (the seen-set check in ``ancestor_chain_pids``), but this bounds the
# work done against a snapshot that is malformed or adversarial in some other
# way, so the walk can never spin. Real chains are a handful of hops.
MAX_CHAIN_HOPS = 4096


class ProcRow(NamedTuple):
    """One snapshot row: the parent pid and (where known) the creation stamp.

    ``created`` is a creation time comparable only between rows of the same
    snapshot (Windows: ``psutil`` ``create_time()``, epoch seconds derived from
    the kernel's UTC FILETIME). ``None`` means unknown, and an unknown stamp
    never disqualifies a parent link. Every row of one snapshot uses one unit.
    """

    ppid: int
    created: float | None = None


class _ProcessRowSource(Protocol):
    """Structural type ``proc_rows`` reads: pid, ppid, and a creation stamp."""

    pid: int
    ppid: int
    created: float | None


def proc_rows(processes: Iterable[_ProcessRowSource]) -> dict[int, ProcRow]:
    """Reduce a process snapshot to the ``pid -> ProcRow`` map a walk needs."""
    return {proc.pid: ProcRow(ppid=proc.ppid, created=proc.created) for proc in processes}


def win32_process_ppid_snapshot() -> dict[int, ProcRow]:
    """Snapshot ``pid -> (ppid, creation time)`` for every process via ``psutil``.

    The creation time exists because Windows never reparents: a child's
    parent PID keeps naming its parent after the parent exits, and that PID
    can then be recycled by an unrelated, *younger* process. A walk that
    trusts the bare ppid follows the recycled PID into a stranger (see
    ``ancestor_chain_pids``).

    ``create_time()`` is taken from the kernel's process-creation FILETIME,
    which is UTC by construction. The earlier CIM ``CreationDate`` source is a
    ``DateTime`` built from local-time fields and can be off by an hour around
    a DST transition -- enough to make a real parent look *newer* than its
    child, stop the ancestor walk early, and unprotect a real ancestor (the
    dangerous direction). ``psutil`` is a declared dependency; one
    ``process_iter`` pass yields pid, ppid and create_time from the same
    source, so the rows are mutually consistent, with no PowerShell spawn, no
    JSON round-trip, and no 10s timeout to stall the kill path
    (``kill_process_tree`` / ``kill_orphan_pid``). A process whose creation
    time is unreadable (``AccessDenied`` -> ``None``) simply keeps its link.

    Returns ``{}`` on any failure (``psutil`` error, OS error). The caller
    degrades to the bare self-pid guard rather than disabling process reaping
    on a host whose process-listing substrate is broken -- see
    ``orphan_sweep._self_ancestor_pids``.
    """
    ppid_by_pid: dict[int, ProcRow] = {}
    try:
        for proc in psutil.process_iter(["pid", "ppid", "create_time"]):
            info = proc.info
            try:
                pid = int(info["pid"])
                ppid = int(info.get("ppid") or 0)
            except (KeyError, TypeError, ValueError):
                continue
            created = info.get("create_time")
            ppid_by_pid[pid] = ProcRow(
                ppid, float(created) if isinstance(created, (int, float)) else None
            )
    except (psutil.Error, OSError):
        logger.warning("psutil process snapshot failed", exc_info=True)
        return {}
    return ppid_by_pid


def posix_process_ppid_snapshot(proc_root: Path = Path("/proc")) -> dict[int, ProcRow]:
    """Snapshot ``pid -> ppid`` from procfs (``/proc/<pid>/stat``).

    ``proc_root`` is a parameter -- not a constant read at call time -- so
    tests can point it at a fabricated procfs tree without touching the
    host's (same convention as ``host_load._list_processes_posix``).
    Processes that exit or become unreadable mid-scan are skipped: a
    snapshot can never be perfectly atomic, and a vanished entry is not
    worth failing the walk over. Returns ``{}`` where procfs is absent.

    The creation stamp is left unknown: POSIX reparents orphans to init, so a
    recorded ppid always names a live process and cannot go stale.
    """
    ppid_by_pid: dict[int, ProcRow] = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return ppid_by_pid
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            # ppid is the first field after ``state`` once comm (which can
            # contain spaces/parens) is split off on the LAST ')'.
            fields = stat_text.rpartition(")")[2].split()
            ppid_by_pid[int(entry.name)] = ProcRow(int(fields[1]))
        except (OSError, ValueError, IndexError):
            continue
    return ppid_by_pid


def ancestor_chain_pids(
    pid: int,
    rows: Mapping[int, ProcRow],
    *,
    max_hops: int = MAX_CHAIN_HOPS,
) -> tuple[int, ...]:
    """Ordered ancestor chain ``pid -> ... -> top`` over a ``ProcRow`` map.

    ``pid`` itself is always the first element, even when it has no row in
    ``rows``. A recorded parent absent from ``rows`` is appended as the final
    element before the walk ends (it can never be matched or killed by a
    caller working over the same snapshot, but naming it keeps the chain
    honest -- the same boundary ``quiesce.self_process_chain`` has always
    reported).

    Termination rules, each ending the walk:
      - ``current`` has no row in ``rows`` (root of the snapshot).
      - The recorded parent is ``<= 0`` or already on the chain (a cycle).
      - The parent was *created after its child*: Windows never reparents, so
        a stale ``ParentProcessId`` naming a recycled, unrelated process is
        not an ancestor. A missing ``created`` stamp on either side keeps the
        link -- unknown never disqualifies.
      - ``max_hops`` iterations: the hard backstop against a snapshot that is
        malformed in a way the seen-set cannot catch.
    """
    chain = [pid]
    seen = {pid}
    current = pid
    for _ in range(max_hops):
        row = rows.get(current)
        if row is None:
            break
        parent = row.ppid
        if parent <= 0 or parent in seen:
            break
        parent_row = rows.get(parent)
        if (
            parent_row is not None
            and row.created is not None
            and parent_row.created is not None
            and parent_row.created > row.created
        ):
            # Recycled pid: the recorded parent exited and an unrelated,
            # younger process now holds its number. Not an ancestor.
            break
        chain.append(parent)
        seen.add(parent)
        current = parent
    return tuple(chain)
