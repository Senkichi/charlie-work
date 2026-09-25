"""Orphan-process sweep machinery and the caller-ancestry kill guard.

Extracted from ``process_utils.py`` (issue #1842 rework): that module is a
tracked file under the issue-#1442 file-size ratchet (800-line cap), and the
#1842 guard additions would have pushed it over its cap. The seam is a real
domain boundary, not a line-count convenience — everything here serves the
orphan-sweep pipeline:

* ``sweep_orphan_processes`` finds processes whose CommandLine references a
  dead session's worktree, and the ``_MIN_SWEEP_WORKTREE_PATH_LENGTH`` /
  ``_SWEEP_PATH_FORBIDDEN_CHARS`` validation keeps a degenerate needle from
  turning its ``-like "*<path>*"`` filter into a match-everything query.
* ``_win32_process_ppid_snapshot`` / ``_posix_process_ppid_snapshot`` /
  ``_self_ancestor_pids`` answer "who is the caller's ancestry" — the set the
  kill primitives (``process_utils.kill_process_tree`` /
  ``process_utils.kill_orphan_pid``) refuse to terminate, because a sweep-hit
  PID naming any ancestor (uv, the pytest controller, the step's pwsh) fells
  the subtree containing the caller — the mid-run controller death issue
  #1842 tracks.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from .subprocess_runner import run_captured

logger = logging.getLogger(__name__)


# Hard backstop on the parent-PID walk in ``_self_ancestor_pids`` — the same
# bound ``quiesce.self_process_chain`` carries as ``_MAX_CHAIN_DEPTH``. A
# malformed or adversarial ppid snapshot (a cycle) must never spin the guard;
# real chains are a handful of hops.
_MAX_ANCESTOR_CHAIN_HOPS = 64


def _win32_process_ppid_snapshot() -> dict[int, int]:
    """Snapshot ``pid -> ppid`` for every process via one CIM query.

    Leaner than ``quiesce.list_processes``: it selects only
    ``ProcessId``/``ParentProcessId``, makes a single attempt, and uses a
    10s timeout — this runs on the kill path inside ``kill_process_tree`` /
    ``kill_orphan_pid``, where quiesce's 60s x 2 retry budget (sized for a
    once-per-invocation operator gate) could stall an orphan sweep for
    minutes under load.

    Routed through ``subprocess_runner.run_captured`` (the codebase's
    never-raises runner) rather than raw ``subprocess.run``: a spawn failure
    such as ``PermissionError`` from a denied ``CreateProcess`` is an
    ``OSError``, not a ``SubprocessError``, and would have slipped the old
    narrow ``except`` to propagate through the ancestor guard — aborting
    ``dead_worker_reap._sweep_orphan_processes_for_dead_sessions`` outright.

    Returns ``{}`` on any failure (no PowerShell, timeout, non-zero exit,
    spawn error, unparseable output). The caller degrades to the bare
    self-pid guard rather than disabling process reaping on a host whose
    process-listing substrate is broken — see ``_self_ancestor_pids``.
    """
    ppid_by_pid: dict[int, int] = {}
    if not shutil.which("powershell"):
        return ppid_by_pid
    result = run_captured(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId, ParentProcessId | ConvertTo-Json",
        ],
        cwd=Path.cwd(),
        timeout_seconds=10,
    )
    if result.returncode != 0:
        return ppid_by_pid
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ppid_by_pid

    # No ``-AsArray``: that switch is PowerShell 6+ and Windows PowerShell
    # 5.1 fails the whole command on it (see ``quiesce.list_processes``).
    # ``ConvertTo-Json`` therefore emits a bare object when exactly one
    # process matches — normalize both shapes.
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return ppid_by_pid

    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            pid = int(entry["ProcessId"])
            ppid = int(entry.get("ParentProcessId") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        ppid_by_pid[pid] = ppid
    return ppid_by_pid


def _posix_process_ppid_snapshot(proc_root: Path = Path("/proc")) -> dict[int, int]:
    """Snapshot ``pid -> ppid`` from procfs (``/proc/<pid>/stat``).

    ``proc_root`` is a parameter — not a constant read at call time — so
    tests can point it at a fabricated procfs tree without touching the
    host's (same convention as ``host_load._list_processes_posix``).
    Processes that exit or become unreadable mid-scan are skipped: a
    snapshot can never be perfectly atomic, and a vanished entry is not
    worth failing the walk over. Returns ``{}`` where procfs is absent.
    """
    ppid_by_pid: dict[int, int] = {}
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
            ppid_by_pid[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    return ppid_by_pid


def _self_ancestor_pids() -> frozenset[int]:
    """Return ``os.getpid()`` plus every ancestor PID, best-effort.

    The kill primitives consult this so a caller can never terminate its own
    ancestry: on Windows ``taskkill /T /PID <ancestor>`` fells the ancestor's
    whole subtree, which contains the caller — the issue #1842 failure shape
    (a hosted-CI pytest controller terminated mid-run, no traceback, no
    junit). The sweep that feeds PIDs to ``kill_orphan_pid`` matches on a
    CommandLine substring, and a too-broad worktree needle legitimately
    matches that ancestry's command lines.

    The walk mirrors ``quiesce.self_process_chain``'s termination rules: a
    parent absent from the snapshot, a cycle, or the hop cap ends it. When
    the snapshot cannot be taken at all the result degrades to
    ``{os.getpid()}`` — on Windows the only producer of orphan PIDs
    (``sweep_orphan_processes``) needs the same CIM substrate, so a broken
    snapshot mostly means there was no sweep output to act on either;
    collapsing to the pre-#1842 self-pid guard keeps direct
    ``kill_process_tree`` callers (which carry start-time fingerprints)
    working on such a host instead of silently disabling reaping.

    Never raises: the snapshot helpers already collapse their own failures
    to ``{}``; the ``except Exception`` here is the second line of defense
    (e.g. ``Path.cwd()`` raising because the caller's cwd was deleted
    mid-sweep) so nothing on this path can propagate into the kill
    primitives. Both the raise and the empty-snapshot degrade are logged —
    whether the #1842 guard ever fires is the only telemetry that can
    confirm or refute the sweep-needle hypothesis, so a silent degrade would
    erase the evidence.
    """
    try:
        if os.name == "nt":
            ppid_by_pid = _win32_process_ppid_snapshot()
        else:
            ppid_by_pid = _posix_process_ppid_snapshot()
    except Exception:
        logger.warning(
            "process ppid snapshot raised; ancestor kill guard degrades to self-pid only",
            exc_info=True,
        )
        ppid_by_pid = {}
    self_pid = os.getpid()
    if not ppid_by_pid:
        logger.warning(
            "process ppid snapshot unavailable or empty; ancestor kill guard "
            "degrades to self-pid %d only (os.name=%r)",
            self_pid,
            os.name,
        )
    chain: set[int] = {self_pid}
    current = self_pid
    for _ in range(_MAX_ANCESTOR_CHAIN_HOPS):
        parent = ppid_by_pid.get(current)
        if parent is None or parent <= 0 or parent in chain:
            break
        chain.add(parent)
        current = parent
    return frozenset(chain)


# Minimum length a ``worktree_path`` needle must reach before
# ``sweep_orphan_processes`` embeds it in the ``-like "*<path>*"`` filter —
# long enough to actually name a directory (``C:\a\b``-class), far below every
# real worktree path this system hands it (``.var/charlie-work/worktrees/...``).
# Shorter values can only be drive roots, bare drive letters, or fragments —
# each of which as a CommandLine substring matches far more than one worktree's
# orphans.
_MIN_SWEEP_WORKTREE_PATH_LENGTH = 8

# Characters that can never appear in a real Windows worktree path but corrupt
# the ``-like`` needle: ``*``, ``?``, ``[]`` are -like wildcard metacharacters
# (a ``*`` needle matches every CommandLine on the host), and ``"`` breaks out
# of the double-quoted pattern inside the -Command string.
_SWEEP_PATH_FORBIDDEN_CHARS = '*?[]"'


def sweep_orphan_processes(worktree_path: str) -> list[dict[str, Any]]:
    """Sweep for orphan processes whose CommandLine references a worktree path.

    On Windows: Uses PowerShell Get-CimInstance Win32_Process to find processes
    whose CommandLine contains the worktree path. This catches detached/daemonized
    processes that survived a process tree kill (e.g., nohup-style background processes).

    On POSIX: Not implemented (returns empty list). POSIX process groups handle
    detachment better via killpg, and /proc enumeration is more complex.

    This is a read-only detection function. Callers should decide whether to kill
    the returned processes based on policy (e.g., janitor warnings vs. automatic cleanup).

    Args:
        worktree_path: The worktree path to search for in process CommandLines.
            Degenerate values — empty, whitespace-only, below
            ``_MIN_SWEEP_WORKTREE_PATH_LENGTH``, containing no path separator,
            or containing ``-like`` wildcard metacharacters — are refused
            (empty result, no query): as a ``-like "*<path>*"`` needle they
            match far more than one worktree's processes, including the
            pytest/uv/pwsh ancestry running the sweep itself (issue #1842).

    Returns:
        A list of dicts describing processes whose CommandLine references the
        worktree path. Each dict contains ``pid`` (int), ``name`` (str), and
        ``command_line`` (str). POSIX callers always get an empty list.
    """
    orphans: list[dict[str, Any]] = []

    # Issue #1842: validate the needle *before* it reaches the -like filter.
    # "" or whitespace produces ``-like "**"``-equivalent matching that returns
    # every process on the host — including this process and its ancestors —
    # and ``kill_orphan_pid`` downstream would then taskkill our own tree.
    # Checked first so the refusal is identical on every platform.
    needle = worktree_path.strip()
    if (
        len(needle) < _MIN_SWEEP_WORKTREE_PATH_LENGTH
        or ("/" not in needle and "\\" not in needle)
        or any(ch in needle for ch in _SWEEP_PATH_FORBIDDEN_CHARS)
    ):
        return orphans

    if os.name != "nt":
        # POSIX: not implemented - process groups handle detachment better
        return orphans

    if not shutil.which("powershell"):
        return orphans

    # Use PowerShell to query Win32_Process for CommandLine matching the worktree path.
    # Select-Object + ConvertTo-Json preserves PID, image name, and command line so
    # callers can log what was killed and identify respawn sources.
    #
    # ``run_captured`` never raises, so a spawn-level ``OSError`` (e.g.
    # ``PermissionError`` from a denied ``CreateProcess``) cannot propagate
    # out of here and abort the dead-session sweep lane mid-loop.
    #
    # Known gap: ``-AsArray`` is a PowerShell 6+ switch — on Windows
    # PowerShell 5.1 the whole command fails (non-zero exit) and this sweep
    # silently returns []. Tracked in a follow-up issue filed from the #1842
    # rework; ``_win32_process_ppid_snapshot`` deliberately omits ``-AsArray``
    # for exactly this reason.
    result = run_captured(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f'Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like "*{needle}*" }} | Select-Object ProcessId, CommandLine, Name | ConvertTo-Json -AsArray',
        ],
        cwd=Path.cwd(),
        timeout_seconds=10,
    )
    if result.returncode != 0:
        return orphans

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return orphans

    if not isinstance(data, list):
        return orphans

    for proc in data:
        if not isinstance(proc, dict):
            continue
        try:
            pid = int(proc["ProcessId"])
        except (KeyError, ValueError, TypeError):
            continue
        if pid <= 0:
            continue
        orphans.append(
            {
                "pid": pid,
                "name": str(proc.get("Name") or ""),
                "command_line": str(proc.get("CommandLine") or ""),
            }
        )

    return orphans
