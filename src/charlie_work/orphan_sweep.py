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
  #1842 tracks. The snapshot row type and the walk itself live in
  ``process_chain`` (shared with ``quiesce`` and ``host_load``, issue #2058);
  the private names here are alias bindings into it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from .process_chain import ancestor_chain_pids
from .process_chain import ProcRow as _ProcRow  # noqa: F401 (deliberate re-export)
from .process_chain import posix_process_ppid_snapshot as _posix_process_ppid_snapshot
from .process_chain import win32_process_ppid_snapshot as _win32_process_ppid_snapshot
from .subprocess_runner import run_captured

logger = logging.getLogger(__name__)


# Hard backstop on the parent-PID walk ``_self_ancestor_pids`` delegates to
# (``process_chain.ancestor_chain_pids``). A malformed or adversarial ppid
# snapshot (a cycle) must never spin the guard; real chains are a handful of
# hops.
_MAX_ANCESTOR_CHAIN_HOPS = 64


def _self_ancestor_pids() -> frozenset[int]:
    """Return ``os.getpid()`` plus every ancestor PID, best-effort.

    The kill primitives consult this so a caller can never terminate its own
    ancestry: on Windows ``taskkill /T /PID <ancestor>`` fells the ancestor's
    whole subtree, which contains the caller — the issue #1842 failure shape
    (a hosted-CI pytest controller terminated mid-run, no traceback, no
    junit). The sweep that feeds PIDs to ``kill_orphan_pid`` matches on a
    CommandLine substring, and a too-broad worktree needle legitimately
    matches that ancestry's command lines.

    The walk is ``process_chain.ancestor_chain_pids`` (issue #2058), whose
    termination rules are a parent absent from the snapshot, a cycle, the
    hop cap, and -- crucially -- a parent *created after its child*. Windows
    never reparents, so once an ancestor exits its PID is free to be
    recycled, and a child's stale ``ParentProcessId`` then names whatever
    unrelated process took the number. Trusting it put a freshly launched
    merge-gate runner on the "ancestor" list, and ``kill_process_tree``
    refused to kill it (the ``test_local_merge_gate_async``
    timeout/restart flake). When the snapshot cannot be taken at all the
    result degrades to ``{os.getpid()}`` — on Windows the only producer of
    orphan PIDs (``sweep_orphan_processes``) needs a working
    process-listing substrate too, so a broken snapshot mostly means there
    was no sweep output to act on either;
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
    return frozenset(ancestor_chain_pids(self_pid, ppid_by_pid, max_hops=_MAX_ANCESTOR_CHAIN_HOPS))


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
    # silently returns [] on every such host (a #1842 follow-up tracks the
    # fix). ``quiesce.list_processes`` deliberately omits ``-AsArray``
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
