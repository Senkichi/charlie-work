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
import signal
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import host as _host
from .process_chain import ancestor_chain_pids
from .process_chain import ProcRow as _ProcRow  # noqa: F401 (deliberate re-export)
from .process_chain import posix_process_ppid_snapshot as _posix_process_ppid_snapshot
from .process_chain import win32_process_ppid_snapshot as _win32_process_ppid_snapshot
from .subprocess_runner import command_failure_message, run_captured

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


def _enumerate_fingerprinted_children(pid: int) -> dict[int, float | None]:
    """Enumerate ``pid``'s children, pairing each with its process start time.

    The fingerprint must be taken at enumeration time — *before* the platform
    tree kill runs — because ``_reap_enumerated_children`` uses it later to
    tell "the process enumeration returned" from a stranger that recycled the
    pid during the kill window (issue #2059). ``None`` means the start time
    was unreadable (a protected process, or the pid exited between
    enumeration and the query); such a pid can never be pinned and must never
    be individually killed.

    The local import avoids a cycle: ``process_utils`` imports this module's
    guard machinery at module scope. Resolving ``_enumerate_child_pids`` /
    ``get_process_start_time`` through the module at call time preserves the
    existing test seam (``monkeypatch.setattr(process_utils, ...)``).
    """
    from .process_utils import _enumerate_child_pids, get_process_start_time

    return {child: get_process_start_time(child) for child in _enumerate_child_pids(pid)}


# Bound on confirming a survivor of the tree kill actually dies after its
# individual kill. TerminateProcess/SIGKILL teardown is prompt; the bound only
# absorbs propagation latency under load, and a still-live result after it is
# reported as a survivor, not retried forever.
_CHILD_KILL_CONFIRM_SECONDS = 1.0
_CHILD_KILL_CONFIRM_POLL_SECONDS = 0.05


def _reap_enumerated_children(
    root_pid: int,
    child_starts: Mapping[int, float | None],
    exempt_pids: frozenset[int],
    expected_root_start_time: float | None,
) -> list[int]:
    """Verify each enumerated child died with its root; reap survivors directly.

    Called by ``kill_process_tree`` after the platform tree kill, whose report
    is optimistic: ``taskkill /T /PID <already-dead root>`` exits 128 ("not
    found") and kills *nothing*, and ``/T`` silently skips a member it cannot
    terminate — yet the children used to be recorded as killed unconditionally
    (issue #2059: the "child still alive after kill_process_tree" flake is a
    real orphan the return value had claimed dead). On POSIX, ``killpg``
    similarly cannot reach an enumerated child that left the group.

    Each enumerated pid is classified with its enumeration-time fingerprint:

    * already dead — or recycled onto another process, which the pinned
      ``is_pid_alive`` check reads as dead — is recorded.
    * alive with a verified matching fingerprint is killed individually
      (``taskkill /T /F`` on Windows so its own subtree goes with it;
      ``SIGKILL`` on POSIX), then re-verified before it is recorded.
    * alive but unverifiable (no fingerprint captured, or the start time can
      no longer be read) is *never* killed by bare pid — an unpinned kill can
      hit a stranger holding a recycled pid — and is logged instead.
    * (Windows only) created before its alleged parent
      (``start <= root_start_time``) is a stale-``ParentProcessId`` artifact
      of pid recycling, not a child: Windows never reparents, so a real child
      always postdates its parent — the #2057 ancestor-walk rule mirrored
      onto descendants. Refused and logged. ``taskkill /T``'s own ppid
      matching may still have hit it; that is outside this function's
      control, but this path never adds an individual kill for it. The check
      is Windows-only because its premise is: on POSIX the kernel reparents
      orphans, so a ppid read at enumeration is never stale, and /proc start
      times are ~10 ms-tick-quantized on an estimated boot base — jitter can
      place a real child's read a few ms before its root's, and refusing it
      would strand the very survivor this function exists to reap.
    * listed in ``exempt_pids`` (the caller's self/ancestor guard, #1842) is
      refused identically: a stale ppid can name an ancestor, and killing it
      would fell this process's own subtree.

    Returns the subset of enumerated pids verified dead. Never raises: this
    runs on the kill path inside ``kill_process_tree``, whose contract is
    best-effort.
    """
    # Routed through ``process_utils``'s own names at call time so a patched
    # ``charlie_work.process_utils.run_captured`` covers the individual kill
    # too — same seam the tree kill uses. Liveness goes through the host
    # probe, which late-binds ``process_utils.is_pid_alive``, so a patch of
    # the primitive still reaches it.
    from .process_utils import get_process_start_time, run_captured

    probe = _host.current().probe

    root_start = expected_root_start_time
    if root_start is None:
        # Post-mortem reads still work while a handle keeps the process object
        # alive; a fully-reaped root yields None and the ordering check below
        # is skipped (the fingerprint pin still applies).
        root_start = get_process_start_time(root_pid)

    confirmed_dead: list[int] = []
    for child, start in child_starts.items():
        if child in exempt_pids:
            logger.warning(
                "kill_process_tree: enumerated 'child' pid %d of %d is this process or "
                "a caller ancestor; refusing to touch it (stale ParentProcessId or "
                "bogus enumeration)",
                child,
                root_pid,
            )
            continue
        if start is None:
            if not probe.is_alive(child):
                confirmed_dead.append(child)
            else:
                logger.warning(
                    "kill_process_tree: child %d of %d survived the tree kill but has "
                    "no start-time fingerprint; refusing an identity-unpinned kill",
                    child,
                    root_pid,
                )
            continue
        if not probe.is_alive(child, start):
            confirmed_dead.append(child)
            continue
        if os.name == "nt" and root_start is not None and start <= root_start:
            logger.warning(
                "kill_process_tree: enumerated 'child' %d of %d was created before its "
                "parent (start %.3f <= %.3f); stale ParentProcessId on a recycled pid, "
                "not a child -- refusing individual kill",
                child,
                root_pid,
                start,
                root_start,
            )
            continue
        current = get_process_start_time(child)
        if current is None or abs(current - start) > 1.0:
            logger.warning(
                "kill_process_tree: child %d of %d reads alive but its start time is no "
                "longer verifiable; refusing an identity-unpinned kill",
                child,
                root_pid,
            )
            continue
        # The direct kill's own failure detail rides into the survivor warning
        # below -- a taskkill/os.kill that never landed is the difference
        # between "kill was resisted" and "kill never happened", and only the
        # latter is actionable from a bare "still alive" line.
        kill_detail = ""
        try:
            if os.name == "nt":
                kill_command = ["taskkill", "/T", "/F", "/PID", str(child)]
                kill_result = run_captured(
                    kill_command,
                    cwd=Path.cwd(),
                    timeout_seconds=10,
                )
                if not kill_result.ok:
                    kill_detail = "; direct kill failed: " + command_failure_message(
                        kill_command, kill_result, "no diagnostic output"
                    )
            else:
                os.kill(child, signal.SIGKILL)
        except Exception as exc:
            kill_detail = f"; direct kill raised {exc!r}"
        deadline = time.monotonic() + _CHILD_KILL_CONFIRM_SECONDS
        while time.monotonic() < deadline:
            if not probe.is_alive(child, start):
                confirmed_dead.append(child)
                break
            time.sleep(_CHILD_KILL_CONFIRM_POLL_SECONDS)
        else:
            logger.warning(
                "kill_process_tree: child %d of %d survived the tree kill and a direct "
                "kill -- still alive%s",
                child,
                root_pid,
                kill_detail,
            )
    return confirmed_dead
