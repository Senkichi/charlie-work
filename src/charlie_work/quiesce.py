"""Testable quiescence predicate: prove no fleet process is still running.

Before migrating a repo's orchestrator state directory, an operator must
prove the fleet (supervisor, workers, reviewers) is fully stopped. That gate
used to be PowerShell prose in a plan doc and had three separate defects that
made it either impossible to pass or falsely green:

1. **Self-match.** A pattern like ``"fleet supervise"`` matches the checking
   shell's own command line (the pattern text is literally inside the command
   that is running the check), and — worse — can match any ancestor shell in
   its parent chain too. A kill loop built on this force-killed the operator's
   own shell mid-run. Fixed by ``self_process_chain`` + the ``self_pid``
   exclusion in ``assert_quiescent``.
2. **Name-based matching hits unrelated applications.** This host runs the
   Windsurf IDE as ``Devin.exe`` (~20 live processes: renderer, gpu-process,
   crashpad-handler, ``devin.exe acp`` language servers, ...). Matching on
   process *name* can never pass. The fleet reviewer is a distinct
   *invocation* (``devin --model ... -p --prompt-file ...``), so matching is
   command-line-only, never name-based — see ``assert_quiescent``.
3. **One logical process, several OS processes.** ``fleet supervise`` is a
   lineage of multiple processes (e.g. a launcher, plus one or more
   interpreters), sometimes under a trampoline parent. A gate that reports
   only one PID lets a caller stop part of the lineage and leave the rest
   running. ``assert_quiescent`` reports every process whose command line
   matches, not just the first hit.

This module is intentionally free of fleet-specific knowledge: ``patterns``
are supplied by the caller (a future CLI layer would source them from
config), never hardcoded here (project rule: no embedded manual lists).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .subprocess_runner import no_console_window_kwargs

# Hard backstop on the parent-PID walk in `self_process_chain`. A cycle in
# the chain already terminates the loop on its own (see that function's
# docstring), but this bounds the work done against a snapshot that is
# malformed or adversarial in some other way, so the walk can never spin.
_MAX_CHAIN_DEPTH = 4096

# Timeout for the PowerShell process-snapshot query. Generous relative to
# `process_utils.sweep_orphan_processes`'s 10s because this queries every
# process on the host, not a CommandLine-filtered subset.
_LIST_PROCESSES_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class ProcessInfo:
    """One OS process, as reported by a `ProcessLister`.

    ``name`` is carried for display/logging only. Quiescence matching in
    `assert_quiescent` uses ``command_line`` exclusively — see defect #2 in
    the module docstring for why name-based matching is unsafe.
    """

    pid: int
    ppid: int
    name: str
    command_line: str


@runtime_checkable
class ProcessLister(Protocol):
    """A callable that returns a live process snapshot.

    Narrows the dependency `check_quiescence` needs down to "give me the
    current process list", mirroring the ``WorktreeCleanGH`` pattern in
    ``worktree.py`` (issue #641): shrink an external/OS dependency to the
    minimal Protocol surface so test doubles can satisfy it structurally,
    with zero subprocess use, instead of subclassing or monkeypatching the
    real implementation.

    Returns ``(processes, error)`` rather than raising — external process
    errors come back as values, per this repo's invariant (see
    ``list_processes``).
    """

    def __call__(self) -> tuple[Sequence[ProcessInfo], str | None]: ...


@dataclass(frozen=True)
class QuiesceReport:
    """Result of `assert_quiescent`.

    ``matched`` carries every offending `ProcessInfo`, not just one PID per
    logical group (defect #3) — a caller stopping processes must stop all of
    them. ``excluded_pids`` is the self/ancestor exclusion set actually
    applied (defect #1), surfaced so a reviewer can audit what was excluded
    and why, rather than trusting ``ok`` blindly (verdict-content-not-just-
    decision: never accept a boolean without the values behind it).
    ``invalid_patterns`` lists any caller-supplied regex that failed to
    compile; those patterns are skipped (never matched) rather than raising,
    but are surfaced so a bad pattern doesn't silently make the gate looser
    than the caller intended.

    ``ok`` is ``False`` whenever zero patterns compiled -- including the
    all-invalid case -- not just when a process matched. A gate with no
    usable pattern cannot prove anything; reporting ``ok=True`` off an empty
    search space is exactly the fail-open this type exists to prevent (#732).
    A caller that supplies *some* valid patterns alongside invalid ones keeps
    the current match-governed behaviour (``invalid_patterns`` narrows what
    was searched but doesn't itself flip an otherwise-clean result), since a
    still-nonempty search space is the check the caller asked for, just
    narrower than intended -- surfaced, not silently strengthened past what
    was requested.
    """

    ok: bool
    matched: tuple[ProcessInfo, ...]
    excluded_pids: frozenset[int]
    summary: str
    invalid_patterns: tuple[str, ...] = field(default_factory=tuple)


def self_process_chain(pid: int, processes: Sequence[ProcessInfo]) -> frozenset[int]:
    """Return ``pid`` and every ancestor PID, walked over an in-memory snapshot.

    Pure and side-effect free (no subprocess calls) so it is directly
    testable against a fabricated ``processes`` list. Used to build the
    exclusion set that keeps the process performing a quiescence check (and
    every shell/wrapper that launched it) from ever matching a pattern that
    merely appears inside its own invocation (defect #1).

    Termination is guaranteed three ways:
      - A parent PID already seen in the chain (a cycle) stops the walk
        before it is re-added.
      - A parent PID absent from ``processes`` (root of the tree, or a
        snapshot that doesn't include it) stops the walk.
      - `_MAX_CHAIN_DEPTH` hops is a hard backstop regardless of the above.
    """
    ppid_by_pid = {proc.pid: proc.ppid for proc in processes}
    chain: set[int] = set()
    current = pid
    hops = 0
    while current not in chain and hops < _MAX_CHAIN_DEPTH:
        chain.add(current)
        hops += 1
        parent = ppid_by_pid.get(current)
        if parent is None:
            break
        current = parent
    return frozenset(chain)


def _compile_patterns(patterns: Sequence[str]) -> tuple[list[re.Pattern[str]], tuple[str, ...]]:
    """Compile ``patterns``, skipping (never raising on) invalid regexes.

    A caller-supplied pattern that fails to compile (e.g. a Windows path
    pasted in verbatim, where a backslash-letter pair like ``\\Users`` is an
    invalid regex escape) must not crash the quiescence check. It is
    dropped from matching and reported in the second return value so the
    caller can see the gate is narrower than intended, instead of failing
    silently.
    """
    compiled: list[re.Pattern[str]] = []
    invalid: list[str] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error:
            invalid.append(pattern)
    return compiled, tuple(invalid)


def _build_summary(
    *,
    ok: bool,
    matched: Sequence[ProcessInfo],
    excluded_pids: frozenset[int],
    invalid_patterns: Sequence[str],
    compiled_count: int,
) -> str:
    lines: list[str] = []
    if ok:
        lines.append(
            f"quiescent: no process command line matched (excluded {len(excluded_pids)} "
            "self/ancestor pid(s))"
        )
    elif compiled_count == 0:
        # Distinct from "processes matched": here nothing *could* have
        # matched, because no supplied pattern compiled. Reporting this the
        # same way as a real process match would bury the actual reason
        # (unusable patterns, not a live fleet) in the summary text.
        lines.append(
            "NOT quiescent: no usable pattern (0 patterns compiled) -- "
            "quiescence cannot be established"
        )
    else:
        lines.append(f"NOT quiescent: {len(matched)} matching process(es):")
        for proc in matched:
            lines.append(
                f"  pid={proc.pid} ppid={proc.ppid} name={proc.name!r} "
                f"command_line={proc.command_line!r}"
            )
    if invalid_patterns:
        lines.append(f"invalid pattern(s) skipped (never matched): {list(invalid_patterns)!r}")
    return "\n".join(lines)


def assert_quiescent(
    *,
    patterns: Sequence[str],
    processes: Sequence[ProcessInfo],
    self_pid: int,
) -> QuiesceReport:
    """Check whether any process (other than the caller's own chain) matches.

    Pure: takes a process snapshot and never touches subprocess/OS state
    itself, so it is fully unit-testable. ``patterns`` are regexes matched
    against each process's ``command_line`` only (never ``name`` — defect
    #2). ``self_pid`` and its full ancestor chain (via
    `self_process_chain`) are excluded before matching, so the caller and
    whatever launched it can never be reported (defect #1). Every process
    whose command line matches is included in ``matched`` — not just the
    first hit for a given pattern — so a caller stopping the reported
    processes stops the whole matched lineage (defect #3).
    """
    excluded = self_process_chain(self_pid, processes)
    compiled, invalid_patterns = _compile_patterns(patterns)

    matched: list[ProcessInfo] = []
    for proc in processes:
        if proc.pid in excluded:
            continue
        if any(rx.search(proc.command_line) for rx in compiled):
            matched.append(proc)

    # A gate with zero usable patterns can never observe a match regardless
    # of what's actually running, so "nothing matched" alone is not
    # sufficient for "quiescent" -- see the `ok` docstring on `QuiesceReport`
    # for why this must be enforced here (single point of enforcement) and
    # not left to each caller.
    ok = bool(compiled) and not matched
    summary = _build_summary(
        ok=ok,
        matched=matched,
        excluded_pids=excluded,
        invalid_patterns=invalid_patterns,
        compiled_count=len(compiled),
    )
    return QuiesceReport(
        ok=ok,
        matched=tuple(matched),
        excluded_pids=excluded,
        summary=summary,
        invalid_patterns=invalid_patterns,
    )


def list_processes() -> tuple[Sequence[ProcessInfo], str | None]:
    """Snapshot every OS process via ``Win32_Process`` (Windows only).

    Built on the same PowerShell invocation style as
    ``process_utils.sweep_orphan_processes``/``_enumerate_child_pids``:
    ``Get-CimInstance Win32_Process`` piped through ``ConvertTo-Json``, no
    ``psutil`` dependency (that module deliberately avoids one; see its
    docstrings). Returns ``(processes, error)`` instead of raising — a
    PowerShell hiccup must surface as a value so a caller can fail the
    quiescence check *closed* (i.e. "unknown" is never reported as
    "quiescent"), matching this repo's "errors from external processes come
    back as values" invariant.
    """
    if sys.platform != "win32":
        return (), "list_processes is only implemented for win32"

    if not shutil.which("powershell"):
        return (), "powershell executable not found on PATH"

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                # No `-AsArray`: that switch is PowerShell 6+. Windows PowerShell
                # 5.1 -- the edition this host and the self-hosted runners ship --
                # fails the whole command with a ParameterBindingException, which
                # made every quiescence check on 5.1 fail closed and report
                # "not quiescent" regardless of what was actually running. The
                # single-result normalization below is what makes dropping it safe.
                "Get-CimInstance Win32_Process | "
                "Select-Object ProcessId, ParentProcessId, Name, CommandLine | "
                "ConvertTo-Json",
            ],
            capture_output=True,
            text=True,
            timeout=_LIST_PROCESSES_TIMEOUT_SECONDS,
            **no_console_window_kwargs(),
        )
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as exc:
        return (), f"powershell invocation failed: {exc}"

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return (), f"powershell exited {result.returncode}: {stderr}"

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return (), f"could not parse powershell output as JSON: {exc}"

    # Required, not defensive: without `-AsArray` (see above), `ConvertTo-Json`
    # emits a bare object whenever the pipeline yields exactly one result, and a
    # JSON array otherwise. Both shapes are normal input here.
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return (), "powershell output was not a JSON array"

    processes: list[ProcessInfo] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            pid = int(entry["ProcessId"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            ppid = int(entry.get("ParentProcessId") or 0)
        except (TypeError, ValueError):
            ppid = 0
        processes.append(
            ProcessInfo(
                pid=pid,
                ppid=ppid,
                name=str(entry.get("Name") or ""),
                command_line=str(entry.get("CommandLine") or ""),
            )
        )
    return tuple(processes), None


def check_quiescence(
    *,
    patterns: Sequence[str],
    self_pid: int | None = None,
    lister: ProcessLister = list_processes,
) -> QuiesceReport:
    """Fetch a process snapshot via ``lister`` and evaluate `assert_quiescent`.

    ``lister`` defaults to `list_processes` (the real Windows implementation)
    but is the seam a caller — test or future CLI wiring — substitutes to
    inject a fake with zero subprocess use, per the `ProcessLister` Protocol.
    If ``lister`` reports an error, this fails *closed*: ``ok=False``, since
    an unknown process snapshot must never be reported as quiescent.
    """
    resolved_self_pid = self_pid if self_pid is not None else os.getpid()
    processes, error = lister()
    if error is not None:
        return QuiesceReport(
            ok=False,
            matched=(),
            excluded_pids=frozenset(),
            summary=f"process listing failed, failing closed (not quiescent): {error}",
        )
    return assert_quiescent(patterns=patterns, processes=processes, self_pid=resolved_self_pid)
