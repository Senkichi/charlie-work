"""``ps``-backed process listing for POSIX hosts without ``/proc`` (spec C2, CW-3).

macOS has no procfs, so every ``/proc`` reader in charlie-work degrades to "no
processes" there: the orphan sweep, child enumeration and the host-load probe.
``ps -axww -o pid= -o ppid= -o command=`` is the one listing both BSD (macOS)
and procps (Linux) ``ps`` accept: ``-ax`` lists every process, ``ww`` lifts the
column-width cap so command lines are never truncated, and the trailing ``=``
drops each column header. Like ``quiesce.list_processes`` this returns
``(processes, error)`` and never raises.
"""

from __future__ import annotations

from collections.abc import Callable

from .quiesce import ProcessInfo
from .subprocess_runner import RunResult, command_failure_message, run_captured

PS_COMMAND: tuple[str, ...] = ("ps", "-axww", "-o", "pid=", "-o", "ppid=", "-o", "command=")
PS_TIMEOUT_SECONDS = 10


def parse_ps(stdout: str) -> tuple[ProcessInfo, ...]:
    """Parse ``PS_COMMAND`` output; malformed lines are skipped."""
    rows: list[ProcessInfo] = []
    for line in stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        command_line = parts[2].strip() if len(parts) == 3 else ""
        name = command_line.split(" ", 1)[0].rsplit("/", 1)[-1]
        rows.append(ProcessInfo(int(parts[0]), int(parts[1]), name, command_line))
    return tuple(rows)


def list_processes_ps(
    run_command: Callable[..., RunResult] = run_captured,
) -> tuple[tuple[ProcessInfo, ...], str | None]:
    """Every process on the host via ``ps``, or ``((), error)`` when ``ps`` fails."""
    command = list(PS_COMMAND)
    result = run_command(command, cwd=None, timeout_seconds=PS_TIMEOUT_SECONDS)
    if not result.ok:
        return (), command_failure_message(command, result, "ps failed")
    return parse_ps(result.stdout), None
