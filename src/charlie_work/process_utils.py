"""Shared utilities for process management across adapters."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path


def parse_proc_stat_starttime(stat_text: str) -> int | None:
    """Parse /proc/<pid>/stat to extract the starttime field.

    Args:
        stat_text: The content of /proc/<pid>/stat as a string.

    Returns:
        The starttime value in clock ticks since boot, or None if parsing fails.

    The /proc/<pid>/stat format is:
        pid (comm) state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt
        cmajflt utime stime cutime cstime priority nice num_threads itrealvalue
        starttime vsize rss rsslim startcode endcode startstack kstkesp kstkeip
        signal blocked sigignore sigcatch wchan ...

    The comm field can contain spaces and parentheses, so we must split on the
    LAST ')' to correctly separate it from the remaining fields. Using rpartition
    ensures we handle embedded ')' characters in comm (e.g., "(tmux: (0) server)").
    """
    # Split on the LAST ')' to handle comm containing embedded ')'
    before_paren, sep, after_paren = stat_text.rpartition(")")
    if not sep:
        # No ')' found - malformed stat line
        return None

    # After the comm field, we have the remaining space-separated fields
    after_comm = after_paren.strip().split()

    # starttime is at index 19 (0-indexed) after comm
    # Format: state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt
    #         cmajflt utime stime cutime cstime priority nice num_threads
    #         itrealvalue starttime ...
    if len(after_comm) < 20:
        return None

    try:
        return int(after_comm[19])
    except (ValueError, IndexError):
        return None


def is_session_stalled(
    log_path: Path, stall_threshold_minutes: int
) -> tuple[bool, str | None]:
    """Check if a session is stalled based on log file mtime and terminal error markers.

    A session is stalled if:
    1. The log file's mtime is older than the stall threshold (default 20 minutes), OR
    2. The log ends with a terminal error marker (e.g., "Error: A tool was rejected")

    Returns a tuple of (is_stalled, last_log_line).
    """
    if not log_path.exists():
        return False, None

    try:
        log_mtime = datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC)
        age = datetime.now(UTC) - log_mtime
        is_stalled_by_mtime = age > timedelta(minutes=stall_threshold_minutes)

        # Check for terminal error markers in the log
        last_log_line = None
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            lines = log_text.splitlines()
            if lines:
                last_log_line = lines[-1].strip()
        except OSError:
            last_log_line = None

        # Terminal error markers that indicate a dead agent regardless of mtime
        terminal_error_patterns = [
            "Error: A tool was rejected",
            "Error:",
        ]

        has_terminal_error = False
        if last_log_line:
            for pattern in terminal_error_patterns:
                if pattern in last_log_line:
                    has_terminal_error = True
                    break

        is_stalled = is_stalled_by_mtime or has_terminal_error
        return is_stalled, last_log_line
    except OSError:
        return False, None


def kill_process_tree(pid: int) -> list[int]:
    """Kill a process and all its children (process tree).

    Returns a list of PIDs that were killed (including the root PID).
    On Windows, uses taskkill /T /F to terminate the process tree.
    On POSIX, uses os.killpg to kill the process group.

    This is used to clean up stalled sessions where the wrapper PID is alive
    but the agent loop has died, leaving child processes holding resources.
    """
    killed_pids = []

    if pid <= 0:
        return killed_pids

    try:
        if os.name == "nt":
            # Windows: use taskkill to terminate the process tree
            import subprocess

            result = subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
            )
            # taskkill returns 0 for success, 1 for "process not found" (which is fine)
            if result.returncode in (0, 1):
                killed_pids.append(pid)
                # taskkill /T kills children but doesn't list them
                # We could parse the output, but the spec only requires
                # that children are killed, not enumerated
        else:
            # POSIX: kill the process group
            try:
                os.killpg(os.getpgid(pid), 9)  # SIGKILL
                killed_pids.append(pid)
            except (ProcessLookupError, OSError):
                # Process may have already exited
                pass
    except Exception:
        # Best-effort kill - don't raise
        pass

    return killed_pids
