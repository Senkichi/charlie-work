"""Shared utilities for process management across adapters."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .subprocess_runner import no_console_window_kwargs


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
    log_path: Path,
    stall_threshold_minutes: int,
    terminal_error_markers: Sequence[str] | None = None,
) -> tuple[bool, str | None]:
    """Check if a session is stalled based on log file mtime and terminal error markers.

    A session is stalled if:
    1. The log file's mtime is older than the stall threshold (default 20 minutes), OR
    2. The log ends with a terminal error marker (e.g., "Error: A tool was rejected")

    Returns a tuple of (is_stalled, last_log_line).

    Args:
        log_path: Path to the log file
        stall_threshold_minutes: Minutes of log inactivity to consider stalled
        terminal_error_markers: List of terminal error marker strings. If None, uses
            the default hardcoded list for backward compatibility.
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
        # Anchored to known-fatal shapes to avoid false positives from retry messages
        if terminal_error_markers is None:
            # Default hardcoded list for backward compatibility
            terminal_error_patterns = [
                "Error: A tool was rejected",
                "Error: Agent error:",  # Devin fatal prefix
            ]
        else:
            terminal_error_patterns = terminal_error_markers

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


def _enumerate_child_pids(pid: int) -> list[int]:
    """Enumerate child PIDs of a given process.

    On POSIX: scans /proc to find processes with the given parent PID.
    On Windows: uses PowerShell CIM to query ParentProcessId (primary) with wmic fallback.

    Returns a list of child PIDs (may be empty).
    """
    children = []

    if os.name == "nt":
        # Windows: use PowerShell CIM (primary) with wmic fallback
        try:
            import subprocess
            import shutil

            # Try PowerShell CIM first (modern Windows 11+)
            if shutil.which("powershell"):
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        f'Get-CimInstance Win32_Process -Filter "ParentProcessId={pid}" | Select-Object -ExpandProperty ProcessId',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    **no_console_window_kwargs(),
                )
                # Parse output: extract PIDs (one per line)
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line and line.isdigit():
                        children.append(int(line))
            else:
                # Fallback to wmic if PowerShell not available
                if shutil.which("wmic"):
                    result = subprocess.run(
                        ["wmic", "process", "where", f"ParentProcessId={pid}", "get", "ProcessId"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        **no_console_window_kwargs(),
                    )
                    # Parse output: skip header line, extract PIDs
                    for line in result.stdout.splitlines():
                        line = line.strip()
                        if line and line.isdigit():
                            children.append(int(line))
        except (
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
            ValueError,
            FileNotFoundError,
        ):
            # Best-effort enumeration - don't fail the kill if enumeration fails
            pass
    else:
        # POSIX: scan /proc to find children
        try:
            proc_path = Path("/proc")
            if not proc_path.exists():
                return children

            for entry in proc_path.iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    stat_path = entry / "stat"
                    if not stat_path.exists():
                        continue
                    stat_text = stat_path.read_text(encoding="utf-8", errors="replace")
                    # Parse PPID (field 4 in /proc/<pid>/stat)
                    # Format: pid (comm) state ppid ...
                    _, _, after_paren = stat_text.rpartition(")")
                    fields = after_paren.strip().split()
                    if len(fields) >= 4:
                        ppid = int(fields[1])  # PPID is at index 1 after comm
                        if ppid == pid:
                            children.append(int(entry.name))
                except (OSError, ValueError, IndexError):
                    continue
        except OSError:
            # Best-effort enumeration - don't fail the kill if enumeration fails
            pass

    return children


def get_process_start_time(pid: int) -> float | None:
    """Return the process start time as a Unix timestamp, or None if unavailable.

    Used to verify process identity and detect PID recycling. The value is
    resolved from the operating system, not from a cached sidecar, so it can be
    compared against a stored fingerprint to decide whether a PID still refers
    to the same process.
    """
    if pid <= 0:
        return None

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        _WIN_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(_WIN_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            creation_time = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel_time = wintypes.FILETIME()
            user_time = wintypes.FILETIME()
            if kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation_time),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                filetime = (creation_time.dwHighDateTime << 32) | creation_time.dwLowDateTime
                return filetime / 10_000_000 - 11644473600
            return None
        finally:
            kernel32.CloseHandle(handle)
    else:
        # POSIX: read /proc/<pid>/stat
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                stat = f.read()
            starttime_ticks = parse_proc_stat_starttime(stat)
            if starttime_ticks is None:
                return None
            tick_hz = os.sysconf("SC_CLK_TCK")
            if tick_hz <= 0:
                tick_hz = 100
            try:
                with open("/proc/uptime", "r") as f:
                    uptime_seconds = float(f.read().split()[0])
            except (OSError, ValueError, IndexError):
                uptime_seconds = 0
            boot_time = time.time() - uptime_seconds
            return boot_time + (starttime_ticks / tick_hz)
        except (OSError, ValueError, IndexError):
            return None


def is_pid_alive(pid: int, expected_start_time: float | None = None) -> bool:
    """Return True if ``pid`` is a live process and (optionally) start time matches.

    On Windows this uses ``OpenProcess`` + ``GetExitCodeProcess``; on POSIX it
    uses ``os.kill(pid, 0)``. When ``expected_start_time`` is provided, the
    current process start time is also fetched and compared against the stored
    fingerprint, returning False if the PID has been recycled.

    Indeterminate probes (e.g. a transient ``OpenProcess`` failure or a
    start-time query that returns ``None`` while the PID still appears to exist)
    are treated as ``True``.  Reaping a live worker on a false-negative liveness
    signal is far worse than delaying a reap by one pass, so this function only
    returns ``False`` when it can prove the process is dead or has been
    recycled.
    """
    if pid <= 0:
        return False

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        _WIN_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        _WIN_STILL_ACTIVE = 259
        _ERROR_ACCESS_DENIED = 5
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(_WIN_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # ``OpenProcess`` can fail because the PID does not exist (definitive
            # dead signal) or because the process exists but we are denied access
            # (indeterminate).  Treat access-denied as alive; all other open
            # failures are treated as not running.
            return kernel32.GetLastError() == _ERROR_ACCESS_DENIED
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                # Could not read the exit code; we cannot prove the process is
                # dead, so treat it as indeterminate rather than failing open.
                return True
            if exit_code.value != _WIN_STILL_ACTIVE:
                return False
        finally:
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we cannot signal it; treat as indeterminate.
            return True
        except (OSError, ValueError):
            return False

    if expected_start_time is not None:
        current_start_time = get_process_start_time(pid)
        if current_start_time is None:
            # Could not verify identity.  Treat as indeterminate (alive) rather
            # than reaping a worker whose liveness probe merely failed
            # (issue #360 criterion #1 / issue #343).
            return True
        if abs(current_start_time - expected_start_time) > 1.0:
            # Start time mismatch - PID has been recycled
            return False

    return True


def kill_process_tree(pid: int, expected_start_time: float | None = None) -> list[int]:
    """Kill a process and all its children (process tree).

    Returns a list of PIDs that were killed (including the root PID and children).
    On Windows, uses taskkill /T /F to terminate the process tree.
    On POSIX, uses os.killpg to kill the process group.

    This is used to clean up stalled sessions where the wrapper PID is alive
    but the agent loop has died, leaving child processes holding resources.

    Args:
        pid: The process ID to kill.
        expected_start_time: The expected process start time (Unix timestamp in seconds).
            If provided, this function will re-verify the process identity immediately
            before killing to prevent PID recycling attacks. If the start time mismatch,
            the process is NOT killed and an empty list is returned.

    Returns:
        A list of PIDs that were killed (including the root PID and children). Returns empty list
        if the process could not be killed or if the start time verification failed.
    """
    killed_pids = []

    if pid <= 0:
        return killed_pids

    # Re-verify process identity via start time if provided
    if expected_start_time is not None:
        current_start_time = get_process_start_time(pid)

        # If we couldn't get current start time, conservatively don't kill
        if current_start_time is None:
            return killed_pids

        # Allow 1-second tolerance for unit conversion differences
        if abs(current_start_time - expected_start_time) > 1.0:
            # Start time mismatch - PID has been recycled
            return killed_pids

    # Enumerate children before killing
    child_pids = _enumerate_child_pids(pid)

    try:
        if os.name == "nt":
            # Windows: use taskkill to terminate the process tree
            import subprocess

            result = subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                **no_console_window_kwargs(),
            )
            # taskkill returns 0 for success, 1 for "process not found" (which is fine)
            if result.returncode in (0, 1):
                killed_pids.append(pid)
                # taskkill /T kills children but doesn't list them reliably
                # Add enumerated children to the killed list
                killed_pids.extend(child_pids)
        else:
            # POSIX: kill the process group
            try:
                pgid = os.getpgid(pid)
                # Defense-in-depth: refuse to kill our own process group
                if pgid == os.getpgid(0):
                    # Would kill the orchestrator and all in-flight workers
                    return killed_pids
                os.killpg(pgid, 9)  # SIGKILL
                killed_pids.append(pid)
                # Add enumerated children to the killed list
                killed_pids.extend(child_pids)
            except (ProcessLookupError, OSError):
                # Process may have already exited
                pass
    except Exception:
        # Best-effort kill - don't raise
        pass

    return killed_pids


def sweep_orphan_processes(worktree_path: str) -> list[int]:
    """Sweep for orphan processes whose CommandLine references a worktree path.

    On Windows: Uses PowerShell Get-CimInstance Win32_Process to find processes
    whose CommandLine contains the worktree path. This catches detached/daemonized
    processes that survived a process tree kill (e.g., nohup-style background processes).

    On POSIX: Not implemented (returns empty list). POSIX process groups handle
    detachment better via killpg, and /proc enumeration is more complex.

    This is a read-only detection function. Callers should decide whether to kill
    the returned PIDs based on policy (e.g., janitor warnings vs. automatic cleanup).

    Args:
        worktree_path: The worktree path to search for in process CommandLines.

    Returns:
        A list of PIDs whose CommandLine references the worktree path.
    """
    orphans = []

    if os.name != "nt":
        # POSIX: not implemented - process groups handle detachment better
        return orphans

    try:
        import subprocess
        import shutil

        if not shutil.which("powershell"):
            return orphans

        # Use PowerShell to query Win32_Process for CommandLine matching the worktree path
        # We use -like with wildcards for partial matching (handles both forward and backslashes)
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f'Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like "*{worktree_path}*" }} | Select-Object -ExpandProperty ProcessId',
            ],
            capture_output=True,
            text=True,
            timeout=10,
            **no_console_window_kwargs(),
        )

        # Parse output: extract PIDs (one per line)
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and line.isdigit():
                orphans.append(int(line))
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        # Best-effort sweep - don't fail if PowerShell fails
        pass

    return orphans
