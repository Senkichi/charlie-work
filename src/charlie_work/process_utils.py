"""Shared utilities for process management across adapters."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .subprocess_runner import hidden_console_kwargs, no_console_window_kwargs


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
    _, sep, after_paren = stat_text.rpartition(")")
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


def worker_terminal_status_path(
    sessions_dir: Path, issue_number: int, sidecar_suffix: str
) -> Path:
    """Path to a worker's durable terminal-status record.

    Named ``issue-<n>.<suffix>.terminal.json`` -- distinct from the adapter's
    own sidecar (``issue-<n>.<suffix>.json``) so the terminal record is never
    touched by sidecar reaping (``WorkerView.reap_sidecar``) and survives
    independently of it (issue #773). ``sidecar_suffix`` mirrors the adapter's
    own sidecar suffix (e.g. ``"claude"`` for ``adapter_kind="claude-code"``)
    purely so both files sit side by side and are easy to correlate by eye;
    readers should not need to know the suffix (see ``find_worker_terminal_status``).
    """
    return sessions_dir / f"issue-{issue_number}.{sidecar_suffix}.terminal.json"


def write_worker_terminal_status(
    path: Path,
    *,
    pid: int,
    exit_code: int | None,
    started_at: str,
    ended_at: str,
    duration_seconds: float,
    worker_outcome: dict[str, Any] | None = None,
) -> None:
    """Atomically persist a worker process's terminal status (issue #773).

    Written by ``start_terminal_status_watcher`` once its ``Popen.poll()``
    loop observes the process has exited. Uses the tmp-file + ``replace()``
    pattern required by CLAUDE.md for every JSON state write so a reader (the
    orphan detector's polling pass) never observes a partially-written file.
    """
    payload: dict[str, Any] = {
        "pid": pid,
        "exit_code": exit_code,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
    }
    if worker_outcome is not None:
        payload["worker_outcome"] = worker_outcome
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def find_worker_terminal_status(sessions_dir: Path, issue_number: int) -> dict[str, Any] | None:
    """Return the most recently written terminal-status record for ``issue_number``.

    Globs ``issue-<n>.*.terminal.json`` rather than requiring a caller-known
    adapter suffix, because this is consulted by the state.json-driven orphan
    sweep (``workflow._detect_and_handle_orphaned_workers``), which is
    explicitly the fallback path for when an issue's own sidecar (and thus
    its adapter identity) may already be gone. Ties are broken by mtime so a
    redispatch's fresh record wins over a stale one left by an earlier
    attempt. Never raises: a missing directory, no matching file, an
    unreadable file, or malformed JSON all resolve to ``None`` -- this is the
    "no terminal record" case that callers must treat as today's legacy
    behavior (issue #773 acceptance criterion: fully backward compatible
    when no record exists).
    """
    if not sessions_dir.is_dir():
        return None
    candidates = list(sessions_dir.glob(f"issue-{issue_number}.*.terminal.json"))
    if not candidates:
        return None

    def _mtime(candidate: Path) -> float:
        try:
            return candidate.stat().st_mtime
        except OSError:
            return -1.0

    candidates.sort(key=_mtime, reverse=True)
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data
    return None


# Sleep interval for the terminal-status watcher's poll() loop. Workers run
# for minutes, so this granularity costs nothing; kept as a named constant
# (rather than an inline literal) since it's the one knob that trades
# poll-loop wakeups against exit-timestamp precision.
_TERMINAL_STATUS_POLL_INTERVAL_SECONDS = 2.0


def start_terminal_status_watcher(
    process: subprocess.Popen[Any], path: Path, worktree_path: Path | None = None
) -> threading.Thread:
    """Spawn a daemon thread that records ``process``'s terminal status at exit.

    This is the durable half of issue #773's fix: reading ``GetExitCodeProcess``
    at orphan-sweep poll time is unreliable (the sweep runs every
    ``full_pass_interval_seconds`` -- 300s by default -- by which point the
    process is long reaped and the PID may already be recycled), so the exit
    code must be captured once, at the moment the process actually exits, and
    persisted where a later poll can find it.

    When ``worktree_path`` is provided, the watcher also reads the worker's
    structured outcome file (``WORKER_OUTCOME_FILENAME``) after the process
    exits and copies it into the durable terminal status. This preserves the
    worker's push/PR-failure signal even after the worktree is removed
    (issue #935).

    Does NOT call ``Popen.wait()`` or ``Popen.communicate()`` -- on the caller
    thread or any other. CLAUDE.md's "adapters must not block on worker
    completion" invariant bans those two calls outright, not just a
    caller-thread version of them (this project was previously bitten by
    exactly this deadlock class). Instead this polls ``Popen.poll()`` in a
    sleep loop, so the banned calls are structurally absent from every code
    path here -- a later edit cannot reintroduce the deadlock class by moving
    a `wait()`/`communicate()` call onto a different thread, because there is
    no such call to move. Never raises: a failure inside the thread (a
    poll() error, or a write failure) is swallowed, since this is best-effort
    telemetry that must not crash the orchestrator or the worker's own I/O
    threads (e.g. the tee_stream_json reader thread also draining
    ``process.stdout`` concurrently in ``claude_code.launch_claude_worker``).
    """
    pid = process.pid
    started_monotonic = time.monotonic()
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def _watch() -> None:
        try:
            while process.poll() is None:
                time.sleep(_TERMINAL_STATUS_POLL_INTERVAL_SECONDS)
            exit_code = process.returncode
        except Exception:
            exit_code = None
        duration_seconds = time.monotonic() - started_monotonic
        ended_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        try:
            worker_outcome = None
            if worktree_path is not None:
                # Local import to avoid a circular import: worktree.py already
                # imports is_pid_alive from this module (issue #935).
                from .worktree import read_worker_outcome

                worker_outcome = read_worker_outcome(worktree_path)
            write_worker_terminal_status(
                path,
                pid=pid,
                exit_code=exit_code,
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=duration_seconds,
                worker_outcome=worker_outcome,
            )
        except Exception:
            # Best-effort telemetry: a write failure (disk full, permissions,
            # or -- observed under test mocking -- a non-JSON-serializable
            # field) must not raise inside a daemon thread the caller isn't
            # watching. Narrower than OSError deliberately, matching the
            # `_tee_output` thread's `except Exception` precedent elsewhere
            # in this codebase (claude_code.py) for the same reason.
            pass

    thread = threading.Thread(target=_watch, daemon=True, name=f"terminal-status-watcher-{pid}")
    thread.start()
    return thread


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

    # Defense-in-depth (issue #627): never kill the calling process. The fleet
    # supervisor reaps stalled workers from within its own process image, so
    # ``os.getpid()`` here IS the supervisor. A tree walk or ``killpg`` that
    # reached the supervisor would terminate it silently (exit=-1, no event, no
    # alert) — exactly the #627 failure shape. Exempt it explicitly by PID
    # rather than by name matching: process names are toothpick-brittle (every
    # binary rename breaks them) and #608 shows child enumeration is
    # untrustworthy under load. On Windows ``taskkill /T /PID`` only kills
    # descendants so the supervisor (an ancestor) is already safe, but this
    # guard also covers a recycled-PID or bogus-caller case where ``pid``
    # passed in is the supervisor itself. On POSIX the existing
    # ``pgid == os.getpgid(0)`` guard covers the process-group shape; this PID
    # guard covers the direct case.
    if pid == os.getpid():
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
            result = subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                **no_console_window_kwargs(),
            )
            # taskkill returns 0 for success, 1 for "process not found" (which
            # is fine). But the return code is not a reliable kill signal under
            # load: taskkill can return codes outside (0, 1) even when it
            # successfully terminates the process (e.g., when the process is
            # already exiting, or under box contention). Without the liveness
            # fallback below, callers that retry on [] (and tests that assert
            # ``pid in killed``) see a phantom miss: the process is dead but
            # ``kill_process_tree`` reported []. Verify the process is actually
            # dead rather than trusting the return code alone.
            if result.returncode in (0, 1) or not is_pid_alive(pid):
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

    Returns:
        A list of dicts describing processes whose CommandLine references the
        worktree path. Each dict contains ``pid`` (int), ``name`` (str), and
        ``command_line`` (str). POSIX callers always get an empty list.
    """
    orphans: list[dict[str, Any]] = []

    if os.name != "nt":
        # POSIX: not implemented - process groups handle detachment better
        return orphans

    try:
        if not shutil.which("powershell"):
            return orphans

        # Use PowerShell to query Win32_Process for CommandLine matching the worktree path.
        # Select-Object + ConvertTo-Json preserves PID, image name, and command line so
        # callers can log what was killed and identify respawn sources.
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f'Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like "*{worktree_path}*" }} | Select-Object ProcessId, CommandLine, Name | ConvertTo-Json -AsArray',
            ],
            capture_output=True,
            text=True,
            timeout=10,
            **no_console_window_kwargs(),
        )

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
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        # Best-effort sweep - don't fail if PowerShell fails
        pass

    return orphans


def popen_worker(
    args: Sequence[str] | str,
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    **popen_kwargs: Any,
) -> subprocess.Popen[Any]:
    """Launch a worker process as the single point for creationflags/process-group composition.

    Injects the worker hidden-console policy into the ``subprocess.Popen`` call:
    - ``creationflags`` and ``startupinfo`` are composed through
      ``hidden_console_kwargs()`` so ``CREATE_NEW_CONSOLE`` is combined with
      ``CREATE_NEW_PROCESS_GROUP`` on Windows and the console is hidden via
      ``STARTF_USESHOWWINDOW`` / ``SW_HIDE``. On POSIX it is a no-op.
    - ``start_new_session`` defaults to ``True`` on POSIX and is omitted on
      Windows; callers may override by passing it explicitly.

    All other ``Popen`` keyword arguments are passed through. The helper returns
    the ``Popen`` object immediately and never waits or communicates.
    """
    if cwd is not None:
        popen_kwargs["cwd"] = cwd
    if env is not None:
        popen_kwargs["env"] = env

    if "start_new_session" not in popen_kwargs and os.name != "nt":
        popen_kwargs["start_new_session"] = True

    extra_flags = popen_kwargs.pop("creationflags", 0)
    process_group_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    popen_kwargs.update(hidden_console_kwargs(extra_flags | process_group_flag))

    return subprocess.Popen(args, **popen_kwargs)
