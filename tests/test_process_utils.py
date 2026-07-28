from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from charlie_work.process_utils import (
    is_pid_alive,
    is_session_stalled,
    kill_process_tree,
    popen_worker,
    sweep_orphan_processes,
)


def test_is_session_stalled_with_old_mtime(tmp_path: Path) -> None:
    """Test that a session is stalled when log mtime is older than threshold."""
    log_file = tmp_path / "test.log"
    log_file.write_text("some log content\n", encoding="utf-8")

    # Set mtime to 25 minutes ago (older than default 20 minute threshold)
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    is_stalled, last_line = is_session_stalled(log_file, stall_threshold_minutes=20)
    assert is_stalled is True
    assert last_line == "some log content"


def test_is_session_stalled_with_fresh_mtime(tmp_path: Path) -> None:
    """Test that a session is not stalled when log mtime is recent."""
    log_file = tmp_path / "test.log"
    log_file.write_text("some log content\n", encoding="utf-8")

    # Set mtime to 5 minutes ago (younger than 20 minute threshold)
    recent_time = datetime.now(UTC) - timedelta(minutes=5)
    timestamp = recent_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    is_stalled, last_line = is_session_stalled(log_file, stall_threshold_minutes=20)
    assert is_stalled is False
    assert last_line == "some log content"


def test_is_session_stalled_with_terminal_error_marker(tmp_path: Path) -> None:
    """Test that a session is stalled when log ends with terminal error marker."""
    log_file = tmp_path / "test.log"
    log_file.write_text("some log content\nError: A tool was rejected\n", encoding="utf-8")

    # Even with fresh mtime, terminal error should trigger stall
    is_stalled, last_line = is_session_stalled(log_file, stall_threshold_minutes=20)
    assert is_stalled is True
    assert last_line == "Error: A tool was rejected"


def test_is_session_stalled_with_agent_error_marker(tmp_path: Path) -> None:
    """Test that a session is stalled when log ends with Agent error marker."""
    log_file = tmp_path / "test.log"
    log_file.write_text(
        "some log content\nError: Agent error: something failed\n", encoding="utf-8"
    )

    is_stalled, last_line = is_session_stalled(log_file, stall_threshold_minutes=20)
    assert is_stalled is True
    assert last_line == "Error: Agent error: something failed"


def test_is_session_stalled_healthy_session(tmp_path: Path) -> None:
    """Test that a healthy session (recent log, no terminal marker) is not stalled."""
    log_file = tmp_path / "test.log"
    log_file.write_text("working on issue\nmaking progress\n", encoding="utf-8")

    # Fresh mtime, no terminal error
    is_stalled, last_line = is_session_stalled(log_file, stall_threshold_minutes=20)
    assert is_stalled is False
    assert last_line == "making progress"


def test_is_session_stalled_retry_message_not_stalled(tmp_path: Path) -> None:
    """Test that a retry-style error message with fresh mtime does NOT trigger stall."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Retrying after Error: connection reset\n", encoding="utf-8")

    # Fresh mtime, retry message should not trigger stall (not a terminal error)
    is_stalled, last_line = is_session_stalled(log_file, stall_threshold_minutes=20)
    assert is_stalled is False
    assert last_line == "Retrying after Error: connection reset"


def test_is_session_stalled_nonexistent_log(tmp_path: Path) -> None:
    """Test that a non-existent log file is not considered stalled."""
    log_file = tmp_path / "nonexistent.log"

    is_stalled, last_line = is_session_stalled(log_file, stall_threshold_minutes=20)
    assert is_stalled is False
    assert last_line is None


def test_is_session_stalled_empty_log(tmp_path: Path) -> None:
    """Test that an empty log file is handled correctly."""
    log_file = tmp_path / "empty.log"
    log_file.write_text("", encoding="utf-8")

    is_stalled, last_line = is_session_stalled(log_file, stall_threshold_minutes=20)
    # Empty log with fresh mtime should not be stalled
    assert is_stalled is False
    assert last_line is None


def test_kill_process_tree_invalid_pid() -> None:
    """Test that kill_process_tree handles invalid PIDs gracefully."""
    # Invalid PID should return empty list
    killed = kill_process_tree(-1)
    assert killed == []

    killed = kill_process_tree(0)
    assert killed == []


def test_kill_process_tree_nonexistent_pid() -> None:
    """Test that kill_process_tree handles non-existent PIDs gracefully."""
    from unittest.mock import patch

    # Mock _enumerate_child_pids to avoid wmic on Windows
    with patch("charlie_work.process_utils._enumerate_child_pids", return_value=[]):
        # Use a PID that likely doesn't exist
        killed = kill_process_tree(999999)
        # Should return empty list or just the PID if the kill attempt was made
        # The important thing is it doesn't raise an exception
        assert isinstance(killed, list)


def test_kill_process_tree_start_time_verification() -> None:
    """Test that kill_process_tree verifies start time when provided."""
    from unittest.mock import patch

    # Spawn a real child process
    # On POSIX, use start_new_session=True to avoid sharing pytest's process group
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )

    try:
        # Get the actual start time
        from charlie_work.process_utils import parse_proc_stat_starttime
        import time

        actual_start_time = None
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            _WIN_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(_WIN_PROCESS_QUERY_LIMITED_INFORMATION, False, proc.pid)
            if handle:
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
                        filetime = (
                            creation_time.dwHighDateTime << 32
                        ) | creation_time.dwLowDateTime
                        unix_time = filetime / 10_000_000 - 11644473600
                        actual_start_time = unix_time
                finally:
                    kernel32.CloseHandle(handle)
        else:
            # POSIX: read /proc/<pid>/stat
            try:
                with open(f"/proc/{proc.pid}/stat", "r") as f:
                    stat = f.read()
                starttime_ticks = parse_proc_stat_starttime(stat)
                if starttime_ticks is not None:
                    tick_hz = os.sysconf("SC_CLK_TCK")
                    if tick_hz <= 0:
                        tick_hz = 100
                    try:
                        with open("/proc/uptime", "r") as f:
                            uptime_seconds = float(f.read().split()[0])
                    except (OSError, ValueError, IndexError):
                        uptime_seconds = 0
                    boot_time = time.time() - uptime_seconds
                    actual_start_time = boot_time + (starttime_ticks / tick_hz)
            except (OSError, ValueError, IndexError):
                pass

        # If we couldn't get start time, skip this test
        if actual_start_time is None:
            proc.terminate()
            proc.wait()
            return

        # Mock _enumerate_child_pids to avoid wmic on Windows
        with patch("charlie_work.process_utils._enumerate_child_pids", return_value=[]):
            # Test 1: Wrong start time should NOT kill
            wrong_start_time = actual_start_time - 1000  # 1000 seconds in the past
            killed = kill_process_tree(proc.pid, wrong_start_time)
            assert killed == []  # Should not kill due to start time mismatch

            # Test 2: Correct start time should kill
            killed = kill_process_tree(proc.pid, actual_start_time)
            assert proc.pid in killed  # Should kill the process

    finally:
        # Clean up if still alive
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
def test_kill_process_tree_own_group_guard_posix() -> None:
    """Test that kill_process_tree refuses to kill its own process group on POSIX."""
    # Try to kill our own process group - should refuse
    own_pid = os.getpid()
    os.getpgid(own_pid)

    # Attempt to kill our own process group
    killed = kill_process_tree(own_pid)

    # Should return empty list because we refused to kill our own group
    assert killed == []


def test_kill_process_tree_self_pid_exempt() -> None:
    """kill_process_tree never kills the calling process (issue #627).

    The fleet supervisor reaps stalled workers from within its own process
    image, so a ``kill_process_tree(os.getpid())`` call (e.g. from a
    recycled-PID or bogus-caller case) would terminate it silently. The
    PID self-exemption guard returns an empty list without attempting
    the kill, regardless of platform.
    """
    own_pid = os.getpid()
    killed = kill_process_tree(own_pid)
    assert killed == []
    # The calling process is still alive.
    assert os.getpid() == own_pid


def test_kill_process_tree_enumerates_children() -> None:
    """Test that kill_process_tree enumerates and includes child PIDs."""
    # Spawn a real parent process that will spawn a child
    # On POSIX, use start_new_session=True to avoid sharing pytest's process group
    parent_proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']); import time; time.sleep(10)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )

    try:
        # Give the parent time to spawn the child
        import time

        time.sleep(0.5)

        # Get the actual child PID(s)
        from charlie_work.process_utils import _enumerate_child_pids

        child_pids = _enumerate_child_pids(parent_proc.pid)

        if not child_pids:
            # If no children were spawned, skip this test
            parent_proc.terminate()
            parent_proc.wait()
            return

        # Kill the parent process tree
        killed = kill_process_tree(parent_proc.pid, expected_start_time=None)

        # Check that the parent PID is in the killed list
        assert parent_proc.pid in killed

        # Check that the enumerated child PIDs are in the killed list
        for child_pid in child_pids:
            assert child_pid in killed
    finally:
        # Clean up if still alive
        if parent_proc.poll() is None:
            parent_proc.terminate()
            parent_proc.wait()


def test_is_session_stalled_custom_terminal_markers(tmp_path: Path) -> None:
    """Test that is_session_stalled uses custom terminal_error_markers when provided."""
    log_file = tmp_path / "test.log"
    log_file.write_text("some log content\nCustom fatal error\n", encoding="utf-8")

    # Default markers should not trigger stall
    is_stalled, last_line = is_session_stalled(
        log_file, stall_threshold_minutes=20, terminal_error_markers=None
    )
    assert is_stalled is False
    assert last_line == "Custom fatal error"

    # Custom marker should trigger stall
    is_stalled, last_line = is_session_stalled(
        log_file,
        stall_threshold_minutes=20,
        terminal_error_markers=("Custom fatal error",),
    )
    assert is_stalled is True
    assert last_line == "Custom fatal error"


def test_is_session_stalled_default_markers_backward_compat(tmp_path: Path) -> None:
    """Test that is_session_stalled with None markers uses the default hardcoded list for backward compatibility."""
    log_file = tmp_path / "test.log"
    log_file.write_text("some log content\nError: A tool was rejected\n", encoding="utf-8")

    # None should use default markers
    is_stalled, last_line = is_session_stalled(
        log_file, stall_threshold_minutes=20, terminal_error_markers=None
    )
    assert is_stalled is True
    assert last_line == "Error: A tool was rejected"

    # Explicit default markers should behave the same
    is_stalled, last_line = is_session_stalled(
        log_file,
        stall_threshold_minutes=20,
        terminal_error_markers=("Error: A tool was rejected", "Error: Agent error:"),
    )
    assert is_stalled is True
    assert last_line == "Error: A tool was rejected"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
def test_sweep_orphan_processes_posix_returns_empty() -> None:
    """Test that sweep_orphan_processes returns empty list on POSIX."""
    # On POSIX, the function is not implemented and should return empty list
    orphans = sweep_orphan_processes("/some/worktree/path")
    assert orphans == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_sweep_orphan_processes_windows_no_powershell() -> None:
    """Test that sweep_orphan_processes handles missing PowerShell gracefully."""
    from unittest.mock import patch

    # Mock shutil.which to return None (PowerShell not found)
    with patch("shutil.which", return_value=None):
        orphans = sweep_orphan_processes("/some/worktree/path")
        assert orphans == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_sweep_orphan_processes_windows_parsing() -> None:
    """Test that sweep_orphan_processes parses PowerShell JSON output correctly."""
    import json
    from unittest.mock import patch

    sample = [
        {
            "ProcessId": 1234,
            "Name": "python.exe",
            "CommandLine": "python worker.py /some/worktree/path",
        },
        {
            "ProcessId": 5678,
            "Name": "node.exe",
            "CommandLine": "node server.js /some/worktree/path",
        },
    ]
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = json.dumps(sample)
        orphans = sweep_orphan_processes("/some/worktree/path")
        assert orphans == [
            {
                "pid": 1234,
                "name": "python.exe",
                "command_line": "python worker.py /some/worktree/path",
            },
            {
                "pid": 5678,
                "name": "node.exe",
                "command_line": "node server.js /some/worktree/path",
            },
        ]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_sweep_orphan_processes_windows_empty_output() -> None:
    """Test that sweep_orphan_processes handles empty PowerShell output."""
    from unittest.mock import patch

    # Mock subprocess.run to return empty output
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = ""
        orphans = sweep_orphan_processes("/some/worktree/path")
        assert orphans == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_sweep_orphan_processes_windows_subprocess_error() -> None:
    """Test that sweep_orphan_processes handles subprocess errors gracefully."""
    from unittest.mock import patch

    # Mock subprocess.run to raise an exception
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("powershell", 10)
        orphans = sweep_orphan_processes("/some/worktree/path")
        assert orphans == []


def test_is_pid_alive_true_for_live_process(tmp_path: Path) -> None:
    """``is_pid_alive`` returns True for a running process with a matching start time."""
    from charlie_work.process_utils import get_process_start_time

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )
    try:
        start_time = get_process_start_time(proc.pid)
        assert start_time is not None
        assert is_pid_alive(proc.pid, start_time) is True
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_is_pid_alive_false_for_dead_process() -> None:
    """``is_pid_alive`` returns False for a process that has already exited."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc.wait(timeout=5)
    assert is_pid_alive(proc.pid) is False


def test_is_pid_alive_false_for_mismatched_start_time(tmp_path: Path) -> None:
    """``is_pid_alive`` returns False when the start time does not match (PID recycled)."""
    from charlie_work.process_utils import get_process_start_time

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )
    try:
        start_time = get_process_start_time(proc.pid)
        assert start_time is not None
        assert is_pid_alive(proc.pid, start_time - 600) is False
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_is_pid_alive_treats_start_time_none_as_indeterminate(tmp_path: Path) -> None:
    """Issue #360 criterion #1: a start-time probe failure is not a definitive dead signal.

    When ``get_process_start_time`` returns ``None`` for a process that is still
    alive, ``is_pid_alive`` must return ``True`` (indeterminate) rather than
    treating the worker as dead.
    """
    from unittest.mock import patch

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )
    try:
        with patch("charlie_work.process_utils.get_process_start_time", return_value=None):
            assert is_pid_alive(proc.pid, 123.456) is True
    finally:
        proc.kill()
        proc.wait(timeout=5)


def _capture_popen_call(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace subprocess.Popen with a MagicMock and return the mock."""
    mock = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", mock)
    return mock


def test_popen_worker_routes_creationflags_through_hidden_console_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`popen_worker` is the single chokepoint for creationflags/process-group composition."""
    mock = _capture_popen_call(monkeypatch)
    sentinel_kwargs = {"creationflags": 0xDEADBEEF}

    with patch(
        "charlie_work.process_utils.hidden_console_kwargs",
        return_value=sentinel_kwargs,
    ) as mock_helper:
        popen_worker([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL)

    expected_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    mock_helper.assert_called_once_with(expected_flag)
    assert mock.call_args.kwargs.get("creationflags") == 0xDEADBEEF


def test_popen_worker_combines_existing_creationflags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing creationflags from the caller are merged with the process-group flag."""
    monkeypatch.setattr(subprocess, "Popen", MagicMock())

    with patch("charlie_work.process_utils.hidden_console_kwargs") as mock_helper:
        popen_worker(
            [sys.executable, "-c", "pass"],
            creationflags=0x00000400,
            stdout=subprocess.DEVNULL,
        )

    expected_flag = 0x00000400 | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    mock_helper.assert_called_once_with(expected_flag)


def test_popen_worker_defaults_start_new_session_on_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`popen_worker` defaults POSIX workers to a new session."""
    monkeypatch.setattr(os, "name", "posix")
    mock = _capture_popen_call(monkeypatch)

    popen_worker([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL)

    assert mock.call_args.kwargs.get("start_new_session") is True


def test_popen_worker_omits_start_new_session_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`popen_worker` does not inject start_new_session on Windows by default."""
    monkeypatch.setattr(os, "name", "nt")
    mock = _capture_popen_call(monkeypatch)

    popen_worker([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL)

    assert "start_new_session" not in mock.call_args.kwargs


def test_popen_worker_respects_explicit_start_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers can override the default start_new_session behavior."""
    monkeypatch.setattr(os, "name", "nt")
    mock = _capture_popen_call(monkeypatch)

    popen_worker(
        [sys.executable, "-c", "pass"],
        start_new_session=False,
        stdout=subprocess.DEVNULL,
    )

    assert mock.call_args.kwargs.get("start_new_session") is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_popen_worker_composes_hidden_console_for_worker_spawns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker spawns inherit a hidden console via CREATE_NEW_CONSOLE + SW_HIDE."""
    mock = _capture_popen_call(monkeypatch)

    popen_worker([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL)

    kwargs = mock.call_args.kwargs
    flags = kwargs.get("creationflags", 0)
    assert flags & subprocess.CREATE_NEW_CONSOLE
    assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
    assert not (flags & subprocess.CREATE_NO_WINDOW)
    assert not (flags & subprocess.DETACHED_PROCESS)
    startupinfo = kwargs.get("startupinfo")
    assert startupinfo is not None
    assert startupinfo.wShowWindow == subprocess.SW_HIDE
    assert startupinfo.dwFlags & subprocess.STARTF_USESHOWWINDOW
