from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from charlie_work.process_utils import is_session_stalled, kill_process_tree, sweep_orphan_processes


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
    """Test that sweep_orphan_processes parses PowerShell output correctly."""
    from unittest.mock import patch

    # Mock subprocess.run to return sample PowerShell output
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "1234\n5678\n"
        orphans = sweep_orphan_processes("/some/worktree/path")
        assert orphans == [1234, 5678]


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
