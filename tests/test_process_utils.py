from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.process_utils import is_session_stalled, kill_process_tree


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
    import sys
    import subprocess
    from unittest.mock import patch

    # Spawn a real child process
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Get the actual start time
        from charlie_work.process_utils import parse_proc_stat_starttime
        import os
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


def test_kill_process_tree_own_group_guard_posix() -> None:
    """Test that kill_process_tree refuses to kill its own process group on POSIX."""
    import os
    import sys

    # This test is only for POSIX systems
    if sys.platform == "win32":
        return  # Skip on Windows

    # Try to kill our own process group - should refuse
    own_pid = os.getpid()
    os.getpgid(own_pid)

    # Attempt to kill our own process group
    killed = kill_process_tree(own_pid)

    # Should return empty list because we refused to kill our own group
    assert killed == []


def test_kill_process_tree_enumerates_children() -> None:
    """Test that kill_process_tree enumerates and includes child PIDs."""
    from unittest.mock import patch, MagicMock

    # Mock _enumerate_child_pids to return fake child PIDs
    with patch("charlie_work.process_utils._enumerate_child_pids", return_value=[12345, 67890]):
        # Mock the actual kill to avoid killing real processes
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            # Kill a fake PID
            killed = kill_process_tree(99999, expected_start_time=None)

            # Check that the parent PID is in the killed list
            assert 99999 in killed

            # Check that the enumerated child PIDs are in the killed list
            assert 12345 in killed
            assert 67890 in killed
