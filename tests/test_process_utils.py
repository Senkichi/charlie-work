from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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


def test_is_session_stalled_with_generic_error_marker(tmp_path: Path) -> None:
    """Test that a session is stalled when log ends with generic Error: marker."""
    log_file = tmp_path / "test.log"
    log_file.write_text("some log content\nError: something failed\n", encoding="utf-8")

    is_stalled, last_line = is_session_stalled(log_file, stall_threshold_minutes=20)
    assert is_stalled is True
    assert last_line == "Error: something failed"


def test_is_session_stalled_healthy_session(tmp_path: Path) -> None:
    """Test that a healthy session (recent log, no terminal marker) is not stalled."""
    log_file = tmp_path / "test.log"
    log_file.write_text("working on issue\nmaking progress\n", encoding="utf-8")

    # Fresh mtime, no terminal error
    is_stalled, last_line = is_session_stalled(log_file, stall_threshold_minutes=20)
    assert is_stalled is False
    assert last_line == "making progress"


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
    # Use a PID that likely doesn't exist
    killed = kill_process_tree(999999)
    # Should return empty list or just the PID if the kill attempt was made
    # The important thing is it doesn't raise an exception
    assert isinstance(killed, list)
