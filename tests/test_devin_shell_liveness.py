"""Probe/liveness tests for the devin-shell adapter.

Split out of ``tests/test_devin_shell.py`` (issue #1542, Track-1 pilot):
``probe_devin`` binary checks, ``is_session_alive`` PID + start-time
matching (including PID-recycling rejection and the indeterminate-probe
fallback), and the ``/proc/<pid>/stat`` comm-parsing unit tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from _devin_shell_fixtures import _FAKE_DEVIN_VERSION, _write_fake_devin

from charlie_work.devin_shell import (
    SessionRecord,
    is_session_alive,
    probe_devin,
)


def test_probe_devin_ok_for_zero_exit_fake(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_VERSION)

    result = probe_devin(repo_root, command=(sys.executable, str(script)))

    assert result.ok is True
    assert result.returncode == 0
    assert "devin-fake" in result.stdout


def test_probe_devin_not_ok_for_missing_binary(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    result = probe_devin(repo_root, command=("definitely-not-a-real-devin-binary-xyz",))

    assert result.ok is False
    assert result.error is not None


def test_is_session_alive_reflects_real_process(tmp_path: Path) -> None:
    script = _write_fake_devin(tmp_path, "import time; time.sleep(2)")
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        alive_record = SessionRecord(
            issue_number=1,
            branch="agent/issue-1",
            worktree_path="/tmp/wt/issue-1",
            prompt_path="p.md",
            command=("x",),
            pid=process.pid,
            started_at="2026-01-01T00:00:00Z",
            log_path="log.txt",
        )
        assert is_session_alive(alive_record) is True
    finally:
        process.kill()
        process.wait(timeout=5)

    # Regression guard for the Windows os.kill(pid, 0) trap: that call keeps
    # reporting a reaped PID as alive indefinitely (verified empirically —
    # see is_session_alive's docstring), so this must be an exact
    # post-wait() assertion, not a "poll until it settles" retry loop.
    dead_record = SessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/wt/issue-1",
        prompt_path="p.md",
        command=("x",),
        pid=process.pid,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
    )
    assert is_session_alive(dead_record) is False


def test_is_session_alive_false_for_none_pid() -> None:
    record = SessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="",
        prompt_path="p.md",
        command=("x",),
        pid=None,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
        error="devin not found",
    )

    assert is_session_alive(record) is False


def test_is_session_alive_false_for_implausible_pid() -> None:
    # A PID this large was never handed out by Popen; OpenProcess should
    # simply fail to find it (and on POSIX os.kill raises ProcessLookupError).
    # Confirms the "not found" path returns False without raising.
    record = SessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="",
        prompt_path="p.md",
        command=("x",),
        pid=999_999_999,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
    )

    assert is_session_alive(record) is False


def test_is_session_alive_rejects_pid_recycling_mismatched_start_time(tmp_path: Path) -> None:
    """A record with an alive PID but mismatched start time is treated as dead.

    This prevents false positives from PID recycling: if the OS has reused the PID
    for a different process, the start time will not match.
    """
    from charlie_work.devin_shell import _get_process_start_time

    # Spawn a short-lived process to get a valid PID
    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(2)", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Get the actual start time of this process
        actual_start_time = _get_process_start_time(process.pid)
        assert actual_start_time is not None

        # Create a record with a deliberately wrong start time (10 minutes ago)
        # This simulates a recycled PID
        wrong_start_time = actual_start_time - 600  # 10 minutes in the past

        record = SessionRecord(
            issue_number=1,
            branch="agent/issue-1",
            worktree_path="/tmp/wt/issue-1",
            prompt_path="p.md",
            command=("x",),
            pid=process.pid,
            started_at="2026-01-01T00:00:00Z",
            log_path="log.txt",
            process_start_time=wrong_start_time,
        )

        # Should return False because start time doesn't match
        assert is_session_alive(record) is False
    finally:
        process.kill()
        process.wait(timeout=5)


def test_is_session_alive_accepts_matching_start_time(tmp_path: Path) -> None:
    """A record with matching start time is counted as live."""
    from charlie_work.devin_shell import _get_process_start_time

    # Spawn a short-lived process
    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(2)", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Get the actual start time of this process
        actual_start_time = _get_process_start_time(process.pid)
        assert actual_start_time is not None

        record = SessionRecord(
            issue_number=1,
            branch="agent/issue-1",
            worktree_path="/tmp/wt/issue-1",
            prompt_path="p.md",
            command=("x",),
            pid=process.pid,
            started_at="2026-01-01T00:00:00Z",
            log_path="log.txt",
            process_start_time=actual_start_time,
        )

        # Should return True because start time matches
        assert is_session_alive(record) is True
    finally:
        process.kill()
        process.wait(timeout=5)


def test_is_session_alive_legacy_record_fallback() -> None:
    """Legacy records without process_start_time fall back to pid-only liveness.

    This preserves backward compatibility for old sidecar files.
    """
    # Use the test process's own PID (guaranteed to be alive)
    record = SessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/wt/issue-1",
        prompt_path="p.md",
        command=("x",),
        pid=os.getpid(),
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
        process_start_time=None,  # Legacy record
    )

    # Should return True using pid-only fallback
    assert is_session_alive(record) is True


def test_posix_stat_parse_with_spaces_in_comm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit test for /proc/<pid>/stat parsing with comm containing spaces and closing paren.

    This test exercises the real parse_proc_stat_starttime function (used by both adapters)
    by monkeypatching the /proc read. The comm field MUST contain spaces and a closing paren
    (e.g., '(tmux: server)') to prove the last-')' splitting logic works correctly.
    """
    from charlie_work.process_utils import parse_proc_stat_starttime

    # Format: pid (comm) state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt utime stime cutime cstime priority nice num_threads itrealvalue starttime vsize rss ...
    # starttime is at index 19 (0-indexed) after splitting on ')'
    # This example has comm="(tmux: server)" which contains spaces
    stat_line = "1234 (tmux: server) S 1 1234 1234 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000 31000 32000 33000 34000 35000 36000 37000 38000 39000 40000 41000 42000 43000 44000 45000 46000 47000 48000 49000 50000 51000 52000"

    # Call the real parser function
    starttime_ticks = parse_proc_stat_starttime(stat_line)
    assert starttime_ticks == 100, f"Expected starttime to be 100, got {starttime_ticks}"


def test_posix_stat_parse_with_embedded_paren_in_comm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit test for /proc/<pid>/stat parsing with comm containing embedded ')'.

    This test proves that rpartition (last-')' split) is required: a comm like
    '(tmux: (0) server)' contains an embedded ')', which would break split(')')
    by shifting all field offsets. The parser must split on the LAST ')' to correctly
    extract the starttime field.
    """
    from charlie_work.process_utils import parse_proc_stat_starttime

    # Comm contains embedded ')': (tmux: (0) server)
    # Using split(')') would split on the first ')', giving wrong field offsets
    # Using rpartition(')') splits on the last ')', giving correct field offsets
    stat_line = "1234 (tmux: (0) server) S 1 1234 1234 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000 31000 32000 33000 34000 35000 36000 37000 38000 39000 40000 41000 42000 43000 44000 45000 46000 47000 48000 49000 50000 51000 52000"

    # Call the real parser function
    starttime_ticks = parse_proc_stat_starttime(stat_line)
    assert starttime_ticks == 100, f"Expected starttime to be 100, got {starttime_ticks}"


def test_is_session_alive_probe_none_treats_indeterminate_as_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #360 criterion #1: a start-time probe failure is not a definitive dead signal.

    When ``get_process_start_time`` returns ``None`` for a live PID, ``is_session_alive``
    treats the liveness signal as indeterminate and returns ``True`` rather than
    reaping a potentially-live worker.
    """
    import charlie_work.process_utils as process_utils

    # Spawn a short-lived process to get a valid PID
    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(2)", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Simulate a start-time probe failure while the process is still alive.
        monkeypatch.setattr(process_utils, "get_process_start_time", lambda pid: None)

        record = SessionRecord(
            issue_number=1,
            branch="agent/issue-1",
            worktree_path="/tmp/wt/issue-1",
            prompt_path="p.md",
            command=("x",),
            pid=process.pid,  # PID is alive
            started_at="2026-01-01T00:00:00Z",
            log_path="log.txt",
            process_start_time=123.456,  # Record has a start time
        )

        # Should return True because the probe was indeterminate, not definitive dead.
        assert is_session_alive(record) is True
    finally:
        process.kill()
        process.wait(timeout=5)
