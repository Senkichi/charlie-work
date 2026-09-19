"""Worker liveness: ``is_worker_alive`` pid-recycling guards,
``_posix_stat`` comm parsing, process-start-time capture, and the
worker-census log line.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

from _claude_adapter_fixtures import (
    _fake_claude_script,
    _install_fake_create_worktree,
)

from charlie_work.claude_code import (
    ClaudeWorkerRecord,
    is_worker_alive,
    launch_claude_worker,
)


def test_log_worker_census_emits_alive_worker_with_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Issue #646: the per-pass census must actually surface a real alive worker.

    Regression guard against a census that only ever exercises the empty-
    sessions-dir path (n_alive=0): writes a real sidecar for a genuinely
    live process (matching pid + process_start_time, like production
    sidecars) and asserts the emitted INFO line names its worktree, pid,
    and resolved xdist_cap -- proving `cap` is populated end-to-end rather
    than silently `None`.
    """
    from charlie_work.claude_code import _get_process_start_time
    from charlie_work.workflow import _log_worker_census

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        start_time = _get_process_start_time(process.pid)
        assert start_time is not None

        record = ClaudeWorkerRecord(
            issue_number=646,
            branch="agent/issue-646-census",
            worktree_path=str(tmp_path / "worktrees" / "issue-646-census"),
            prompt_path="p.md",
            command=("x",),
            pid=process.pid,
            started_at="2026-01-01T00:00:00Z",
            log_path="log.txt",
            process_start_time=start_time,
            xdist_cap="2",
        )
        (sessions_dir / "issue-646.claude.json").write_text(
            json.dumps(record.to_dict()), encoding="utf-8"
        )

        # Issue #1317: _log_worker_census moved (verbatim) to
        # dead_worker_reap.py -- its `logging.getLogger(__name__)` call
        # resolves to that module's own real name, not workflow.py's.
        with caplog.at_level("INFO", logger="charlie_work.dead_worker_reap"):
            _log_worker_census(sessions_dir)

        [census_record] = [r for r in caplog.records if "worker census" in r.message]
        assert "n_alive=1" in census_record.message
        assert f"pid={process.pid}" in census_record.message
        assert "worktree=" in census_record.message and "issue-646-census" in census_record.message
        assert "cap=2" in census_record.message
    finally:
        process.kill()
        process.wait(timeout=5)


def test_is_worker_alive_reflects_real_process(tmp_path: Path) -> None:
    """Mirror of test_is_session_alive_reflects_real_process from test_devin_shell.py."""
    # Spawn a short-lived process
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        alive_record = ClaudeWorkerRecord(
            issue_number=1,
            branch="agent/issue-1",
            worktree_path="/tmp/wt/issue-1",
            prompt_path="p.md",
            command=("x",),
            pid=process.pid,
            started_at="2026-01-01T00:00:00Z",
            log_path="log.txt",
        )
        assert is_worker_alive(alive_record) is True
    finally:
        process.kill()
        process.wait(timeout=5)

    # Regression guard for the Windows os.kill(pid, 0) trap: that call keeps
    # reporting a reaped PID as alive indefinitely (verified empirically —
    # see is_worker_alive's docstring), so this must be an exact
    # post-wait() assertion, not a "poll until it settles" retry loop.
    dead_record = ClaudeWorkerRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/wt/issue-1",
        prompt_path="p.md",
        command=("x",),
        pid=process.pid,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
    )
    assert is_worker_alive(dead_record) is False

    # pid=None case (never launched)
    none_record = ClaudeWorkerRecord(
        issue_number=2,
        branch="agent/issue-2",
        worktree_path="/tmp/wt/issue-2",
        prompt_path="p2.md",
        command=("y",),
        pid=None,
        started_at="2026-01-01T00:00:00Z",
        log_path="log2.txt",
    )
    assert is_worker_alive(none_record) is False


def test_is_worker_alive_rejects_pid_recycling_mismatched_start_time(tmp_path: Path) -> None:
    """A record with an alive PID but mismatched start time is treated as dead.

    This prevents false positives from PID recycling: if the OS has reused the PID
    for a different process, the start time will not match.
    """
    from charlie_work.claude_code import _get_process_start_time

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

        record = ClaudeWorkerRecord(
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
        assert is_worker_alive(record) is False
    finally:
        process.kill()
        process.wait(timeout=5)


def test_is_worker_alive_accepts_matching_start_time(tmp_path: Path) -> None:
    """A record with matching start time is counted as live."""
    from charlie_work.claude_code import _get_process_start_time

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

        record = ClaudeWorkerRecord(
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
        assert is_worker_alive(record) is True
    finally:
        process.kill()
        process.wait(timeout=5)


def test_is_worker_alive_legacy_record_fallback() -> None:
    """Legacy records without process_start_time fall back to pid-only liveness.

    This preserves backward compatibility for old sidecar files.
    """
    # Use the test process's own PID (guaranteed to be alive)
    record = ClaudeWorkerRecord(
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
    assert is_worker_alive(record) is True


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


def test_is_worker_alive_probe_none_treats_indeterminate_as_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #360 criterion #1: a start-time probe failure is not a definitive dead signal.

    When ``get_process_start_time`` returns ``None`` for a live PID, ``is_worker_alive``
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

        record = ClaudeWorkerRecord(
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
        assert is_worker_alive(record) is True
    finally:
        process.kill()
        process.wait(timeout=5)


def test_launch_captures_process_start_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The launch path must capture process_start_time at spawn time.

    This test goes through the real launch path and asserts the resulting record's
    process_start_time is not None. Mutation gate: forcing spawn capture to None MUST fail it.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    # The record must have a captured process_start_time
    assert record.process_start_time is not None, (
        "launch_claude_worker must capture process_start_time at spawn time"
    )
    assert isinstance(record.process_start_time, float), (
        "process_start_time must be a float (Unix timestamp)"
    )
