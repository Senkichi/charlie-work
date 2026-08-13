from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from charlie_work.process_utils import (
    find_worker_terminal_status,
    is_pid_alive,
    is_session_stalled,
    kill_process_tree,
    popen_worker,
    start_terminal_status_watcher,
    sweep_orphan_processes,
    worker_terminal_status_path,
    write_worker_terminal_status,
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

    # Spawn a real child process.
    # On POSIX, use start_new_session=True to avoid sharing pytest's process group.
    #
    # The sleep must outlast the whole test body, not just "long enough" (it was
    # 10s, which flaked on CI: this suite runs under `-n 2` on a shared Windows
    # box alongside the live fleet). If the child exits on its own first,
    # `taskkill /T /F /PID` reports the PID as not found and exits outside the
    # (0, 1) codes kill_process_tree accepts, so it returns [] -- which makes the
    # *negative* case below pass vacuously and the positive case fail with a
    # baffling `assert <pid> in []`. The finally block terminates the child
    # regardless, so a long sleep costs nothing.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
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
            from charlie_work.process_utils import get_process_start_time

            # Both assertions below are only meaningful while the child is
            # still alive. If it has already exited, `taskkill /F /PID` exits
            # with a not-found code outside (0, 1), kill_process_tree returns [],
            # and the mismatch case below passes for a reason that has nothing to
            # do with start-time verification -- while the positive case fails as
            # a baffling `assert <pid> in []`. That is precisely the CI flake
            # this test had.
            #
            # Probe with is_pid_alive, NOT proc.poll() and NOT
            # get_process_start_time. Both of those still report "alive" for a
            # child that Popen still holds a handle to: poll() until Python reaps
            # it, and get_process_start_time because a terminated process object
            # stays queryable. Each was tried first, and each sailed straight
            # through a mutation that killed the fixture mid-test.
            assert is_pid_alive(proc.pid), (
                "fixture process was already gone before the mismatch case"
            )

            # kill_process_tree can conservatively skip (return []) when
            # get_process_start_time transiently returns None inside it:
            # OpenProcess can fail under box load even for a child we spawned,
            # and the conservative-skip path returns [] before reaching the
            # start-time comparison. The is_pid_alive probe above does not catch
            # this because the failure happens *inside* kill_process_tree, not
            # before it. This is the second race left open after #1097: the
            # negative case passes vacuously (conservative skip, not mismatch)
            # and the positive case fails as `assert <pid> in []`.
            #
            # The retry distinguishes the two [] causes by re-checking
            # get_process_start_time *after* the call. If it returns None, the
            # skip was a transient query failure (retry). If it returns a value,
            # the [] is the correct result -- start-time mismatch for the
            # negative case, or the process is gone for the positive case.

            # Test 1: Wrong start time should NOT kill
            wrong_start_time = actual_start_time - 1000  # 1000 seconds in the past
            deadline_t1 = time.monotonic() + 10.0
            killed = []
            while time.monotonic() < deadline_t1:
                killed = kill_process_tree(proc.pid, wrong_start_time)
                if killed != []:
                    break  # unexpected kill -- let the assertion below fail
                # [] -- could be start-time mismatch (correct) or transient skip
                if get_process_start_time(proc.pid) is not None:
                    break  # query works, [] is genuinely from mismatch
                time.sleep(0.1)  # transient query failure, retry
            assert killed == [], f"wrong start time must not kill, but got killed={killed}"
            assert is_pid_alive(proc.pid), "mismatched start time must leave the process alive"

            # Test 2: Correct start time should kill
            # Retry while the process is alive: a successful kill makes it dead,
            # and a transient skip leaves it alive, so the loop converges either
            # way. The process sleeps 300s, so it will not exit on its own.
            deadline_t2 = time.monotonic() + 10.0
            killed = []
            while time.monotonic() < deadline_t2 and is_pid_alive(proc.pid):
                killed = kill_process_tree(proc.pid, actual_start_time)
                if proc.pid in killed:
                    break
                time.sleep(0.1)  # transient query failure, retry
            assert proc.pid in killed, (
                f"kill_process_tree did not kill {proc.pid} within retry deadline; "
                f"killed={killed}, alive={is_pid_alive(proc.pid)}"
            )

    finally:
        # Clean up if still alive
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


@pytest.mark.skipif(os.name != "nt", reason="Windows-only: taskkill return-code race")
def test_kill_process_tree_records_kill_when_taskkill_returncode_unexpected() -> None:
    """taskkill can return codes outside (0, 1) even when it kills the process.

    Under CI box load, ``taskkill /F /T /PID`` may exit with a code like 128
    even though it successfully terminated the target.  The old code only
    recorded the PID as killed when ``returncode in (0, 1)``, so the caller
    saw ``killed=[]`` for a dead process — a phantom miss that broke retry
    loops and tests alike.  The fix verifies the process is actually dead
    via ``is_pid_alive`` after taskkill, recording the kill regardless of
    the return code.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=False,
    )
    try:
        import time

        # Wait for the child to be alive before proceeding.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if is_pid_alive(proc.pid):
                break
            time.sleep(0.1)
        if not is_pid_alive(proc.pid):
            pytest.skip("fixture process failed to start")

        from charlie_work.process_utils import get_process_start_time

        actual_start_time = get_process_start_time(proc.pid)
        if actual_start_time is None:
            pytest.skip("could not read fixture process start time")

        # Simulate taskkill returning an unexpected exit code (e.g. 128)
        # while the process is actually dead. We patch subprocess.run to
        # first kill the process for real (so is_pid_alive returns False),
        # then report a non-(0,1) return code.
        def fake_run(*args: Any, **kwargs: Any) -> Any:
            # Only intercept the taskkill call; let everything else through.
            if args and isinstance(args[0], list) and "taskkill" in args[0]:
                # Kill the process for real so is_pid_alive sees it dead.
                proc.terminate()
                proc.wait()
                return subprocess.CompletedProcess(
                    args=args[0], returncode=128, stdout="", stderr=""
                )
            return subprocess.run(*args, **kwargs)

        with patch("charlie_work.process_utils.subprocess.run", side_effect=fake_run):
            with patch("charlie_work.process_utils._enumerate_child_pids", return_value=[]):
                killed = kill_process_tree(proc.pid, actual_start_time)

        assert proc.pid in killed, (
            f"kill_process_tree must record the PID as killed when the process "
            f"is dead even if taskkill returned an unexpected exit code; "
            f"killed={killed}"
        )
    finally:
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


def test_kill_process_tree_self_pid_exempt(monkeypatch: Any) -> None:
    """kill_process_tree returns empty without invoking the platform kill when the target PID is the caller.

    The fleet supervisor reaps stalled workers from within its own process
    image, so a ``kill_process_tree(os.getpid())`` call (e.g. from a
    recycled-PID or bogus-caller case) would terminate it silently. The
    explicit PID self-exemption guard must win even if the process-group
    guard would also block, because a target PID matching the supervisor is
    unambiguous.
    """
    own_pid = 424242
    monkeypatch.setattr("os.getpid", lambda: own_pid, raising=False)

    # Make the process-group guard think the target is in a different group so
    # the only thing preventing self-termination is the explicit PID guard.
    def fake_getpgid(pid: int) -> int:
        return 100 if pid == own_pid else 200

    # Add these on the process_utils os module even if the host os module does
    # not provide them (e.g. Windows), because we force the POSIX branch below.
    monkeypatch.setattr("charlie_work.process_utils.os.getpgid", fake_getpgid, raising=False)

    # Avoid querying the real system for children of a non-existent PID.
    monkeypatch.setattr("charlie_work.process_utils._enumerate_child_pids", lambda _pid: [])

    kill_attempts: list[Any] = []

    def fake_subprocess_run(cmd, **kwargs):
        kill_attempts.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    def fake_killpg(pgid: int, sig: int) -> None:
        kill_attempts.append((pgid, sig))

    monkeypatch.setattr("charlie_work.process_utils.os.killpg", fake_killpg, raising=False)

    # Force the POSIX branch so both platform kill paths are gated by the guard.
    monkeypatch.setattr("charlie_work.process_utils.os.name", "posix")

    killed = kill_process_tree(own_pid)
    assert killed == []
    assert kill_attempts == []


def test_kill_process_tree_enumerates_children() -> None:
    """Test that kill_process_tree enumerates and includes child PIDs."""
    # Spawn a real parent process that will spawn a child
    # On POSIX, use start_new_session=True to avoid sharing pytest's process group
    # The parent imports ``sys`` before referencing ``sys.executable``; without it
    # the parent crashes with NameError before spawning the child, and the test
    # silently no-ops via the empty-children skip path (a false positive).
    #
    # The sleeps must outlast the enumeration deadline below by a wide margin.
    # They previously matched it exactly (both 10s), so under full-suite CPU
    # contention a child that took several seconds to become visible left the
    # parent with almost no lifetime remaining: it exited between the retry
    # loop and ``kill_process_tree``, ``taskkill`` found nothing to kill, and
    # the test failed on an empty ``killed`` list rather than skipping. Both
    # processes are terminated explicitly below and in ``finally``, so a long
    # sleep costs nothing -- nothing here waits for them to expire on their own.
    parent_proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
            "time.sleep(120)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )

    # Bound outside the ``try`` so the ``finally`` cleanup can always read it,
    # including when a skip fires before the retry loop assigns it.
    child_pids: list[int] = []

    try:
        import time

        from charlie_work.process_utils import _enumerate_child_pids

        # Poll for the child to appear instead of sampling once. Under full-suite
        # CPU contention the child may not be visible to a single CIM/proc
        # snapshot immediately, and on Windows the enumeration itself may
        # transiently time out and return ``[]`` (swallowed by design). A bounded
        # retry distinguishes "child not yet visible" (keep waiting) from "no
        # child was spawned / enumeration failed" (skip), avoiding a spurious
        # assertion failure on unrelated PRs. See issue #608.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if parent_proc.poll() is not None:
                pytest.skip(f"parent process {parent_proc.pid} exited before spawning a child")
            child_pids = _enumerate_child_pids(parent_proc.pid)
            if child_pids:
                break
            time.sleep(0.25)

        if not child_pids:
            pytest.skip(
                f"no child of parent {parent_proc.pid} became visible within "
                f"the deadline; enumeration may have failed under load"
            )

        # Re-check liveness immediately before the kill. The retry loop above
        # only proves the parent was alive when enumeration started; a parent
        # that died in between leaves nothing for the platform kill to find and
        # returns an empty list, which would fail the assertion below for a
        # reason that has nothing to do with child enumeration. Skipping here
        # keeps that failure mode legible instead of surfacing as
        # ``assert <pid> in []``.
        if parent_proc.poll() is not None:
            pytest.skip(
                f"parent process {parent_proc.pid} exited between enumeration and kill; "
                f"cannot assert on the killed set"
            )

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

        # ``parent_proc.terminate()`` does not reap the grandchild, and the
        # sleeps are long enough now that a skipped run would otherwise leave
        # it resident on the runner for two minutes. Reap any child the
        # enumeration found; on the assertion path they are already dead, so
        # this is a no-op there. Best-effort by design -- a failure to clean up
        # a stray sleep must not mask the test's own result.
        for stray_pid in child_pids:
            try:
                os.kill(stray_pid, signal.SIGTERM)
            except OSError:
                pass


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


# --- issue #773: durable worker terminal-status record -----------------------


def test_worker_terminal_status_path_naming(tmp_path: Path) -> None:
    """The terminal-status path follows issue-<n>.<suffix>.terminal.json."""
    path = worker_terminal_status_path(tmp_path, 207, "claude")
    assert path == tmp_path / "issue-207.claude.terminal.json"


def test_write_worker_terminal_status_round_trips_and_is_atomic(tmp_path: Path) -> None:
    """Writing a terminal-status record is atomic (no leftover .tmp) and round-trips."""
    path = tmp_path / "issue-1.claude.terminal.json"
    write_worker_terminal_status(
        path,
        pid=4242,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:05:00Z",
        duration_seconds=300.0,
    )

    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "pid": 4242,
        "exit_code": 0,
        "started_at": "2026-07-30T00:00:00Z",
        "ended_at": "2026-07-30T00:05:00Z",
        "duration_seconds": 300.0,
    }


def test_write_worker_terminal_status_round_trips_worker_outcome(tmp_path: Path) -> None:
    """Issue #935: terminal status preserves the worker's push/PR-failure signal."""
    path = tmp_path / "issue-1.claude.terminal.json"
    worker_outcome = {"push_succeeded": True, "pr_created": False, "error": "gh unauthenticated"}
    write_worker_terminal_status(
        path,
        pid=4242,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:05:00Z",
        duration_seconds=300.0,
        worker_outcome=worker_outcome,
    )

    record = find_worker_terminal_status(tmp_path, 1)
    assert record is not None
    assert record["worker_outcome"] == worker_outcome


def test_write_worker_terminal_status_creates_missing_parent(tmp_path: Path) -> None:
    """The sessions directory does not need to pre-exist."""
    path = tmp_path / "nested" / "issue-1.claude.terminal.json"
    write_worker_terminal_status(
        path,
        pid=1,
        exit_code=1,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:00:01Z",
        duration_seconds=1.0,
    )
    assert path.exists()


def test_find_worker_terminal_status_returns_none_when_absent(tmp_path: Path) -> None:
    """No terminal record for the issue -> None (legacy/fallback path)."""
    assert find_worker_terminal_status(tmp_path, 999) is None


def test_find_worker_terminal_status_returns_none_when_dir_missing(tmp_path: Path) -> None:
    """A sessions dir that does not exist at all is handled without raising."""
    assert find_worker_terminal_status(tmp_path / "does-not-exist", 1) is None


def test_find_worker_terminal_status_reads_written_record(tmp_path: Path) -> None:
    """A record written via write_worker_terminal_status is found by issue number."""
    path = worker_terminal_status_path(tmp_path, 55, "claude")
    write_worker_terminal_status(
        path,
        pid=100,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:01:00Z",
        duration_seconds=60.0,
    )

    record = find_worker_terminal_status(tmp_path, 55)
    assert record is not None
    assert record["pid"] == 100
    assert record["exit_code"] == 0
    assert record["duration_seconds"] == 60.0


def test_find_worker_terminal_status_picks_most_recent_by_mtime(tmp_path: Path) -> None:
    """When multiple adapter suffixes wrote a record, the newest mtime wins."""
    older = worker_terminal_status_path(tmp_path, 7, "claude")
    newer = worker_terminal_status_path(tmp_path, 7, "api")

    write_worker_terminal_status(
        older,
        pid=1,
        exit_code=1,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:00:01Z",
        duration_seconds=1.0,
    )
    write_worker_terminal_status(
        newer,
        pid=2,
        exit_code=0,
        started_at="2026-07-30T00:01:00Z",
        ended_at="2026-07-30T00:01:02Z",
        duration_seconds=2.0,
    )
    # Force an unambiguous mtime ordering regardless of filesystem timestamp
    # resolution (some Windows filesystems only resolve mtime to ~2 seconds).
    old_time = (datetime.now(UTC) - timedelta(minutes=10)).timestamp()
    os.utime(older, (old_time, old_time))

    record = find_worker_terminal_status(tmp_path, 7)
    assert record is not None
    assert record["pid"] == 2
    assert record["exit_code"] == 0


def test_find_worker_terminal_status_skips_corrupt_json(tmp_path: Path) -> None:
    """A corrupt record for one adapter suffix does not hide a valid one."""
    corrupt = worker_terminal_status_path(tmp_path, 8, "claude")
    valid = worker_terminal_status_path(tmp_path, 8, "api")

    corrupt.write_text("{not valid json", encoding="utf-8")
    write_worker_terminal_status(
        valid,
        pid=3,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:00:05Z",
        duration_seconds=5.0,
    )

    record = find_worker_terminal_status(tmp_path, 8)
    assert record is not None
    assert record["pid"] == 3


def test_start_terminal_status_watcher_records_exit_code_without_blocking(
    tmp_path: Path,
) -> None:
    """The watcher thread captures the real exit code and does not block the caller."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(1); raise SystemExit(7)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )
    path = tmp_path / "issue-1.claude.terminal.json"

    import time as time_module

    before = time_module.monotonic()
    thread = start_terminal_status_watcher(proc, path)
    call_duration = time_module.monotonic() - before

    # start_terminal_status_watcher must return immediately -- the subprocess
    # sleeps for 1 second before exiting, so a blocking implementation would
    # make this call take >= 1 second.
    assert call_duration < 0.5

    thread.join(timeout=5)
    assert not thread.is_alive()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pid"] == proc.pid
    assert payload["exit_code"] == 7
    assert payload["duration_seconds"] >= 1.0
