from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pytest

from charlie_work.process_utils import (
    is_pid_alive,
    kill_process_tree,
    run_captured,
)
from charlie_work.subprocess_runner import RunResult


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
        # while the process is actually dead. We patch run_captured to
        # first kill the process for real (so is_pid_alive returns False),
        # then report a non-(0,1) return code.
        def fake_run_captured(*args: Any, **kwargs: Any) -> RunResult:
            # Only intercept the taskkill call; let everything else through.
            if args and isinstance(args[0], list) and "taskkill" in args[0]:
                # Kill the process for real so is_pid_alive sees it dead.
                proc.terminate()
                proc.wait()
                return RunResult(returncode=128, stdout="", stderr="", error="command exited 128")
            return run_captured(*args, **kwargs)

        with patch("charlie_work.process_utils.run_captured", side_effect=fake_run_captured):
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


def test_kill_process_tree_enumerates_children(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that kill_process_tree enumerates children and kills the process tree.

    The assertion strategy decouples from ``kill_process_tree``'s internal
    enumeration *result*, which is best-effort by design (``_enumerate_child_pids``
    swallows transient CIM/proc failures and returns ``[]``). Under full-suite
    load the internal single-shot enumeration can fail even when the test's
    setup enumeration succeeded, so asserting ``child_pid in killed`` couples
    two independent enumerations and flakes (#608, #681, #1064).

    Instead we verify three things:
    1. The parent PID is in ``killed`` (the kill was reported).
    2. ``kill_process_tree`` called ``_enumerate_child_pids`` (enumeration was
       attempted — structural, via spy). Catches a regression that removes the
       enumeration call.
    3. The child the parent *reports* is dead (behavioral). The report is the
       ground truth of which process is the child: a ``ParentProcessId`` match
       can also name a stranger holding a pid recycled from a dead parent
       (stale ppid; Windows never reparents — the same recycled-pid family as
       #2057), and asserting on an enumerated-but-unrelated pid was a source of
       the "child still alive" flake (#2059). Liveness is checked with a
       start-time fingerprint so a pid recycled onto a new process after its
       death cannot read alive either.
    """
    import logging
    import threading
    import time

    import charlie_work.process_utils as _pu

    # Spawn a real parent process that will spawn a child and print the child's
    # pid on stdout -- the test then asserts on that pid, not on whichever pids
    # a ppid match happened to return. On POSIX use start_new_session=True to
    # avoid sharing pytest's process group.
    #
    # ``sys._base_executable`` (fall back to ``sys.executable``) skips the
    # uv-managed venv trampoline: under uv, ``sys.executable`` is a trampoline
    # whose spawned interpreter is a *child* of the pid ``Popen`` returns, so
    # the real parent was one level down and the sleeper a grandchild -- extra
    # layers inside the enumerate->kill window this test exists to exercise.
    # Spawning the base interpreter directly gives the intended two-level tree.
    #
    # The sleeps must outlast the setup deadline below by a wide margin. They
    # previously matched it exactly (both 10s), so under full-suite CPU
    # contention a child that took several seconds to become visible left the
    # parent with almost no lifetime remaining: it exited between enumeration
    # and ``kill_process_tree``, and the test failed on an empty ``killed``
    # list rather than skipping. Both processes are terminated explicitly
    # below and in ``finally``, so a long sleep costs nothing.
    python = getattr(sys, "_base_executable", sys.executable)
    parent_proc = subprocess.Popen(
        [
            python,
            "-c",
            "import subprocess, sys, time; "
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
            "print(f'CHILD {child.pid}', flush=True); "
            "time.sleep(120)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=(os.name != "nt"),
    )

    # Bound outside the ``try`` so the ``finally`` cleanup can always read it,
    # including when a skip fires before the setup assigns it.
    child_pid: int | None = None

    try:
        # Read the child's pid from the parent's stdout. A thread is required:
        # ``readline`` blocks until the parent prints, and the parent may take
        # seconds to spawn the child under full-suite CPU contention. A bounded
        # join distinguishes "not yet printed" from "parent died / pipe broken".
        reported: list[int] = []

        def _read_reported_child() -> None:
            assert parent_proc.stdout is not None
            parts = parent_proc.stdout.readline().split()
            if len(parts) == 2 and parts[0] == "CHILD" and parts[1].isdigit():
                reported.append(int(parts[1]))

        reader = threading.Thread(target=_read_reported_child, daemon=True)
        reader.start()
        reader.join(timeout=10.0)
        if not reported:
            if parent_proc.poll() is not None:
                pytest.skip(f"parent process {parent_proc.pid} exited before reporting a child")
            pytest.skip(
                f"parent process {parent_proc.pid} did not report a child pid within the deadline"
            )
        child_pid = reported[0]

        # Fingerprint the reported child so the post-kill liveness check is
        # pinned to this exact process: a pid recycled onto a new process after
        # the real child's death must not satisfy "still alive" OR "alive at
        # assert time" (the recycled-pid failure shape, #2057/#2059).
        from charlie_work.process_utils import get_process_start_time

        child_start = get_process_start_time(child_pid)
        if child_start is None:
            pytest.skip(f"could not read reported child {child_pid}'s start time")
        if parent_proc.poll() is not None or not is_pid_alive(child_pid, child_start):
            pytest.skip(
                f"fixture died before the kill (parent={parent_proc.poll()}, child={child_pid})"
            )

        # Spy on ``_enumerate_child_pids`` to verify ``kill_process_tree`` calls
        # it (structural assertion). The patched module attribute is resolved
        # by ``kill_process_tree`` at call time -- inside this test it is only
        # used for ``finally`` cleanup, which is unaffected by the monkeypatch.
        # The internal enumeration's *result* is best-effort (can return ``[]``
        # under load), so we assert on the *call*, not the result -- a
        # regression that removes the enumeration call fails here.
        enumerate_calls: list[int] = []
        _real_enumerate = _pu._enumerate_child_pids

        def _spy_enumerate(spy_pid: int) -> list[int]:
            enumerate_calls.append(spy_pid)
            return _real_enumerate(spy_pid)

        monkeypatch.setattr(_pu, "_enumerate_child_pids", _spy_enumerate)

        # Kill the parent process tree. ``caplog`` captures kill_process_tree's
        # warnings so a future failure is self-diagnosing: a "refused pid"
        # warning exposes the #2057-style stale-parent-PID refusal, a taskkill
        # failure surfaces the platform kill, and the *absence* of warnings
        # points at the tree-kill-reach path itself.
        with caplog.at_level(logging.WARNING, logger="charlie_work"):
            killed = kill_process_tree(parent_proc.pid, expected_start_time=None)
        warnings = "; ".join(r.getMessage() for r in caplog.records) or "<none>"

        # 1. The parent PID must be in the killed list (kill was reported).
        assert parent_proc.pid in killed, (
            f"parent {parent_proc.pid} not in killed={killed}; "
            f"kill_process_tree warnings: {warnings}"
        )

        # 2. ``kill_process_tree`` must call ``_enumerate_child_pids`` with the
        #    parent PID (enumeration was attempted). This catches a regression
        #    that removes or skips the enumeration call -- the structural
        #    guarantee the test name promises. We do NOT assert on the result
        #    (which children appear in ``killed``) because that result is
        #    best-effort and transiently empty under load (#1064).
        assert parent_proc.pid in enumerate_calls, (
            f"kill_process_tree did not call _enumerate_child_pids for "
            f"parent {parent_proc.pid}; enumeration was skipped; "
            f"kill_process_tree warnings: {warnings}"
        )

        # 3. Behavioral assertion: the reported child must be actually dead.
        #    ``taskkill /T`` (Windows) or ``killpg`` (POSIX) kills the tree
        #    regardless of enumeration, so child liveness is the reliable signal
        #    that the tree kill worked. The fingerprint pins identity so a
        #    recycled pid cannot read alive; a bounded wait accommodates
        #    kill-signal propagation latency under load.
        child_dead = False
        child_deadline = time.monotonic() + 5.0
        while time.monotonic() < child_deadline:
            if not is_pid_alive(child_pid, child_start):
                child_dead = True
                break
            time.sleep(0.1)
        assert child_dead, (
            f"child {child_pid} still alive after kill_process_tree; "
            f"tree kill did not reach the child; killed={killed}; "
            f"parent_poll={parent_proc.poll()}; "
            f"kill_process_tree warnings: {warnings}"
        )
    finally:
        # Clean up if still alive
        if parent_proc.poll() is None:
            parent_proc.terminate()
            parent_proc.wait()

        # ``parent_proc.terminate()`` does not reap the grandchild, and the
        # sleeps are long enough now that a skipped run would otherwise leave
        # it resident on the runner for two minutes. Reap the reported child;
        # on the assertion path it is already dead, so this is a no-op there.
        # Best-effort by design -- a failure to clean up a stray sleep must not
        # mask the test's own result.
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except OSError:
                pass


def _stub_kill_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    children: list[int],
    alive: dict[int, bool],
    starts: dict[int, float | None],
    taskkill_calls: list[list[str]],
) -> None:
    """Install the process_utils seams ``kill_process_tree`` touches.

    ``children`` is the enumeration result; ``starts`` is pid -> start-time
    fingerprint (what ``get_process_start_time`` returns). ``alive`` is pid ->
    liveness as ``is_pid_alive`` sees it. ``taskkill_calls`` records every
    command handed to ``run_captured``; the default stub models the platform
    faithfully -- a ``/T`` kill takes the target's enumerated children with it.
    """

    import charlie_work.process_utils as _pu

    def fake_enumerate(pid: int) -> list[int]:
        return list(children)

    def fake_start(pid: int) -> float | None:
        return starts.get(pid)

    def fake_alive(pid: int, expected_start_time: float | None = None) -> bool:
        if not alive.get(pid, False):
            return False
        if expected_start_time is not None:
            current = starts.get(pid)
            return current is not None and abs(current - expected_start_time) <= 1.0
        return True

    def fake_run_captured(command: list[str], **_kwargs: Any) -> RunResult:
        taskkill_calls.append(list(command))
        # taskkill /T kills the tree: the target plus its enumerated children.
        try:
            killed_pid = int(command[-1])
        except (ValueError, IndexError):
            killed_pid = -1
        if killed_pid in alive:
            alive[killed_pid] = False
            if "/T" in command:
                for child in children:
                    alive[child] = False
        return RunResult(returncode=0, stdout="", stderr="", error=None)

    monkeypatch.setattr(_pu, "_enumerate_child_pids", fake_enumerate)
    monkeypatch.setattr(_pu, "get_process_start_time", fake_start)
    monkeypatch.setattr(_pu, "is_pid_alive", fake_alive)
    monkeypatch.setattr(_pu, "run_captured", fake_run_captured)
    monkeypatch.setattr(_pu.os, "name", "nt", raising=False)


def test_kill_process_tree_reaps_survivor_when_root_died_before_taskkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root dies between enumeration and taskkill: the tree kill kills nothing
    (taskkill exits 128 on a dead root -- verified platform behavior), so a
    verified-alive child is reaped individually and then recorded (#2059)."""
    import charlie_work.process_utils as _pu

    root, child = 60001, 60002
    children = [child]
    alive = {root: False, child: True}
    starts = {root: 1000.0, child: 2000.0}
    taskkill_calls: list[list[str]] = []
    _stub_kill_env(
        monkeypatch,
        children=children,
        alive=alive,
        starts=starts,
        taskkill_calls=taskkill_calls,
    )

    # Root already dead by the time taskkill runs: "not found", rc=128, and the
    # /T traversal has no live root to walk -- the child is untouched.
    def dead_root_run_captured(command: list[str], **_kwargs: Any) -> RunResult:
        taskkill_calls.append(list(command))
        if int(command[-1]) == root:
            return RunResult(returncode=128, stdout="", stderr="not found", error="exit 128")
        alive[int(command[-1])] = False
        return RunResult(returncode=0, stdout="", stderr="", error=None)

    monkeypatch.setattr(_pu, "run_captured", dead_root_run_captured)

    killed = kill_process_tree(root, expected_start_time=None)

    assert killed == [root, child]
    assert ["taskkill", "/T", "/F", "/PID", str(child)] in taskkill_calls


def test_kill_process_tree_records_child_dead_with_root_without_extra_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that died with the tree kill is recorded without an individual kill."""
    root, child = 60011, 60012
    children = [child]
    alive = {root: True, child: True}
    starts = {root: 1000.0, child: 2000.0}
    taskkill_calls: list[list[str]] = []
    _stub_kill_env(
        monkeypatch,
        children=children,
        alive=alive,
        starts=starts,
        taskkill_calls=taskkill_calls,
    )

    killed = kill_process_tree(root, expected_start_time=None)

    assert killed == [root, child]
    assert taskkill_calls == [["taskkill", "/T", "/F", "/PID", str(root)]]


def test_kill_process_tree_refuses_unpinned_surviving_child(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A survivor with no start-time fingerprint can't be identity-pinned, so
    it is never individually killed nor recorded -- a bare-pid kill could hit a
    stranger holding a recycled pid (#2059)."""
    import logging

    import charlie_work.process_utils as _pu

    root, child = 60021, 60022
    children = [child]  # enumeration could not fingerprint it
    alive = {root: True, child: True}
    starts = {root: 1000.0, child: None}
    taskkill_calls: list[list[str]] = []
    _stub_kill_env(
        monkeypatch,
        children=children,
        alive=alive,
        starts=starts,
        taskkill_calls=taskkill_calls,
    )
    # The tree kill spares the child (e.g. it could not be terminated).
    alive[child] = True

    def sparing_run_captured(command: list[str], **_kwargs: Any) -> RunResult:
        taskkill_calls.append(list(command))
        if int(command[-1]) == root:
            alive[root] = False
        return RunResult(returncode=0, stdout="", stderr="", error=None)

    monkeypatch.setattr(_pu, "run_captured", sparing_run_captured)

    with caplog.at_level(logging.WARNING, logger="charlie_work"):
        killed = kill_process_tree(root, expected_start_time=None)

    assert killed == [root]
    assert ["taskkill", "/T", "/F", "/PID", str(child)] not in taskkill_calls
    assert any("no start-time fingerprint" in r.getMessage() for r in caplog.records)


def test_kill_process_tree_refuses_child_predating_its_parent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A 'child' created before its alleged parent is a stale-ParentProcessId
    artifact (a pid recycled onto an unrelated process -- Windows never
    reparents). It is refused, warned about, and not recorded (#2059)."""
    import logging

    import charlie_work.process_utils as _pu

    root, phantom = 60031, 60032
    children = [phantom]  # "child" predates root's start of 1000.0
    alive = {root: True, phantom: True}
    starts = {root: 1000.0, phantom: 500.0}
    taskkill_calls: list[list[str]] = []
    _stub_kill_env(
        monkeypatch,
        children=children,
        alive=alive,
        starts=starts,
        taskkill_calls=taskkill_calls,
    )
    # The platform kill does not reach the phantom (protected or its own tree).
    alive[phantom] = True

    def sparing_run_captured(command: list[str], **_kwargs: Any) -> RunResult:
        taskkill_calls.append(list(command))
        if int(command[-1]) == root:
            alive[root] = False
        return RunResult(returncode=0, stdout="", stderr="", error=None)

    monkeypatch.setattr(_pu, "run_captured", sparing_run_captured)

    with caplog.at_level(logging.WARNING, logger="charlie_work"):
        killed = kill_process_tree(root, expected_start_time=None)

    assert killed == [root]
    assert ["taskkill", "/T", "/F", "/PID", str(phantom)] not in taskkill_calls
    assert any("stale ParentProcessId" in r.getMessage() for r in caplog.records)


def test_kill_process_tree_refuses_enumerated_caller_ancestor(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An enumerated 'child' naming a caller ancestor gets the same #1842
    refusal as the root: a stale ppid can name an ancestor, and killing it
    would fell this process's own subtree."""
    import logging

    import charlie_work.process_utils as _pu

    root, ancestor = 60041, 60042
    children = [ancestor]
    alive = {root: True, ancestor: True}
    starts = {root: 1000.0, ancestor: 2000.0}
    taskkill_calls: list[list[str]] = []
    _stub_kill_env(
        monkeypatch,
        children=children,
        alive=alive,
        starts=starts,
        taskkill_calls=taskkill_calls,
    )
    monkeypatch.setattr(_pu, "_self_ancestor_pids", lambda: frozenset({ancestor}))
    alive[ancestor] = True

    def sparing_run_captured(command: list[str], **_kwargs: Any) -> RunResult:
        taskkill_calls.append(list(command))
        if int(command[-1]) == root:
            alive[root] = False
        return RunResult(returncode=0, stdout="", stderr="", error=None)

    monkeypatch.setattr(_pu, "run_captured", sparing_run_captured)

    with caplog.at_level(logging.WARNING, logger="charlie_work"):
        killed = kill_process_tree(root, expected_start_time=None)

    assert killed == [root]
    assert ["taskkill", "/T", "/F", "/PID", str(ancestor)] not in taskkill_calls
    assert any("caller ancestor" in r.getMessage() for r in caplog.records)
