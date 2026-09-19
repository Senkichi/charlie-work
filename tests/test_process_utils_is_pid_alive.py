from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from charlie_work.process_utils import is_pid_alive


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


# --- issue #1371: ACCESS_DENIED fail-closed ---------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_is_pid_alive_false_for_protected_system_process() -> None:
    """Issue #1371 acceptance criterion #1.

    ``is_pid_alive`` must return ``False`` for a real protected/system PID
    discovered at test runtime -- not a hardcoded guess.  PID 4 (the Windows
    ``System`` process) is always present and always returns
    ``ERROR_ACCESS_DENIED`` for ``OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)``
    from a non-administrator token.  If no protected PID is discoverable, the
    test skips cleanly.
    """
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    _WIN_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    # Discover a protected PID at runtime.  PID 4 (System) is the canonical
    # always-present protected process on Windows NT; OpenProcess with limited
    # rights returns 0 + ERROR_ACCESS_DENIED for it from a non-elevated token.
    candidate_pids = [4]
    protected_pid: int | None = None
    for candidate in candidate_pids:
        handle = kernel32.OpenProcess(_WIN_PROCESS_QUERY_LIMITED_INFORMATION, False, candidate)
        if not handle:
            if kernel32.GetLastError() == 5:  # ERROR_ACCESS_DENIED
                protected_pid = candidate
                break
        else:
            kernel32.CloseHandle(handle)

    if protected_pid is None:
        pytest.skip("no protected PID discoverable on this host")

    # Any start time -- the identity check is never reached because the handle
    # open fails with ACCESS_DENIED.
    assert is_pid_alive(protected_pid, 123.456) is False
    assert is_pid_alive(protected_pid) is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_is_pid_alive_access_denied_returns_false_with_monkeypatched_openprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1371 acceptance criterion #3.

    Simulate the ACCESS_DENIED path with an injected ``OpenProcess`` and assert
    ``False`` -- without any snapshot fallback being able to verify identity.
    The monkeypatched ``OpenProcess`` returns 0 (failure) and
    ``GetLastError`` returns 5 (``ERROR_ACCESS_DENIED``), which is exactly the
    recycled-pid-onto-protected-process scenario.
    """

    class _FakeKernel32:
        """Minimal kernel32 stub that makes OpenProcess fail with ACCESS_DENIED."""

        def OpenProcess(self, access: int, inherit: bool, pid: int) -> int:
            return 0  # failure -> caller checks GetLastError

        def GetLastError(self) -> int:
            return 5  # ERROR_ACCESS_DENIED

        def CloseHandle(self, handle: int) -> int:
            return 1

        def GetExitCodeProcess(self, handle: int, exit_code: Any) -> int:
            return 0  # not reached

    fake = _FakeKernel32()
    # ``is_pid_alive`` resolves ``ctypes.windll.kernel32`` at call time, so
    # patch the attribute on the ctypes.windll proxy.
    import ctypes

    monkeypatch.setattr(ctypes.windll, "kernel32", fake, raising=False)

    # No snapshot fallback is present in the implementation, so identity cannot
    # be verified -- the assertion must hold purely from the ACCESS_DENIED path.
    assert is_pid_alive(6262, 123.456) is False
    assert is_pid_alive(6262) is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_is_pid_alive_invalid_parameter_dead_pid_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1371 acceptance criterion #4: no behavior change for the dead-pid path.

    ``ERROR_INVALID_PARAMETER`` (87) from ``OpenProcess`` means the PID does not
    exist.  This path must still return ``False`` -- the fix only changed the
    ``ERROR_ACCESS_DENIED`` branch.
    """

    class _FakeKernel32:
        def OpenProcess(self, access: int, inherit: bool, pid: int) -> int:
            return 0  # failure

        def GetLastError(self) -> int:
            return 87  # ERROR_INVALID_PARAMETER

        def CloseHandle(self, handle: int) -> int:
            return 1

        def GetExitCodeProcess(self, handle: int, exit_code: Any) -> int:
            return 0  # not reached

    fake = _FakeKernel32()
    import ctypes

    monkeypatch.setattr(ctypes.windll, "kernel32", fake, raising=False)

    assert is_pid_alive(999999, 123.456) is False
    assert is_pid_alive(999999) is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only: ctypes.windll kernel32 stub")
def test_is_pid_alive_get_exit_code_failure_fallthrough_uses_start_time_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1371 audit: Windows ``GetExitCodeProcess`` failure fallthrough.

    When ``OpenProcess`` succeeds but ``GetExitCodeProcess`` fails, the function
    must NOT declare the PID alive without verifying start-time identity.  It
    falls through to the identity check: ``True`` on a matching
    ``expected_start_time`` and ``False`` on a mismatched one (recycled PID).
    """

    class _FakeKernel32:
        def OpenProcess(self, access: int, inherit: bool, pid: int) -> int:
            return 1  # success -- a nonzero handle

        def GetLastError(self) -> int:
            return 0

        def GetExitCodeProcess(self, handle: int, exit_code: Any) -> int:
            return 0  # failure -- cannot read exit code

        def CloseHandle(self, handle: int) -> int:
            return 1

    fake = _FakeKernel32()
    import ctypes

    monkeypatch.setattr(ctypes.windll, "kernel32", fake, raising=False)

    matching_start = 1000.0
    mismatched_start = 5000.0

    # Matching start time -> identity verified -> alive (True).
    with patch(
        "charlie_work.process_utils.get_process_start_time",
        return_value=matching_start,
    ):
        assert is_pid_alive(6262, matching_start) is True

    # Mismatched start time -> identity mismatch -> recycled -> False.
    with patch(
        "charlie_work.process_utils.get_process_start_time",
        return_value=mismatched_start,
    ):
        assert is_pid_alive(6262, matching_start) is False


def test_is_pid_alive_posix_permission_error_fallthrough_uses_start_time_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1371 audit: POSIX ``PermissionError`` fallthrough.

    When ``os.kill(pid, 0)`` raises ``PermissionError`` (process exists but we
    cannot signal it), the function must NOT declare the PID alive without
    verifying start-time identity.  It falls through to the identity check:
    ``True`` on a matching ``expected_start_time`` and ``False`` on a mismatched
    one (recycled PID).

    Not gated behind a POSIX-only skip: CI runs self-hosted on Windows, so the
    POSIX branch is forced via a ``sys.platform`` monkeypatch to ensure the path
    is actually exercised rather than skipped on every CI run.
    """
    # Force the POSIX branch regardless of the host platform.
    monkeypatch.setattr("charlie_work.process_utils.sys.platform", "linux")

    def fake_kill(pid: int, sig: int) -> None:
        raise PermissionError("simulated: process exists but cannot signal")

    monkeypatch.setattr("charlie_work.process_utils.os.kill", fake_kill)

    matching_start = 1000.0
    mismatched_start = 5000.0

    # Matching start time -> identity verified -> alive (True).
    with patch(
        "charlie_work.process_utils.get_process_start_time",
        return_value=matching_start,
    ):
        assert is_pid_alive(6262, matching_start) is True

    # Mismatched start time -> identity mismatch -> recycled -> False.
    with patch(
        "charlie_work.process_utils.get_process_start_time",
        return_value=mismatched_start,
    ):
        assert is_pid_alive(6262, matching_start) is False
