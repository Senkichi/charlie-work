from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from charlie_work.process_utils import CpuPriority, popen_worker


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
        popen_worker(
            [sys.executable, "-c", "pass"], priority=CpuPriority.NORMAL, stdout=subprocess.DEVNULL
        )

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
            priority=CpuPriority.NORMAL,
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

    popen_worker(
        [sys.executable, "-c", "pass"], priority=CpuPriority.NORMAL, stdout=subprocess.DEVNULL
    )

    assert mock.call_args.kwargs.get("start_new_session") is True


def test_popen_worker_omits_start_new_session_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`popen_worker` does not inject start_new_session on Windows by default."""
    monkeypatch.setattr(os, "name", "nt")
    mock = _capture_popen_call(monkeypatch)

    popen_worker(
        [sys.executable, "-c", "pass"], priority=CpuPriority.NORMAL, stdout=subprocess.DEVNULL
    )

    assert "start_new_session" not in mock.call_args.kwargs


def test_popen_worker_respects_explicit_start_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers can override the default start_new_session behavior."""
    monkeypatch.setattr(os, "name", "nt")
    mock = _capture_popen_call(monkeypatch)

    popen_worker(
        [sys.executable, "-c", "pass"],
        priority=CpuPriority.NORMAL,
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

    popen_worker(
        [sys.executable, "-c", "pass"], priority=CpuPriority.NORMAL, stdout=subprocess.DEVNULL
    )

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


def test_popen_worker_requires_an_explicit_priority() -> None:
    """No default: a launch site cannot silently inherit either class."""
    with pytest.raises(TypeError, match="priority"):
        popen_worker([sys.executable, "-c", "pass"])  # type: ignore[call-arg]


def test_popen_worker_below_normal_composes_priority_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BELOW_NORMAL adds the Windows priority-class flag; NORMAL adds nothing."""
    below = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    monkeypatch.setattr(subprocess, "Popen", MagicMock())
    with patch("charlie_work.process_utils.hidden_console_kwargs") as mock_helper:
        popen_worker([sys.executable, "-c", "pass"], priority=CpuPriority.BELOW_NORMAL)
        popen_worker([sys.executable, "-c", "pass"], priority=CpuPriority.NORMAL)
    assert [c.args[0] for c in mock_helper.call_args_list] == [below | group, group]


_REPORT_CHILD_PRIORITY = """
import ctypes, subprocess, sys
code = (
    "import ctypes; k = ctypes.windll.kernel32; "
    "k.GetCurrentProcess.restype = ctypes.c_void_p; "
    "k.GetPriorityClass.argtypes = [ctypes.c_void_p]; "
    "print(k.GetPriorityClass(k.GetCurrentProcess()))"
)
print(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip())
"""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows priority classes")
@pytest.mark.parametrize(
    ("priority", "expected_flag"),
    [
        (CpuPriority.BELOW_NORMAL, getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)),
        (CpuPriority.NORMAL, getattr(subprocess, "NORMAL_PRIORITY_CLASS", 0)),
    ],
)
def test_popen_worker_priority_is_inherited_by_grandchildren(
    priority: CpuPriority, expected_flag: int
) -> None:
    """The real mechanism: an agent's pytest is a grandchild, so the class must
    reach it through an unflagged intermediate spawn (Windows inherits
    BELOW_NORMAL by default; this pins that the design relies on it)."""
    proc = popen_worker(
        [sys.executable, "-c", _REPORT_CHILD_PRIORITY],
        priority=priority,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    assert int(out.strip()) == expected_flag
