from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from charlie_work.process_utils import popen_worker


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
