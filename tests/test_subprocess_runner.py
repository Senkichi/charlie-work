"""Tests for the shared subprocess spawn helpers in ``subprocess_runner``.

Covers ``no_console_window_kwargs`` — the single point of enforcement for
suppressing the transient console window Windows allocates for spawned
children (issue #393) — and ``hidden_console_kwargs`` for long-lived worker
spawns that need an inherited hidden console (issue #459). Confirms
``run_captured`` routes through the appropriate helper.
"""

from __future__ import annotations

import subprocess

import pytest
from unittest.mock import patch

from charlie_work.subprocess_runner import (
    hidden_console_kwargs,
    no_console_window_kwargs,
    run_captured,
)


class TestNoConsoleWindowKwargsWindows:
    """On Windows, CREATE_NO_WINDOW must be OR'd into whatever flags the call
    site already needs -- except when DETACHED_PROCESS is present, since that
    combination is invalid/contradictory on Windows."""

    def test_default_adds_create_no_window(self):
        with (
            patch("charlie_work.subprocess_runner.sys.platform", "win32"),
            patch("charlie_work.subprocess_runner._CREATE_NO_WINDOW", 0x08000000),
            patch("charlie_work.subprocess_runner._DETACHED_PROCESS", 0x00000008),
        ):
            result = no_console_window_kwargs()
        assert result == {"creationflags": 0x08000000}

    def test_preserves_requested_group_flags(self):
        create_new_process_group = 0x00000200
        with (
            patch("charlie_work.subprocess_runner.sys.platform", "win32"),
            patch("charlie_work.subprocess_runner._CREATE_NO_WINDOW", 0x08000000),
            patch("charlie_work.subprocess_runner._DETACHED_PROCESS", 0x00000008),
        ):
            result = no_console_window_kwargs(create_new_process_group)
        assert result["creationflags"] == create_new_process_group | 0x08000000
        assert result["creationflags"] & create_new_process_group

    def test_never_combined_with_detached_process(self):
        detached_process = 0x00000008
        create_new_process_group = 0x00000200
        with (
            patch("charlie_work.subprocess_runner.sys.platform", "win32"),
            patch("charlie_work.subprocess_runner._CREATE_NO_WINDOW", 0x08000000),
            patch("charlie_work.subprocess_runner._DETACHED_PROCESS", detached_process),
        ):
            result = no_console_window_kwargs(detached_process | create_new_process_group)
        # Flags come back unchanged -- CREATE_NO_WINDOW must not appear.
        assert result == {"creationflags": detached_process | create_new_process_group}
        assert not result["creationflags"] & 0x08000000

    def test_noop_when_create_no_window_unavailable(self):
        # Defensive: even reporting as win32, if the constant genuinely isn't
        # exposed by the subprocess module (e.g. a stub), never invent a flag.
        with (
            patch("charlie_work.subprocess_runner.sys.platform", "win32"),
            patch("charlie_work.subprocess_runner._CREATE_NO_WINDOW", 0),
        ):
            assert no_console_window_kwargs() == {}


class TestNoConsoleWindowKwargsPosix:
    def test_returns_empty_dict_regardless_of_requested_flags(self):
        with patch("charlie_work.subprocess_runner.sys.platform", "linux"):
            assert no_console_window_kwargs() == {}
            assert no_console_window_kwargs(0x00000200) == {}


class TestHiddenConsoleKwargsWindows:
    """On Windows, worker spawns get CREATE_NEW_CONSOLE plus a STARTUPINFO
    that hides the console so descendants inherit a hidden console."""

    def test_default_adds_create_new_console_and_hidden_startupinfo(self):
        with (
            patch("charlie_work.subprocess_runner.sys.platform", "win32"),
            patch("charlie_work.subprocess_runner._CREATE_NEW_CONSOLE", 0x00000010),
        ):
            result = hidden_console_kwargs()
        assert result["creationflags"] == 0x00000010
        assert "startupinfo" in result
        assert result["startupinfo"].wShowWindow == subprocess.SW_HIDE
        assert result["startupinfo"].dwFlags & subprocess.STARTF_USESHOWWINDOW

    def test_preserves_requested_group_flags(self):
        create_new_console = 0x00000010
        create_new_process_group = 0x00000200
        with (
            patch("charlie_work.subprocess_runner.sys.platform", "win32"),
            patch("charlie_work.subprocess_runner._CREATE_NEW_CONSOLE", create_new_console),
        ):
            result = hidden_console_kwargs(create_new_process_group)
        assert result["creationflags"] == create_new_process_group | create_new_console
        assert result["startupinfo"].wShowWindow == subprocess.SW_HIDE

    def test_rejects_create_no_window(self):
        create_new_console = 0x00000010
        create_no_window = 0x08000000
        with (
            patch("charlie_work.subprocess_runner.sys.platform", "win32"),
            patch("charlie_work.subprocess_runner._CREATE_NEW_CONSOLE", create_new_console),
            patch("charlie_work.subprocess_runner._CREATE_NO_WINDOW", create_no_window),
        ):
            with pytest.raises(ValueError, match="mutually-exclusive"):
                hidden_console_kwargs(create_no_window)

    def test_rejects_detached_process(self):
        create_new_console = 0x00000010
        detached_process = 0x00000008
        with (
            patch("charlie_work.subprocess_runner.sys.platform", "win32"),
            patch("charlie_work.subprocess_runner._CREATE_NEW_CONSOLE", create_new_console),
            patch("charlie_work.subprocess_runner._DETACHED_PROCESS", detached_process),
        ):
            with pytest.raises(ValueError, match="mutually-exclusive"):
                hidden_console_kwargs(detached_process)


class TestHiddenConsoleKwargsPosix:
    def test_returns_empty_dict_regardless_of_requested_flags(self):
        with patch("charlie_work.subprocess_runner.sys.platform", "linux"):
            assert hidden_console_kwargs() == {}
            assert hidden_console_kwargs(0x00000200) == {}


class TestRunCapturedUsesHelper:
    def test_run_captured_passes_helper_kwargs_to_subprocess_run(self, tmp_path):
        sentinel = {"creationflags": 0x08000000}
        fake_completed = subprocess.CompletedProcess(
            args=["echo", "hi"], returncode=0, stdout="hi\n", stderr=""
        )
        with (
            patch(
                "charlie_work.subprocess_runner.no_console_window_kwargs",
                return_value=sentinel,
            ) as mock_kwargs,
            patch(
                "charlie_work.subprocess_runner.subprocess.run", return_value=fake_completed
            ) as mock_run,
        ):
            result = run_captured(["echo", "hi"], cwd=tmp_path, timeout_seconds=5)

        assert result.ok
        mock_kwargs.assert_called_once_with()
        _, call_kwargs = mock_run.call_args
        assert call_kwargs["creationflags"] == 0x08000000

    def test_run_captured_passes_stdin_to_subprocess_run(self, tmp_path):
        sentinel = {"creationflags": 0x08000000}
        fake_completed = subprocess.CompletedProcess(
            args=["cat"], returncode=0, stdout="hello\n", stderr=""
        )
        with (
            patch(
                "charlie_work.subprocess_runner.no_console_window_kwargs",
                return_value=sentinel,
            ),
            patch(
                "charlie_work.subprocess_runner.subprocess.run", return_value=fake_completed
            ) as mock_run,
        ):
            result = run_captured(["cat"], cwd=tmp_path, timeout_seconds=5, stdin="hello\n")

        assert result.ok
        assert result.stdout == "hello\n"
        _, call_kwargs = mock_run.call_args
        assert call_kwargs["input"] == "hello\n"
