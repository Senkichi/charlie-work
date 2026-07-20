"""Tests for the shared subprocess spawn helpers in ``subprocess_runner``.

Covers ``no_console_window_kwargs`` — the single point of enforcement for
suppressing the transient console window Windows allocates for spawned
children (issue #393) — and ``hidden_console_kwargs`` for long-lived worker
spawns that need an inherited hidden console (issue #459). Confirms
``run_captured`` routes through the appropriate helper. Also covers
``resolve_cli_binary`` — the single point of enforcement for unwrapping
npm ``.CMD``/``.bat`` shims (e.g. ``claude.CMD``) to their underlying
``.exe`` so ``Popen(shell=False)`` can find and invoke them without going
through ``cmd.exe`` (issue #487).
"""

from __future__ import annotations

import subprocess

import pytest
from unittest.mock import patch

from charlie_work.subprocess_runner import (
    hidden_console_kwargs,
    no_console_window_kwargs,
    resolve_cli_binary,
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


class TestResolveCliBinary:
    """``resolve_cli_binary`` unwraps an npm ``.CMD`` shim to the ``.exe`` it
    wraps, so a bare ``"claude"`` resolves to something ``CreateProcessW``
    can execute directly without going through ``cmd.exe`` (issue #487)."""

    def test_binary_not_on_path_returns_name_unchanged(self):
        with patch("charlie_work.subprocess_runner.shutil.which", return_value=None):
            assert resolve_cli_binary("this-binary-does-not-exist-xyz") == (
                "this-binary-does-not-exist-xyz"
            )

    def test_posix_returns_which_result_unchanged(self):
        with (
            patch("charlie_work.subprocess_runner.os.name", "posix"),
            patch("charlie_work.subprocess_runner.shutil.which", return_value="/usr/bin/claude"),
        ):
            assert resolve_cli_binary("claude") == "/usr/bin/claude"

    def test_non_shim_extension_returned_unchanged(self, tmp_path):
        exe_path = tmp_path / "claude.exe"
        exe_path.write_bytes(b"")
        with (
            patch("charlie_work.subprocess_runner.os.name", "nt"),
            patch("charlie_work.subprocess_runner.shutil.which", return_value=str(exe_path)),
        ):
            assert resolve_cli_binary("claude") == str(exe_path)

    def test_npm_cmd_shim_unwrapped_to_underlying_exe(self, tmp_path):
        # Mirror the real npm shim layout:
        #   node_modules/.bin/claude.CMD
        #   node_modules/@anthropic-ai/claude-code/cli.exe
        bin_dir = tmp_path / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        pkg_dir = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code"
        pkg_dir.mkdir(parents=True)
        real_exe = pkg_dir / "cli.exe"
        real_exe.write_bytes(b"")

        shim_path = bin_dir / "claude.CMD"
        shim_path.write_text(
            '"%dp0%\\..\\@anthropic-ai\\claude-code\\cli.exe" %*\n',
            encoding="utf-8",
        )

        with (
            patch("charlie_work.subprocess_runner.os.name", "nt"),
            patch("charlie_work.subprocess_runner.shutil.which", return_value=str(shim_path)),
        ):
            resolved = resolve_cli_binary("claude")

        assert resolved == str(real_exe)
        assert resolved.lower().endswith(".exe")

    def test_shim_target_missing_falls_back_to_shim_path(self, tmp_path):
        bin_dir = tmp_path / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        shim_path = bin_dir / "claude.CMD"
        shim_path.write_text(
            '"%dp0%\\..\\nonexistent-pkg\\cli.exe" %*\n',
            encoding="utf-8",
        )

        with (
            patch("charlie_work.subprocess_runner.os.name", "nt"),
            patch("charlie_work.subprocess_runner.shutil.which", return_value=str(shim_path)),
        ):
            resolved = resolve_cli_binary("claude")

        # Target .exe referenced by the shim does not exist on disk: fall
        # back to the shim path itself rather than fabricating a path.
        assert resolved == str(shim_path)

    def test_unparseable_shim_falls_back_to_shim_path(self, tmp_path):
        bin_dir = tmp_path / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        shim_path = bin_dir / "claude.CMD"
        shim_path.write_text(
            "@echo off\r\nrem not the expected npm shim shape\r\n", encoding="utf-8"
        )

        with (
            patch("charlie_work.subprocess_runner.os.name", "nt"),
            patch("charlie_work.subprocess_runner.shutil.which", return_value=str(shim_path)),
        ):
            assert resolve_cli_binary("claude") == str(shim_path)
