from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pytest

import charlie_work.process_utils as _pu
from charlie_work.process_utils import sweep_orphan_processes


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


@pytest.mark.parametrize(
    "worktree_path",
    [
        "",  # empty needle: ``-like "**"``-equivalent matches every process
        "   ",  # whitespace-only — same match-everything shape after strip
        " \t\n ",  # mixed whitespace
        "C:\\",  # drive root — matches anything under C:\
        "C:\\a",  # below _MIN_SWEEP_WORKTREE_PATH_LENGTH
        "a/b",  # below minimum length despite a separator
        "worktrees",  # bare name, no separator — never a worktree path
        "D:\\worktrees\\*",  # -like wildcard metachar — matches the whole host
        "D:\\worktrees\\a?b",  # -like '?' metacharacter
        'D:\\worktrees\\"x',  # quote breaks out of the -like pattern string
        "D:\\worktrees\\a[b]",  # -like character-class metacharacters
    ],
)
def test_sweep_orphan_processes_rejects_degenerate_needles(
    monkeypatch: pytest.MonkeyPatch, worktree_path: str
) -> None:
    """Issue #1842: a degenerate ``worktree_path`` must be refused before the
    ``-like "*<path>*"`` filter is ever built.

    ``""``/whitespace produces a needle that matches *every* CommandLine on
    the host — including the pytest controller/uv/pwsh ancestry running the
    sweep itself — and would feed ``os.getpid()``'s own tree to
    ``kill_orphan_pid``. The refusal must fire before any PowerShell/CIM
    query runs, so this test forces the Windows branch on every platform and
    spies on ``subprocess.run``: any query at all is a failure, and the fake
    answer hands back this process's own PID so a regression is also caught
    by the result assertions.
    """
    import json

    # Force the Windows code path regardless of host so path validation is
    # the only thing that can prevent the CIM query.
    monkeypatch.setattr(_pu.os, "name", "nt")
    monkeypatch.setattr(_pu.shutil, "which", lambda _name: "powershell")

    run_calls: list[Any] = []

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        run_calls.append(args)
        # If the degenerate needle ever reaches CIM it matches this process
        # too — return os.getpid() so a regression shows up in the result
        # assertions, not just in the spy.
        payload = json.dumps(
            [
                {
                    "ProcessId": os.getpid(),
                    "Name": "pytest.exe",
                    "CommandLine": f"pytest.exe {worktree_path}",
                }
            ]
        )
        return subprocess.CompletedProcess(args, 0, payload, "")

    monkeypatch.setattr(_pu.subprocess, "run", fake_run)

    orphans = sweep_orphan_processes(worktree_path)

    assert run_calls == [], f"degenerate worktree_path {worktree_path!r} reached the CIM query"
    assert all(orphan["pid"] != os.getpid() for orphan in orphans)
    assert orphans == []
