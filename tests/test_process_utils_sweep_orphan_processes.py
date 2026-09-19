from __future__ import annotations

import subprocess
import sys

import pytest

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
