"""Launch process-spawn mechanics: POSIX ``start_new_session``,
Windows creationflags routing, and tee-stream-JSON output handling.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
import pytest

from _claude_adapter_fixtures import _install_fake_create_worktree
from _worker_marker_wait import read_worker_marker

from charlie_work.claude_code import launch_claude_worker


def test_launch_claude_worker_includes_start_new_session_on_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """launch_claude_worker should include start_new_session=True on POSIX systems."""
    from unittest.mock import patch

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Capture the kwargs passed to subprocess.Popen
    popen_kwargs: dict = {}
    original_popen = subprocess.Popen

    def capture_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        # Return a fake process that exits immediately
        return original_popen([sys.executable, "-c", "pass"], **kwargs)

    with patch("subprocess.Popen", side_effect=capture_popen):
        launch_claude_worker(
            999,
            "agent/issue-999-start-new-session",
            "prompt",
            repo_root=repo_root,
            sessions_dir=sessions_dir,
            command_template=(sys.executable, "-c", "pass"),
        )

    # Worker spawns use hidden_console_kwargs: CREATE_NEW_CONSOLE plus a
    # STARTUPINFO with wShowWindow=SW_HIDE so descendants inherit a hidden
    # console. Policy A survival flags (DETACHED_PROCESS, CREATE_BREAKAWAY_FROM_JOB)
    # are out of scope for issue #360.
    if os.name != "nt":
        assert popen_kwargs.get("start_new_session") is True
        assert "creationflags" not in popen_kwargs
        assert "startupinfo" not in popen_kwargs
    else:
        assert popen_kwargs.get("start_new_session") is False
        flags = popen_kwargs.get("creationflags", 0)
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
        assert flags & subprocess.CREATE_NEW_CONSOLE
        assert not (flags & subprocess.CREATE_NO_WINDOW)
        assert not (flags & subprocess.DETACHED_PROCESS)
        assert not (flags & subprocess.CREATE_BREAKAWAY_FROM_JOB)
        startupinfo = popen_kwargs.get("startupinfo")
        assert startupinfo is not None
        assert startupinfo.wShowWindow == subprocess.SW_HIDE
        assert startupinfo.dwFlags & subprocess.STARTF_USESHOWWINDOW


def test_launch_claude_worker_routes_creationflags_through_hidden_console_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """launch_claude_worker must obtain its Popen kwargs via
    ``hidden_console_kwargs`` (issue #459) so worker descendants inherit a
    hidden console instead of each allocating their own visible window.

    Note: patching ``subprocess.Popen`` globally also intercepts the
    internal ``Popen`` calls that ``subprocess.run`` makes under the hood
    (e.g. from any incidental git cleanup on the error path), so we record
    kwargs *per call* and assert on the first one -- the actual worker
    launch -- rather than a merged/overwritten dict.
    """
    from unittest.mock import MagicMock, patch

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    popen_calls: list[dict] = []

    def capture_popen(*args, **kwargs):
        popen_calls.append(kwargs)
        return MagicMock(pid=12345)

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    sentinel_kwargs = {
        "creationflags": subprocess.CREATE_NEW_CONSOLE,
        "startupinfo": startupinfo,
    }
    with (
        patch("subprocess.Popen", side_effect=capture_popen),
        patch(
            "charlie_work.process_utils.hidden_console_kwargs",
            return_value=sentinel_kwargs,
        ) as mock_helper,
    ):
        launch_claude_worker(
            998,
            "agent/issue-998-hidden-console",
            "prompt",
            repo_root=repo_root,
            sessions_dir=sessions_dir,
            command_template=(sys.executable, "-c", "pass"),
        )

    expected_group_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    mock_helper.assert_called_once_with(expected_group_flag)
    assert popen_calls, "expected at least one Popen call from the worker launch"
    assert popen_calls[0].get("creationflags") == subprocess.CREATE_NEW_CONSOLE
    assert popen_calls[0].get("startupinfo") is startupinfo


# Critical regression test for issue #160: tee thread file handle closure bug
# ---------------------------------------------------------------------------


def test_launch_claude_worker_tee_stream_json_writes_to_both_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test for issue #160: tee thread must write to both log and events files.

    This test verifies that when tee_stream_json=True, the background tee thread
    successfully writes output to both the plaintext log file and the events.jsonl file.
    The original bug closed file handles immediately after starting the thread,
    causing all writes to fail silently and leaving both files empty.

    This test would fail against the buggy code (empty files) and pass after the fix.
    """
    from charlie_work.process_utils import is_session_stalled
    from charlie_work.claude_code import _classify_session_failure

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Create a fake claude script that outputs multiple lines
    script_path = tmp_path / "fake_claude_tee.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path

            data = sys.stdin.read()
            Path("worker-ran.txt").write_text(data, encoding="utf-8")
            # Output multiple lines to verify tee writes all of them
            print("line 1")
            print("line 2")
            print("line 3")
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        160,
        "agent/issue-160-tee-test",
        "test prompt for tee",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
        tee_stream_json=True,  # Enable the tee feature
    )

    assert record.ok
    worktree_path = Path(record.worktree_path)

    # Wait for the worker to complete (content, not just the path appearing).
    marker_path = worktree_path / "worker-ran.txt"
    read_worker_marker(marker_path)

    # Give the tee thread a moment to finish writing
    time.sleep(0.2)

    # Verify both files exist and have content
    log_path = Path(record.log_path)
    events_path = sessions_dir / "issue-160.events.jsonl"

    assert log_path.exists(), "Log file should exist"
    assert events_path.exists(), "Events file should exist"

    log_content = log_path.read_text(encoding="utf-8")
    events_content = events_path.read_text(encoding="utf-8")

    # Both files should have content (the bug would leave them empty)
    assert len(log_content) > 0, "Log file should have content (bug would leave it empty)"
    assert len(events_content) > 0, "Events file should have content (bug would leave it empty)"

    # Verify the content matches what we expect
    assert "line 1" in log_content
    assert "line 2" in log_content
    assert "line 3" in log_content

    # Events file should have the same content (it's a tee)
    assert "line 1" in events_content
    assert "line 2" in events_content
    assert "line 3" in events_content

    # Verify is_session_stalled works correctly on the tee'd log
    # The log should not be stalled (it was just written)
    is_stalled, last_line = is_session_stalled(log_path, stall_threshold_minutes=20)
    assert is_stalled is False, "Freshly written log should not be stalled"
    assert last_line is not None, "Should be able to read last line from log"

    # Verify _classify_session_failure works correctly on the tee'd log
    # Should return None (no failure) for a successful run
    failure_kind, throttled_until = _classify_session_failure(log_path)
    assert failure_kind is None, "Successful run should not be classified as a failure"
    assert throttled_until is None


def test_launch_claude_worker_tee_stream_json_popen_failure_closes_handles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test: in the tee_stream_json branch, log_handle/events_handle
    are opened without a `with` block so the background tee thread can own their
    lifecycle (closing them itself once the process exits). But if
    subprocess.Popen raises before the thread ever starts, nobody closes them.
    Popen failure must close both handles before the OSError propagates.
    """
    from unittest.mock import patch

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    opened_handles: list = []
    original_open = Path.open

    def tracking_open(self, *args, **kwargs):
        handle = original_open(self, *args, **kwargs)
        if self.name.endswith((".claude.log", ".events.jsonl")):
            opened_handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)

    with patch("subprocess.Popen", side_effect=OSError("mock spawn failure")):
        record = launch_claude_worker(
            161,
            "agent/issue-161-tee-popen-failure",
            "prompt text",
            repo_root=repo_root,
            sessions_dir=sessions_dir,
            command_template=("claude",),
            tee_stream_json=True,
        )

    assert not record.ok
    assert record.error is not None
    assert "failed to launch claude" in record.error

    assert len(opened_handles) == 2, "expected exactly log_handle and events_handle to be opened"
    assert all(handle.closed for handle in opened_handles), (
        "log_handle/events_handle must be closed when Popen fails, not leaked"
    )
