"""Worker-session lifecycle: death-bounded runtime, live-session census, stalled-session event, orphan sweep, watchdog-off.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8; sub-split for the PR #1750 review's 800-line cap).
"""

from __future__ import annotations

import json
import os
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from typing import Any
import pytest
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_worker_death_bounded_runtime_last_activity_at_fallback(tmp_path: Path) -> None:
    """Issue #1106: ``_worker_death_bounded_runtime_seconds`` must fall back to
    the sidecar's ``last_activity_at`` timestamp when the log file is gone
    (``log_stat()`` returns None).

    The log file may be deleted after the CLI exits (cleanup, tmpfs recycle,
    operator intervention), but the sidecar's ``last_activity_at`` was updated
    each pass by ``update_worker_log_stat`` and is frozen at the last observed
    mtime.  The death-bounded runtime derived from it must be the gap between
    ``started_at`` and that last-activity timestamp — not 0.0 (which would
    only be correct when *no* activity was ever recorded).

    Mutation gate: removing the ``last_activity_at`` fallback branch (so the
    function returns 0.0 when ``log_stat()`` is None) makes this test fail
    (0.0 != ~5.0).
    """
    from charlie_work.worker import WorkerView
    from charlie_work.workflow import _worker_death_bounded_runtime_seconds

    now = datetime.now(UTC)
    started_at = now - timedelta(seconds=300)
    death_at = started_at + timedelta(seconds=5)

    # Log path does not exist — log_stat() will return None.
    missing_log = tmp_path / "nonexistent" / "issue-123.log"

    worker = WorkerView(
        adapter_kind="devin",
        issue_number=123,
        repo_key="",
        pid=99999,
        started_at=started_at.isoformat().replace("+00:00", "Z"),
        process_start_time=started_at.timestamp(),
        log_path=str(missing_log),
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        error=None,
        failure_kind=None,
        reclaimed=None,
        branch="agent/issue-123-fix-search",
        last_activity_at=death_at.isoformat().replace("+00:00", "Z"),
    )

    runtime = _worker_death_bounded_runtime_seconds(worker)
    # The fallback must derive ~5s from last_activity_at, not 0.0.
    assert runtime == pytest.approx(5.0, abs=0.01)


def test_worker_death_bounded_runtime_no_signal_returns_zero(tmp_path: Path) -> None:
    """Issue #1106: ``_worker_death_bounded_runtime_seconds`` must return 0.0
    when neither ``log_stat()`` nor ``last_activity_at`` is available — the CLI
    never wrote anything, which is a startup death by construction.

    Mutation gate: changing the final fallback to return a non-zero value
    (e.g. ``worker.runtime_seconds()``) makes this test fail.
    """
    from charlie_work.worker import WorkerView
    from charlie_work.workflow import _worker_death_bounded_runtime_seconds

    now = datetime.now(UTC)
    started_at = now - timedelta(seconds=300)

    missing_log = tmp_path / "nonexistent" / "issue-123.log"

    worker = WorkerView(
        adapter_kind="devin",
        issue_number=123,
        repo_key="",
        pid=99999,
        started_at=started_at.isoformat().replace("+00:00", "Z"),
        process_start_time=started_at.timestamp(),
        log_path=str(missing_log),
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        error=None,
        failure_kind=None,
        reclaimed=None,
        branch="agent/issue-123-fix-search",
        last_activity_at=None,  # no sidecar activity recorded
    )

    runtime = _worker_death_bounded_runtime_seconds(worker)
    assert runtime == 0.0


def test_count_live_sessions_counts_both_adapters(tmp_path: Path) -> None:
    """_count_live_sessions should count sessions from both devin-shell and claude-code adapters."""
    from charlie_work.workflow import _count_live_sessions
    from charlie_work.devin_shell import SessionRecord as DevinSessionRecord
    from charlie_work.claude_code import ClaudeWorkerRecord

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a devin-shell session record with a valid PID (will be checked for liveness)
    # Since we can't easily create a real live process, we'll just test the file reading
    devin_record = DevinSessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/test",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--print"),
        pid=None,  # None means not alive
        started_at="2024-01-01T00:00:00Z",
        log_path="/tmp/log.log",
    )
    devin_path = sessions_dir / "issue-1.json"
    import json

    devin_path.write_text(json.dumps(devin_record.to_dict()), encoding="utf-8")

    # Create a claude-code session record
    claude_record = ClaudeWorkerRecord(
        issue_number=2,
        branch="agent/issue-2",
        worktree_path="/tmp/test2",
        prompt_path="/tmp/prompt2.md",
        command=("claude", "-p"),
        pid=None,  # None means not alive
        started_at="2024-01-01T00:00:00Z",
        log_path="/tmp/log2.log",
    )
    claude_path = sessions_dir / "issue-2.claude.json"
    claude_path.write_text(json.dumps(claude_record.to_dict()), encoding="utf-8")

    # Count live sessions (both have pid=None, so count should be 0)
    count = _count_live_sessions(sessions_dir)
    assert count == 0  # No live sessions since both have pid=None


def test_count_live_sessions_corroborates_ghost_worker_via_state_json(
    tmp_path: Path,
) -> None:
    """Issue #343: a live ``worker_pid`` recorded in state.json with NO
    corresponding session sidecar (a "ghost") must still be counted against
    the concurrency governor.

    Before this fix, ``_count_live_sessions`` only counted sidecar files on
    disk. If a sidecar goes missing for a still-live process -- e.g. the
    dead-session reap lane removed it on ambiguous evidence, or any other
    path stranded state.json's dispatch record -- the live worker became
    invisible to the governor and looked like free capacity, letting the
    next dispatch pass launch past the configured concurrency cap even
    though the ghost's process was still actually running (issue #343's
    concrete production instance: pid 23440 verified alive via
    ``Get-Process`` with its sidecar already gone).

    This test uses the current test process's own real, genuinely-alive pid
    (recorded only in state.json, never in a sidecar) to prove the ghost is
    now counted, without needing to spawn or mock a child process.

    MUTATION GATE: removing the ``if state_file is not None:`` state.json
    corroboration block in ``_count_live_sessions``
    (src/charlie_work/workflow.py) makes this test fail -- the count would
    revert to 0 and the ghost worker would look like free capacity again.
    """
    from charlie_work.devin_shell import _get_process_start_time
    from charlie_work.workflow import _count_live_sessions

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    # No sidecar file is written for issue 343 -- this is the "ghost" case:
    # a live worker_pid with no session sidecar on disk at all.

    current_pid = os.getpid()
    current_start_time = _get_process_start_time(current_pid)
    state = load_state(paths.state_file)
    state["issues"]["343"] = {
        "status": "dispatched",
        "worker_pid": current_pid,
        "worker_process_start_time": current_start_time,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)

    count = _count_live_sessions(sessions_dir, paths.state_file)
    assert count == 1, "a ghost worker_pid that is genuinely alive must count against the cap"

    # Without state.json corroboration (the pre-fix behavior), the same ghost
    # is invisible -- pin the contrast so a future regression that silently
    # drops the state_file argument elsewhere is easy to diagnose.
    assert _count_live_sessions(sessions_dir) == 0


def test_stalled_session_emits_event_with_required_fields(tmp_path: Path) -> None:
    """Issue #109: stalled session detection should emit session_stalled event with required fields."""
    from datetime import UTC, datetime, timedelta
    import os
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(
            harness="devin-shell"
        ),  # Use devin-shell adapter for watchdog support
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub
    class FakeGitHubForEvent(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []

        def issue_list(self, labels=None, state=None):
            # Support both old and new signature
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

    fake_gh = FakeGitHubForEvent()
    OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a fake stalled session sidecar
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file with old mtime (stalled by time)
    log_file = sessions_dir / "issue-109.log"
    log_file.write_text("working on issue\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a sidecar with a fake PID
    sidecar = sessions_dir / "issue-109.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 109,
                "branch": "agent/issue-109",
                "worktree_path": "/fake/path",
                "prompt_path": "/fake/prompt",
                "command": ["devin", "--print"],
                "pid": 99999,
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_file),
                "error": None,
                "process_start_time": 1234567890.0,  # Fake start time
            }
        ),
        encoding="utf-8",
    )

    # Mock read_session_records to return a fake record
    from charlie_work.devin_shell import SessionRecord

    fake_record = SessionRecord(
        issue_number=109,
        branch="agent/issue-109",
        worktree_path="/fake/path",
        prompt_path="/fake/prompt",
        command=("devin", "--print"),
        pid=99999,
        started_at=datetime.now(UTC).isoformat(),
        log_path=str(log_file),
        error=None,
        process_start_time=None,  # No start time verification in this test
    )

    # Mock is_session_alive to return True and kill_process_tree to return killed PIDs
    with (
        patch("charlie_work.devin_shell.read_session_records", return_value=[fake_record]),
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.write_gate.kill_process_tree", return_value=[99999]),
        patch(
            "charlie_work.dead_worker_reap.sweep_orphan_processes",
            return_value=[{"pid": 3492, "name": "python.exe", "command_line": "python worker.py"}],
        ),  # Fixed mock return
        patch(
            "charlie_work.devin_shell.update_session_record_with_failure_classification",
            return_value=(None, None),
        ),
    ):
        # Run the stall detection and handling
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        result = _detect_and_handle_stalled_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )

    # Check that the stalled issue was detected
    assert any(entry["issue"] == 109 for entry in result)

    # Load state and check for the event
    state = load_state(paths.state_file)
    events = state.get("events", [])

    # Find the session_stalled event
    stalled_events = [e for e in events if e.get("kind") == "session_stalled"]
    assert len(stalled_events) == 1

    event = stalled_events[0]
    # Check required fields (they're in the payload)
    payload = event.get("payload", {})
    assert payload.get("issue_number") == 109
    assert payload.get("pid") == 99999
    assert "log_mtime" in payload
    assert "last_log_line" in payload
    # killed_pids now includes both the session PID and any orphan PIDs
    # The mock returns [99999] for kill_process_tree, and sweep_orphan_processes
    # returns [3492] as a fixed mock value
    assert 99999 in payload.get("killed_pids", [])
    assert 3492 in payload.get("killed_pids", [])  # Orphan PID from mock
    # orphan_pids is included in the event payload with the exact mock value
    assert payload.get("orphan_pids") == [3492]


def test_sweep_orphan_processes_for_dead_sessions_unit(tmp_path: Path) -> None:
    """Unit test for _sweep_orphan_processes_for_dead_sessions (issue #139)."""
    from datetime import UTC, datetime
    from unittest.mock import patch, MagicMock
    import subprocess

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create fake session records
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.claude_code import ClaudeWorkerRecord

    dead_session = SessionRecord(
        issue_number=100,
        branch="agent/issue-100",
        worktree_path="/dead/worktree",
        prompt_path="/fake/prompt",
        command=("devin", "--print"),
        pid=1000,
        started_at=datetime.now(UTC).isoformat(),
        log_path="/fake/log",
        error=None,
        process_start_time=1234567890.0,
    )

    live_session = SessionRecord(
        issue_number=101,
        branch="agent/issue-101",
        worktree_path="/live/worktree",
        prompt_path="/fake/prompt",
        command=("devin", "--print"),
        pid=1001,
        started_at=datetime.now(UTC).isoformat(),
        log_path="/fake/log",
        error=None,
        process_start_time=1234567890.0,
    )

    dead_worker = ClaudeWorkerRecord(
        issue_number=102,
        branch="agent/issue-102",
        worktree_path="/dead/worker",
        prompt_path="/fake/prompt",
        command=("devin", "--print"),
        pid=1002,
        started_at=datetime.now(UTC).isoformat(),
        log_path="/fake/log",
        error=None,
        process_start_time=1234567890.0,
    )

    # Mock sweep_orphan_processes to return fixed orphan process details for dead worktrees
    def mock_sweep_orphan(worktree_path: str) -> list[dict[str, Any]]:
        if worktree_path == "/dead/worktree":
            return [
                {
                    "pid": 5000,
                    "name": "python.exe",
                    "command_line": "python script.py /dead/worktree",
                },
                {"pid": 5001, "name": "node.exe", "command_line": "node server.js /dead/worktree"},
            ]
        elif worktree_path == "/dead/worker":
            return [
                {
                    "pid": 6000,
                    "name": "python.exe",
                    "command_line": "python worker.py /dead/worker",
                },
            ]
        return []

    # Mock subprocess.run to track taskkill calls
    taskkill_calls = []
    original_run = subprocess.run

    def mock_subprocess_run(*args, **kwargs):
        if args and args[0] and args[0][0] == "taskkill":
            taskkill_calls.append(args[0])
            # Return a successful result
            return MagicMock(returncode=0, stdout="", stderr="")
        return original_run(*args, **kwargs)

    with (
        patch(
            "charlie_work.devin_shell.read_session_records",
            return_value=[dead_session, live_session],
        ),
        patch("charlie_work.claude_code.read_worker_records", return_value=[dead_worker]),
        patch("charlie_work.devin_shell.is_session_alive", side_effect=lambda r: r.pid != 1000),
        patch("charlie_work.claude_code.is_worker_alive", side_effect=lambda r: r.pid != 1002),
        patch(
            "charlie_work.dead_worker_reap.sweep_orphan_processes", side_effect=mock_sweep_orphan
        ),
        patch("os.name", "nt"),  # Force Windows path (os.name check lives in dead_worker_reap;
        # patching the os module directly avoids depending on which module happens to
        # `import os` into its own namespace)
        patch("subprocess.run", side_effect=mock_subprocess_run),
    ):
        from charlie_work.workflow import _sweep_orphan_processes_for_dead_sessions

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _sweep_orphan_processes_for_dead_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )

    # Verify taskkill was called for the orphan PIDs
    assert len(taskkill_calls) == 3
    killed_pids = [
        int(call[3]) for call in taskkill_calls
    ]  # Extract PID from taskkill /F /PID <pid>
    assert 5000 in killed_pids
    assert 5001 in killed_pids
    assert 6000 in killed_pids

    # Verify the event was logged
    state = load_state(paths.state_file)
    events = state.get("events", [])
    orphan_events = [e for e in events if e.get("kind") == "orphan_processes_killed"]
    assert len(orphan_events) == 2

    # Check the first event (dead/worktree)
    event1 = next(e for e in orphan_events if e["payload"]["worktree_path"] == "/dead/worktree")
    assert event1["payload"]["orphan_pids"] == [5000, 5001]
    assert event1["payload"]["killed_orphans"] == [5000, 5001]

    # Check the second event (dead/worker)
    event2 = next(e for e in orphan_events if e["payload"]["worktree_path"] == "/dead/worker")
    assert event2["payload"]["orphan_pids"] == [6000]
    assert event2["payload"]["killed_orphans"] == [6000]


def test_sweep_orphan_processes_called_from_production_loop(tmp_path: Path) -> None:
    """Integration test: verify _sweep_orphan_processes_for_dead_sessions is called from production loop (issue #139)."""
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub
    class FakeGitHubForSweep(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []

        def issue_list(self, labels=None, state=None):
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

        def pr_list(self):
            return []

    fake_gh = FakeGitHubForSweep()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Mock _sweep_orphan_processes_for_dead_sessions to track if it's called
    sweep_called = []

    def mock_sweep(*args, **kwargs):
        sweep_called.append(True)
        # Don't actually do anything

    with patch(
        "charlie_work.workflow._sweep_orphan_processes_for_dead_sessions", side_effect=mock_sweep
    ):
        # Run the production loop (loop calls the sweep)
        app.loop(limit=1)

    # Verify the sweep was called from the production loop
    assert len(sweep_called) == 1, (
        "Expected _sweep_orphan_processes_for_dead_sessions to be called from production loop"
    )


def test_watchdog_disabled_no_detection_no_kill_no_event(tmp_path: Path) -> None:
    """Issue #109: when watchdog.enabled=False, no detection, no kill, no event."""
    from datetime import UTC, datetime, timedelta
    import os
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        watchdog=WatchdogConfig(enabled=False, stall_minutes=20),  # Disabled
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Create a fake GitHub
    class FakeGitHubDisabled(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = []

        def issue_list(self, labels=None, state=None):
            # Support both old and new signature
            return self.issues

        def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
            return set()

    fake_gh = FakeGitHubDisabled()
    OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Create a fake stalled session sidecar
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file with old mtime (stalled by time)
    log_file = sessions_dir / "issue-109.log"
    log_file.write_text("working on issue\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    os.utime(log_file, (timestamp, timestamp))

    # Create a sidecar with a fake PID
    sidecar = sessions_dir / "issue-109.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 109,
                "branch": "agent/issue-109",
                "worktree_path": "/fake/path",
                "prompt_path": "/fake/prompt",
                "command": ["devin", "--print"],
                "pid": 99999,
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_file),
                "error": None,
                "process_start_time": 1234567890.0,  # Fake start time
            }
        ),
        encoding="utf-8",
    )

    # Mock is_session_alive and kill_process_tree to track calls
    with (
        patch("charlie_work.worker.is_session_alive", return_value=True) as mock_alive,
        patch("charlie_work.process_utils.kill_process_tree", return_value=[]) as mock_kill,
    ):
        # Run the stall detection and handling
        from charlie_work.workflow import _detect_and_handle_stalled_sessions

        _detect_and_handle_stalled_sessions(
            sessions_dir, paths.state_file, config, write_gate=_wg(paths.state_file)
        )

    # Check that is_session_alive was NOT called (detection skipped)
    mock_alive.assert_not_called()

    # Check that kill_process_tree was NOT called (no kill)
    mock_kill.assert_not_called()

    # Load state and check for the event
    state = load_state(paths.state_file)
    events = state.get("events", [])

    # No reap event of either kind may be emitted while the watchdog is off.
    #
    # This filter previously read e.get("type"), but events are keyed on
    # "kind" — so it matched nothing regardless of what was emitted and the
    # assertion below was true by construction. Found while splitting the
    # reap kinds for #873; fixed here rather than left as a silent no-op.
    # Both kinds are checked because #873 split the single "session_stalled"
    # into session_stalled (STALLED) + session_exited (DEAD), and a
    # disabled watchdog must emit neither.
    reap_events = [e for e in events if e.get("kind") in {"session_stalled", "session_exited"}]
    assert reap_events == []
