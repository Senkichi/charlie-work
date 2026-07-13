from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import os
import time

from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.claude_code import _sidecar_path as claude_sidecar_path
from charlie_work.config import OrchestratorConfig, PostMortemConfig, WatchdogConfig
from charlie_work.devin_shell import (
    SessionRecord,
    _sidecar_path as devin_sidecar_path,
    _write_json,
)
from charlie_work.post_mortem import ActivitySource, RealActivityProbe
from charlie_work.worker import WorkerHealth, WorkerView, _log_is_stalled_at_shim, iter_workers


def test_iter_workers_empty_dir(tmp_path: Path) -> None:
    """iter_workers on an empty sessions_dir returns []."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    workers = iter_workers(sessions_dir)
    assert workers == []


def test_iter_workers_mixed_adapters(tmp_path: Path) -> None:
    """iter_workers on a mix of devin-shell and claude-code sidecars returns two WorkerViews with correct adapter_kind tagging."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Write a devin-shell sidecar
    devin_sidecar = sessions_dir / "issue-1.json"
    devin_sidecar.write_text(
        json.dumps(
            {
                "issue_number": 1,
                "branch": "agent/issue-1",
                "worktree_path": "/tmp/worktree-1",
                "prompt_path": "/tmp/prompt-1.md",
                "command": ["devin", "prompt.md"],
                "pid": 12345,
                "started_at": "2026-07-06T00:00:00Z",
                "log_path": str(sessions_dir / "issue-1.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
            }
        ),
        encoding="utf-8",
    )

    # Write a claude-code sidecar
    claude_sidecar = sessions_dir / "issue-2.claude.json"
    claude_sidecar.write_text(
        json.dumps(
            {
                "issue_number": 2,
                "branch": "agent/issue-2",
                "worktree_path": "/tmp/worktree-2",
                "prompt_path": "/tmp/prompt-2.md",
                "command": ["claude", "prompt.md"],
                "pid": 67890,
                "started_at": "2026-07-06T00:00:00Z",
                "log_path": str(sessions_dir / "issue-2.claude.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
            }
        ),
        encoding="utf-8",
    )

    workers = iter_workers(sessions_dir)
    assert len(workers) == 2

    # Find devin worker
    devin_worker = next(w for w in workers if w.adapter_kind == "devin")
    assert devin_worker.issue_number == 1
    assert devin_worker.pid == 12345
    assert devin_worker.adapter_kind == "devin"

    # Find claude-code worker
    claude_worker = next(w for w in workers if w.adapter_kind == "claude-code")
    assert claude_worker.issue_number == 2
    assert claude_worker.pid == 67890
    assert claude_worker.adapter_kind == "claude-code"


def test_iter_workers_skips_malformed_sidecar(tmp_path: Path) -> None:
    """iter_workers skips a malformed/corrupt sidecar file without raising (matches read_session_records/read_worker_records' existing skip-on-OSError/JSONDecodeError behavior)."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Write a valid devin-shell sidecar
    devin_sidecar = sessions_dir / "issue-1.json"
    devin_sidecar.write_text(
        json.dumps(
            {
                "issue_number": 1,
                "branch": "agent/issue-1",
                "worktree_path": "/tmp/worktree-1",
                "prompt_path": "/tmp/prompt-1.md",
                "command": ["devin", "prompt.md"],
                "pid": 12345,
                "started_at": "2026-07-06T00:00:00Z",
                "log_path": str(sessions_dir / "issue-1.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
            }
        ),
        encoding="utf-8",
    )

    # Write a malformed sidecar (invalid JSON)
    malformed_sidecar = sessions_dir / "issue-2.json"
    malformed_sidecar.write_text("{ invalid json", encoding="utf-8")

    # Write a valid claude-code sidecar
    claude_sidecar = sessions_dir / "issue-3.claude.json"
    claude_sidecar.write_text(
        json.dumps(
            {
                "issue_number": 3,
                "branch": "agent/issue-3",
                "worktree_path": "/tmp/worktree-3",
                "prompt_path": "/tmp/prompt-3.md",
                "command": ["claude", "prompt.md"],
                "pid": 67890,
                "started_at": "2026-07-06T00:00:00Z",
                "log_path": str(sessions_dir / "issue-3.claude.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
            }
        ),
        encoding="utf-8",
    )

    workers = iter_workers(sessions_dir)
    assert len(workers) == 2  # Only the two valid sidecars
    assert all(w.issue_number in (1, 3) for w in workers)


def test_worker_view_is_alive_devin(monkeypatch: pytest.MonkeyPatch) -> None:
    """WorkerView.is_alive() for a devin-kind view delegates to is_session_alive-equivalent logic (mock the underlying PID probe)."""
    worker = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        process_start_time=1710000000.0,
        log_path="/tmp/issue-1.log",
        worktree_path="/tmp/worktree-1",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    # Mock is_session_alive in the worker module where it's imported
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    assert worker.is_alive() is True

    # Mock is_session_alive to return False
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: False)
    assert worker.is_alive() is False


def test_worker_view_is_alive_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    """WorkerView.is_alive() for a claude-code kind view delegates to is_worker_alive-equivalent logic (mock the underlying PID probe)."""
    worker = WorkerView(
        adapter_kind="claude-code",
        issue_number=2,
        repo_key="",
        pid=67890,
        started_at="2026-07-06T00:00:00Z",
        process_start_time=1710000000.0,
        log_path="/tmp/issue-2.claude.log",
        worktree_path="/tmp/worktree-2",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    # Mock is_worker_alive in the worker module where it's imported
    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: True)
    assert worker.is_alive() is True

    # Mock is_worker_alive to return False
    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: False)
    assert worker.is_alive() is False


def test_worker_view_is_alive_unknown_adapter() -> None:
    """WorkerView.is_alive() for an unknown adapter kind conservatively returns False."""
    worker = WorkerView(
        adapter_kind="unknown-adapter",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        process_start_time=1710000000.0,
        log_path="/tmp/issue-1.log",
        worktree_path="/tmp/worktree-1",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )
    assert worker.is_alive() is False


def test_worker_view_log_stat(tmp_path: Path) -> None:
    """WorkerView.log_stat() stats the log_path and returns the result, or None if the file doesn't exist."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_file = sessions_dir / "issue-1.log"
    log_file.write_text("test log content", encoding="utf-8")

    worker = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="/tmp/worktree-1",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    stat_result = worker.log_stat()
    assert stat_result is not None
    assert stat_result.st_size > 0

    # Test with non-existent file
    worker_missing_log = WorkerView(
        adapter_kind="devin",
        issue_number=2,
        repo_key="",
        pid=67890,
        started_at="2026-07-06T00:00:00Z",
        process_start_time=1710000000.0,
        log_path=str(sessions_dir / "issue-2.log"),
        worktree_path="/tmp/worktree-2",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    assert worker_missing_log.log_stat() is None


def test_claude_worker_record_from_dict_roundtrip() -> None:
    """ClaudeWorkerRecord.from_dict(): round-trip a to_dict() payload back through from_dict() and assert equality."""
    original = ClaudeWorkerRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/worktree-1",
        prompt_path="/tmp/prompt-1.md",
        command=("claude", "prompt.md"),
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        log_path="/tmp/issue-1.claude.log",
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        last_activity_at="2026-07-06T01:00:00Z",
        log_bytes=1024,
    )

    payload = original.to_dict()
    reconstructed = ClaudeWorkerRecord.from_dict(payload)
    assert reconstructed == original


def test_claude_worker_record_from_dict_missing_optional_fields() -> None:
    """A payload missing optional fields (reclaimed, process_start_time, last_activity_at, log_bytes) still constructs with the documented defaults."""
    payload = {
        "issue_number": 1,
        "branch": "agent/issue-1",
        "worktree_path": "/tmp/worktree-1",
        "prompt_path": "/tmp/prompt-1.md",
        "command": ["claude", "prompt.md"],
        "pid": 12345,
        "started_at": "2026-07-06T00:00:00Z",
        "log_path": "/tmp/issue-1.claude.log",
        # error, failure_kind, process_start_time, reclaimed, last_activity_at, log_bytes omitted
    }

    record = ClaudeWorkerRecord.from_dict(payload)
    assert record.issue_number == 1
    assert record.error is None
    assert record.failure_kind is None
    assert record.process_start_time is None
    assert record.reclaimed is None
    assert record.last_activity_at is None
    assert record.log_bytes is None


def test_claude_worker_record_log_stat_fields_roundtrip() -> None:
    """ClaudeWorkerRecord with last_activity_at and log_bytes fields round-trips through to_dict/from_dict."""
    original = ClaudeWorkerRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/worktree-1",
        prompt_path="/tmp/prompt-1.md",
        command=("claude", "prompt.md"),
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        log_path="/tmp/issue-1.claude.log",
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        last_activity_at="2026-07-06T01:30:45Z",
        log_bytes=2048,
    )

    payload = original.to_dict()
    assert "last_activity_at" in payload
    assert "log_bytes" in payload
    assert payload["last_activity_at"] == "2026-07-06T01:30:45Z"
    assert payload["log_bytes"] == 2048

    reconstructed = ClaudeWorkerRecord.from_dict(payload)
    assert reconstructed.last_activity_at == "2026-07-06T01:30:45Z"
    assert reconstructed.log_bytes == 2048


def test_worker_view_reap_sidecar_devin(tmp_path: Path) -> None:
    """WorkerView.reap_sidecar() deletes the devin-shell sidecar file for a dead session (issue #113)."""
    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Write a devin-shell sidecar
    issue_number = 1
    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 1,
                "branch": "agent/issue-1",
                "worktree_path": "/tmp/worktree-1",
                "prompt_path": "/tmp/prompt-1.md",
                "command": ["devin", "prompt.md"],
                "pid": 12345,
                "started_at": "2026-07-06T00:00:00Z",
                "log_path": str(sessions_dir / "issue-1.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
            }
        ),
        encoding="utf-8",
    )

    # Verify the sidecar exists
    assert sidecar_path.exists()

    # Create a WorkerView and reap the sidecar
    worker = WorkerView(
        adapter_kind="devin",
        issue_number=issue_number,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        process_start_time=1710000000.0,
        log_path=str(sessions_dir / "issue-1.log"),
        worktree_path="/tmp/worktree-1",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    worker.reap_sidecar(sessions_dir)

    # Verify the sidecar was deleted
    assert not sidecar_path.exists()


def test_devin_sidecars_never_populate_claude_progress_fields(tmp_path: Path) -> None:
    """Devin SessionRecord instances never have ClaudeProgress fields (issue #160).

    ClaudeProgress is specific to Claude Code workers (from events.jsonl).
    Devin sidecars should never populate these fields, and downstream code
    should treat their absence as "no signal," not "unhealthy."
    """
    from charlie_work.devin_shell import SessionRecord

    # Create a devin SessionRecord with all fields populated
    devin_record = SessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/worktree-1",
        prompt_path="/tmp/prompt-1.md",
        command=("devin", "prompt.md"),
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        log_path="/tmp/issue-1.log",
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        last_activity_at="2026-07-06T01:30:45Z",
        log_bytes=2048,
    )

    # Verify SessionRecord has no ClaudeProgress-related fields
    assert not hasattr(devin_record, "tool_call_count")
    assert not hasattr(devin_record, "turn_count")
    assert not hasattr(devin_record, "tokens")
    assert not hasattr(devin_record, "cost_usd")

    # Verify the same when round-tripping through to_dict/from_dict
    payload = devin_record.to_dict()
    assert "tool_call_count" not in payload
    assert "turn_count" not in payload
    assert "tokens" not in payload
    assert "cost_usd" not in payload

    reconstructed = SessionRecord.from_dict(payload)
    assert not hasattr(reconstructed, "tool_call_count")
    assert not hasattr(reconstructed, "turn_count")
    assert not hasattr(reconstructed, "tokens")
    assert not hasattr(reconstructed, "cost_usd")


def test_worker_view_reap_sidecar_claude(tmp_path: Path) -> None:
    """WorkerView.reap_sidecar() deletes the claude-code sidecar file for a dead session (issue #113)."""
    from charlie_work.claude_code import _sidecar_path as claude_sidecar_path

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Write a claude-code sidecar
    issue_number = 2
    sidecar_path = claude_sidecar_path(sessions_dir, issue_number)
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 2,
                "branch": "agent/issue-2",
                "worktree_path": "/tmp/worktree-2",
                "prompt_path": "/tmp/prompt-2.md",
                "command": ["claude", "prompt.md"],
                "pid": 67890,
                "started_at": "2026-07-06T00:00:00Z",
                "log_path": str(sessions_dir / "issue-2.claude.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
            }
        ),
        encoding="utf-8",
    )

    # Verify the sidecar exists
    assert sidecar_path.exists()

    # Create a WorkerView and reap the sidecar
    worker = WorkerView(
        adapter_kind="claude-code",
        issue_number=issue_number,
        repo_key="",
        pid=67890,
        started_at="2026-07-06T00:00:00Z",
        process_start_time=1710000000.0,
        log_path=str(sessions_dir / "issue-2.claude.log"),
        worktree_path="/tmp/worktree-2",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    worker.reap_sidecar(sessions_dir)

    # Verify the sidecar was deleted
    assert not sidecar_path.exists()


def test_worker_view_reap_sidecar_unknown_adapter(tmp_path: Path) -> None:
    """WorkerView.reap_sidecar() for an unknown adapter kind does nothing (no-op)."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create a WorkerView with an unknown adapter kind
    worker = WorkerView(
        adapter_kind="unknown-adapter",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        process_start_time=1710000000.0,
        log_path="/tmp/issue-1.log",
        worktree_path="/tmp/worktree-1",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    # Should not raise, just do nothing
    worker.reap_sidecar(sessions_dir)


def test_workflow_classify_dead_sessions_reaps_sidecar(tmp_path: Path) -> None:
    """Integration test: _classify_dead_sessions_and_update_throttle_state reaps sidecars for dead sessions (issue #113)."""
    from charlie_work.config import AutoMergeConfig, DevinConfig, OrchestratorConfig
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state
    from datetime import UTC, datetime
    import sys

    # Use command adapter to avoid needing real devin binary
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
    )

    # Create a sessions directory with a dead session
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log (empty is fine for this test)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text("Session log\n", encoding="utf-8")

    # Write a session record for a dead session (non-existent PID)
    issue_number = 123
    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path

    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    record = SessionRecord(
        issue_number=issue_number,
        branch="agent/issue-123",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=99999,  # Non-existent PID
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Verify the sidecar exists before the test
    assert sidecar_path.exists()

    # Create a fake GitHub instance (no PRs for this issue)
    class FakeGitHub:
        def __init__(self) -> None:
            self.issues = [
                {
                    "number": issue_number,
                    "title": "Test issue",
                    "url": "https://example.test/issues/123",
                    "body": "Test",
                    "labels": [{"name": config.labels.ready}],
                }
            ]
            self.prs = []
            self.labels_added = []
            self.labels_removed = []

        def issue_list(self, labels=None, state=None):
            return self.issues

        def issue_view(self, number: int):
            for issue in self.issues:
                if issue["number"] == number:
                    return issue
            raise ValueError(f"Issue {number} not found")

        def pr_list(self):
            return self.prs

        def add_issue_label(self, number: int, label: str) -> bool:
            self.labels_added.append((number, label))
            return True

        def remove_issue_label(self, number: int, label: str) -> bool:
            self.labels_removed.append((number, label))
            return True

    fake_gh = FakeGitHub()

    # Create a state file
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")

    # Run the production function that should reap the sidecar
    _classify_dead_sessions_and_update_throttle_state(sessions_dir, state_file, fake_gh, config)

    # Verify the sidecar was deleted as a side effect
    assert not sidecar_path.exists(), "Sidecar should be reaped after dead session classification"


def test_iter_workers_backward_compatibility(tmp_path: Path) -> None:
    """A fixture sidecar JSON written in the pre-refactor on-disk shape (captured from current to_dict() output) still loads via iter_workers — locks backward compatibility."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Write a devin-shell sidecar in the exact on-disk shape
    devin_sidecar = sessions_dir / "issue-1.json"
    devin_sidecar.write_text(
        json.dumps(
            {
                "issue_number": 1,
                "branch": "agent/issue-1",
                "worktree_path": "/tmp/worktree-1",
                "prompt_path": "/tmp/prompt-1.md",
                "command": ["devin", "prompt.md"],
                "pid": 12345,
                "started_at": "2026-07-06T00:00:00Z",
                "log_path": str(sessions_dir / "issue-1.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
            }
        ),
        encoding="utf-8",
    )

    # Write a claude-code sidecar in the exact on-disk shape
    claude_sidecar = sessions_dir / "issue-2.claude.json"
    claude_sidecar.write_text(
        json.dumps(
            {
                "issue_number": 2,
                "branch": "agent/issue-2",
                "worktree_path": "/tmp/worktree-2",
                "prompt_path": "/tmp/prompt-2.md",
                "command": ["claude", "prompt.md"],
                "pid": 67890,
                "started_at": "2026-07-06T00:00:00Z",
                "log_path": str(sessions_dir / "issue-2.claude.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
            }
        ),
        encoding="utf-8",
    )

    # Write a sidecar missing an optional field (reclaimed) to test legacy compatibility
    legacy_sidecar = sessions_dir / "issue-3.json"
    legacy_sidecar.write_text(
        json.dumps(
            {
                "issue_number": 3,
                "branch": "agent/issue-3",
                "worktree_path": "/tmp/worktree-3",
                "prompt_path": "/tmp/prompt-3.md",
                "command": ["devin", "prompt.md"],
                "pid": 54321,
                "started_at": "2026-07-06T00:00:00Z",
                "log_path": str(sessions_dir / "issue-3.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                # reclaimed omitted (legacy field)
            }
        ),
        encoding="utf-8",
    )

    workers = iter_workers(sessions_dir)
    assert len(workers) == 3

    # Verify all workers loaded correctly
    devin_worker = next(w for w in workers if w.issue_number == 1)
    assert devin_worker.adapter_kind == "devin"
    assert devin_worker.reclaimed is None

    claude_worker = next(w for w in workers if w.issue_number == 2)
    assert claude_worker.adapter_kind == "claude-code"
    assert claude_worker.reclaimed is None

    legacy_worker = next(w for w in workers if w.issue_number == 3)
    assert legacy_worker.adapter_kind == "devin"
    assert legacy_worker.reclaimed is None  # Should default to None for missing field


def test_log_is_stalled_at_shim_with_marker(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns True when log has shim marker and is stale."""
    log_path = tmp_path / "issue-1.log"
    # Write a log with the shim marker (typical frozen log size ~424-425 bytes)
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    # Set mtime to 10 minutes ago (past the default 5-minute grace period)
    old_time = datetime.now(UTC) - timedelta(minutes=10)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    assert _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_without_marker(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns False when log lacks shim marker."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("Some other log content\n", encoding="utf-8")

    # Set mtime to 10 minutes ago
    old_time = datetime.now(UTC) - timedelta(minutes=10)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    assert not _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_within_grace_period(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns False when log is within grace period."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    # Set mtime to 2 minutes ago (within the 5-minute grace period)
    old_time = datetime.now(UTC) - timedelta(minutes=2)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    assert not _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_large_log(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns False when log is large (>1KB)."""
    log_path = tmp_path / "issue-1.log"
    # Write a large log with the shim marker
    large_content = "[shim] .devin infra materialized\n" + "x" * 2000
    log_path.write_text(large_content, encoding="utf-8")

    # Set mtime to 10 minutes ago
    old_time = datetime.now(UTC) - timedelta(minutes=10)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    assert not _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_nonexistent_log(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns False when log file doesn't exist."""
    log_path = tmp_path / "issue-1.log"
    now = datetime.now(UTC)
    assert not _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_with_fresh_real_activity(tmp_path: Path) -> None:
    """Issue #280: frozen sidecar log is ignored when real-session activity is fresh."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    fresh_timestamp = now - timedelta(minutes=1)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=fresh_timestamp,
                staleness_seconds=(now - fresh_timestamp).total_seconds(),
                error=None,
            ),
        )
    )

    assert not _log_is_stalled_at_shim(
        log_path, grace_minutes=5, now=now, real_activity_probe=probe
    )


def test_log_is_stalled_at_shim_with_stale_real_activity(tmp_path: Path) -> None:
    """Issue #280: launch stall is still detected when real activity is also stale."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    stale_timestamp = now - timedelta(minutes=10)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=stale_timestamp,
                staleness_seconds=(now - stale_timestamp).total_seconds(),
                error=None,
            ),
        )
    )

    assert _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now, real_activity_probe=probe)


# ---------------------------------------------------------------------------
# Rate-limit stall deferral (issue #247)
# ---------------------------------------------------------------------------


def _make_stalled_devin_session(
    tmp_path: Path,
    issue_number: int,
    log_text: str,
    *,
    rate_limit_defer_until: str | None = None,
) -> tuple[Path, Path, Path]:
    """Create a sessions directory, sidecar, and stale log for a live worker."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text(log_text, encoding="utf-8")
    # Set mtime to 30 minutes ago so the log looks stalled at the default
    # 20-minute stall threshold.
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    record = SessionRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("devin", "--prompt-file", str(tmp_path / "prompt.md")),
        pid=99999,
        started_at=(datetime.now(UTC) - timedelta(minutes=31)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        last_activity_at=old_time.isoformat().replace("+00:00", "Z"),
        log_bytes=log_path.stat().st_size,
        rate_limit_defer_until=rate_limit_defer_until,
    )
    _write_json(sidecar_path, record.to_dict())
    state_file = tmp_path / "state.json"
    return sessions_dir, state_file, log_path


def test_stalled_worker_with_rate_limit_signature_is_deferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled-looking worker whose log tail contains a rate-limit signature is deferred, not killed."""
    from charlie_work import workflow

    issue_number = 247
    log_text = (
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n"
    )
    sessions_dir, state_file, _ = _make_stalled_devin_session(tmp_path, issue_number, log_text)

    killed = []
    monkeypatch.setattr(
        workflow, "kill_process_tree", lambda pid, start_time: killed.append(pid) or [pid]
    )
    monkeypatch.setattr(workflow, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    result = workflow._detect_and_handle_stalled_sessions(sessions_dir, state_file, config)

    assert result == []
    assert killed == []

    sidecar = json.loads(
        (devin_sidecar_path(sessions_dir, issue_number)).read_text(encoding="utf-8")
    )
    assert sidecar["rate_limit_defer_until"] is not None
    defer_until = datetime.fromisoformat(sidecar["rate_limit_defer_until"].replace("Z", "+00:00"))
    expected_min = datetime.now(UTC) + timedelta(minutes=10 + 2 - 1)
    expected_max = datetime.now(UTC) + timedelta(minutes=10 + 2 + 1)
    assert expected_min <= defer_until <= expected_max

    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = [e for e in state.get("events", []) if e.get("kind") == "session_rate_limit_deferred"]
    assert len(events) == 1
    assert events[0]["payload"]["issue_number"] == issue_number
    assert events[0]["payload"]["defer_until"] == sidecar["rate_limit_defer_until"]


def test_deferred_worker_log_resumes_exits_deferred_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deferred worker whose log resumes growing exits the deferred state and is not killed."""
    from charlie_work import workflow

    issue_number = 248
    log_text = (
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n"
    )
    future_defer = (datetime.now(UTC) + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    sessions_dir, state_file, log_path = _make_stalled_devin_session(
        tmp_path, issue_number, log_text, rate_limit_defer_until=future_defer
    )

    # Now the log resumes: update mtime and size to "now".
    log_path.write_text(log_text + "Resumed work after provider window reset\n", encoding="utf-8")
    recent_time = datetime.now(UTC) - timedelta(minutes=1)
    os.utime(log_path, (recent_time.timestamp(), recent_time.timestamp()))

    killed = []
    monkeypatch.setattr(
        workflow, "kill_process_tree", lambda pid, start_time: killed.append(pid) or [pid]
    )
    monkeypatch.setattr(workflow, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    result = workflow._detect_and_handle_stalled_sessions(sessions_dir, state_file, config)

    assert result == []
    assert killed == []

    sidecar = json.loads(
        (devin_sidecar_path(sessions_dir, issue_number)).read_text(encoding="utf-8")
    )
    assert sidecar["rate_limit_defer_until"] is None


def test_deferred_worker_past_deadline_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deferred worker still silent past the deadline is killed and classified via the rate-limit path."""
    from charlie_work import workflow

    issue_number = 249
    log_text = (
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n"
    )
    past_defer = (datetime.now(UTC) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    sessions_dir, state_file, _ = _make_stalled_devin_session(
        tmp_path, issue_number, log_text, rate_limit_defer_until=past_defer
    )

    killed = []
    monkeypatch.setattr(
        workflow, "kill_process_tree", lambda pid, start_time: killed.append(pid) or [pid]
    )
    monkeypatch.setattr(workflow, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    result = workflow._detect_and_handle_stalled_sessions(sessions_dir, state_file, config)

    assert result == [{"issue": issue_number, "pid": 99999}]
    assert killed == [99999]

    sidecar = json.loads(
        (devin_sidecar_path(sessions_dir, issue_number)).read_text(encoding="utf-8")
    )
    assert sidecar["failure_kind"] == "rate_limited"
    # The sidecar still carries the expired defer deadline; the global throttle
    # state is set to the freshly computed cooldown from the log tail.
    assert sidecar["rate_limit_defer_until"] is not None

    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state.get("throttled_until") is not None
    throttled_until = datetime.fromisoformat(state["throttled_until"].replace("Z", "+00:00"))
    expected_min = datetime.now(UTC) + timedelta(minutes=10 - 1)
    expected_max = datetime.now(UTC) + timedelta(minutes=10 + 1)
    assert expected_min <= throttled_until <= expected_max


def test_stalled_worker_without_rate_limit_signature_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled worker with a genuinely quiet tail (no rate-limit signature) keeps existing stall-kill behavior."""
    from charlie_work import workflow

    issue_number = 250
    log_text = "Working on task...\nLast line\n"
    sessions_dir, state_file, _ = _make_stalled_devin_session(tmp_path, issue_number, log_text)

    killed = []
    monkeypatch.setattr(
        workflow, "kill_process_tree", lambda pid, start_time: killed.append(pid) or [pid]
    )
    monkeypatch.setattr(workflow, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    result = workflow._detect_and_handle_stalled_sessions(sessions_dir, state_file, config)

    assert result == [{"issue": issue_number, "pid": 99999}]
    assert killed == [99999]

    sidecar = json.loads(
        (devin_sidecar_path(sessions_dir, issue_number)).read_text(encoding="utf-8")
    )
    assert sidecar["failure_kind"] == "stalled"
    assert sidecar.get("rate_limit_defer_until") is None


def test_detect_stalled_sessions_passes_real_activity_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #301: workflow.py stall paths must construct and pass a RealActivityProbe.

    A future edit that drops the probe argument will make this test fail because
    the spy will receive None instead of a constructed probe.
    """
    from charlie_work import workflow

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    issue_number = 301

    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    log_path.write_text("Working on task...\nLast line", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_path, (time.time(), old_time.timestamp()))

    events_path = sessions_dir / f"issue-{issue_number}.events.jsonl"
    fresh_time = datetime.now(UTC) - timedelta(minutes=1)
    events_path.write_text(
        f'{{"type": "tool_call", "timestamp": "{fresh_time.isoformat()}"}}\n',
        encoding="utf-8",
    )
    os.utime(events_path, (time.time(), fresh_time.timestamp()))

    sidecar_path = claude_sidecar_path(sessions_dir, issue_number)
    record = ClaudeWorkerRecord(
        issue_number=issue_number,
        branch="agent/issue-301",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("claude", "prompt.md"),
        pid=99999,
        started_at=(datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    captured: list[RealActivityProbe | None] = []

    def spy_classify(
        view: WorkerView,
        config: OrchestratorConfig,
        now: datetime,
        real_activity_probe: RealActivityProbe | None = None,
    ) -> WorkerHealth:
        captured.append(real_activity_probe)
        return WorkerHealth.HEALTHY

    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: True)
    monkeypatch.setattr("charlie_work.worker.classify_worker_health", spy_classify)

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    result = workflow._detect_stalled_sessions(sessions_dir, config)

    assert result == []
    assert len(captured) == 1
    assert captured[0] is not None
    assert isinstance(captured[0], RealActivityProbe)
    assert captured[0].latest_source == "claude_events_jsonl"
