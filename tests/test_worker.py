from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.worker import WorkerView, iter_workers


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
