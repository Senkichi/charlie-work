from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import os
import subprocess
import time

from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.claude_code import _sidecar_path as claude_sidecar_path
from charlie_work.config import (
    AutoMergeConfig,
    DevinConfig,
    OrchestratorConfig,
    PostMortemConfig,
    WatchdogConfig,
)
from charlie_work.devin_shell import (
    SessionRecord,
    _sidecar_path as devin_sidecar_path,
    _write_json,
)
from charlie_work.post_mortem import ActivitySource, RealActivityProbe, real_activity_for_worker
from charlie_work.worker import WorkerHealth, WorkerView, _log_is_stalled_at_shim, iter_workers
from charlie_work.write_gate import WriteGate


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


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


def test_session_record_started_at_canonical_at_construction() -> None:
    """Issue #354: SessionRecord normalizes a non-canonical started_at to UTC/Z/no-microseconds."""
    record = SessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/worktree-1",
        prompt_path="/tmp/prompt-1.md",
        command=("devin",),
        pid=12345,
        started_at="2026-07-13T16:04:45.267596+00:00",
        log_path="/tmp/issue-1.log",
    )
    assert record.started_at == "2026-07-13T16:04:45Z"


def test_claude_worker_record_started_at_canonical_at_construction() -> None:
    """Issue #354: ClaudeWorkerRecord normalizes a non-canonical started_at to UTC/Z/no-microseconds."""
    record = ClaudeWorkerRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/worktree-1",
        prompt_path="/tmp/prompt-1.md",
        command=("claude",),
        pid=12345,
        started_at="2026-07-13T16:04:45.267596+00:00",
        log_path="/tmp/issue-1.claude.log",
    )
    assert record.started_at == "2026-07-13T16:04:45Z"


def test_session_record_from_dict_rejects_missing_started_at() -> None:
    """Issue #354: a foreign payload (e.g. a post-mortem sidecar) lacks started_at;
    from_dict must raise instead of producing an unparseable empty string."""
    payload = {
        "issue_number": 203,
        "generated_at": "2026-07-13T19:52:10.811933+00:00",
        "matched": False,
        "db_path": "C:/fake/sessions.db",
    }
    with pytest.raises(ValueError, match="started_at"):
        SessionRecord.from_dict(payload)


def test_claude_worker_record_from_dict_rejects_missing_started_at() -> None:
    """Issue #354: a foreign payload lacks started_at; from_dict must raise."""
    payload = {
        "issue_number": 203,
        "generated_at": "2026-07-13T19:52:10.811933+00:00",
        "matched": False,
    }
    with pytest.raises(ValueError, match="started_at"):
        ClaudeWorkerRecord.from_dict(payload)


def test_session_record_from_dict_derives_started_at_from_process_start_time() -> None:
    """Issue #354: if started_at is missing but process_start_time is present, recover the canonical start timestamp."""
    payload = {
        "issue_number": 1,
        "branch": "agent/issue-1",
        "worktree_path": "/tmp/worktree-1",
        "prompt_path": "/tmp/prompt-1.md",
        "command": ["devin", "prompt.md"],
        "pid": 12345,
        "started_at": "",
        "log_path": "/tmp/issue-1.log",
        "process_start_time": 1710000000.0,
    }
    record = SessionRecord.from_dict(payload)
    assert record.started_at == "2024-03-09T16:00:00Z"


def test_claude_worker_record_from_dict_derives_started_at_from_process_start_time() -> None:
    """Issue #354: if started_at is missing but process_start_time is present, recover the canonical start timestamp."""
    payload = {
        "issue_number": 1,
        "branch": "agent/issue-1",
        "worktree_path": "/tmp/worktree-1",
        "prompt_path": "/tmp/prompt-1.md",
        "command": ["claude", "prompt.md"],
        "pid": 12345,
        "started_at": "",
        "log_path": "/tmp/issue-1.claude.log",
        "process_start_time": 1710000000.0,
    }
    record = ClaudeWorkerRecord.from_dict(payload)
    assert record.started_at == "2024-03-09T16:00:00Z"


def test_canonical_started_at_normalizes_non_utc_offsets_to_utc() -> None:
    """Issue #354: _canonical_started_at must normalize non-UTC offsets to UTC/Z."""
    from charlie_work.state import _canonical_started_at

    assert _canonical_started_at("2026-07-13T16:04:45+02:00") == "2026-07-13T14:04:45Z"
    assert _canonical_started_at("2026-07-13T16:04:45.267596+02:00") == "2026-07-13T14:04:45Z"
    assert _canonical_started_at("2026-07-13T16:04:45-07:00") == "2026-07-13T23:04:45Z"


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


def test_worker_view_is_alive_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """WorkerView.is_alive() for an api-kind view delegates to is_worker_alive
    (api workers are Claude Code CLI processes with provider env injected)."""
    worker = WorkerView(
        adapter_kind="api",
        issue_number=3,
        repo_key="",
        pid=54321,
        started_at="2026-07-22T00:00:00Z",
        process_start_time=1710000000.0,
        log_path="/tmp/issue-3.claude.log",
        worktree_path="/tmp/worktree-3",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: True)
    assert worker.is_alive() is True

    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: False)
    assert worker.is_alive() is False


def test_worker_view_reap_sidecar_api(tmp_path: Path) -> None:
    """WorkerView.reap_sidecar() deletes the issue-<n>.api.json sidecar file
    for a dead api worker session (issue #478 third-arm)."""
    from charlie_work.claude_code import _sidecar_path as claude_sidecar_path

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    issue_number = 3
    sidecar_path = claude_sidecar_path(sessions_dir, issue_number, "api")
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 3,
                "branch": "agent/issue-3",
                "worktree_path": "/tmp/worktree-3",
                "prompt_path": "/tmp/prompt-3.md",
                "command": ["claude"],
                "pid": 54321,
                "started_at": "2026-07-22T00:00:00Z",
                "log_path": str(sessions_dir / "issue-3.claude.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
                "adapter_kind": "api",
                "provider": "kimi-k3",
            }
        ),
        encoding="utf-8",
    )

    assert sidecar_path.exists()

    worker = WorkerView(
        adapter_kind="api",
        issue_number=issue_number,
        repo_key="",
        pid=54321,
        started_at="2026-07-22T00:00:00Z",
        process_start_time=1710000000.0,
        log_path=str(sessions_dir / "issue-3.claude.log"),
        worktree_path="/tmp/worktree-3",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    worker.reap_sidecar(sessions_dir)

    assert not sidecar_path.exists()


def test_iter_workers_surfaces_api_sidecars(tmp_path: Path) -> None:
    """iter_workers reads issue-<n>.api.json sidecars and tags them
    adapter_kind='api' (issue #478 third-arm)."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    api_sidecar = sessions_dir / "issue-4.api.json"
    api_sidecar.write_text(
        json.dumps(
            {
                "issue_number": 4,
                "branch": "agent/issue-4",
                "worktree_path": "/tmp/worktree-4",
                "prompt_path": "/tmp/prompt-4.md",
                "command": ["claude"],
                "pid": 44444,
                "started_at": "2026-07-22T00:00:00Z",
                "log_path": str(sessions_dir / "issue-4.claude.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
                "adapter_kind": "api",
                "provider": "kimi-k3",
            }
        ),
        encoding="utf-8",
    )

    workers = iter_workers(sessions_dir)
    api_workers = [w for w in workers if w.adapter_kind == "api"]
    assert len(api_workers) == 1
    assert api_workers[0].issue_number == 4
    assert api_workers[0].adapter_kind == "api"


def test_update_worker_log_stat_api_writes_api_sidecar(tmp_path: Path) -> None:
    """update_worker_log_stat for an api-kind worker writes back to the
    issue-<n>.api.json sidecar (not the .claude.json sidecar)."""
    from charlie_work.claude_code import _sidecar_path as claude_sidecar_path
    from charlie_work.worker import update_worker_log_stat

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    issue_number = 5
    log_path = sessions_dir / "issue-5.claude.log"
    log_path.write_text("log content\n", encoding="utf-8")

    sidecar_path = claude_sidecar_path(sessions_dir, issue_number, "api")
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 5,
                "branch": "agent/issue-5",
                "worktree_path": "/tmp/worktree-5",
                "prompt_path": "/tmp/prompt-5.md",
                "command": ["claude"],
                "pid": 55555,
                "started_at": "2026-07-22T00:00:00Z",
                "log_path": str(log_path),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
                "adapter_kind": "api",
                "provider": "kimi-k3",
                "last_activity_at": None,
                "log_bytes": None,
            }
        ),
        encoding="utf-8",
    )

    worker = WorkerView(
        adapter_kind="api",
        issue_number=issue_number,
        repo_key="",
        pid=55555,
        started_at="2026-07-22T00:00:00Z",
        process_start_time=1710000000.0,
        log_path=str(log_path),
        worktree_path="/tmp/worktree-5",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    update_worker_log_stat(sessions_dir, worker)

    # The .api.json sidecar was updated; no .claude.json sidecar was created.
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["last_activity_at"] is not None
    assert payload["log_bytes"] is not None
    assert not (sessions_dir / "issue-5.claude.json").exists()


def test_workflow_classify_dead_sessions_reaps_sidecar(tmp_path: Path) -> None:
    """Integration test: _classify_dead_sessions_and_update_throttle_state reaps sidecars for dead sessions (issue #113)."""
    from charlie_work.config import (
        AutoMergeConfig,
        DevinConfig,
        OrchestratorConfig,
        WatchdogConfig,
    )
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
        # Issue #343: the dead-session lane now corroborates a not-alive pid
        # against real-session activity before reaping (never fail-open on a
        # merely-inconclusive probe). This test's concern is the reap-on-dead
        # mechanics (issue #113), not corroboration timing, so pin the
        # deferral cap to 0 to keep its original immediate-reap assertion --
        # a bare test environment with no real sessions.db always produces an
        # inconclusive probe, which would otherwise defer for
        # max_inconclusive_probe_deferrals passes before reaping.
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=0),
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
    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, fake_gh, config, write_gate=_wg(state_file)
    )

    # Verify the sidecar was deleted as a side effect
    assert not sidecar_path.exists(), "Sidecar should be reaped after dead session classification"


def test_workflow_classify_dead_sessions_reaps_probe_error_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #426: a dead-pid sidecar with a stale ``probe_error``/``live_worker_redispatch_averted``
    record is not invisible to the confirmed-dead lane. It is deferred up to
    ``max_inconclusive_probe_deferrals`` and then reaped/relabeled.
    """
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe
    import sys

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
        # Cap at 1 so the test reaches the reap in two passes.
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=1),
    )

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-123.log"
    log_path.write_text("Session log\n", encoding="utf-8")

    issue_number = 123
    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    record = SessionRecord(
        issue_number=issue_number,
        branch="agent/issue-123",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=99999,  # Dead PID
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="probe_error",
        failure_kind="live_worker_redispatch_averted",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    class FakeGitHub:
        def __init__(self) -> None:
            self.issues = [
                {
                    "number": issue_number,
                    "title": "Test issue",
                    "url": "https://example.test/issues/123",
                    "body": "Test",
                    "labels": [{"name": config.labels.in_progress}],
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

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")

    # Force an all-errored (inconclusive) real-activity probe so the test is
    # deterministic and exercises the deferral cap.
    def _inconclusive_probe(*_args: object, **_kwargs: object) -> RealActivityProbe:
        return RealActivityProbe(
            sources=(
                ActivitySource(
                    name="sessions.db",
                    timestamp=None,
                    staleness_seconds=None,
                    error="no session found matching working_directory",
                ),
                ActivitySource(
                    name="devin_per_pid_log",
                    timestamp=None,
                    staleness_seconds=None,
                    error="no per-PID log found",
                ),
            )
        )

    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _inconclusive_probe)
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda _record: False)

    # First pass: defer and advance the counter.
    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, fake_gh, config, write_gate=_wg(state_file)
    )
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar.get("inconclusive_probe_deferred_count") == 1
    assert fake_gh.labels_removed == []
    assert fake_gh.labels_added == []

    # Second pass: deferral cap reached, sidecar reaped and issue relabeled.
    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, fake_gh, config, write_gate=_wg(state_file)
    )
    assert not sidecar_path.exists()
    assert (123, config.labels.in_progress) in fake_gh.labels_removed
    assert (123, config.labels.ready) in fake_gh.labels_added


# --- helpers for issue #656 workflow call-site regression coverage ----------


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True)


def _init_bare_remote_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """A bare remote + a non-bare clone with one commit on main, pushed."""
    remote = tmp_path / "remote.git"
    remote.mkdir(parents=True, exist_ok=True)
    _git(remote, "init", "--bare", "--initial-branch=main")
    clone = tmp_path / "clone"
    clone.mkdir(parents=True, exist_ok=True)
    _git(clone, "init", "--initial-branch=main")
    _git(clone, "config", "user.email", "test@example.test")
    _git(clone, "config", "user.name", "Test User")
    _git(clone, "config", "commit.gpgSign", "false")
    _git(clone, "remote", "add", "origin", str(remote))
    (clone / "README.md").write_text("hello\n", encoding="utf-8")
    _git(clone, "add", "README.md")
    _git(clone, "commit", "-m", "initial commit")
    _git(clone, "push", "-u", "origin", "main")
    return remote, clone


def _setup_completed_worktree(repo_root: Path, issue_number: int) -> tuple[Path, str]:
    """Create a worktree with one commit beyond origin/main (WorktreeState.COMPLETED)."""
    from charlie_work.worktree import create_worktree

    branch = f"agent/issue-{issue_number}"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")
    return info.path, branch


# Adapter-kind -> sidecar filename suffix (mirrors claude_code sidecar naming).
_WF_ADAPTER_SIDECAR_SUFFIX = {"devin": "", "claude-code": ".claude", "api": ".api"}


def _wf_write_dead_session_sidecar(
    sessions_dir: Path,
    issue_number: int,
    branch: str,
    worktree_path: Path,
    adapter_kind: str,
    log_text: str,
) -> Path:
    """Write a dead-session sidecar (pid=None) for any adapter kind + its log."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text(log_text, encoding="utf-8")
    suffix = _WF_ADAPTER_SIDECAR_SUFFIX[adapter_kind]
    sidecar_path = sessions_dir / f"issue-{issue_number}{suffix}.json"
    if adapter_kind == "devin":
        record = SessionRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree_path),
            prompt_path="/tmp/prompt.md",
            command=("devin", "--prompt-file", "/tmp/prompt.md"),
            pid=None,
            started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            log_path=str(log_path),
            error=None,
        )
        sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    else:
        # claude-code / api share the ClaudeWorkerRecord on-disk shape; the
        # ``adapter_kind`` field disambiguates api from claude-code.
        sidecar_path.write_text(
            json.dumps(
                {
                    "issue_number": issue_number,
                    "branch": branch,
                    "worktree_path": str(worktree_path),
                    "prompt_path": "/tmp/prompt.md",
                    "command": ["claude", "-p"],
                    "pid": None,
                    "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "log_path": str(log_path),
                    "error": None,
                    "adapter_kind": adapter_kind,
                }
            ),
            encoding="utf-8",
        )
    return sidecar_path


class _NoOpGitHub:
    """Minimal GitHub stub for the dead-session lane.

    Returns no open PRs and an issue carrying only a non-active label, so the
    post-classification relabel/salvage block hits ``if not active_labels:
    continue`` and the test's concern -- the throttle-state write -- is
    exercised in isolation. ``repo_root`` is deliberately NOT set so the
    salvage path (which needs push/PR creation) is skipped.
    """

    def __init__(self, issue_number: int, label: str) -> None:
        self._issue = {
            "number": issue_number,
            "title": f"issue {issue_number}",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "",
            "labels": [{"name": label}],
            "state": "OPEN",
        }
        self.labels_added: list[tuple[int, str]] = []
        self.labels_removed: list[tuple[int, str]] = []

    def pr_list(self) -> list[dict]:
        return []

    def issue_view(self, number: int):
        if number != self._issue["number"]:
            raise ValueError(f"issue {number} not found")
        return self._issue

    def add_issue_label(self, number: int, label: str) -> bool:
        self.labels_added.append((number, label))
        return True

    def remove_issue_label(self, number: int, label: str) -> bool:
        self.labels_removed.append((number, label))
        return True


@pytest.mark.parametrize("adapter_kind", ["devin", "claude-code", "api"])
def test_workflow_classify_dead_sessions_completed_skips_log_tail_throttle(
    tmp_path: Path, adapter_kind: str
) -> None:
    """Issue #656 regression: a completed worktree's log-tail throttle markers
    must NOT set ``throttled_until`` in state.json.

    Guards the three ``session_completed=True`` call sites in
    ``workflow._classify_dead_sessions_and_update_throttle_state`` (one per
    adapter kind, including the claude-code branch that caused the live #651
    incident). The log file is seeded with ``"usage limit"`` -- a
    ``_QUOTA_EXHAUSTED_PATTERN`` substring that, if log-tail classification
    ran, would return ``quota_exhausted`` plus a 24h ``throttled_until`` and
    write it into state.json. The worktree inspection is ground truth the
    session completed, so ``session_completed=True`` must skip log-tail
    matching entirely and leave ``failure_kind="unpublished_work"``.

    If ``session_completed=True`` is silently dropped from any of the three
    call sites, this test fails on both assertions: ``failure_kind`` becomes
    ``quota_exhausted`` and ``state["throttled_until"]`` is set to a 24h
    future timestamp.
    """
    import sys

    from charlie_work.state import load_state
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
        # pid=None skips the corroboration/deferral lane entirely, so the
        # deferral cap is irrelevant here; pin it for determinism anyway.
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=0),
    )

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    issue_number = 656
    worktree_path, branch = _setup_completed_worktree(repo_root, issue_number)

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    sidecar_path = _wf_write_dead_session_sidecar(
        sessions_dir,
        issue_number,
        branch,
        worktree_path,
        adapter_kind,
        log_text=(
            '## Summary\n\nFixed generic substrings ("rate limit", "usage limit") '
            "that legitimately appear in this codebase's rate-limit/quota domain.\n"
        ),
    )

    # Issue carries only a non-active label so the relabel/salvage block
    # short-circuits and the throttle-state write is the only side effect.
    gh = _NoOpGitHub(issue_number, config.labels.ready)

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")

    reaped = _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    # The dead lane ran and reaped the sidecar.
    assert len(reaped) == 1
    assert reaped[0]["adapter_kind"] == adapter_kind
    # The is_completed lane was taken: failure_kind is the caller's
    # "unpublished_work" fallback, NOT log-tail-derived "quota_exhausted".
    assert reaped[0]["failure_kind"] == "unpublished_work", (
        f"completed {adapter_kind} session was reclassified from log tail "
        f"(issue #656 regression): {reaped[0]['failure_kind']}"
    )
    assert not sidecar_path.exists()

    # The throttle must NOT fire despite the "usage limit" marker in the log --
    # session_completed=True skipped log-tail classification entirely, so no
    # throttled_until window was written into state.json.
    final_state = load_state(state_file)
    assert final_state.get("throttled_until") is None, (
        f"completed {adapter_kind} session set a fleet-wide throttle despite "
        f"session_completed=True (issue #656 regression): "
        f"{final_state.get('throttled_until')}"
    )


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


def test_log_is_stalled_at_shim_with_all_errored_probe_deferred(tmp_path: Path) -> None:
    """Issue #307 scope-extension: _log_is_stalled_at_shim must not fail open on an
    all-errored probe.

    This is the site the reviewer reproduced directly: reconcile.py:264 reaches
    this function only for a CONFIRMED-ALIVE worker, and a True return here
    drives an immediate kill_process_tree (reconcile.py:288). An all-errored
    probe is insufficient evidence of a stall.
    """
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=None,
                staleness_seconds=None,
                error="message_nodes query failed (schema drift?): no such column: id",
            ),
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error="no per-PID log found",
            ),
        )
    )

    assert not _log_is_stalled_at_shim(
        log_path, grace_minutes=5, now=now, real_activity_probe=probe
    )


def test_log_is_stalled_at_shim_with_no_match_yet_probe_deferred(tmp_path: Path) -> None:
    """Issue #307 scope-extension: the second inconclusive shape at the shim site.

    Distinct from test_log_is_stalled_at_shim_with_all_errored_probe_deferred:
    here every source is error-free but returned no timestamp match at all
    (e.g. a young session within launch_stall_grace_minutes whose sessions.db
    row hasn't landed yet). This must also defer rather than report a stall.
    """
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=None,
                staleness_seconds=None,
                error=None,
            ),
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error=None,
            ),
        )
    )

    assert not _log_is_stalled_at_shim(
        log_path, grace_minutes=5, now=now, real_activity_probe=probe
    )


def test_log_is_stalled_at_shim_worktree_files_mtime_fresh_beyond_grace(
    tmp_path: Path,
) -> None:
    """Issue #353: worktree mtime freshness uses its own generous threshold."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    now = datetime.now(UTC)
    old_time = now - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    source_file = worktree_path / "foo.py"
    source_file.write_text("# change", encoding="utf-8")
    worktree_mtime = now - timedelta(minutes=8)
    os.utime(source_file, (worktree_mtime.timestamp(), worktree_mtime.timestamp()))

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
    )
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        (now - timedelta(minutes=10)).isoformat(),
        None,
        now,
        watchdog_config=watchdog,
    )

    # 8 minutes is past the 5-minute grace, but within the 45-minute worktree threshold.
    assert not _log_is_stalled_at_shim(
        log_path, grace_minutes=5, now=now, real_activity_probe=probe
    )


def test_log_is_stalled_at_shim_worktree_files_mtime_checkout_noise_stalls(
    tmp_path: Path,
) -> None:
    """Issue #353: checkout-time mtimes do not mask a launch stall."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    now = datetime.now(UTC)
    old_time = now - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    source_file = worktree_path / "foo.py"
    source_file.write_text("# change", encoding="utf-8")
    started_at = now - timedelta(minutes=30)
    os.utime(source_file, (started_at.timestamp(), started_at.timestamp()))

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
    )
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        started_at.isoformat(),
        None,
        now,
        watchdog_config=watchdog,
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


def _stale_devin_probe(*_args: object, **_kwargs: object) -> RealActivityProbe:
    """Return a probe that is stale (not fresh) and not all-errored.

    Issue #307: a worker with a stale sidecar log and a stale real-session
    activity signal must still be classified as STALLED. Tests that exercise
    the rate-limit defer path must not be tripped up by an all-errored probe,
    which now defers to avoid the fail-open bug in Signal 3.
    """
    now = datetime.now(UTC)
    timestamp = now - timedelta(minutes=30)
    return RealActivityProbe(
        sources=(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=timestamp,
                staleness_seconds=(now - timestamp).total_seconds(),
                error=None,
            ),
        )
    )


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
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    # Issue #828: freeze the clock and inject it so the assertion below is an
    # exact equality, not a wall-clock-tolerance window that a CI stall (the
    # same failure class as PRs #700/#690) can blow through.
    frozen_now = datetime.now(UTC)
    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, now=frozen_now
    )

    assert result == []
    assert killed == []

    sidecar = json.loads(
        (devin_sidecar_path(sessions_dir, issue_number)).read_text(encoding="utf-8")
    )
    assert sidecar["rate_limit_defer_until"] is not None
    defer_until = datetime.fromisoformat(sidecar["rate_limit_defer_until"].replace("Z", "+00:00"))
    margin_seconds = config.runtime.throttle_resume_margin_s
    expected = (frozen_now + timedelta(minutes=10 + 2, seconds=margin_seconds)).replace(
        microsecond=0
    )
    assert defer_until == expected

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
    # Issue #828: freeze the clock and inject it into the call below so the
    # deadline comparison is deterministic rather than depending on two
    # independently-sampled wall-clock reads (setup here vs. the sweep's own
    # internal sample) staying within 5 minutes of each other under CI load.
    frozen_now = datetime.now(UTC)
    past_defer = (frozen_now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    sessions_dir, state_file, _ = _make_stalled_devin_session(
        tmp_path, issue_number, log_text, rate_limit_defer_until=past_defer
    )

    killed = []
    monkeypatch.setattr(
        workflow, "kill_process_tree", lambda pid, start_time: killed.append(pid) or [pid]
    )
    monkeypatch.setattr(workflow, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, now=frozen_now
    )

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
    margin = timedelta(seconds=config.runtime.throttle_resume_margin_s)
    expected_min = datetime.now(UTC) + timedelta(minutes=10) + margin - timedelta(minutes=1)
    expected_max = datetime.now(UTC) + timedelta(minutes=10) + margin + timedelta(minutes=1)
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
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)

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


def test_detect_and_handle_stalled_sessions_not_killed_when_real_activity_probe_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #301 kill-path wiring: a claude-code worker whose sidecar log is frozen
    but whose events.jsonl sibling carries fresh activity must NOT be killed by
    _detect_and_handle_stalled_sessions.

    This is the kill-path counterpart to
    test_detect_stalled_sessions_passes_real_activity_probe above, which only
    exercises the read-only _detect_stalled_sessions status/digest path and
    never drives an actual kill decision. A future edit that drops the
    ``probe`` argument from the classify_worker_health call inside
    _detect_and_handle_stalled_sessions (workflow.py), or that neuters the
    claude_events_jsonl Source-3 construction in
    post_mortem.real_activity_for_worker, must make this test fail (the
    worker gets killed) rather than silently reverting to mtime-only kills.
    """
    from charlie_work import workflow

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    issue_number = 302

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
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("claude", "prompt.md"),
        pid=88888,
        started_at=(datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    killed: list[int] = []
    monkeypatch.setattr(
        workflow, "kill_process_tree", lambda pid, start_time: killed.append(pid) or [pid]
    )
    monkeypatch.setattr(workflow, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: True)

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    state_file = tmp_path / "state.json"

    result = workflow._detect_and_handle_stalled_sessions(sessions_dir, state_file, config)

    assert result == []
    assert killed == []

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar.get("failure_kind") is None


def test_detect_and_handle_stalled_sessions_tolerates_none_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #329: workflow.py stall-event logging must tolerate a None RealActivityProbe."""
    from charlie_work import workflow

    issue_number = 329
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text("Working on task...\nLast line\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_path, (time.time(), old_time.timestamp()))

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
        rate_limit_defer_until=None,
    )
    _write_json(sidecar_path, record.to_dict())

    killed: list[int] = []
    monkeypatch.setattr(
        workflow, "kill_process_tree", lambda pid, start_time: killed.append(pid) or [pid]
    )
    monkeypatch.setattr(workflow, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", lambda *args: None)

    state_file = tmp_path / "state.json"
    config = OrchestratorConfig()

    result = workflow._detect_and_handle_stalled_sessions(sessions_dir, state_file, config)

    assert result == [{"issue": issue_number, "pid": 99999}]
    assert killed == [99999]

    state = json.loads(state_file.read_text(encoding="utf-8"))
    stalled_events = [e for e in state.get("events", []) if e.get("kind") == "session_stalled"]
    assert len(stalled_events) == 1
    payload = stalled_events[0]["payload"]
    assert payload["latest_real_activity_source"] == "probe unavailable"
    assert payload["latest_real_activity_at"] is None
    assert payload["activity_sources"] == []


def _inconclusive_probe_for_signal_1(
    view: WorkerView,
    config: OrchestratorConfig,
    now: datetime,
) -> RealActivityProbe:
    """Return an all-errored real-activity probe for issue #338 kill-path tests."""
    return RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=None,
                staleness_seconds=None,
                error="message_nodes query failed (schema drift?): no such column: id",
            ),
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error="no per-PID log found",
            ),
        )
    )


def test_detect_and_handle_stalled_sessions_inconclusive_probe_deferred_then_escalated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #338 kill-path: a dead worker with an inconclusive probe is deferred,
    the sidecar counter increments, and the worker is reaped once the cap is hit.
    """
    from charlie_work import workflow

    issue_number = 338
    log_text = "Working on task...\n"
    sessions_dir, state_file, _ = _make_stalled_devin_session(tmp_path, issue_number, log_text)

    killed: list[int] = []
    monkeypatch.setattr(
        workflow, "kill_process_tree", lambda pid, start_time: killed.append(pid) or [pid]
    )
    monkeypatch.setattr(workflow, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: False)
    monkeypatch.setattr(
        "charlie_work.worker.real_activity_probe_for", _inconclusive_probe_for_signal_1
    )

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=1),
    )

    # First pass: dead PID + inconclusive probe → defer, counter advances.
    result = workflow._detect_and_handle_stalled_sessions(sessions_dir, state_file, config)
    assert result == []
    assert killed == []

    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar.get("inconclusive_probe_deferred_count") == 1
    assert sidecar.get("failure_kind") is None

    # Second pass: counter is now at the cap → reap.
    result = workflow._detect_and_handle_stalled_sessions(sessions_dir, state_file, config)
    assert result == [{"issue": issue_number, "pid": 99999}]
    assert killed == [99999]

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["failure_kind"] == "stalled"
    assert sidecar.get("inconclusive_probe_deferred_count") == 0


def test_detect_and_handle_stalled_sessions_emits_provider_suspended_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1342: when the stall lane kills an api worker whose log carries a
    provider account-suspension signature, the dead-session lane (run in
    ``loop()``'s own order immediately after) emits a distinct
    ``api_worker_provider_suspended`` event on first detection so the operator
    learns about a billing problem in minutes, and classifies the sidecar
    ``failure_kind=provider_suspended`` so the issue escalates immediately
    instead of burning the redispatch cap."""
    from charlie_work import workflow
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    issue_number = 1342
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    # Fresh log (not stalled by mtime) carrying the suspension signature —
    # Signal 2.5 must fire on the tail regardless of mtime.
    log_path.write_text(
        "Working...\n"
        "Error: suspended due to insufficient balance, please recharge your "
        "account or check your plan and billing details.\n"
        "Retrying in 60s...\n",
        encoding="utf-8",
    )

    sidecar_path = claude_sidecar_path(sessions_dir, issue_number, "api")
    record = ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("claude", "prompt.md"),
        pid=77777,
        started_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        adapter_kind="api",
        provider="example",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    killed: list[int] = []
    monkeypatch.setattr(workflow, "sweep_orphan_processes", lambda worktree_path: [])

    # The stall lane sees a live worker (Signal 2.5 fires on the log tail);
    # after it kills the PID, the dead lane sees a dead worker. Use a mutable
    # flag flipped by kill_process_tree to model the real liveness transition.
    alive = {"yes": True}
    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: alive["yes"])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: alive["yes"])
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", lambda *args: None)

    def _kill_and_flip(pid, start_time):
        alive["yes"] = False
        return killed.append(pid) or [pid]

    monkeypatch.setattr(workflow, "kill_process_tree", _kill_and_flip)

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": [], "issues": {}}), encoding="utf-8")

    # Stall lane: kills the worker (Signal 2.5 → DEAD) and classifies the
    # sidecar failure_kind=provider_suspended.
    result = workflow._detect_and_handle_stalled_sessions(sessions_dir, state_file, config)

    # The worker is killed within one supervision pass (no stall-minute wait).
    assert result == [{"issue": issue_number, "pid": 77777}]
    assert killed == [77777]

    # The sidecar is classified provider_suspended (terminal).
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["failure_kind"] == "provider_suspended"

    # Dead lane (run in loop()'s own order immediately after the stall lane):
    # reaps the sidecar, emits the distinct event, and escalates.
    class FakeGitHub:
        def __init__(self) -> None:
            self.issues = [
                {
                    "number": issue_number,
                    "title": "Test issue",
                    "url": "https://example.test/issues/1342",
                    "body": "Test",
                    "labels": [{"name": config.labels.in_progress}],
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

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir,
        state_file,
        fake_gh,
        config,
        write_gate=_wg(state_file),
    )

    # The sidecar is reaped by the dead lane.
    assert not sidecar_path.exists(), "Sidecar must be reaped after dead-session classification"

    # A distinct event fires on first detection.
    state = json.loads(state_file.read_text(encoding="utf-8"))
    kinds = [e.get("kind") for e in state.get("events", [])]
    assert "api_worker_provider_suspended" in kinds
    suspended_event = next(
        e for e in state["events"] if e.get("kind") == "api_worker_provider_suspended"
    )
    assert suspended_event["payload"]["issue_number"] == issue_number
    assert suspended_event["payload"]["provider"] == "example"

    # The issue is escalated (deterministic escalation), not redispatched.
    assert "session_failed_escalated" in kinds
    escalated_event = next(
        e for e in state["events"] if e.get("kind") == "session_failed_escalated"
    )
    assert escalated_event["payload"]["failure_kind"] == "provider_suspended"
