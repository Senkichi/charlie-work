"""Worker sidecar records: ``read_worker_records`` round-trips and
corrupt-entry skipping, ``ClaudeWorkerRecord`` serialization, ``_sidecar_path``.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from _claude_adapter_fixtures import (
    _fake_claude_script,
    _install_fake_create_worktree,
)

from charlie_work.claude_code import (
    ClaudeWorkerRecord,
    launch_claude_worker,
    read_worker_records,
    _sidecar_path,
)


def test_read_worker_records_round_trips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    launch_claude_worker(
        1,
        "agent/issue-1-a",
        "prompt a",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )
    launch_claude_worker(
        2,
        "agent/issue-2-b",
        "prompt b",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    records = read_worker_records(sessions_dir)

    assert len(records) == 2
    assert {r.issue_number for r in records} == {1, 2}
    assert all(isinstance(r, ClaudeWorkerRecord) for r in records)


def test_read_worker_records_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert read_worker_records(tmp_path / "does-not-exist") == []


def test_read_worker_records_skips_corrupt_sidecar(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "issue-5.claude.json").write_text("not json{{{", encoding="utf-8")

    assert read_worker_records(sessions_dir) == []


def test_read_worker_records_skips_sidecar_missing_required_fields(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "issue-6.claude.json").write_text(
        json.dumps({"branch": "agent/issue-6-x"}), encoding="utf-8"
    )

    assert read_worker_records(sessions_dir) == []


def test_sidecar_path_returns_correct_path(tmp_path: Path) -> None:
    """_sidecar_path returns the expected path for a given issue number."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    path = _sidecar_path(sessions_dir, 123)
    assert path == sessions_dir / "issue-123.claude.json"


def test_claude_worker_record_round_trips_new_fields(tmp_path: Path) -> None:
    """``to_dict`` / ``from_dict`` carry ``adapter_kind`` and ``provider``."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    record = ClaudeWorkerRecord(
        issue_number=7,
        branch="agent/issue-7",
        worktree_path=str(tmp_path / "wt"),
        prompt_path="p.md",
        command=("claude", "-p", "p.md"),
        pid=123,
        started_at="2026-07-20T00:00:00Z",
        log_path="log.txt",
        adapter_kind="api",
        provider="openai",
    )

    sidecar_path = sessions_dir / "issue-7.api.json"
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    [restored] = read_worker_records(sessions_dir, adapter_kind=None)
    assert restored == record


def test_claude_worker_record_from_dict_defaults_legacy_keys() -> None:
    """Sidecars written before the new fields default them on read."""
    payload = {
        "issue_number": 9,
        "branch": "agent/issue-9",
        "worktree_path": "/wt",
        "prompt_path": "p.md",
        "command": ["claude", "-p", "p.md"],
        "pid": 999,
        "started_at": "2026-07-20T00:00:00Z",
        "log_path": "log.txt",
    }

    record = ClaudeWorkerRecord.from_dict(payload)

    assert record.adapter_kind == "claude-code"
    assert record.provider == ""
