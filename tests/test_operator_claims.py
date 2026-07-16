from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    empty_state,
    load_state,
    operator_claimed_issues,
    release_operator_claimed,
    set_operator_claimed,
    stale_operator_claims,
)
from charlie_work.worktree import (
    WorktreeForeignWriterError,
    _check_worktree_writer_marker,
    read_worktree_marker,
    remove_worktree_marker,
    worktree_path_for_branch,
    write_worktree_marker,
)
from charlie_work.workflow import OrchestratorApp


def test_operator_claim_state_helpers() -> None:
    state = empty_state()
    state = set_operator_claimed(state, 123)
    assert 123 in operator_claimed_issues(state)
    assert 456 not in operator_claimed_issues(state)
    assert not stale_operator_claims(state, threshold_minutes=30)

    # A stale claim is still reported as claimed, but also flagged stale.
    old = (datetime.now(UTC) - timedelta(minutes=31)).isoformat().replace("+00:00", "Z")
    state = set_operator_claimed(state, 456, timestamp=old)
    assert operator_claimed_issues(state) == {123, 456}
    assert stale_operator_claims(state, threshold_minutes=30) == {456}

    # Releasing a claim removes it.
    state = release_operator_claimed(state, 123)
    assert operator_claimed_issues(state) == {456}


def test_worktree_marker_write_read_remove(tmp_path: Path) -> None:
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()
    write_worktree_marker(worktree_path, 1234, "session-abc")
    marker = read_worktree_marker(worktree_path)
    assert marker is not None
    assert marker["pid"] == 1234
    assert marker["session_id"] == "session-abc"
    assert marker["started_at"].endswith("Z")

    # Reading a missing marker returns None; removing one is a no-op.
    other = tmp_path / "other"
    other.mkdir()
    assert read_worktree_marker(other) is None
    assert remove_worktree_marker(other) is False

    # Session-id-gated removal only succeeds for the matching session.
    assert remove_worktree_marker(worktree_path, session_id="wrong") is False
    assert read_worktree_marker(worktree_path) is not None
    assert remove_worktree_marker(worktree_path, session_id="session-abc") is True
    assert read_worktree_marker(worktree_path) is None


def test_check_worktree_marker_stale_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_worktree_marker(worktree_path, 1234, "session-abc")

    monkeypatch.setattr("charlie_work.worktree.is_pid_alive", lambda pid, start: False)
    _check_worktree_writer_marker(worktree_path, sessions_dir)
    assert read_worktree_marker(worktree_path) is None


def test_check_worktree_marker_foreign_live_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_worktree_marker(worktree_path, 1234, "foreign-session")

    monkeypatch.setattr("charlie_work.worktree.is_pid_alive", lambda pid, start: True)
    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        _check_worktree_writer_marker(worktree_path, sessions_dir)

    assert exc_info.value.worktree_path == worktree_path
    assert exc_info.value.pid == 1234
    assert exc_info.value.session_id == "foreign-session"


def test_check_worktree_marker_owned_session_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    write_worktree_marker(worktree_path, 1234, "owned-session")

    # A sidecar for the same live session lets the orchestrator re-enter the worktree.
    sidecar = sessions_dir / "issue-55.json"
    sidecar.write_text(
        json.dumps({"session_id": "owned-session", "pid": 1234, "process_start_time": 1.0}),
        encoding="utf-8",
    )

    monkeypatch.setattr("charlie_work.worktree.is_pid_alive", lambda pid, start: True)
    _check_worktree_writer_marker(worktree_path, sessions_dir)
    # Marker should not be removed when the session is owned and live.
    assert read_worktree_marker(worktree_path) is not None


class _MinimalGitHub:
    def __init__(self) -> None:
        pass

    def validate_field_lists(self) -> None:
        pass

    def issue_view(self, number: int) -> dict[str, Any]:
        return {
            "number": number,
            "title": f"Issue {number}",
            "body": "",
            "labels": [],
            "state": "OPEN",
            "url": f"https://example.test/issues/{number}",
        }


def test_claim_records_operator_claim_in_state(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, _MinimalGitHub())

    result = app.claim(999)
    assert result.ok is True
    assert result.data["issue_number"] == 999
    assert result.data["released"] is False

    state = load_state(paths.state_file)
    assert state["issues"]["999"]["operator_claimed_at"] is not None
    assert state["events"][-1]["kind"] == "operator_claim"


def test_claim_release_removes_operator_claim(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, _MinimalGitHub())

    app.claim(999)
    result = app.claim(999, release=True)
    assert result.ok is True
    assert result.data["released"] is True

    state = load_state(paths.state_file)
    assert "operator_claimed_at" not in state["issues"].get("999", {})
    assert state["events"][-1]["kind"] == "operator_claim_released"


def test_claim_writes_worktree_marker_when_worktree_exists(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, _MinimalGitHub())

    branch_name = f"{config.dispatch.branch_prefix}-999-Issue-999"
    worktree_path = worktree_path_for_branch(tmp_path, branch_name)
    worktree_path.mkdir(parents=True, exist_ok=True)

    result = app.claim(999)
    assert result.ok is True
    assert result.data["marker_written"] is True

    marker = read_worktree_marker(worktree_path)
    assert marker is not None
    assert marker["session_id"].startswith("operator-")
    assert marker["pid"] is not None
