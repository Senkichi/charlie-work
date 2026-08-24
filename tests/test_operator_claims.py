from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from charlie_work.config import OrchestratorConfig
from charlie_work.github import GitHubLike
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    empty_state,
    load_state,
    operator_claimed_issues,
    release_operator_claimed,
    save_state,
    set_operator_claimed,
    stale_operator_claims,
)
from charlie_work.worktree import (
    OPERATOR_MARKER_KIND,
    OPERATOR_MARKER_SESSION_ID,
    WorktreeForeignWriterError,
    _check_worktree_writer_marker,
    create_worktree,
    read_worktree_marker,
    remove_worktree_marker,
    worktree_path_for_branch,
    write_worktree_marker,
)
from charlie_work.workflow import OrchestratorApp


def _init_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )
    run(["git", "init", "--initial-branch=main"])
    run(["git", "config", "user.email", "test@example.test"])
    run(["git", "config", "user.name", "Test User"])
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    run(["git", "add", "README.md"])
    run(["git", "commit", "-m", "initial commit"])


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
    assert marker["kind"] == "worker"

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


def _write_marker_with_started_at(
    worktree_path: Path,
    pid: int,
    session_id: str,
    started_at: str,
    *,
    process_start_time: float | None = 1234567890.0,
) -> None:
    """Write a marker with a custom ``started_at`` (for staleness tests).

    Also sets the marker file's mtime to the ``started_at`` time so the
    worktree-mtime activity probe sees a realistic old marker (in production
    the marker is written once at worker start and never touched again).

    Issue #1423 review: ``process_start_time`` is the OS process-creation
    fingerprint the reap path passes to ``kill_process_tree`` to defend
    against PID recycling. It defaults to a fixed fake value so the reap
    tests exercise the identity-checked kill path; pass ``None`` to simulate
    a legacy marker written before the field existed (which must NOT be
    reaped).
    """
    from charlie_work.config import WRITER_MARKER_FILENAME

    marker: dict[str, Any] = {
        "pid": pid,
        "session_id": session_id,
        "started_at": started_at,
        "kind": "worker",
    }
    if process_start_time is not None:
        marker["process_start_time"] = process_start_time
    marker_path = worktree_path / WRITER_MARKER_FILENAME
    tmp = marker_path.with_suffix(marker_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(marker, handle)
        handle.write("\n")
    tmp.replace(marker_path)
    # Set the marker file's mtime to the started_at time so the worktree
    # mtime probe does not see a freshly-written marker as "activity".
    try:
        old_ts = datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        old_ts = None
    if old_ts is not None:
        os.utime(marker_path, (old_ts, old_ts))


def test_foreign_writer_live_pid_stale_mtime_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1423: a foreign writer with a live pid but stale worktree mtime
    is reaped (killed + marker removed) instead of blocking dispatch."""
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Marker with an old started_at (2 hours ago) — the writer has been
    # alive but idle for a long time.
    old_started = (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    _write_marker_with_started_at(worktree_path, 1234, "foreign-session", old_started)

    # Monkeypatch process kills so the test doesn't actually kill anything.
    monkeypatch.setattr("charlie_work.worktree.is_pid_alive", lambda pid, start: True)
    monkeypatch.setattr("charlie_work.worktree.kill_process_tree", lambda pid, st: [pid])
    monkeypatch.setattr("charlie_work.worktree.sweep_orphan_processes", lambda wt: [])
    monkeypatch.setattr("charlie_work.worktree.kill_orphan_pid", lambda pid: None)

    config = OrchestratorConfig()
    state_file = tmp_path / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)

    # Should NOT raise — the idle foreign writer is reaped and the marker
    # is cleaned, so dispatch proceeds.
    _check_worktree_writer_marker(
        worktree_path,
        sessions_dir,
        config=config,
        state_file=state_file,
    )

    # Marker must be removed after reaping.
    assert read_worktree_marker(worktree_path) is None


def test_foreign_writer_live_pid_fresh_mtime_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1423: a foreign writer with a live pid AND fresh worktree mtime
    is still blocked (not reaped) — it is genuinely active."""
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Marker with an old started_at (2 hours ago).
    old_started = (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    _write_marker_with_started_at(worktree_path, 1234, "foreign-session", old_started)

    # Write a file with a recent mtime (1 minute ago) — the writer is active.
    fresh_file = worktree_path / "worker_output.txt"
    fresh_file.write_text("recent work\n", encoding="utf-8")
    recent_mtime = (datetime.now(UTC) - timedelta(minutes=1)).timestamp()
    os.utime(fresh_file, (recent_mtime, recent_mtime))

    # Monkeypatch process kills so the test doesn't actually kill anything.
    monkeypatch.setattr("charlie_work.worktree.is_pid_alive", lambda pid, start: True)
    monkeypatch.setattr("charlie_work.worktree.kill_process_tree", lambda pid, st: [pid])
    monkeypatch.setattr("charlie_work.worktree.sweep_orphan_processes", lambda wt: [])
    monkeypatch.setattr("charlie_work.worktree.kill_orphan_pid", lambda pid: None)

    config = OrchestratorConfig()
    state_file = tmp_path / "state.json"

    # Should raise — the writer is active, so dispatch is blocked.
    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        _check_worktree_writer_marker(
            worktree_path,
            sessions_dir,
            config=config,
            state_file=state_file,
        )

    assert exc_info.value.pid == 1234
    # Marker must still be present (not reaped).
    assert read_worktree_marker(worktree_path) is not None


def test_foreign_writer_reaped_event_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1423: reaping a foreign writer logs a ``foreign_writer_reaped``
    event retrievable via ``query_events``."""
    from charlie_work.instrumentation import query_events
    from charlie_work.state import empty_state, save_state

    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    old_started = (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    _write_marker_with_started_at(worktree_path, 1234, "foreign-session", old_started)

    monkeypatch.setattr("charlie_work.worktree.is_pid_alive", lambda pid, start: True)
    monkeypatch.setattr("charlie_work.worktree.kill_process_tree", lambda pid, st: [pid])
    monkeypatch.setattr("charlie_work.worktree.sweep_orphan_processes", lambda wt: [])
    monkeypatch.setattr("charlie_work.worktree.kill_orphan_pid", lambda pid: None)

    config = OrchestratorConfig()
    state_file = tmp_path / "state.json"
    save_state(state_file, empty_state())

    _check_worktree_writer_marker(
        worktree_path,
        sessions_dir,
        config=config,
        state_file=state_file,
        issue_number=42,
    )

    events = query_events(state_file, kind="foreign_writer_reaped")
    assert len(events) == 1
    assert events[0]["payload"]["pid"] == 1234
    assert events[0]["issue_number"] == 42


def test_foreign_writer_reap_passes_process_start_time_to_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1423 review: the reap path must pass the marker's
    ``process_start_time`` fingerprint to ``kill_process_tree`` so the kill
    re-verifies process identity immediately before terminating — the same
    PID-recycling defense every other ``kill_process_tree`` call site uses.
    Passing ``None`` (the pre-fix behavior) drops that defense and can kill
    an unrelated process that reused the PID during the hours-long idle
    window."""
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    old_started = (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    fake_start_time = 1234567890.0
    _write_marker_with_started_at(
        worktree_path, 1234, "foreign-session", old_started, process_start_time=fake_start_time
    )

    kill_calls: list[tuple[int, float | None]] = []

    def _fake_kill(pid: int, start: float | None) -> list[int]:
        kill_calls.append((pid, start))
        return [pid]

    monkeypatch.setattr("charlie_work.worktree.is_pid_alive", lambda pid, start: True)
    monkeypatch.setattr("charlie_work.worktree.kill_process_tree", _fake_kill)
    monkeypatch.setattr("charlie_work.worktree.sweep_orphan_processes", lambda wt: [])
    monkeypatch.setattr("charlie_work.worktree.kill_orphan_pid", lambda pid: None)

    config = OrchestratorConfig()
    state_file = tmp_path / "state.json"

    _check_worktree_writer_marker(
        worktree_path, sessions_dir, config=config, state_file=state_file
    )

    assert len(kill_calls) == 1
    assert kill_calls[0][0] == 1234
    # The fingerprint MUST be forwarded, not dropped to None.
    assert kill_calls[0][1] == fake_start_time


def test_foreign_writer_legacy_marker_without_start_time_not_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1423 review: a marker written before the ``process_start_time``
    field existed (legacy) carries no identity fingerprint. Reaping it would
    call ``kill_process_tree(pid, None)``, dropping the PID-recycling defense
    — and an idle window can be hours long, exactly when the OS recycles
    PIDs. Such a marker must NOT be reaped; it falls back to the
    block/escalate path instead."""
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    old_started = (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    # Legacy marker: no process_start_time field.
    _write_marker_with_started_at(
        worktree_path, 1234, "foreign-session", old_started, process_start_time=None
    )

    kill_calls: list[tuple[int, float | None]] = []
    monkeypatch.setattr("charlie_work.worktree.is_pid_alive", lambda pid, start: True)
    monkeypatch.setattr(
        "charlie_work.worktree.kill_process_tree",
        lambda pid, st: kill_calls.append((pid, st)) or [pid],
    )
    monkeypatch.setattr("charlie_work.worktree.sweep_orphan_processes", lambda wt: [])
    monkeypatch.setattr("charlie_work.worktree.kill_orphan_pid", lambda pid: None)

    config = OrchestratorConfig()
    state_file = tmp_path / "state.json"

    # Must raise — the legacy marker cannot be safely reaped, so dispatch is
    # blocked (and would eventually escalate) instead of killing an
    # unverified PID.
    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        _check_worktree_writer_marker(
            worktree_path, sessions_dir, config=config, state_file=state_file
        )
    assert exc_info.value.pid == 1234
    # No kill must have occurred.
    assert kill_calls == []
    # Marker must still be present (not reaped).
    assert read_worktree_marker(worktree_path) is not None


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
    dry_run = False

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
    app = OrchestratorApp(tmp_path, paths, config, cast(GitHubLike, _MinimalGitHub()))

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
    app = OrchestratorApp(tmp_path, paths, config, cast(GitHubLike, _MinimalGitHub()))

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
    app = OrchestratorApp(tmp_path, paths, config, cast(GitHubLike, _MinimalGitHub()))

    branch_name = f"{config.dispatch.branch_prefix}-999-Issue-999"
    worktree_path = worktree_path_for_branch(tmp_path, branch_name)
    worktree_path.mkdir(parents=True, exist_ok=True)

    result = app.claim(999)
    assert result.ok is True
    assert result.data["marker_written"] is True

    marker = read_worktree_marker(worktree_path)
    assert marker is not None
    assert marker["session_id"] == OPERATOR_MARKER_SESSION_ID
    assert marker["kind"] == OPERATOR_MARKER_KIND
    assert marker["pid"] == 0


def test_claim_marker_liveness_derived_from_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator markers must stay live after the CLI exits (pid is a sentinel)."""
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    state_file = tmp_path / "state.json"
    state = set_operator_claimed(empty_state(), 42)
    save_state(state_file, state)

    write_worktree_marker(worktree_path, 0, OPERATOR_MARKER_SESSION_ID, kind=OPERATOR_MARKER_KIND)

    # Even though the sentinel pid is "not alive", the marker is live because
    # state.json shows an active operator claim for issue 42.
    monkeypatch.setattr("charlie_work.worktree.is_pid_alive", lambda pid, start: False)
    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        _check_worktree_writer_marker(worktree_path, sessions_dir, 42, state_file)

    assert exc_info.value.worktree_path == worktree_path
    assert exc_info.value.session_id == OPERATOR_MARKER_SESSION_ID


def test_claim_marker_removed_when_claim_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A released operator claim lets create_worktree clean the stale marker."""
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    state_file = tmp_path / "state.json"

    # No operator_claimed_at in state.
    save_state(state_file, empty_state())

    write_worktree_marker(worktree_path, 0, OPERATOR_MARKER_SESSION_ID, kind=OPERATOR_MARKER_KIND)

    monkeypatch.setattr("charlie_work.worktree.is_pid_alive", lambda pid, start: False)
    _check_worktree_writer_marker(worktree_path, sessions_dir, 42, state_file)
    assert read_worktree_marker(worktree_path) is None


def test_claim_release_does_not_remove_worker_marker(tmp_path: Path) -> None:
    """--release must only remove the operator's own marker, not a worker's."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, cast(GitHubLike, _MinimalGitHub()))

    branch_name = f"{config.dispatch.branch_prefix}-999-Issue-999"
    worktree_path = worktree_path_for_branch(tmp_path, branch_name)
    worktree_path.mkdir(parents=True, exist_ok=True)

    # Simulate a worker marker left behind by an active session.
    write_worktree_marker(worktree_path, 1234, "worker-uuid")

    # Releasing a claim without a prior operator marker must not unlink the worker.
    result = app.claim(999, release=True)
    assert result.ok is True

    # Worker marker must survive the operator release.
    marker = read_worktree_marker(worktree_path)
    assert marker is not None
    assert marker["session_id"] == "worker-uuid"


def test_create_worktree_refuses_operator_marker_while_claimed(
    tmp_path: Path,
) -> None:
    """Real create_worktree(sessions_dir=...) refuses a foreign operator writer."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    config = OrchestratorConfig()
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    paths.ensure()
    state = set_operator_claimed(empty_state(), 42)
    save_state(paths.state_file, state)

    branch_name = f"{config.dispatch.branch_prefix}-42-fix"
    worktree_path = worktree_path_for_branch(repo_root, branch_name)
    worktree_path.mkdir(parents=True, exist_ok=True)
    write_worktree_marker(worktree_path, 0, OPERATOR_MARKER_SESSION_ID, kind=OPERATOR_MARKER_KIND)

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            issue_number=42,
            config=config,
            sessions_dir=sessions_dir,
        )

    assert exc_info.value.worktree_path == worktree_path
