"""Recovery behaviour per worker kind: the Devin-source absence rules for Devin,
non-Devin and api workers."""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from _worktree_fixtures import _git, _init_repo
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    PostMortemConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.process_utils import get_process_start_time
from charlie_work.worktree import (
    WorktreeInfo,
    LiveWorkerRedispatchError,
    _default_worktrees_dir,
    create_worktree,
    _slugify,
)


def test_recovery_proceeds_for_non_devin_worker_despite_devin_source_absence(
    tmp_path: Path,
) -> None:
    """Issue #639: a non-Devin worker (claude-code/api) never writes rows to
    sessions.db or per-PID Devin logs. Before #639, the redispatch probe
    consulted those Devin sources anyway, got permanent "no session found" /
    "no pid" errors, and fail-closed on them — blocking recovery forever
    because the ``inconclusive_probe_deferred_count`` cap was never reached
    and ``confirmed_dead`` required a recorded ``worker_pid`` (absent in 75
    of 81 live-host worktrees).

    The fix (option 3 in the issue): distinguish "no Devin subject exists at
    all" from "Devin subject exists but could not be read". When the recovery
    record's ``adapter_history`` shows the worker was routed to a non-Devin
    adapter, the Devin sources are skipped entirely — they have no subject to
    look up. With no errored sources, the probe falls through and recovery
    proceeds (the PID check alone gates the reset, same as when
    ``devin.adapter != "devin-shell"``).
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-639-non-devin-worker"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db exists but has no row for this worktree — the exact shape
    # on the live host (12 of 81 worktrees hit "no session found matching
    # working_directory within the time window").
    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (id TEXT, working_directory TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=3),
    )

    # The worker was routed to claude-code (api routing enabled, fallback to
    # claude-code). No worker_pid recorded (75 of 81 live-host worktrees).
    # Before #639 this raised LiveWorkerRedispatchError(probe_error); now the
    # Devin sources are skipped and recovery proceeds.
    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": None,
        "worker_process_start_time": 0.0,
        "started_at": now,
        "adapter_history": [
            {"ts": now, "kind": "claude-code", "provider": "", "reason": "fallback:disabled"}
        ],
    }

    result = create_worktree(
        repo_root,
        branch_name,
        base_ref="HEAD",
        recovery=recovery_record,
        config=config,
    )

    assert isinstance(result, WorktreeInfo)
    assert worktree_path.exists()
    assert branch_name in _git(repo_root, "branch", "--list").stdout


def test_recovery_proceeds_for_api_worker_despite_devin_source_absence(
    tmp_path: Path,
) -> None:
    """Issue #639 companion: an ``api``-routed worker (which delegates to
    claude-code) also has no Devin subject. The ``adapter_history`` records
    ``kind="api"`` for these workers, and the Devin sources must be skipped
    just as for ``claude-code``.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-639-api-worker"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (id TEXT, working_directory TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=3),
    )

    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": None,
        "worker_process_start_time": 0.0,
        "started_at": now,
        "adapter_history": [
            {"ts": now, "kind": "api", "provider": "anthropic", "reason": "policy:complexity"}
        ],
    }

    result = create_worktree(
        repo_root,
        branch_name,
        base_ref="HEAD",
        recovery=recovery_record,
        config=config,
    )

    assert isinstance(result, WorktreeInfo)
    assert worktree_path.exists()
    assert branch_name in _git(repo_root, "branch", "--list").stdout


def test_recovery_still_aborts_for_devin_worker_with_devin_source_errors(
    tmp_path: Path,
) -> None:
    """Issue #639 regression guard: a Devin-shell worker with the same
    sessions.db / per-PID log absence-of-record must STILL abort recovery.
    The fix only skips Devin sources for *non-Devin* workers; a Devin worker
    whose probe is genuinely inconclusive remains fail-closed (issue #282).
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-639-devin-worker-still-aborts"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (id TEXT, working_directory TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=3),
    )

    # adapter_history records "devin-shell" — the worker IS a Devin subject.
    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": None,
        "worker_process_start_time": 0.0,
        "started_at": now,
        "adapter_history": [
            {"ts": now, "kind": "devin-shell", "provider": "", "reason": "policy:default"}
        ],
    }

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            recovery=recovery_record,
            config=config,
        )

    assert exc_info.value.probe_result == "probe_error"
    assert exc_info.value.inconclusive_probe_deferred_count == 1
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_non_devin_worker_with_live_pid_still_aborts(
    tmp_path: Path,
) -> None:
    """Issue #639 safety guard: skipping Devin sources for a non-Devin worker
    does NOT bypass the PID liveness check. A non-Devin worker whose recorded
    PID is still alive must still abort recovery — the PID check runs before
    the activity probe and is unaffected by the Devin-source skip.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-639-non-devin-live-pid"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (id TEXT, working_directory TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=3),
    )

    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": os.getpid(),
        "worker_process_start_time": get_process_start_time(os.getpid()),
        "started_at": now,
        "adapter_history": [
            {"ts": now, "kind": "claude-code", "provider": "", "reason": "fallback:disabled"}
        ],
    }

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            recovery=recovery_record,
            config=config,
        )

    assert exc_info.value.probe_result == "pid_alive"
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout
