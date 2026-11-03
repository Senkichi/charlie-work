"""Recovery aborts/proceeds depending on worker liveness evidence (pid, sessions.db,
per-pid log)."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from _sessions_db_fixtures import make_sessions_db
from _worktree_fixtures import _git, _init_repo
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    PostMortemConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.process_utils import get_process_start_time
from charlie_work import worktree as worktree_module
from charlie_work.worktree import (
    WorktreeInfo,
    LiveWorkerRedispatchError,
    _default_worktrees_dir,
    create_worktree,
    _slugify,
)


def test_recovery_aborts_when_worker_pid_alive(tmp_path: Path) -> None:
    """Issue #282: recovery must not remove a worktree if the recorded worker PID is still alive."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-live-worker"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Spawn a real long-running child process so we have a live PID and a valid start time.
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )
    try:
        start_time = get_process_start_time(process.pid)
        assert start_time is not None

        recovery_record = {
            "branch_name": branch_name,
            "status": "dispatched",
            "worker_pid": process.pid,
            "worker_process_start_time": start_time,
        }

        with pytest.raises(LiveWorkerRedispatchError) as exc_info:
            create_worktree(repo_root, branch_name, base_ref="HEAD", recovery=recovery_record)

        assert exc_info.value.probe_result == "pid_alive"
        assert exc_info.value.pid == process.pid
        # The worktree and branch must survive the aborted redispatch.
        assert info1.path.exists()
        assert branch_name in _git(repo_root, "branch", "--list").stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_recovery_aborts_on_sessions_db_activity(tmp_path: Path) -> None:
    """Issue #282: recovery must not remove a worktree if sessions.db shows fresh activity."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-db-activity"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # Build a fake Devin sessions.db with a recent session/message for this worktree.
    db_path = tmp_path / "sessions.db"
    now = datetime.now(UTC).isoformat()
    make_sessions_db(
        db_path,
        session_id="session-1",
        working_directory=str(worktree_path),
        created_at=now,
        rows=[{"role": "tool", "content": "tool result", "created_at": now}],
    )

    # The per-PID Devin log source must also resolve WITHOUT an error (issue
    # #282 rework: any errored source makes the whole probe inconclusive and
    # aborts recovery on its own, which would mask this test's actual signal
    # under test — sessions.db activity alone being sufficient). Give it a
    # confirmed-stale (not fresh) timestamp so sessions.db remains the
    # freshest, deciding source.
    logs_dir = db_path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stale_log = logs_dir / "devin_test_999999.log"
    stale_log.write_text("old log\n", encoding="utf-8")
    stale_time = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
    os.utime(stale_log, (stale_time, stale_time))

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )

    # Use a dead PID (no such process) so the sessions.db probe is the deciding signal.
    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
        "started_at": now,
    }

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            recovery=recovery_record,
            config=config,
        )

    assert exc_info.value.probe_result == "sessions_db_activity"
    # No worktree should have been created yet and the branch should not exist.
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_aborts_on_sessions_db_schema_error_other_source_silent(tmp_path: Path) -> None:
    """Issue #282 rework: an errored sessions.db probe is INCONCLUSIVE, not
    confirmed-dead — even when the per-PID log source has no signal of its
    own either. Reproduces the exact live-incident signature (sqlite schema
    drift, "no such column: id") that let the prior fail-open guard proceed
    into a destructive reset of a live worker.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-db-schema-error"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions table is missing the `id` column the probe's query selects -
    # the exact schema-drift shape from the live incident.
    db_path = tmp_path / "sessions.db"

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (working_directory TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )

    # No logs/ directory at all -> devin_per_pid_log is silent too (its own
    # "not found" error), never a confirmed timestamp either way.
    # No worker PID is recorded, so the probe is genuinely inconclusive.
    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": None,
        "worker_process_start_time": 0.0,
        "started_at": now,
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
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_aborts_when_all_sources_errored(tmp_path: Path) -> None:
    """Issue #282 rework: if every activity source errors, liveness is
    unknown, not confirmed-dead — recovery must abort rather than proceed as
    if the worker were genuinely stale.

    Mutation-checked: reverting ``_probe_recovery_liveness`` to the prior
    fail-open guard (``if source.name == "sessions.db" and
    source.staleness_seconds is not None``) makes this test FAIL, because
    that guard silently returns without raising when every source errors.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-all-sources-errored"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db does not exist on disk at all -> _open_readonly errors
    # immediately, before any query is even attempted.
    db_path = tmp_path / "missing-sessions.db"

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )

    # No logs/ directory either -> devin_per_pid_log also errors.
    # No worker PID is recorded, so the probe is genuinely inconclusive.
    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": None,
        "worker_process_start_time": 0.0,
        "started_at": now,
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
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_aborts_on_fresh_per_pid_log_despite_sessions_db_error(tmp_path: Path) -> None:
    """Issue #282 rework: fresh devin_per_pid_log activity must abort
    recovery even when sessions.db is the source that errored — this is the
    exact tonight's-incident shape (an 8-second-fresh per-PID log) that the
    prior sessions.db-only check never looked at.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-per-pid-log-fresh-db-error"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db missing entirely -> errored source.
    db_path = tmp_path / "missing-sessions.db"

    # Fresh per-PID Devin log for the recorded worker pid.
    logs_dir = db_path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fresh_log = logs_dir / "devin_test_999999.log"
    fresh_log.write_text("fresh activity\n", encoding="utf-8")

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )

    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
        "started_at": now,
    }

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            recovery=recovery_record,
            config=config,
        )

    # Aborted either way (the errored sessions.db source is already enough on
    # its own) - the point of this regression test is that the fresh
    # devin_per_pid_log signal is never silently ignored just because it
    # isn't the sessions.db source.
    assert exc_info.value.probe_result == "devin_per_pid_log_activity"
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_aborts_on_fresh_per_pid_log_when_sessions_db_confirmed_stale(
    tmp_path: Path,
) -> None:
    """Issue #282 rework, requirement 2 in isolation: even when sessions.db
    positively confirms only stale (non-error) activity, a fresh
    devin_per_pid_log signal on its own must still abort recovery. The prior
    guard only ever inspected the "sessions.db" source by name and would have
    silently ignored this signal.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-per-pid-log-only-fresh"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    db_path = tmp_path / "sessions.db"
    stale_iso = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    make_sessions_db(
        db_path,
        session_id="session-1",
        working_directory=str(worktree_path),
        created_at=stale_iso,
        rows=[{"role": "tool", "content": "tool result", "created_at": stale_iso}],
    )

    # Fresh per-PID Devin log (mtime defaults to "now" via write_text).
    logs_dir = db_path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fresh_log = logs_dir / "devin_test_999999.log"
    fresh_log.write_text("fresh activity\n", encoding="utf-8")

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )

    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
        "started_at": stale_iso,
    }

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            recovery=recovery_record,
            config=config,
        )

    assert exc_info.value.probe_result == "devin_per_pid_log_activity"
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_increments_deferral_count_for_permanent_no_match(tmp_path: Path) -> None:
    """Issue #426: a permanent sessions.db no-match increments the recovery
    deferral counter on each attempt and still aborts below the cap.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-permanent-no-match"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db exists but has no row for this worktree (permanent no-match).
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

    # No worker PID is recorded, so the permanent no-match is genuinely
    # inconclusive and the deferral counter must advance.
    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": None,
        "worker_process_start_time": 0.0,
        "started_at": now,
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


def test_recovery_allows_permanent_no_match_after_deferral_cap(tmp_path: Path) -> None:
    """Issue #426: a structurally permanent sessions.db no-match is not an
    unconditional recovery block. After ``max_inconclusive_probe_deferrals``
    consecutive inconclusive probes, the destructive reset is allowed.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-permanent-no-match-capped"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db exists but has no row for this worktree (permanent no-match).
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
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=2),
    )

    # No worker PID is recorded, so the deferral cap is the reason reset is
    # allowed, not the confirmed-dead PID short-circuit.
    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": None,
        "worker_process_start_time": 0.0,
        "started_at": now,
        "inconclusive_probe_deferred_count": 2,
    }

    # Should not raise: the permanent no-match has reached the deferral cap.
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


def test_recovery_allows_reset_when_worker_pid_dead_and_probe_inconclusive(
    tmp_path: Path,
) -> None:
    """Issue #506: a confirmed-dead worker PID overrides an inconclusive probe."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-dead-pid-inconclusive"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db exists but has no row for this worktree (permanent no-match).
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
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
        "started_at": now,
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


def test_recovery_aborts_on_transient_probe_error_despite_dead_pid(
    tmp_path: Path,
) -> None:
    """Issue #506 rework: a confirmed-dead PID does NOT override a probe that
    contains transient errors (locked/corrupt DB, schema drift, I/O failures).
    Only structurally permanent absence-of-record errors may be overridden by
    a dead PID; transient errors remain fail-closed.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-dead-pid-transient-probe"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db exists on disk but is not a valid SQLite file — a transient
    # "failed to open sessions.db (locked or corrupt)" error, not a permanent
    # no-match.
    db_path = tmp_path / "sessions.db"
    db_path.write_bytes(b"this is not a sqlite database")

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
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
        "started_at": now,
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


def test_recovery_proceeds_when_no_source_errored_and_pid_dead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #640: when no probe source errors, ``all_permanent`` keeps its
    pre-bound default and is never read in a decision — the second
    ``if errored_sources:`` block is skipped entirely. A confirmed-dead PID
    with a fully successful (but stale) probe must let recovery proceed
    without raising and without any ``UnboundLocalError`` on ``all_permanent``.

    This is the no-errored-source counterpart to the errored-source tests
    above; it guards the binding fix that makes ``all_permanent`` provably
    bound to Pyright.
    """
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-no-source-errored-dead-pid"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # A probe where every source returned successfully (no error) but with only
    # stale timestamps — liveness is genuinely inconclusive, yet no source
    # errored, so neither ``if errored_sources:`` branch runs.
    stale = datetime.now(UTC) - timedelta(hours=2)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=stale,
                staleness_seconds=7200.0,
                error=None,
            ),
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=stale,
                staleness_seconds=7200.0,
                error=None,
            ),
        )
    )

    monkeypatch.setattr(worktree_module, "real_activity_for_worker", lambda *a, **k: probe)

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "unused.db")),
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=3),
    )

    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
        "started_at": datetime.now(UTC).isoformat(),
    }

    result = create_worktree(
        repo_root,
        branch_name,
        base_ref="HEAD",
        recovery=recovery_record,
        config=config,
    )

    # Recovery proceeds: the dead PID and stale-but-error-free probe do not
    # abort redispatch.
    assert isinstance(result, WorktreeInfo)
    assert worktree_path.exists()
    assert branch_name in _git(repo_root, "branch", "--list").stdout
