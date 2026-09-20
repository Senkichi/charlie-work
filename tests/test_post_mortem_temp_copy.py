"""Temp-copy fallback safeguard tests (issue #1234).

Split out of ``tests/test_post_mortem.py`` (issue #1567, Track 1): the
size-cap refusal, rmtree cleanup of -shm/-wal residue, the reclamation
sweep, log-on-cleanup-failure, and the classify_and_record integration
paths for locked / oversized sessions.db.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _post_mortem_fixtures import (
    _NOW,
    _build_sessions_db,
    _config_with_db,
    _make_worker,
)

from charlie_work.post_mortem import (
    _cleanup_temp_copy,
    _open_readonly,
    _reclaim_stale_temp_copies,
    classify_and_record,
    read_post_mortem,
)


def test_classify_and_record_locked_db_falls_back_to_temp_copy(
    tmp_path: Path, monkeypatch
) -> None:
    """A sessions.db held open with an exclusive lock by another connection
    must not prevent extraction — the copy-to-temp fallback must still
    succeed (or, if it can't, degrade gracefully; either way, never raise).

    The reclamation sweep (issue #1234) and mkdtemp are isolated to tmp_path
    via monkeypatched ``tempfile.gettempdir`` so the test never touches the
    real system temp dir.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[("tool", json.dumps({"tool": "bash"}), "2026-07-11T11:57:00")],
    )

    # Isolate the temp-copy fallback's mkdtemp and the reclamation sweep to
    # tmp_path so the test never creates/deletes dirs in the real system temp.
    monkeypatch.setattr("charlie_work.post_mortem.tempfile.gettempdir", lambda: str(tmp_path))

    # Hold an exclusive lock on the DB via a separate connection + BEGIN EXCLUSIVE,
    # simulating the live Devin CLI process holding the file open.
    locker = sqlite3.connect(db_path, timeout=1)
    locker.execute("BEGIN EXCLUSIVE")

    try:
        config = _config_with_db(db_path)
        worker = _make_worker()
        sessions_dir = tmp_path / "sessions"

        # Must not raise regardless of whether the fallback succeeds.
        classify_and_record(sessions_dir, config, worker, now=_NOW)

        record = read_post_mortem(sessions_dir, worker.issue_number)
        assert record is not None
        # Either the temp-copy fallback succeeded (matched=True) or it also
        # failed and extraction_error was recorded — both are acceptable
        # degradation outcomes; a raised exception is the only failure.
        assert record.matched is True or record.extraction_error is not None
    finally:
        locker.rollback()
        locker.close()


# ---------------------------------------------------------------------------
# Issue #1234: temp-copy fallback disk leak safeguards
# (size-cap, rmtree cleanup, reclamation sweep, log-on-cleanup-failure)
# ---------------------------------------------------------------------------


def test_open_readonly_refuses_oversized_temp_copy(tmp_path: Path, monkeypatch) -> None:
    """When sessions.db exceeds temp_copy_max_bytes and the read-only URI
    fails (locked DB), the fallback refuses to copy and returns a size error
    instead of writing a multi-GB temp copy (issue #1234 root cause #1).

    MUTATION GATE: reverting the size-cap check in _open_readonly (so the
    fallback always copies) makes this test fail — the connection opens
    against the temp copy instead of returning the size-limit error.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[("tool", json.dumps({"tool": "bash"}), "2026-07-11T11:57:00")],
    )
    db_size = db_path.stat().st_size

    # Isolate mkdtemp + sweep to tmp_path so the test never touches the real
    # system temp dir.
    monkeypatch.setattr("charlie_work.post_mortem.tempfile.gettempdir", lambda: str(tmp_path))

    # Force the read-only URI path to fail so the fallback is exercised.
    real_connect = sqlite3.connect
    call_count = {"n": 0}

    def fake_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr("charlie_work.post_mortem.sqlite3.connect", fake_connect)

    # Set the cap below the fixture DB's actual size.
    conn, temp_copy, error = _open_readonly(
        db_path,
        temp_copy_max_bytes=db_size - 1,
        reclaim_max_age_hours=999999,  # no-op sweep
    )

    assert conn is None
    assert temp_copy is None
    assert error is not None
    assert "limit" in error
    # No temp copy dir was created.
    assert not list(tmp_path.glob("charlie-work-postmortem-*"))


def test_cleanup_temp_copy_removes_shm_wal_residue(tmp_path: Path) -> None:
    """_cleanup_temp_copy must remove the whole mkdtemp dir, including
    sessions.db-shm/-wal siblings created by opening the copy — the old
    unlink+rmdir missed them and leaked the dir forever (issue #1234 #2).

    MUTATION GATE: reverting _cleanup_temp_copy to the old unlink+rmdir
    form makes this test fail — rmdir raises on the non-empty dir (the
    -shm/-wal files remain) and the dir survives.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="charlie-work-postmortem-", dir=str(tmp_path)))
    db_copy = tmp_dir / "sessions.db"
    db_copy.write_bytes(b"dummy")
    # Simulate SQLite opening the copy and creating -shm/-wal siblings.
    (tmp_dir / "sessions.db-shm").write_bytes(b"shm")
    (tmp_dir / "sessions.db-wal").write_bytes(b"wal")

    _cleanup_temp_copy(db_copy)

    assert not tmp_dir.exists()
    assert not db_copy.exists()


def test_reclaim_stale_temp_copies_removes_old_keeps_fresh(tmp_path: Path, monkeypatch) -> None:
    """_reclaim_stale_temp_copies removes charlie-work-postmortem-* dirs
    older than the threshold but leaves fresh ones (issue #1234 #4).

    MUTATION GATE: reverting the mtime>cutoff guard (so all dirs are
    removed) makes this test fail — the fresh dir would be deleted too.
    """
    monkeypatch.setattr("charlie_work.post_mortem.tempfile.gettempdir", lambda: str(tmp_path))

    old_dir = tmp_path / "charlie-work-postmortem-old"
    old_dir.mkdir()
    (old_dir / "sessions.db").write_bytes(b"x")
    # Backdate the dir's mtime to 3 hours ago.
    old_time = (datetime.now(UTC) - timedelta(hours=3)).timestamp()
    os.utime(old_dir, (old_time, old_time))

    fresh_dir = tmp_path / "charlie-work-postmortem-fresh"
    fresh_dir.mkdir()
    (fresh_dir / "sessions.db").write_bytes(b"x")

    # A non-matching dir must be ignored.
    other_dir = tmp_path / "something-else"
    other_dir.mkdir()

    reclaimed = _reclaim_stale_temp_copies(max_age_hours=2)

    assert reclaimed == 1
    assert not old_dir.exists()
    assert fresh_dir.exists()
    assert other_dir.exists()


def test_cleanup_temp_copy_logs_when_rmtree_fails(tmp_path: Path, monkeypatch, caplog) -> None:
    """When rmtree cannot remove the temp dir, a warning is logged instead
    of silently swallowing the failure (issue #1234: log-on-cleanup-failure).

    MUTATION GATE: reverting _cleanup_temp_copy to the old except-OSError-pass
    form (no logging) makes this test fail — no warning record is emitted.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="charlie-work-postmortem-", dir=str(tmp_path)))
    db_copy = tmp_dir / "sessions.db"
    db_copy.write_bytes(b"x")

    # Make rmtree a no-op so the dir survives and the warning fires.
    monkeypatch.setattr("charlie_work.post_mortem.shutil.rmtree", lambda *a, **k: None)

    with caplog.at_level(logging.WARNING, logger="charlie_work.post_mortem"):
        _cleanup_temp_copy(db_copy)

    assert tmp_dir.exists()  # rmtree was a no-op, dir still there
    assert any("failed to clean up" in r.message for r in caplog.records)


def test_classify_and_record_oversized_db_records_size_error(tmp_path: Path, monkeypatch) -> None:
    """Integration: a locked sessions.db exceeding temp_copy_max_bytes must
    record an extraction_error mentioning the size limit, never raise, and
    leave no temp copy behind (issue #1234 #1).
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[("tool", json.dumps({"tool": "bash"}), "2026-07-11T11:57:00")],
    )
    db_size = db_path.stat().st_size

    monkeypatch.setattr("charlie_work.post_mortem.tempfile.gettempdir", lambda: str(tmp_path))

    # Force the read-only URI to fail so the fallback is exercised.
    real_connect = sqlite3.connect
    call_count = {"n": 0}

    def fake_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr("charlie_work.post_mortem.sqlite3.connect", fake_connect)

    config = _config_with_db(db_path, temp_copy_max_bytes=db_size - 1)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None  # no worker_blocked signal from an unopenable DB
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False
    assert record.extraction_error is not None
    assert "limit" in record.extraction_error
    # No temp copy dir leaked.
    assert not list(tmp_path.glob("charlie-work-postmortem-*"))
