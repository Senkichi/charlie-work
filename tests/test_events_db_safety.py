"""``events.db`` corruption quarantine and unsafe-path guard (events_db_safety).

Reproduces the 2026-11-09 fleet incident: page 1 of ``events.db`` was
overwritten by one of its own index leaf pages, every open then failed with
"file is not a database", and ~30k events over three days were dropped while
the corrupt file stayed in place. The contract pinned here: the corrupt file
and its sidecars are preserved under one ``.corrupt-<stamp>`` name, a fresh
database is rebuilt at the original path, and the rebuild is itself recorded.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

import charlie_work.events_db_safety as safety
import charlie_work.instrumentation as instrumentation
from charlie_work.instrumentation import close_db, log_event, query_events

PAGE = 4096


@pytest.fixture
def state_path(tmp_path: Path):
    path = tmp_path / "state.json"
    yield path
    close_db(path)


def _seed(state_path: Path, n: int = 50) -> Path:
    for i in range(n):
        log_event(state_path, "dispatch", {"issue_number": i})
    close_db(state_path)
    db = state_path.parent / "events.db"
    assert db.read_bytes()[:16] == b"SQLite format 3\x00"
    return db


def _overwrite_page_one_with_page_two(db: Path) -> bytes:
    """The 11-09 signature: a legitimate non-header page written at offset 0."""
    data = bytearray(db.read_bytes())
    data[:PAGE] = data[PAGE : 2 * PAGE]
    db.write_bytes(bytes(data))
    return bytes(data)


def _preserved(tmp_path: Path) -> list[Path]:
    return sorted(tmp_path.glob("events.db.corrupt-*"))


def test_corrupt_page_one_is_preserved_and_rebuilt(state_path: Path, tmp_path: Path) -> None:
    db = _seed(state_path)
    corrupt_bytes = _overwrite_page_one_with_page_two(db)
    stale_wal = db.with_name("events.db-wal")
    stale_wal.write_bytes(b"stale frames that must never replay into the fresh db")

    log_event(state_path, "dispatch", {"issue_number": 999})

    preserved = _preserved(tmp_path)
    main = [p for p in preserved if not p.name.endswith(safety.SIDECAR_SUFFIXES)]
    assert len(main) == 1, preserved
    assert main[0].read_bytes() == corrupt_bytes  # moved, byte-for-byte, never deleted
    moved_wal = main[0].with_name(main[0].name + "-wal")
    assert moved_wal.read_bytes().startswith(b"stale frames")
    assert not stale_wal.exists() or stale_wal.read_bytes()[:5] != b"stale"

    kinds = [e["kind"] for e in query_events(state_path)]
    assert "dispatch" in kinds
    quarantine = query_events(state_path, kind="events_db_quarantined")
    assert len(quarantine) == 1
    assert quarantine[0]["payload"]["preserved_as"] == str(main[0])
    assert "not a database" in quarantine[0]["payload"]["error"]


def test_healthy_database_is_never_moved(state_path: Path, tmp_path: Path) -> None:
    _seed(state_path)
    log_event(state_path, "dispatch", {"issue_number": 1})
    assert _preserved(tmp_path) == []
    assert query_events(state_path, kind="events_db_quarantined") == []


def test_write_time_corruption_evicts_and_quarantines(state_path: Path, tmp_path: Path) -> None:
    log_event(state_path, "dispatch", {"issue_number": 1})
    key = instrumentation._db_key(state_path)
    assert key in instrumentation._db_connections
    corrupt_conn = instrumentation._db_connections[key]
    exc = sqlite3.DatabaseError("database disk image is malformed")
    exc.sqlite_errorcode = 11  # SQLITE_CORRUPT

    instrumentation._quarantine_on_corruption(state_path, exc)

    # The corrupt connection was evicted; emitting the quarantine record then
    # reopened and re-cached a FRESH connection on the rebuilt database.
    assert instrumentation._db_connections[key] is not corrupt_conn
    assert len(_preserved(tmp_path)) == 1
    # The quarantine itself is recorded in the rebuilt database: the next
    # _get_db opens a fresh file with quarantined=None, so the write-time
    # path emits events_db_quarantined directly or the rebuild would go
    # unrecorded.
    quarantine = query_events(state_path, kind="events_db_quarantined")
    assert len(quarantine) == 1
    assert quarantine[0]["payload"]["preserved_as"] == str(_preserved(tmp_path)[0])
    assert "malformed" in quarantine[0]["payload"]["error"]
    log_event(state_path, "dispatch", {"issue_number": 2})
    assert [e["payload"]["issue_number"] for e in query_events(state_path, kind="dispatch")] == [2]


def test_quarantine_record_write_corruption_is_re_entrancy_guarded(
    state_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The events_db_quarantined record's own write reporting corruption on
    the fresh rebuild must not recurse: the write-time guard bounds it at
    one nested quarantine, and the OPEN-time emission (previously unguarded
    -- adversarial review round 2) funnels through the same guard."""
    log_event(state_path, "dispatch", {"issue_number": 1})
    real_log_event = instrumentation.log_event
    depth: list[str] = []

    def corrupting_record(*args, **kwargs):
        if len(args) >= 2 and args[1] == "events_db_quarantined":
            depth.append("record")
            exc = sqlite3.DatabaseError("database disk image is malformed")
            exc.sqlite_errorcode = 11
            instrumentation._quarantine_on_corruption(state_path, exc)
            return
        return real_log_event(*args, **kwargs)

    monkeypatch.setattr(instrumentation, "log_event", corrupting_record)
    exc = sqlite3.DatabaseError("database disk image is malformed")
    exc.sqlite_errorcode = 11
    instrumentation._quarantine_on_corruption(state_path, exc)
    # Exactly one nested record write was attempted; the nested quarantine
    # found no recorded identity (the connection was already evicted), so
    # the fresh file is untouched and the loop is bounded.
    assert len(depth) == 1
    assert len(_preserved(tmp_path)) == 1


def test_open_time_quarantine_record_is_guarded_too(
    state_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The open-time events_db_quarantined emission shares the
    _quarantining guard: a record write that itself reports corruption
    quarantines the fresh rebuild once and does not loop."""
    db = tmp_path / "events.db"
    db.write_bytes(b"not a sqlite header at all")
    real_log_event = instrumentation.log_event
    depth: list[str] = []

    def corrupting_record(*args, **kwargs):
        if len(args) >= 2 and args[1] == "events_db_quarantined":
            depth.append("record")
            exc = sqlite3.DatabaseError("database disk image is malformed")
            exc.sqlite_errorcode = 11
            instrumentation._quarantine_on_corruption(state_path, exc)
            return
        return real_log_event(*args, **kwargs)

    monkeypatch.setattr(instrumentation, "log_event", corrupting_record)
    instrumentation._get_db(state_path)
    assert len(depth) == 1
    assert len(_preserved(tmp_path)) <= 2
    instrumentation.close_db(state_path)


def test_busy_or_locked_is_not_corruption() -> None:
    busy = sqlite3.OperationalError("database is locked")
    busy.sqlite_errorcode = 5  # SQLITE_BUSY
    assert not safety.is_corruption_error(busy)
    notadb = sqlite3.DatabaseError("file is not a database")
    notadb.sqlite_errorcode = 26
    assert safety.is_corruption_error(notadb)
    assert safety.is_corruption_error(sqlite3.DatabaseError("file is not a database"))
    assert not safety.is_corruption_error(OSError("file is not a database"))


def test_quarantine_skips_a_file_rebuilt_by_another_process(tmp_path: Path) -> None:
    db = tmp_path / "events.db"
    db.write_bytes(b"corrupt")
    observed = safety.file_identity(db)
    db.unlink()  # another process quarantined it ...
    db.write_bytes(b"fresh")  # ... and rebuilt a new file at the same name
    if safety.file_identity(db) == observed:
        pytest.skip("filesystem reused the inode; identity cannot discriminate here")
    assert safety.quarantine_db(db, observed=observed) is None
    assert db.read_bytes() == b"fresh"


def test_quarantine_is_all_or_nothing_when_a_sidecar_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "events.db"
    db.write_bytes(b"corrupt")
    wal = tmp_path / "events.db-wal"
    wal.write_bytes(b"frames")
    real_rename = os.rename

    def held_wal(src, dst):
        if str(src).endswith("-wal"):
            raise PermissionError("held open by another process")
        return real_rename(src, dst)

    monkeypatch.setattr(safety.os, "rename", held_wal)
    assert safety.quarantine_db(db, observed=safety.file_identity(db)) is None
    assert db.read_bytes() == b"corrupt"  # rolled back to its original name
    assert wal.read_bytes() == b"frames"
    assert _preserved(tmp_path) == []


def test_quarantine_moves_sidecar_that_appears_mid_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lexists-then-rename pre-check used to leave a window where a
    sidecar materializing between the check and the rename stayed beside
    the freshly rebuilt database -- the replay hazard the all-or-nothing
    rollback exists to prevent. Sidecars are renamed FIRST-MOVE now: an
    absent sidecar's FileNotFoundError is the skip, and one that appears in
    the window is moved with the rest (adversarial review round 2)."""
    db = tmp_path / "events.db"
    db.write_bytes(b"corrupt")
    real_rename = os.rename

    def racing_rename(src, dst):
        # While the main file is being moved aside, another process's
        # SQLite writer creates the -wal sidecar.
        if str(src) == str(db):
            (tmp_path / "events.db-wal").write_bytes(b"late frames")
        return real_rename(src, dst)

    monkeypatch.setattr(safety.os, "rename", racing_rename)
    moved = safety.quarantine_db(db, observed=safety.file_identity(db))
    assert moved is not None
    assert (tmp_path / f"{moved.name}-wal").read_bytes() == b"late frames"
    assert not (tmp_path / "events.db-wal").exists()


def test_quarantine_returns_none_when_main_file_vanishes_mid_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A main file deleted between the identity check and the rename is the
    identity-mismatch case: roll back any already-moved sidecars and leave
    everything in place."""
    db = tmp_path / "events.db"
    db.write_bytes(b"corrupt")
    wal = tmp_path / "events.db-wal"
    wal.write_bytes(b"frames")
    real_rename = os.rename

    def vanishing_rename(src, dst):
        if str(src) == str(db):
            db.unlink()  # another process quarantined it first
            raise FileNotFoundError(src)
        return real_rename(src, dst)

    monkeypatch.setattr(safety.os, "rename", vanishing_rename)
    assert safety.quarantine_db(db, observed=safety.file_identity(db)) is None
    assert wal.read_bytes() == b"frames"  # sidecar rolled back / never moved
    assert _preserved(tmp_path) == []


def test_network_path_is_unsafe_for_wal() -> None:
    reason = safety.wal_unsafe_reason(Path("//localhost/C$/Users/x/charlie-work/events.db"))
    assert reason is not None and "network path" in reason


def test_msix_forked_sidecar_is_unsafe_for_wal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "events.db"
    (tmp_path / "events.db-wal").write_bytes(b"")
    private = tmp_path / "LocalCache" / "events.db-wal"
    monkeypatch.setattr(
        safety,
        "detect_path_virtualization",
        lambda p: (p, private) if p.name.endswith("-wal") else None,
    )
    reason = safety.wal_unsafe_reason(db)
    assert reason is not None and "MSIX" in reason


def test_unsafe_path_drops_events_without_touching_the_database(
    state_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.setattr(safety, "wal_unsafe_reason", lambda p: "on a network path")
    with caplog.at_level("WARNING", logger="charlie_work.instrumentation"):
        log_event(state_path, "dispatch", {"issue_number": 1})
        log_event(state_path, "dispatch", {"issue_number": 2})
    assert not (tmp_path / "events.db").exists()
    warnings = [r for r in caplog.records if "Not opening event database" in r.getMessage()]
    assert len(warnings) == 1  # once per path, not once per event
    instrumentation._unsafe_db_warned.discard(instrumentation._db_key(state_path))


def test_open_or_rebuild_retries_once_after_post_open_corruption(tmp_path: Path) -> None:
    db = tmp_path / "events.db"
    db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)  # header intact, body malformed
    calls: list[int] = []

    def opener() -> str:
        calls.append(1)
        if len(calls) == 1:
            exc = sqlite3.DatabaseError("database disk image is malformed")
            exc.sqlite_errorcode = 11
            raise exc
        return "fresh-connection"

    conn, quarantined = safety.open_or_rebuild(db, opener)
    assert conn == "fresh-connection" and len(calls) == 2
    assert quarantined is not None and quarantined[0].exists()
    assert "malformed" in quarantined[1]


def test_open_or_rebuild_reraises_non_corruption(tmp_path: Path) -> None:
    def opener() -> str:
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError):
        safety.open_or_rebuild(tmp_path / "events.db", opener)
    assert _preserved(tmp_path) == []


def test_headerless_db_with_held_sidecar_never_reaches_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A headerless file that cannot be quarantined must not be opened at
    all: connecting raises SQLITE_NOTADB and closing that connection makes
    SQLite delete the -wal/-shm sidecars (the forensic copy the quarantine
    rollback just restored)."""
    db = tmp_path / "events.db"
    db.write_bytes(b"not a sqlite header at all")
    wal = tmp_path / "events.db-wal"
    wal.write_bytes(b"uncheckpointed frames")
    shm = tmp_path / "events.db-shm"
    shm.write_bytes(b"wal index")
    real_rename = os.rename

    def held_wal(src, dst):
        if str(src).endswith("-wal"):
            raise PermissionError("held open by another process")
        return real_rename(src, dst)

    monkeypatch.setattr(safety.os, "rename", held_wal)
    calls: list[int] = []

    def opener() -> str:
        calls.append(1)
        return "should never be reached"

    with pytest.raises(sqlite3.DatabaseError, match="could not be quarantined"):
        safety.open_or_rebuild(db, opener)
    assert calls == []  # opener() was never invoked on the headerless file
    # Everything is still in place for the retry -- and for forensics.
    assert db.read_bytes().startswith(b"not a sqlite")
    assert wal.read_bytes() == b"uncheckpointed frames"
    assert shm.read_bytes() == b"wal index"
    assert _preserved(tmp_path) == []


def test_headerless_db_quarantined_then_rebuilt(tmp_path: Path) -> None:
    """The 11-09 incident shape end to end: headerless main file quarantined
    with its sidecars under one .corrupt-<stamp> name, fresh db rebuilt."""
    db = tmp_path / "events.db"
    db.write_bytes(b"not a sqlite header at all")
    wal = tmp_path / "events.db-wal"
    wal.write_bytes(b"frames")

    conn, quarantined = safety.open_or_rebuild(db, lambda: "new-connection")
    assert conn == "new-connection"
    assert quarantined is not None
    preserved, reason = quarantined
    assert preserved.read_bytes() == b"not a sqlite header at all"
    assert "not a database" in reason
    assert preserved.with_name(preserved.name + "-wal").read_bytes() == b"frames"
    assert not wal.exists()
