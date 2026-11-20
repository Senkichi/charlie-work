"""Safety guards for opening ``events.db`` (SQLite, WAL mode).

Two failure classes, one module, both enforced at the single place
``instrumentation._get_db`` opens the database:

1. **Unsafe access paths (prevention).** ``events.db`` runs in WAL mode, and
   WAL coordinates every connection through a memory-mapped ``-shm`` index.
   That only works when every process maps the *same* file through the
   *same* filesystem view (https://www.sqlite.org/wal.html, "WAL does not work
   over a network filesystem"). This host has two ways to break it:

   * an SMB route such as the ``\\\\localhost\\C$`` administrative share that
     packaged (MSIX) sessions use for fleet state since #2806 -- the
     redirector's view of ``-shm`` is not coherent with the daemon's local
     mapping, so the WAL index the checkpointer trusts can map a frame to the
     wrong page;
   * MSIX copy-on-write redirection, which forks ``-wal``/``-shm`` (new files)
     into a private LocalCache while the main ``events.db`` (an existing file)
     is modified in place -- two WALs over one database.

   :func:`wal_unsafe_reason` names either condition so the caller skips the
   database instead of joining it (events are best-effort by contract; a
   dropped event is recoverable, a corrupted database is not).

2. **Corruption (recovery).** A database that already fails to open as
   SQLite ("file is not a database") or reports ``SQLITE_CORRUPT`` used to
   fail every open forever, silently dropping every event (the 2026-11-09
   fleet ``events.db`` page-1 overwrite logged ~30k warnings over three days).
   :func:`quarantine_db` moves the file and its ``-wal``/``-shm``/``-journal``
   sidecars aside under one ``.corrupt-<UTC stamp>`` name -- never deleting
   anything -- so the next open rebuilds a fresh database. The sidecars must
   move with it: a leftover ``-wal`` whose frames checksum correctly would be
   replayed into the fresh file.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, TypeVar

from charlie_work.fleet_paths import detect_path_virtualization

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_SQLITE_HEADER = b"SQLite format 3\x00"

# SQLite primary result codes (the low byte of an extended code).
_SQLITE_CORRUPT = 11
_SQLITE_NOTADB = 26
_CORRUPTION_CODES = frozenset({_SQLITE_CORRUPT, _SQLITE_NOTADB})
_CORRUPTION_MESSAGES = ("file is not a database", "database disk image is malformed")

# Every sidecar SQLite may keep next to a database, in move order after the
# main file. ``-wal``/``-shm`` are WAL mode; ``-journal`` is rollback mode.
SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

# GetDriveTypeW: DRIVE_REMOTE. Covers both UNC roots and mapped network drives.
_DRIVE_REMOTE = 4

FileIdentity = tuple[int, int]


def is_corruption_error(exc: BaseException) -> bool:
    """True when ``exc`` reports a corrupt database (not a transient failure).

    Prefers ``sqlite_errorcode`` (Python >= 3.11, extended code; the primary
    code is its low byte). Falls back to the message for errors raised
    without a code, so a locked or busy database is never classified as
    corrupt.
    """
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int):
        return (code & 0xFF) in _CORRUPTION_CODES
    message = str(exc).lower()
    return any(marker in message for marker in _CORRUPTION_MESSAGES)


def _is_remote_path(path: Path) -> bool:
    """True when ``path`` lives on a network filesystem (UNC or mapped drive)."""
    if path.drive.startswith(("\\\\", "//")):
        return True
    if sys.platform != "win32" or not path.drive:
        return False
    try:
        import ctypes

        drive_type = ctypes.windll.kernel32.GetDriveTypeW(path.drive + "\\")
    except (AttributeError, OSError):
        return False
    return drive_type == _DRIVE_REMOTE


def wal_unsafe_reason(db_path: Path) -> str | None:
    """Why ``db_path`` must not be opened in WAL mode by this process, if so.

    Returns ``None`` when the path is a local, unvirtualized file -- the only
    configuration in which SQLite WAL is safe across processes. The
    virtualization probe resolves only sidecars that already exist (a fresh
    database has none), so the one-``Path.resolve()``-per-``state_path``
    contract (#2566) is unchanged on the normal path.
    """
    if _is_remote_path(db_path):
        return (
            f"{db_path} is on a network path; SQLite WAL needs one shared-memory "
            "index on the same host, so this process would corrupt the database "
            "a local writer (the fleet daemon) holds open"
        )
    for suffix in SIDECAR_SUFFIXES:
        sidecar = db_path.with_name(db_path.name + suffix)
        if not os.path.lexists(sidecar):
            continue
        diverged = detect_path_virtualization(sidecar)
        if diverged is not None:
            return (
                f"{diverged[0]} resolves to {diverged[1]} (per-file MSIX fork); "
                "this process would use a private WAL over the shared database"
            )
    return None


def lacks_sqlite_header(db_path: Path) -> bool:
    """True when a non-empty ``db_path`` does not start with the SQLite header.

    Checked *before* SQLite touches the file: closing a connection that failed
    with ``SQLITE_NOTADB`` deletes the ``-wal``/``-shm`` sidecars, which
    destroys the forensic copy and any frames not yet checkpointed. An empty
    file is a valid (new) database.
    """
    try:
        with open(db_path, "rb") as fh:
            head = fh.read(len(_SQLITE_HEADER))
    except OSError:
        return False
    return bool(head) and head != _SQLITE_HEADER


def open_or_rebuild(db_path: Path, opener: Callable[[], _T]) -> tuple[_T, tuple[Path, str] | None]:
    """Open via ``opener``; on corruption, quarantine ``db_path`` and reopen once.

    Returns ``(connection, quarantined)`` where ``quarantined`` is
    ``(preserved_path, reason)`` when a rebuild happened. Re-raises the
    original error when it is not corruption or the file could not be moved
    aside, so the caller's best-effort handling stays the single fallback.
    """
    observed = file_identity(db_path)
    if lacks_sqlite_header(db_path):
        moved = quarantine_db(db_path, observed=observed)
        if moved is not None:
            return opener(), (moved, "file is not a database (no SQLite header on page 1)")
        # A headerless file that could NOT be moved aside must never be
        # opened anyway: connecting raises SQLITE_NOTADB, and closing that
        # connection deletes the -wal/-shm sidecars -- destroying the
        # forensic copy and any uncheckpointed frames (the very thing the
        # pre-open header check exists to protect). Report corruption
        # instead and let the caller's best-effort path drop events until
        # the rename blockage clears and the next open retries.
        exc = sqlite3.DatabaseError(
            f"{db_path}: file is not a database and could not be quarantined"
        )
        exc.sqlite_errorcode = _SQLITE_NOTADB
        raise exc
    try:
        return opener(), None
    except sqlite3.Error as exc:
        if not is_corruption_error(exc):
            raise
        moved = quarantine_db(db_path, observed=observed)
        if moved is None:
            raise
        return opener(), (moved, str(exc))


def file_identity(path: Path) -> FileIdentity | None:
    """``(st_dev, st_ino)`` of ``path``, or ``None`` when it does not exist.

    Used to make sure :func:`quarantine_db` moves the file whose corruption
    was observed and not a fresh database another process rebuilt meanwhile.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def quarantine_db(db_path: Path, *, observed: FileIdentity | None) -> Path | None:
    """Move a corrupt ``db_path`` and its sidecars aside; return the new path.

    The file is renamed to ``<name>.corrupt-<UTC stamp>`` and each sidecar to
    ``<name>.corrupt-<stamp><suffix>``, so SQLite still pairs them when the
    preserved copy is opened for forensics. Nothing is deleted.

    Sidecars are renamed FIRST-MOVE rather than pre-checked with
    ``lexists``: a pre-check leaves a window where a sidecar materializes
    between the check and the rename, and the skip would then leave it
    beside the freshly rebuilt database -- exactly the replay hazard the
    all-or-nothing rollback exists to prevent. The rename of an absent
    sidecar raises ``FileNotFoundError``, which is the skip; a sidecar that
    appears in the window is moved with the rest.

    Returns ``None`` (and leaves everything in place) when the file is gone,
    is no longer the file whose corruption was observed (``observed``
    identity mismatch -- another process already quarantined it and built a
    fresh one), or cannot be renamed (on Windows, another process still has
    it open; the next failing open retries).
    """
    if observed is None or file_identity(db_path) != observed:
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = db_path.with_name(f"{db_path.name}.corrupt-{stamp}")
    moves = [(db_path, target)] + [
        (db_path.with_name(db_path.name + s), target.with_name(target.name + s))
        for s in SIDECAR_SUFFIXES
    ]
    done: list[tuple[Path, Path]] = []
    for src, dst in moves:
        try:
            os.rename(src, dst)
        except FileNotFoundError:
            if src is not db_path:
                continue  # sidecar absent (or vanished mid-window): nothing to move
            # The main file vanished between the identity check and the
            # rename -- another process quarantined/rebuilt it; treat like
            # the identity-mismatch case.
            logger.warning("Corrupt event database %s disappeared before quarantine", db_path)
            for moved_src, moved_dst in reversed(done):
                try:
                    os.rename(moved_dst, moved_src)
                except OSError as undo_exc:
                    logger.error("Could not restore %s to %s: %s", moved_dst, moved_src, undo_exc)
            return None
        except OSError as exc:
            # All or nothing: a fresh database next to a left-behind ``-wal``
            # would replay the corrupt file's frames into itself.
            logger.warning("Could not move corrupt event database file %s aside: %s", src, exc)
            for moved_src, moved_dst in reversed(done):
                try:
                    os.rename(moved_dst, moved_src)
                except OSError as undo_exc:
                    logger.error("Could not restore %s to %s: %s", moved_dst, moved_src, undo_exc)
            return None
        done.append((src, dst))
    logger.warning(
        "Event database %s was corrupt; preserved it as %s and rebuilding a fresh one",
        db_path,
        target,
    )
    return target


# Memoized ``state_path`` -> resolved ``events.db`` key. Every write primitive
# used to re-resolve the db path with ``Path.resolve()`` on each call (two
# resolves per ``log_event`` call: once in ``_get_db``, once for the lock key),
# and ``nt._getfinalpathname`` is expensive on filesystems holding
# ``tmp_path`` (network/junctioned volumes), so a burst of events paid it
# 2N times per N events (issue #2566). Keys are absolute
# (``os.path.abspath``): a relative ``state_path`` outliving a cwd change
# keys a fresh entry against the new cwd instead of returning the previous
# cwd's resolved form. While that holds, the only way the memo can go stale
# without the cwd changing is the filesystem it names being re-linked
# mid-process, which also invalidates the open sqlite handle ``_get_db``
# would have returned — so the memo is invalidated exactly where the
# connection cache is (``close_db``).
_resolved_db_keys: dict[str, str] = {}


def _db_path(state_path: Path) -> Path:
    """Derive the ``events.db`` SQLite path from a ``state.json`` path."""
    return state_path.parent / "events.db"


def _db_memo_key(state_path: Path) -> str:
    """Return the memo/connection key spelling for a ``state_path``.

    Relative paths are normalized against the current working directory at
    lookup time (:func:`os.path.abspath`), so a relative ``state_path``
    outliving a cwd change keys a fresh cache entry instead of colliding
    with the previous cwd's.
    """
    return os.path.abspath(str(state_path))


def _db_key(state_path: Path) -> str:
    """Return the memoized resolved key for a ``state_path``'s ``events.db``.

    The key identifies the per-path connection and lock entries. Resolution
    is memoized, keyed on the ``state_path`` string normalized to its
    absolute form via :func:`os.path.abspath`: ``Path.resolve()`` bottoms
    out in ``nt._getfinalpathname``, which is a real syscall cost on
    network / junctioned filesystems, and every event write used to pay it
    twice per call. Keying on the absolute form keeps relative
    ``state_path`` values correct across cwd changes — a cwd change yields
    a different key (and a re-resolve) instead of serving the previous
    cwd's memo — while the absolute-spelling lookup itself is a string
    operation, not a syscall.

    The memo is invalidated by ``close_db`` alongside the connection and
    lock the key identifies. Within one open connection, the only way a
    key's resolved form can change is the filesystem it names being
    re-linked mid-process, which would invalidate the open sqlite handle
    ``_get_db`` would have returned — so the memo cannot outlive a
    connection the cache would still hand out under it.
    """
    key_input = _db_memo_key(state_path)
    cached = _resolved_db_keys.get(key_input)
    if cached is None:
        cached = str(_db_path(state_path).resolve())
        _resolved_db_keys[key_input] = cached
    return cached


def _jsonl_path(state_path: Path) -> Path:
    """Derive the legacy ``events.jsonl`` path from a ``state.json`` path."""
    return state_path.parent / "events.jsonl"
