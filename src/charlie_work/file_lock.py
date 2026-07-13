"""Shared byte-range file lock helper.

Used by fleet and supervisor locks so the acquisition/release logic cannot
silently drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path


class ByteRangeFileLock:
    """Holds an OS-level non-blocking byte-range lock on a file.

    Released in ``__del__`` and explicit ``release()``; also released on process
    death (OS closes all file handles).
    """

    def __init__(self, path: Path, handle: object) -> None:
        self._path = path
        self._handle = handle
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        # Unlock and close are independent failure modes: an msvcrt/fcntl
        # unlock raising OSError must not skip closing the handle (that would
        # leak the file descriptor and, on Windows, keep the lock file
        # undeletable/unlockable by anyone else).
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        except OSError:
            pass
        try:
            self._handle.close()  # type: ignore[attr-defined]
        except OSError:
            pass

    def __del__(self) -> None:
        self.release()


def try_acquire_byte_range_lock(lock_path: Path) -> ByteRangeFileLock | None:
    """Try to acquire a non-blocking byte-range lock on ``lock_path``.

    Returns a ``ByteRangeFileLock`` if acquired; ``None`` if another process
    (or thread) holds it. Never raises.
    """
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Write 1 byte on creation so msvcrt.locking(..., 1) has a byte to lock.
        # touch() creates an empty (0-byte) file; msvcrt locks specific byte ranges
        # and will raise EACCES on a 0-byte file even for a non-blocking attempt.
        if not lock_path.exists():
            lock_path.write_bytes(b"\x00")

        handle = lock_path.open("r+b", encoding=None)
        if sys.platform == "win32":
            import msvcrt

            # Guard against a pre-existing 0-byte lock file (e.g. left over
            # from an older touch()-based implementation, or a race with
            # another process's creation) — the write above only fires when
            # the file doesn't exist yet, so a stale empty file would still
            # make msvcrt.locking raise EACCES.
            if handle.seek(0, 2) == 0:
                handle.write(b"\x00")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                handle.close()
                return None
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, BlockingIOError):
                handle.close()
                return None
        return ByteRangeFileLock(lock_path, handle)
    except OSError:
        return None
