"""Regression tests for ``charlie_work.file_lock``."""

from __future__ import annotations

from pathlib import Path

from charlie_work.file_lock import try_acquire_byte_range_lock


def test_byte_range_lock_acquires_zero_byte_file(tmp_path: Path) -> None:
    """A pre-existing 0-byte lock file can be acquired and released.

    Probe on the deployed runtime (Python 3.13.5, Windows 11):
    ``msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)`` succeeds on a genuine 0-byte
    file and the file size remains 0. The byte-range lock helper must
    therefore acquire the lock without padding the file.
    """
    lock_path = tmp_path / "lock.lock"
    lock_path.write_bytes(b"")  # simulate an old touch()-created 0-byte file
    assert lock_path.stat().st_size == 0

    lock = try_acquire_byte_range_lock(lock_path)
    assert lock is not None, "0-byte pre-existing lock file should be acquirable"
    assert lock_path.stat().st_size == 0, "lock acquisition should not pad the file"

    lock.release()

    lock2 = try_acquire_byte_range_lock(lock_path)
    assert lock2 is not None, "lock should be released after first release"
    lock2.release()
