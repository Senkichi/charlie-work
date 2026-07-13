"""Regression tests for ``charlie_work.state.state_lock`` lock lifecycle.

These target two review findings from PR #248 (supervised-infill-loop):

- finding #4: on the 30s timeout path (lock never acquired), the opened lock
  file handle must still be closed -- previously ``close()`` was gated on
  ``acquired``, so a timed-out lock leaked the handle for the life of the
  process.
- finding #8: a pre-existing 0-byte lock file (e.g. left over from an older
  ``touch()``-based implementation) must not permanently block acquisition.
  Finding #8 originally claimed ``msvcrt.locking`` raises ``EACCES`` on a
  0-byte file; probing the deployed runtime (Python 3.13.5, Windows 11) in
  #324/#328 disproved that -- ``LK_NBLCK`` with ``nbytes=1`` succeeds on a
  genuine 0-byte file, so the write-1-byte guards were removed as dead code
  and the test below now characterizes lock acquisition on the bare 0-byte
  file.

Both behaviors are Windows-specific (``msvcrt`` byte-range locking); these
tests are skipped on non-Windows platforms.
"""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

import pytest

from charlie_work import state as state_module

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="msvcrt byte-range locking is Windows-specific"
)


def test_state_lock_timeout_closes_handle(tmp_path: Path, monkeypatch) -> None:
    """Regression for finding #4: the timeout path (acquired=False) must
    still close the handle it opened, not just skip unlocking.
    """
    import msvcrt

    state_path = tmp_path / "state.json"
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.write_bytes(b"\x00")

    # Hold a real competing lock on the same file so every retry inside
    # state_lock fails and the timeout branch (acquired=False) is exercised.
    blocker = lock_path.open("r+b")
    msvcrt.locking(blocker.fileno(), msvcrt.LK_NBLCK, 1)

    opened: list = []
    orig_open = pathlib.Path.open

    def tracking_open(self, *args, **kwargs):
        handle = orig_open(self, *args, **kwargs)
        if self == lock_path:
            opened.append(handle)
        return handle

    monkeypatch.setattr(pathlib.Path, "open", tracking_open)
    monkeypatch.setattr(state_module, "_LOCK_TIMEOUT_SECONDS", 0.05)

    try:
        with state_module.state_lock(state_path):
            pass
    finally:
        msvcrt.locking(blocker.fileno(), msvcrt.LK_UNLCK, 1)
        blocker.close()

    assert len(opened) == 1, "state_lock should open exactly one handle on the lock file"
    assert opened[0].closed is True, "handle opened on the timeout path was never closed (leak)"


def test_state_lock_zero_byte_existing_file_acquires(tmp_path: Path, caplog) -> None:
    """Characterization for finding #8: a pre-existing 0-byte lock file must
    not permanently block acquisition on the state_lock path either.

    Probe on the deployed runtime (Python 3.13.5, Windows 11):
    ``msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)`` succeeds on a genuine 0-byte
    file and the file size remains 0, so ``state_lock`` carries no padding
    guard (#324/#328) and must acquire the bare 0-byte file directly.

    ``state_lock`` is best-effort: it yields regardless of whether the lock
    was actually acquired, so a bare ``with state_lock(...): pass`` succeeding
    proves nothing on its own -- a genuine 0-byte acquisition failure would
    still "succeed" after silently falling through the 30s timeout. Assert
    the acquire-failed warning is NOT logged, proving the lock was genuinely
    acquired on first try rather than via the best-effort timeout fallback.
    """
    import logging

    state_path = tmp_path / "state.json"
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.write_bytes(b"")  # simulate an old touch()-created 0-byte file
    assert lock_path.stat().st_size == 0

    with caplog.at_level(logging.WARNING, logger=state_module.__name__):
        with state_module.state_lock(state_path):
            pass

    assert not any("Failed to acquire lock" in record.message for record in caplog.records), (
        "state_lock fell through to the best-effort timeout path instead of "
        "genuinely acquiring the 0-byte lock file on the first try"
    )
    assert lock_path.stat().st_size == 0, "lock acquisition should not pad the file"
