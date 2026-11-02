"""Regression tests for ``charlie_work.state.state_lock`` lock lifecycle.

These target two review findings from PR #248 (supervised-infill-loop):

- finding #4: on the 30s timeout path (lock never acquired), the opened lock
  file handle must still be closed -- previously ``close()`` was gated on
  ``acquired``, so a timed-out lock leaked the handle for the life of the
  process. After issue #398 the timeout path raises ``StateLockBusy`` instead
  of proceeding unlocked, but the handle-closing requirement still holds.
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
import time
from pathlib import Path

import pytest

from charlie_work import state as state_module

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="msvcrt byte-range locking is Windows-specific"
)


def test_state_lock_timeout_closes_handle(tmp_path: Path, monkeypatch) -> None:
    """Regression for finding #4: the timeout path must raise StateLockBusy
    and still close the handle it opened, not just skip unlocking.
    """
    import msvcrt

    state_path = tmp_path / "state.json"
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.write_bytes(b"\x00")

    # Hold a real competing lock on the same file so every retry inside
    # state_lock fails and the timeout branch is exercised.
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
        with pytest.raises(state_module.StateLockBusy):
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

    ``state_lock`` now raises ``StateLockBusy`` on timeout, so a bare
    ``with state_lock(...): pass`` would fail loudly if the lock could not be
    acquired. Assert the acquire-failed warning is NOT logged, proving the lock
    was genuinely acquired on the first try rather than via the timeout path.
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
        "state_lock fell through to the timeout path instead of "
        "genuinely acquiring the 0-byte lock file on the first try"
    )
    assert lock_path.stat().st_size == 0, "lock acquisition should not pad the file"


def test_state_lock_timeout_immune_to_wall_clock_forward_jump(tmp_path: Path, monkeypatch) -> None:
    """Issue #2231: the lock-timeout retry loop must measure elapsed time on
    a monotonic clock, not wall-clock ``time.time()``.

    With wall-clock timing, a forward step (NTP correction, manual clock
    change, VM resume) makes ``time.time() - start`` exceed the timeout on
    the first check, so the loop raised ``StateLockBusy`` without ever
    attempting the lock — a spurious failure of a unit of work on a FREE
    lock. Patch ``time.time`` to jump far past the timeout between the
    ``start`` sample and the loop check: the lock must still be acquired.
    """
    real_time = time.time
    calls = 0

    def jumping_wall_clock() -> float:
        nonlocal calls
        calls += 1
        # First call is the ``start =`` sample; every later call reports the
        # wall clock stepped far past the lock timeout.
        offset = 0.0 if calls == 1 else 10 * state_module._LOCK_TIMEOUT_SECONDS
        return real_time() + offset

    monkeypatch.setattr(time, "time", jumping_wall_clock)

    with state_module.state_lock(tmp_path / "state.json"):
        pass  # must acquire — must not raise StateLockBusy


def test_state_lock_timeout_immune_to_wall_clock_backward_jump(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #2231: a backward wall-clock step must not extend the timeout.

    With wall-clock timing against a HELD lock, ``time.time() - start``
    stays negative forever and the retry loop never times out — the unit of
    work waits far longer than ``_LOCK_TIMEOUT_SECONDS`` instead of failing
    with ``StateLockBusy``. Patch ``time.time`` to report the clock stepped
    backward; the timeout must still fire. A bounded ``time.sleep`` stands
    in for the hang unfixed code would produce, so this test fails fast
    rather than spinning forever on the old implementation.
    """
    import msvcrt

    state_path = tmp_path / "state.json"
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.write_bytes(b"\x00")

    # Hold a real competing lock so every retry inside state_lock fails.
    blocker = lock_path.open("r+b")
    msvcrt.locking(blocker.fileno(), msvcrt.LK_NBLCK, 1)

    real_time = time.time
    calls = 0

    def backward_jumping_clock() -> float:
        nonlocal calls
        calls += 1
        # First call is the ``start =`` sample; every later call reports the
        # wall clock stepped backward, so ``time.time() - start`` stays
        # negative forever under wall-clock timing.
        offset = 0.0 if calls == 1 else -1000.0
        return real_time() + offset

    monkeypatch.setattr(time, "time", backward_jumping_clock)
    monkeypatch.setattr(state_module, "_LOCK_TIMEOUT_SECONDS", 0.3)

    real_sleep = time.sleep
    sleeps = 0

    def bounded_sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps > 30:
            raise AssertionError("lock retry loop never timed out — is it using wall-clock time?")
        real_sleep(min(seconds, 0.02))

    monkeypatch.setattr(time, "sleep", bounded_sleep)

    try:
        with pytest.raises(state_module.StateLockBusy):
            with state_module.state_lock(state_path):
                pass
    finally:
        msvcrt.locking(blocker.fileno(), msvcrt.LK_UNLCK, 1)
        blocker.close()


def test_state_lock_timeout_uses_monotonic_on_posix_branch(tmp_path: Path, monkeypatch) -> None:
    """Issue #2231: the POSIX (fcntl) retry loop gets the same monotonic
    timeout as the Windows branch. Force the POSIX path on Windows via a
    stub ``fcntl`` module and a patched ``sys.platform`` so CI exercises
    both loops.
    """
    import types

    flocked: list[int] = []
    fake_fcntl = types.SimpleNamespace(
        LOCK_EX=2,
        LOCK_NB=4,
        LOCK_UN=8,
        flock=lambda _fd, op: flocked.append(op),
    )
    monkeypatch.setitem(sys.modules, "fcntl", fake_fcntl)
    monkeypatch.setattr(state_module.sys, "platform", "linux")

    real_time = time.time
    calls = 0

    def jumping_wall_clock() -> float:
        nonlocal calls
        calls += 1
        offset = 0.0 if calls == 1 else 10 * state_module._LOCK_TIMEOUT_SECONDS
        return real_time() + offset

    monkeypatch.setattr(time, "time", jumping_wall_clock)

    with state_module.state_lock(tmp_path / "state.json"):
        pass

    assert flocked, "the POSIX branch never attempted the lock"
