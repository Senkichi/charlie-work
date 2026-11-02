"""Tests for ``process_utils.get_process_start_time`` (issue #2232)."""

from __future__ import annotations

import io
import time
from typing import Any

import pytest

import charlie_work.process_utils as process_utils
from charlie_work.process_utils import get_process_start_time

# A well-formed /proc/<pid>/stat line: comm="(python)", starttime=100 ticks
# at index 19 after the last ')' (same field layout as the
# parse_proc_stat_starttime tests).
_STAT_LINE = (
    "1234 (python) S 1 1234 1234 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 "
    "100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 "
    "1600 1700 1800 1900 2000 21000 22000 23000 24000 25000 26000 27000 "
    "28000 29000 30000 31000 32000 33000 34000 35000 36000 37000 38000 "
    "39000 40000 41000 42000 43000 44000 45000 46000 47000 48000 49000 "
    "50000 51000 52000"
)


def _force_posix_procfs(monkeypatch: pytest.MonkeyPatch, uptime: str | BaseException) -> None:
    """Force the POSIX branch and fabricate /proc reads for ``get_process_start_time``.

    CI runs self-hosted on Windows, so the POSIX branch is forced via a
    ``sys.platform`` monkeypatch (same seam as ``test_process_utils_is_pid_alive``),
    ``open`` is shadowed at module scope to serve ``_STAT_LINE`` for
    ``/proc/<pid>/stat``, and ``/proc/uptime`` either reads ``uptime`` or
    raises it when it is an exception. ``os.sysconf`` is stubbed because it
    does not exist on Windows.
    """
    monkeypatch.setattr("charlie_work.process_utils.sys.platform", "linux")

    def fake_open(path: object, *args: Any, **kwargs: Any) -> Any:
        name = str(path)
        if name == "/proc/uptime":
            if isinstance(uptime, BaseException):
                raise uptime
            return io.StringIO(uptime)
        if name.startswith("/proc/") and name.endswith("/stat"):
            return io.StringIO(_STAT_LINE)
        raise AssertionError(f"unexpected open: {name}")

    monkeypatch.setattr(process_utils, "open", fake_open, raising=False)
    monkeypatch.setattr(process_utils.os, "sysconf", lambda name: 100, raising=False)


def test_get_process_start_time_returns_none_when_proc_uptime_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #2232: unreadable /proc/uptime must yield ``None``, not a drifting value.

    Substituting ``uptime_seconds = 0`` made the result ``time.time() +
    ticks/hz`` -- a value that drifts with the wall clock, so a start time
    recorded at launch never matches a later re-read and ``is_pid_alive``
    can report a live worker as dead. ``None`` is indeterminate (fail-open:
    the caller treats the process as alive), matching the pre-wave-D copies.
    """
    _force_posix_procfs(monkeypatch, OSError("simulated unreadable /proc/uptime"))

    assert get_process_start_time(1234) is None


def test_get_process_start_time_posix_computes_from_proc_uptime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: a readable /proc/uptime still yields a start time.

    Asserts the ``time.time() - uptime + ticks/hz`` plumbing so the None
    path is proven specific to the uptime read failing, not a blanket None.
    """
    _force_posix_procfs(monkeypatch, "1000.00 2500.00")

    result = get_process_start_time(1234)

    # boot_time = time.time() - 1000; starttime 100 ticks / hz 100 = +1.0s.
    assert result is not None
    assert result == pytest.approx(time.time() - 1000.0 + 1.0, abs=5.0)
