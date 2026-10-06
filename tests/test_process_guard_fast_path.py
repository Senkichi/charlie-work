"""Tests for the leak guard's Job Object fast path (HS-CW-2).

Each deliberately leaked sleeper is killed inside the test body (by the guard
under test, or by the test itself), so the autouse guard in ``conftest.py``
observes a clean teardown.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import _process_guard
from _process_guard import (
    _THIS_PID,
    FAST_GUARDS_ENV,
    job_process_ids,
    leak_guard,
)

_BASE_EXECUTABLE = getattr(sys, "_base_executable", sys.executable)


def _spawn_sleeper(seconds: int = 60) -> subprocess.Popen:
    # The base interpreter, not the venv launcher: one fully materialized child.
    return subprocess.Popen(
        [_BASE_EXECUTABLE, "-c", f"import time; time.sleep({seconds})"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_job_process_ids_lists_this_process_and_a_new_child() -> None:
    if job_process_ids() is None:
        pytest.skip("no kill-on-close job in this process (off Windows, or nesting refused)")
    proc = _spawn_sleeper()
    try:
        ids = job_process_ids()
        assert ids is not None
        assert _THIS_PID in ids
        assert proc.pid in ids
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_job_process_ids_is_none_without_a_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_process_guard, "_job_handle", None)
    assert job_process_ids() is None


def test_fast_guards_switch_is_read_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FAST_GUARDS_ENV, "off")
    assert _process_guard.fast_guards_enabled() is False
    monkeypatch.setenv(FAST_GUARDS_ENV, "on")
    assert _process_guard.fast_guards_enabled() is True


@pytest.mark.parametrize("switch", ["on", "off"])
def test_leak_guard_fails_a_leaked_child_naming_its_pid(
    monkeypatch: pytest.MonkeyPatch, switch: str
) -> None:
    """Positive control, on both paths: a real leak fails with the pid in the message."""
    monkeypatch.setenv(FAST_GUARDS_ENV, switch)
    spawned: list[subprocess.Popen] = []
    with pytest.raises(
        pytest.fail.Exception, match=r"test left live child process\(es\) behind:\n  pid=\d+"
    ) as excinfo:
        with leak_guard(grace=0.2):
            spawned.append(_spawn_sleeper())
    [proc] = spawned
    assert f"pid={proc.pid} " in str(excinfo.value)
    proc.wait(timeout=10)  # the guard killed it; reap the Popen handle


def test_unchanged_job_skips_the_psutil_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FAST_GUARDS_ENV, raising=False)
    calls: list[str] = []
    monkeypatch.setattr(_process_guard, "job_member_snapshot", lambda: {4242: 1.0})
    monkeypatch.setattr(_process_guard, "descendant_snapshot", lambda: calls.append("walk") or {})
    monkeypatch.setattr(
        _process_guard,
        "reap_leaked_descendants",
        lambda before, grace: calls.append("reap") or [],
    )
    with leak_guard():
        pass
    assert calls == []


def test_kill_switch_takes_the_psutil_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FAST_GUARDS_ENV, "off")
    calls: list[str] = []
    reaped: list[dict[int, float]] = []
    monkeypatch.setattr(_process_guard, "job_member_snapshot", lambda: calls.append("job") or {})
    monkeypatch.setattr(
        _process_guard, "descendant_snapshot", lambda: calls.append("walk") or {7: 1.0}
    )
    monkeypatch.setattr(
        _process_guard,
        "reap_leaked_descendants",
        lambda before, grace: reaped.append(before) or [],
    )
    with leak_guard():
        pass
    assert calls == ["walk"]
    assert reaped == [{7: 1.0}]


def test_changed_job_reaps_against_pre_job_and_setup_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(FAST_GUARDS_ENV, raising=False)
    # Third value: the autouse guard's own teardown also sees this patched function
    # (monkeypatch outlives it), so the iterator must not run dry.
    snapshots = iter([{100: 1.0}, {100: 1.0, 200: 2.0}, {100: 1.0, 200: 2.0}])
    reaped: list[dict[int, float]] = []
    monkeypatch.setattr(_process_guard, "job_member_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(_process_guard, "_pre_job_descendants", {7: 0.5})
    monkeypatch.setattr(
        _process_guard,
        "descendant_snapshot",
        lambda: pytest.fail("the fast path must not walk psutil at setup"),
    )
    monkeypatch.setattr(
        _process_guard,
        "reap_leaked_descendants",
        lambda before, grace: reaped.append(before) or [],
    )
    with leak_guard():
        pass
    assert reaped == [{7: 0.5, 100: 1.0}]


def test_no_job_falls_back_to_the_psutil_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FAST_GUARDS_ENV, raising=False)
    reaped: list[dict[int, float]] = []
    monkeypatch.setattr(_process_guard, "job_member_snapshot", lambda: None)
    monkeypatch.setattr(_process_guard, "descendant_snapshot", lambda: {9: 3.0})
    monkeypatch.setattr(
        _process_guard,
        "reap_leaked_descendants",
        lambda before, grace: reaped.append(before) or [],
    )
    with leak_guard():
        pass
    assert reaped == [{9: 3.0}]
