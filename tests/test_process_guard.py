"""Self-tests for ``tests/_process_guard.py`` (issue #1851).

The conftest guard fails any test that leaves a live descendant of pytest
behind at teardown; these tests exercise the detection/report/reap machinery
that guard is built on. Each deliberately leaked sleeper is cleaned up inside
the test body, so the autouse guard itself observes a clean teardown.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from types import SimpleNamespace

import psutil
import pytest

from _process_guard import (
    _pid_collecting_wrapper,
    descendant_snapshot,
    enter_kill_on_close_job,
    leaked_descendants,
    reap_leaked_descendants,
    reap_pid,
    wrap_launchers,
)


def _spawn_sleeper(seconds: int = 60, *, direct: bool = False) -> subprocess.Popen:
    # ``sys.executable`` in a venv is a launcher that spawns the real
    # interpreter as a *grandchild* a beat later — the exact pair shape the
    # issue describes. ``direct=True`` uses the base interpreter instead, so
    # tests asserting on a single, fully-materialized child get one.
    executable = getattr(sys, "_base_executable", sys.executable) if direct else sys.executable
    return subprocess.Popen(
        [executable, "-c", f"import time; time.sleep({seconds})"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _reap(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)


def test_leaked_sleeper_is_reported_and_killed() -> None:
    """The guard's report path: a deliberately leaked sleeper is named by
    pid and command line, and killed rather than merely flagged."""
    before = descendant_snapshot()
    proc = _spawn_sleeper()
    try:
        report = reap_leaked_descendants(before, grace=0.0)
        assert report, "expected the leaked sleeper to be reported"
        assert any(str(proc.pid) in line and "time.sleep" in line for line in report), (
            f"report does not name the sleeper's cmdline: {report}"
        )
        proc.wait(timeout=10)
        assert proc.poll() is not None, "surviving leak was not killed"
    finally:
        _reap(proc)


def test_leaked_descendants_names_the_new_child() -> None:
    before = descendant_snapshot()
    proc = _spawn_sleeper()
    try:
        leaked = leaked_descendants(before)
        assert any(p.pid == proc.pid for p in leaked), (
            f"sleeper pid={proc.pid} missing from leaked descendants"
        )
    finally:
        _reap(proc)


def test_preexisting_children_are_not_leaks() -> None:
    """A descendant already alive at snapshot time is not this test's leak."""
    # direct=True: no venv-launcher grandchild can appear after the snapshot.
    proc = _spawn_sleeper(direct=True)
    try:
        before = descendant_snapshot()
        assert leaked_descendants(before) == []
        assert reap_leaked_descendants(before, grace=0.0) == []
    finally:
        _reap(proc)


def test_child_that_exits_within_grace_is_not_a_leak() -> None:
    """A fast-exiting child mid-flight at teardown must not fail the test."""
    before = descendant_snapshot()
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert reap_leaked_descendants(before, grace=10.0) == []
    finally:
        _reap(proc)


def test_reap_pid_kills_a_live_child() -> None:
    proc = _spawn_sleeper()
    reap_pid(proc.pid)
    proc.wait(timeout=10)
    assert proc.poll() is not None


def test_reap_pid_never_touches_foreign_or_missing_pids() -> None:
    """Fabricated record pids (delegation-test stubs) must be safe to offer."""
    reap_pid(None)
    reap_pid(0)
    reap_pid(-1)
    reap_pid(psutil.Process().ppid())  # live but not our descendant
    reap_pid(2**24)  # implausibly large; NoSuchProcess → no-op


def test_pid_collecting_wrapper_records_each_returned_pid() -> None:
    pids: list[int] = []
    wrapper = _pid_collecting_wrapper(lambda *a, **k: SimpleNamespace(pid=4242), pids)
    wrapper()
    wrapper()
    assert pids == [4242, 4242]


def test_wrap_launchers_rebinds_real_launcher_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every sys.modules binding of a real launcher gets the pid collector;
    bindings of anything else (a test's own fake) are left alone."""
    captured = SimpleNamespace(pid=7777)
    real = lambda *a, **k: captured  # noqa: E731
    fake_module = types.ModuleType("_process_guard_fake_ns")
    fake_module.launch_claude_worker = real
    fake_module.launch_api_worker = lambda *a, **k: None  # not the real function
    monkeypatch.setitem(sys.modules, "_process_guard_fake_ns", fake_module)

    pids: list[int] = []
    wrapped = wrap_launchers(monkeypatch, pids, real_launchers={"launch_claude_worker": real})
    assert wrapped == 1
    assert fake_module.launch_claude_worker("ignored") is captured
    assert pids == [7777]
    assert fake_module.launch_api_worker() is None  # untouched


def test_enter_kill_on_close_job_skips_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "name", "posix")
    assert enter_kill_on_close_job() is False


def test_enter_kill_on_close_job_tolerates_reentry() -> None:
    """A second assignment attempt must be tolerated, never raised.

    On Windows the session fixture has already assigned this process to a
    job; assigning it to a fresh root job is refused (ERROR_ACCESS_DENIED) —
    exactly the "already in a job that forbids nesting" path the issue calls
    out. Off Windows the call is a plain no-op. Either way: bool, no raise.
    """
    assert isinstance(enter_kill_on_close_job(), bool)
