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
import time
import types
from pathlib import Path
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

_BASE_EXECUTABLE = getattr(sys, "_base_executable", sys.executable)


def _spawn_sleeper(seconds: int = 60, *, direct: bool = False) -> subprocess.Popen:
    # ``sys.executable`` in a venv is a launcher that spawns the real
    # interpreter as a *grandchild* a beat later — the exact pair shape the
    # issue describes. ``direct=True`` uses the base interpreter instead, so
    # tests asserting on a single, fully-materialized child get one.
    executable = _BASE_EXECUTABLE if direct else sys.executable
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


def _wait_dead(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if not psutil.Process(pid).is_running():
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


def _spawn_orphan_sleeper(tmp_path: Path, seconds: int = 120) -> int:
    """Launch a sleeper through a helper that exits immediately.

    The sleeper ends up parented to a dead pid — alive and killable, but not
    a descendant of pytest — which is exactly the "foreign pid" shape a
    fabricated launch record could carry.
    """
    script = tmp_path / "orphan_helper.py"
    script.write_text(
        "import subprocess, sys\n"
        "sleeper = subprocess.Popen(\n"
        f"    [sys.argv[1], '-c', 'import time; time.sleep({seconds})'],\n"
        "    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
        ")\n"
        "print(sleeper.pid, flush=True)\n",
        encoding="utf-8",
    )
    # Base interpreter for both processes: single-process, fully materialized,
    # so the orphan has no venv-launcher grandchild to escape cleanup.
    helper = subprocess.Popen(
        [_BASE_EXECUTABLE, str(script), _BASE_EXECUTABLE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = helper.communicate(timeout=30)
        assert helper.returncode == 0, f"orphan helper failed: {err!r}"
        return int(out.strip())
    finally:
        if helper.poll() is None:
            helper.kill()
            helper.wait(timeout=10)


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


def test_reap_pid_never_touches_foreign_or_missing_pids(tmp_path: Path) -> None:
    """Fabricated record pids (delegation-test stubs) must be safe to offer:
    the descendant check is the only thing standing between ``reap_pid`` and
    an unrelated live process, so the assertions — not just the calls — are
    the test.
    """
    parent_pid = os.getppid()
    orphan_pid = _spawn_orphan_sleeper(tmp_path)
    try:
        ours = {p.pid for p in psutil.Process().children(recursive=True)}
        assert parent_pid not in ours, "test premise broken: pytest's parent is our descendant"
        assert orphan_pid not in ours, "test premise broken: orphan is still our descendant"

        reap_pid(None)
        reap_pid(0)
        reap_pid(-1)
        reap_pid(2**24)  # implausibly large; NoSuchProcess → no-op

        # A live, killable process that is NOT our descendant must survive.
        # Checked first: under a deleted descendant-check mutation this fails
        # here, before the parent's pid is ever offered.
        reap_pid(orphan_pid)
        try:
            orphan_alive = psutil.Process(orphan_pid).is_running()
        except psutil.NoSuchProcess:
            orphan_alive = False
        assert orphan_alive, "reap_pid killed a process that is not a descendant of pytest"

        reap_pid(parent_pid)
        assert psutil.pid_exists(parent_pid), "reap_pid killed pytest's parent"
    finally:
        if psutil.pid_exists(orphan_pid):
            psutil.Process(orphan_pid).kill()
            psutil.Process(orphan_pid).wait(timeout=10)


@pytest.mark.skipif(os.name != "nt", reason="Job Objects exist only on Windows")
def test_kill_on_close_job_kills_descendants_when_handle_closes(tmp_path: Path) -> None:
    """The structural fix's core promise, observed end to end: closing the
    last handle to a kill-on-close Job Object terminates every member —
    including the process that owned the handle — at any depth.

    A helper child enters a fresh kill-on-close job via the production
    ``enter_kill_on_close_job`` (a nested job, permitted since Windows 8),
    spawns a sleeper through the venv launcher — so a real interpreter
    grandchild materializes a beat later, the exact pair shape the issue
    describes — reports both pids, then closes its job handle. That is the
    same trigger as the owner process exiting (the kernel closes all handles
    at exit), chosen deliberately: in this environment an owner process's
    exit already kills its descendants through an independent outer
    containment, which would mask a missing flag, while an explicit
    ``CloseHandle`` isolates ``KILL_ON_JOB_CLOSE`` as the only thing that can
    have fired. A host that refuses the nested assignment reports
    JOB_UNAVAILABLE and the test skips, matching the tolerated no-op path in
    ``enter_kill_on_close_job``.
    """
    # The helper prints SLEEPER=/LEAF=, then closes the job handle. If the
    # flag is set, the close kills the helper itself mid-call — STILL_ALIVE
    # is never printed and the helper exits abnormally. If the flag were
    # absent, the close would merely release the members and STILL_ALIVE
    # would appear.
    script = tmp_path / "job_helper.py"
    script.write_text(
        "import ctypes, subprocess, sys, time\n"
        "from ctypes import wintypes\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import _process_guard\n"
        "if not _process_guard.enter_kill_on_close_job():\n"
        "    print('JOB_UNAVAILABLE', flush=True)\n"
        "    raise SystemExit(2)\n"
        "sleeper = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(300)'],\n"
        "    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
        ")\n"
        "time.sleep(2)  # let the venv launcher's real interpreter materialize\n"
        "import psutil\n"
        "kids = psutil.Process(sleeper.pid).children(recursive=True)\n"
        "print(f'SLEEPER={sleeper.pid}', flush=True)\n"
        "print(f'LEAF={(kids[-1].pid if kids else sleeper.pid)}', flush=True)\n"
        "kernel32 = ctypes.windll.kernel32\n"
        "kernel32.CloseHandle.argtypes = [wintypes.HANDLE]\n"
        "kernel32.CloseHandle(wintypes.HANDLE(_process_guard._job_handle))\n"
        "print('STILL_ALIVE', flush=True)\n",
        encoding="utf-8",
    )
    helper = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pids: list[int] = []
    try:
        out, err = helper.communicate(timeout=30)
        if "JOB_UNAVAILABLE" in out:
            pytest.skip("host refused a nested Job Object for a process already inside one")
        pids = [
            int(line.partition("=")[2])
            for line in out.splitlines()
            if line.startswith(("SLEEPER=", "LEAF="))
        ]
        assert len(pids) == 2, (
            f"helper did not report both descendant pids: stdout={out!r} stderr={err!r}"
        )
        assert "STILL_ALIVE" not in out, (
            "helper survived closing its kill-on-close job — "
            "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE did not fire"
        )
        for pid in pids:
            assert _wait_dead(pid), f"descendant pid={pid} survived the job's last handle closing"
    finally:
        if helper.poll() is None:
            helper.kill()
            helper.wait(timeout=10)
        for pid in pids:
            if psutil.pid_exists(pid):
                try:
                    psutil.Process(pid).kill()
                    psutil.Process(pid).wait(timeout=10)
                except psutil.NoSuchProcess:
                    pass


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
