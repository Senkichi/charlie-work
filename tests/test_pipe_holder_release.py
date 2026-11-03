"""``pipe_holder_release.release_pipe_holders`` against real processes (issue #2794).

A stalled opencode worker is blocked on a tool call whose output pipe an orphaned
process still holds. The primitive kills exactly those orphans -- selected by a
cwd inside the worker's worktree, a start after the worker's own, and no
parentage under the worker -- and must touch nothing when it cannot anchor on the
worker. Each row spawns real interpreters: the selection rule *is* the behaviour,
so a fake process table would only restate it.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest
from charlie_work.pipe_holder_release import release_pipe_holders
from charlie_work.process_utils import get_process_start_time

# Longer than the primitive's 1s start-time tolerance, so "started before the
# worker" is unambiguous.
_BEFORE_WORKER_GAP_SECONDS = 1.3
_SLEEPER = "import time; print('r', flush=True); time.sleep(120)"
# The interpreter the children run under. A venv's ``python.exe`` on Windows is a
# launcher that spawns the real interpreter, so its ``Popen.pid`` is a process
# whose cwd psutil misreports; the base interpreter is the process itself, the
# shape of the real worker (a native ``opencode.exe`` the launcher resolves to).
_PYTHON = getattr(sys, "_base_executable", sys.executable)


@pytest.fixture
def spawned() -> Iterator[list[subprocess.Popen[str]]]:
    procs: list[subprocess.Popen[str]] = []
    yield procs
    for proc in procs:
        _reap(proc)


def _reap(proc: subprocess.Popen[str]) -> None:
    for live in _tree(proc.pid):
        try:
            live.kill()
        except psutil.Error:
            pass
    proc.wait(timeout=10)
    if proc.stdout is not None:
        proc.stdout.close()


def _tree(pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except psutil.Error:
        return []


def _spawn(
    spawned: list[subprocess.Popen[str]], cwd: Path, code: str = _SLEEPER
) -> subprocess.Popen[str]:
    """Start ``python -c code`` in ``cwd`` and wait until it has announced itself."""
    proc = subprocess.Popen(
        [_PYTHON, "-c", code],
        cwd=cwd,
        stdout=subprocess.PIPE,
        text=True,
    )
    spawned.append(proc)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "r"
    return proc


def _start_time(proc: subprocess.Popen[str]) -> float:
    started = get_process_start_time(proc.pid)
    assert started is not None
    return started


def _alive(proc: subprocess.Popen[str]) -> bool:
    return proc.poll() is None


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    path = tmp_path / "wt"
    path.mkdir()
    return path.resolve()


def test_a_holder_started_after_the_worker_is_killed_and_the_worker_is_spared(
    worktree: Path, spawned: list[subprocess.Popen[str]]
) -> None:
    worker = _spawn(spawned, worktree)
    holder = _spawn(spawned, worktree)

    released = release_pipe_holders(worktree, worker.pid, _start_time(worker))

    assert holder.pid in released
    assert worker.pid not in released
    assert holder.wait(timeout=10) is not None
    assert _alive(worker)


def test_an_orphan_with_a_dead_parent_is_found_though_no_process_tree_reaches_it(
    worktree: Path, spawned: list[subprocess.Popen[str]]
) -> None:
    """The real shape: the tool's shell backgrounds a child and exits (#2794).

    Windows never reparents, so the holder's ppid names a dead process; ancestry
    cannot find it, the shared cwd can.
    """
    worker = _spawn(spawned, worktree)
    shell = subprocess.run(
        [
            _PYTHON,
            "-c",
            "import subprocess, sys; "
            f"p = subprocess.Popen([{_PYTHON!r}, '-c', {_SLEEPER!r}], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            "print(p.pid, flush=True)",
        ],
        cwd=worktree,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    holder_pid = int(shell.stdout.strip())
    deadline = time.monotonic() + 10
    while not psutil.pid_exists(holder_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert psutil.pid_exists(holder_pid), "positive control: the orphan must be running"
    assert holder_pid not in {p.pid for p in _tree(worker.pid)}, "not a descendant of the worker"

    released = release_pipe_holders(worktree, worker.pid, _start_time(worker))

    assert holder_pid in released
    assert _alive(worker)


def test_a_process_that_started_before_the_worker_is_spared(
    worktree: Path, spawned: list[subprocess.Popen[str]]
) -> None:
    """An operator shell opened in the worktree before the dispatch is not a holder."""
    earlier = _spawn(spawned, worktree)
    time.sleep(_BEFORE_WORKER_GAP_SECONDS)
    worker = _spawn(spawned, worktree)
    holder = _spawn(spawned, worktree)

    released = release_pipe_holders(worktree, worker.pid, _start_time(worker))

    assert holder.pid in released
    assert earlier.pid not in released
    assert _alive(earlier)


def test_a_live_descendant_of_the_worker_is_spared(
    worktree: Path, spawned: list[subprocess.Popen[str]]
) -> None:
    """The worker's own helpers (LSP/MCP servers) share its cwd and start after it."""
    worker = _spawn(
        spawned,
        worktree,
        "import subprocess, sys, time\n"
        f"c = subprocess.Popen([{_PYTHON!r}, '-c', {_SLEEPER!r}], stdout=subprocess.PIPE, text=True)\n"
        "c.stdout.readline()\n"
        "print('r', flush=True)\n"
        "time.sleep(120)\n",
    )
    helpers = [p for p in _tree(worker.pid) if p.pid != worker.pid]
    assert helpers, "positive control: the worker must have a live descendant"

    released = release_pipe_holders(worktree, worker.pid, _start_time(worker))

    assert not set(released) & {p.pid for p in helpers}
    assert all(p.is_running() for p in helpers)
    assert _alive(worker)


def test_a_process_whose_cwd_is_outside_the_worktree_is_spared(
    worktree: Path, tmp_path: Path, spawned: list[subprocess.Popen[str]]
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    worker = _spawn(spawned, worktree)
    bystander = _spawn(spawned, elsewhere)
    holder = _spawn(spawned, worktree)

    released = release_pipe_holders(worktree, worker.pid, _start_time(worker))

    assert holder.pid in released
    assert bystander.pid not in released
    assert _alive(bystander)


def test_no_holder_means_nothing_is_killed_and_an_empty_result(
    worktree: Path, spawned: list[subprocess.Popen[str]]
) -> None:
    worker = _spawn(spawned, worktree)

    assert release_pipe_holders(worktree, worker.pid, _start_time(worker)) == ()
    assert _alive(worker)


class TestRefusesWithoutAnAnchor:
    """Each row leaves a would-be holder untouched: the guard, not luck, saves it."""

    def test_a_missing_pid_or_start_time(
        self, worktree: Path, spawned: list[subprocess.Popen[str]]
    ) -> None:
        worker = _spawn(spawned, worktree)
        holder = _spawn(spawned, worktree)
        start = _start_time(worker)

        assert release_pipe_holders(worktree, None, start) == ()
        assert release_pipe_holders(worktree, 0, start) == ()
        assert release_pipe_holders(worktree, worker.pid, None) == ()
        assert _alive(holder)

    def test_a_start_time_that_names_another_process_than_the_pid(
        self, worktree: Path, spawned: list[subprocess.Popen[str]]
    ) -> None:
        """A recycled pid: same number, different process."""
        worker = _spawn(spawned, worktree)
        holder = _spawn(spawned, worktree)

        assert release_pipe_holders(worktree, worker.pid, _start_time(worker) - 3600) == ()
        assert _alive(holder)

    def test_a_worker_that_is_gone(
        self, worktree: Path, spawned: list[subprocess.Popen[str]]
    ) -> None:
        worker = _spawn(spawned, worktree)
        started = _start_time(worker)
        holder = _spawn(spawned, worktree)
        _reap(worker)

        assert release_pipe_holders(worktree, worker.pid, started) == ()
        assert _alive(holder)

    def test_a_worker_whose_own_cwd_is_outside_the_worktree(
        self, worktree: Path, tmp_path: Path, spawned: list[subprocess.Popen[str]]
    ) -> None:
        """The anchor assertion: a wrong ``worktree_path`` must not select processes."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        worker = _spawn(spawned, elsewhere)
        holder = _spawn(spawned, worktree)

        assert release_pipe_holders(worktree, worker.pid, _start_time(worker)) == ()
        assert _alive(holder)

    @pytest.mark.parametrize("bad", [Path(""), Path("."), Path("wt")], ids=str)
    def test_an_empty_or_relative_worktree_path(
        self,
        bad: Path,
        worktree: Path,
        monkeypatch: pytest.MonkeyPatch,
        spawned: list[subprocess.Popen[str]],
    ) -> None:
        """``Path("")`` is this process's cwd; run from the worktree it would match."""
        worker = _spawn(spawned, worktree)
        holder = _spawn(spawned, worktree)
        monkeypatch.chdir(worktree.parent)

        assert release_pipe_holders(bad, worker.pid, _start_time(worker)) == ()
        assert _alive(holder)

    def test_a_worktree_path_that_is_not_a_directory(
        self, worktree: Path, spawned: list[subprocess.Popen[str]]
    ) -> None:
        worker = _spawn(spawned, worktree)
        holder = _spawn(spawned, worktree)

        assert release_pipe_holders(worktree / "missing", worker.pid, _start_time(worker)) == ()
        assert _alive(holder)
