"""``pipe_holder_release.release_pipe_holders`` against real processes (issue #2794).

A stalled opencode worker is blocked on a tool call whose output pipe an orphaned
process still holds. The primitive kills exactly those orphans -- selected by a
cwd inside the worker's worktree, a start after the worker's own, and a parent
that is gone -- and must touch nothing when it cannot anchor on the worker. Each
row spawns real interpreters: the selection rule *is* the behaviour, so a fake
process table would only restate it. (The one branch a real process cannot
produce on demand -- a recycled parent pid -- is a unit row on ``_is_orphan``.)

Every would-be holder in a refusal row is a *real orphan*. A holder with a live
parent would be spared by the orphan rule whatever the guard under test did, so
the row would pass with the guard deleted -- the guard, not luck, must save it.
Windows never reparents, so an orphan keeps its dead parent's pid; on POSIX the
kernel hands it to init, which is a live parent and so is (by design) not a
holder. The suite therefore runs on Windows, the fleet's host and CI's OS.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import psutil
import pytest
from charlie_work import pipe_holder_release
from charlie_work.pipe_holder_release import _is_orphan, release_pipe_holders
from charlie_work.process_utils import get_process_start_time

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="orphans are only parentless on Windows; POSIX reparents them to a live init",
)

# Longer than the primitive's 1s start-time tolerance, so "started before the
# worker" is unambiguous.
_BEFORE_WORKER_GAP_SECONDS = 1.3
_SLEEPER = "import time; print('r', flush=True); time.sleep(120)"
# The interpreter the children run under. A venv's ``python.exe`` on Windows is a
# launcher that spawns the real interpreter, so its ``Popen.pid`` is a process
# whose cwd psutil misreports; the base interpreter is the process itself, the
# shape of the real worker (a native ``opencode.exe`` the launcher resolves to).
_PYTHON = getattr(sys, "_base_executable", sys.executable)
# The real shape of the bug: a tool's shell backgrounds a child and exits.
_ORPHAN_LAUNCHER = (
    "import subprocess, sys; "
    f"p = subprocess.Popen([{_PYTHON!r}, '-c', {_SLEEPER!r}], "
    "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
    "print(p.pid, flush=True)"
)


@pytest.fixture
def spawned() -> Iterator[list[subprocess.Popen[str]]]:
    procs: list[subprocess.Popen[str]] = []
    yield procs
    for proc in procs:
        _reap(proc)


@pytest.fixture
def orphans() -> Iterator[list[psutil.Process]]:
    """Orphans the test made; killed on teardown (pinned to pid + create time)."""
    procs: list[psutil.Process] = []
    yield procs
    for proc in procs:
        try:
            proc.kill()
        except psutil.Error:
            pass


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


def _spawn_orphan(orphans: list[psutil.Process], cwd: Path) -> psutil.Process:
    """Start a sleeper in ``cwd`` whose parent has already exited -- a real orphan."""
    launched = subprocess.run(
        [_PYTHON, "-c", _ORPHAN_LAUNCHER],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    orphan = psutil.Process(int(launched.stdout.strip()))
    orphans.append(orphan)
    deadline = time.monotonic() + 10
    while not orphan.is_running() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert orphan.is_running(), "positive control: the orphan must be running"
    # psutil's own parent() rule (missing, or a recycled pid) as an independent
    # control that the process really is parentless in the table.
    assert orphan.parent() is None, "positive control: the orphan's parent must be gone"
    return orphan


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


def test_an_orphan_started_after_the_worker_is_killed_and_the_worker_is_spared(
    worktree: Path, spawned: list[subprocess.Popen[str]], orphans: list[psutil.Process]
) -> None:
    """The real shape: the tool's shell backgrounds a child and exits (#2794).

    Windows never reparents, so the holder's ppid names a dead process; ancestry
    cannot find it, the shared cwd can.
    """
    worker = _spawn(spawned, worktree)
    holder = _spawn_orphan(orphans, worktree)
    assert holder.pid not in {p.pid for p in _tree(worker.pid)}, "not a descendant of the worker"

    released = release_pipe_holders(worktree, worker.pid, _start_time(worker))

    assert released == (holder.pid,)
    assert not holder.is_running()
    assert _alive(worker)


def test_a_live_parent_process_started_after_the_worker_is_spared(
    worktree: Path, spawned: list[subprocess.Popen[str]], orphans: list[psutil.Process]
) -> None:
    """A process with a live parent is legitimate even with a cwd in the worktree.

    An operator shell opened in the worktree *after* the dispatch, or a tool
    another live process started there, shares the cwd and starts after the worker.
    Only its live parent distinguishes it from the real holder next to it.
    """
    worker = _spawn(spawned, worktree)
    bystander = _spawn(spawned, worktree)  # parent: this (live, older) test process
    holder = _spawn_orphan(orphans, worktree)
    assert psutil.Process(bystander.pid).parent() is not None, "positive control: a live parent"

    released = release_pipe_holders(worktree, worker.pid, _start_time(worker))

    assert released == (holder.pid,)
    assert bystander.pid not in released
    assert _alive(bystander)
    assert _alive(worker)


def test_with_only_live_parent_processes_nothing_is_killed_and_the_result_is_empty(
    worktree: Path, spawned: list[subprocess.Popen[str]]
) -> None:
    """A miss is an empty result, which the stalled lane turns into the ordinary reap."""
    worker = _spawn(spawned, worktree)
    bystander = _spawn(spawned, worktree)

    assert release_pipe_holders(worktree, worker.pid, _start_time(worker)) == ()
    assert _alive(bystander)
    assert _alive(worker)


def test_a_process_that_started_before_the_worker_is_spared(
    worktree: Path, spawned: list[subprocess.Popen[str]], orphans: list[psutil.Process]
) -> None:
    """An orphan already in the worktree before the dispatch is not this call's holder."""
    earlier = _spawn_orphan(orphans, worktree)
    time.sleep(_BEFORE_WORKER_GAP_SECONDS)
    worker = _spawn(spawned, worktree)
    holder = _spawn_orphan(orphans, worktree)

    released = release_pipe_holders(worktree, worker.pid, _start_time(worker))

    assert released == (holder.pid,)
    assert earlier.is_running()


def test_a_live_descendant_of_the_worker_is_spared(
    worktree: Path, spawned: list[subprocess.Popen[str]], orphans: list[psutil.Process]
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
    holder = _spawn_orphan(orphans, worktree)

    released = release_pipe_holders(worktree, worker.pid, _start_time(worker))

    assert released == (holder.pid,)
    assert all(p.is_running() for p in helpers)
    assert _alive(worker)


def test_an_orphan_whose_cwd_is_outside_the_worktree_is_spared(
    worktree: Path,
    tmp_path: Path,
    spawned: list[subprocess.Popen[str]],
    orphans: list[psutil.Process],
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    worker = _spawn(spawned, worktree)
    bystander = _spawn_orphan(orphans, elsewhere)
    holder = _spawn_orphan(orphans, worktree)

    released = release_pipe_holders(worktree, worker.pid, _start_time(worker))

    assert released == (holder.pid,)
    assert bystander.is_running()


def test_no_holder_means_nothing_is_killed_and_an_empty_result(
    worktree: Path, spawned: list[subprocess.Popen[str]]
) -> None:
    worker = _spawn(spawned, worktree)

    assert release_pipe_holders(worktree, worker.pid, _start_time(worker)) == ()
    assert _alive(worker)


class _FakeProc:
    """The two reads ``_is_orphan`` makes of a process: its ppid and its create time."""

    def __init__(self, ppid: int, created: float) -> None:
        self._ppid = ppid
        self._created = created

    def ppid(self) -> int:
        return self._ppid

    def create_time(self) -> float:
        return self._created


_CHILD_CREATED = 1001.0


def _parent_missing(pid: int) -> _FakeProc:
    raise psutil.NoSuchProcess(pid)


def _parent_unreadable(pid: int) -> _FakeProc:
    raise psutil.AccessDenied(pid)


def _parent_created_at(created: float) -> Callable[[int], _FakeProc]:
    return lambda pid: _FakeProc(ppid=0, created=created)


@pytest.mark.parametrize(
    ("lookup", "expected"),
    [
        pytest.param(_parent_missing, True, id="parent-missing"),
        pytest.param(_parent_created_at(_CHILD_CREATED - 1), False, id="parent-older-so-live"),
        pytest.param(
            _parent_created_at(_CHILD_CREATED + 5), True, id="parent-younger-so-recycled-pid"
        ),
        pytest.param(_parent_unreadable, False, id="parent-unreadable-fails-safe"),
    ],
)
def test_orphan_rule_on_the_parent_link(
    lookup: Callable[[int], _FakeProc], expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recycled-pid branch cannot be staged with a real process; pin it here."""
    monkeypatch.setattr(pipe_holder_release.psutil, "Process", lookup)

    assert _is_orphan(_FakeProc(ppid=7, created=_CHILD_CREATED)) is expected  # type: ignore[arg-type]


@pytest.mark.parametrize("ppid", [0, -1])
def test_a_process_with_no_recorded_parent_is_not_an_orphan(ppid: int) -> None:
    """No parent on record is not the same as a dead one: spare it."""
    assert _is_orphan(_FakeProc(ppid=ppid, created=1.0)) is False  # type: ignore[arg-type]


class TestRefusesWithoutAnAnchor:
    """Each row leaves a would-be holder -- a real orphan -- untouched: the guard saves it."""

    def test_a_missing_pid_or_start_time(
        self,
        worktree: Path,
        spawned: list[subprocess.Popen[str]],
        orphans: list[psutil.Process],
    ) -> None:
        worker = _spawn(spawned, worktree)
        holder = _spawn_orphan(orphans, worktree)
        start = _start_time(worker)

        assert release_pipe_holders(worktree, None, start) == ()
        assert release_pipe_holders(worktree, 0, start) == ()
        assert release_pipe_holders(worktree, worker.pid, None) == ()
        assert holder.is_running()

    def test_a_start_time_that_names_another_process_than_the_pid(
        self,
        worktree: Path,
        spawned: list[subprocess.Popen[str]],
        orphans: list[psutil.Process],
    ) -> None:
        """A recycled pid: same number, different process."""
        worker = _spawn(spawned, worktree)
        holder = _spawn_orphan(orphans, worktree)

        assert release_pipe_holders(worktree, worker.pid, _start_time(worker) - 3600) == ()
        assert holder.is_running()

    def test_a_worker_that_is_gone(
        self,
        worktree: Path,
        spawned: list[subprocess.Popen[str]],
        orphans: list[psutil.Process],
    ) -> None:
        worker = _spawn(spawned, worktree)
        started = _start_time(worker)
        holder = _spawn_orphan(orphans, worktree)
        _reap(worker)

        assert release_pipe_holders(worktree, worker.pid, started) == ()
        assert holder.is_running()

    def test_a_worker_whose_own_cwd_is_outside_the_worktree(
        self,
        worktree: Path,
        tmp_path: Path,
        spawned: list[subprocess.Popen[str]],
        orphans: list[psutil.Process],
    ) -> None:
        """The anchor assertion: a wrong ``worktree_path`` must not select processes."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        worker = _spawn(spawned, elsewhere)
        holder = _spawn_orphan(orphans, worktree)

        assert release_pipe_holders(worktree, worker.pid, _start_time(worker)) == ()
        assert holder.is_running()

    @pytest.mark.parametrize("bad", [Path(""), Path("."), Path("wt")], ids=str)
    def test_an_empty_or_relative_worktree_path(
        self,
        bad: Path,
        worktree: Path,
        monkeypatch: pytest.MonkeyPatch,
        spawned: list[subprocess.Popen[str]],
        orphans: list[psutil.Process],
    ) -> None:
        """``Path("")`` is this process's cwd; run from the worktree it would match."""
        worker = _spawn(spawned, worktree)
        holder = _spawn_orphan(orphans, worktree)
        monkeypatch.chdir(worktree.parent)

        assert release_pipe_holders(bad, worker.pid, _start_time(worker)) == ()
        assert holder.is_running()

    def test_a_worktree_path_that_is_not_a_directory(
        self,
        worktree: Path,
        spawned: list[subprocess.Popen[str]],
        orphans: list[psutil.Process],
    ) -> None:
        worker = _spawn(spawned, worktree)
        holder = _spawn_orphan(orphans, worktree)

        assert release_pipe_holders(worktree / "missing", worker.pid, _start_time(worker)) == ()
        assert holder.is_running()
