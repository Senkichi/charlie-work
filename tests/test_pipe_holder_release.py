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

import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import psutil
import pytest
from _pipe_holder_fixtures import _PYTHON, _SLEEPER, _spawn_orphan
from charlie_work import pipe_holder_release
from charlie_work.pipe_holder_release import _fresh_processes, _is_orphan, release_pipe_holders
from charlie_work.process_utils import get_process_start_time

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="orphans are only parentless on Windows; POSIX reparents them to a live init",
)

# Longer than the primitive's 1s start-time tolerance, so "started before the
# worker" is unambiguous.
_BEFORE_WORKER_GAP_SECONDS = 1.3
# A live look-alike that is nobody's sleeper: same shape, different script, so
# the identity check must reject a printed pid naming it -- and never touch
# the process behind that pid.
_STRANGER = "import time; print('f', flush=True); time.sleep(120)"


@pytest.fixture
def spawned() -> Iterator[list[subprocess.Popen[str]]]:
    procs: list[subprocess.Popen[str]] = []
    yield procs
    for proc in procs:
        _reap(proc)


# Same ``orphans`` lifecycle guard the settle rows of
# tests/test_pipe_holder_spawn_parent_settle.py declare: at import time the
# fixture name would collide with every test's ``orphans`` parameter (ruff's
# redefinition rules), so each suite keeps this one-line teardown local rather
# than importing it from _pipe_holder_fixtures.


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
    spawned: list[subprocess.Popen[str]],
    cwd: Path,
    code: str = _SLEEPER,
    *,
    announce: str = "r",
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
    assert proc.stdout.readline().strip() == announce
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


@pytest.fixture
def stale_process_cache() -> Iterator[Callable[[int, float], psutil.Process]]:
    """Plant, in psutil's own ``process_iter`` cache, a ``Process`` that remembers an older owner.

    ``process_iter`` caches one ``Process`` per pid and evicts a pid only when it
    is absent from a sweep. Windows recycles pids, so a pid an earlier sweep cached
    can be reborn as a new process before the next sweep; the cached instance then
    still carries the previous owner's start time. Which earlier sweep populated the
    cache is test order, so the failure appears only in a full, shuffled run.

    The cache is the process-global one, so teardown empties it (public
    ``cache_clear``): a planted entry must not outlive the pid it names.
    """

    def plant(pid: int, created: float) -> psutil.Process:
        stale = psutil.Process(pid)
        # psutil caches a Process's identity at construction and exposes no way to
        # age one; these two attributes are what a previous owner's instance holds.
        stale._create_time = created  # type: ignore[attr-defined]
        stale._ident = (pid, created)  # type: ignore[attr-defined]
        psutil._pmap[pid] = stale  # type: ignore[attr-defined]
        # Positive controls: the aging took, and the cache really serves the stale
        # instance -- so a pass cannot mean the plant silently missed (a psutil
        # upgrade that renames these internals fails here, not by passing).
        assert stale.create_time() == created, "positive control: the instance must be aged"
        assert any(p is stale for p in psutil.process_iter()), (
            "positive control: process_iter must serve the planted instance"
        )
        return stale

    yield plant
    psutil.process_iter.cache_clear()


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


def test_a_stale_cached_process_for_the_holders_pid_does_not_hide_the_holder(
    worktree: Path,
    spawned: list[subprocess.Popen[str]],
    orphans: list[psutil.Process],
    stale_process_cache: Callable[[int, float], psutil.Process],
) -> None:
    """The holder's pid is one ``process_iter`` already cached for an older process (#3028).

    Selecting from the cached instance reads the previous owner's start time, so a
    holder that started *after* the worker looks like it started before it and is
    spared: the call stays blocked and the lane falls through to its ordinary reap.
    The holder's own start time, not the cache's, decides.
    """
    worker = _spawn(spawned, worktree)
    holder = _spawn_orphan(orphans, worktree)
    worker_started = _start_time(worker)
    stale = stale_process_cache(holder.pid, worker_started - 3600)
    assert stale.pid == holder.pid
    assert holder.create_time() > worker_started, "positive control: it did start after the worker"

    released = release_pipe_holders(worktree, worker.pid, worker_started)

    assert released == (holder.pid,)
    assert not holder.is_running()
    assert _alive(worker)


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


def _exited_interpreters_pid() -> int:
    """A pid whose owner already exited: the recycling race, staged on demand.

    Freed under no handle holders, exactly like the real sleeper's pid after the
    launcher that printed it exits. Whether the pid is still free (psutil answers
    ``NoSuchProcess``) or already reborn as an unrelated process (a foreign
    cmdline), the identity check rejects it either way.
    """
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)
    return child.pid


def test_a_printed_pid_that_names_a_live_stranger_is_rejected_untouched(
    monkeypatch: pytest.MonkeyPatch,
    worktree: Path,
    spawned: list[subprocess.Popen[str]],
    orphans: list[psutil.Process],
) -> None:
    """A live pid that is not the sleeper is rejected on its cmdline, untouched.

    Staged: the first launch attempt's launcher prints the pid of a live
    process that is not the sleeper (its cmdline is a different script), so
    ``psutil.Process(pid)`` succeeds and is_running says yes. The identity
    check must reject it as the attempt's evidence, never report it as the
    orphan, retry the spawn for real, and leave the stranger running -- it was
    never ours, whatever it shares the worktree with.
    """
    stranger = _spawn(spawned, worktree, code=_STRANGER, announce="f")
    real_run = subprocess.run
    state = {"stranger": True}

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if state["stranger"]:
            state["stranger"] = False
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{stranger.pid}\n", stderr="")
        return real_run(cmd, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "run", run)

    holder = _spawn_orphan(orphans, worktree)

    assert state == {"stranger": False}, "the retry must have re-run the real launcher"
    assert [proc.pid for proc in orphans] == [holder.pid]
    assert any(_SLEEPER in arg for arg in holder.cmdline()), "the retry must be ours"
    assert holder.pid != stranger.pid
    assert _alive(stranger), "the stranger was never ours: it must survive the rejection"


def test_a_printed_pid_that_names_a_stray_sleeper_is_killed_and_the_spawn_retried(
    monkeypatch: pytest.MonkeyPatch,
    worktree: Path,
    spawned: list[subprocess.Popen[str]],
    orphans: list[psutil.Process],
) -> None:
    """A printed pid can name a sleeper that is not an orphan: kill it, retry.

    Staged: the first launch attempt's launcher prints the pid of a sleeper
    this test process itself parented. The cmdline check accepts it (it really
    is the sleeper code), but psutil's ``parent()`` answers a live process
    even after the poll -- the helper must record that as the attempt's
    evidence, kill the stray (a verified sleeper is pinned to its (pid, create
    time) either way), retry the spawn, and record only the verified orphan.
    """
    stray = _spawn(spawned, worktree)
    assert psutil.Process(stray.pid).parent() is not None, "positive control: a live parent"
    real_run = subprocess.run
    state = {"stray": True}

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if state["stray"]:
            state["stray"] = False
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{stray.pid}\n", stderr="")
        return real_run(cmd, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "run", run)

    holder = _spawn_orphan(orphans, worktree)

    assert state == {"stray": False}, "the retry must have re-run the real launcher"
    assert [proc.pid for proc in orphans] == [holder.pid]
    assert any(_SLEEPER in arg for arg in holder.cmdline()), "the retry must be ours"
    assert not _alive(stray), "the mis-parented stray was ours and got killed"


def test_spawn_orphan_never_touches_a_foreign_pid_and_fails_loudly_when_raced_out(
    monkeypatch: pytest.MonkeyPatch, worktree: Path
) -> None:
    """A printed pid whose owner already exited is dropped, never reported.

    Every attempt is handed a pid that no longer names the sleeper: gone, or
    reborn as someone else. The helper must (a) refuse to trust that pid
    whatever now holds it -- it was never ours -- and (b) raise with the last
    attempt's evidence rather than an assert against a phantom, so the failure
    names the race instead of blaming the guard under test.
    """
    foreign_pid = _exited_interpreters_pid()
    no_owners: list[psutil.Process] = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{foreign_pid}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(AssertionError, match="raced out from under the test"):
        _spawn_orphan(no_owners, worktree)

    assert no_owners == [], "a foreign pid must not be recorded for teardown"


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


_UNREADABLE_PID = 4_000_001  # patched ``psutil.pids`` names it; no real process is asked about it


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(psutil.NoSuchProcess(_UNREADABLE_PID), id="vanished-mid-sweep"),
        pytest.param(psutil.AccessDenied(_UNREADABLE_PID), id="identity-unreadable"),
        pytest.param(OSError("handle could not be opened"), id="os-error"),
    ],
)
def test_a_pid_that_cannot_be_read_is_skipped_and_the_sweep_goes_on(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pid that vanishes or cannot be opened mid-sweep must not end the sweep (#3028).

    ``_fresh_processes`` replaced ``process_iter``'s own handling of this with a
    ``try``/``except``. Nothing else exercises it: a real sweep rarely meets an unreadable
    pid, and an exception escaping the generator is swallowed by ``release_pipe_holders``'
    outer ``except`` -- the lane would silently become a no-op. The unreadable pid comes
    *first*, so a ``return`` or an uncaught raise in its place would lose the live pid.
    """
    live_pid = os.getpid()
    real_process = psutil.Process

    def process(pid: int) -> psutil.Process:
        if pid == _UNREADABLE_PID:
            raise error
        return real_process(pid)

    monkeypatch.setattr(pipe_holder_release.psutil, "pids", lambda: [_UNREADABLE_PID, live_pid])
    monkeypatch.setattr(pipe_holder_release.psutil, "Process", process)

    assert [proc.pid for proc in _fresh_processes()] == [live_pid]


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
