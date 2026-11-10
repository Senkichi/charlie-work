"""Shared real-process spawn helpers for the pipe-holder suites.

The zero-cross-test-import guard (``tests/test_zero_cross_test_import_guard.py``,
issue #1284) bans imports between ``test_*.py`` modules and directs shared
fixtures into a ``tests/_*.py`` module instead -- this is that module. What
lives here is what both ``test_pipe_holder_release.py`` and
``test_pipe_holder_spawn_parent_settle.py`` stage against: the sleeper/launcher
source strings and interpreter identity every pipe-holder row spawns, and the
``_spawn_orphan`` positive control whose parent-settle window issue #3047 pins.
Each suite still declares its own tiny ``orphans`` teardown fixture -- at import
time the fixture name would collide with every test's ``orphans`` parameter
(ruff's redefinition rules), so sharing it is not worth the collision.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil

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
# The launcher-ppid race, and how many times ``_spawn_orphan`` restarts when
# the host wins it. The launcher prints the sleeper's pid and exits right
# away, so from that moment the pid is held by no handle this process owns;
# a busy host (a CI shard running another pytest worker that spawns and
# kills interpreters constantly) races the checks below on every attempt:
# * the printed pid can name no process at all, or an unrelated one whose
#   cmdline is someone else's code;
# * if the pid was reborn, its owner's recorded ppid can name a *live* real
#   parent (e.g. a pytest xdist worker), so psutil's ``parent()`` answers
#   ``Process(that live parent)``; conversely, a sweep one table behind can
#   still list the just-exited launcher as the sleeper's live parent for a
#   moment, so a verified sleeper's ``parent()`` is polled briefly before it
#   is believed and killed.
# Every gate is checked against this evidence each attempt, and a failed gate
# restarts the spawn rather than asserting on a first look.
_SPAWN_ORPHAN_ATTEMPTS = 3
# How long a verified sleeper's ``parent()`` is believed to linger -- polls
# absorb the process-table lag only; a parent that is alive (a stray sleeper
# one of the suites stages) is detected after the window.
_PARENT_POLL_SECONDS = 2.0


def _spawn_orphan(orphans: list[psutil.Process], cwd: Path) -> psutil.Process:
    """Start a sleeper in ``cwd`` whose parent has already exited -- a real orphan.

    Every gate is identity-verified before the controls below are trusted: the
    pid must still be running and must still name the sleeper code (its
    cmdline), and its parent must be gone -- polled for ``_PARENT_POLL_SECONDS``,
    because a process table can lag the launcher's exit by a sweep. What a gate
    saw is kept as the attempt's last look and the spawn restarts; a stranger's
    pid is never touched (it was never ours), while a verified sleeper whose
    parent shows up anyway is killed before the retry (it is pinned to its
    (pid, create time) either way). Exits with the last attempt's evidence,
    never silently.
    """
    last_look = ""
    for _ in range(_SPAWN_ORPHAN_ATTEMPTS):
        launched = subprocess.run(
            [_PYTHON, "-c", _ORPHAN_LAUNCHER],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        try:
            orphan = psutil.Process(int(launched.stdout.strip()))
        except psutil.Error as err:
            # The pid printed by the launcher names no process now: the sleeper
            # died and its pid was not reborn either. Nothing was ours to leak
            # or touch; restart the spawn.
            last_look = f"the printed pid was already gone: {err!r}"
            continue
        if not orphan.is_running():
            # psutil stamps this instance gone once it stops finding the pid,
            # so polling ``is_running()`` could only spin to the deadline on
            # the same dead psutil instance no matter the truth: drop it and
            # let the next attempt build a fresh one from a fresh launcher.
            last_look = f"pid {orphan.pid} not running: {orphan!r}"
            continue
        cmdline: list[str] = []
        try:
            cmdline = list(orphan.cmdline())
        except psutil.Error as err:
            last_look = f"pid {orphan.pid} unreadable: {err!r}"
            continue
        if not any(_SLEEPER in arg for arg in cmdline):
            # A reborn pid: the owner was never ours, so it is not touched.
            last_look = (
                f"pid {orphan.pid} no longer names the sleeper code: {cmdline!r} ({orphan!r})"
            )
            continue
        # psutil's own parent() rule (missing, or a recycled pid) as an
        # independent control that the process really is parentless; a just-
        # exited launcher can linger in the process table a sweep long.
        deadline = time.monotonic() + _PARENT_POLL_SECONDS
        parent = orphan.parent()
        while parent is not None and time.monotonic() < deadline:
            time.sleep(0.05)
            parent = orphan.parent()
        if parent is None:
            orphans.append(orphan)
            return orphan
        last_look = f"orphan {orphan!r} claimed parent {parent!r}"
        try:
            orphan.kill()  # ours: a verified sleeper, pinned to its (pid, create time)
        except psutil.Error:
            pass
    raise AssertionError(
        "positive control: the orphan's parent must be gone (the spawn raced "
        f"out from under the test {_SPAWN_ORPHAN_ATTEMPTS} times): {last_look}"
    )
