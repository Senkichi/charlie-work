"""The #3047 staging rows for ``_spawn_orphan``'s parent-settle window.

The fakes below stage the lingering-row window of issue #3047, which a real
process cannot produce on demand: ~1 in a few hundred orphan spawns met it under
churn, measured while fixing this (see the PR). ``subprocess.run`` has waited
for the launcher to exit, but its row can still answer ``Process(ppid)`` +
``create_time()`` in the few milliseconds right after -- a create time that
precedes the child's, so psutil's parent() rule returns it, while
``status()``/``ppid()`` already raise. That is what the nightly's annotation
rendered as ``status='terminated'``. The controls in ``_spawn_orphan`` must
wait for the row to drop rather than read the table once.

These rows live in their own module because ``test_pipe_holder_release.py`` sits
near the module-size cap (issue #1442's file-size ratchet), so the shared
helpers are imported from there rather than copied: the fakes stay pinned to the
spawn helper whose parent control they stage.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from test_pipe_holder_release import (
    _PYTHON,
    _SLEEPER,
    _SPAWN_ORPHAN_ATTEMPTS,
    _spawn_orphan,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="orphans are only parentless on Windows; POSIX reparents them to a live init",
)

# Same lifecycle guard the real-process rows use (test_pipe_holder_release's
# ``orphans`` fixture: killed on teardown, pinned so no foreign pid is touched).
# Declared here rather than re-exported, because a fixture import shadows the
# parameter name and trips ruff's redefinition rules.


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


_FAKE_ORPHAN_PID = 4_000_003
_FAKE_LAUNCHER_PID = 4_000_004
_FAKE_ORPHAN_CREATED = 1000.0


class _FakeOrphan:
    """A real-shaped orphan whose ``parent()`` is replayed through psutil's rule.

    It mirrors ``psutil.Process.parent()``'s own two-part rule -- raise-backed
    absence, and a holder created after the child read as a recycled pid, not a
    parent -- against whatever ``psutil.Process(ppid)`` serves, so the fake stays
    on psutil's rule instead of short-circuiting through a canned ``None``.
    """

    def is_running(self) -> bool:
        return True

    def cmdline(self) -> list[str]:
        # The identity gate in ``_spawn_orphan`` reads the sleeper code off the
        # cmdline; the fake must name its code like the real orphan does.
        return [_PYTHON, "-c", _SLEEPER]

    def parent(self) -> object | None:
        try:
            holder = psutil.Process(_FAKE_LAUNCHER_PID)
        except psutil.NoSuchProcess:
            return None
        if holder.create_time() > _FAKE_ORPHAN_CREATED:
            return None  # recycled pid, psutil's rule
        return holder

    def kill(self) -> None:
        pass  # teardown hygiene: the fake holds no real process


class _LingeringLauncherRow:
    """The dead launcher's row in the window: identity readable, the rest gone."""

    def __init__(self) -> None:
        self._created = _FAKE_ORPHAN_CREATED - 100  # an older process, "the parent"

    def create_time(self) -> float:
        return self._created

    def status(self) -> str:
        raise psutil.NoSuchProcess(_FAKE_LAUNCHER_PID)


def _canned_run(*args: object, **kwargs: object) -> SimpleNamespace:
    """Stand-in for the launcher's ``subprocess.run``: no process is spawned."""
    return SimpleNamespace(stdout=f"{_FAKE_ORPHAN_PID}\n")


def test_the_orphan_control_waits_out_a_temporarily_readable_dead_parent_row(
    monkeypatch: pytest.MonkeyPatch, orphans: list[psutil.Process]
) -> None:
    """The dead launcher's row is readable for a beat; the control waits it out (#3047).

    Staged: psutil's ``parent()`` rule is exercised through exactly the reads the
    rule makes (``Process(ppid)``, then ``create_time()``), so the lingering row
    answers for the first two (the window) and is reported gone from the third
    read on. The fake also carries the sleeper cmdline, so it clears the
    identity gates ahead of the parent poll. A control that read the parent
    link once instead of polling would drop the attempt inside such a window
    and fail only by exhausting the retries; polling absorbs it on the first
    attempt.
    """
    real_process = psutil.Process
    seen: list[int] = []
    built: list[object] = []

    def process(pid: int) -> psutil.Process:
        if pid == _FAKE_ORPHAN_PID:
            fake = _FakeOrphan()
            built.append(fake)
            return fake  # type: ignore[return-value]
        if pid == _FAKE_LAUNCHER_PID:
            seen.append(pid)
            if len(seen) <= 2:
                return _LingeringLauncherRow()  # type: ignore[return-value]
            raise psutil.NoSuchProcess(_FAKE_LAUNCHER_PID)
        return real_process(pid)

    monkeypatch.setattr(psutil, "Process", process)
    monkeypatch.setattr(subprocess, "run", _canned_run)

    orphan = _spawn_orphan(orphans, Path("."))
    assert built == [orphan], "the control returned the settled orphan"
    assert len(seen) == 3, "the settle-wait polled through the lingering row to its drop"


def test_the_orphan_control_fails_when_a_dead_parent_row_never_drops(
    monkeypatch: pytest.MonkeyPatch, orphans: list[psutil.Process]
) -> None:
    """The settle-wait is bounded: a parent link that never settles fails the control.

    Without the deadline a pathological row (a handle held open somewhere)
    would hang ``_spawn_orphan`` -- and, with it, every real-process row of the
    pipe-holder suite -- instead of failing loudly. The read count also pins the
    shape of the failure: each attempt polls the link (two reads at minimum,
    since the poll re-checks after every sleep) rather than reading it once and
    handing the pid straight to the retry, so this cannot pass against a control
    that dropped the poll.
    """
    real_process = psutil.Process
    seen: list[int] = []

    def process(pid: int) -> psutil.Process:
        if pid == _FAKE_ORPHAN_PID:
            return _FakeOrphan()  # type: ignore[return-value]
        if pid == _FAKE_LAUNCHER_PID:
            seen.append(pid)
            return _LingeringLauncherRow()  # type: ignore[return-value]
        return real_process(pid)

    monkeypatch.setattr(psutil, "Process", process)
    monkeypatch.setattr(subprocess, "run", _canned_run)

    with pytest.raises(AssertionError, match="the orphan's parent must be gone"):
        _spawn_orphan(orphans, Path("."))

    assert len(seen) >= 2 * _SPAWN_ORPHAN_ATTEMPTS, (
        "each attempt must poll the parent link, not read it once"
    )
