"""Unit tests for ``charlie_work.orphan_sweep._reap_enumerated_children`` (#2059).

Split out of ``test_process_utils_kill_process_tree.py`` when the #2059
coverage pushed that file over the 800-line cap. ``_stub_reap_env`` stubs
only what the reaper calls, so each case can select the platform branch
under test on any host.
"""

from __future__ import annotations

import os
import signal
from typing import Any

import pytest

from charlie_work.process_utils import kill_process_tree
from charlie_work.subprocess_runner import RunResult


def _stub_reap_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    alive: dict[int, bool],
    queries: dict[int, Any],
    os_name: str = "posix",
) -> dict[str, list[Any]]:
    """Install the seams ``_reap_enumerated_children`` touches.

    Unlike ``_stub_kill_env`` (which drives ``kill_process_tree`` end-to-end
    on a forced-Windows model), this stubs only what the reaper calls --
    ``is_pid_alive`` / ``get_process_start_time`` / ``run_captured`` on
    ``process_utils`` plus ``os.kill``/``os.name`` -- so the platform branch
    under test is selectable per case. ``alive`` is the liveness oracle;
    ``queries`` is what ``get_process_start_time`` returns per pid, and a
    list value is consumed one entry per call (modeling a start-time read
    that changes between the pinned liveness probe and the identity
    re-check). ``is_pid_alive`` mirrors the real one: a dead pid is dead, an
    indeterminate start-time read counts as alive, and a start drifted more
    than a second from the fingerprint means the pid was recycled onto
    another process.

    The fake ``os.kill``/``run_captured`` model a successful kill -- the
    target flips dead; a test needing a resisted or failed kill re-patches
    the seam afterwards. ``signal.SIGKILL`` does not exist on win32, so it is
    defined just for the test duration there.
    """
    import charlie_work.process_utils as _pu

    sigkill_calls: list[tuple[int, int]] = []
    taskkill_calls: list[list[str]] = []
    start_queries: list[int] = []

    def fake_start(pid: int) -> float | None:
        start_queries.append(pid)
        value = queries.get(pid)
        if isinstance(value, list):
            return value.pop(0) if value else None
        return value

    def fake_alive(pid: int, expected_start_time: float | None = None) -> bool:
        if not alive.get(pid, False):
            return False
        if expected_start_time is None:
            return True
        # Resolved through the module attribute so a later monkeypatch of
        # get_process_start_time is honored, like the real is_pid_alive.
        current = _pu.get_process_start_time(pid)
        if current is None:
            return True
        return abs(current - expected_start_time) <= 1.0

    def fake_os_kill(pid: int, sig: int) -> None:
        sigkill_calls.append((pid, sig))
        alive[pid] = False

    def fake_run_captured(command: list[str], **_kwargs: Any) -> RunResult:
        taskkill_calls.append(list(command))
        try:
            killed_pid = int(command[-1])
        except (ValueError, IndexError):
            killed_pid = -1
        if killed_pid in alive:
            alive[killed_pid] = False
        return RunResult(returncode=0, stdout="", stderr="", error=None)

    monkeypatch.setattr(_pu, "get_process_start_time", fake_start)
    monkeypatch.setattr(_pu, "is_pid_alive", fake_alive)
    monkeypatch.setattr(_pu, "run_captured", fake_run_captured)
    monkeypatch.setattr(os, "kill", fake_os_kill)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(os, "name", os_name)
    return {"sigkill": sigkill_calls, "taskkill": taskkill_calls, "start_queries": start_queries}


def test_reap_enumerated_children_posix_sigkill_reaps_pinned_survivor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSIX individual reap: a survivor verified alive under its fingerprint
    gets ``os.kill(pid, SIGKILL)`` and is recorded only after the pinned
    ``is_pid_alive`` re-check confirms death (#2059).

    The existing stub suite forces ``os.name == "nt"``; this exercises the
    POSIX ``os.kill`` branch -- ``killpg`` cannot reach a child that left the
    process group -- on any host."""
    import charlie_work.orphan_sweep as _osweep

    root, child = 60101, 60102
    alive = {root: False, child: True}
    queries: dict[int, Any] = {root: 1000.0, child: 2000.0}
    calls = _stub_reap_env(monkeypatch, alive=alive, queries=queries)

    dead = _osweep._reap_enumerated_children(root, {child: 2000.0}, frozenset(), 1000.0)

    assert dead == [child]
    assert calls["sigkill"] == [(child, 9)]
    assert calls["taskkill"] == []


@pytest.mark.parametrize("start_offset", [-0.010, 0.0, 0.005])
def test_reap_enumerated_children_posix_tolerates_start_jitter(
    monkeypatch: pytest.MonkeyPatch, start_offset: float
) -> None:
    """A real POSIX child can *read* as starting a few ms before its root:
    /proc starttime is tick-quantized (~10 ms at HZ=100) on an estimated boot
    base, so root/child comparisons jitter -- and POSIX reparents orphans, so
    a ppid read is never stale anyway. The stale-ParentProcessId refusal is
    therefore Windows-only, and such a child must still be reaped (#2059)."""
    import charlie_work.orphan_sweep as _osweep

    root, child = 60111, 60112
    child_start = 1000.0 + start_offset
    alive = {root: False, child: True}
    queries: dict[int, Any] = {root: 1000.0, child: child_start}
    calls = _stub_reap_env(monkeypatch, alive=alive, queries=queries)

    dead = _osweep._reap_enumerated_children(root, {child: child_start}, frozenset(), 1000.0)

    assert dead == [child]
    assert calls["sigkill"] == [(child, 9)]


def test_reap_enumerated_children_survivor_of_direct_kill_not_recorded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A child still pinned-alive after its direct kill is warned about and
    NOT recorded -- reporting it dead would repeat the #2059 phantom-report
    bug. ``_CHILD_KILL_CONFIRM_SECONDS`` is zeroed so the confirm poll exits
    immediately."""
    import logging

    import charlie_work.orphan_sweep as _osweep

    root, child = 60121, 60122
    alive = {root: False, child: True}
    queries: dict[int, Any] = {root: 1000.0, child: 2000.0}
    calls = _stub_reap_env(monkeypatch, alive=alive, queries=queries)
    monkeypatch.setattr(_osweep, "_CHILD_KILL_CONFIRM_SECONDS", 0.0)
    # The signal is delivered but the pid is still live at the deadline
    # (e.g. an immune/protected holder).
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls["sigkill"].append((pid, sig)))

    with caplog.at_level(logging.WARNING, logger="charlie_work"):
        dead = _osweep._reap_enumerated_children(root, {child: 2000.0}, frozenset(), 1000.0)

    assert dead == []
    assert calls["sigkill"] == [(child, 9)]
    assert any(
        "survived the tree kill and a direct kill" in r.getMessage() for r in caplog.records
    )


def test_reap_enumerated_children_survivor_warning_reports_taskkill_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The 'survived a direct kill' warning carries the individual taskkill's
    own result, so a kill that never landed is diagnosable from the log
    instead of reading identically to a resisted kill (#2059)."""
    import logging

    import charlie_work.orphan_sweep as _osweep
    import charlie_work.process_utils as _pu

    root, child = 60131, 60132
    alive = {root: False, child: True}
    queries: dict[int, Any] = {root: 1000.0, child: 2000.0}
    calls = _stub_reap_env(monkeypatch, alive=alive, queries=queries, os_name="nt")
    monkeypatch.setattr(_osweep, "_CHILD_KILL_CONFIRM_SECONDS", 0.0)

    def failed_taskkill(command: list[str], **_kwargs: Any) -> RunResult:
        calls["taskkill"].append(list(command))
        return RunResult(
            returncode=128,
            stdout="",
            stderr="ERROR: The process was not found.",
            error="command exited 128",
        )

    monkeypatch.setattr(_pu, "run_captured", failed_taskkill)

    with caplog.at_level(logging.WARNING, logger="charlie_work"):
        dead = _osweep._reap_enumerated_children(root, {child: 2000.0}, frozenset(), 1000.0)

    assert dead == []
    assert calls["taskkill"] == [["taskkill", "/T", "/F", "/PID", str(child)]]
    survivor_warnings = [
        r.getMessage()
        for r in caplog.records
        if "survived the tree kill and a direct kill" in r.getMessage()
    ]
    assert survivor_warnings
    assert "not found" in survivor_warnings[0]


@pytest.mark.parametrize("child_queries", [None, [2000.0, 2005.5]], ids=["unreadable", "drifted"])
def test_reap_enumerated_children_refuses_when_recheck_unverifiable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    child_queries: Any,
) -> None:
    """A child alive under its fingerprint but whose start-time re-read is
    unreadable (None -- protected, or exited mid-check) or drifted is never
    killed by bare pid: the kill could land on a stranger holding a recycled
    pid. Warned, not recorded (#2059)."""
    import logging

    import charlie_work.orphan_sweep as _osweep

    root, child = 60141, 60142
    alive = {root: False, child: True}
    queries: dict[int, Any] = {root: 1000.0, child: child_queries}
    calls = _stub_reap_env(monkeypatch, alive=alive, queries=queries)

    with caplog.at_level(logging.WARNING, logger="charlie_work"):
        dead = _osweep._reap_enumerated_children(root, {child: 2000.0}, frozenset(), 1000.0)

    assert dead == []
    assert calls["sigkill"] == []
    assert calls["taskkill"] == []
    assert any("no longer verifiable" in r.getMessage() for r in caplog.records)


def test_reap_enumerated_children_recycled_pid_recorded_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pid recycled onto a different process reads dead under the pinned
    fingerprint -- the enumerated child is genuinely gone -- so it is
    recorded without any individual kill (#2059)."""
    import charlie_work.orphan_sweep as _osweep

    root, child = 60151, 60152
    alive = {root: False, child: True}  # pid live, but held by a stranger
    queries: dict[int, Any] = {root: 1000.0, child: 3000.0}  # != the 2000.0 fingerprint
    calls = _stub_reap_env(monkeypatch, alive=alive, queries=queries)

    dead = _osweep._reap_enumerated_children(root, {child: 2000.0}, frozenset(), 1000.0)

    assert dead == [child]
    assert calls["sigkill"] == []
    assert calls["taskkill"] == []


def test_reap_enumerated_children_uses_caller_supplied_root_start(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A caller-supplied ``expected_root_start_time`` is used as-is: the
    root's start is never re-read (the root is dead and possibly fully
    reaped by now), and on Windows the supplied value drives the stale-ppid
    refusal."""
    import logging

    import charlie_work.orphan_sweep as _osweep

    root, phantom = 60161, 60162
    alive = {root: False, phantom: True}
    # No root entry: any re-read would return None and show up in
    # start_queries. The phantom predates the supplied root start.
    queries: dict[int, Any] = {phantom: 500.0}
    calls = _stub_reap_env(monkeypatch, alive=alive, queries=queries, os_name="nt")

    with caplog.at_level(logging.WARNING, logger="charlie_work"):
        dead = _osweep._reap_enumerated_children(root, {phantom: 500.0}, frozenset(), 1000.0)

    assert dead == []
    assert root not in calls["start_queries"]
    assert calls["taskkill"] == []
    assert any(
        "stale ParentProcessId" in r.getMessage() and "500.000 <= 1000.000" in r.getMessage()
        for r in caplog.records
    )


def test_kill_process_tree_posix_reaps_survivor_killpg_missed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end POSIX: ``killpg`` reaches only the process group; an
    enumerated child that left it survives the tree kill and is individually
    SIGKILLed, then recorded (#2059)."""
    import charlie_work.process_utils as _pu

    root, child = 60171, 60172
    alive = {root: True, child: True}
    queries: dict[int, Any] = {root: 1000.0, child: 2000.0}
    calls = _stub_reap_env(monkeypatch, alive=alive, queries=queries)

    monkeypatch.setattr(_pu, "_enumerate_child_pids", lambda _pid: [child])
    # os.getpgid/os.killpg do not exist on win32 -- defined here for the
    # forced-POSIX branch. The root's group (500) differs from the caller's
    # (900) so the own-group guard does not fire.
    monkeypatch.setattr(os, "getpgid", lambda pid: 900 if pid == 0 else 500, raising=False)

    def fake_killpg(pgid: int, sig: int) -> None:
        # The group kill drops the root; the child already left the group.
        alive[root] = False

    monkeypatch.setattr(os, "killpg", fake_killpg, raising=False)

    killed = kill_process_tree(root, expected_start_time=None)

    assert killed == [root, child]
    assert calls["sigkill"] == [(child, 9)]
