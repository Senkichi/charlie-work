"""Issue #1842: kill primitives must never terminate the caller's own ancestry.

``kill_process_tree`` self-guarded only ``os.getpid()`` — a target PID that is
an *ancestor* of the caller (the pytest controller, ``uv``, the step's pwsh)
slipped through, and ``taskkill /T /PID <ancestor>`` fells the whole subtree
containing the caller. ``kill_orphan_pid`` had no self-guard at all. These
tests pin both guards: a fabricated ancestor set (deterministic, both
platforms) and the real parent PID exercised end-to-end against the live
ppid snapshot.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

import charlie_work.orphan_sweep as _sweep
import charlie_work.process_utils as _pu
from charlie_work.process_utils import kill_orphan_pid, kill_process_tree
from charlie_work.subprocess_runner import RunResult


def _patch_self_chain(monkeypatch: pytest.MonkeyPatch, chain: frozenset[int]) -> None:
    """Install a fabricated ancestor set for ``_self_ancestor_pids``.

    ``raising=False`` so the same test, run against unfixed code where the
    helper does not exist, creates an inert attribute instead of erroring —
    the kill itself is what the assertions observe.
    """
    monkeypatch.setattr(_pu, "_self_ancestor_pids", lambda: chain, raising=False)


def test_kill_process_tree_refuses_ancestor_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """A target PID on the caller's ancestor chain must be refused outright —
    before child enumeration, before start-time verification, before any
    platform kill primitive runs.
    """
    own_pid, ancestor_pid = 424242, 777
    monkeypatch.setattr(_pu.os, "getpid", lambda: own_pid)
    _patch_self_chain(monkeypatch, frozenset({own_pid, ancestor_pid, 1}))
    monkeypatch.setattr(_pu, "_enumerate_child_pids", lambda _pid: [])

    kill_attempts: list[Any] = []

    def fake_run_captured(command: Any, **kwargs: Any) -> RunResult:
        kill_attempts.append(command)
        return RunResult(returncode=0, stdout="", stderr="", error=None)

    monkeypatch.setattr(_pu, "run_captured", fake_run_captured)
    # POSIX leg: distinct pgids so the process-group guard passes and the
    # ancestor guard is the only thing that can prevent the kill.
    monkeypatch.setattr(_pu.os, "getpgid", lambda pid: 1000 + pid, raising=False)
    monkeypatch.setattr(
        _pu.os,
        "killpg",
        lambda pgid, sig: kill_attempts.append(("killpg", pgid, sig)),
        raising=False,
    )

    killed = kill_process_tree(ancestor_pid)

    assert killed == []
    assert kill_attempts == []


def test_kill_orphan_pid_refuses_ancestor_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """``kill_orphan_pid`` must refuse a target on the caller's ancestor chain.

    Orphan PIDs arrive from ``sweep_orphan_processes``'s CommandLine substring
    match; a too-broad worktree needle legitimately matches this process's own
    ancestry, so the same guard ``kill_process_tree`` carries applies here.
    """
    own_pid, ancestor_pid = 424242, 777
    monkeypatch.setattr(_pu.os, "getpid", lambda: own_pid)
    _patch_self_chain(monkeypatch, frozenset({own_pid, ancestor_pid, 1}))

    kill_attempts: list[Any] = []
    monkeypatch.setattr(
        _pu,
        "run_captured",
        lambda command, **kwargs: (
            kill_attempts.append(command)
            or RunResult(returncode=0, stdout="", stderr="", error=None)
        ),
    )
    monkeypatch.setattr(
        _pu.os,
        "kill",
        lambda pid, sig: kill_attempts.append(("os.kill", pid, sig)),
        raising=False,
    )

    kill_orphan_pid(ancestor_pid)

    assert kill_attempts == []


def test_kill_orphan_pid_still_kills_unrelated_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: a PID outside the ancestor chain still reaches the
    platform kill primitive — the guard narrows kills, it does not disable them.
    """
    own_pid, ancestor_pid, orphan_pid = 424242, 777, 31337
    monkeypatch.setattr(_pu.os, "getpid", lambda: own_pid)
    _patch_self_chain(monkeypatch, frozenset({own_pid, ancestor_pid, 1}))

    kill_attempts: list[Any] = []
    monkeypatch.setattr(
        _pu,
        "run_captured",
        lambda command, **kwargs: (
            kill_attempts.append(command)
            or RunResult(returncode=0, stdout="", stderr="", error=None)
        ),
    )
    monkeypatch.setattr(
        _pu.os,
        "kill",
        lambda pid, sig: kill_attempts.append(("os.kill", pid, sig)),
        raising=False,
    )

    kill_orphan_pid(orphan_pid)

    assert len(kill_attempts) == 1


def _self_chain_or_skip(helper: Any, parent_pid: int) -> None:
    """Require the real ppid snapshot to place ``parent_pid`` on our chain.

    Distinguishes "snapshot substrate broken on this host" (skip — the guard
    cannot fire on a chain it cannot see) from "guard absent or misfired"
    (fail). A bounded retry absorbs a transient CIM hiccup under xdist load,
    the same convention the kill_process_tree tests already carry.
    """
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if parent_pid in helper():
            return
        time.sleep(0.25)
    pytest.skip("ancestor snapshot cannot see this process's parent on this host")


def test_kill_orphan_pid_refuses_real_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end against the real snapshot: ``os.getppid()`` is a genuine,
    live, killable ancestor — ``kill_orphan_pid`` must refuse it without any
    fabricated chain.
    """
    helper = getattr(_pu, "_self_ancestor_pids", None)
    parent_pid = os.getppid()

    kill_attempts: list[Any] = []
    if os.name == "nt":
        monkeypatch.setattr(
            _pu,
            "run_captured",
            lambda command, **kwargs: (
                kill_attempts.append(command)
                or RunResult(returncode=0, stdout="", stderr="", error=None)
            ),
        )
    else:
        monkeypatch.setattr(
            _pu.os,
            "kill",
            lambda pid, sig: kill_attempts.append(("os.kill", pid, sig)),
            raising=False,
        )

    if helper is not None:
        _self_chain_or_skip(helper, parent_pid)

    # A transient snapshot failure inside the kill call can slip the guard
    # once; retry so the verdict reflects the guard, not the substrate. Under
    # unfixed code every call reaches the spy, so the loop still fails.
    for _ in range(5):
        kill_orphan_pid(parent_pid)
        if not kill_attempts:
            return
        kill_attempts.clear()
    pytest.fail(
        f"kill_orphan_pid reached the platform kill primitive for ancestor "
        f"{parent_pid}; the ancestor guard did not fire"
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows-only: real ppid chain via CIM")
def test_kill_process_tree_refuses_real_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows end-to-end: ``os.getppid()`` resolved through the real CIM
    snapshot must be refused before ``taskkill`` runs.

    Windows-gated because on POSIX the pre-existing ``killpg`` group guard can
    already refuse our parent's group, which would mask the ancestor check the
    test means to exercise.
    """
    helper = getattr(_pu, "_self_ancestor_pids", None)
    parent_pid = os.getppid()

    kill_attempts: list[Any] = []
    monkeypatch.setattr(
        _pu,
        "run_captured",
        lambda command, **kwargs: (
            kill_attempts.append(command)
            or RunResult(returncode=0, stdout="", stderr="", error=None)
        ),
    )
    monkeypatch.setattr(_pu, "_enumerate_child_pids", lambda _pid: [])

    if helper is not None:
        _self_chain_or_skip(helper, parent_pid)

    for _ in range(5):
        killed = kill_process_tree(parent_pid)
        if not kill_attempts and killed == []:
            return
        kill_attempts.clear()
    pytest.fail(
        f"kill_process_tree reached taskkill for ancestor {parent_pid}; "
        "the ancestor guard did not fire"
    )


def test_self_ancestor_pids_walks_win32_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fabricated CIM rows: the walk must return self plus the whole chain,
    including the final ancestor whose own ppid is absent from the snapshot —
    the same boundary ``quiesce.self_process_chain`` tests pin.
    """
    monkeypatch.setattr(_sweep.os, "name", "nt")
    monkeypatch.setattr(
        _sweep,
        "_win32_process_ppid_snapshot",
        lambda: {300: 200, 200: 100, 100: 1, 50: 1},
    )
    monkeypatch.setattr(_sweep.os, "getpid", lambda: 300)

    assert _sweep._self_ancestor_pids() == frozenset({300, 200, 100, 1})


def test_self_ancestor_pids_terminates_on_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cyclic ppid map must not spin the walk — it terminates on the
    already-seen ancestor."""
    monkeypatch.setattr(_sweep.os, "name", "nt")
    monkeypatch.setattr(_sweep, "_win32_process_ppid_snapshot", lambda: {5: 7, 7: 5})
    monkeypatch.setattr(_sweep.os, "getpid", lambda: 5)

    assert _sweep._self_ancestor_pids() == frozenset({5, 7})


def test_self_ancestor_pids_degrades_to_self_when_snapshot_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken snapshot degrades to the pre-#1842 bare self-pid guard rather
    than disabling reaping — or pretending to protect ancestors it cannot see.
    """
    monkeypatch.setattr(_sweep.os, "name", "nt")
    monkeypatch.setattr(_sweep, "_win32_process_ppid_snapshot", lambda: {})
    monkeypatch.setattr(_sweep.os, "getpid", lambda: 424242)

    assert _sweep._self_ancestor_pids() == frozenset({424242})


def test_posix_process_ppid_snapshot_parses_procfs(tmp_path: Path) -> None:
    """Fabricated procfs: ``/proc/<pid>/stat`` ppid is the first field after
    ``state``, split off comm on the LAST ``)`` (comm itself can hold spaces
    and parens)."""
    rows = {100: 50, 200: 100, 50: 1}
    for pid, ppid in rows.items():
        proc_dir = tmp_path / str(pid)
        proc_dir.mkdir()
        comm = "we(rd)name" if pid == 200 else "python"
        (proc_dir / "stat").write_text(f"{pid} ({comm}) S {ppid} 1 2 3", encoding="utf-8")
    (tmp_path / "notapid").mkdir()  # non-numeric entry is skipped

    assert _sweep._posix_process_ppid_snapshot(tmp_path) == rows


def test_win32_process_ppid_snapshot_normalizes_single_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``-AsArray`` (dropped for PS 5.1 compat), ``ConvertTo-Json``
    emits a bare object for a single result — the snapshot must treat a dict
    payload as a one-element list, same as ``quiesce.list_processes``.
    """
    import json

    monkeypatch.setattr(_sweep.shutil, "which", lambda _name: "powershell")
    # The snapshot shells out through ``run_captured``, which reaches
    # ``subprocess.run`` via the shared module — patch it there.
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 0, json.dumps({"ProcessId": 7, "ParentProcessId": 3}), ""
        ),
    )

    assert _sweep._win32_process_ppid_snapshot() == {7: 3}


def test_win32_process_ppid_snapshot_failure_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No PowerShell (or a failed/unparseable query) must yield ``{}`` so the
    ancestor guard degrades rather than raising."""
    monkeypatch.setattr(_sweep.shutil, "which", lambda _name: None)
    assert _sweep._win32_process_ppid_snapshot() == {}


@pytest.mark.parametrize("bad_pid", [0, -1])
def test_kill_orphan_pid_refuses_nonpositive_pid(
    monkeypatch: pytest.MonkeyPatch, bad_pid: int
) -> None:
    """``pid <= 0`` is refused before the ancestor lookup and before any
    platform kill primitive — a degenerate sweep entry must never reach
    ``taskkill``/``os.kill``."""
    kill_attempts: list[Any] = []
    monkeypatch.setattr(
        _pu,
        "run_captured",
        lambda command, **kwargs: (
            kill_attempts.append(command)
            or RunResult(returncode=0, stdout="", stderr="", error=None)
        ),
    )
    monkeypatch.setattr(
        _pu.os,
        "kill",
        lambda pid, sig: kill_attempts.append(("os.kill", pid, sig)),
        raising=False,
    )

    kill_orphan_pid(bad_pid)

    assert kill_attempts == []


def test_kill_process_tree_ancestor_guard_precedes_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ancestor guard fires *before* start-time fingerprinting — a
    matching fingerprint must not rescue a target that names a caller
    ancestor (pins guard-before-fingerprint ordering)."""
    own_pid, ancestor_pid = 424242, 777
    fingerprint = 1_700_000_000.0
    monkeypatch.setattr(_pu.os, "getpid", lambda: own_pid)
    _patch_self_chain(monkeypatch, frozenset({own_pid, ancestor_pid, 1}))

    fingerprint_calls: list[int] = []
    monkeypatch.setattr(
        _pu,
        "get_process_start_time",
        lambda pid: fingerprint_calls.append(pid) or fingerprint,
    )
    monkeypatch.setattr(_pu, "_enumerate_child_pids", lambda _pid: [])

    kill_attempts: list[Any] = []
    monkeypatch.setattr(
        _pu,
        "run_captured",
        lambda command, **kwargs: (
            kill_attempts.append(command)
            or RunResult(returncode=0, stdout="", stderr="", error=None)
        ),
    )
    monkeypatch.setattr(_pu.os, "getpgid", lambda pid: 1000 + pid, raising=False)
    monkeypatch.setattr(
        _pu.os,
        "killpg",
        lambda pgid, sig: kill_attempts.append(("killpg", pgid, sig)),
        raising=False,
    )

    killed = kill_process_tree(ancestor_pid, expected_start_time=fingerprint)

    assert killed == []
    assert kill_attempts == []
    assert fingerprint_calls == []  # the guard refused before fingerprinting


def test_ancestor_refusal_logs_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Refusals are the only telemetry that can confirm or refute whether the
    #1842 guard ever fires — both kill primitives must log them, not swallow
    them."""
    own_pid, ancestor_pid = 424242, 777
    monkeypatch.setattr(_pu.os, "getpid", lambda: own_pid)
    _patch_self_chain(monkeypatch, frozenset({own_pid, ancestor_pid, 1}))
    monkeypatch.setattr(_pu, "_enumerate_child_pids", lambda _pid: [])
    monkeypatch.setattr(
        _pu,
        "run_captured",
        lambda *a, **k: RunResult(returncode=0, stdout="", stderr="", error=None),
    )
    monkeypatch.setattr(_pu.os, "kill", lambda *a: None, raising=False)

    with caplog.at_level(logging.WARNING, logger="charlie_work.process_utils"):
        assert kill_process_tree(ancestor_pid) == []
        kill_orphan_pid(ancestor_pid)

    messages = [record.getMessage() for record in caplog.records]
    assert any("kill_process_tree" in m and str(ancestor_pid) in m for m in messages)
    assert any("kill_orphan_pid" in m and str(ancestor_pid) in m for m in messages)


@pytest.mark.parametrize("exc_type", [PermissionError, OSError])
def test_snapshot_spawn_oserror_degrades_without_raising(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    exc_type: type[OSError],
) -> None:
    """A spawn-level ``OSError`` (e.g. ``PermissionError`` from a denied
    ``CreateProcess``) is not a ``SubprocessError`` — the snapshot must
    return ``{}``, ``_self_ancestor_pids`` must degrade to ``{os.getpid()}``
    with a warning, and both kill primitives must return without raising.
    """
    monkeypatch.setattr(_sweep.os, "name", "nt")
    monkeypatch.setattr(shutil, "which", lambda _name: "powershell")

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise exc_type("denied")

    monkeypatch.setattr(subprocess, "run", boom)

    assert _sweep._win32_process_ppid_snapshot() == {}

    monkeypatch.setattr(_sweep.os, "getpid", lambda: 555)
    with caplog.at_level(logging.WARNING, logger="charlie_work.orphan_sweep"):
        assert _sweep._self_ancestor_pids() == frozenset({555})
    assert any("self-pid" in r.getMessage() for r in caplog.records)

    # The kill primitives consult the degraded set and still function.
    kill_attempts: list[Any] = []
    monkeypatch.setattr(
        _pu,
        "run_captured",
        lambda command, **kwargs: (
            kill_attempts.append(command)
            or RunResult(returncode=0, stdout="", stderr="", error=None)
        ),
    )
    monkeypatch.setattr(
        _pu.os,
        "kill",
        lambda pid, sig: kill_attempts.append(("os.kill", pid, sig)),
        raising=False,
    )
    monkeypatch.setattr(_pu.os, "getpgid", lambda pid: 1000 + pid, raising=False)
    monkeypatch.setattr(
        _pu.os,
        "killpg",
        lambda pgid, sig: kill_attempts.append(("killpg", pgid, sig)),
        raising=False,
    )
    monkeypatch.setattr(_pu, "_enumerate_child_pids", lambda _pid: [])

    kill_orphan_pid(555)  # degraded guard must still refuse the self-pid
    assert kill_attempts == []
    assert kill_process_tree(555) == []

    kill_orphan_pid(31337)
    killed = kill_process_tree(31337)

    # The degraded guard covers only self-pid — an unrelated target still
    # reaches the kill primitive (reaping is degraded, not disabled).
    assert len(kill_attempts) == 2
    assert 31337 in killed


def test_self_ancestor_pids_degrades_when_snapshot_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If the snapshot helper itself raises (e.g. ``Path.cwd()`` on a deleted
    cwd), ``_self_ancestor_pids`` degrades to ``{os.getpid()}`` with a warning
    instead of propagating into the kill path."""
    monkeypatch.setattr(_sweep.os, "name", "nt")

    def boom() -> dict[int, int]:
        raise PermissionError("cwd gone")

    monkeypatch.setattr(_sweep, "_win32_process_ppid_snapshot", boom)
    monkeypatch.setattr(_sweep.os, "getpid", lambda: 555)

    with caplog.at_level(logging.WARNING, logger="charlie_work.orphan_sweep"):
        assert _sweep._self_ancestor_pids() == frozenset({555})
    assert any("self-pid" in r.getMessage() for r in caplog.records)


def test_kill_primitives_survive_ancestor_lookup_raise(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If the ancestor lookup itself raises, the kill boundary degrades to
    the bare self-pid guard (with a warning) instead of propagating —
    ``kill_orphan_pid`` is documented never-raise, and an escape here would
    abort ``_sweep_orphan_processes_for_dead_sessions`` mid-pass."""
    monkeypatch.setattr(_pu.os, "getpid", lambda: 555)

    def boom() -> frozenset[int]:
        raise OSError("snapshot substrate exploded")

    monkeypatch.setattr(_pu, "_self_ancestor_pids", boom, raising=False)

    kill_attempts: list[Any] = []
    monkeypatch.setattr(
        _pu,
        "run_captured",
        lambda command, **kwargs: (
            kill_attempts.append(command)
            or RunResult(returncode=0, stdout="", stderr="", error=None)
        ),
    )
    monkeypatch.setattr(
        _pu.os,
        "kill",
        lambda pid, sig: kill_attempts.append(("os.kill", pid, sig)),
        raising=False,
    )
    monkeypatch.setattr(_pu.os, "getpgid", lambda pid: 1000 + pid, raising=False)
    monkeypatch.setattr(
        _pu.os,
        "killpg",
        lambda pgid, sig: kill_attempts.append(("killpg", pgid, sig)),
        raising=False,
    )
    monkeypatch.setattr(_pu, "_enumerate_child_pids", lambda _pid: [])

    with caplog.at_level(logging.WARNING, logger="charlie_work.process_utils"):
        kill_orphan_pid(555)  # degraded guard refuses self
        kill_orphan_pid(31337)  # unrelated target still killed
        killed_self = kill_process_tree(555)
        killed_other = kill_process_tree(31337)

    assert killed_self == []
    assert 31337 in killed_other
    assert len(kill_attempts) == 2
    assert any("degrades to self-pid" in r.getMessage() for r in caplog.records)
