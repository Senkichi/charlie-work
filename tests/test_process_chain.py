"""Unit tests for the shared process-ancestry machinery (issue #2058).

``process_chain`` holds the ``pid -> ProcRow`` row shape and
``ancestor_chain_pids`` -- the walk ``orphan_sweep._self_ancestor_pids``,
``quiesce.self_process_chain`` and ``host_load._self_tree`` share. These
tests pin the walk's termination rules directly (ordered output included)
rather than through any one consumer.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from charlie_work import process_chain as pc


def test_ancestor_chain_returns_ordered_full_chain() -> None:
    """The chain is ordered self -> ... -> top, and a recorded parent with no
    row of its own is still appended as the final ancestor."""
    rows = {
        300: pc.ProcRow(200, created=5_000.0),
        200: pc.ProcRow(100, created=3_000.0),
        100: pc.ProcRow(1, created=1_000.0),
    }

    assert pc.ancestor_chain_pids(300, rows) == (300, 200, 100, 1)


def test_ancestor_chain_rejects_parent_created_after_child() -> None:
    """Windows never reparents: a dead parent's pid can be recycled by an
    unrelated, younger process. The recorded parent created *after* its
    child is not an ancestor -- the walk stops there."""
    rows = {
        300: pc.ProcRow(200, created=5_000.0),
        200: pc.ProcRow(100, created=9_000.0),  # younger than its "child"
        100: pc.ProcRow(1, created=1_000.0),
    }

    assert pc.ancestor_chain_pids(300, rows) == (300,)


def test_ancestor_chain_keeps_links_when_stamps_unknown() -> None:
    """A ``created=None`` stamp never disqualifies a link -- the POSIX
    snapshot deliberately leaves every stamp unknown (reparenting makes
    stale links impossible there), so the no-stamp case is the full chain."""
    rows = {300: pc.ProcRow(200), 200: pc.ProcRow(100), 100: pc.ProcRow(1)}

    assert pc.ancestor_chain_pids(300, rows) == (300, 200, 100, 1)


def test_ancestor_chain_ends_at_absent_parent_row() -> None:
    rows = {300: pc.ProcRow(999)}

    assert pc.ancestor_chain_pids(300, rows) == (300, 999)


def test_ancestor_chain_terminates_on_cycle() -> None:
    rows = {5: pc.ProcRow(7), 7: pc.ProcRow(5)}

    assert pc.ancestor_chain_pids(5, rows) == (5, 7)


def test_ancestor_chain_ignores_nonpositive_parent() -> None:
    rows = {4: pc.ProcRow(0)}

    assert pc.ancestor_chain_pids(4, rows) == (4,)


def test_ancestor_chain_honors_hop_cap() -> None:
    rows = {pid: pc.ProcRow(pid - 1) for pid in range(10, 4, -1)}

    assert pc.ancestor_chain_pids(10, rows, max_hops=2) == (10, 9, 8)


def test_proc_rows_reduces_process_infos() -> None:
    """``proc_rows`` is the shared (ppid, created)-row producer for consumers
    holding ``ProcessInfo``-shaped rows (quiesce, host_load)."""
    infos = [
        SimpleNamespace(pid=10, ppid=5, created=100.0),
        SimpleNamespace(pid=20, ppid=10, created=None),
    ]

    assert pc.proc_rows(infos) == {10: pc.ProcRow(5, 100.0), 20: pc.ProcRow(10, None)}


# ---------------------------------------------------------------------------
# The OS snapshot helpers (moved here from orphan_sweep in #2058). The
# per-platform shapes are exercised end to end by
# tests/test_process_utils_ancestor_guard.py through orphan_sweep's aliases;
# these pin the same contract directly on the public names.
# ---------------------------------------------------------------------------


class _FakePsutilProc:
    """Minimal ``psutil.Process`` stand-in exposing only ``.info``."""

    def __init__(self, **info: Any) -> None:
        self.info = info


def test_win32_snapshot_normalizes_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pc, "_bulk_ppid_map", lambda: {7: 3, 8: 7})
    monkeypatch.setattr(
        pc,
        "_bulk_create_time_map",
        lambda: {7: 99.5, 8: None, 4: 1.0},
    )

    assert pc.win32_process_ppid_snapshot() == {
        7: pc.ProcRow(3, 99.5),
        8: pc.ProcRow(7, None),
        4: pc.ProcRow(0, 1.0),
    }


def test_win32_snapshot_never_reads_ppid_per_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #2353 regression pin: on Windows ``Process.ppid()`` rebuilds the
    kernel's whole ppid table per call, so a per-row ``ppid`` column on
    ``process_iter`` is O(n^2) — ~4 s at ~470 processes, paid by every
    ``kill_process_tree`` ancestor guard. The snapshot must take the bulk map
    once and never request a ``ppid`` column — including on the
    ``process_iter`` leg that runs when no ``cext.proc_times`` exists."""
    requested: list[Any] = []
    monkeypatch.setattr(pc, "_bulk_ppid_map", lambda: {7: 3})
    monkeypatch.setattr(pc, "_bulk_create_time_map", lambda: None)
    monkeypatch.setattr(
        pc.psutil,
        "process_iter",
        lambda attrs=None: (
            requested.append(attrs) or iter([_FakePsutilProc(pid=7, create_time=1.0)])
        ),
    )

    assert pc.win32_process_ppid_snapshot() == {7: pc.ProcRow(3, 1.0)}
    assert requested and all("ppid" not in (attrs or []) for attrs in requested)


def test_win32_snapshot_never_enumerates_via_process_iter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #2372 regression pin: ``process_iter``'s ``create_time`` column
    pays a full ``NtQuerySystemInformation`` walk per AccessDenied process
    (~1.7 ms each — ~1 s on a busy host, the ledger's measured regression).
    While the bulk ``cext.proc_times`` table exists, the snapshot must not
    touch ``process_iter`` at all."""
    monkeypatch.setattr(pc, "_bulk_ppid_map", lambda: {7: 3})
    monkeypatch.setattr(pc, "_bulk_create_time_map", lambda: {7: 1.0})

    def boom(attrs: Any = None) -> Any:
        raise AssertionError("process_iter must not run while the bulk map exists")

    monkeypatch.setattr(pc.psutil, "process_iter", boom)

    assert pc.win32_process_ppid_snapshot() == {7: pc.ProcRow(3, 1.0)}


def test_win32_snapshot_falls_back_to_process_iter_without_bulk_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A platform with no ``cext.proc_times`` (``_bulk_create_time_map`` ->
    ``None``) still gathers ``create_time`` from one ``process_iter`` pass —
    the pre-#2372 leg, kept so a hypothetical port stays correct."""
    monkeypatch.setattr(pc, "_bulk_ppid_map", lambda: {7: 3, 8: 7})
    monkeypatch.setattr(pc, "_bulk_create_time_map", lambda: None)
    monkeypatch.setattr(
        pc.psutil,
        "process_iter",
        lambda attrs=None: iter(
            [
                _FakePsutilProc(pid=7, create_time=99.5),
                _FakePsutilProc(pid=8, create_time=None),
            ]
        ),
    )

    assert pc.win32_process_ppid_snapshot() == {
        7: pc.ProcRow(3, 99.5),
        8: pc.ProcRow(7, None),
    }


def test_win32_snapshot_failure_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(attrs: Any = None) -> Any:
        raise pc.psutil.Error("substrate broken")

    monkeypatch.setattr(pc, "_bulk_ppid_map", lambda: {})
    monkeypatch.setattr(pc, "_bulk_create_time_map", boom)

    assert pc.win32_process_ppid_snapshot() == {}


def test_bulk_ppid_map_with_platform_ppid_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """psutil._psplatform.ppid_map present: returns int-coerced pid->ppid map
    in a single call, without touching process_iter."""
    fake_platform = SimpleNamespace(ppid_map=lambda: {"100": "50", 200: 100, 300: 0})
    monkeypatch.setattr(pc.psutil, "_psplatform", fake_platform, raising=False)

    def boom(attrs: Any = None) -> Any:
        raise AssertionError("process_iter must not be called when ppid_map is present")

    monkeypatch.setattr(pc.psutil, "process_iter", boom)

    assert pc._bulk_ppid_map() == {100: 50, 200: 100, 300: 0}


def test_bulk_ppid_map_fallback_when_platform_ppid_map_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """psutil._psplatform.ppid_map absent: falls back to per-row process_iter
    ppid read and returns the same int-coerced pid->ppid map."""
    monkeypatch.setattr(pc.psutil, "_psplatform", None, raising=False)
    monkeypatch.setattr(
        pc.psutil,
        "process_iter",
        lambda attrs=None: iter(
            [
                _FakePsutilProc(pid="100", ppid="50"),
                _FakePsutilProc(pid=200, ppid=100),
                _FakePsutilProc(pid=300, ppid=None),
                _FakePsutilProc(pid="invalid", ppid=1),
                _FakePsutilProc(ppid=1),
            ]
        ),
    )

    assert pc._bulk_ppid_map() == {100: 50, 200: 100, 300: 0}


def test_bulk_ppid_map_fallback_when_ppid_map_attribute_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """psutil._psplatform exists without ppid_map: falls back to process_iter."""
    monkeypatch.setattr(pc.psutil, "_psplatform", SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        pc.psutil,
        "process_iter",
        lambda attrs=None: iter([_FakePsutilProc(pid=100, ppid=50)]),
    )

    assert pc._bulk_ppid_map() == {100: 50}


def test_bulk_create_time_map_uses_proc_times(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cext.proc_times`` present: one call per pid from ``psutil.pids()``,
    the creation stamp read off the third tuple element, ``None`` kept for
    a denied open."""
    calls: list[int] = []

    def fake_proc_times(pid: int) -> tuple[float, float, float]:
        calls.append(pid)
        if pid == 8:
            raise PermissionError("denied")
        return (0.0, 0.0, {7: 99.5, 4: 1.0}[pid])

    monkeypatch.setattr(
        pc.psutil,
        "_psplatform",
        SimpleNamespace(cext=SimpleNamespace(proc_times=fake_proc_times)),
        raising=False,
    )
    monkeypatch.setattr(pc.psutil, "pids", lambda: [7, 8, 4])

    assert pc._bulk_create_time_map() == {7: 99.5, 8: None, 4: 1.0}
    assert calls == [7, 8, 4]


def test_bulk_create_time_map_drops_pid_gone_mid_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pid that exits between ``pids()`` and its ``proc_times`` read
    (``ProcessLookupError``) yields no row — the same boundary
    ``process_iter`` applies when it removes a gone pid."""

    def fake_proc_times(pid: int) -> tuple[float, float, float]:
        if pid == 8:
            raise ProcessLookupError(pid)
        return (0.0, 0.0, 1.0)

    monkeypatch.setattr(
        pc.psutil,
        "_psplatform",
        SimpleNamespace(cext=SimpleNamespace(proc_times=fake_proc_times)),
        raising=False,
    )
    monkeypatch.setattr(pc.psutil, "pids", lambda: [7, 8])

    assert pc._bulk_create_time_map() == {7: 1.0}


def test_bulk_create_time_map_absent_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A platform shipping no ``cext.proc_times`` returns ``None`` so the
    snapshot takes its ``process_iter`` fallback leg."""
    monkeypatch.setattr(
        pc.psutil, "_psplatform", SimpleNamespace(cext=SimpleNamespace()), raising=False
    )

    assert pc._bulk_create_time_map() is None


@pytest.mark.skipif(os.name != "nt", reason="Windows-only: real psutil snapshot")
def test_win32_snapshot_real_reports_actual_parent() -> None:
    """Real psutil snapshot on Windows (issue #2353 review): assert the live
    snapshot reports this process's actual parent (``os.getppid()``), so an
    all-zero or malformed ppid table from ``psutil._psplatform.ppid_map`` fails
    loudly instead of silently unprotecting ancestors."""
    snapshot = pc.win32_process_ppid_snapshot()
    row = snapshot.get(os.getpid())
    assert row is not None, "own pid missing from the real snapshot"
    assert row.ppid == os.getppid()
    assert row.created is not None


class _FakePsutilProcess:
    """``psutil.Process`` stand-in for ``win32_ancestor_rows``: looks up
    ``(ppid, create_time)`` rows in a dict, raising ``NoSuchProcess`` for a
    pid with no row — the fabricated dead-pid boundary. ``probes`` records
    which pids were touched so tests can pin the walk's scope."""

    def __init__(
        self,
        pid: int,
        rows: dict[int, tuple[int, float | None]],
        probes: list[int],
    ) -> None:
        probes.append(pid)
        self._pid = pid
        self._rows = rows

    def ppid(self) -> int:
        if self._pid not in self._rows:
            raise pc.psutil.NoSuchProcess(self._pid)
        return self._rows[self._pid][0]

    def create_time(self) -> float:
        if self._pid not in self._rows:
            raise pc.psutil.NoSuchProcess(self._pid)
        created = self._rows[self._pid][1]
        if created is None:
            raise pc.psutil.AccessDenied(self._pid)
        return created


def _patch_win32_process(
    monkeypatch: pytest.MonkeyPatch,
    rows: dict[int, tuple[int, float | None]],
) -> list[int]:
    """Install the fabricated ``psutil.Process`` and return its probe log.

    Also installs a fabricated ppid table at the ``_optional_bulk_ppid_map``
    seam: the walk consults that table first, and fabricated pids like 200
    or 300 can collide with real live host pids — a real table would hand
    the walk the host's ppid for them instead of the fabricated one.
    ``raising=False`` so the semantic pins still exercise the walk against
    code where the helper does not exist (the per-hop path) instead of
    erroring at patch time.
    """
    probes: list[int] = []
    monkeypatch.setattr(
        pc.psutil,
        "Process",
        lambda pid: _FakePsutilProcess(pid, rows, probes),
    )
    monkeypatch.setattr(
        pc,
        "_optional_bulk_ppid_map",
        lambda: {pid: row[0] for pid, row in rows.items()},
        raising=False,
    )
    return probes


def test_win32_ancestor_rows_walks_only_the_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chain-scoped (issue #2332): rows are gathered hop by hop through
    ``psutil.Process`` — the host-wide ``process_iter`` must never run, and
    pids off the chain are never touched. The ppid table is taken once per
    walk (issue #2362): ``Process.ppid()`` pays a full ``ppid_map()``
    enumeration per call, so per-hop ``ppid()`` would be O(hops x host
    processes); the saving is one enumeration per walk plus a handle probe
    per ancestor, not one enumeration per ancestor or per process."""
    rows = {300: (200, 5.0), 200: (100, 3.0), 100: (1, 1.0), 50: (1, 0.5)}
    probes = _patch_win32_process(monkeypatch, rows)

    def boom(attrs: Any = None) -> Any:
        raise AssertionError("win32_ancestor_rows must not enumerate all processes")

    monkeypatch.setattr(pc.psutil, "process_iter", boom)

    # pid 1 has no row: the walk names it in the child's ppid but cannot
    # read it — the same absent-row boundary a full snapshot produces.
    assert pc.win32_ancestor_rows(300) == {
        300: pc.ProcRow(200, 5.0),
        200: pc.ProcRow(100, 3.0),
        100: pc.ProcRow(1, 1.0),
    }
    assert 50 not in probes


def test_win32_ancestor_rows_keeps_link_when_create_time_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hop whose creation time cannot be read keeps its link via
    ``created=None`` — the same ``AccessDenied`` rule the full snapshot
    applies, so the walk adjudicates the link, not the gather."""
    rows = {300: (200, 5.0), 200: (100, None), 100: (1, 1.0)}
    _patch_win32_process(monkeypatch, rows)

    assert pc.win32_ancestor_rows(300) == {
        300: pc.ProcRow(200, 5.0),
        200: pc.ProcRow(100, None),
        100: pc.ProcRow(1, 1.0),
    }


def test_win32_ancestor_rows_returns_empty_when_self_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead/unreadable start pid yields ``{}`` — indistinguishable from a
    failed snapshot, so the caller's degrade path treats it the same."""
    _patch_win32_process(monkeypatch, rows={})

    assert pc.win32_ancestor_rows(99999) == {}


def test_win32_ancestor_rows_terminates_on_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = {5: (7, 1.0), 7: (5, 1.0)}
    probes = _patch_win32_process(monkeypatch, rows)

    assert pc.win32_ancestor_rows(5) == {5: pc.ProcRow(7, 1.0), 7: pc.ProcRow(5, 1.0)}
    assert probes == [5, 7]


def test_win32_ancestor_rows_takes_ppid_table_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #2362 regression pin: on psutil 7.x ``Process.ppid()`` re-runs
    ``ppid_map()`` — a full ``CreateToolhelp32Snapshot`` enumeration of every
    process on the host — per call, so a per-hop ``ppid()`` is O(hops x host
    processes) paid by every guarded kill. The walk must take the bulk table
    exactly once and consult ``ppid()`` only for a pid the table does not
    cover (the absent-row boundary)."""
    rows = {300: (200, 5.0), 200: (100, 3.0), 100: (1, 1.0)}
    _patch_win32_process(monkeypatch, rows)

    map_calls: list[int] = []

    def spy_map() -> dict[int, int]:
        map_calls.append(1)
        return {pid: row[0] for pid, row in rows.items()}

    monkeypatch.setattr(pc, "_optional_bulk_ppid_map", spy_map)

    ppid_calls: list[int] = []
    real_ppid = _FakePsutilProcess.ppid

    def spy_ppid(self: _FakePsutilProcess) -> int:
        ppid_calls.append(self._pid)
        return real_ppid(self)

    monkeypatch.setattr(_FakePsutilProcess, "ppid", spy_ppid)

    assert pc.win32_ancestor_rows(300) == {
        300: pc.ProcRow(200, 5.0),
        200: pc.ProcRow(100, 3.0),
        100: pc.ProcRow(1, 1.0),
    }
    assert map_calls == [1]
    # Only pid 1 — the chain's absent-row boundary, which the fabricated
    # table deliberately does not cover — reaches the per-hop fallback.
    assert ppid_calls == [1]


def test_win32_ancestor_rows_falls_back_when_no_bulk_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A platform with no ``ppid_map`` (``_optional_bulk_ppid_map`` ->
    ``None``, e.g. POSIX where ``ppid()`` is a cheap per-process read) still
    resolves every hop through per-hop ``Process.ppid()``."""
    rows = {300: (200, 5.0), 200: (100, 3.0)}
    _patch_win32_process(monkeypatch, rows)
    monkeypatch.setattr(pc, "_optional_bulk_ppid_map", lambda: None)

    assert pc.win32_ancestor_rows(300) == {
        300: pc.ProcRow(200, 5.0),
        200: pc.ProcRow(100, 3.0),
    }


def test_win32_ancestor_rows_real_chain_contains_self_and_parent() -> None:
    """Real psutil, no fakes (issue #2332 review): the live walk for this
    process must contain its own pid, record ``os.getppid()`` as its parent,
    and carry a row for that parent — and every ``created`` stamp must be a
    float or ``None``.

    ``win32_ancestor_rows`` only uses cross-platform ``psutil.Process``
    calls despite its name, so this runs on every CI platform rather than
    being Windows-gated."""
    rows = pc.win32_ancestor_rows(os.getpid())

    assert os.getpid() in rows
    assert rows[os.getpid()].ppid == os.getppid()
    assert os.getppid() in rows
    assert all(row.created is None or isinstance(row.created, float) for row in rows.values())


def test_posix_snapshot_reads_fabricated_procfs(tmp_path: Path) -> None:
    for pid, ppid in {100: 50, 200: 100}.items():
        proc_dir = tmp_path / str(pid)
        proc_dir.mkdir()
        (proc_dir / "stat").write_text(f"{pid} (weird (name)) S {ppid} 1 2 3", encoding="utf-8")
    (tmp_path / "notapid").mkdir()

    assert pc.posix_process_ppid_snapshot(tmp_path) == {
        100: pc.ProcRow(50),
        200: pc.ProcRow(100),
    }
