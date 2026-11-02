"""Issue #2058: ``quiesce.self_process_chain`` must not follow a stale parent link.

On Windows a recorded ``ParentProcessId`` is never updated when the parent
exits, and the freed pid can be recycled by an unrelated, *younger* process.
``self_process_chain`` builds the exclusion set the quiescence gate applies
before pattern matching; trusting a recycled pid pulls a stranger (and its
ancestry) into the exclusion set, so a process whose command line still
matches a fleet pattern is skipped and the gate reports "quiescent" while
it is running.

(``tests/test_quiesce.py``'s module attachment point is saturated, so the
#2058 cases live in this sibling file.)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys

import pytest

from charlie_work.process_chain import ProcRow
from charlie_work.quiesce import (
    ProcessInfo,
    assert_quiescent,
    list_processes,
    self_process_chain,
)

# Same stand-in for the config-supplied fleet-process regex as
# tests/test_quiesce.py uses (never hardcoded inside quiesce.py itself).
_SUPERVISE_PATTERN = r"fleet supervise"


def _proc(
    pid: int, ppid: int, name: str, command_line: str, created: float | None = None
) -> ProcessInfo:
    return ProcessInfo(pid=pid, ppid=ppid, name=name, command_line=command_line, created=created)


def _fake_completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["powershell"], returncode=returncode, stdout=stdout, stderr=""
    )


def _win32_listing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Get ``list_processes`` past its win32/PATH guards on any platform."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(shutil, "which", lambda name: name)


def test_self_process_chain_rejects_recycled_parent() -> None:
    """300's recorded parent (200) exited and its pid was recycled by a
    process created *after* 300. A real parent always predates its child, so
    the walk must stop at 300 -- the stranger and everything above it are
    not ancestors."""
    processes = [
        _proc(100, 1, "a.exe", "a", created=1_000.0),
        _proc(200, 100, "b.exe", "b", created=9_000.0),  # younger than its "child"
        _proc(300, 200, "c.exe", "c", created=5_000.0),
    ]

    assert self_process_chain(300, processes) == frozenset({300})


def test_self_process_chain_returns_full_chain_with_consistent_stamps() -> None:
    """Control for the recycled-parent rule: with creation stamps present and
    every parent older than its child, the whole chain is still returned --
    including the final ancestor (1) that has no row of its own."""
    processes = [
        _proc(100, 1, "a.exe", "a", created=1_000.0),
        _proc(200, 100, "b.exe", "b", created=3_000.0),
        _proc(300, 200, "c.exe", "c", created=5_000.0),
    ]

    assert self_process_chain(300, processes) == frozenset({300, 200, 100, 1})


def test_recycled_parent_stranger_still_matches_pattern() -> None:
    """End-to-end through ``assert_quiescent``: an unrelated process holding
    a recycled parent pid must NOT land in the exclusion set, so its
    matching command line is still reported -- the false-quiet shape the
    issue reports (the exclusion hid it)."""
    stranger = _proc(
        200,
        100,
        "charlie.exe",
        "charlie.exe fleet supervise --repo x",
        created=9_000.0,  # younger than its "child" -- a recycled holder
    )
    processes = [
        _proc(100, 1, "a.exe", "a", created=1_000.0),
        stranger,
        _proc(300, 200, "python.exe", "python quiesce_cli.py", created=5_000.0),
    ]

    report = assert_quiescent(patterns=[_SUPERVISE_PATTERN], processes=processes, self_pid=300)

    assert report.ok is False
    assert report.matched == (stranger,)
    assert report.excluded_pids == frozenset({300})


def test_list_processes_populates_created_from_ppid_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The UTC-sourced creation stamps ride on the CIM rows: the overlay is
    pid-keyed, and a pid the chain probe did not cover keeps ``created=None``.

    Issue #2376: the probe is scoped to ``os.getpid()``'s own ancestor chain
    (``win32_ancestor_rows``), not a host-wide ``process_iter`` pass --
    ``created`` is only ever consulted on that one chain, and stamping all
    ~450 host processes cost about a second per listing on this host.

    The leaf name predates the #2376 re-scope and is kept verbatim: the
    collect-only gate (issue #1538) fails a required check on any leaf-name
    removal, rename included, absent the operator-applied
    ``collect-gate-exempt`` label.
    """
    _win32_listing_env(monkeypatch)
    sample = [
        {
            "ProcessId": 111,
            "ParentProcessId": 1,
            "Name": "a.exe",
            "CommandLine": "a.exe",
        },
        {
            "ProcessId": 222,
            "ParentProcessId": 111,
            "Name": "b.exe",
            "CommandLine": "b.exe",
        },
    ]
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_completed(json.dumps(sample)))
    calls: list[int] = []

    def _chain_rows(pid: int, **_kwargs: object) -> dict[int, ProcRow]:
        calls.append(pid)
        return {111: ProcRow(1, created=1234.5)}

    monkeypatch.setattr("charlie_work.quiesce.win32_ancestor_rows", _chain_rows)

    processes, error = list_processes()

    assert error is None
    assert calls == [os.getpid()]
    assert [p.created for p in processes] == [1234.5, None]


def test_list_processes_created_overlay_failure_degrades_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed creation-time snapshot must not fail the listing: rows keep
    ``created=None`` (unknown never disqualifies a parent link) and the
    degrade is logged -- an inert recycled-parent guard must be visible."""
    _win32_listing_env(monkeypatch)
    sample = [{"ProcessId": 111, "ParentProcessId": 1, "Name": "a.exe", "CommandLine": "a"}]
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _fake_completed(json.dumps(sample)))
    # An empty row map is the chain probe's failure contract.
    monkeypatch.setattr("charlie_work.quiesce.win32_ancestor_rows", lambda _pid, **_k: {})

    with caplog.at_level(logging.WARNING, logger="charlie_work.quiesce"):
        processes, error = list_processes()

    assert error is None
    assert [p.created for p in processes] == [None]
    assert any("creation-time" in r.getMessage() for r in caplog.records)
