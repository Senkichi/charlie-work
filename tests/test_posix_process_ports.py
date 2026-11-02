"""POSIX ports for the Mac host (spec C2, CW-3): ``ps`` listing and the launchd probe.

macOS has no ``/proc`` and no ``schtasks``. Each test forces the macOS branch by
replacing the module's own ``os``/``sys`` binding, so the suite runs the same on
the Windows CI shards as on a Mac.

"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _src_ast import source_text

import charlie_work.fleet_dispatch as fleet_dispatch
import charlie_work.host_load as host_load
import charlie_work.launchd_probe as launchd_probe
import charlie_work.orphan_sweep as orphan_sweep
import charlie_work.posix_processes as posix_processes
import charlie_work.process_utils as process_utils
from charlie_work.quiesce import ProcessInfo
from charlie_work.subprocess_runner import RunResult

_EFFECTS_SESSIONS = (
    Path(__file__).resolve().parent.parent
    / "src/charlie_work/dead_worker_sweep/effects_sessions.py"
)
_WORKTREE = "/Users/fleet/repos/app/.var/charlie-work/worktrees/issue-41"
_NODE = f"/opt/homebrew/bin/node {_WORKTREE}/server.js --port 3000"
_PS_OUT = (
    "    1     0 /sbin/launchd\n"
    "  501     1 /usr/bin/python3 -m charlie_work fleet supervise-loop\n"
    f"  777   501 {_NODE}\n"
    "  778     1 \n"
    "garbage line\n"
)
_LABEL = "com.senkichi.hosts.charlie-fleet"
# Bound at import, before conftest's autouse stub replaces the module attribute
# for every test (the test_host_load.py idiom).
_REAL_LIST_HOST_PROCESSES = host_load.list_host_processes


def _rows() -> tuple[ProcessInfo, ...]:
    return posix_processes.parse_ps(_PS_OUT)


class _Runner:
    """Records each command and answers from ``replies`` in order."""

    def __init__(self, *replies: RunResult) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[str], Any, int]] = []

    def __call__(self, command: list[str], *, cwd: Any, timeout_seconds: int) -> RunResult:
        self.calls.append((command, cwd, timeout_seconds))
        return self.replies.pop(0)


def _ok(stdout: str = "") -> RunResult:
    return RunResult(returncode=0, stdout=stdout, stderr="")


def _fail() -> RunResult:
    return RunResult(returncode=1, stdout="", stderr="nope", error="command exited 1")


def _disabled(label: str, value: str) -> str:
    return f'disabled services = {{\n\t"{label}" => {value}\n}}\n'


def test_parse_ps_reads_pid_ppid_name_and_full_command_line() -> None:
    rows = _rows()
    assert [(r.pid, r.ppid, r.name) for r in rows] == [
        (1, 0, "launchd"),
        (501, 1, "python3"),
        (777, 501, "node"),
        (778, 1, ""),
    ]
    assert rows[2].command_line == _NODE


def test_list_processes_ps_runs_the_portable_ps_command() -> None:
    runner = _Runner(_ok(_PS_OUT))
    rows, error = posix_processes.list_processes_ps(runner)
    assert error is None
    assert len(rows) == 4
    assert runner.calls == [
        (["ps", "-axww", "-o", "pid=", "-o", "ppid=", "-o", "command="], None, 10)
    ]


def test_list_processes_ps_returns_the_error_instead_of_raising() -> None:
    rows, error = posix_processes.list_processes_ps(_Runner(_fail()))
    assert rows == ()
    assert error


def _posix_sweep(monkeypatch: pytest.MonkeyPatch, calls: list[int]) -> None:
    monkeypatch.setattr(orphan_sweep, "os", SimpleNamespace(name="posix"))

    def fake_list() -> tuple[tuple[ProcessInfo, ...], None]:
        calls.append(1)
        return _rows(), None

    monkeypatch.setattr(posix_processes, "list_processes_ps", fake_list)


def test_posix_sweep_finds_processes_naming_the_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    _posix_sweep(monkeypatch, calls)
    assert orphan_sweep.sweep_orphan_processes(_WORKTREE) == [
        {"pid": 777, "name": "node", "command_line": _NODE}
    ]


def test_posix_sweep_still_refuses_a_degenerate_needle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    _posix_sweep(monkeypatch, calls)
    assert orphan_sweep.sweep_orphan_processes("/usr") == []
    assert orphan_sweep.sweep_orphan_processes("/Users/*") == []
    assert calls == []


def test_dead_session_sweep_has_no_windows_only_guard() -> None:
    assert "os.name" not in source_text(_EFFECTS_SESSIONS)


def test_ancestor_guard_on_macos_walks_the_psutil_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orphan_sweep, "os", SimpleNamespace(name="posix", getpid=lambda: 300))
    monkeypatch.setattr(orphan_sweep, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(
        orphan_sweep,
        "_win32_ancestor_rows",
        lambda _pid, **_k: {300: orphan_sweep._ProcRow(200), 200: orphan_sweep._ProcRow(1)},
    )
    assert orphan_sweep._self_ancestor_pids() == frozenset({300, 200, 1})


def test_child_pids_without_procfs_come_from_ps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_utils, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(process_utils, "Path", lambda _p: SimpleNamespace(exists=lambda: False))
    monkeypatch.setattr(process_utils, "list_processes_ps", lambda: (_rows(), None))
    assert process_utils._enumerate_child_pids(501) == [777]
    assert process_utils._enumerate_child_pids(1) == [501, 778]


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        if pid == 404:
            raise process_utils.psutil.NoSuchProcess(pid)
        self.pid = pid

    def create_time(self) -> float:
        return 1_760_000_000.25


def test_start_time_on_macos_comes_from_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_utils, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(process_utils.psutil, "Process", _FakeProcess)
    assert process_utils.get_process_start_time(777) == 1_760_000_000.25
    assert process_utils.get_process_start_time(404) is None


def test_host_load_on_macos_lists_processes_with_ps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_load, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(posix_processes, "list_processes_ps", lambda: (_rows(), None))
    rows, error = _REAL_LIST_HOST_PROCESSES()
    assert error is None
    assert [r.pid for r in rows] == [1, 501, 777, 778]


def test_launchd_probe_is_unknown_without_a_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(launchd_probe.LAUNCHD_LABEL_ENV, raising=False)
    runner = _Runner()
    armed, detail = launchd_probe.probe_launchd_job(run_command=runner)
    assert armed is None
    assert "CHARLIE_WORK_LAUNCHD_LABEL" in detail
    assert runner.calls == []


@pytest.mark.parametrize(
    ("replies", "expected"),
    [
        ((_fail(),), None),
        ((_ok(), _fail()), None),
        ((_ok(), _ok(_disabled(_LABEL, "disabled"))), False),
        ((_ok(), _ok(_disabled(_LABEL, "true"))), False),
        ((_ok(), _ok(_disabled(_LABEL, "enabled"))), True),
        ((_ok(), _ok(_disabled("com.other", "disabled"))), True),
    ],
)
def test_launchd_probe_reads_loaded_and_disabled_state(
    monkeypatch: pytest.MonkeyPatch, replies: tuple[RunResult, ...], expected: bool | None
) -> None:
    monkeypatch.setenv(launchd_probe.LAUNCHD_LABEL_ENV, _LABEL)
    runner = _Runner(*replies)
    armed, _detail = launchd_probe.probe_launchd_job(run_command=runner)
    assert armed is expected
    assert runner.calls[0][0] == ["launchctl", "print", f"system/{_LABEL}"]
    if len(runner.calls) > 1:
        assert runner.calls[1][0] == ["launchctl", "print-disabled", "system"]


def test_fleet_watchdog_probe_on_macos_uses_launchd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fleet_dispatch, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setenv(launchd_probe.LAUNCHD_LABEL_ENV, _LABEL)
    runner = _Runner(_ok(), _ok("disabled services = {\n}\n"))
    probe = fleet_dispatch.probe_fleet_watchdog(run_command=runner)
    assert probe.armed is True
    assert probe.detail == f"launchd job system/{_LABEL} is loaded and enabled"
