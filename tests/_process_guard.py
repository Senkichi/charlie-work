"""Test-session process containment for the launch-path suite (issue #1851).

Tests that exercise the non-blocking worker launch paths spawn real child
processes — venv ``python.exe`` stand-ins, ``python -c ...`` stubs, fake
``claude`` scripts — and the launch functions return immediately by design
(CLAUDE.md: "Adapters must not block on worker completion"), so nothing ever
waited on them. When a worker's test run was stopped mid-run, every in-flight
child was orphaned, and on Windows the orphaned venv launchers busy-spun for
hours until killed by PID.

This module supplies the machinery ``tests/conftest.py`` wires in:

* ``enter_kill_on_close_job`` — the structural fix. On Windows it places the
  running pytest process in a Job Object with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` so every descendant — present and
  future, at any depth — dies with pytest however it exits. Off Windows, and
  when the process is already inside a job that forbids nesting, it is a
  no-op that only logs.
* Leak detection and reaping — ``descendant_snapshot`` /
  ``reap_leaked_descendants`` power the autouse teardown guard that fails a
  test leaving a live child behind; ``reap_pid`` / ``wrap_launchers`` let the
  suite wait on or kill the pid each returned launch record carries.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import psutil
import pytest

logger = logging.getLogger(__name__)

# Bound for how long teardown waits for a just-launched child to exit on its
# own before declaring it leaked. Fake worker scripts finish in well under a
# second; the grace exists so a fast child mid-exit at teardown time is not
# misreported as a leak. Only genuinely stuck children pay the full wait.
LEAK_GRACE_S = 5.0

_POLL_INTERVAL_S = 0.05

# Kept so the Job Object handle is never closed early: KILL_ON_JOB_CLOSE fires
# when the LAST handle to the job closes — intended to be process exit, when
# the kernel closes it for us.
_job_handle: int | None = None

# Captured at import: tests may monkeypatch ``os.getpid`` mid-test (the
# self-exemption tests do), and ``psutil.Process()`` with no argument resolves
# the current pid through it — so it must never be consulted at fixture time.
_THIS_PID = os.getpid()


def _this_process() -> psutil.Process:
    return psutil.Process(_THIS_PID)


def descendant_snapshot() -> dict[int, float]:
    """Map ``pid -> create_time`` for every live descendant of this process.

    Recursive: a grandchild (e.g. the real interpreter under a venv launcher)
    is covered just like a direct child. The create_time value distinguishes
    a recycled pid from the process that was alive at snapshot time.
    """
    snapshot: dict[int, float] = {}
    for child in _this_process().children(recursive=True):
        try:
            snapshot[child.pid] = child.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return snapshot


def leaked_descendants(before: dict[int, float]) -> list[psutil.Process]:
    """Live descendants of this process that are not in ``before``.

    Zombies are excluded: an exited-but-unreaped child is already dead and
    cannot outlive the session.
    """
    leaked: list[psutil.Process] = []
    for child in _this_process().children(recursive=True):
        try:
            if before.get(child.pid) == child.create_time():
                continue
            if child.status() == psutil.STATUS_ZOMBIE:
                continue
            leaked.append(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return leaked


def _describe(proc: psutil.Process) -> str:
    try:
        cmdline = proc.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        cmdline = []
    if not cmdline:
        try:
            cmdline = [proc.name()]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cmdline = ["<unknown>"]
    return f"pid={proc.pid} cmdline={' '.join(cmdline)!r}"


def _kill_tree(proc: psutil.Process) -> None:
    """Kill ``proc`` and every descendant it still has, then reap them all."""
    try:
        tree = proc.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        tree = []
    tree.append(proc)
    for member in tree:
        try:
            member.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for member in tree:
        try:
            member.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue


def reap_pid(pid: int | None) -> None:
    """Kill ``pid`` and its descendants — but only while ``pid`` is a live
    descendant of this process.

    Launch records in delegation tests carry fabricated pids (``pid=4242`` on
    a stubbed record is just a number, and could collide with a real,
    unrelated host process). The descendant check is what makes it safe to
    offer every returned pid here unconditionally.
    """
    if not pid or pid <= 0:
        return
    try:
        proc = psutil.Process(pid)
        ours = any(child.pid == pid for child in _this_process().children(recursive=True))
    except psutil.NoSuchProcess:
        return
    if ours:
        _kill_tree(proc)


def reap_leaked_descendants(before: dict[int, float], grace: float = LEAK_GRACE_S) -> list[str]:
    """Give new descendants ``grace`` seconds to exit, then kill survivors.

    Returns one report line (``pid=... cmdline=...``) per survivor; an empty
    list means nothing was left running.
    """
    deadline = time.monotonic() + grace
    leaked = leaked_descendants(before)
    while leaked and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)
        leaked = leaked_descendants(before)
    report = [_describe(proc) for proc in leaked]
    for proc in leaked:
        _kill_tree(proc)
    return report


def wrap_launchers(
    monkeypatch: pytest.MonkeyPatch,
    pids: list[int],
    real_launchers: dict[str, Any] | None = None,
) -> int:
    """Rebind every visible binding of each real launch function so every
    returned record's pid lands in ``pids``.

    Launch functions return ``SessionRecord``/``ClaudeWorkerRecord`` values
    whose ``pid`` is the only handle the non-blocking API hands back; the
    caller's teardown can then kill each one via ``reap_pid``.

    The scan covers ``sys.modules`` — not just the test module — because
    dispatch tests reach the launchers through production bindings
    (``workflow.launch_devin_session`` and friends are ``from``-imported
    there). Modules imported mid-test ``from``-bind through the already-
    patched source attribute, so they pick up the wrapper too. Bindings that
    do not point at the real function (a test's own fake, ``None``) are left
    alone. Returns the number of bindings wrapped.
    """
    if real_launchers is None:
        from charlie_work.api_worker import launch_api_worker
        from charlie_work.claude_code import launch_claude_worker
        from charlie_work.devin_shell import launch_devin_session

        real_launchers = {
            "launch_api_worker": launch_api_worker,
            "launch_claude_worker": launch_claude_worker,
            "launch_devin_session": launch_devin_session,
        }
    wrapped = 0
    for module in list(sys.modules.values()):
        if module is None:
            continue
        for name, real in real_launchers.items():
            try:
                bound = getattr(module, name, None)
            except Exception:
                continue
            if bound is not real:
                continue
            monkeypatch.setattr(module, name, _pid_collecting_wrapper(real, pids))
            wrapped += 1
    return wrapped


def _pid_collecting_wrapper(real: Any, pids: list[int]) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        record = real(*args, **kwargs)
        pid = getattr(record, "pid", None)
        if pid is not None:
            pids.append(pid)
        return record

    return wrapper


def enter_kill_on_close_job() -> bool:
    """Place this process in a kill-on-close Job Object. Windows only.

    Returns True when the process was assigned to a fresh job. Returns False —
    logging a warning — off Windows, or when creation/assignment fails (the
    usual failure is ERROR_ACCESS_DENIED because the process is already inside
    a job that forbids nesting, e.g. a second call, or a CI runner that jobs
    its test step). A failed setup leaves the suite's other two mechanisms
    (per-test reaping, the teardown guard) in place, so it is tolerated
    rather than raised.
    """
    if os.name != "nt":
        return False

    import ctypes
    from ctypes import wintypes

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9

    kernel32 = ctypes.windll.kernel32
    # Signatures are mandatory, not decoration: ctypes defaults to c_int for
    # every return/argument, which truncates 64-bit HANDLEs. In particular
    # GetCurrentProcess() yields the -1 pseudo-handle; passed without a HANDLE
    # restype it arrives as 0x00000000FFFFFFFF and AssignProcessToJobObject
    # fails with ERROR_INVALID_HANDLE.
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        logger.warning(
            "CreateJobObjectW failed (error %s); kill-on-close containment is off",
            ctypes.GetLastError(),
        )
        return False

    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job, _JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
    ):
        err = ctypes.GetLastError()
        kernel32.CloseHandle(job)
        logger.warning(
            "SetInformationJobObject failed (error %s); kill-on-close containment is off",
            err,
        )
        return False

    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        err = ctypes.GetLastError()
        kernel32.CloseHandle(job)
        logger.warning(
            "AssignProcessToJobObject failed (error %s); the process is likely "
            "already inside a job that forbids nesting — continuing without "
            "kill-on-close containment",
            err,
        )
        return False

    global _job_handle
    _job_handle = job
    return True
