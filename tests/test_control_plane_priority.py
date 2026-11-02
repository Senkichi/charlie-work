"""The fleet control plane raises itself to NORMAL at startup (issue #2140)."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from charlie_work import control_plane_priority as cpp

NORMAL = 0x20
BELOW_NORMAL = 0x4000

_CHILD = (
    "import ctypes, sys\n"
    "from charlie_work.control_plane_priority import raise_to_normal_priority\n"
    "k = ctypes.windll.kernel32\n"
    "k.GetCurrentProcess.restype = ctypes.c_void_p\n"
    "k.GetPriorityClass.argtypes = [ctypes.c_void_p]\n"
    "h = k.GetCurrentProcess()\n"
    "before = k.GetPriorityClass(h)\n"
    "r = raise_to_normal_priority()\n"
    "print(before, k.GetPriorityClass(h), r.ok)\n"
)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows priority classes")
def test_below_normal_process_is_raised_to_normal() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    before, after, ok = proc.stdout.split()
    assert (int(before), int(after), ok) == (BELOW_NORMAL, NORMAL, "True")


def test_failure_is_nonfatal_and_emits_warning_event(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cpp.os, "name", "nt")

    class _Boom:
        def nice(self, *_a):
            raise psutil.AccessDenied(1)

    fake = SimpleNamespace(
        Process=lambda: _Boom(),
        Error=psutil.Error,
        NORMAL_PRIORITY_CLASS=NORMAL,
    )
    monkeypatch.setattr(cpp, "psutil", fake)
    events = []
    monkeypatch.setattr(cpp, "log_event", lambda *a, **k: events.append((a, k)))
    result = cpp.raise_to_normal_priority(tmp_path / "state.json")
    assert result.ok is False
    assert events and events[0][0][1] == cpp.EVENT_FAILED
    assert events[0][1]["level"] == "warning"


def test_posix_is_noop(monkeypatch) -> None:
    monkeypatch.setattr(cpp.os, "name", "posix")
    assert cpp.raise_to_normal_priority().ok is True


def test_unexpected_exception_type_is_still_nonfatal(monkeypatch, tmp_path) -> None:
    """The 'never raises' contract holds for non-psutil errors too."""
    monkeypatch.setattr(cpp.os, "name", "nt")

    def _boom():
        raise RuntimeError("unexpected")

    monkeypatch.setattr(
        cpp, "psutil", SimpleNamespace(Process=_boom, NORMAL_PRIORITY_CLASS=NORMAL)
    )
    events = []
    monkeypatch.setattr(cpp, "log_event", lambda *a, **k: events.append((a, k)))
    result = cpp.raise_to_normal_priority(tmp_path / "state.json")
    assert result.ok is False
    assert events and events[0][0][1] == cpp.EVENT_FAILED


def test_raise_supervisor_to_normal_resolves_supervisor_state_path(monkeypatch, tmp_path) -> None:
    seen = []
    monkeypatch.setattr(cpp, "raise_to_normal_priority", lambda p=None: seen.append(p))
    monkeypatch.setattr(cpp.layout, "DEFAULT_STATE_DIR", tmp_path)
    cpp.raise_supervisor_to_normal()
    assert seen == [cpp.supervisor_runtime_paths(tmp_path).state_file]
