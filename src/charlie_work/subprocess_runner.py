"""One subprocess runner for every adapter and cross-family invocation.

Centralizes the Windows-safe capture contract: text mode with explicit UTF-8
decoding and ``errors="replace"`` (never the cp1252 default), and bytes-safe
handling of ``TimeoutExpired`` partial output. Callers get a plain result and
never an encoding crash.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0)
_CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
_STARTF_USESHOWWINDOW = getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
_SW_HIDE = getattr(subprocess, "SW_HIDE", 0)


class _SpawnKwargs(TypedDict, total=False):
    creationflags: int
    startupinfo: Any


def no_console_window_kwargs(extra_creationflags: int = 0) -> _SpawnKwargs:
    """Return the ``creationflags`` kwargs that suppress the transient console
    window Windows allocates for a spawned child when the parent has no
    console/window of its own (e.g. an orchestrator running headless or from
    a service). This is the single point of enforcement for
    ``CREATE_NO_WINDOW`` — every ``subprocess.run``/``Popen`` call site in
    this codebase should route its creationflags through this helper rather
    than hard-coding the flag itself.

    ``extra_creationflags`` composes in whatever flags the call site already
    needs (e.g. ``CREATE_NEW_PROCESS_GROUP`` so a launched worker can still be
    killed by process group). ``CREATE_NO_WINDOW`` is never combined with
    ``DETACHED_PROCESS`` — that combination is invalid/contradictory on
    Windows, since ``DETACHED_PROCESS`` already fully detaches the child from
    any console. Callers passing ``DETACHED_PROCESS`` get their flags back
    unchanged.

    On POSIX platforms (no ``creationflags`` concept), returns an empty dict
    so callers can unconditionally do
    ``subprocess.Popen(..., **no_console_window_kwargs(...))`` cross-platform.
    """
    if sys.platform != "win32" or not _CREATE_NO_WINDOW:
        return {}
    if extra_creationflags & _DETACHED_PROCESS:
        return {"creationflags": extra_creationflags}
    return {"creationflags": extra_creationflags | _CREATE_NO_WINDOW}


def hidden_console_kwargs(extra_creationflags: int = 0) -> _SpawnKwargs:
    """Return the ``creationflags`` and ``startupinfo`` kwargs that allocate a
    hidden console for a long-lived worker spawn (e.g. ``devin`` or ``claude``).

    A ``CREATE_NEW_CONSOLE`` process is created with ``STARTF_USESHOWWINDOW`` and
    ``wShowWindow=SW_HIDE``. The worker and all of its console-subsystem
    descendants (pytest, git, gh, bash) inherit that hidden console, so they
    do not each allocate their own visible console window. This is the worker-
    spawn boundary; short-lived leaf spawns continue to use
    ``no_console_window_kwargs()``.

    ``extra_creationflags`` composes in whatever flags the call site already
    needs (e.g. ``CREATE_NEW_PROCESS_GROUP`` so a launched worker can still be
    killed by process group). ``CREATE_NEW_CONSOLE`` is never combined with
    ``CREATE_NO_WINDOW`` or ``DETACHED_PROCESS`` — those console modes are
    mutually exclusive on Windows. A ``ValueError`` is raised if the caller
    tries to combine them.

    On POSIX platforms (no ``creationflags`` concept), returns an empty dict so
    callers can unconditionally do
    ``subprocess.Popen(..., **hidden_console_kwargs(...))`` cross-platform.
    """
    if sys.platform != "win32" or not _CREATE_NEW_CONSOLE:
        return {}
    forbidden = extra_creationflags & (_CREATE_NO_WINDOW | _DETACHED_PROCESS)
    if forbidden:
        raise ValueError(
            f"CREATE_NEW_CONSOLE cannot be combined with mutually-exclusive "
            f"console modes (CREATE_NO_WINDOW, DETACHED_PROCESS); got {forbidden:#x}"
        )
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= _STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = _SW_HIDE
    return {
        "creationflags": extra_creationflags | _CREATE_NEW_CONSOLE,
        "startupinfo": startupinfo,
    }


@dataclass(frozen=True)
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value) if value else ""


def run_captured(
    command: list[str] | str,
    *,
    cwd: Path | str,
    timeout_seconds: int,
    shell: bool = False,
    stdin: str | None = None,
) -> RunResult:
    """Run ``command`` and capture output. Never raises for runtime failures —
    timeouts, missing binaries, and non-zero exits all come back as a result."""
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
            shell=shell,
            check=False,
            input=stdin,
            **no_console_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            returncode=None,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            timed_out=True,
            error=f"command timed out after {timeout_seconds}s",
        )
    except OSError as exc:
        return RunResult(returncode=None, stdout="", stderr="", error=str(exc))
    except subprocess.SubprocessError as exc:
        return RunResult(returncode=None, stdout="", stderr="", error=str(exc))
    return RunResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        error=None if completed.returncode == 0 else f"command exited {completed.returncode}",
    )
