"""One subprocess runner for every adapter and cross-family invocation.

Centralizes the Windows-safe capture contract: text mode with explicit UTF-8
decoding and ``errors="replace"`` (never the cp1252 default), and bytes-safe
handling of ``TimeoutExpired`` partial output. Callers get a plain result and
never an encoding crash.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
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


_NPM_SHIM_EXE_PATTERN = re.compile(r'"%dp0%\\([^"]+\.exe)"', re.IGNORECASE)


def resolve_cli_binary(name: str) -> str:
    """Resolve an npm-installed CLI tool name to its underlying ``.exe`` on Windows.

    npm-published shims (``claude.CMD``, ``gemini.CMD``, ...) are batch-file
    wrappers around the real ``.exe``. Two distinct problems follow from that,
    and this is the single point of enforcement that fixes both:

    1. ``subprocess.Popen``/``subprocess.run`` with ``shell=False`` and a bare
       name (e.g. ``"claude"``) goes straight to Windows' ``CreateProcessW``,
       which does **not** perform the ``PATHEXT``-based extension search that
       ``cmd.exe`` does — so it cannot find ``claude.CMD`` at all and fails
       with ``OSError: [WinError 2] The system cannot find the file
       specified``, even though ``claude`` is on ``PATH`` and works fine when
       typed at a shell prompt (see charlie-work issue #487).
    2. Pre-resolving via ``shutil.which()`` alone (which *does* honor
       ``PATHEXT`` and returns the full ``.CMD`` path) fixes (1) but trades it
       for a subtler bug: ``CreateProcessW`` on a ``.CMD`` file implicitly
       routes the child through ``cmd.exe``, whose argv parser uses
       caret-escaping rather than the C-runtime backslash-escaping that
       Python's ``subprocess.list2cmdline`` emits. A literal ``|`` in an
       argument value (e.g. a prompt containing ``"small|mid|large"``) is
       then interpreted as a ``cmd.exe`` pipeline separator, breaking the
       invocation.

    The real fix is to skip ``cmd.exe`` entirely: parse the ``.CMD`` shim to
    find the underlying ``.exe`` it wraps (the npm shim pattern is stable —
    ``"%dp0%\\node_modules\\<pkg>\\bin\\<name>.exe" %*``) and use that ``.exe``
    path directly as ``argv[0]``. ``CreateProcessW`` then invokes it without
    any shell interposed. Ported from a sibling repo's
    ``claude_client._resolve_cli_binary`` (commit fe0cdde7), which fixed the
    identical class of bug for the same npm-installed CLI shims.

    On POSIX, or when ``shutil.which`` resolves to something other than a
    ``.cmd``/``.bat`` file, the resolved path is returned unchanged (already
    directly executable). If the binary cannot be found on ``PATH`` at all,
    or the shim cannot be parsed/its target ``.exe`` does not exist, the
    original (or ``shutil.which``-resolved) value is returned unchanged — a
    deliberately conservative fallback that preserves the caller's existing
    "binary not found" error handling rather than masking a missing install.

    Args:
        name: CLI tool name or path, e.g. ``"claude"``.

    Returns:
        A path safe to pass as ``argv[0]`` to ``subprocess.Popen``/``.run``
        with ``shell=False``.
    """
    path = shutil.which(name)
    if path is None:
        return name
    if os.name != "nt" or not path.lower().endswith((".cmd", ".bat")):
        return path
    try:
        shim_text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return path
    match = _NPM_SHIM_EXE_PATTERN.search(shim_text)
    if match is None:
        return path
    exe_path = Path(os.path.normpath(Path(path).parent / match.group(1)))
    return str(exe_path) if exe_path.exists() else path


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


def command_failure_message(command: list[str], result: RunResult, fallback: str) -> str:
    """Build a diagnostic failure message that prefers the specific stderr.

    ``run_captured`` (the real production runner) always populates
    ``RunResult.error`` on any non-zero exit with a generic
    ``"command exited {code}"``. Under a naive ``result.error or result.stderr``
    fallback chain that generic string is always truthy and permanently
    shadows ``.stderr`` in production, even when stderr carries the actual
    diagnostic -- e.g. ``git pull --ff-only`` names the exact colliding paths
    on a dirty-tree collision, and that name was unreachable (issue #817
    item 3). Prefer ``.stderr``; fall back to ``.error``, then to
    ``fallback``, only when stderr is empty. The failing argv is always
    included so the message is actionable without cross-referencing a log.

    This is the repo's single implementation of that ordering -- every
    caller building a failure message from a ``RunResult`` should route
    through here rather than re-deriving its own ``or`` chain, which is
    exactly how the shadowing bug above went unnoticed at a second call site
    (``main_ci_reclaim.py``, issue #1777) after it was already fixed here.
    """
    detail = (result.stderr or "").strip() or (result.error or "").strip() or fallback
    return f"{' '.join(command)}: {detail}"


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value) if value else ""


# Layered onto every child's env (issue #2139). ``GIT_ASK_YESNO=false`` makes
# Git for Windows answer its "Unlink of file ... failed. Should I try again?
# (y/n)" retry prompt with "no" instead of waiting on a console;
# ``GIT_TERMINAL_PROMPT=0`` does the same for credential prompts. A caller's
# ``extra_env`` still wins if it sets either key.
_NON_INTERACTIVE_ENV: dict[str, str] = {"GIT_ASK_YESNO": "false", "GIT_TERMINAL_PROMPT": "0"}

# Bound on draining the pipes after the process tree is killed. If a
# descendant the tree kill could not reach still holds the inherited pipe
# handles, the drain gives up after this long and the pipes are closed.
_POST_KILL_DRAIN_SECONDS = 5


def _kill_tree_and_drain(proc: subprocess.Popen[str]) -> tuple[str, str]:
    """Kill ``proc``'s whole process tree, then drain its pipes with a bound.

    ``subprocess.run(timeout=...)`` is not a bound on Windows: it kills only the
    direct child and then calls ``communicate()`` with no timeout, which blocks
    for as long as any surviving descendant (e.g. the real git under the
    ``git.exe`` launcher) holds the inherited pipe handles (issue #2139). Never
    raises and never blocks unbounded; returns whatever output was drained.
    """
    # Lazy import: ``process_utils`` imports this module at module scope.
    from .process_utils import kill_process_tree

    try:
        kill_process_tree(proc.pid)
    except Exception:  # noqa: BLE001 - best-effort; the direct kill below still runs
        pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        stdout, stderr = proc.communicate(timeout=_POST_KILL_DRAIN_SECONDS)
        return stdout or "", stderr or ""
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        partial = (
            (_as_text(exc.stdout), _as_text(exc.stderr))
            if isinstance(exc, subprocess.TimeoutExpired)
            else ("", "")
        )
    # A survivor still holds the pipes. Closing them synchronously would block
    # too: ``communicate``'s reader thread holds each buffered reader's lock
    # while blocked in ``ReadFile``, and ``close()`` waits for that lock.
    # Hand the close to a daemon thread so it completes whenever the survivor
    # finally exits, without ever stalling the caller.
    threading.Thread(
        target=_close_pipes, args=(proc,), daemon=True, name=f"run-captured-close-{proc.pid}"
    ).start()
    return partial


def _close_pipes(proc: subprocess.Popen[str]) -> None:
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if pipe is not None:
                pipe.close()
        except (OSError, ValueError):
            pass


def run_captured(
    command: list[str] | str,
    *,
    cwd: Path | str | None,
    timeout_seconds: int,
    shell: bool = False,
    stdin: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> RunResult:
    """Run ``command`` and capture output. Never raises for runtime failures —
    timeouts, missing binaries, and non-zero exits all come back as a result.

    ``extra_env``, when given, is layered on top of (never in place of) the
    current process environment -- a full replacement would drop ``PATH`` and
    break the child's ability to even find ``git``. ``_NON_INTERACTIVE_ENV``
    is always layered between the two, so ``extra_env`` can still override it.

    ``timeout_seconds`` bounds the whole call, descendants included: on
    timeout the process *tree* is killed and the pipe drain is itself bounded
    (issue #2139). ``stdin=None`` gives the child ``DEVNULL`` -- never the
    parent's stdin -- so an interactive prompt reads EOF instead of waiting.
    ``cwd=None`` inherits the current directory -- it must reach ``Popen`` as
    ``None``, never as ``str(None)`` (issue #2450).
    """
    env = {**os.environ, **_NON_INTERACTIVE_ENV, **(extra_env or {})}
    # POSIX: a new session puts the child in its own process group so
    # ``kill_process_tree``'s ``killpg`` can reach its descendants (it refuses
    # the caller's own group), and drops the controlling terminal.
    session_kwargs = {"start_new_session": True} if os.name != "nt" else {}
    try:
        proc = subprocess.Popen(
            command,
            cwd=None if cwd is None else str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL if stdin is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
            env=env,
            **session_kwargs,
            **hidden_console_kwargs(),
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return RunResult(returncode=None, stdout="", stderr="", error=str(exc))
    try:
        stdout, stderr = proc.communicate(input=stdin, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        stdout, stderr = _kill_tree_and_drain(proc)
        return RunResult(
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            error=f"command timed out after {timeout_seconds}s",
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        _kill_tree_and_drain(proc)
        return RunResult(returncode=None, stdout="", stderr="", error=str(exc))
    returncode = proc.returncode
    return RunResult(
        returncode=returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        error=None if returncode == 0 else f"command exited {returncode}",
    )
