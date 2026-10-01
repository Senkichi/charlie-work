"""Route ``run_captured``'s spawn through a ``subprocess.run``-shaped fake.

``run_captured`` used to call ``subprocess.run``, so tests faked every
external command it runs by patching ``subprocess.run`` globally. Since issue
#2139 it spawns with ``subprocess.Popen`` + a bounded ``communicate`` (so a
timeout can kill the whole process tree), and a global ``subprocess.run``
patch no longer reaches it: such a test silently runs the *real* command --
``taskkill`` on a fixture pid, a real ``git fetch`` -- or passes vacuously.

``patch_run_captured(fake_run)`` swaps the ``subprocess`` name inside
``charlie_work.subprocess_runner`` only, for a proxy whose ``Popen`` drives
``fake_run`` with the same arguments ``subprocess.run`` used to receive. The
real ``subprocess`` module is untouched, so a fake that passes non-matching
commands through to the real ``subprocess.run`` keeps working.

The fake process reports ``pid=-1``, which ``kill_process_tree`` rejects up
front, so a fake that raises ``TimeoutExpired`` exercises the timeout path
without any real kill being attempted.
"""

from __future__ import annotations

import subprocess
import types
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from charlie_work import subprocess_runner

# Popen-only keyword arguments that ``subprocess.run`` never received from the
# old ``run_captured`` (it used ``capture_output``/``input`` instead).
_POPEN_ONLY_KWARGS = frozenset({"stdin", "stdout", "stderr", "start_new_session"})


class _RunBackedProcess:
    def __init__(self, fake_run: Callable[..., Any], command: Any, kwargs: dict[str, Any]) -> None:
        self._fake_run = fake_run
        self._command = command
        self._kwargs = {k: v for k, v in kwargs.items() if k not in _POPEN_ONLY_KWARGS}
        self._ran = False
        self.pid = -1
        self.returncode: int | None = None
        self.stdin = self.stdout = self.stderr = None

    def communicate(
        self, input: str | None = None, timeout: float | None = None
    ) -> tuple[Any, Any]:
        if self._ran:  # the bounded post-kill drain after a faked timeout
            return "", ""
        self._ran = True
        completed = self._fake_run(
            self._command,
            capture_output=True,
            timeout=timeout,
            check=False,
            input=input,
            **self._kwargs,
        )
        self.returncode = completed.returncode
        return completed.stdout, completed.stderr

    def kill(self) -> None:
        return None


def _proxy_subprocess(fake_run: Callable[..., Any]) -> types.SimpleNamespace:
    proxy = types.SimpleNamespace(**vars(subprocess))
    proxy.Popen = lambda command, **kwargs: _RunBackedProcess(fake_run, command, kwargs)
    return proxy


def patch_run_captured(fake_run: Callable[..., Any]) -> Any:
    """Context manager: ``run_captured`` spawns through ``fake_run``."""
    return patch.object(subprocess_runner, "subprocess", _proxy_subprocess(fake_run))


def monkeypatch_run_captured(monkeypatch: Any, fake_run: Callable[..., Any]) -> None:
    """``monkeypatch`` form of :func:`patch_run_captured`."""
    monkeypatch.setattr(subprocess_runner, "subprocess", _proxy_subprocess(fake_run))
