"""Worker process launch: the single point for creationflags composition.

Extracted from ``process_utils`` (which re-exports both names) so the launch
policy lives in one small module.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from .subprocess_runner import hidden_console_kwargs


class CpuPriority(Enum):
    """CPU scheduling class for a process launched through ``popen_worker``.

    Every launch site must choose one explicitly (the parameter has no
    default). Agent sessions -- workers and reviewers -- launch at
    ``BELOW_NORMAL``; on Windows a ``BELOW_NORMAL`` process's children inherit
    that class, so every pytest suite an agent runs is deprioritized without
    any per-suite plumbing. The merge-gate suite runner launches at
    ``NORMAL``: it is the one suite every merge waits on, and under host
    oversubscription it must win the CPU rather than share it evenly with
    exploratory worker runs (mdls PR #41's gate ran ~2x its 25-35 min
    baseline while starved by worker suites, 2026-09-30).

    POSIX has no inheritable priority class here; the value is ignored there.
    """

    NORMAL = "normal"
    BELOW_NORMAL = "below_normal"


# NORMAL is set explicitly, never left to inheritance: a child with no class
# flag inherits BELOW_NORMAL from a below-normal parent, so "no flag" would
# silently demote the gate whenever its launcher ran deprioritized.
_PRIORITY_CREATIONFLAGS: dict[CpuPriority, int] = {
    CpuPriority.NORMAL: getattr(subprocess, "NORMAL_PRIORITY_CLASS", 0),
    CpuPriority.BELOW_NORMAL: getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
}


def popen_worker(
    args: Sequence[str] | str,
    *,
    priority: CpuPriority,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    **popen_kwargs: Any,
) -> subprocess.Popen[Any]:
    """Launch a worker process as the single point for creationflags/process-group composition.

    Injects the worker hidden-console policy into the ``subprocess.Popen`` call:
    - ``creationflags`` and ``startupinfo`` are composed through
      ``hidden_console_kwargs()`` so ``CREATE_NEW_CONSOLE`` is combined with
      ``CREATE_NEW_PROCESS_GROUP`` on Windows and the console is hidden via
      ``STARTF_USESHOWWINDOW`` / ``SW_HIDE``. On POSIX it is a no-op.
    - ``start_new_session`` defaults to ``True`` on POSIX and is omitted on
      Windows; callers may override by passing it explicitly.
    - ``priority`` (required) sets the process's CPU priority class; see
      ``CpuPriority``.

    All other ``Popen`` keyword arguments are passed through. The helper returns
    the ``Popen`` object immediately and never waits or communicates.
    """
    if cwd is not None:
        popen_kwargs["cwd"] = cwd
    if env is not None:
        popen_kwargs["env"] = env

    if "start_new_session" not in popen_kwargs and os.name != "nt":
        popen_kwargs["start_new_session"] = True

    extra_flags = popen_kwargs.pop("creationflags", 0) | _PRIORITY_CREATIONFLAGS[priority]
    process_group_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    popen_kwargs.update(hidden_console_kwargs(extra_flags | process_group_flag))

    return subprocess.Popen(args, **popen_kwargs)
