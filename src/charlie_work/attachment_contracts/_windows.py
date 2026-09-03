"""Inlined console-suppression helper for the attachment_contracts subpackage.

This is a verbatim copy of ``charlie_work.subprocess_runner.no_console_window_kwargs``
inlined into the subpackage so the ``charlie-work-attachment-contracts``
distribution has zero intra-repo imports (issue #1544 Stage 1).  charlie-work
and the wheel share this single in-tree implementation; the
``charlie_work.subprocess_runner`` copy remains for the rest of charlie-work.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, TypedDict

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0)


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
