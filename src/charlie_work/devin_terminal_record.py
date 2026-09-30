"""Devin-shell's side of the terminal-record contract (issue #2052).

Since #2049 ``worker_fate.resolve_fate`` consumes a per-adapter terminal
record, and #2052 made the devin launcher run the same
``process_utils.start_terminal_status_watcher`` claude-code's launcher runs
(#773): a daemon thread that polls ``Popen.poll()`` (never ``wait()``/
``communicate()`` — CLAUDE.md's non-blocking-adapter invariant) and persists
``issue-<n>.devin.terminal.json`` — exit code, duration, and a copy of the
worktree's ``.worker-outcome.json`` — when the spawned process exits.

Extracted from ``devin_shell.py`` (PR #2069 rework, file-size ratchet #1442).
What stays here is the profile gate around the watcher start: whether the
record is written is read off the harness's ``AdapterFateProfile`` —
``writes_terminal_record`` — never an ``adapter_kind ==`` branch.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

from charlie_work.process_utils import (
    start_terminal_status_watcher,
    worker_terminal_status_path,
)


def maybe_start_terminal_status_watcher(
    process: subprocess.Popen[Any],
    sessions_dir: Path,
    issue_number: int,
    *,
    worktree_path: Path | None,
) -> threading.Thread | None:
    """Start the terminal-status watcher for a freshly spawned devin process.

    Consults ``worker_fate.profile_for("devin").writes_terminal_record``: the
    declaration lives on the profile so flipping it off (or a harness without
    the capability) suppresses the record — the mutation-gate seam the
    terminal-record tests pin. The ``worker_fate`` import is lazy (function
    body): the profile table reaches back into ``devin_shell``, so a
    top-level import would cycle. A profile lookup failure still writes the
    record — a registry hiccup must never suppress durable exit evidence on
    a live process.

    ``worktree_path`` is ``None`` for review launches (matching claude_code's
    #1354 handling): a review checkout holds no ``.worker-outcome.json`` —
    only the exit code matters to the review-verdict reaper — so the caller
    resolves ``None if review else worktree.path``. Worker launches pass the
    worktree so the watcher embeds the outcome file (#935 semantics).

    Returns the watcher thread, or ``None`` when the profile gate suppresses
    the start. Launch returns immediately either way: the thread polls
    ``Popen.poll()`` on a daemon thread and never joins.
    """
    try:
        from . import worker_fate  # lazy: profile table reaches back into devin_shell

        profile = worker_fate.profile_for("devin")
        writes_terminal_record = profile is None or profile.writes_terminal_record
    except Exception:
        writes_terminal_record = True
    if not writes_terminal_record:
        return None
    return start_terminal_status_watcher(
        process,
        worker_terminal_status_path(sessions_dir, issue_number, "devin"),
        worktree_path=worktree_path,
    )
