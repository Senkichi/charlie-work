"""Shared helpers for the ``test_worker.py`` family of test modules.

Hoisted out of ``test_worker.py`` under the #1574 Track-1 shoulder split:
``_wg`` is consumed by tests in both ``test_worker.py`` and the new
``tests/test_worker_stalled_sessions.py`` sibling, and test modules may not
import from each other, so the shared ``WriteGate`` builder lives here.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.write_gate import WriteGate


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")
