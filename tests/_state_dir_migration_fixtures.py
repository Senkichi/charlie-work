"""Shared module constants for the state-DIR migration test siblings.

Hoisted verbatim out of ``tests/test_state_dir_migration.py`` (issue #1566,
Track-1 seam split) when that module was split into seam-named siblings --
the ``tests/_*.py`` hoisted-fixture convention is the sanctioned import
target for shared test helpers (see
``tests/test_zero_cross_test_import_guard.py``).

The ``state_dir`` prefix in the sibling names (and in this module) is
deliberate: ``test_state_migration`` is already taken by a pre-existing,
unrelated suite covering ``charlie_work.state`` schema round-tripping
(PRs #306/#321/#531). See ``src/charlie_work/state_migration.py`` for the
planner these tests exercise.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path("C:/repos/job-cannon")
SRC_ROOT = Path("C:/repos/job-cannon/.var/devin-orchestrator")
DST_ROOT = Path("C:/repos/job-cannon/.var/charlie-work")
