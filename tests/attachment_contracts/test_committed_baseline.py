"""Regression guard: the committed repo-root ``.attachment-budgets/``
directory must itself satisfy the ``load_dir()`` contract.

PR #1739 round-3 finding: a branch merge produced two byte-identical
``tests/test_backlog_reachability.py::module`` entries, and nothing in the
suite exercised the committed document -- ``loads()`` raised ``TamperError``,
disabling the bump/G4 check for the whole PR. Issue #1839 moved the store to
the per-entry directory; the guard now exercises that layout end-to-end,
including the path/content key check (an entry file's name must match the
``(kind, file, identity)`` its content declares).
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.attachment_contracts import baseline_dir
from charlie_work.attachment_contracts.baseline import (
    BASELINE_DIRNAME,
    BASELINE_FILENAME,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_committed_baseline_directory_loads() -> None:
    # load_dir() enforces the schema, bump validation (G4), the unique
    # (kind, file, identity) key invariant, and the entry-filename/content
    # match -- it raises TamperError on any violation, so a passing load is
    # the whole assertion.
    document = baseline_dir.load_dir(_REPO_ROOT / BASELINE_DIRNAME)
    assert document["entries"], "committed baseline must carry at least one entry"


def test_committed_baseline_is_directory_not_file() -> None:
    """Issue #1839: the shared-append-point ``.attachment-budgets.json``
    must be gone -- the directory layout is the only committed store."""
    assert (_REPO_ROOT / BASELINE_DIRNAME).is_dir()
    assert not (_REPO_ROOT / BASELINE_FILENAME).exists()
