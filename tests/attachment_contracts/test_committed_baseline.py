"""Regression guard: the committed repo-root ``.attachment-budgets.json``
must itself satisfy the ``loads()`` contract.

PR #1739 round-3 finding: a branch merge produced two byte-identical
``tests/test_backlog_reachability.py::module`` entries, and nothing in the
suite exercised the committed document -- ``loads()`` raised ``TamperError``,
disabling the bump/G4 check for the whole PR.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.attachment_contracts.baseline import BASELINE_FILENAME, loads

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_committed_baseline_file_loads() -> None:
    # loads() enforces the schema, bump validation (G4), and the unique
    # (kind, file, identity) key invariant -- it raises TamperError on any
    # duplicate entry, so a passing load is the whole assertion.
    document = loads((_REPO_ROOT / BASELINE_FILENAME).read_text(encoding="utf-8"))
    assert document["entries"], "committed baseline must carry at least one entry"
