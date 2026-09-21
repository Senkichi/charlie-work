"""Round-trip tests for ``read_worker_outcome``'s ``pr_title``/``pr_body`` fields.

cw#1771 (Option C step 1): the outcome-file contract grew two new optional
string fields so a worker can draft the PR content it would have used for
``gh pr create`` and let the orchestrator open the PR from that draft
verbatim. This module pins the read side of that contract; the write side is
prose in ``prompts/worker_sections/push_pr_outcome.md`` (workers write the
file, never this codebase), and the *consumption* side (preferring the draft
over synthesis) is covered in ``test_salvage_body.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.config import WORKER_OUTCOME_FILENAME
from charlie_work.worktree import read_worker_outcome


def _write_outcome(worktree_path: Path, payload: dict) -> None:
    worktree_path.mkdir(parents=True, exist_ok=True)
    (worktree_path / WORKER_OUTCOME_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def test_read_worker_outcome_round_trips_drafted_pr_title_and_body(tmp_path: Path) -> None:
    payload = {
        "push_succeeded": True,
        "pr_created": False,
        "pr_title": "fix(worktree): handle stale lock",
        "pr_body": "Closes #123\n\nRan the targeted suite; all green.",
    }
    _write_outcome(tmp_path, payload)

    result = read_worker_outcome(tmp_path)

    assert result == payload
    assert result["pr_title"] == "fix(worktree): handle stale lock"
    assert result["pr_body"] == "Closes #123\n\nRan the targeted suite; all green."


def test_read_worker_outcome_without_drafted_fields_omits_them(tmp_path: Path) -> None:
    """The pre-#1771 shape (no ``pr_title``/``pr_body``) is still valid -- optional
    means callers must not assume presence, not that older workers are rejected."""
    payload = {"push_succeeded": True, "pr_created": False, "error": "gh unauthenticated"}
    _write_outcome(tmp_path, payload)

    result = read_worker_outcome(tmp_path)

    assert result == payload
    assert "pr_title" not in result
    assert "pr_body" not in result


def test_read_worker_outcome_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert read_worker_outcome(tmp_path) is None


def test_read_worker_outcome_returns_none_for_malformed_json(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / WORKER_OUTCOME_FILENAME).write_text("{not json", encoding="utf-8")

    assert read_worker_outcome(tmp_path) is None


def test_read_worker_outcome_returns_none_for_a_non_dict_payload(tmp_path: Path) -> None:
    """A worker that writes a JSON array or scalar must not crash the reader --
    the file is worker-authored text, not a trusted schema."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / WORKER_OUTCOME_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")

    assert read_worker_outcome(tmp_path) is None


def test_read_worker_outcome_tolerates_non_string_drafted_fields(tmp_path: Path) -> None:
    """``read_worker_outcome`` does no type-checking of individual fields (per its
    own docstring) -- that validation lives at the consumption site
    (``_open_salvage_pr``), which must reject a non-string draft rather than
    trusting it. This pins the reader's half of that split responsibility."""
    payload = {"push_succeeded": True, "pr_created": False, "pr_title": 123, "pr_body": None}
    _write_outcome(tmp_path, payload)

    result = read_worker_outcome(tmp_path)

    assert result == payload
