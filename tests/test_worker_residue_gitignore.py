"""Repo-hygiene regression for issue #1138.

Worker/orchestrator protocol residue (``.worker-outcome.json``,
``.charlie-writer.json``) is state written into each worktree root at runtime,
not source. Tracking either file on ``main`` materializes a phantom outcome in
every fresh worktree and turns a worker's own write into a modification to a
tracked file -- the exact shape of dirt that trips the ``worktree_unsafe``
escalation path. These tests pin the structural prevention: the names are
gitignored and not tracked.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from charlie_work.config import WORKER_OUTCOME_FILENAME, WRITER_MARKER_FILENAME

# The protocol residue filenames that every worker worktree has written into
# its root. Sourced from the canonical constants rather than re-declared, so a
# rename in config.py flows through here instead of silently rotting the guard.
_RESIDUE_NAMES = (WORKER_OUTCOME_FILENAME, WRITER_MARKER_FILENAME)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def test_worker_residue_filenames_are_gitignored() -> None:
    """Each protocol residue filename must appear in ``.gitignore``.

    Without this, a salvage that ``git add -A``s a worktree with residue
    re-commits the sentinel -- the recurrence path that put
    ``.worker-outcome.json`` on ``main`` via PR #1114 in the first place.
    """
    gitignore = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    missing = [name for name in _RESIDUE_NAMES if name not in gitignore]
    assert not missing, f"worker residue not in .gitignore: {missing}"


def test_worker_residue_filenames_are_not_tracked() -> None:
    """No protocol residue filename may be tracked on this branch.

    ``git ls-files`` against the current index must not list either sentinel.
    This is the positive control for the ``git rm`` half of the fix: it fails
    the moment a tracked copy reappears.
    """
    tracked = set(_git("ls-files").splitlines())
    tracked_residue = sorted(tracked.intersection(_RESIDUE_NAMES))
    assert not tracked_residue, f"worker residue tracked as source: {tracked_residue}"


def test_worker_residue_is_ignored_by_git_check_ignore() -> None:
    """``git check-ignore`` must report each residue filename as ignored.

    This exercises git's actual ignore resolution (not just a substring match
    on ``.gitignore``), so a malformed pattern that fails to match is caught.
    """
    result = subprocess.run(
        ["git", "check-ignore", *_RESIDUE_NAMES],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    ignored = set(result.stdout.splitlines())
    not_ignored = sorted(set(_RESIDUE_NAMES) - ignored)
    assert not not_ignored, f"git check-ignore did not match: {not_ignored}"
