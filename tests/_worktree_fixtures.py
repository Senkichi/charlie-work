"""Shared git plumbing helpers for worktree-adjacent tests.

Hoisted out of ``test_worktree.py`` (issue #1284): a thin ``git`` subprocess
runner and a fresh-clone-with-identity builder, both imported by other test
modules that need a real (non-bare) git repo to exercise worktree behaviour
against. ``test_worktree.py`` itself is one of the three monoliths issue
#1284 marks out of scope for a full split -- only these two exported
symbols move; the rest of the file is untouched.

Note: ``test_reconcile.py`` defines its own, byte-identical top-level
``_git`` helper. That copy has zero external importers (unlike this one)
so it stays where it is per the same out-of-scope-monolith carve-out;
``tests/_reconcile_fixtures.py`` imports this module's ``_git`` rather than
keeping a third copy.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _clone_repo(remote_repo: Path, repo_root: Path) -> None:
    subprocess.run(
        ["git", "clone", str(remote_repo), str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    # A fresh clone has no committer identity on CI runners.
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "Test User")
