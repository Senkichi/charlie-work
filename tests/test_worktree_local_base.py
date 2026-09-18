"""Tests for ``inspect_worktree_state``'s HEAD re-anchoring on a pure-local
repo (no origin remote).

``inspect_worktree_state`` resolves the comparison base via
``_resolve_default_branch_ref``, which returns the literal string
``"HEAD"`` when a repo has no origin remote at all. Naively comparing
``git merge-base HEAD HEAD`` *inside the worker's own worktree* always
yields the worker's own tip (ahead_count 0 by construction), because
``"HEAD"`` there means the worktree's own branch, not the commit the
worktree was cut from. ``inspect_worktree_state`` re-anchors the
*comparison* ref to ``_main_worktree_head`` (the main checkout's live
HEAD) while still *reporting* ``resolved_base_ref == "HEAD"`` -- see
``worktree.py``'s ``inspect_worktree_state`` docstring around the
``resolved_base_ref == "HEAD"`` branch. These tests exercise that fix
against real git repos/worktrees, not mocks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from charlie_work.worktree import (
    WorktreeState,
    _main_worktree_head,
    inspect_worktree_state,
)

# ---------------------------------------------------------------------------
# Inlined git helpers -- self-contained per this repo's test-file convention,
# not shared with the other two new files.
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo with one commit and NO origin remote."""
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(
        repo_root,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "--allow-empty",
        "-m",
        "chore: seed",
    )


def _create_linked_worktree(repo_root: Path, worktree_path: Path, branch: str) -> None:
    _git(repo_root, "worktree", "add", "-b", branch, str(worktree_path))


def _commit_file(worktree_path: Path, name: str, content: str, message: str) -> None:
    (worktree_path / name).write_text(content, encoding="utf-8")
    _git(worktree_path, "add", name)
    _git(worktree_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", message)


# ---------------------------------------------------------------------------
# inspect_worktree_state
# ---------------------------------------------------------------------------


def test_inspect_worktree_state_no_origin_completed_ahead_one(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    worktree_path = tmp_path / "wt"
    _create_linked_worktree(repo_root, worktree_path, "agent/issue-1-work")

    _commit_file(worktree_path, "feature.txt", "feature\n", "feat: add feature")

    inspection = inspect_worktree_state(worktree_path)

    assert inspection.state == WorktreeState.COMPLETED
    assert inspection.ahead_count == 1
    assert inspection.resolved_base_ref == "HEAD"
    assert inspection.dirty is False


def test_inspect_worktree_state_ahead_count_survives_main_advancing(tmp_path: Path) -> None:
    """MUTATION CHECK target: without the HEAD re-anchoring fix in
    ``inspect_worktree_state`` (comparing against the *live* main-worktree
    HEAD rather than a value cached at worktree-creation time), this would
    still pass by accident only if main never moved. Advancing main here
    (after the worker's worktree was cut) is what would break a naive
    "cache the base ref at creation time" implementation while the
    re-anchoring fix -- which re-queries ``_main_worktree_head`` on every
    call -- tolerates it, per the source docstring's own claim that
    "merge-base still finds the fork point.\""""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    worktree_path = tmp_path / "wt"
    _create_linked_worktree(repo_root, worktree_path, "agent/issue-1-work")

    _commit_file(worktree_path, "feature.txt", "feature\n", "feat: add feature")

    # Main checkout advances AFTER the worktree was cut.
    _commit_file(repo_root, "unrelated.txt", "unrelated\n", "chore: advance main")

    inspection = inspect_worktree_state(worktree_path)

    assert inspection.state == WorktreeState.COMPLETED
    assert inspection.ahead_count == 1
    assert inspection.resolved_base_ref == "HEAD"


def test_inspect_worktree_state_no_origin_no_commits(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    worktree_path = tmp_path / "wt"
    _create_linked_worktree(repo_root, worktree_path, "agent/issue-1-idle")

    inspection = inspect_worktree_state(worktree_path)

    assert inspection.state == WorktreeState.NO_COMMITS
    assert inspection.ahead_count == 0
    assert inspection.resolved_base_ref == "HEAD"
    assert inspection.dirty is False


# ---------------------------------------------------------------------------
# _main_worktree_head
# ---------------------------------------------------------------------------


def test_main_worktree_head_from_linked_worktree(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    main_head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()

    worktree_path = tmp_path / "wt"
    _create_linked_worktree(repo_root, worktree_path, "agent/issue-1-work")

    assert _main_worktree_head(worktree_path) == main_head


def test_main_worktree_head_none_for_non_git_directory(tmp_path: Path) -> None:
    plain_dir = tmp_path / "not-a-repo"
    plain_dir.mkdir()

    assert _main_worktree_head(plain_dir) is None
