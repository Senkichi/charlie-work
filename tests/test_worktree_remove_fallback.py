"""Regression tests for ``remove_worktree``'s force-fallback path (issue #1786).

When ``git worktree remove --force`` fails, ``remove_worktree`` falls back to
a reparse-point-safe ``shutil.rmtree``. ``git worktree prune`` ran *before*
that fallback, while the directory still existed and looked live -- so a
fully successful rmtree left git's own admin entry
(``<repo>/.git/worktrees/<name>``) behind, and the independent
``git branch -D`` cleanup then failed with "cannot delete branch ... used by
worktree at ...", leaking the branch and reporting a false failure.

``test_worktree.py`` is over its file-size ceiling (issue #1442 ratchet), so
new ``remove_worktree`` coverage lives in this focused sibling module -- the
same move ``test_worktree_launcher_owned_repair.py`` made for #1688. The
``failing_git_remove`` monkeypatch technique mirrors
``test_worker_tmp_isolation.py``, whose #1767 tests document this exact
defect in a comment (they omit ``branch=`` precisely because the stale admin
entry used to block ``git branch -D``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _git, _init_repo
from charlie_work import worktree
from charlie_work.subprocess_runner import RunResult
from charlie_work.worktree import remove_worktree


def _force_git_remove_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``git worktree remove`` fail while every other git call runs for
    real, driving ``remove_worktree`` into its rmtree fallback path (e.g. a
    locked file under the tree at the moment git tries)."""
    real_run_captured = worktree.run_captured

    def failing_git_remove(args, **kwargs):
        if args[:3] == ["git", "worktree", "remove"]:
            return RunResult(returncode=1, stdout="", stderr="simulated lock")
        return real_run_captured(args, **kwargs)

    monkeypatch.setattr(worktree, "run_captured", failing_git_remove)


def _add_branch_worktree(tmp_path: Path, branch: str) -> tuple[Path, Path]:
    """A real git worktree on ``branch``. Returns (repo_root, worktree_path)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    worktree_path = tmp_path / "wt"
    subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(worktree_path), "main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return repo_root, worktree_path


def test_remove_worktree_prunes_stale_admin_entry_before_branch_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1786 regression: ``git worktree remove --force`` fails but the
    rmtree fallback fully removes the directory. Git's admin entry for the
    worktree then goes stale ("prunable") -- it must be pruned *again* before
    the independent ``git branch -D`` step, or git refuses the delete with
    "cannot delete branch ... used by worktree at ..." and the branch leaks
    behind a false failure report."""
    branch = "agent/issue-1786-prune-before-branch-d"
    repo_root, worktree_path = _add_branch_worktree(tmp_path, branch)
    _force_git_remove_failure(monkeypatch)

    removed = remove_worktree(repo_root, worktree_path, force=True, branch=branch)

    assert removed is True
    assert not worktree_path.exists()
    # The stale admin entry was pruned before `git branch -D` ran, so the
    # branch is genuinely gone -- not merely skipped by a failed delete.
    assert branch not in _git(repo_root, "branch", "--list", branch).stdout
    # And git no longer tracks the worktree at all (no "prunable" residue).
    assert str(worktree_path) not in _git(repo_root, "worktree", "list").stdout


def test_remove_worktree_prunes_before_branch_delete_after_worker_tmp_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1786 companion on the worker-tmp-reclaim sub-path: when the
    first whole-tree rmtree fails and removal only succeeds via the
    worker-tmp reclaim retry, the second ``git worktree prune`` must still
    run before ``git branch -D`` -- the prune belongs to the *final* removed
    state, not to the first rmtree attempt."""
    branch = "agent/issue-1786-prune-after-retry"
    repo_root, worktree_path = _add_branch_worktree(tmp_path, branch)
    _force_git_remove_failure(monkeypatch)

    # First whole-tree rmtree attempt fails (as if a lingering handle were
    # still held); the retry after worker-tmp reclaim runs for real.
    real_robust_rmtree = worktree._robust_rmtree
    whole_tree_attempts = {"n": 0}

    def flaky_robust_rmtree(path: Path) -> bool:
        if path == worktree_path:
            whole_tree_attempts["n"] += 1
            if whole_tree_attempts["n"] == 1:
                return False
        return real_robust_rmtree(path)

    monkeypatch.setattr(worktree, "_robust_rmtree", flaky_robust_rmtree)

    removed = remove_worktree(
        repo_root, worktree_path, force=True, branch=branch, sleep=lambda _seconds: None
    )

    assert removed is True
    assert whole_tree_attempts["n"] == 2
    assert not worktree_path.exists()
    assert branch not in _git(repo_root, "branch", "--list", branch).stdout
    assert str(worktree_path) not in _git(repo_root, "worktree", "list").stdout
