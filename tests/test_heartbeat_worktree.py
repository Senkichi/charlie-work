"""Issue #1476: heartbeat worktree resolution is registry-first.

``scripts/heartbeat_worktree.py``'s ``_worktree_path_for_branch`` mirrors
``charlie_work.worktree.worktree_path_for_branch``: a branch already checked
out in a registered worktree resolves to that path — an adopted foreign
checkout lives outside the managed root, and reporting the managed slug for a
live adopted worker would be a false ANOMALY in the in-progress-staleness
check. An unregistered branch falls back to the managed slug path, matching
the pre-#1476 behavior.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from _heartbeat_check_fixtures import _load_heartbeat_check
from _worktree_fixtures import _git, _init_repo


def _repo_info(hb: ModuleType, repo_root: Path, tmp_path: Path) -> Any:
    return hb.RepoInfo(
        slug="owner/repo",
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        config_path=tmp_path / "orchestrator.config.yaml",
    )


def test_worktree_path_for_branch_resolves_registered_foreign_checkout(
    tmp_path: Path,
) -> None:
    """A branch checked out in a registered foreign worktree resolves to that
    path — ``git worktree list`` is the source of truth for where the branch
    actually lives, ahead of the managed-path computation."""
    hb = _load_heartbeat_check()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1476-hb-foreign"
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch)

    repo = _repo_info(hb, repo_root, tmp_path)

    assert hb._worktree_path_for_branch(repo, branch) == foreign_wt
    assert hb._worktree_helpers._registered_worktree_for_branch(repo, branch) == foreign_wt

    _git(repo_root, "worktree", "remove", str(foreign_wt), "--force")


def test_worktree_path_for_branch_falls_back_to_managed_slug(tmp_path: Path) -> None:
    """An unregistered branch degrades to the managed-path computation —
    ``<state_dir>/worktrees/<slug>`` — matching the pre-#1476 behavior."""
    hb = _load_heartbeat_check()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    repo = _repo_info(hb, repo_root, tmp_path)

    branch = "agent/issue-9999-unregistered"
    assert hb._worktree_helpers._registered_worktree_for_branch(repo, branch) is None
    assert hb._worktree_path_for_branch(repo, branch) == (
        tmp_path / "state" / "worktrees" / "agent-issue-9999-unregistered"
    )
