"""Rework reclaim of detached/deleted worktrees and foreign-worktree adoption
refusals."""

from __future__ import annotations

from pathlib import Path

import pytest

from _worktree_fixtures import _git, _init_repo
from _worktree_shared_fixtures import _force_capture_failure
from charlie_work.worktree import (
    WorktreeForeignWriterError,
    WorktreeUnsafeError,
    create_worktree,
    list_worktrees,
    remove_worktree,
)


def test_rework_reclaims_detached_worktree_at_target_path(tmp_path: Path) -> None:
    """Issue #461: a leftover worktree registered at the rework target path but
    left DETACHED (crashed mid-rework, reboot) is invisible to the branch-name
    lookup (`git worktree list --porcelain` emits no `branch` line for it).
    The by-path reclaim must remove it first so the attach path can succeed."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-9-x"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")
    _git(info1.path, "checkout", "--detach")

    worktrees = list_worktrees(repo_root)
    stale = next(wt for wt in worktrees if Path(wt["worktree"]) == info1.path)
    assert not stale.get("branch")

    info2 = create_worktree(repo_root, branch_name, rework=True)

    assert info2.path == info1.path
    assert info2.path.exists()
    current_branch = _git(info2.path, "branch", "--show-current").stdout.strip()
    assert current_branch == branch_name

    remove_worktree(repo_root, info2.path)


def test_rework_reclaim_refuses_dirty_detached_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detached leftover worktree with uncommitted work must not be silently
    clobbered by the by-path reclaim — WorktreeUnsafeError, work survives."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-10-dirty-detach"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")
    _git(info1.path, "checkout", "--detach")
    (info1.path / "dirty.txt").write_text("uncommitted worker edit\n", encoding="utf-8")

    _force_capture_failure(monkeypatch)
    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted source work"):
        create_worktree(repo_root, branch_name, rework=True)

    assert info1.path.exists()
    assert (info1.path / "dirty.txt").read_text(encoding="utf-8") == "uncommitted worker edit\n"

    remove_worktree(repo_root, info1.path)


def test_rework_reclaims_worktree_when_directory_deleted(tmp_path: Path) -> None:
    """A worktree registered at the rework target path whose directory was
    deleted out-of-band (not via `git worktree remove`) is pruned before the
    attach path runs `git worktree add`."""
    import shutil

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-11-pruned-detach"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")
    _git(info1.path, "checkout", "--detach")
    shutil.rmtree(info1.path)

    info2 = create_worktree(repo_root, branch_name, rework=True)

    assert info2.path == info1.path
    assert info2.path.exists()
    current_branch = _git(info2.path, "branch", "--show-current").stdout.strip()
    assert current_branch == branch_name

    remove_worktree(repo_root, info2.path)


def test_rework_refuses_foreign_worktree_at_unexpected_path(tmp_path: Path) -> None:
    """Issue #1118: a rework dispatch must never adopt an operator's foreign
    checkout with uncommitted edits in it — the operator's uncommitted work
    must not be committed as worker output. Issue #1476 reworked the blanket
    foreign-path refusal into a checked adoption: a CLEAN foreign checkout is
    now adopted (see tests/test_worktree_foreign_adoption.py), while a dirty
    one stays refused with WorktreeForeignWriterError.

    The test name is pinned: the collect-only gate (#1538) fails a PR on any
    leaf name removed from base, so renaming this test requires the
    operator-applied exemption label.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1118-foreign"
    # Create the branch and check it out in a FOREIGN worktree (simulating an
    # operator's interactive checkout at a different directory).
    foreign_wt = tmp_path / "operator-worktree"
    _git(repo_root, "worktree", "add", str(foreign_wt), "-b", branch_name)
    (foreign_wt / "README.md").write_text("operator's uncommitted edit\n", encoding="utf-8")

    # The orchestrator's expected worktrees dir is different from the foreign
    # worktree's location.
    worktrees_dir = tmp_path / "charlie-worktrees"

    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        create_worktree(repo_root, branch_name, rework=True, worktrees_dir=worktrees_dir)

    assert exc_info.value.worktree_path == foreign_wt
    assert exc_info.value.pid is None
    assert exc_info.value.session_id is None
    # The operator's edits are untouched — never captured, cleaned, or reset.
    assert (foreign_wt / "README.md").read_text(encoding="utf-8") == (
        "operator's uncommitted edit\n"
    )

    # Clean up the foreign worktree.
    _git(repo_root, "worktree", "remove", str(foreign_wt), "--force")


def test_rework_refuses_dirty_worktree_at_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1118: a rework dispatch (non-recovery) must refuse to adopt a
    worktree that is dirty at adoption time — never commit tracked
    modifications the shim did not itself produce.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1118-dirty"
    # Create the worktree at the orchestrator's expected path and make a commit
    # so the branch exists (rework requires an existing branch).
    worktrees_dir = tmp_path / "charlie-worktrees"
    info = create_worktree(repo_root, branch_name, base_ref="HEAD", worktrees_dir=worktrees_dir)
    (info.path / "file.txt").write_text("committed\n", encoding="utf-8")
    _git(info.path, "add", "file.txt")
    _git(info.path, "commit", "-m", "initial work")

    # Simulate foreign uncommitted edits (e.g. operator editing in the same
    # worktree directory).
    (info.path / "file.txt").write_text("uncommitted foreign edit\n", encoding="utf-8")

    _force_capture_failure(monkeypatch)
    with pytest.raises(WorktreeUnsafeError):
        create_worktree(repo_root, branch_name, rework=True, worktrees_dir=worktrees_dir)

    # Clean up: discard the dirty change and remove the worktree.
    _git(info.path, "checkout", "--", "file.txt")
    remove_worktree(repo_root, info.path, branch=branch_name)


def test_rework_recovery_allows_dirty_worktree(tmp_path: Path) -> None:
    """Issue #1118: the dirty-at-adoption hard stop must NOT fire in recovery
    mode — the dirt is a prior (owned) worker's partial work, which the
    recovery redispatch is supposed to continue from.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1118-recovery"
    worktrees_dir = tmp_path / "charlie-worktrees"
    info = create_worktree(repo_root, branch_name, base_ref="HEAD", worktrees_dir=worktrees_dir)
    (info.path / "file.txt").write_text("committed\n", encoding="utf-8")
    _git(info.path, "add", "file.txt")
    _git(info.path, "commit", "-m", "initial work")

    # Simulate a crashed worker's uncommitted partial work.
    (info.path / "partial.txt").write_text("partial work\n", encoding="utf-8")

    recovery = {"branch_name": branch_name}
    info2 = create_worktree(
        repo_root,
        branch_name,
        rework=True,
        recovery=recovery,
        worktrees_dir=worktrees_dir,
    )

    # The dirty worktree should have been adopted, not refused.
    assert info2.path == info.path
    assert (info2.path / "partial.txt").read_text(encoding="utf-8") == "partial work\n"

    # Clean up.
    _git(info.path, "checkout", "--", ".")
    remove_worktree(repo_root, info.path, branch=branch_name)
