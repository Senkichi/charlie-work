from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from charlie_work.worktree import (
    WorktreeInfo,
    _default_worktrees_dir,
    create_worktree,
    is_junction,
    list_worktrees,
    remove_worktree,
)


def _init_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )
    run(["git", "init", "--initial-branch=main"])
    run(["git", "config", "user.email", "test@example.test"])
    run(["git", "config", "user.name", "Test User"])
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    run(["git", "add", "README.md"])
    run(["git", "commit", "-m", "initial commit"])


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


def test_create_and_remove_round_trip(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    info = create_worktree(repo_root, "agent/issue-1-fix", base_ref="HEAD")

    assert isinstance(info, WorktreeInfo)
    assert info.branch == "agent/issue-1-fix"
    assert info.path.exists()
    assert (info.path / "README.md").exists()
    assert info.venv_junction is None

    removed = remove_worktree(repo_root, info.path)

    assert removed is True
    assert not info.path.exists()


def test_create_worktree_junctions_shared_venv(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    venv_source = tmp_path / "shared-venv"
    venv_source.mkdir()
    marker = venv_source / "site-packages-marker.txt"
    marker.write_text("shared contents\n", encoding="utf-8")

    info = create_worktree(
        repo_root, "agent/issue-2-junction", base_ref="HEAD", venv_source=venv_source
    )

    assert info.venv_junction == info.path / ".venv"
    assert is_junction(info.venv_junction)
    # Junction resolves through to the shared venv's contents.
    assert (info.path / ".venv" / "site-packages-marker.txt").read_text(
        encoding="utf-8"
    ) == "shared contents\n"


def test_remove_worktree_refuses_when_venv_is_real_directory(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-3-real-venv", base_ref="HEAD")
    # Simulate a worker that cold-built its own venv instead of junctioning.
    real_venv = info.path / ".venv"
    real_venv.mkdir()
    (real_venv / "pyvenv.cfg").write_text("home = somewhere\n", encoding="utf-8")

    removed = remove_worktree(repo_root, info.path)

    assert removed is False
    assert info.path.exists()
    assert real_venv.exists()


def test_remove_worktree_force_removes_real_venv_dir_but_not_junction_targets(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-4-force", base_ref="HEAD")
    real_venv = info.path / ".venv"
    real_venv.mkdir()
    (real_venv / "pyvenv.cfg").write_text("home = somewhere\n", encoding="utf-8")

    removed = remove_worktree(repo_root, info.path, force=True)

    assert removed is True
    assert not info.path.exists()


def test_remove_worktree_junction_removal_preserves_shared_venv_contents(
    tmp_path: Path,
) -> None:
    """THE regression test: removing a worktree whose .venv is a junction must
    delete only the reparse point, never the shared venv's real contents."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    venv_source = tmp_path / "shared-venv"
    venv_source.mkdir()
    marker = venv_source / "site-packages-marker.txt"
    marker.write_text("shared contents\n", encoding="utf-8")

    info = create_worktree(
        repo_root, "agent/issue-5-survive", base_ref="HEAD", venv_source=venv_source
    )
    assert is_junction(info.path / ".venv")

    removed = remove_worktree(repo_root, info.path)

    assert removed is True
    assert not info.path.exists()
    # The shared venv itself, and its contents, must survive.
    assert venv_source.exists()
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "shared contents\n"


def test_create_worktree_rejects_existing_venv_link_target(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    venv_source = tmp_path / "shared-venv"
    venv_source.mkdir()

    # First creation succeeds and leaves a worktree at the deterministic slug path.
    info = create_worktree(
        repo_root, "agent/issue-6-dup", base_ref="HEAD", venv_source=venv_source
    )
    remove_worktree(repo_root, info.path)

    # Recreate the directory structure manually with a pre-existing .venv to
    # simulate stale state that create_worktree must not silently clobber.
    stale_path = info.path
    stale_path.mkdir(parents=True)
    (stale_path / ".venv").mkdir()

    with pytest.raises(RuntimeError):
        create_worktree(
            repo_root,
            "agent/issue-6-dup",
            base_ref="HEAD",
            worktrees_dir=info.path.parent,
            venv_source=venv_source,
        )


def test_list_worktrees_parses_porcelain_output(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-7-list", base_ref="HEAD")

    worktrees = list_worktrees(repo_root)

    assert len(worktrees) == 2  # main checkout + the new worktree
    paths = [entry["worktree"] for entry in worktrees]
    assert repo_root.resolve() in [p.resolve() for p in paths]
    assert info.path.resolve() in [p.resolve() for p in paths]
    branches = [entry.get("branch") for entry in worktrees]
    assert any(branch and branch.endswith("agent/issue-7-list") for branch in branches)


def test_list_worktrees_returns_empty_list_for_non_git_dir(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()

    assert list_worktrees(not_a_repo) == []


def test_is_junction_false_for_ordinary_directory(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()

    assert is_junction(ordinary) is False


def test_is_junction_false_for_missing_path(tmp_path: Path) -> None:
    assert is_junction(tmp_path / "does-not-exist") is False


@pytest.mark.skipif(os.name != "nt", reason="junction semantics are Windows-specific")
def test_is_junction_true_for_windows_junction(tmp_path: Path) -> None:
    import _winapi

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    _winapi.CreateJunction(str(target), str(link))

    assert is_junction(link) is True


def test_rework_reuses_existing_worktree(tmp_path: Path) -> None:
    """Rework mode should reuse an existing worktree and fetch+fast-forward it."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create a branch and worktree (simulating a previous PR cycle)
    branch_name = "agent/issue-1-rework"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")
    (info1.path / "file1.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "file1.txt"],
        cwd=info1.path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "add file1"],
        cwd=info1.path,
        check=True,
        capture_output=True,
    )

    # Verify the worktree is in the list
    worktrees = list_worktrees(repo_root)
    assert any(
        wt.get("branch", "").endswith(f"/{branch_name}") or wt.get("branch") == branch_name
        for wt in worktrees
    )

    # In rework mode, create_worktree should reuse the existing worktree
    # The fetch will fail in a test repo without a remote, but the reuse logic
    # should still work (we just skip the fetch in this case)
    info2 = create_worktree(repo_root, branch_name, rework=True)

    # Should return the same path
    assert info2.path == info1.path
    # File should still exist
    assert (info2.path / "file1.txt").read_text(encoding="utf-8") == "original\n"

    # Clean up
    remove_worktree(repo_root, info1.path)


def test_rework_attaches_to_existing_branch(tmp_path: Path) -> None:
    """Rework mode should attach to an existing branch when no worktree exists."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create a branch without a worktree (simulating a branch that exists but
    # its worktree was cleaned up after merge)
    branch_name = "agent/issue-2-reattach"
    subprocess.run(
        ["git", "branch", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # In rework mode, create_worktree should attach to the existing branch
    # (no -b flag, so it doesn't try to create a new branch)
    info = create_worktree(repo_root, branch_name, rework=True)

    assert info.branch == branch_name
    assert info.path.exists()
    # Should be on the existing branch, not a new one
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=info.path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == branch_name

    # Clean up
    remove_worktree(repo_root, info.path)


def test_rework_reuse_fetches_and_fast_forwards_to_origin_tip(tmp_path: Path) -> None:
    """Rework reuse path must fast-forward the existing worktree to the origin tip."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create the agent branch + worktree locally and push it to origin.
    branch_name = "agent/issue-1-rework-ff"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")
    (info1.path / "file1.txt").write_text("original\n", encoding="utf-8")
    _git(info1.path, "add", "file1.txt")
    _git(info1.path, "commit", "-m", "add file1")
    _git(repo_root, "push", "origin", branch_name)

    # Advance the AGENT branch on the remote so the local worktree is behind.
    _git(remote_repo, "checkout", branch_name)
    (remote_repo / "file2.txt").write_text("remote change\n", encoding="utf-8")
    _git(remote_repo, "add", "file2.txt")
    _git(remote_repo, "commit", "-m", "add file2")
    remote_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()
    _git(remote_repo, "checkout", "main")

    info2 = create_worktree(repo_root, branch_name, rework=True)

    # Same worktree, fast-forwarded to the origin tip.
    assert info2.path == info1.path
    assert (info2.path / "file1.txt").read_text(encoding="utf-8") == "original\n"
    assert (info2.path / "file2.txt").read_text(encoding="utf-8") == "remote change\n"
    assert _git(info2.path, "rev-parse", "HEAD").stdout.strip() == remote_tip

    remove_worktree(repo_root, info1.path)


def test_rework_attach_fetches_to_origin_tip(tmp_path: Path) -> None:
    """Rework attach path must materialize the existing branch at the origin tip."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create the agent branch locally (no worktree) and push it to origin.
    branch_name = "agent/issue-2-attach-ff"
    _git(repo_root, "branch", branch_name)
    _git(repo_root, "push", "origin", branch_name)

    # Advance the AGENT branch on the remote so the local ref is behind.
    _git(remote_repo, "checkout", branch_name)
    (remote_repo / "file1.txt").write_text("remote change\n", encoding="utf-8")
    _git(remote_repo, "add", "file1.txt")
    _git(remote_repo, "commit", "-m", "add file1")
    remote_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()
    _git(remote_repo, "checkout", "main")

    info = create_worktree(repo_root, branch_name, rework=True)

    # Attached to the existing branch at the origin tip.
    assert info.branch == branch_name
    assert (info.path / "file1.txt").read_text(encoding="utf-8") == "remote change\n"
    assert _git(info.path, "rev-parse", "HEAD").stdout.strip() == remote_tip

    remove_worktree(repo_root, info.path)


def test_rework_fetch_failure_raises_when_origin_exists(tmp_path: Path) -> None:
    """Rework with origin present but fetch failure should raise RuntimeError."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create the agent branch locally (no worktree) and push it to origin.
    branch_name = "agent/issue-3-fetch-fail"
    _git(repo_root, "branch", branch_name)
    _git(repo_root, "push", "origin", branch_name)

    # Break the origin remote to simulate a fetch failure
    _git(repo_root, "remote", "set-url", "origin", "file:///nonexistent/path")

    # Rework attach path should raise on fetch failure
    with pytest.raises(RuntimeError, match="Fetch failed for rework branch"):
        create_worktree(repo_root, branch_name, rework=True)


def test_remove_worktree_deletes_branch_when_provided(tmp_path: Path) -> None:
    """remove_worktree should delete the branch when branch parameter is provided."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-delete-branch"
    info = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Verify the branch exists
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Remove the worktree with branch deletion
    removed = remove_worktree(repo_root, info.path, branch=branch_name)
    assert removed is True
    assert not info.path.exists()

    # Verify the branch is deleted
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name not in result.stdout


def test_remove_worktree_without_branch_parameter_preserves_branch(tmp_path: Path) -> None:
    """remove_worktree without branch parameter should preserve the branch."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-2-preserve-branch"
    info = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Verify the branch exists
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Remove the worktree without branch deletion
    removed = remove_worktree(repo_root, info.path)
    assert removed is True
    assert not info.path.exists()

    # Verify the branch still exists
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Clean up the branch manually
    subprocess.run(
        ["git", "branch", "-D", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def test_junction_creation_failure_cleans_up_worktree_and_branch(tmp_path: Path) -> None:
    """Junction creation failure should clean up the orphan worktree and branch."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create a non-existent venv_source to trigger junction failure
    venv_source = tmp_path / "nonexistent-venv"
    branch_name = "agent/issue-3-junction-fail"

    with pytest.raises((OSError, RuntimeError)):
        create_worktree(repo_root, branch_name, base_ref="HEAD", venv_source=venv_source)

    # Verify the worktree is cleaned up
    worktrees_dir = _default_worktrees_dir(repo_root)
    worktree_path = worktrees_dir / branch_name.replace("/", "-")
    assert not worktree_path.exists()

    # Verify the branch is deleted
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name not in result.stdout


def test_junction_creation_failure_in_rework_preserves_branch(tmp_path: Path) -> None:
    """Junction creation failure in rework mode should preserve the existing branch."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create a branch first (simulating a previous PR cycle)
    branch_name = "agent/issue-5-junction-fail-rework"
    subprocess.run(
        ["git", "branch", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Verify the branch exists
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Create a non-existent venv_source to trigger junction failure in rework mode
    venv_source = tmp_path / "nonexistent-venv"

    with pytest.raises((OSError, RuntimeError)):
        create_worktree(repo_root, branch_name, rework=True, venv_source=venv_source)

    # Verify the branch is preserved (not deleted)
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Clean up
    subprocess.run(
        ["git", "branch", "-D", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def test_recovery_clean_worktree_removed_and_recreated(tmp_path: Path) -> None:
    """Recovery mode with clean leftover worktree (no commits past base): remove and create fresh."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Simulate a previous dispatch that created a worktree but crashed before committing
    branch_name = "agent/issue-1-recovery-clean"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Verify worktree exists
    assert info1.path.exists()
    assert (info1.path / "README.md").exists()

    # Recovery dispatch should remove the clean worktree and create fresh
    info2 = create_worktree(repo_root, branch_name, base_ref="HEAD", recovery=recovery_record)

    # Should be a fresh worktree (same path, but recreated)
    assert info2.path == info1.path
    assert info2.path.exists()
    assert (info2.path / "README.md").exists()

    # Clean up
    remove_worktree(repo_root, info2.path)


def test_recovery_with_commits_reuses_worktree(tmp_path: Path) -> None:
    """Recovery mode with leftover worktree containing commits: reuse via rework-style attach."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Simulate a previous dispatch that created a worktree and committed work
    branch_name = "agent/issue-2-recovery-commits"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Add a commit to simulate partial work
    (info1.path / "file1.txt").write_text("partial work\n", encoding="utf-8")
    _git(info1.path, "add", "file1.txt")
    _git(info1.path, "commit", "-m", "partial work")

    # Recovery dispatch should reuse the worktree with commits
    info2 = create_worktree(repo_root, branch_name, base_ref="HEAD", recovery=recovery_record)

    # Should reuse the same worktree
    assert info2.path == info1.path
    # The partial work should still be there
    assert (info2.path / "file1.txt").read_text(encoding="utf-8") == "partial work\n"

    # Clean up
    remove_worktree(repo_root, info2.path)


def test_recovery_with_dirty_tree_reuses_worktree(tmp_path: Path) -> None:
    """Recovery mode with dirty working tree: reuse via rework-style attach."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Simulate a previous dispatch that created a worktree with dirty changes
    branch_name = "agent/issue-3-recovery-dirty"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Add uncommitted changes to simulate dirty state
    (info1.path / "file2.txt").write_text("uncommitted\n", encoding="utf-8")

    # Recovery dispatch should reuse the worktree despite dirty state
    info2 = create_worktree(repo_root, branch_name, base_ref="HEAD", recovery=recovery_record)

    # Should reuse the same worktree
    assert info2.path == info1.path
    # The uncommitted work should still be there
    assert (info2.path / "file2.txt").read_text(encoding="utf-8") == "uncommitted\n"

    # Clean up
    remove_worktree(repo_root, info2.path, force=True)


def test_remove_worktree_branch_deletion_independent_of_worktree_removal(tmp_path: Path) -> None:
    """Branch deletion should be attempted even when worktree removal fails."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-4-branch-delete-on-wt-fail"
    info = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Verify the branch exists
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Force worktree removal to fail by making the worktree path non-writable
    # (simulating a Windows file lock scenario)
    # We'll do this by removing the worktree from git's metadata first
    subprocess.run(
        ["git", "worktree", "remove", str(info.path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Now the directory still exists but git doesn't know about it
    # remove_worktree should fail on the git worktree remove step
    # but still attempt branch deletion
    removed = remove_worktree(repo_root, info.path, branch=branch_name)

    # Should return False because worktree removal failed
    assert removed is False

    # But the branch should still be deleted
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name not in result.stdout

    # Clean up the orphaned directory if it still exists
    if info.path.exists():
        import shutil
        shutil.rmtree(info.path)


def test_recovery_branch_mismatch_raises(tmp_path: Path) -> None:
    """Recovery mode with branch name mismatch: raise RuntimeError."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Recovery record for a different branch
    recovery_record = {"branch_name": "agent/issue-1-other", "status": "dispatched"}

    with pytest.raises(RuntimeError, match="Recovery record branch_name"):
        create_worktree(
            repo_root, "agent/issue-1-different", base_ref="HEAD", recovery=recovery_record
        )


def test_recovery_foreign_branch_fails_loudly(tmp_path: Path) -> None:
    """AC #3: Recovery mode with leftover worktree on foreign branch: fail loudly."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create a worktree on a branch
    branch_name = "agent/issue-4-foreign"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")
    (info1.path / "foreign.txt").write_text("foreign work\n", encoding="utf-8")
    _git(info1.path, "add", "foreign.txt")
    _git(info1.path, "commit", "-m", "foreign work")

    # Manually switch the worktree to a different branch to simulate foreign state
    # (the worktree path is the same, but the branch is different)
    _git(info1.path, "checkout", "-b", "agent/issue-5-different")

    # Try to recover with a record for the original branch, but the worktree is now on a different branch
    recovery_record = {"branch_name": "agent/issue-4-foreign", "status": "dispatched"}

    with pytest.raises(RuntimeError, match="Recovery mode found leftover worktree"):
        create_worktree(
            repo_root, "agent/issue-4-foreign", base_ref="HEAD", recovery=recovery_record
        )

    # Clean up
    remove_worktree(repo_root, info1.path)


def test_recovery_with_venv_dir_reuses_worktree(tmp_path: Path) -> None:
    """Recovery mode with .venv directory (documented danger case): reuse via rework-style attach."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Simulate a previous dispatch that created a worktree with a real .venv directory
    # (the documented danger case: worker cold-built its own venv instead of junctioning)
    branch_name = "agent/issue-5-recovery-venv"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Create a real .venv directory (not a junction)
    real_venv = info1.path / ".venv"
    real_venv.mkdir()
    (real_venv / "pyvenv.cfg").write_text("home = somewhere\n", encoding="utf-8")

    # Add a commit to ensure we reuse (not remove-and-recreate)
    (info1.path / "file1.txt").write_text("partial work\n", encoding="utf-8")
    _git(info1.path, "add", "file1.txt")
    _git(info1.path, "commit", "-m", "partial work")

    # Recovery dispatch should reuse the worktree despite the .venv directory
    info2 = create_worktree(repo_root, branch_name, base_ref="HEAD", recovery=recovery_record)

    # Should reuse the same worktree
    assert info2.path == info1.path
    # The partial work should still be there
    assert (info2.path / "file1.txt").read_text(encoding="utf-8") == "partial work\n"
    # The .venv directory should still be there (not deleted)
    assert (info2.path / ".venv").exists()
    assert (info2.path / ".venv" / "pyvenv.cfg").exists()

    # Clean up with force=True since .venv is a real directory
    remove_worktree(repo_root, info2.path, force=True)
