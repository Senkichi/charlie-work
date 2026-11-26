"""Redispatch/recovery over an existing worktree: reuse, recreate and reset
refusals."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _clone_repo, _git, _init_repo
from _worktree_shared_fixtures import _force_capture_failure
from charlie_work.config import OrchestratorConfig
from charlie_work.worktree import (
    WorktreeUnsafeError,
    create_worktree,
    list_worktrees,
    remove_worktree,
)


def test_recovery_clean_worktree_removed_and_recreated(tmp_path: Path) -> None:
    """Issue #110: Recovery mode with clean leftover worktree (no commits past base): remove and create fresh."""
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
    assert info2.reclaimed == "pruned"

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


def test_recovery_branch_prefix_change_fails_loudly(tmp_path: Path) -> None:
    """AC #1: Recovery mode with branch_prefix change fails loudly on worktree collision.

    When dispatch.branch_prefix changes between dispatches, the recovery record's
    branch_name (old prefix) no longer matches the derived branch (new prefix).
    The recovery logic validates this mismatch and raises RuntimeError immediately.

    The test simulates this by:
    1. Creating a worktree with branch "agent/issue-123"
    2. Attempting recovery with a mismatched branch name "worker/issue-123"
    3. The recovery check should fail loudly with a branch mismatch error

    Mutation to verify: comment out the recovery branch check in worktree.py (line 134),
    and the test will fail (it will skip recovery and attempt fresh worktree creation).
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Simulate a previous dispatch with old prefix "agent/issue"
    old_branch_name = "agent/issue-123"
    recovery_record = {"branch_name": old_branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, old_branch_name, base_ref="HEAD")

    # Verify the worktree exists at the old path
    assert info1.path.exists()
    assert (info1.path / "README.md").exists()

    # Try to recover with a new branch name (simulating branch_prefix change)
    # The recovery record's branch_name doesn't match, so this should fail loudly
    new_branch_name = "worker/issue-123"

    with pytest.raises(RuntimeError, match="Recovery record branch_name"):
        create_worktree(repo_root, new_branch_name, base_ref="HEAD", recovery=recovery_record)

    # The old worktree should still exist (not clobbered)
    assert info1.path.exists()
    assert (info1.path / "README.md").exists()

    # Clean up
    remove_worktree(repo_root, info1.path)


def test_recovery_fetch_fallback_on_missing_remote_branch(tmp_path: Path) -> None:
    """Issue #110: Recovery with no remote branch should fall through to fresh dispatch.

    When a worker is killed before its first push, the branch exists locally but not on origin.
    Recovery should detect this via _remote_branch_exists and fall through to fresh dispatch
    instead of failing with fetch error 128.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Simulate a killed-before-push session: create branch locally but don't push
    branch_name = "agent/issue-110-no-remote"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Verify worktree exists locally
    assert info1.path.exists()
    assert (info1.path / "README.md").exists()

    # Verify branch does NOT exist on origin
    ls_remote_result = subprocess.run(
        ["git", "ls-remote", "origin", f"refs/heads/{branch_name}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not ls_remote_result.stdout.strip()

    # Recovery dispatch should succeed via fetch-fallback path
    info2 = create_worktree(repo_root, branch_name, base_ref="HEAD", recovery=recovery_record)

    # Should be a fresh worktree (same path, but recreated)
    assert info2.path == info1.path
    assert info2.path.exists()
    assert info2.reclaimed == "fetch-fallback"

    # Clean up
    remove_worktree(repo_root, info2.path)


def test_redispatch_refuses_to_reset_with_unpushed_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #257: a redispatch with local commits not on the remote branch
    must hard-refuse to reset the worktree rather than discarding work.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    branch_name = "agent/issue-261-attempt-preserve"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD", issue_number=261)

    # Simulate the dead worker having committed real work before dying
    # (never pushed — this is exactly the fetch-fallback scenario).
    work_file = info1.path / "work.txt"
    work_file.write_text("unpushed work\n", encoding="utf-8")
    _git(info1.path, "add", "work.txt")
    _git(info1.path, "commit", "-m", "worker commit before death")

    # Redispatch must refuse to reset the worktree because it has local commits.
    _force_capture_failure(monkeypatch)
    with pytest.raises(WorktreeUnsafeError, match="worktree has 1 local commit"):
        create_worktree(
            repo_root, branch_name, base_ref="HEAD", recovery=recovery_record, issue_number=261
        )

    # The original worktree and branch must remain intact.
    assert info1.path.exists()
    branches = _git(repo_root, "branch", "--list", branch_name).stdout.strip()
    assert branch_name in branches

    # Clean up
    remove_worktree(repo_root, info1.path)


def test_second_redispatch_refuses_with_unpushed_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #257: each redispatch with local commits not on the remote branch
    must hard-refuse to reset; the worktree is left intact.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    branch_name = "agent/issue-262-double-attempt"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}

    # Attempt 1: dies with unpushed work.
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD", issue_number=262)
    (info1.path / "attempt1.txt").write_text("attempt 1\n", encoding="utf-8")
    _git(info1.path, "add", "attempt1.txt")
    _git(info1.path, "commit", "-m", "attempt 1 work")

    _force_capture_failure(monkeypatch)
    with pytest.raises(WorktreeUnsafeError, match="worktree has 1 local commit"):
        create_worktree(
            repo_root, branch_name, base_ref="HEAD", recovery=recovery_record, issue_number=262
        )

    # The original worktree remains intact.
    assert info1.path.exists()
    assert (info1.path / "attempt1.txt").exists()

    # Clean up
    remove_worktree(repo_root, info1.path)


def test_recovery_stale_worktree_pruned_on_missing_directory(tmp_path: Path) -> None:
    """Issue #110: Stale worktree with missing directory should be pruned before fresh dispatch.

    When a worktree is registered in git's metadata but the directory is missing
    (e.g., manually deleted), fresh dispatch should prune the stale registration
    and succeed.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create a worktree
    branch_name = "agent/issue-110-prune"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Manually delete the directory (simulating stale state)
    import shutil

    shutil.rmtree(info1.path)

    # Verify directory is missing but worktree is still registered
    assert not info1.path.exists()
    worktrees = list_worktrees(repo_root)
    assert any(Path(wt["worktree"]) == info1.path for wt in worktrees)

    # Fresh dispatch should prune and succeed
    info2 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Should be a fresh worktree
    assert info2.path == info1.path
    assert info2.path.exists()
    # Note: this test doesn't use recovery mode, so reclaimed might be None or "pruned"
    # depending on whether the branch exists
    assert info2.reclaimed in (None, "pruned")

    # Clean up
    remove_worktree(repo_root, info2.path)


def test_recovery_dirty_worktree_salvaged(tmp_path: Path) -> None:
    """Issue #110: Recovery mode with dirty changes reuses worktree (existing behavior).

    In recovery mode, dirty worktrees are reused via rework-style attach.
    The salvage logic is for fresh dispatch stale worktree reclamation.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create a worktree with dirty changes
    branch_name = "agent/issue-110-dirty"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Push the branch to origin first
    _git(repo_root, "push", "origin", branch_name)

    # Add uncommitted changes
    (info1.path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    # Verify dirty state
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=info1.path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status_result.stdout.strip()

    # Recovery dispatch should reuse via rework
    info2 = create_worktree(repo_root, branch_name, base_ref="HEAD", recovery=recovery_record)

    # Should reuse the same worktree
    assert info2.path == info1.path
    assert info2.path.exists()
    # The dirty file should still be there
    assert (info2.path / "dirty.txt").read_text(encoding="utf-8") == "uncommitted\n"

    # Clean up
    remove_worktree(repo_root, info2.path)


def test_fresh_dispatch_dirty_worktree_refuses_to_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #257: Fresh dispatch with a dirty stale worktree must hard-refuse
    to reset rather than discarding uncommitted modifications.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo, bare=True)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create a worktree with dirty changes (simulating a killed session)
    branch_name = "agent/issue-110-dirty-fresh"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Add uncommitted changes
    (info1.path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    # Verify dirty state
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=info1.path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status_result.stdout.strip()

    # Fresh dispatch must refuse to reset the dirty worktree.
    _force_capture_failure(monkeypatch)
    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted source work"):
        create_worktree(repo_root, branch_name, base_ref="HEAD")

    # The original worktree and the dirty file must remain intact.
    assert info1.path.exists()
    assert (info1.path / "dirty.txt").exists()

    # Clean up
    remove_worktree(repo_root, info1.path)


def test_fresh_dispatch_ignores_injected_only_dirtiness(tmp_path: Path) -> None:
    """Issue #381: a stale worktree with dirty orchestrator-injected prompt files
    but no worker-authored changes can be safely reset and recreated."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo, bare=True)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    branch_name = "agent/issue-381-injected"
    config = OrchestratorConfig()
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD", config=config)

    # Only orchestrator-injected prompt files are dirty.
    for injected in config.dispatch.injected_paths:
        prompt = info1.path / injected
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text("injected prompt", encoding="utf-8")

    # Fresh dispatch should prune the stale worktree and recreate it.
    info2 = create_worktree(repo_root, branch_name, base_ref="HEAD", config=config)
    assert info2.path.exists()
    assert info2.reclaimed == "pruned"

    # Clean up
    remove_worktree(repo_root, info2.path)


def test_fresh_dispatch_still_refuses_worker_authored_dirtiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #381: worker-authored uncommitted changes still hard-refuse fresh dispatch."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo, bare=True)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    branch_name = "agent/issue-381-worker-dirty"
    config = OrchestratorConfig()
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD", config=config)

    for injected in config.dispatch.injected_paths:
        prompt = info1.path / injected
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text("injected prompt", encoding="utf-8")

    (info1.path / "worker-change.txt").write_text("worker work\n", encoding="utf-8")

    _force_capture_failure(monkeypatch)
    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted source work"):
        create_worktree(repo_root, branch_name, base_ref="HEAD", config=config)

    assert info1.path.exists()
    assert (info1.path / "worker-change.txt").exists()

    remove_worktree(repo_root, info1.path)


def test_recovery_with_commits_reuses_worktree(tmp_path: Path) -> None:
    """Issue #110: Recovery mode with commits reuses worktree (existing behavior).

    In recovery mode, worktrees with commits are reused via rework-style attach.
    The salvage logic is for fresh dispatch stale worktree reclamation.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create a worktree with commits
    branch_name = "agent/issue-110-commits"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Add a commit
    (info1.path / "file1.txt").write_text("partial work\n", encoding="utf-8")
    _git(info1.path, "add", "file1.txt")
    _git(info1.path, "commit", "-m", "partial work")

    # Push the branch to origin
    _git(repo_root, "push", "origin", branch_name)

    # Add another commit that is NOT pushed
    (info1.path / "file2.txt").write_text("more work\n", encoding="utf-8")
    _git(info1.path, "add", "file2.txt")
    _git(info1.path, "commit", "-m", "more work")

    # Recovery dispatch should reuse via rework
    info2 = create_worktree(repo_root, branch_name, base_ref="HEAD", recovery=recovery_record)

    # Should reuse the same worktree
    assert info2.path == info1.path
    assert info2.path.exists()
    # The partial work should still be there
    assert (info2.path / "file1.txt").read_text(encoding="utf-8") == "partial work\n"
    assert (info2.path / "file2.txt").read_text(encoding="utf-8") == "more work\n"

    # Clean up
    remove_worktree(repo_root, info2.path)


def test_recovery_transient_fetch_failure_still_aborts(tmp_path: Path) -> None:
    """Issue #110: Transient fetch failure should still abort, not fall through to fresh dispatch.

    When a real network/auth error occurs (not just missing remote branch),
    the dispatch should fail with the error, not silently fall through.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create a branch and push it to origin
    branch_name = "agent/issue-110-transient"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")
    _git(repo_root, "push", "origin", branch_name)

    # Break the origin remote to simulate a transient network error
    _git(repo_root, "remote", "set-url", "origin", "file:///nonexistent/path")

    # Recovery with rework should fail on fetch error
    with pytest.raises(RuntimeError, match="Fetch failed for rework branch"):
        create_worktree(repo_root, branch_name, rework=True)

    # Clean up
    remove_worktree(repo_root, info1.path)


def test_recovery_junction_safety_preserves_shared_venv(tmp_path: Path) -> None:
    """Issue #110: Reclamation path must never follow .venv junction into shared venv.

    When salvaging a worktree with a .venv junction, the junction-safe removal
    must preserve the shared venv target.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)
    venv_source = tmp_path / "shared-venv"
    venv_source.mkdir()
    marker = venv_source / "site-packages-marker.txt"
    marker.write_text("shared contents\n", encoding="utf-8")

    # Create a worktree with a junctioned .venv
    branch_name = "agent/issue-110-junction"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD", venv_source=venv_source)

    # Push the branch to origin first
    _git(repo_root, "push", "origin", branch_name)

    # Add dirty changes to trigger salvage
    (info1.path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    # Recovery dispatch should reuse via rework (junction-safe)
    info2 = create_worktree(
        repo_root, branch_name, base_ref="HEAD", recovery=recovery_record, venv_source=venv_source
    )

    # Should reuse the same worktree
    assert info2.path == info1.path
    assert info2.path.exists()
    # The dirty file should still be there
    assert (info2.path / "dirty.txt").read_text(encoding="utf-8") == "uncommitted\n"

    # The shared venv must survive (junction safety)
    assert venv_source.exists()
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "shared contents\n"

    # Clean up
    remove_worktree(repo_root, info2.path)
