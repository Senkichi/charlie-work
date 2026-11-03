"""Redispatch/recovery over an existing worktree: reuse, reset refusals, probe
failures and salvage of dirty or committed work."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _clone_repo, _git, _init_repo
from _worktree_shared_fixtures import _force_capture_failure
from charlie_work.config import OrchestratorConfig
from charlie_work.subprocess_runner import RunResult
from charlie_work.worktree import (
    WorktreeProbeFailedError,
    WorktreeUnsafeError,
    _default_worktrees_dir,
    _has_origin_remote,
    _is_confirmed_missing_ref,
    _worktree_refuse_to_reset_reason,
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


def test_recovery_transient_fetch_failure_via_probe_aborts(tmp_path: Path) -> None:
    """Issue #110: Transient fetch failure via _remote_branch_exists probe should abort, not fall through.

    When the _remote_branch_exists probe fails (network/auth error), the dispatch should abort
    with an error instead of falling through to the fetch-fallback path (which would delete
    the local worktree and branch). This test exercises the actual recovery= path, not the
    pre-existing rework=True fetch-raise branch.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Simulate a killed-before-push session: create branch locally but don't push
    branch_name = "agent/issue-110-transient-probe"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Verify worktree exists locally
    assert info1.path.exists()
    assert (info1.path / "README.md").exists()

    # Break the origin remote to simulate a transient network error
    _git(repo_root, "remote", "set-url", "origin", "file:///nonexistent/path")

    # Recovery dispatch should abort with probe failure, NOT fall through to fetch-fallback
    with pytest.raises(
        RuntimeError, match="Failed to probe remote branch.*transient network or auth error"
    ):
        create_worktree(repo_root, branch_name, base_ref="HEAD", recovery=recovery_record)

    # Verify the local worktree and branch still exist (not deleted by fallback)
    assert info1.path.exists()
    assert (info1.path / "README.md").exists()

    # Verify the branch still exists
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Clean up
    _git(repo_root, "remote", "set-url", "origin", str(remote_repo))
    remove_worktree(repo_root, info1.path)


def test_fresh_dispatch_dirty_worktree_with_broken_remote_refuses_to_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #257: A dirty worktree must be refused even if the remote is broken.

    The salvage path is no longer used; the guard should refuse to reset before
    any push is attempted and leave the worktree intact.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create a worktree with dirty changes
    branch_name = "agent/issue-110-salvage-fail"
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

    # Break the origin remote to simulate push failure
    _git(repo_root, "remote", "set-url", "origin", "file:///nonexistent/path")

    # Fresh dispatch should refuse to reset the dirty worktree without trying to push.
    _force_capture_failure(monkeypatch)
    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted source work"):
        create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Verify the worktree still exists (not removed)
    assert info1.path.exists()
    # Verify the dirty file is still there
    assert (info1.path / "dirty.txt").read_text(encoding="utf-8") == "uncommitted\n"

    # Clean up
    _git(repo_root, "remote", "set-url", "origin", str(remote_repo))
    remove_worktree(repo_root, info1.path, force=True)


def test_dirty_probe_failure_refuses_to_reset(tmp_path: Path) -> None:
    """Issue #257: Failed dirty-probe should be treated as dirty and refused.

    When git status --porcelain fails (index lock, corruption, permissions),
    the guard should refuse to reset rather than risk discarding work. Per
    the issue #288 follow-up review (PR #314), this is raised as the distinct
    ``WorktreeProbeFailedError`` rather than ``WorktreeUnsafeError`` — a probe
    failure is transient contention, not a confirmed-dirty worktree, so the
    launch shim must classify it under a separate failure_kind that does not
    escalate to a human on first occurrence.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create a worktree with dirty changes
    branch_name = "agent/issue-110-dirty-probe-fail"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Add uncommitted changes
    (info1.path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    # Monkeypatch run_captured to simulate a failed git status probe
    import charlie_work.worktree

    original_run_captured = charlie_work.worktree.run_captured

    def mock_run_captured(*args: object, **kwargs: object) -> object:
        # If this is a git status --porcelain=v2 call, return a failure. Matches
        # by prefix since the flag carries a mode suffix ("--porcelain=v2").
        if (
            isinstance(args[0], list)
            and "status" in args[0]
            and any(str(a).startswith("--porcelain") for a in args[0])
        ):
            from charlie_work.subprocess_runner import RunResult

            return RunResult(
                returncode=1,
                stdout="",
                stderr="index.lock: File exists",
                error="index.lock: File exists",
            )
        return original_run_captured(*args, **kwargs)

    charlie_work.worktree.run_captured = mock_run_captured

    try:
        # Fresh dispatch should refuse to reset because the probe could not confirm
        # the worktree is clean — but as a distinct exception type from a
        # confirmed-dirty worktree (see WorktreeProbeFailedError docstring).
        with pytest.raises(WorktreeProbeFailedError, match="worktree status probe failed"):
            create_worktree(repo_root, branch_name, base_ref="HEAD")

        # Verify the worktree still exists (not removed)
        assert info1.path.exists()
        # Verify the dirty file is still there
        assert (info1.path / "dirty.txt").read_text(encoding="utf-8") == "uncommitted\n"
    finally:
        charlie_work.worktree.run_captured = original_run_captured


def test_is_confirmed_missing_ref_true_for_quiet_verify_miss() -> None:
    """Issue #1011: with ``-q``, ``git rev-parse --verify`` reserves exit code
    1 exclusively for "the given ref does not resolve" (confirmed against git
    2.45 for both an unborn ``HEAD`` in an empty repo and a missing branch
    name -- both produce ``returncode=1`` with empty stdout/stderr). This is
    the one shape the discriminator must treat as "nothing to lose".
    """
    result = RunResult(returncode=1, stdout="", stderr="", error="command exited 1")
    assert _is_confirmed_missing_ref(result) is True


def test_is_confirmed_missing_ref_false_for_git_error_exit_128() -> None:
    """A broken probe -- not a git repository, corrupted refs, permissions --
    exits 128, not 1. Before this fix, ANY non-ok `rev-parse --verify` result
    (including this one) fell through to "nothing to lose"; this pins that
    the discriminator refuses instead of misreading a broken probe as a
    confirmed-absent ref.
    """
    result = RunResult(
        returncode=128,
        stdout="",
        stderr="fatal: not a git repository (or any of the parent directories): .git",
        error="command exited 128",
    )
    assert _is_confirmed_missing_ref(result) is False


def test_is_confirmed_missing_ref_false_for_timeout() -> None:
    """A timed-out probe must never be read as a confirmed-absent ref."""
    result = RunResult(
        returncode=None,
        stdout="",
        stderr="",
        timed_out=True,
        error="command timed out after 60s",
    )
    assert _is_confirmed_missing_ref(result) is False


def test_is_confirmed_missing_ref_false_for_missing_git_binary() -> None:
    """An ``OSError`` from a git binary that is not on PATH at all comes back
    as ``returncode=None`` with ``error`` set (not a timeout). This must
    refuse too, not be misread as a confirmed-absent ref -- the discriminator
    must survive the case where git cannot be found at all, not just where it
    runs and errors.
    """
    result = RunResult(
        returncode=None,
        stdout="",
        stderr="",
        error="[WinError 2] The system cannot find the file specified",
    )
    assert _is_confirmed_missing_ref(result) is False


def test_local_tip_probe_failure_refuses_to_reset(tmp_path: Path) -> None:
    """Issue #1011: the confirmed data-loss path. When the local-tip
    ``git rev-parse --verify -q`` probe fails for a reason OTHER than a
    confirmed-absent ref (index lock, AV-held handle, transient I/O error --
    simulated here as a "not a git repository" failure, exit 128), a fresh
    branch that was never created still has no worktree directory to trip
    the dirty check, so this is the pre-fix code's only way in: before this
    fix, ANY non-ok probe result fell through to ``return None`` --
    "nothing to lose" -- and authorized discarding unpushed commits. It must
    now raise ``WorktreeProbeFailedError`` (refuse) instead.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    branch_name = "agent/issue-1011-probe-fail"
    _git(repo_root, "branch", branch_name)

    import charlie_work.worktree

    original_run_captured = charlie_work.worktree.run_captured

    def mock_run_captured(*args: object, **kwargs: object) -> object:
        if (
            isinstance(args[0], list)
            and args[0][:4] == ["git", "rev-parse", "--verify", "-q"]
            and args[0][-1] == branch_name
        ):
            return RunResult(
                returncode=128,
                stdout="",
                stderr="fatal: not a git repository (or any of the parent directories): .git",
                error="command exited 128",
            )
        return original_run_captured(*args, **kwargs)

    charlie_work.worktree.run_captured = mock_run_captured
    try:
        with pytest.raises(WorktreeProbeFailedError, match="not a confirmed-missing ref"):
            _worktree_refuse_to_reset_reason(repo_root, branch_name, "HEAD")
    finally:
        charlie_work.worktree.run_captured = original_run_captured


def test_absent_ref_still_returns_none_not_a_permanent_refusal(tmp_path: Path) -> None:
    """Issue #1011's other failure direction: a branch that genuinely does
    not exist (never dispatched, or already cleaned up) must still resolve
    to ``None`` ("nothing to lose") through the real ``git`` binary -- not
    mocked, unlike the probe-failure test above. Getting this backwards (
    treating every non-ok ``rev-parse --verify`` as a refusal) would make
    every legitimately-absent branch permanently un-reclaimable, leaking a
    worktree/reclaim slot forever.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    reason = _worktree_refuse_to_reset_reason(repo_root, "agent/does-not-exist-1011", "HEAD")
    assert reason is None


def test_salvage_worktree_with_origin_non_main_default_with_commit(tmp_path: Path) -> None:
    """Issue #141: _salvage_worktree with origin, non-main default, worktree has commit.

    This is test 1a of the differentiating pair: origin exists, default branch is 'master'
    (not 'main'), and the worktree has one commit beyond origin/master. Salvage should
    return non-None because there is work to preserve.
    """
    from charlie_work.worktree import _salvage_worktree

    # Create a remote repo with 'master' as the default branch
    remote_repo = tmp_path / "remote"
    remote_repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=master"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (remote_repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    # Clone the remote repo
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Set up origin/HEAD to point to origin/master
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Create a worktree with a commit beyond origin/master
    branch_name = "agent/issue-141-salvage-with-commit"
    worktrees_dir = _default_worktrees_dir(repo_root)
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = worktrees_dir / branch_name.replace("/", "-")
    subprocess.run(
        ["git", "worktree", "add", "-b", branch_name, str(worktree_path), "origin/master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Add a commit to the worktree (so it has unpushed commits)
    (worktree_path / "file1.txt").write_text("partial work\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "file1.txt"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "partial work"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )

    # Verify the branch does NOT exist on origin (killed-before-push scenario)
    ls_remote_result = subprocess.run(
        ["git", "ls-remote", "origin", f"refs/heads/{branch_name}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not ls_remote_result.stdout.strip()

    # Call _salvage_worktree directly
    salvage_ref = _salvage_worktree(repo_root, worktree_path, branch_name)

    # Should return a salvage ref (not None) because there are unpushed commits
    assert salvage_ref is not None
    assert salvage_ref.startswith("salvage/")

    # Verify the salvage ref was created locally
    salvage_refs = subprocess.run(
        ["git", "show-ref"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert salvage_refs.returncode == 0
    assert "salvage/" in salvage_refs.stdout or "refs/salvage/" in salvage_refs.stdout

    # Clean up
    remove_worktree(repo_root, worktree_path, force=True, branch=branch_name)


def test_salvage_worktree_with_origin_non_main_default_clean(tmp_path: Path) -> None:
    """Issue #141: _salvage_worktree with origin, non-main default, clean worktree.

    This is test 1b of the differentiating pair: origin exists, default branch is 'master'
    (not 'main'), and the worktree is CLEAN (exactly at origin/master tip). Salvage should
    return None because there is nothing to preserve.

    This is the REAL mutation gate: a hardcoded "main" (or any wrong ref) makes merge-base
    fail → safe-default True → non-None → this test FAILS.
    """
    from charlie_work.worktree import _salvage_worktree

    # Create a remote repo with 'master' as the default branch
    remote_repo = tmp_path / "remote"
    remote_repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=master"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (remote_repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    # Clone the remote repo
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Set up origin/HEAD to point to origin/master
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Create a worktree exactly at origin/master tip (clean)
    branch_name = "agent/issue-141-salvage-clean"
    worktrees_dir = _default_worktrees_dir(repo_root)
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = worktrees_dir / branch_name.replace("/", "-")
    subprocess.run(
        ["git", "worktree", "add", "-b", branch_name, str(worktree_path), "origin/master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Verify the worktree is clean (no commits beyond origin/master)
    merge_base_result = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/master"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    merge_base = merge_base_result.stdout.strip()
    rev_list_result = subprocess.run(
        ["git", "rev-list", "--count", f"{merge_base}..HEAD"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    commit_count = int(rev_list_result.stdout.strip())
    assert commit_count == 0, (
        f"Expected clean worktree, got {commit_count} commits beyond merge-base"
    )

    # Verify that merge-base with "main" (the hardcoded bug) would FAIL
    # This is the mutation gate: if the code uses "main" instead of "origin/master",
    # merge-base will fail and the test should catch it
    merge_base_main_result = subprocess.run(
        ["git", "merge-base", "HEAD", "main"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    assert merge_base_main_result.returncode != 0, (
        "merge-base with 'main' should fail in a 'master' repo"
    )

    # Call _salvage_worktree directly
    salvage_ref = _salvage_worktree(repo_root, worktree_path, branch_name)

    # Should return None because the worktree is clean
    assert salvage_ref is None

    # Clean up
    remove_worktree(repo_root, worktree_path, force=True, branch=branch_name)


def test_salvage_worktree_no_origin_conservative(tmp_path: Path) -> None:
    """Issue #141: _salvage_worktree with no origin returns non-None conservatively.

    When there is no origin remote, there is no authoritative default-branch tip to compare
    against. The salvage logic conservatively assumes there are unpushed commits to preserve
    work. This is an intentional, documented trade-off.
    """
    from charlie_work.worktree import _salvage_worktree

    # Create a repo with 'master' as the default branch (no origin)
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Create a worktree with commits
    branch_name = "agent/issue-141-salvage-no-origin"
    worktrees_dir = _default_worktrees_dir(repo_root)
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = worktrees_dir / branch_name.replace("/", "-")
    subprocess.run(
        ["git", "worktree", "add", "-b", branch_name, str(worktree_path), "master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Add a commit to the worktree
    (worktree_path / "file1.txt").write_text("partial work\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "file1.txt"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "partial work"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )

    # Verify no origin remote exists
    assert not _has_origin_remote(repo_root)

    # Call _salvage_worktree directly
    salvage_ref = _salvage_worktree(repo_root, worktree_path, branch_name)

    # Should return a salvage ref (not None) due to conservative no-origin default
    assert salvage_ref is not None
    assert salvage_ref.startswith("salvage/")

    # Verify the salvage ref was created locally
    salvage_refs = subprocess.run(
        ["git", "show-ref"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert salvage_refs.returncode == 0
    assert "salvage/" in salvage_refs.stdout or "refs/salvage/" in salvage_refs.stdout

    # Clean up
    remove_worktree(repo_root, worktree_path, force=True, branch=branch_name)
