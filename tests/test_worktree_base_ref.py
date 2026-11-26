"""Default-branch resolution and base-ref fetch behaviour of ``create_worktree``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _clone_repo, _git, _init_repo
from charlie_work.worktree import _resolve_default_branch_ref, create_worktree, remove_worktree


def test_resolve_default_branch_ref_with_origin(tmp_path: Path) -> None:
    """_resolve_default_branch_ref should return origin/<branch> when origin exists."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Set up origin/HEAD to point to origin/main
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    default_ref = _resolve_default_branch_ref(repo_root)
    assert default_ref == "origin/main"


def test_resolve_default_branch_ref_without_origin(tmp_path: Path) -> None:
    """_resolve_default_branch_ref should return HEAD when origin doesn't exist."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    default_ref = _resolve_default_branch_ref(repo_root)
    assert default_ref == "HEAD"


def test_resolve_default_branch_ref_heals_missing_origin_head(tmp_path: Path) -> None:
    """A clone whose origin/HEAD symref is unset gets healed via set-head --auto.

    Issue #239: this exact gap made fresh dispatches base on stale local HEAD.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Simulate the incident state: origin remote present, symref absent.
    subprocess.run(
        ["git", "symbolic-ref", "--delete", "refs/remotes/origin/HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    default_ref = _resolve_default_branch_ref(repo_root)
    assert default_ref == "origin/main"

    # The heal must persist in-repo, not just resolve transiently.
    symref = subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert symref.stdout.strip() == "refs/remotes/origin/main"


def test_resolve_default_branch_ref_raises_when_unhealable(tmp_path: Path) -> None:
    """Origin exists but is unreachable: raise instead of silently using local HEAD."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    _git(repo_root, "remote", "add", "origin", str(tmp_path / "does-not-exist"))

    with pytest.raises(RuntimeError, match="issue #239"):
        _resolve_default_branch_ref(repo_root)


def test_fresh_dispatch_autoresolve_ignores_stale_local_head(tmp_path: Path) -> None:
    """base_ref='' must base on the fetched origin tip even when origin/HEAD is
    unset and local HEAD is stale AND carries operator-only work (issue #239)."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Incident state 1: origin/HEAD symref absent on the clone.
    subprocess.run(
        ["git", "symbolic-ref", "--delete", "refs/remotes/origin/HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    # Incident state 2: the remote advanced past the local checkout.
    (remote_repo / "remote-only.txt").write_text("remote change\n", encoding="utf-8")
    _git(remote_repo, "add", "remote-only.txt")
    _git(remote_repo, "commit", "-m", "remote advance")
    remote_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()

    # Incident state 3: local main carries an unpublished operator commit.
    (repo_root / "operator-wip.txt").write_text("unpublished\n", encoding="utf-8")
    _git(repo_root, "add", "operator-wip.txt")
    _git(repo_root, "commit", "-m", "operator WIP never pushed")

    info = create_worktree(repo_root, "agent/issue-239-regression", base_ref="")

    worktree_tip = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    assert worktree_tip == remote_tip
    assert not (info.path / "operator-wip.txt").exists()
    assert (info.path / "remote-only.txt").exists()


def test_fresh_dispatch_with_base_ref_fetches_remote_ref(tmp_path: Path) -> None:
    """Fresh dispatch with base_ref=origin/main should fetch before worktree creation."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Set up origin/HEAD to point to origin/main
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Add a commit to the remote main branch
    (remote_repo / "remote-file.txt").write_text("remote change\n", encoding="utf-8")
    _git(remote_repo, "add", "remote-file.txt")
    _git(remote_repo, "commit", "-m", "add remote file")
    remote_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()

    # Park the local repo on a side branch with unique commits
    _git(repo_root, "checkout", "-b", "side-branch")
    (repo_root / "local-file.txt").write_text("local change\n", encoding="utf-8")
    _git(repo_root, "add", "local-file.txt")
    _git(repo_root, "commit", "-m", "add local file")
    local_tip = _git(repo_root, "rev-parse", "HEAD").stdout.strip()

    # Verify local is ahead of origin
    assert local_tip != remote_tip

    # Fresh dispatch with base_ref="" (auto-resolves to origin/main)
    branch_name = "agent/issue-103-fresh-fetch"
    info = create_worktree(repo_root, branch_name, base_ref="")

    # The worktree should be based on the remote tip, not the local side branch
    worktree_tip = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    assert worktree_tip == remote_tip
    assert worktree_tip != local_tip

    # Clean up
    remove_worktree(repo_root, info.path)


def test_fresh_dispatch_with_explicit_base_ref_fetches(tmp_path: Path) -> None:
    """Fresh dispatch with explicit base_ref=origin/main should fetch before worktree creation."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Add a commit to the remote main branch
    (remote_repo / "remote-file.txt").write_text("remote change\n", encoding="utf-8")
    _git(remote_repo, "add", "remote-file.txt")
    _git(remote_repo, "commit", "-m", "add remote file")
    remote_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()

    # Park the local repo on a side branch with unique commits
    _git(repo_root, "checkout", "-b", "side-branch")
    (repo_root / "local-file.txt").write_text("local change\n", encoding="utf-8")
    _git(repo_root, "add", "local-file.txt")
    _git(repo_root, "commit", "-m", "add local file")
    local_tip = _git(repo_root, "rev-parse", "HEAD").stdout.strip()

    # Fresh dispatch with explicit base_ref="origin/main"
    branch_name = "agent/issue-103-explicit-base-ref"
    info = create_worktree(repo_root, branch_name, base_ref="origin/main")

    # The worktree should be based on the remote tip, not the local side branch
    worktree_tip = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    assert worktree_tip == remote_tip
    assert worktree_tip != local_tip

    # Clean up
    remove_worktree(repo_root, info.path)


def test_fresh_dispatch_with_local_base_ref_does_not_fetch(tmp_path: Path) -> None:
    """Fresh dispatch with base_ref=HEAD should not fetch (local ref)."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Add a commit to the remote main branch
    (remote_repo / "remote-file.txt").write_text("remote change\n", encoding="utf-8")
    _git(remote_repo, "add", "remote-file.txt")
    _git(remote_repo, "commit", "-m", "add remote file")
    remote_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()

    # Park the local repo on a side branch with unique commits
    _git(repo_root, "checkout", "-b", "side-branch")
    (repo_root / "local-file.txt").write_text("local change\n", encoding="utf-8")
    _git(repo_root, "add", "local-file.txt")
    _git(repo_root, "commit", "-m", "add local file")
    local_tip = _git(repo_root, "rev-parse", "HEAD").stdout.strip()

    # Fresh dispatch with base_ref="HEAD" (local ref, no fetch)
    branch_name = "agent/issue-103-local-base-ref"
    info = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # The worktree should be based on the local tip, not the remote tip
    worktree_tip = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    assert worktree_tip == local_tip
    assert worktree_tip != remote_tip

    # Clean up
    remove_worktree(repo_root, info.path)


def test_rework_dispatch_does_not_fetch_base_ref(tmp_path: Path) -> None:
    """Rework dispatch should not fetch base_ref (preserves existing tip)."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create the agent branch + worktree locally and push it to origin.
    branch_name = "agent/issue-103-rework-no-fetch"
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

    # Rework dispatch with base_ref="" (should NOT fetch base_ref, only the branch itself)
    info2 = create_worktree(repo_root, branch_name, rework=True, base_ref="")

    # The worktree should be fast-forwarded to the origin tip (via rework logic),
    # but the base_ref fetch should not have happened (it's rework mode)
    assert info2.path == info1.path
    assert _git(info2.path, "rev-parse", "HEAD").stdout.strip() == remote_tip

    remove_worktree(repo_root, info1.path)


def test_recovery_dispatch_does_not_fetch_base_ref(tmp_path: Path) -> None:
    """Recovery dispatch should not fetch base_ref (only the branch itself in rework mode)."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create the agent branch + worktree locally with commits and push it
    branch_name = "agent/issue-103-recovery-no-fetch"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")
    (info1.path / "file1.txt").write_text("original\n", encoding="utf-8")
    _git(info1.path, "add", "file1.txt")
    _git(info1.path, "commit", "-m", "add file1")
    _git(repo_root, "push", "origin", branch_name)
    local_tip = _git(info1.path, "rev-parse", "HEAD").stdout.strip()

    # Add a commit to the remote main branch (this should NOT be fetched during recovery)
    _git(remote_repo, "checkout", "main")
    (remote_repo / "remote-file.txt").write_text("remote change\n", encoding="utf-8")
    _git(remote_repo, "add", "remote-file.txt")
    _git(remote_repo, "commit", "-m", "add remote file")
    remote_main_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()

    # Recovery dispatch with base_ref="" (should NOT fetch base_ref, only the branch itself)
    info2 = create_worktree(repo_root, branch_name, base_ref="", recovery=recovery_record)

    # The worktree should be reused (same path)
    assert info2.path == info1.path
    # The local commit should still be present (file1.txt exists)
    assert (info2.path / "file1.txt").read_text(encoding="utf-8") == "original\n"
    # The worktree tip should still be the local tip (recovery reuses existing worktree)
    assert _git(info2.path, "rev-parse", "HEAD").stdout.strip() == local_tip
    # The remote main tip should NOT have been fetched into the worktree
    assert _git(info2.path, "rev-parse", "HEAD").stdout.strip() != remote_main_tip

    remove_worktree(repo_root, info1.path)


def test_fresh_dispatch_fetch_failure_raises_error(tmp_path: Path) -> None:
    """Fresh dispatch with base_ref fetch failure should raise RuntimeError."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Set up origin/HEAD to point to origin/main
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Break the origin remote to simulate a fetch failure
    _git(repo_root, "remote", "set-url", "origin", "file:///nonexistent/path")

    # Fresh dispatch with base_ref="" should raise on fetch failure
    branch_name = "agent/issue-103-fetch-fail"
    with pytest.raises(RuntimeError, match="Failed to fetch base ref"):
        create_worktree(repo_root, branch_name, base_ref="")

    # Clean up
    _git(repo_root, "remote", "set-url", "origin", str(remote_repo))
