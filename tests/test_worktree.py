from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from _sessions_db_fixtures import make_sessions_db
from charlie_work.config import DevinConfig, OrchestratorConfig, PostMortemConfig, WatchdogConfig
from charlie_work.github import GitHubRunResult
from charlie_work.process_utils import get_process_start_time
from charlie_work.subprocess_runner import RunResult
from charlie_work import worktree as worktree_module
from charlie_work.worktree import (
    OPERATOR_MARKER_KIND,
    OPERATOR_MARKER_SESSION_ID,
    WorktreeCleanResult,
    WorktreeCleanGH,
    WorktreeInfo,
    WorktreeProbeFailedError,
    WorktreeState,
    WorktreeUnsafeError,
    LiveWorkerRedispatchError,
    ReworkMergeConflict,
    _default_worktrees_dir,
    _has_origin_remote,
    _resolve_default_branch_ref,
    _worker_authored_dirty,
    clean_worktrees,
    create_review_checkout,
    create_worktree,
    inspect_worktree_state,
    is_junction,
    list_worktrees,
    push_branch,
    remove_review_checkout,
    remove_worktree,
    verify_shared_venv,
    write_worktree_marker,
    _is_git_tracked,
    _materialize_directory,
    _slugify,
    _unlink_reparse_point,
)


def _init_repo(repo_root: Path, bare: bool = False) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )
    if bare:
        # Create a temporary non-bare repo, initialize it, then convert to bare
        temp_repo = repo_root.parent / f"{repo_root.name}-temp"
        temp_repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", "--initial-branch=main"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.test"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        (temp_repo / "README.md").write_text("hello\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "README.md"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "initial commit"],
            cwd=temp_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        # Convert to bare by cloning with --bare
        subprocess.run(
            ["git", "clone", "--bare", str(temp_repo), str(repo_root)],
            check=True,
            capture_output=True,
            text=True,
        )
        # Clean up temp repo (ignore errors on Windows due to file locks)
        import shutil

        shutil.rmtree(temp_repo, ignore_errors=True)
    else:
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


def test_create_worktree_no_venv_source_isolates_uv_sync_writes(tmp_path: Path) -> None:
    """Issue #274: with no venv_source, a worker's uv sync creates a local .venv
    and cannot poison the operator's shared venv editable-install metadata.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    pth = shared_venv / "Lib" / "site-packages" / "_editable_impl_charlie_work.pth"
    pth.parent.mkdir(parents=True)
    pth.write_text("operator/src\n", encoding="utf-8")

    info = create_worktree(
        repo_root, "agent/issue-274-no-junction", base_ref="HEAD", venv_source=None
    )

    # With the fix, no .venv junction is created at worktree creation time.
    assert info.venv_junction is None
    assert not (info.path / ".venv").exists()
    assert not is_junction(info.path / ".venv")

    # Simulate a worker running uv sync and installing an editable .pth in the
    # worktree's own (cold-built) .venv.
    local_venv = info.path / ".venv"
    local_venv.mkdir(parents=True)
    local_pth = local_venv / "Lib" / "site-packages" / "_editable_impl_charlie_work.pth"
    local_pth.parent.mkdir(parents=True)
    local_pth.write_text("worktree/src\n", encoding="utf-8")

    # The shared venv's editable .pth must remain untouched.
    assert pth.read_text(encoding="utf-8") == "operator/src\n"


def test_rework_unlinks_pre_existing_venv_junction_when_venv_source_none(
    tmp_path: Path,
) -> None:
    """Issue #274: reusing a worktree with no venv_source must remove a pre-existing
    .venv junction so the worker's uv sync writes to a local .venv instead of
    poisoning the operator's shared venv.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    pth = shared_venv / "Lib" / "site-packages" / "_editable_impl_charlie_work.pth"
    pth.parent.mkdir(parents=True)
    pth.write_text("operator/src\n", encoding="utf-8")

    branch_name = "agent/issue-274-reuse-no-junction"

    # Pre-PR era: worktree was created with a shared-venv junction.
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD", venv_source=shared_venv)
    assert info1.venv_junction == info1.path / ".venv"
    assert is_junction(info1.venv_junction)

    # Re-dispatch with the new default (venv_source=None) should unlink the
    # junction, not follow it, so the shared venv is untouched.
    info2 = create_worktree(repo_root, branch_name, rework=True, venv_source=None)

    assert info2.path == info1.path
    assert info2.venv_junction is None
    assert not is_junction(info2.path / ".venv")
    assert not (info2.path / ".venv").exists()

    # The shared venv's editable .pth must remain untouched and unreachable.
    assert pth.read_text(encoding="utf-8") == "operator/src\n"
    assert not (
        info2.path / ".venv" / "Lib" / "site-packages" / "_editable_impl_charlie_work.pth"
    ).exists()

    # Simulate a worker running uv sync into a now-isolated per-worktree venv.
    local_venv = info2.path / ".venv"
    local_venv.mkdir(parents=True)
    local_pth = local_venv / "Lib" / "site-packages" / "_editable_impl_charlie_work.pth"
    local_pth.parent.mkdir(parents=True)
    local_pth.write_text("worktree/src\n", encoding="utf-8")

    # The shared venv's editable .pth must still be untouched.
    assert pth.read_text(encoding="utf-8") == "operator/src\n"

    # Clean up
    remove_worktree(repo_root, info2.path)


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


def test_rework_merge_update_conflicts_with_local_base(tmp_path: Path) -> None:
    """Rework mode must NOT abort the launch when the branch conflicts with the
    local base (no origin remote): the merge is aborted internally and the
    worktree is returned with a populated ``rework_conflict`` notice instead of
    raising ReworkBranchConflictError (see worktree.ReworkMergeConflict)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-rework-conflict-local"
    info1 = create_worktree(repo_root, branch_name, base_ref="")
    (info1.path / "file.txt").write_text("feature line\n", encoding="utf-8")
    _git(info1.path, "add", "file.txt")
    _git(info1.path, "commit", "-m", "add feature")
    pre_merge_head = _git(info1.path, "rev-parse", "HEAD").stdout.strip()

    # Advance the local base branch with a conflicting edit.
    _git(repo_root, "checkout", "main")
    (repo_root / "file.txt").write_text("main line\n", encoding="utf-8")
    _git(repo_root, "add", "file.txt")
    _git(repo_root, "commit", "-m", "advance main")

    info2 = create_worktree(repo_root, branch_name, rework=True, base_ref="")

    assert isinstance(info2.rework_conflict, ReworkMergeConflict)
    assert "file.txt" in info2.rework_conflict.conflicted_files
    assert info2.rework_conflict.base_branch == "main"

    # The branch head must be unchanged, and no merge left mid-flight.
    post_merge_head = _git(info2.path, "rev-parse", "HEAD").stdout.strip()
    assert post_merge_head == pre_merge_head
    merge_head_check = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"],
        cwd=info2.path,
        capture_output=True,
        text=True,
    )
    assert merge_head_check.returncode != 0

    remove_worktree(repo_root, info1.path)


def test_rework_merge_update_conflicts_with_remote_base(tmp_path: Path) -> None:
    """Rework mode must NOT abort the launch when the branch conflicts with the
    origin base: the merge is aborted internally and the worktree is returned
    with a populated ``rework_conflict`` notice instead of raising
    ReworkBranchConflictError (see worktree.ReworkMergeConflict)."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    branch_name = "agent/issue-rework-conflict-remote"
    info1 = create_worktree(repo_root, branch_name, base_ref="")
    (info1.path / "file.txt").write_text("feature line\n", encoding="utf-8")
    _git(info1.path, "add", "file.txt")
    _git(info1.path, "commit", "-m", "add feature")
    _git(repo_root, "push", "origin", branch_name)
    pre_merge_head = _git(info1.path, "rev-parse", "HEAD").stdout.strip()

    # Advance origin/main with a conflicting edit.
    _git(remote_repo, "checkout", "main")
    (remote_repo / "file.txt").write_text("main line\n", encoding="utf-8")
    _git(remote_repo, "add", "file.txt")
    _git(remote_repo, "commit", "-m", "advance main")
    _git(remote_repo, "checkout", branch_name)

    info2 = create_worktree(repo_root, branch_name, rework=True, base_ref="")

    assert isinstance(info2.rework_conflict, ReworkMergeConflict)
    assert "file.txt" in info2.rework_conflict.conflicted_files
    assert info2.rework_conflict.base_branch == "main"

    post_merge_head = _git(info2.path, "rev-parse", "HEAD").stdout.strip()
    assert post_merge_head == pre_merge_head
    merge_head_check = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"],
        cwd=info2.path,
        capture_output=True,
        text=True,
    )
    assert merge_head_check.returncode != 0

    remove_worktree(repo_root, info1.path)


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


def test_rework_reuse_resets_on_non_ff_identical_patch_id(tmp_path: Path) -> None:
    """Rework reuse path must reset an existing worktree when local branch diverged
    non-FF from origin but the patch-id is identical (e.g. rebase-only rewrite)."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    branch_name = "agent/issue-451-reuse-identical"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")
    (info1.path / "feature.txt").write_text("feature content\n", encoding="utf-8")
    _git(info1.path, "add", "feature.txt")
    _git(info1.path, "commit", "-m", "add feature")
    _git(repo_root, "push", "origin", branch_name)

    # Advance main on remote and rebase feature with the same content.
    _git(remote_repo, "checkout", "main")
    (remote_repo / "base.txt").write_text("base v2\n", encoding="utf-8")
    _git(remote_repo, "add", "base.txt")
    _git(remote_repo, "commit", "-m", "advance main")
    _git(remote_repo, "checkout", branch_name)
    _git(remote_repo, "rebase", "main")
    remote_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()
    _git(remote_repo, "checkout", "main")

    info2 = create_worktree(
        repo_root,
        branch_name,
        rework=True,
        base_ref="",
        issue_number=451,
    )

    assert info2.path == info1.path
    assert _git(info2.path, "rev-parse", "HEAD").stdout.strip() == remote_tip
    assert (info2.path / "feature.txt").read_text(encoding="utf-8") == "feature content\n"
    assert (info2.path / "base.txt").read_text(encoding="utf-8") == "base v2\n"
    assert info2.reclaimed == "reset-origin:identical-patch-id"
    assert info2.attempt_snapshot is not None
    assert info2.attempt_snapshot.ref_name is not None

    remove_worktree(repo_root, info1.path)


def test_rework_reuse_resets_on_non_ff_different_patch_id(tmp_path: Path) -> None:
    """Rework reuse path must reset to origin and report a different patch-id
    when the local-only commits genuinely diverge from the rebased origin tip."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    branch_name = "agent/issue-451-reuse-different"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")
    (info1.path / "feature.txt").write_text("local version\n", encoding="utf-8")
    _git(info1.path, "add", "feature.txt")
    _git(info1.path, "commit", "-m", "add feature")
    _git(repo_root, "push", "origin", branch_name)

    # Remote rebases and changes the feature commit content.
    _git(remote_repo, "checkout", "main")
    (remote_repo / "base.txt").write_text("base v2\n", encoding="utf-8")
    _git(remote_repo, "add", "base.txt")
    _git(remote_repo, "commit", "-m", "advance main")
    _git(remote_repo, "checkout", branch_name)
    _git(remote_repo, "rebase", "main")
    (remote_repo / "feature.txt").write_text("remote version\n", encoding="utf-8")
    _git(remote_repo, "add", "feature.txt")
    _git(remote_repo, "commit", "--amend", "-m", "add feature (remote)")
    remote_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()
    _git(remote_repo, "checkout", "main")

    info2 = create_worktree(
        repo_root,
        branch_name,
        rework=True,
        base_ref="",
        issue_number=451,
    )

    assert info2.path == info1.path
    assert _git(info2.path, "rev-parse", "HEAD").stdout.strip() == remote_tip
    assert (info2.path / "feature.txt").read_text(encoding="utf-8") == "remote version\n"
    assert (info2.path / "base.txt").read_text(encoding="utf-8") == "base v2\n"
    assert info2.reclaimed == "reset-origin:different-patch-id"
    assert info2.attempt_snapshot is not None
    assert info2.attempt_snapshot.ref_name is not None

    remove_worktree(repo_root, info1.path)


def test_rework_reuse_refuses_non_ff_with_dirty_worktree(tmp_path: Path) -> None:
    """Rework reuse path must refuse to reset an existing worktree that has
    uncommitted modifications and is non-FF diverged from origin."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    branch_name = "agent/issue-451-reuse-dirty"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")
    (info1.path / "feature.txt").write_text("feature content\n", encoding="utf-8")
    _git(info1.path, "add", "feature.txt")
    _git(info1.path, "commit", "-m", "add feature")
    _git(repo_root, "push", "origin", branch_name)

    # Leave an uncommitted edit in the existing worktree.
    (info1.path / "dirty.txt").write_text("uncommitted worker edit\n", encoding="utf-8")

    # Diverge origin so the reuse path cannot fast-forward.
    _git(remote_repo, "checkout", branch_name)
    (remote_repo / "feature.txt").write_text("remote version\n", encoding="utf-8")
    _git(remote_repo, "add", "feature.txt")
    _git(remote_repo, "commit", "--amend", "-m", "add feature (remote)")
    _git(remote_repo, "checkout", "main")

    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted modifications"):
        create_worktree(
            repo_root,
            branch_name,
            rework=True,
            base_ref="",
            issue_number=451,
        )

    # The dirty file must survive untouched.
    assert info1.path.exists()
    assert (info1.path / "dirty.txt").read_text(encoding="utf-8") == "uncommitted worker edit\n"

    remove_worktree(repo_root, info1.path)


def test_rework_attach_resets_on_non_ff_identical_patch_id(tmp_path: Path) -> None:
    """Rework attach path must reset a non-FF diverged local branch ref to the
    origin tip when the patch-id is identical."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    branch_name = "agent/issue-451-attach-identical"
    _git(repo_root, "checkout", "-b", branch_name)
    (repo_root / "feature.txt").write_text("feature content\n", encoding="utf-8")
    _git(repo_root, "add", "feature.txt")
    _git(repo_root, "commit", "-m", "add feature")
    _git(repo_root, "push", "origin", branch_name)
    _git(repo_root, "checkout", "main")

    # Advance main on remote and rebase feature with the same content.
    _git(remote_repo, "checkout", "main")
    (remote_repo / "base.txt").write_text("base v2\n", encoding="utf-8")
    _git(remote_repo, "add", "base.txt")
    _git(remote_repo, "commit", "-m", "advance main")
    _git(remote_repo, "checkout", branch_name)
    _git(remote_repo, "rebase", "main")
    remote_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()
    _git(remote_repo, "checkout", "main")

    info = create_worktree(
        repo_root,
        branch_name,
        rework=True,
        base_ref="",
        issue_number=451,
    )

    assert info.branch == branch_name
    assert _git(info.path, "rev-parse", "HEAD").stdout.strip() == remote_tip
    assert (info.path / "feature.txt").read_text(encoding="utf-8") == "feature content\n"
    assert (info.path / "base.txt").read_text(encoding="utf-8") == "base v2\n"
    assert info.reclaimed == "reset-origin:identical-patch-id"
    assert info.attempt_snapshot is not None
    assert info.attempt_snapshot.ref_name is not None

    remove_worktree(repo_root, info.path)


def test_rework_attach_resets_on_non_ff_different_patch_id(tmp_path: Path) -> None:
    """Rework attach path must reset a non-FF diverged local branch ref to the
    origin tip and report a different patch-id."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    branch_name = "agent/issue-451-attach-different"
    _git(repo_root, "checkout", "-b", branch_name)
    (repo_root / "feature.txt").write_text("local version\n", encoding="utf-8")
    _git(repo_root, "add", "feature.txt")
    _git(repo_root, "commit", "-m", "add feature")
    _git(repo_root, "push", "origin", branch_name)
    _git(repo_root, "checkout", "main")

    # Remote rebases and changes the feature commit content.
    _git(remote_repo, "checkout", "main")
    (remote_repo / "base.txt").write_text("base v2\n", encoding="utf-8")
    _git(remote_repo, "add", "base.txt")
    _git(remote_repo, "commit", "-m", "advance main")
    _git(remote_repo, "checkout", branch_name)
    _git(remote_repo, "rebase", "main")
    (remote_repo / "feature.txt").write_text("remote version\n", encoding="utf-8")
    _git(remote_repo, "add", "feature.txt")
    _git(remote_repo, "commit", "--amend", "-m", "add feature (remote)")
    remote_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()
    _git(remote_repo, "checkout", "main")

    info = create_worktree(
        repo_root,
        branch_name,
        rework=True,
        base_ref="",
        issue_number=451,
    )

    assert info.branch == branch_name
    assert _git(info.path, "rev-parse", "HEAD").stdout.strip() == remote_tip
    assert (info.path / "feature.txt").read_text(encoding="utf-8") == "remote version\n"
    assert (info.path / "base.txt").read_text(encoding="utf-8") == "base v2\n"
    assert info.reclaimed == "reset-origin:different-patch-id"
    assert info.attempt_snapshot is not None
    assert info.attempt_snapshot.ref_name is not None

    remove_worktree(repo_root, info.path)


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


def test_rework_reclaim_refuses_dirty_detached_worktree(tmp_path: Path) -> None:
    """A detached leftover worktree with uncommitted work must not be silently
    clobbered by the by-path reclaim — WorktreeUnsafeError, work survives."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-10-dirty-detach"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")
    _git(info1.path, "checkout", "--detach")
    (info1.path / "dirty.txt").write_text("uncommitted worker edit\n", encoding="utf-8")

    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted modifications"):
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

    # Create a valid venv_source but monkeypatch junction creation to fail
    venv_source = tmp_path / "shared-venv"
    venv_source.mkdir()
    branch_name = "agent/issue-3-junction-fail"

    # Monkeypatch the junction creation function to raise OSError
    import charlie_work.worktree

    original_create = charlie_work.worktree._create_junction_or_symlink

    def mock_create_junction(*args: object, **kwargs: object) -> None:
        raise OSError("Mock junction creation failure")

    charlie_work.worktree._create_junction_or_symlink = mock_create_junction

    try:
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
    finally:
        charlie_work.worktree._create_junction_or_symlink = original_create


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

    # Create a valid venv_source but monkeypatch junction creation to fail
    venv_source = tmp_path / "shared-venv"
    venv_source.mkdir()

    # Monkeypatch the junction creation function to raise OSError
    import charlie_work.worktree

    original_create = charlie_work.worktree._create_junction_or_symlink

    def mock_create_junction(*args: object, **kwargs: object) -> None:
        raise OSError("Mock junction creation failure")

    charlie_work.worktree._create_junction_or_symlink = mock_create_junction

    try:
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
    finally:
        charlie_work.worktree._create_junction_or_symlink = original_create

    # Clean up
    subprocess.run(
        ["git", "branch", "-D", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
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


def test_redispatch_refuses_to_reset_with_unpushed_commits(tmp_path: Path) -> None:
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


def test_second_redispatch_refuses_with_unpushed_commits(tmp_path: Path) -> None:
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


def test_fresh_dispatch_dirty_worktree_refuses_to_reset(tmp_path: Path) -> None:
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
    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted modifications"):
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


def test_fresh_dispatch_still_refuses_worker_authored_dirtiness(tmp_path: Path) -> None:
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

    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted modifications"):
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
    tmp_path: Path,
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
    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted modifications"):
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


def test_list_worktrees_porcelain_parser_handles_flag_lines(tmp_path: Path) -> None:
    """Porcelain parser should handle flag lines (bare, detached, prunable) correctly."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create a worktree to ensure we have porcelain output to parse
    info = create_worktree(repo_root, "agent/issue-131-flags", base_ref="HEAD")

    # The parser should handle flag lines without crashing
    worktrees = list_worktrees(repo_root)

    # Should have at least 2 worktrees (main + the new one)
    assert len(worktrees) >= 2

    # Each worktree entry should have a worktree key with a Path value
    for wt in worktrees:
        assert "worktree" in wt
        assert isinstance(wt["worktree"], Path)

    # Flag keys like "bare", "detached" should be present as True if applicable
    # (main worktree is typically not bare/detached, but the parser should handle them)
    for wt in worktrees:
        if "bare" in wt:
            assert wt["bare"] is True
        if "detached" in wt:
            assert wt["detached"] is True

    # Clean up
    remove_worktree(repo_root, info.path)


# ---------------------------------------------------------------------------
# Tests for materialize_dirs functionality
# ---------------------------------------------------------------------------


def test_is_git_tracked_returns_true_for_tracked_file(tmp_path: Path) -> None:
    """_is_git_tracked should return True for tracked files."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # README.md is tracked by default
    assert _is_git_tracked(repo_root, repo_root / "README.md") is True


def test_is_git_tracked_returns_false_for_untracked_file(tmp_path: Path) -> None:
    """_is_git_tracked should return False for untracked files."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create an untracked file
    untracked = repo_root / "untracked.txt"
    untracked.write_text("untracked\n", encoding="utf-8")

    assert _is_git_tracked(repo_root, untracked) is False


def test_is_git_tracked_returns_false_for_nonexistent_path(tmp_path: Path) -> None:
    """_is_git_tracked should return False for nonexistent paths."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    nonexistent = repo_root / "does-not-exist.txt"
    assert _is_git_tracked(repo_root, nonexistent) is False


def test_materialize_directory_copies_untracked_dir(tmp_path: Path) -> None:
    """_materialize_directory should copy untracked directories."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create an untracked directory in repo_root
    untracked_dir = repo_root / ".devin"
    untracked_dir.mkdir()
    (untracked_dir / "config.json").write_text("config\n", encoding="utf-8")

    # Create a worktree
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    # Materialize the directory
    _materialize_directory(repo_root, worktree_path, ".devin")

    # Verify the directory was copied
    target_dir = worktree_path / ".devin"
    assert target_dir.exists()
    assert (target_dir / "config.json").read_text(encoding="utf-8") == "config\n"


def test_materialize_directory_preserves_tracked_files_and_copies_untracked_infra(
    tmp_path: Path,
) -> None:
    """_materialize_directory should copy untracked infra but not overwrite
    tracked files that are already identical in the target."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "committed worker\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "config.json").write_text("{}\n", encoding="utf-8")
    _git(repo_root, "add", ".devin/prompts/worker.md")
    _git(repo_root, "commit", "-m", "add prompt template")

    # Pre-existing target with the same tracked prompt template
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".devin" / "prompts").mkdir(parents=True)
    (worktree_path / ".devin" / "prompts" / "worker.md").write_text(
        "committed worker\n", encoding="utf-8"
    )

    written = _materialize_directory(repo_root, worktree_path, ".devin")

    # The tracked prompt template should be preserved (not overwritten)
    assert (worktree_path / ".devin" / "prompts" / "worker.md").read_text(
        encoding="utf-8"
    ) == "committed worker\n"
    # The untracked infra file should be copied
    assert (worktree_path / ".devin" / "config.json").read_text(encoding="utf-8") == "{}\n"
    assert ".devin/config.json" in written
    assert ".devin/prompts/worker.md" not in written


def test_materialize_directory_skips_nonexistent_source(tmp_path: Path) -> None:
    """_materialize_directory should skip nonexistent source directories."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    # Try to materialize a nonexistent directory - should not raise
    _materialize_directory(repo_root, worktree_path, "does-not-exist")

    # Worktree should still be empty
    assert not (worktree_path / "does-not-exist").exists()


def test_materialize_directory_merges_into_existing_target(tmp_path: Path) -> None:
    """_materialize_directory should copy missing files and preserve existing ones."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create an untracked directory in repo_root
    untracked_dir = repo_root / ".devin"
    untracked_dir.mkdir()
    (untracked_dir / "config.json").write_text("original\n", encoding="utf-8")

    # Create a worktree with a pre-existing .devin directory
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    target_dir = worktree_path / ".devin"
    target_dir.mkdir()
    (target_dir / "existing.txt").write_text("existing\n", encoding="utf-8")

    # Materialize the directory - should merge, not skip wholesale
    written = _materialize_directory(repo_root, worktree_path, ".devin")

    # The existing file should still be there (not overwritten)
    assert (target_dir / "existing.txt").read_text(encoding="utf-8") == "existing\n"
    # The missing source file should have been copied
    assert (target_dir / "config.json").read_text(encoding="utf-8") == "original\n"
    assert ".devin/config.json" in written


def test_materialize_directory_preserves_empty_subdirectories(tmp_path: Path) -> None:
    """Empty source subdirectories must be recreated in the target (copytree did)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / ".devin" / "empty_subdir").mkdir(parents=True)
    (repo_root / ".devin" / "config.json").write_text("{}\n", encoding="utf-8")

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    _materialize_directory(repo_root, worktree_path, ".devin")

    assert (worktree_path / ".devin" / "empty_subdir").is_dir()
    assert not any((worktree_path / ".devin" / "empty_subdir").iterdir())
    assert (worktree_path / ".devin" / "config.json").read_text(encoding="utf-8") == "{}\n"


def test_create_worktree_with_materialize_dirs(tmp_path: Path) -> None:
    """create_worktree should materialize specified directories."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create an untracked directory in repo_root
    untracked_dir = repo_root / ".devin"
    untracked_dir.mkdir()
    (untracked_dir / "hooks.json").write_text("hooks\n", encoding="utf-8")

    # Create a worktree with materialize_dirs
    info = create_worktree(
        repo_root,
        "agent/issue-1-materialize",
        base_ref="HEAD",
        materialize_dirs=(".devin",),
    )

    # Verify the directory was copied to the worktree
    target_dir = info.path / ".devin"
    assert target_dir.exists()
    assert (target_dir / "hooks.json").read_text(encoding="utf-8") == "hooks\n"
    assert ".devin/hooks.json" in info.materialized_paths


def _simulated_require_ci_clean(worktree_path: Path) -> bool:
    """Return True if the worktree has no staged or unstaged tracked changes."""
    staged = subprocess.run(
        ["git", "diff-index", "--cached", "--exit-code", "HEAD"],
        cwd=worktree_path,
        check=False,
        capture_output=True,
        text=True,
    )
    unstaged = subprocess.run(
        ["git", "diff", "--exit-code"],
        cwd=worktree_path,
        check=False,
        capture_output=True,
        text=True,
    )
    return staged.returncode == 0 and unstaged.returncode == 0


def test_materialize_directory_overwrites_tracked_prompt_and_hides_dirt(
    tmp_path: Path,
) -> None:
    """Issue #469: per-dispatch prompt files injected over tracked templates
    must not leave the worktree dirty in ``git status``."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "committed worker\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "prompts" / "rework.md").write_text(
        "committed rework\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "config.json").write_text("{}\n", encoding="utf-8")
    _git(repo_root, "add", "-f", ".devin/prompts/worker.md", ".devin/prompts/rework.md")
    _git(repo_root, "commit", "-m", "add prompt templates")

    # Simulate the orchestrator writing per-dispatch prompt content into the
    # source tree (the content that the materializer will copy).
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "per-dispatch worker\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "prompts" / "rework.md").write_text(
        "per-dispatch rework\n", encoding="utf-8"
    )

    # Create a target worktree that already has the committed templates tracked.
    worktree_path = tmp_path / "wt"
    _git(repo_root, "clone", str(repo_root), str(worktree_path))
    # Re-initialize the clone as a normal repo so git status is meaningful there.
    _git(worktree_path, "config", "user.email", "test@example.test")
    _git(worktree_path, "config", "user.name", "Test User")

    written = _materialize_directory(repo_root, worktree_path, ".devin")

    assert set(written) == {
        ".devin/prompts/worker.md",
        ".devin/prompts/rework.md",
        ".devin/config.json",
    }
    assert (worktree_path / ".devin" / "prompts" / "worker.md").read_text(
        encoding="utf-8"
    ) == "per-dispatch worker\n"
    assert (worktree_path / ".devin" / "prompts" / "rework.md").read_text(
        encoding="utf-8"
    ) == "per-dispatch rework\n"

    # The copied prompt files should be marked assume-unchanged in the target.
    ls_files = _git(worktree_path, "ls-files", "-v", ".devin/prompts/worker.md")
    assert ls_files.stdout.startswith("h ")

    # No tracked-file dirt visible to git status or a strict CI clean-tree gate.
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
    assert _simulated_require_ci_clean(worktree_path)


def test_create_worktree_materialize_dirs_injects_prompt_and_keeps_clean_tree(
    tmp_path: Path,
) -> None:
    """create_worktree with materialize_dirs must inject per-dispatch prompt
    content over tracked templates while leaving the worktree clean."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "committed worker\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "prompts" / "rework.md").write_text(
        "committed rework\n", encoding="utf-8"
    )
    _git(repo_root, "add", "-f", ".devin/prompts/worker.md", ".devin/prompts/rework.md")
    _git(repo_root, "commit", "-m", "add prompt templates")

    # Per-dispatch prompt content in the source tree.
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "per-dispatch worker\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "prompts" / "rework.md").write_text(
        "per-dispatch rework\n", encoding="utf-8"
    )

    info = create_worktree(
        repo_root,
        "agent/issue-469-materialize",
        base_ref="HEAD",
        materialize_dirs=(".devin",),
    )

    assert set(info.materialized_paths) == {
        ".devin/prompts/worker.md",
        ".devin/prompts/rework.md",
    }
    assert (info.path / ".devin" / "prompts" / "worker.md").read_text(
        encoding="utf-8"
    ) == "per-dispatch worker\n"

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=info.path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
    assert _simulated_require_ci_clean(info.path)


def test_create_worktree_materialize_dirs_survives_external_tracked_prompt_write(
    tmp_path: Path,
) -> None:
    """Issue #469: tracked prompt files overwritten by an external shim after
    create_worktree returns must not dirty the worktree.

    The assume-unchanged bit must be applied proactively to every tracked path
    under the configured materialize surface at worktree-creation time, not only
    to paths whose content differed from the source tree during the copy loop.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / ".devin" / "prompts").mkdir(parents=True)
    (repo_root / ".devin" / "prompts" / "worker.md").write_text(
        "committed worker\n", encoding="utf-8"
    )
    (repo_root / ".devin" / "prompts" / "rework.md").write_text(
        "committed rework\n", encoding="utf-8"
    )
    _git(repo_root, "add", "-f", ".devin/prompts/worker.md", ".devin/prompts/rework.md")
    _git(repo_root, "commit", "-m", "add prompt templates")

    info = create_worktree(
        repo_root,
        "agent/issue-469-shim-write",
        base_ref="HEAD",
        materialize_dirs=(".devin",),
    )

    # Source content matches HEAD, so the materializer does not rewrite anything.
    assert info.materialized_paths == ()

    # Simulate the external launch shim writing per-dispatch content into the
    # worktree after create_worktree has already returned.
    (info.path / ".devin" / "prompts" / "worker.md").write_text("shim worker\n", encoding="utf-8")

    ls_files = _git(info.path, "ls-files", "-v", ".devin/prompts/worker.md")
    assert ls_files.stdout.startswith("h ")

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=info.path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
    assert _simulated_require_ci_clean(info.path)


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


def test_list_worktrees_porcelain_parser_handles_malformed_worktree_line(tmp_path: Path) -> None:
    """Porcelain parser should drop malformed worktree entries entirely.

    When a worktree line is malformed (bare "worktree" with no path), the entire
    entry is dropped from the result. Sibling valid entries are unaffected.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Monkeypatch run_captured to return malformed porcelain output
    import charlie_work.worktree

    original_run_captured = charlie_work.worktree.run_captured

    def mock_run_captured(*args: object, **kwargs: object) -> object:
        # If this is a git worktree list --porcelain call, return malformed output
        if isinstance(args[0], list) and "worktree" in args[0] and "list" in args[0]:
            from charlie_work.subprocess_runner import RunResult

            # Malformed output: a bare "worktree" line with no path (flag-style)
            malformed_output = """worktree
bare
HEAD abc123
branch refs/heads/main

worktree /path/to/valid
HEAD def456
branch refs/heads/feature
"""
            return RunResult(
                returncode=0,
                stdout=malformed_output,
                stderr="",
                error=None,
            )
        return original_run_captured(*args, **kwargs)

    charlie_work.worktree.run_captured = mock_run_captured

    try:
        # The parser should not crash on malformed input
        worktrees = list_worktrees(repo_root)

        # Should parse only the valid entry; the malformed one is dropped entirely
        assert len(worktrees) == 1
        # The valid entry should be present
        assert worktrees[0]["worktree"] == Path("/path/to/valid")
        assert worktrees[0]["branch"] == "refs/heads/feature"
    finally:
        charlie_work.worktree.run_captured = original_run_captured


def test_list_worktrees_porcelain_parser_drops_unknown_flag_keys(tmp_path: Path) -> None:
    """Porcelain parser should drop entries with unknown space-less keys.

    Unknown flag keys (not in KNOWN_FLAG_KEYS) mark the entire entry as malformed
    and cause it to be dropped. Known flag lines (bare, detached) parse as True.
    Valued forms (prunable <reason>, locked <reason>) parse as str.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Monkeypatch run_captured to return porcelain output with unknown keys
    import charlie_work.worktree

    original_run_captured = charlie_work.worktree.run_captured

    def mock_run_captured(*args: object, **kwargs: object) -> object:
        # If this is a git worktree list --porcelain call, return output with unknown keys
        if isinstance(args[0], list) and "worktree" in args[0] and "list" in args[0]:
            from charlie_work.subprocess_runner import RunResult

            # Output with unknown flag key "garbage" and valued forms
            malformed_output = """worktree /path/to/valid1
HEAD abc123
branch refs/heads/feature1
bare

worktree /path/to/valid2
HEAD def456
branch refs/heads/feature2
garbage

worktree /path/to/valid3
HEAD ghi789
branch refs/heads/feature3
prunable some reason

worktree /path/to/valid4
HEAD jkl012
branch refs/heads/feature4
locked another reason
"""
            return RunResult(
                returncode=0,
                stdout=malformed_output,
                stderr="",
                error=None,
            )
        return original_run_captured(*args, **kwargs)

    charlie_work.worktree.run_captured = mock_run_captured

    try:
        worktrees = list_worktrees(repo_root)

        # Should parse only entries 1, 3, 4; entry 2 with unknown "garbage" key is dropped
        assert len(worktrees) == 3
        # Entry 1: valid with known flag "bare"
        assert worktrees[0]["worktree"] == Path("/path/to/valid1")
        assert worktrees[0]["bare"] is True
        # Entry 3: valid with valued "prunable" (not in KNOWN_FLAG_KEYS, but has a value)
        assert worktrees[1]["worktree"] == Path("/path/to/valid3")
        assert worktrees[1]["prunable"] == "some reason"
        # Entry 4: valid with valued "locked" (not in KNOWN_FLAG_KEYS, but has a value)
        assert worktrees[2]["worktree"] == Path("/path/to/valid4")
        assert worktrees[2]["locked"] == "another reason"
    finally:
        charlie_work.worktree.run_captured = original_run_captured


def test_list_worktrees_consumer_path_safe_with_malformed_entries(tmp_path: Path) -> None:
    """Consumer-path test: fresh-dispatch lookup with malformed porcelain output.

    Drives the real fresh-dispatch/existing-worktree lookup code (around L529 in
    worktree.py) with porcelain output containing one malformed entry and one valid
    entry. Verifies no KeyError occurs and the valid entry is triaged correctly.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create a worktree to establish a valid path
    branch_name = "agent/issue-131-consumer"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Monkeypatch run_captured to return porcelain output with a malformed entry
    import charlie_work.worktree

    original_run_captured = charlie_work.worktree.run_captured

    def mock_run_captured(*args: object, **kwargs: object) -> object:
        # If this is a git worktree list --porcelain call, return output with malformed entry
        if isinstance(args[0], list) and "worktree" in args[0] and "list" in args[0]:
            from charlie_work.subprocess_runner import RunResult

            # Output with a malformed entry (bare "worktree") and the real valid entry
            # We need to include the actual worktree path from info1
            malformed_output = f"""worktree
bare
HEAD abc123
branch refs/heads/malformed

worktree {info1.path}
HEAD def456
branch refs/heads/{branch_name}
"""
            return RunResult(
                returncode=0,
                stdout=malformed_output,
                stderr="",
                error=None,
            )
        return original_run_captured(*args, **kwargs)

    charlie_work.worktree.run_captured = mock_run_captured

    try:
        # Drive the actual consumer code: call create_worktree again (fresh-dispatch path)
        # This will call list_worktrees internally and do the real lookup around L529
        # The existing worktree should be found and triaged correctly despite the malformed entry
        info2 = create_worktree(repo_root, branch_name, base_ref="HEAD")

        # Should succeed without KeyError (the malformed entry is dropped by the parser)
        assert info2 is not None
        # The worktree path should be the same (existing worktree was found)
        assert info2.path == info1.path
        # The worktree should still exist
        assert info2.path.exists()
    finally:
        charlie_work.worktree.run_captured = original_run_captured
        remove_worktree(repo_root, info1.path)


def test_create_worktree_materialize_dirs_error_cleanup(tmp_path: Path) -> None:
    """Materialization failure should clean up the worktree and branch."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # Create an untracked directory in repo_root
    untracked_dir = repo_root / ".devin"
    untracked_dir.mkdir()
    (untracked_dir / "config.json").write_text("config\n", encoding="utf-8")

    # Monkeypatch _materialize_directory to raise an error
    import charlie_work.worktree

    original_materialize = charlie_work.worktree._materialize_directory

    def mock_materialize(*args: object, **kwargs: object) -> None:
        raise OSError("Mock materialization failure")

    charlie_work.worktree._materialize_directory = mock_materialize

    try:
        with pytest.raises(RuntimeError, match="Failed to materialize directory"):
            create_worktree(
                repo_root,
                "agent/issue-2-materialize-fail",
                base_ref="HEAD",
                materialize_dirs=(".devin",),
            )

        # Verify the worktree was cleaned up
        worktrees_dir = _default_worktrees_dir(repo_root)
        worktree_path = worktrees_dir / "agent-issue-2-materialize-fail"
        assert not worktree_path.exists()

        # Verify the branch was deleted
        result = subprocess.run(
            ["git", "branch", "--list", "agent/issue-2-materialize-fail"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "agent/issue-2-materialize-fail" not in result.stdout
    finally:
        charlie_work.worktree._materialize_directory = original_materialize


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


def _init_repo_with_remote(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare remote and a local clone, return (remote, repo)."""
    remote = tmp_path / "remote"
    _init_repo(remote, bare=True)
    repo = tmp_path / "repo"
    _clone_repo(remote, repo)
    return remote, repo


def test_inspect_worktree_state_completed(tmp_path: Path) -> None:
    """A clean worktree with commits beyond the base is completed."""
    remote, repo = _init_repo_with_remote(tmp_path)
    info = create_worktree(repo, "agent/issue-1", base_ref="origin/main")

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")

    inspection = inspect_worktree_state(info.path, base_ref="origin/main")
    assert inspection.state == WorktreeState.COMPLETED
    assert inspection.ahead_count == 1
    assert inspection.dirty is False
    assert inspection.resolved_base_ref == "origin/main"

    remove_worktree(repo, info.path, branch="agent/issue-1")


def test_inspect_worktree_state_partial_dirty(tmp_path: Path) -> None:
    """A worktree with uncommitted changes is partial, regardless of commits."""
    remote, repo = _init_repo_with_remote(tmp_path)
    info = create_worktree(repo, "agent/issue-2", base_ref="origin/main")

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")
    (info.path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    inspection = inspect_worktree_state(info.path, base_ref="origin/main")
    assert inspection.state == WorktreeState.PARTIAL
    assert inspection.dirty is True

    remove_worktree(repo, info.path, branch="agent/issue-2")


def test_inspect_worktree_state_completed_ignores_injected_prompts(tmp_path: Path) -> None:
    """Issue #381: a completed worktree is still COMPLETED if the only dirty
    files are orchestrator-injected prompt paths from the frozen config."""
    remote, repo = _init_repo_with_remote(tmp_path)
    config = OrchestratorConfig()
    info = create_worktree(repo, "agent/issue-381", base_ref="origin/main", config=config)

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")

    # Simulate orchestrator-injected prompt files being modified in the worktree.
    for injected in config.dispatch.injected_paths:
        prompt = info.path / injected
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text("injected prompt", encoding="utf-8")

    inspection = inspect_worktree_state(
        info.path,
        base_ref="origin/main",
        injected_paths=config.dispatch.injected_paths,
    )
    assert inspection.state == WorktreeState.COMPLETED
    assert inspection.ahead_count == 1
    assert inspection.dirty is False

    remove_worktree(repo, info.path, branch="agent/issue-381")


def test_inspect_worktree_state_completed_ignores_tracked_injected_prompts(
    tmp_path: Path,
) -> None:
    """Issue #381: a tracked injected prompt file modified in place is not dirty.

    This is the root-cause scenario: the orchestrator writes the prompt file,
    the worker commits it, then rewrites it in place without staging. The
    porcelain parser must not strip the leading status-column space, which would
    shift the path and drop its leading dot.
    """
    remote, repo = _init_repo_with_remote(tmp_path)
    config = OrchestratorConfig()
    info = create_worktree(repo, "agent/issue-381-tracked", base_ref="origin/main", config=config)

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")

    # Track the injected prompt file, then rewrite it in place without staging.
    prompt = info.path / config.dispatch.injected_paths[0]
    prompt.write_text("original prompt", encoding="utf-8")
    _git(info.path, "add", str(prompt))
    _git(info.path, "commit", "-m", "track prompt")
    prompt.write_text("rewritten prompt", encoding="utf-8")

    inspection = inspect_worktree_state(
        info.path,
        base_ref="origin/main",
        injected_paths=config.dispatch.injected_paths,
    )
    assert inspection.state == WorktreeState.COMPLETED
    assert inspection.ahead_count == 2
    assert inspection.dirty is False

    remove_worktree(repo, info.path, branch="agent/issue-381-tracked")


def test_inspect_worktree_state_partial_with_worker_changes_and_injected(tmp_path: Path) -> None:
    """Issue #381: worker-authored uncommitted changes still block COMPLETED."""
    remote, repo = _init_repo_with_remote(tmp_path)
    config = OrchestratorConfig()
    info = create_worktree(repo, "agent/issue-381", base_ref="origin/main", config=config)

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")

    for injected in config.dispatch.injected_paths:
        prompt = info.path / injected
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text("injected prompt", encoding="utf-8")

    (info.path / "worker-change.txt").write_text("worker work\n", encoding="utf-8")

    inspection = inspect_worktree_state(
        info.path,
        base_ref="origin/main",
        injected_paths=config.dispatch.injected_paths,
    )
    assert inspection.state == WorktreeState.PARTIAL
    assert inspection.dirty is True

    remove_worktree(repo, info.path, branch="agent/issue-381")


def test_worker_authored_dirty_ignores_worker_outcome_marker(tmp_path: Path) -> None:
    """Issue #989: ``.worker-outcome.json`` is protocol scaffolding, not worker work.

    It is written on the one path where the worker pushed a branch but could not open
    a PR, so treating it as worker-authored made every such worktree permanently
    ineligible for ``clean_worktrees`` -- including after the salvage path opened the
    PR and it merged.

    Goes through ``DispatchConfig``'s real ``injected_paths`` rather than passing the
    filename in by hand: the property under test is that the shipped configuration
    excludes it, which a hand-passed tuple would assert nothing about.
    """
    from charlie_work.config import DispatchConfig, WORKER_OUTCOME_FILENAME

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / WORKER_OUTCOME_FILENAME).write_text(
        '{"push_succeeded": true, "pr_created": false, "error": "gh unauthenticated"}',
        encoding="utf-8",
    )

    assert _worker_authored_dirty(repo_root, DispatchConfig().injected_paths) is False


def test_worker_authored_dirty_still_flags_work_beside_worker_outcome_marker(
    tmp_path: Path,
) -> None:
    """Issue #989 negative control: excluding the marker must not excuse its neighbours.

    Without this, a fix that over-broadly reported every worktree clean would pass the
    test above and silently strand real work.
    """
    from charlie_work.config import DispatchConfig, WORKER_OUTCOME_FILENAME

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / WORKER_OUTCOME_FILENAME).write_text(
        '{"push_succeeded": true, "pr_created": false, "error": "gh unauthenticated"}',
        encoding="utf-8",
    )
    (repo_root / "worker_authored.py").write_text("# real work\n", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, DispatchConfig().injected_paths) is True


def test_worker_outcome_filename_matches_the_prompt_contract() -> None:
    """Issue #989: the constant and the prompt that tells workers what to write must agree.

    ``push_pr_outcome.md`` names the file literally, so a rename of the constant that
    missed the prompt would leave workers writing a name nothing reads -- and the
    failure is silent, because a missing outcome file is indistinguishable from a
    worker that never hit the unauthenticated path.
    """
    from charlie_work.config import WORKER_OUTCOME_FILENAME

    prompt = (
        Path(worktree_module.__file__).parent
        / "prompts"
        / "worker_sections"
        / "push_pr_outcome.md"
    ).read_text(encoding="utf-8")

    assert WORKER_OUTCOME_FILENAME in prompt


def test_worker_outcome_filename_is_re_exported_from_worktree() -> None:
    """Issue #989: the constant moved to ``config`` to break a circular import.

    ``worktree.WORKER_OUTCOME_FILENAME`` stays valid for existing importers, and must
    remain the same object -- two independent string literals would drift apart
    silently, which is the bug this move exists to prevent.
    """
    from charlie_work.config import WORKER_OUTCOME_FILENAME as CONFIG_NAME

    assert worktree_module.WORKER_OUTCOME_FILENAME is CONFIG_NAME


def test_worker_authored_dirty_ignores_custom_override_path(tmp_path: Path) -> None:
    """Issue #381: a custom injected_paths override excludes the configured path."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    injected = repo_root / ".devin" / "prompts" / "worker.md"
    injected.parent.mkdir(parents=True)
    injected.write_text("injected prompt", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".devin/prompts/worker.md",)) is False


def test_worker_authored_dirty_normalizes_backslash_in_override(tmp_path: Path) -> None:
    """Issue #381: a Windows-style backslash override matches git's forward-slash path."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    injected = repo_root / ".devin" / "prompts" / "worker.md"
    injected.parent.mkdir(parents=True)
    injected.write_text("injected prompt", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".devin\\prompts\\worker.md",)) is False


def test_worker_authored_dirty_detects_sibling_in_collapsed_untracked_dir(
    tmp_path: Path,
) -> None:
    """Issue #381 follow-up, now structural: ``--untracked-files=all`` means
    git never collapses a wholly-untracked directory into a single ``??
    dir/`` line in the first place — every file, including a worker-authored
    sibling living next to a nested injected path, gets its own record. There
    is no collapsed line left to special-case or re-probe.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    injected = repo_root / ".devin" / "prompts" / "worker.md"
    injected.parent.mkdir(parents=True)
    injected.write_text("injected prompt", encoding="utf-8")
    sibling = repo_root / ".devin" / "worker-output.txt"
    sibling.write_text("real worker output", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".devin/prompts/worker.md",)) is True


def test_worker_authored_dirty_untracked_dir_with_only_injected_stays_clean(
    tmp_path: Path,
) -> None:
    """Inverse control: a directory containing ONLY the injected path (no
    siblings) must still be excused, preserving issue #381's original
    carve-out under full per-file enumeration.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    injected = repo_root / ".devin" / "prompts" / "worker.md"
    injected.parent.mkdir(parents=True)
    injected.write_text("injected prompt", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".devin/prompts/worker.md",)) is False


def test_worker_authored_dirty_tracked_leading_dot_path_modified_in_place(
    tmp_path: Path,
) -> None:
    """Issue #381 bug 1 (v1 rework history): the old line-based parser called
    ``.strip()`` on each porcelain line before a fixed-column ``line[3:]``
    slice, which corrupted the leading-space status column of an unstaged
    tracked modification (e.g. ``" M .orchestrator-prompt.md"``) — the strip
    shifted the string left by one character and dropped the leading dot.
    ``--porcelain=v2 -z`` has no line-based whitespace to strip in the first
    place: fields are parsed by splitting the record on single spaces with a
    bounded maxsplit, never by slicing a stripped string.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    prompt = repo_root / ".orchestrator-prompt.md"
    prompt.write_text("original prompt", encoding="utf-8")
    _git(repo_root, "add", str(prompt))
    _git(repo_root, "commit", "-m", "track prompt")
    # Modify in place without staging: unstaged tracked change, XY = ".M".
    prompt.write_text("rewritten prompt", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".orchestrator-prompt.md",)) is False

    # Control: the same leading-dot tracked-modification shape, but for a path
    # NOT in injected_paths, must still be detected as dirty.
    other = repo_root / ".other-tracked.md"
    other.write_text("original", encoding="utf-8")
    _git(repo_root, "add", str(other))
    _git(repo_root, "commit", "-m", "track other")
    other.write_text("changed", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".orchestrator-prompt.md",)) is True


def test_worker_authored_dirty_directory_injected_path_scoped_to_its_own_tree(
    tmp_path: Path,
) -> None:
    """Issue #381 bug 4 (v1 rework history): when an injected_paths entry
    names a whole directory rather than a specific file (a documented
    supported convention — see ``DispatchConfig.injected_paths``), the old
    line-based parser matched it via an exact-match fast path that ran
    *before*, and bypassed, the collapse-safety re-probe used for the
    nested-file case (bug 3) — an inconsistency between two code paths meant
    to answer the same question. ``--untracked-files=all`` removes the
    re-probe machinery (and the fast-path/re-probe distinction) entirely:
    every file, whether it matches an injected file or an injected directory,
    is matched individually by the exact same normalized-path predicate.

    A directory-level injected_paths entry legitimately excuses everything
    under it (that is the documented, intentional trade-off of naming a
    whole directory) — but the exclusion must stay scoped to that directory's
    own subtree and not leak to unrelated content, including a directory that
    merely shares a string prefix.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    injected = repo_root / ".devin" / "prompts" / "worker.md"
    injected.parent.mkdir(parents=True)
    injected.write_text("injected prompt", encoding="utf-8")
    sibling = repo_root / ".devin" / "worker-output.txt"
    sibling.write_text("real worker output inside the excluded tree", encoding="utf-8")

    # Naming the directory itself excuses everything inside it, by design.
    assert _worker_authored_dirty(repo_root, (".devin",)) is False

    # An unrelated untracked file OUTSIDE the named directory must still be
    # detected — the exclusion does not leak beyond its own subtree.
    unrelated = repo_root / "worker-result.txt"
    unrelated.write_text("unrelated worker output", encoding="utf-8")
    assert _worker_authored_dirty(repo_root, (".devin",)) is True


def test_worker_authored_dirty_directory_prefix_does_not_collide(tmp_path: Path) -> None:
    """A directory that merely shares a string prefix with a configured
    injected directory (e.g. ``.devin-cache`` vs ``.devin``) must not match —
    path-segment comparison via ``PurePosixPath.parents``, not substring
    matching, is what makes this safe.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    cache_dir = repo_root / ".devin-cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "marker.txt").write_text("not injected", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (".devin",)) is True


def test_worker_authored_dirty_renamed_tracked_file_not_injected(tmp_path: Path) -> None:
    """A rename/copy porcelain=v2 record (tag ``2``) carries an extra
    NUL-delimited ``origPath`` field after the current path. Parsing must
    consume that extra field so it isn't mistaken for the next record's tag,
    and the CURRENT (renamed-to) path is what gets matched against
    injected_paths — mirroring the old parser's ``" -> "`` right-hand-side
    convention.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    tracked = repo_root / "tracked.txt"
    tracked.write_text("content\n", encoding="utf-8")
    _git(repo_root, "add", "tracked.txt")
    _git(repo_root, "commit", "-m", "add tracked")
    _git(repo_root, "mv", "tracked.txt", "renamed.txt")
    # A subsequent untracked file proves the rename record's extra origPath
    # field was correctly consumed rather than corrupting the next record.
    (repo_root / "new-worker-file.txt").write_text("also here", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, ()) is True

    # A rename INTO an injected path is excused by the current-path match.
    _git(repo_root, "reset", "--hard")
    (repo_root / "new-worker-file.txt").unlink(missing_ok=True)
    _git(repo_root, "mv", "tracked.txt", "prompt-renamed.md")
    assert _worker_authored_dirty(repo_root, ("prompt-renamed.md",)) is False


def test_worker_authored_dirty_excludes_materialize_dirs_surface(
    tmp_path: Path,
) -> None:
    """Issue #471: tracked modifications confined to the configured
    ``materialize_dirs`` surface must not count as worker-authored dirt,
    even when the assume-unchanged bit is not set.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    prompt = repo_root / ".devin" / "prompts" / "worker.md"
    prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_text("original prompt\n", encoding="utf-8")
    _git(repo_root, "add", str(prompt))
    _git(repo_root, "commit", "-m", "track prompt template")

    # Simulate an external launch shim rewriting the tracked prompt in place
    # without the assume-unchanged bit set.
    prompt.write_text("per-dispatch prompt\n", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (), (".devin",)) is False


def test_worker_authored_dirty_detects_changes_outside_materialize_and_injected(
    tmp_path: Path,
) -> None:
    """Issue #471: a modification outside both ``injected_paths`` and
    ``materialize_dirs`` must still trip the worktree-safety guard.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    (repo_root / "worker-output.txt").write_text("worker result\n", encoding="utf-8")

    assert _worker_authored_dirty(repo_root, (), (".devin",)) is True


def test_inspect_worktree_state_no_commits(tmp_path: Path) -> None:
    """A clean worktree with no commits beyond the base is no_commits."""
    remote, repo = _init_repo_with_remote(tmp_path)
    info = create_worktree(repo, "agent/issue-3", base_ref="origin/main")

    inspection = inspect_worktree_state(info.path, base_ref="origin/main")
    assert inspection.state == WorktreeState.NO_COMMITS
    assert inspection.ahead_count == 0
    assert inspection.dirty is False

    remove_worktree(repo, info.path, branch="agent/issue-3")


def test_inspect_worktree_state_unknown_missing_path(tmp_path: Path) -> None:
    """A missing worktree path returns unknown."""
    inspection = inspect_worktree_state(tmp_path / "does-not-exist")
    assert inspection.state == WorktreeState.UNKNOWN
    assert inspection.error is not None


def test_inspect_worktree_state_empty_path_returns_unknown(tmp_path: Path) -> None:
    """Issue #660: an empty worktree_path (api workers set worktree_path="")
    must short-circuit to UNKNOWN instead of probing the caller's cwd.

    Path("") normalizes to Path("."), which is a real directory (the cwd).
    Without the guard, inspect_worktree_state would run real git merge-base /
    rev-list against whatever checkout the caller happens to be in, and could
    return COMPLETED if that checkout has local commits ahead of its base --
    silently misclassifying every dead api-worker session as "unpublished_work".
    """
    # Run from inside tmp_path (a real directory) to prove the guard fires
    # before any filesystem/git probe of cwd, regardless of cwd state.
    import os

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        inspection = inspect_worktree_state(Path(""))
    finally:
        os.chdir(old_cwd)
    assert inspection.state == WorktreeState.UNKNOWN
    assert inspection.error is not None
    assert "empty" in inspection.error


def test_inspect_worktree_state_empty_path_short_circuits_before_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coverage gap in test_inspect_worktree_state_empty_path_returns_unknown:
    that test chdirs into a bare ``tmp_path`` (not a git repo), so removing the
    guard entirely still returns UNKNOWN there -- git merge-base/rev-list fail
    with "not a git repository" regardless of the guard, and only the error
    *string* assertion would catch a regression, not the state assertion.

    The actual danger (issue #660) requires a cwd that IS a real git checkout
    with commits ahead of its own base -- that is what turns the misprobe into
    WorktreeState.COMPLETED (see the merge-ahead branch a few lines below the
    guard). This test reproduces that exact precondition: a real repo, checked
    out with an unpublished commit ahead of origin/main, clean working tree.
    Without the guard this asserts COMPLETED (empirically confirmed by
    temporarily deleting the guard and re-running this exact scenario); with
    the guard it must stay UNKNOWN.
    """
    remote, repo = _init_repo_with_remote(tmp_path)
    (repo / "extra.txt").write_text("extra\n", encoding="utf-8")
    _git(repo, "add", "extra.txt")
    _git(repo, "commit", "-m", "ahead of origin/main")
    monkeypatch.chdir(repo)

    inspection = inspect_worktree_state(Path(""))
    assert inspection.state == WorktreeState.UNKNOWN
    assert inspection.error is not None
    assert "empty" in inspection.error


def test_push_branch_publishes_and_verifies(tmp_path: Path) -> None:
    """push_branch pushes a local branch to origin and verifies the remote tip."""
    remote, repo = _init_repo_with_remote(tmp_path)
    info = create_worktree(repo, "agent/issue-4", base_ref="origin/main")

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")

    ok, error = push_branch(repo, "agent/issue-4", worktree_path=info.path)
    assert ok, error

    remote_refs = _git(remote, "show-ref")
    assert "agent/issue-4" in remote_refs.stdout

    remove_worktree(repo, info.path, branch="agent/issue-4")


def test_push_branch_rejects_invalid_ref_name(tmp_path: Path) -> None:
    """Invalid ref names are rejected before any git argv is built (issue #659)."""
    ok, error = push_branch(tmp_path, "--exec=foo")
    assert not ok
    assert error is not None
    assert "not a valid git ref name" in error


def test_recovery_aborts_when_worker_pid_alive(tmp_path: Path) -> None:
    """Issue #282: recovery must not remove a worktree if the recorded worker PID is still alive."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-live-worker"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Spawn a real long-running child process so we have a live PID and a valid start time.
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )
    try:
        start_time = get_process_start_time(process.pid)
        assert start_time is not None

        recovery_record = {
            "branch_name": branch_name,
            "status": "dispatched",
            "worker_pid": process.pid,
            "worker_process_start_time": start_time,
        }

        with pytest.raises(LiveWorkerRedispatchError) as exc_info:
            create_worktree(repo_root, branch_name, base_ref="HEAD", recovery=recovery_record)

        assert exc_info.value.probe_result == "pid_alive"
        assert exc_info.value.pid == process.pid
        # The worktree and branch must survive the aborted redispatch.
        assert info1.path.exists()
        assert branch_name in _git(repo_root, "branch", "--list").stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_recovery_aborts_on_sessions_db_activity(tmp_path: Path) -> None:
    """Issue #282: recovery must not remove a worktree if sessions.db shows fresh activity."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-db-activity"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # Build a fake Devin sessions.db with a recent session/message for this worktree.
    db_path = tmp_path / "sessions.db"
    now = datetime.now(UTC).isoformat()
    make_sessions_db(
        db_path,
        session_id="session-1",
        working_directory=str(worktree_path),
        created_at=now,
        rows=[{"role": "tool", "content": "tool result", "created_at": now}],
    )

    # The per-PID Devin log source must also resolve WITHOUT an error (issue
    # #282 rework: any errored source makes the whole probe inconclusive and
    # aborts recovery on its own, which would mask this test's actual signal
    # under test — sessions.db activity alone being sufficient). Give it a
    # confirmed-stale (not fresh) timestamp so sessions.db remains the
    # freshest, deciding source.
    logs_dir = db_path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stale_log = logs_dir / "devin_test_999999.log"
    stale_log.write_text("old log\n", encoding="utf-8")
    stale_time = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
    os.utime(stale_log, (stale_time, stale_time))

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )

    # Use a dead PID (no such process) so the sessions.db probe is the deciding signal.
    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
        "started_at": now,
    }

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            recovery=recovery_record,
            config=config,
        )

    assert exc_info.value.probe_result == "sessions_db_activity"
    # No worktree should have been created yet and the branch should not exist.
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_aborts_on_sessions_db_schema_error_other_source_silent(tmp_path: Path) -> None:
    """Issue #282 rework: an errored sessions.db probe is INCONCLUSIVE, not
    confirmed-dead — even when the per-PID log source has no signal of its
    own either. Reproduces the exact live-incident signature (sqlite schema
    drift, "no such column: id") that let the prior fail-open guard proceed
    into a destructive reset of a live worker.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-db-schema-error"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions table is missing the `id` column the probe's query selects -
    # the exact schema-drift shape from the live incident.
    db_path = tmp_path / "sessions.db"

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (working_directory TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )

    # No logs/ directory at all -> devin_per_pid_log is silent too (its own
    # "not found" error), never a confirmed timestamp either way.
    # No worker PID is recorded, so the probe is genuinely inconclusive.
    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": None,
        "worker_process_start_time": 0.0,
        "started_at": now,
    }

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            recovery=recovery_record,
            config=config,
        )

    assert exc_info.value.probe_result == "probe_error"
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_aborts_when_all_sources_errored(tmp_path: Path) -> None:
    """Issue #282 rework: if every activity source errors, liveness is
    unknown, not confirmed-dead — recovery must abort rather than proceed as
    if the worker were genuinely stale.

    Mutation-checked: reverting ``_probe_recovery_liveness`` to the prior
    fail-open guard (``if source.name == "sessions.db" and
    source.staleness_seconds is not None``) makes this test FAIL, because
    that guard silently returns without raising when every source errors.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-all-sources-errored"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db does not exist on disk at all -> _open_readonly errors
    # immediately, before any query is even attempted.
    db_path = tmp_path / "missing-sessions.db"

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )

    # No logs/ directory either -> devin_per_pid_log also errors.
    # No worker PID is recorded, so the probe is genuinely inconclusive.
    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": None,
        "worker_process_start_time": 0.0,
        "started_at": now,
    }

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            recovery=recovery_record,
            config=config,
        )

    assert exc_info.value.probe_result == "probe_error"
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_aborts_on_fresh_per_pid_log_despite_sessions_db_error(tmp_path: Path) -> None:
    """Issue #282 rework: fresh devin_per_pid_log activity must abort
    recovery even when sessions.db is the source that errored — this is the
    exact tonight's-incident shape (an 8-second-fresh per-PID log) that the
    prior sessions.db-only check never looked at.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-per-pid-log-fresh-db-error"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db missing entirely -> errored source.
    db_path = tmp_path / "missing-sessions.db"

    # Fresh per-PID Devin log for the recorded worker pid.
    logs_dir = db_path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fresh_log = logs_dir / "devin_test_999999.log"
    fresh_log.write_text("fresh activity\n", encoding="utf-8")

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )

    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
        "started_at": now,
    }

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            recovery=recovery_record,
            config=config,
        )

    # Aborted either way (the errored sessions.db source is already enough on
    # its own) - the point of this regression test is that the fresh
    # devin_per_pid_log signal is never silently ignored just because it
    # isn't the sessions.db source.
    assert exc_info.value.probe_result == "devin_per_pid_log_activity"
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_aborts_on_fresh_per_pid_log_when_sessions_db_confirmed_stale(
    tmp_path: Path,
) -> None:
    """Issue #282 rework, requirement 2 in isolation: even when sessions.db
    positively confirms only stale (non-error) activity, a fresh
    devin_per_pid_log signal on its own must still abort recovery. The prior
    guard only ever inspected the "sessions.db" source by name and would have
    silently ignored this signal.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-per-pid-log-only-fresh"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    db_path = tmp_path / "sessions.db"
    stale_iso = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    make_sessions_db(
        db_path,
        session_id="session-1",
        working_directory=str(worktree_path),
        created_at=stale_iso,
        rows=[{"role": "tool", "content": "tool result", "created_at": stale_iso}],
    )

    # Fresh per-PID Devin log (mtime defaults to "now" via write_text).
    logs_dir = db_path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fresh_log = logs_dir / "devin_test_999999.log"
    fresh_log.write_text("fresh activity\n", encoding="utf-8")

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )

    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
        "started_at": stale_iso,
    }

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            recovery=recovery_record,
            config=config,
        )

    assert exc_info.value.probe_result == "devin_per_pid_log_activity"
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_increments_deferral_count_for_permanent_no_match(tmp_path: Path) -> None:
    """Issue #426: a permanent sessions.db no-match increments the recovery
    deferral counter on each attempt and still aborts below the cap.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-permanent-no-match"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db exists but has no row for this worktree (permanent no-match).
    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (id TEXT, working_directory TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=3),
    )

    # No worker PID is recorded, so the permanent no-match is genuinely
    # inconclusive and the deferral counter must advance.
    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": None,
        "worker_process_start_time": 0.0,
        "started_at": now,
    }

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            recovery=recovery_record,
            config=config,
        )

    assert exc_info.value.probe_result == "probe_error"
    assert exc_info.value.inconclusive_probe_deferred_count == 1
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_allows_permanent_no_match_after_deferral_cap(tmp_path: Path) -> None:
    """Issue #426: a structurally permanent sessions.db no-match is not an
    unconditional recovery block. After ``max_inconclusive_probe_deferrals``
    consecutive inconclusive probes, the destructive reset is allowed.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-permanent-no-match-capped"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db exists but has no row for this worktree (permanent no-match).
    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (id TEXT, working_directory TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=2),
    )

    # No worker PID is recorded, so the deferral cap is the reason reset is
    # allowed, not the confirmed-dead PID short-circuit.
    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": None,
        "worker_process_start_time": 0.0,
        "started_at": now,
        "inconclusive_probe_deferred_count": 2,
    }

    # Should not raise: the permanent no-match has reached the deferral cap.
    result = create_worktree(
        repo_root,
        branch_name,
        base_ref="HEAD",
        recovery=recovery_record,
        config=config,
    )

    assert isinstance(result, WorktreeInfo)
    assert worktree_path.exists()
    assert branch_name in _git(repo_root, "branch", "--list").stdout


def test_recovery_allows_reset_when_worker_pid_dead_and_probe_inconclusive(
    tmp_path: Path,
) -> None:
    """Issue #506: a confirmed-dead worker PID overrides an inconclusive probe."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-dead-pid-inconclusive"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db exists but has no row for this worktree (permanent no-match).
    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (id TEXT, working_directory TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=3),
    )

    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
        "started_at": now,
    }

    result = create_worktree(
        repo_root,
        branch_name,
        base_ref="HEAD",
        recovery=recovery_record,
        config=config,
    )

    assert isinstance(result, WorktreeInfo)
    assert worktree_path.exists()
    assert branch_name in _git(repo_root, "branch", "--list").stdout


def test_recovery_aborts_on_transient_probe_error_despite_dead_pid(
    tmp_path: Path,
) -> None:
    """Issue #506 rework: a confirmed-dead PID does NOT override a probe that
    contains transient errors (locked/corrupt DB, schema drift, I/O failures).
    Only structurally permanent absence-of-record errors may be overridden by
    a dead PID; transient errors remain fail-closed.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-dead-pid-transient-probe"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # sessions.db exists on disk but is not a valid SQLite file — a transient
    # "failed to open sessions.db (locked or corrupt)" error, not a permanent
    # no-match.
    db_path = tmp_path / "sessions.db"
    db_path.write_bytes(b"this is not a sqlite database")

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=3),
    )

    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
        "started_at": now,
    }

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo_root,
            branch_name,
            base_ref="HEAD",
            recovery=recovery_record,
            config=config,
        )

    assert exc_info.value.probe_result == "probe_error"
    assert exc_info.value.inconclusive_probe_deferred_count == 1
    assert not worktree_path.exists()
    assert branch_name not in _git(repo_root, "branch", "--list").stdout


def test_recovery_proceeds_when_no_source_errored_and_pid_dead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #640: when no probe source errors, ``all_permanent`` keeps its
    pre-bound default and is never read in a decision — the second
    ``if errored_sources:`` block is skipped entirely. A confirmed-dead PID
    with a fully successful (but stale) probe must let recovery proceed
    without raising and without any ``UnboundLocalError`` on ``all_permanent``.

    This is the no-errored-source counterpart to the errored-source tests
    above; it guards the binding fix that makes ``all_permanent`` provably
    bound to Pyright.
    """
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch_name = "agent/issue-1-no-source-errored-dead-pid"
    worktree_path = _default_worktrees_dir(repo_root) / _slugify(branch_name)

    # A probe where every source returned successfully (no error) but with only
    # stale timestamps — liveness is genuinely inconclusive, yet no source
    # errored, so neither ``if errored_sources:`` branch runs.
    stale = datetime.now(UTC) - timedelta(hours=2)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=stale,
                staleness_seconds=7200.0,
                error=None,
            ),
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=stale,
                staleness_seconds=7200.0,
                error=None,
            ),
        )
    )

    monkeypatch.setattr(worktree_module, "real_activity_for_worker", lambda *a, **k: probe)

    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "unused.db")),
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=3),
    )

    recovery_record = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
        "started_at": datetime.now(UTC).isoformat(),
    }

    result = create_worktree(
        repo_root,
        branch_name,
        base_ref="HEAD",
        recovery=recovery_record,
        config=config,
    )

    # Recovery proceeds: the dead PID and stale-but-error-free probe do not
    # abort redispatch.
    assert isinstance(result, WorktreeInfo)
    assert worktree_path.exists()
    assert branch_name in _git(repo_root, "branch", "--list").stdout


def _make_state(issue_number: int, pr_number: int, *, status: str = "merged") -> dict[str, Any]:
    return {
        "issues": {str(issue_number): {"number": issue_number}},
        "prs": {
            str(pr_number): {
                "number": pr_number,
                "issue_number": issue_number,
                "status": status,
                "merged": status == "merged",
            }
        },
        "events": [],
    }


class _FakeGH(WorktreeCleanGH):
    """Fake ``GitHub`` for ``clean_worktrees`` tests.

    Implements the ``WorktreeCleanGH`` protocol (the slice of ``GitHub`` the
    cleanup lane depends on) so it is statically assignable to
    ``clean_worktrees(..., gh=...)`` without ``cast`` (issue #641).

    ``available=False`` simulates ``gh`` itself failing/being unreachable
    (``GitHubRunResult(ok=False, ...)``), distinct from ``gh`` succeeding but
    reporting a PR state other than ``MERGED``.
    """

    def __init__(
        self,
        pr_state: str = "MERGED",
        merged_at: str | None = "2026-07-13T01:33:43Z",
        head_sha: str | None = None,
        *,
        available: bool = True,
        error: str = "gh: could not resolve to a PullRequest",
    ) -> None:
        self.pr_state = pr_state
        self.merged_at = merged_at
        self.head_sha = head_sha
        self.available = available
        self.error = error

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> GitHubRunResult:
        if args[:2] == ["pr", "view"] and json_output and allow_failure:
            if not self.available:
                return GitHubRunResult(
                    ok=False,
                    returncode=1,
                    stdout="",
                    stderr=self.error,
                    value=None,
                    error=self.error,
                )
            return GitHubRunResult(
                ok=True,
                returncode=0,
                stdout="",
                stderr="",
                value={
                    "state": self.pr_state,
                    "mergedAt": self.merged_at,
                    "headRefOid": self.head_sha,
                },
                error=None,
            )
        return GitHubRunResult(
            ok=False,
            returncode=1,
            stdout="",
            stderr="",
            value=None,
            error="unexpected fake gh command",
        )


def _create_shared_venv(repo_root: Path, pth_target: Path | None = None) -> Path:
    """Create a fake shared venv with an editable .pth pointing at the target src."""
    pth = repo_root / "shared-venv" / "Lib" / "site-packages" / "_editable_impl_charlie_work.pth"
    pth.parent.mkdir(parents=True)
    target = pth_target or (repo_root / "src")
    pth.write_text(str(target.resolve()) + "\n", encoding="utf-8")
    return repo_root / "shared-venv"


def test_verify_shared_venv_catches_pth_pointing_outside_main_checkout(tmp_path: Path) -> None:
    """Editable .pth pointing at a worker worktree is the poisoned-venv case."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    worktree = tmp_path / "stolen-worktree"
    worktree.mkdir()
    _create_shared_venv(repo_root, pth_target=worktree / "src")

    ok, message = verify_shared_venv(repo_root, repo_root / "shared-venv")

    assert not ok
    assert "points outside main checkout" in message
    assert "uv sync --all-extras --reinstall-package charlie-work" in message


def test_verify_shared_venv_approves_pth_pointing_at_main_checkout(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    ok, message = verify_shared_venv(repo_root, repo_root / "shared-venv")

    assert ok
    assert "main checkout" in message


def test_clean_worktrees_removes_merged_worktree_and_verifies_shared_venv(
    tmp_path: Path,
) -> None:
    """Junction-safe removal deletes the worktree and leaves the shared venv valid."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-1-merged", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=1, pr_number=101)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert isinstance(result, WorktreeCleanResult)
    assert result.ok is True
    assert len(result.data["removed"]) == 1
    assert not info.path.exists()
    assert result.data["venv_ok"] is True
    assert "main checkout" in result.data["venv_message"]


def test_clean_worktrees_skips_dirty_worktree(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-2-dirty", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    (info.path / "dirty_file.txt").write_text("local changes", encoding="utf-8")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=2, pr_number=102)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.ok is True
    assert len(result.data["skipped"]) == 1
    assert "uncommitted" in result.data["skipped"][0]["reason"]
    assert info.path.exists()


def test_clean_worktrees_ignores_injected_only_dirtiness(tmp_path: Path) -> None:
    """Issue #381: merged worktree with only injected prompt files dirty should be removed."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-381-clean", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    for injected in config.dispatch.injected_paths:
        prompt = info.path / injected
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text("injected prompt", encoding="utf-8")
    state = _make_state(issue_number=381, pr_number=1381)

    result = clean_worktrees(
        repo_root,
        _default_worktrees_dir(repo_root),
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.ok is True
    assert len(result.data["removed"]) == 1
    assert result.data["removed"][0]["branch"] == "agent/issue-381-clean"
    assert not info.path.exists()


def test_clean_worktrees_skips_stray_post_merge_commit(tmp_path: Path) -> None:
    """A worktree HEAD that no longer matches the merged PR's headRefOid (a
    stray commit made after GitHub recorded the merge) must be skipped with a
    distinct reason -- it must NOT be silently removed just because the PR
    state.json/gh say "merged" (review finding 1: eligibility requires HEAD ==
    merged PR head, not just "PR is merged somewhere").
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-3-stray-commit", base_ref="HEAD")
    _git(info.path, "config", "user.email", "test@example.test")
    _git(info.path, "config", "user.name", "Test User")
    # This is the SHA GitHub would report as the merged PR's headRefOid --
    # captured BEFORE the stray commit below.
    merged_head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    (info.path / "new_file.txt").write_text("stray post-merge commit", encoding="utf-8")
    _git(info.path, "add", "new_file.txt")
    _git(info.path, "commit", "-m", "stray commit made after the PR was merged")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=3, pr_number=103)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=merged_head_sha),
    )

    assert result.ok is True
    assert len(result.data["skipped"]) == 1
    reason = result.data["skipped"][0]["reason"].lower()
    assert "stray" in reason
    assert info.path.exists()


def test_clean_worktrees_removes_worktree_behind_merged_head(tmp_path: Path) -> None:
    """The mirror of the stray-commit case, and the other half of the bug that
    made ``worktree-clean`` inert.

    The eligibility question is containment, not equality: does the worktree
    hold commits that did NOT get merged? A worktree sitting *behind* the
    merged head holds none — everything in it is reachable from what landed.
    That is the ordinary shape here, because the merge path advances the PR
    branch after the worker's last local commit (Aviator merge-queue rebases,
    merge-train updates, base-into-branch merges).

    The old ``local_head_sha != merged_head_sha`` test could not tell the two
    directions apart and reported 46 of 47 mismatching worktrees on the live
    host as "stray post-merge commit(s)" — a false reason attached to a
    permanent skip.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-3-behind", base_ref="HEAD")
    _git(info.path, "config", "user.email", "test@example.test")
    _git(info.path, "config", "user.name", "Test User")
    # The worktree's HEAD is where the worker stopped.
    local_head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    # The branch then advanced (merge queue rebase / base-into-branch update)
    # and GitHub recorded THAT tip as the merged headRefOid. Build it as a real
    # descendant commit, then put the worktree back where the worker left it.
    (info.path / "queued.txt").write_text("advanced by the merge queue", encoding="utf-8")
    _git(info.path, "add", "queued.txt")
    _git(info.path, "commit", "-m", "merge queue advanced the branch")
    merged_head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    _git(info.path, "reset", "--hard", local_head_sha)
    assert _git(info.path, "rev-parse", "HEAD").stdout.strip() == local_head_sha

    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=3, pr_number=103)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=merged_head_sha),
    )

    assert result.ok is True, result.message
    assert result.data["skipped"] == []
    assert [entry["issue_number"] for entry in result.data["removed"]] == [3]
    assert not info.path.exists()


def test_clean_worktrees_skips_when_merged_head_object_is_absent(tmp_path: Path) -> None:
    """An unknown merged head must fail closed with a legible reason.

    ``git merge-base --is-ancestor`` exits non-zero both for "not an ancestor"
    and for "no such object", so the containment check gates on object
    presence first. Without that gate the skip would still happen, but under
    the wrong reason — and a wrong reason string is what makes a defect like
    this survive a reading of the output.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-3-absent", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=3, pr_number=103)
    # A well-formed SHA that names no object in this repo.
    absent_sha = "0" * 39 + "1"

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=absent_sha),
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "not present in the local object store" in result.data["skipped"][0]["reason"]
    assert info.path.exists()


def test_clean_worktrees_removes_squash_merged_worktree_with_deleted_remote_branch(
    tmp_path: Path,
) -> None:
    """Production-mirroring regression test (review finding 1).

    This repo's ``AutoMergeConfig`` defaults are ``strategy="squash"`` with
    ``delete_branch=True``. After a real merge: the remote branch is gone, and
    the squash commit landed on ``main`` is NOT an ancestor of the worker
    branch's own commit (its tree matches, but it is a distinct commit with a
    different parent). The old ``_worktree_refuse_to_reset_reason``-based
    eligibility check counted the worker's own committed work as "local
    commit(s) not on remote branch" in exactly this shape and refused to clean
    up, forever. ``clean_worktrees`` must still remove the worktree here
    because gh confirms MERGED and the worktree's local HEAD equals the
    merged PR's headRefOid -- local-ahead-of-a-deleted-remote is expected.
    """
    remote, repo_root = _init_repo_with_remote(tmp_path)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _git(repo_root, "push", "origin", "main")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    branch_name = "agent/issue-7-squash-merged"
    info = create_worktree(repo_root, branch_name, base_ref="origin/main")
    (info.path / "feature.txt").write_text("real work\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "real work for issue 7")
    branch_head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    push_ok, push_error = push_branch(repo_root, branch_name, worktree_path=info.path)
    assert push_ok, push_error

    # Simulate a squash-merge into main: a NEW commit on main whose parent is
    # main's own tip, not an ancestor of branch_head_sha.
    _git(repo_root, "merge", "--squash", branch_name)
    _git(repo_root, "commit", "-m", f"squash merge of {branch_name}")
    _git(repo_root, "push", "origin", "main")
    # Simulate delete_branch=True: the remote branch is gone after merge.
    _git(repo_root, "push", "origin", "--delete", branch_name)

    # Sanity-check the exact regression shape: the squash commit on main is
    # NOT an ancestor of the branch's own commit.
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", branch_head_sha, "main"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert is_ancestor.returncode != 0, "test setup invalid: squash commit must not be an ancestor"

    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=7, pr_number=107)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=branch_head_sha),
    )

    assert len(result.data["skipped"]) == 0, result.data["skipped"]
    assert len(result.data["removed"]) == 1
    assert result.ok is True
    assert not info.path.exists()


def test_clean_worktrees_skips_live_worker(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-4-live", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    issue_state = {
        "number": 4,
        "worker_pid": os.getpid(),
        "worker_process_start_time": get_process_start_time(os.getpid()),
        "started_at": datetime.now(UTC).isoformat(),
    }
    state = _make_state(issue_number=4, pr_number=104)
    state["issues"]["4"] = issue_state

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.ok is True
    assert len(result.data["skipped"]) == 1
    assert "live" in result.data["skipped"][0]["reason"].lower()
    assert info.path.exists()


def test_clean_worktrees_removes_merged_worktree_with_no_recorded_worker_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression that made ``worktree-clean`` a no-op on the live host.

    ``clean_worktrees`` used to gate removal on ``_probe_recovery_liveness``,
    the *redispatch* lane's guard. That probe is fail-closed on an inconclusive
    Devin activity probe, and on this repo's hosts it is inconclusive forever:
    finished workers have no recorded pid (so the per-PID log source reports
    ``no pid``), the Devin ``sessions.db`` holds no rows for claude-code/api
    workers, and the probe's only escape hatch — the
    ``inconclusive_probe_deferred_count`` cap — is never persisted by this
    lane, so the counter is pinned at 0. Result: 70 of 81 merged worktrees
    skipped with ``live worker detected: probe_error``, on every pass.

    The old tests missed it because ``real_activity_for_worker`` *raised* under
    tmp_path (no sessions.db at all), and the probe swallows exceptions as
    "not live" — the opposite of what a real host with a real db does. So this
    test forces the production shape directly: the redispatch probe declares a
    live worker, and the merged worktree must still be removed, because the
    cleanup lane must not consult that probe at all.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    def _always_probe_error(*args: Any, **kwargs: Any) -> None:
        raise LiveWorkerRedispatchError(
            issue_number=4,
            pid=None,
            process_start_time=None,
            probe_result="probe_error",
            inconclusive_probe_deferred_count=1,
        )

    monkeypatch.setattr(worktree_module, "_probe_recovery_liveness", _always_probe_error)

    info = create_worktree(repo_root, "agent/issue-4-no-pid", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    # No worker_pid / last_known_worker_pid: the shape of a worker that
    # finished long ago and whose pid was pruned from state.
    state = _make_state(issue_number=4, pr_number=104)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.ok is True, result.message
    assert result.data["skipped"] == []
    assert [entry["issue_number"] for entry in result.data["removed"]] == [4]
    assert not info.path.exists()


def test_clean_worktrees_skips_worktree_with_live_writer_marker(tmp_path: Path) -> None:
    """No recorded pid, but a writer marker whose pid is alive: still in use.

    This is the signal that replaces the redispatch probe — positive evidence
    of a running process rather than an absence of records.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-4-marker", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    write_worktree_marker(info.path, os.getpid(), "session-live")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=4, pr_number=104)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "live writer marker" in result.data["skipped"][0]["reason"]
    assert info.path.exists()


def test_clean_worktrees_skips_operator_claimed_worktree(tmp_path: Path) -> None:
    """Operator markers carry a sentinel pid; state.json is the authority.

    A pid-only liveness check would read the sentinel as dead and delete a
    worktree an operator is actively editing.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-4-operator", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    write_worktree_marker(info.path, 0, OPERATOR_MARKER_SESSION_ID, kind=OPERATOR_MARKER_KIND)
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=4, pr_number=104)
    state["issues"]["4"]["operator_claimed_at"] = datetime.now(UTC).isoformat()

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.data["removed"] == []
    assert len(result.data["skipped"]) == 1
    assert "operator-claimed" in result.data["skipped"][0]["reason"]
    assert info.path.exists()


def test_clean_worktrees_released_operator_marker_does_not_block_removal(
    tmp_path: Path,
) -> None:
    """A leftover operator marker with no live claim in state must not pin the
    worktree forever — the marker alone is not evidence of a live operator."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-4-released", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    write_worktree_marker(info.path, 0, OPERATOR_MARKER_SESSION_ID, kind=OPERATOR_MARKER_KIND)
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    # No operator_claimed_at: the claim was released.
    state = _make_state(issue_number=4, pr_number=104)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.data["skipped"] == []
    assert [entry["issue_number"] for entry in result.data["removed"]] == [4]
    assert not info.path.exists()


def test_clean_worktrees_dry_run_reports_plan_and_does_not_remove(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    info = create_worktree(repo_root, "agent/issue-5-dry-run", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=5, pr_number=105)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
        dry_run=True,
    )

    assert result.ok is True
    assert len(result.data["planned"]) == 1
    assert len(result.data["removed"]) == 0
    assert info.path.exists()


def test_clean_worktrees_surfaces_poisoned_venv_after_removal(tmp_path: Path) -> None:
    """If removal leaves the shared venv .pth pointing at a worktree, report it."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _git(repo_root, "add", "src/charlie_work/__init__.py")
    _git(repo_root, "commit", "-m", "add charlie_work")

    info = create_worktree(repo_root, "agent/issue-6-poison", base_ref="HEAD")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    _create_shared_venv(repo_root, pth_target=info.path / "src")

    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig(devin=DevinConfig(venv_source="shared-venv"))
    state = _make_state(issue_number=6, pr_number=106)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert result.ok is False
    assert len(result.data["removed"]) == 1
    assert result.data["venv_ok"] is False
    assert "points outside main checkout" in result.data["venv_message"]


def test_clean_worktrees_skips_open_pr(tmp_path: Path) -> None:
    """Review finding 2: an open-PR worktree must never be removed.

    This is the highest-severity failure mode -- destroying a worktree whose
    PR is still open and possibly still being worked on.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-8-open", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=8, pr_number=108, status="open")

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(pr_state="OPEN"),
    )

    assert len(result.data["removed"]) == 0
    assert len(result.data["skipped"]) == 1
    assert "not merged" in result.data["skipped"][0]["reason"].lower()
    assert info.path.exists()


def test_clean_worktrees_skips_closed_unmerged_pr(tmp_path: Path) -> None:
    """Review finding 2: a closed-but-not-merged PR's worktree must be skipped,
    not treated as eligible just because the PR is no longer open.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-9-closed", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=9, pr_number=109, status="closed")

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(pr_state="CLOSED", merged_at=None),
    )

    assert len(result.data["removed"]) == 0
    assert len(result.data["skipped"]) == 1
    assert "not merged" in result.data["skipped"][0]["reason"].lower()
    assert info.path.exists()


def test_clean_worktrees_state_json_merged_alone_is_not_sufficient(tmp_path: Path) -> None:
    """Review finding 3: state.json claiming "merged" must NOT authorize
    removal when the live gh pr view disagrees. Mutating the eligibility
    check to ``state_merged or gh_merged`` (the pre-fix behavior) makes this
    test fail, since state.json here says "merged" but gh reports OPEN.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-10-stale-state", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    # state.json (stale/wrong) claims the PR is merged.
    state = _make_state(issue_number=10, pr_number=110, status="merged")

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(pr_state="OPEN"),
    )

    assert len(result.data["removed"]) == 0
    assert len(result.data["skipped"]) == 1
    reason = result.data["skipped"][0]["reason"].lower()
    assert "state.json" in reason
    assert "gh pr view" in reason
    assert info.path.exists()


def test_clean_worktrees_gh_unavailable_fails_closed(tmp_path: Path) -> None:
    """Review finding 3: when gh itself is unavailable/erroring, clean_worktrees
    must fail CLOSED (skip) rather than falling back to trusting state.json,
    even though state.json says "merged".
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-11-gh-down", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=11, pr_number=111, status="merged")

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(available=False, error="gh: network is unreachable"),
    )

    assert len(result.data["removed"]) == 0
    assert len(result.data["skipped"]) == 1
    reason = result.data["skipped"][0]["reason"]
    assert "unavailable" in reason.lower() or "cannot confirm" in reason.lower()
    assert "network is unreachable" in reason
    assert info.path.exists()


def test_clean_worktrees_removes_junctioned_worktree_and_preserves_shared_venv_contents(
    tmp_path: Path,
) -> None:
    """Non-blocking review item: junction safety through the clean_worktrees
    path itself (not just remove_worktree in isolation). A worktree with a
    live ``.venv`` junction removed via clean_worktrees must delete only the
    reparse point, never the shared venv's real contents.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    # Mirror this repo's own .gitignore (.venv/ is ignored); otherwise the
    # .venv junction itself shows up as an untracked path under `git status
    # --porcelain` and the dirty check would (incorrectly, for this test's
    # purposes) treat every junctioned worktree as dirty.
    (repo_root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    _git(repo_root, "add", ".gitignore")
    _git(repo_root, "commit", "-m", "add .gitignore")
    venv_source = tmp_path / "shared-venv"
    venv_source.mkdir()
    marker = venv_source / "site-packages-marker.txt"
    marker.write_text("shared contents\n", encoding="utf-8")

    info = create_worktree(
        repo_root, "agent/issue-12-junction", base_ref="HEAD", venv_source=venv_source
    )
    assert is_junction(info.path / ".venv")
    head_sha = _git(info.path, "rev-parse", "HEAD").stdout.strip()

    worktrees_dir = _default_worktrees_dir(repo_root)
    # No devin.venv_source configured here: this test is only about the
    # removal path's junction safety (remove_worktree), not the separate
    # post-removal poisoned-.pth check, which has its own dedicated tests.
    config = OrchestratorConfig()
    state = _make_state(issue_number=12, pr_number=112)

    result = clean_worktrees(
        repo_root,
        worktrees_dir,
        state,
        config,
        _FakeGH(head_sha=head_sha),
    )

    assert len(result.data["removed"]) == 1
    assert not info.path.exists()
    # The shared venv itself, and its contents, must survive the junction-safe
    # removal driven through clean_worktrees.
    assert venv_source.exists()
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "shared contents\n"


def test_create_review_checkout_isolated_from_worker_worktree(tmp_path: Path) -> None:
    """Issue #370/#397: a reviewer checkout must never alias the worker's
    worktree for the same branch — different path, different key scheme
    (PR number vs branch slug), detached HEAD instead of a branch checkout.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    branch = "agent/issue-1-fix"

    # Simulate a live worker worktree for this branch/PR.
    worker_info = create_worktree(repo_root, branch, base_ref="HEAD")
    assert worker_info.path.exists()
    worker_marker = worker_info.path / "worker-only.txt"
    worker_marker.write_text("worker work in progress\n", encoding="utf-8")

    head_sha = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    reviews_dir = tmp_path / "reviews"

    review_info = create_review_checkout(repo_root, 1, head_sha, reviews_dir=reviews_dir)

    # Distinct path — never the worker's worktree directory.
    assert review_info.path != worker_info.path
    assert review_info.path == reviews_dir / "pr-1"
    assert review_info.path.exists()

    # Detached HEAD at the exact head_sha — no branch checked out.
    symbolic_ref = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=review_info.path,
        capture_output=True,
        text=True,
    )
    assert symbolic_ref.returncode != 0  # symbolic-ref fails in detached HEAD
    review_head = _git(review_info.path, "rev-parse", "HEAD").stdout.strip()
    assert review_head == head_sha

    # The worker's worktree and its in-progress file are completely untouched.
    assert worker_info.path.exists()
    assert worker_marker.exists()
    assert worker_marker.read_text(encoding="utf-8") == "worker work in progress\n"
    worker_branch = _git(worker_info.path, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert worker_branch == branch


def test_create_review_checkout_requires_head_sha(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    with pytest.raises(ValueError):
        create_review_checkout(repo_root, 42, "", reviews_dir=tmp_path / "reviews")


def test_create_review_checkout_replaces_stale_checkout(tmp_path: Path) -> None:
    """A second review round for the same PR at a new head_sha tears down and
    recreates the checkout rather than reusing/fast-forwarding it in place.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    reviews_dir = tmp_path / "reviews"

    first_sha = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    first_checkout = create_review_checkout(repo_root, 7, first_sha, reviews_dir=reviews_dir)
    assert first_checkout.path.exists()

    (repo_root / "second.txt").write_text("second commit\n", encoding="utf-8")
    _git(repo_root, "add", "second.txt")
    _git(repo_root, "commit", "-m", "second commit")
    second_sha = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    assert second_sha != first_sha

    second_checkout = create_review_checkout(repo_root, 7, second_sha, reviews_dir=reviews_dir)

    assert second_checkout.path == first_checkout.path
    assert (second_checkout.path / "second.txt").exists()
    review_head = _git(second_checkout.path, "rev-parse", "HEAD").stdout.strip()
    assert review_head == second_sha


def test_create_review_checkout_skips_fetch_when_commit_already_local(tmp_path: Path) -> None:
    """When head_sha is already present in the local object store, the fetch
    is skipped entirely — a commit unpushed to origin (or an origin whose
    object store doesn't advertise it) must not block the checkout."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo, bare=True)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Local-only commit: present in repo_root's object store but never
    # pushed, so origin's bare object store does not advertise it. If the
    # fetch were attempted (not skipped), it would fail against this origin.
    (repo_root / "local_only.txt").write_text("local work\n", encoding="utf-8")
    _git(repo_root, "add", "local_only.txt")
    _git(repo_root, "commit", "-m", "local only commit")
    head_sha = _git(repo_root, "rev-parse", "HEAD").stdout.strip()

    reviews_dir = tmp_path / "reviews"
    info = create_review_checkout(repo_root, 55, head_sha, reviews_dir=reviews_dir)

    assert info.path.exists()
    assert _git(info.path, "rev-parse", "HEAD").stdout.strip() == head_sha


def test_create_review_checkout_falls_back_to_refs_pull_head(tmp_path: Path) -> None:
    """When head_sha isn't local and a direct-by-SHA fetch is refused (GitHub,
    and most local git configs, disable uploadpack.allowReachableSHA1InWant),
    create_review_checkout falls back to fetching refs/pull/<pr>/head and
    proceeds once the sha becomes locally reachable.

    On this machine's git, local (file://-style path) transport may itself
    take a shortcut that lets the direct SHA fetch succeed anyway — the
    assertions below only require functional success and the correct sha
    checked out, not which internal fetch path ran.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo, bare=True)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # A third clone pushes a commit under refs/pull/7/head only — never as a
    # branch — mirroring how GitHub exposes PR heads that were never merged
    # into a tracked branch on origin.
    third_clone = tmp_path / "third"
    _clone_repo(remote_repo, third_clone)
    (third_clone / "pr_only.txt").write_text("pr head content\n", encoding="utf-8")
    _git(third_clone, "add", "pr_only.txt")
    _git(third_clone, "commit", "-m", "pr head commit")
    head_sha = _git(third_clone, "rev-parse", "HEAD").stdout.strip()
    _git(third_clone, "push", "origin", "HEAD:refs/pull/7/head")

    # repo_root never fetched refs/pull/*, so the object is absent locally.
    cat_file = subprocess.run(
        ["git", "cat-file", "-e", f"{head_sha}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert cat_file.returncode != 0

    reviews_dir = tmp_path / "reviews"
    info = create_review_checkout(repo_root, 7, head_sha, reviews_dir=reviews_dir)

    assert info.path.exists()
    assert _git(info.path, "rev-parse", "HEAD").stdout.strip() == head_sha
    assert (info.path / "pr_only.txt").read_text(encoding="utf-8") == "pr head content\n"


def test_remove_review_checkout_idempotent(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    reviews_dir = tmp_path / "reviews"

    head_sha = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    info = create_review_checkout(repo_root, 99, head_sha, reviews_dir=reviews_dir)
    assert info.path.exists()

    removed_first = remove_review_checkout(repo_root, 99, reviews_dir=reviews_dir)
    assert removed_first is True
    assert not info.path.exists()

    # Idempotent: calling again on an already-absent checkout is still True,
    # never raises.
    removed_second = remove_review_checkout(repo_root, 99, reviews_dir=reviews_dir)
    assert removed_second is True

    # Never dispatched at all: also True, never raises.
    removed_never_created = remove_review_checkout(repo_root, 12345, reviews_dir=reviews_dir)
    assert removed_never_created is True


def test_create_worktree_reuses_pristine_leftover_without_remote_probe(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #461: a pristine orchestrator-created leftover worktree is reclaimed
    directly, without any remote fetch or ls-remote probe.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    branch = "agent/issue-461-pristine"

    info1 = create_worktree(repo_root, branch, base_ref="HEAD")
    assert info1.path.exists()
    assert info1.reclaimed is None

    original_run_captured = create_worktree.__globals__["run_captured"]

    def _no_remote_calls(command, *, cwd, timeout_seconds):
        if command[:2] == ["git", "fetch"] and "origin" in command:
            raise AssertionError(f"Unexpected git fetch during pristine reclaim: {command}")
        if command[:2] == ["git", "ls-remote"]:
            raise AssertionError(f"Unexpected git ls-remote during pristine reclaim: {command}")
        return original_run_captured(command, cwd=cwd, timeout_seconds=timeout_seconds)

    monkeypatch.setattr("charlie_work.worktree.run_captured", _no_remote_calls)

    info2 = create_worktree(repo_root, branch, base_ref="HEAD")
    assert info2.path == info1.path
    assert info2.reclaimed == "reused"


def test_create_worktree_reuses_pristine_leftover_and_resets_to_fresh_base(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #461: a pristine leftover worktree is reset to the fetched base
    when origin/main has advanced, instead of being left at a stale commit.
    """
    from charlie_work import worktree

    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    branch = "agent/issue-461-stale-base"

    # First dispatch: worktree is at the current origin/main.
    info1 = create_worktree(repo_root, branch, base_ref="")
    old_base = _git(info1.path, "rev-parse", "HEAD").stdout.strip()
    assert info1.reclaimed is None

    # Advance origin/main.
    _git(remote_repo, "checkout", "main")
    (remote_repo / "new.txt").write_text("new\n", encoding="utf-8")
    _git(remote_repo, "add", "new.txt")
    _git(remote_repo, "commit", "-m", "advance main")
    new_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()
    assert new_tip != old_base

    original_run_captured = worktree.run_captured

    def _no_ls_remote(command, **kwargs):
        if isinstance(command, list) and command[:3] == ["git", "ls-remote", "origin"]:
            raise AssertionError(f"Unexpected git ls-remote during pristine reclaim: {command}")
        return original_run_captured(command, **kwargs)

    monkeypatch.setattr("charlie_work.worktree.run_captured", _no_ls_remote)

    # Second fresh dispatch should reuse the same worktree but at the new tip.
    info2 = create_worktree(repo_root, branch, base_ref="")
    assert info2.path == info1.path
    assert info2.reclaimed == "reused"
    assert _git(info2.path, "rev-parse", "HEAD").stdout.strip() == new_tip
    assert (info2.path / "new.txt").exists()


def test_create_worktree_remote_probe_failure_names_subcommand_and_uses_shorter_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #461: a failing remote probe uses the shorter network timeout and the
    resulting error names the failing git subcommand.
    """
    from charlie_work import worktree

    remote_repo = tmp_path / "remote.git"
    repo_root = tmp_path / "repo"
    _init_repo(remote_repo, bare=True)
    _clone_repo(remote_repo, repo_root)
    branch = "agent/issue-461-probe"

    info1 = create_worktree(repo_root, branch, base_ref="HEAD")
    # Create a local commit beyond the base so the unsafe-to-reset check must
    # consult the remote.
    (info1.path / "local.txt").write_text("local work\n", encoding="utf-8")
    _git(info1.path, "add", "local.txt")
    _git(info1.path, "commit", "-m", "local commit")

    calls: list[tuple[list[str], int]] = []
    original_run_captured = worktree.run_captured

    def _intercept(command, *, cwd, timeout_seconds):
        calls.append((command, timeout_seconds))
        if command[:3] == ["git", "ls-remote", "origin"]:
            return RunResult(
                returncode=None,
                stdout="",
                stderr="",
                timed_out=True,
                error=f"command timed out after {timeout_seconds}s",
            )
        return original_run_captured(command, cwd=cwd, timeout_seconds=timeout_seconds)

    monkeypatch.setattr("charlie_work.worktree.run_captured", _intercept)

    with pytest.raises(WorktreeProbeFailedError) as exc_info:
        create_worktree(repo_root, branch, base_ref="HEAD")

    assert "git ls-remote" in str(exc_info.value)
    ls_remote_calls = [c for c in calls if c[0][:3] == ["git", "ls-remote", "origin"]]
    assert ls_remote_calls
    assert all(timeout == worktree._REMOTE_TIMEOUT_SECONDS for _cmd, timeout in ls_remote_calls)


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse point regression")
def test_remove_worktree_directory_symlink_preserves_shared_venv_target(
    tmp_path: Path,
) -> None:
    """Issue #462: directory symlinks (and junctions) at .venv must be unlinked,
    never followed into the shared venv target.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    marker = shared_venv / "site-packages-marker.txt"
    marker.write_text("shared contents\n", encoding="utf-8")

    info = create_worktree(repo_root, "agent/issue-462-symlink", base_ref="HEAD")
    venv_path = info.path / ".venv"
    if venv_path.exists() or is_junction(venv_path):
        _unlink_reparse_point(venv_path)

    # Prefer a real directory symlink; fall back to a junction when the
    # process lacks the symlink privilege on this Windows machine.
    try:
        os.symlink(shared_venv, venv_path, target_is_directory=True)
    except OSError:
        import _winapi

        _winapi.CreateJunction(str(shared_venv), str(venv_path))

    assert venv_path.is_symlink() or is_junction(venv_path)

    removed = remove_worktree(repo_root, info.path)

    assert removed is True
    assert not info.path.exists()
    assert shared_venv.exists()
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "shared contents\n"


def test_remove_worktree_force_fallback_rmtree_succeeds_when_git_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #462: when git worktree remove fails, the force fallback rmtree
    removes the tree and reports success.
    """
    from charlie_work.subprocess_runner import RunResult
    import charlie_work.worktree as worktree_mod

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-462-fallback", base_ref="HEAD")

    original_run = worktree_mod.run_captured

    def fake_run(args: list[str], **kwargs: Any) -> RunResult:
        if args[:3] == ["git", "worktree", "remove"]:
            return RunResult(returncode=1, stdout="", stderr="", error="simulated git failure")
        return original_run(args, **kwargs)

    monkeypatch.setattr(worktree_mod, "run_captured", fake_run)

    removed = remove_worktree(repo_root, info.path, force=True)

    assert removed is True
    assert not info.path.exists()


def test_remove_worktree_reports_failure_when_tree_survives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #462: post-delete verification reports failure when the directory
    is not actually removed.
    """
    from charlie_work.subprocess_runner import RunResult
    import charlie_work.worktree as worktree_mod

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    info = create_worktree(repo_root, "agent/issue-462-survives", base_ref="HEAD")

    original_run = worktree_mod.run_captured

    def fake_run(args: list[str], **kwargs: Any) -> RunResult:
        if args[:3] == ["git", "worktree", "remove"]:
            return RunResult(returncode=1, stdout="", stderr="", error="simulated git failure")
        return original_run(args, **kwargs)

    monkeypatch.setattr(worktree_mod, "run_captured", fake_run)
    monkeypatch.setattr(worktree_mod, "_robust_rmtree", lambda path: None)

    removed = remove_worktree(repo_root, info.path, force=True)

    assert removed is False
    assert info.path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse point regression")
def test_clean_worktrees_orphan_sweep_removes_unregistered_tree_with_reparse_point(
    tmp_path: Path,
) -> None:
    """Issue #462: orphan directories left after a failed git worktree remove are
    detected and removed without following reparse points.
    """
    import _winapi

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    _git(repo_root, "add", ".gitignore")
    _git(repo_root, "commit", "-m", "ignore venv")

    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    marker = shared_venv / "site-packages-marker.txt"
    marker.write_text("shared contents\n", encoding="utf-8")

    worktrees_dir = _default_worktrees_dir(repo_root)
    orphan_dir = worktrees_dir / "agent-issue-462-orphan"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "stale.txt").write_text("stale\n", encoding="utf-8")
    _winapi.CreateJunction(str(shared_venv), str(orphan_dir / ".venv"))

    config = OrchestratorConfig()
    state = _make_state(issue_number=462, pr_number=462)
    result = clean_worktrees(repo_root, worktrees_dir, state, config, _FakeGH())

    assert result.ok is True
    assert not orphan_dir.exists()
    assert shared_venv.exists()
    assert marker.read_text(encoding="utf-8") == "shared contents\n"
    assert any(str(orphan_dir) == r["worktree"] for r in result.data["orphans"]["removed"])


def test_clean_worktrees_skips_orphan_sweep_when_worktree_list_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A git worktree list failure must not be read as zero registered worktrees.

    The orphan sweep must be skipped and surfaced as an attention event so a
    transient git hiccup cannot silently destroy live worker state.
    """
    from charlie_work.subprocess_runner import RunResult
    import charlie_work.worktree as worktree_mod

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    info = create_worktree(repo_root, "agent/issue-1-live", base_ref="HEAD")
    worktrees_dir = _default_worktrees_dir(repo_root)
    config = OrchestratorConfig()
    state = _make_state(issue_number=1, pr_number=101)

    original_run = worktree_mod.run_captured

    def fake_run(args: list[str], **kwargs: Any) -> RunResult:
        if args[:4] == ["git", "worktree", "list", "--porcelain"]:
            return RunResult(
                returncode=1,
                stdout="",
                stderr="simulated git failure",
                error="simulated git failure",
            )
        return original_run(args, **kwargs)

    monkeypatch.setattr(worktree_mod, "run_captured", fake_run)

    result = clean_worktrees(repo_root, worktrees_dir, state, config, _FakeGH())

    assert info.path.exists()
    assert not result.data["orphans"]["removed"]
    assert not result.data["orphans"]["failed"]
    assert any(e["type"] == "worktree_list_failed" for e in result.data["attention_events"])
    assert result.ok is False


# --- issue #659: git argv validation (defense-in-depth) -----------------------


def test_create_worktree_rejects_flag_like_branch(tmp_path: Path) -> None:
    """A flag-like branch name must raise before reaching any git argv."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    with pytest.raises(ValueError, match="create_worktree branch"):
        create_worktree(repo_root, "--exec=foo", base_ref="HEAD")


def test_create_worktree_rejects_rev_syntax_branch(tmp_path: Path) -> None:
    """A branch name with rev-syntax metacharacters must raise."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    with pytest.raises(ValueError, match="create_worktree branch"):
        create_worktree(repo_root, "foo~bar", base_ref="HEAD")


def test_create_worktree_rejects_flag_like_base_ref(tmp_path: Path) -> None:
    """A flag-like base_ref must raise before reaching any git argv."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    with pytest.raises(ValueError, match="create_worktree base_ref"):
        create_worktree(repo_root, "agent/issue-1-fix", base_ref="--upload-pack=evil")


def test_create_worktree_accepts_empty_base_ref(tmp_path: Path) -> None:
    """Empty base_ref (auto-resolve sentinel) must pass validation."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    info = create_worktree(repo_root, "agent/issue-659-valid", base_ref="")
    assert info.path.exists()


def test_create_review_checkout_rejects_flag_like_head_sha(tmp_path: Path) -> None:
    """A flag-like head_sha must raise before reaching ``git fetch``/``worktree add``."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    with pytest.raises(ValueError, match="create_review_checkout head_sha"):
        create_review_checkout(repo_root, 1, "--exec=foo", reviews_dir=tmp_path / "reviews")


def test_create_review_checkout_rejects_non_hex_head_sha(tmp_path: Path) -> None:
    """A non-hex head_sha must raise (format check, not just flags)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    with pytest.raises(ValueError, match="create_review_checkout head_sha"):
        create_review_checkout(repo_root, 1, "not-a-sha!", reviews_dir=tmp_path / "reviews")
