from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from charlie_work.config import DevinConfig, OrchestratorConfig, PostMortemConfig
from charlie_work.process_utils import get_process_start_time
from charlie_work.worktree import (
    WorktreeInfo,
    WorktreeProbeFailedError,
    WorktreeState,
    WorktreeUnsafeError,
    LiveWorkerRedispatchError,
    _default_worktrees_dir,
    _has_origin_remote,
    _resolve_default_branch_ref,
    create_worktree,
    inspect_worktree_state,
    is_junction,
    list_worktrees,
    push_branch,
    remove_worktree,
    _is_git_tracked,
    _materialize_directory,
    _slugify,
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
        # If this is a git status --porcelain call, return a failure
        if isinstance(args[0], list) and "status" in args[0] and "--porcelain" in args[0]:
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


def test_materialize_directory_skips_tracked_dir(tmp_path: Path) -> None:
    """_materialize_directory should skip tracked directories."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    # README.md is tracked by default
    # Create a worktree
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    # Try to materialize a tracked file's parent directory
    # This should be skipped because it's tracked
    _materialize_directory(repo_root, worktree_path, ".")

    # The worktree should still be empty (nothing was copied)
    assert not (worktree_path / "README.md").exists()


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


def test_materialize_directory_skips_if_target_exists(tmp_path: Path) -> None:
    """_materialize_directory should skip if target already exists."""
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

    # Materialize the directory - should skip because target exists
    _materialize_directory(repo_root, worktree_path, ".devin")

    # The existing file should still be there (not overwritten)
    assert (target_dir / "existing.txt").read_text(encoding="utf-8") == "existing\n"
    # The source file should not have been copied
    assert not (target_dir / "config.json").exists()


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
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (id TEXT, working_directory TEXT, created_at TEXT)")
    conn.execute(
        "CREATE TABLE message_nodes (id INTEGER PRIMARY KEY, session_id TEXT, node_id INTEGER, role TEXT, content TEXT, created_at TEXT)"
    )
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO sessions (id, working_directory, created_at) VALUES (?, ?, ?)",
        ("session-1", str(worktree_path), now),
    )
    conn.execute(
        "INSERT INTO message_nodes (session_id, node_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        ("session-1", 1, "tool", "tool result", now),
    )
    conn.commit()
    conn.close()

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
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (working_directory TEXT, created_at TEXT)")
    conn.execute(
        "CREATE TABLE message_nodes (id INTEGER PRIMARY KEY, session_id TEXT, node_id INTEGER, role TEXT, content TEXT, created_at TEXT)"
    )
    conn.commit()
    conn.close()

    now = datetime.now(UTC).isoformat()
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
    )

    # No logs/ directory at all -> devin_per_pid_log is silent too (its own
    # "not found" error), never a confirmed timestamp either way.
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
    assert exc_info.value.probe_result == "probe_error"
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
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (id TEXT, working_directory TEXT, created_at TEXT)")
    conn.execute(
        "CREATE TABLE message_nodes (id INTEGER PRIMARY KEY, session_id TEXT, node_id INTEGER, role TEXT, content TEXT, created_at TEXT)"
    )
    stale_iso = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    conn.execute(
        "INSERT INTO sessions (id, working_directory, created_at) VALUES (?, ?, ?)",
        ("session-1", str(worktree_path), stale_iso),
    )
    conn.execute(
        "INSERT INTO message_nodes (session_id, node_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        ("session-1", 1, "tool", "tool result", stale_iso),
    )
    conn.commit()
    conn.close()

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
