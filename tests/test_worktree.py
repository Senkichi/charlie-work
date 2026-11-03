"""Core worktree lifecycle: create/remove, venv junctions, listing, ref validation.

The remaining seams of the former single ``test_worktree.py`` live in the
``test_worktree_*`` siblings (issue #1555, Track 1)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from _worktree_fixtures import _clone_repo, _git, _init_repo, _seed_tracked_venv_branch
from charlie_work.subprocess_runner import RunResult
from charlie_work.worktree import (
    WorktreeInfo,
    WorktreeProbeFailedError,
    _default_worktrees_dir,
    create_worktree,
    is_junction,
    remove_worktree,
    worktree_head_sha,
    _slugify,
    _unlink_reparse_point,
)


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
    # Register the worktree with git directly rather than create_worktree:
    # remove_worktree is the subject here, and a pristine fresh repo
    # exercises none of create_worktree's leftover/reclaim checks (each of
    # which costs a git subprocess — the dominant per-test cost on Windows,
    # issue #2465).
    branch = "agent/issue-4-force"
    worktrees_dir = _default_worktrees_dir(repo_root)
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    wt_path = worktrees_dir / _slugify(branch)
    _git(repo_root, "worktree", "add", "-b", branch, str(wt_path), "HEAD")
    real_venv = wt_path / ".venv"
    real_venv.mkdir()
    (real_venv / "pyvenv.cfg").write_text("home = somewhere\n", encoding="utf-8")

    removed = remove_worktree(repo_root, wt_path, force=True)

    assert removed is True
    assert not wt_path.exists()


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
    # The base ref tracks ``.venv/``, so the fresh worktree already holds the
    # link target (a hand-made stale dir would die earlier, in ``git worktree
    # add``, and never reach the guard -- issue #2915). ``match`` pins the guard.
    base_ref = _seed_tracked_venv_branch(repo_root)

    with pytest.raises(RuntimeError, match="venv link target already exists"):
        create_worktree(repo_root, "agent/issue-6-dup", base_ref=base_ref, venv_source=venv_source)


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


def test_worktree_head_sha_returns_real_head(tmp_path: Path) -> None:
    """Issue #1243: worktree_head_sha against a real repo returns the actual
    HEAD SHA -- the local half of the orphan-sweep progress fingerprint."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    expected = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert worktree_head_sha(repo) == expected

    # A new commit moves the reported SHA (stranded-commit detection).
    (repo / "work.txt").write_text("stranded\n", encoding="utf-8")
    _git(repo, "add", "work.txt")
    _git(repo, "commit", "-m", "stranded work")
    new_expected = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert new_expected != expected
    assert worktree_head_sha(repo) == new_expected


def test_worktree_head_sha_missing_dir_returns_none(tmp_path: Path) -> None:
    assert worktree_head_sha(tmp_path / "does-not-exist") is None


def test_worktree_head_sha_non_repo_dir_returns_none(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert worktree_head_sha(plain) is None
