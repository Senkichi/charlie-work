"""Rework dispatch against an existing worktree/branch: reuse, attach, fast-forward,
reset and merge-update conflicts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _clone_repo, _git, _init_repo
from _worktree_shared_fixtures import _force_capture_failure
from charlie_work.subprocess_runner import RunResult
from charlie_work import worktree as worktree_module
from charlie_work.worktree import (
    WorktreeUnsafeError,
    ReworkBranchConflictError,
    ReworkMergeConflict,
    _merge_update_rework_branch,
    create_worktree,
    list_worktrees,
    remove_worktree,
)


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


def test_merge_update_rework_branch_real_conflict_reports_conflicted_files(
    tmp_path: Path,
) -> None:
    """Case (a) of the three-outcome split: a genuine content conflict returns
    a ``ReworkMergeConflict`` naming the conflicting file, and the internal
    abort leaves no ``MERGE_HEAD`` behind.

    A naive "any merge failure -> ReworkMergeConflict" fix would satisfy a
    test that only checks the return type. Asserting ``conflicted_files`` is
    non-empty and names the actual file is what distinguishes a real conflict
    from the pre-merge failure covered below, which never touches MERGE_HEAD
    at all.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    _git(repo_root, "checkout", "-b", "feature")
    (repo_root / "file.txt").write_text("feature line\n", encoding="utf-8")
    _git(repo_root, "add", "file.txt")
    _git(repo_root, "commit", "-m", "feature change")

    _git(repo_root, "checkout", "main")
    (repo_root / "file.txt").write_text("main line\n", encoding="utf-8")
    _git(repo_root, "add", "file.txt")
    _git(repo_root, "commit", "-m", "main change")

    _git(repo_root, "checkout", "feature")

    result = _merge_update_rework_branch(repo_root, repo_root, "feature", "main")

    assert isinstance(result, ReworkMergeConflict)
    assert result.conflicted_files
    assert "file.txt" in result.conflicted_files

    merge_head_check = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert merge_head_check.returncode != 0


def test_merge_update_rework_branch_abort_failure_still_raises_conflict_stage(
    tmp_path: Path,
) -> None:
    """Case (b) safety property: when a real conflict's ``git merge --abort``
    itself fails, the worktree is genuinely mid-merge and unusable. This must
    still raise ``ReworkBranchConflictError`` with ``stage="conflict"`` — the
    property the fix must not have dropped while adding the pre_merge branch.

    ``run_captured`` is monkeypatched for the ``merge --abort`` call (its
    outcome is simulated entirely) and, separately, to overwrite just the
    ``.stderr`` field of the real ``merge`` call's result with a synthetic
    marker — the merge itself, its conflict, and the MERGE_HEAD it leaves
    behind are still produced by real git; only the text of that one field is
    substituted. This is necessary because real git writes conflict
    diagnostics to stdout, not stderr (``CONFLICT (content): ...`` never
    populates ``RunResult.stderr``), so a real conflict can never by itself
    demonstrate that the merge's stderr is *preferred* over the abort's —
    without the synthetic marker, the "or" fallback in the implementation
    would trivially resolve to the abort's stderr regardless of which side
    the ordering actually favors, and this test would pass for the wrong
    reason.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    _git(repo_root, "checkout", "-b", "feature")
    (repo_root / "file.txt").write_text("feature line\n", encoding="utf-8")
    _git(repo_root, "add", "file.txt")
    _git(repo_root, "commit", "-m", "feature change")

    _git(repo_root, "checkout", "main")
    (repo_root / "file.txt").write_text("main line\n", encoding="utf-8")
    _git(repo_root, "add", "file.txt")
    _git(repo_root, "commit", "-m", "main change")

    _git(repo_root, "checkout", "feature")

    original_run_captured = worktree_module.run_captured
    _MERGE_STDERR_MARKER = "SIMULATED_MERGE_STDERR_MARKER"

    def mock_run_captured(*args: object, **kwargs: object) -> object:
        if args and args[0] == ["git", "merge", "--no-edit", "main"]:
            real_result = original_run_captured(*args, **kwargs)
            return RunResult(
                returncode=real_result.returncode,
                stdout=real_result.stdout,
                stderr=_MERGE_STDERR_MARKER,
                timed_out=real_result.timed_out,
                error=real_result.error,
            )
        if args and args[0] == ["git", "merge", "--abort"]:
            return RunResult(
                returncode=1,
                stdout="",
                stderr="simulated abort failure",
                error="simulated abort failure",
            )
        return original_run_captured(*args, **kwargs)

    worktree_module.run_captured = mock_run_captured
    try:
        with pytest.raises(ReworkBranchConflictError) as exc_info:
            _merge_update_rework_branch(repo_root, repo_root, "feature", "main")
        assert exc_info.value.stage == "conflict"
        # The conflicted paths must come from the real (pre-abort) merge
        # state, not be empty just because the abort path was exercised.
        assert exc_info.value.conflicted_paths
        assert "file.txt" in exc_info.value.conflicted_paths
        # The fix prefers the real merge's stderr over the abort's stderr
        # (`merge_result.stderr or abort_result.stderr`, not the reverse) —
        # the merge conflict is the actionable diagnostic; the abort failure
        # is secondary.
        assert exc_info.value.stderr == _MERGE_STDERR_MARKER

        # The real abort never ran, so the worktree is genuinely still
        # mid-merge — exactly the state that must escalate rather than be
        # handed to a worker.
        merge_head_check = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert merge_head_check.returncode == 0
    finally:
        worktree_module.run_captured = original_run_captured
        # Actually abort the still-mid-merge state so tmp_path teardown isn't
        # fighting a live index lock / MERGE_HEAD.
        subprocess.run(
            ["git", "merge", "--abort"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )


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


def test_rework_reuse_refuses_non_ff_with_dirty_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    _force_capture_failure(monkeypatch)
    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted source work"):
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
