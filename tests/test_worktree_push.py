"""``push_branch`` and ``salvage_push_stranded_commits`` (incl. the dry-run gate)."""

from __future__ import annotations

import os
from pathlib import Path

from _worktree_fixtures import _clone_repo, _git
from _worktree_shared_fixtures import _init_repo_with_remote
from charlie_work.worktree import (
    OPERATOR_MARKER_KIND,
    OPERATOR_MARKER_SESSION_ID,
    SalvagePushResult,
    create_worktree,
    push_branch,
    remove_worktree,
    salvage_push_stranded_commits,
    write_worktree_marker,
)


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


# ---------------------------------------------------------------------------
# Issue #1248: salvage_push_stranded_commits
# ---------------------------------------------------------------------------
#
# All tests here use real local git repos (a bare "origin" remote + a clone
# that owns the linked worktrees under test) -- the established pattern for
# worktree.py functions in this file. No network, no gh.


def test_salvage_push_stranded_fast_forward(tmp_path: Path) -> None:
    """Remote at X, worktree has 2 unpushed commits on top -> pure FF push."""
    remote, repo = _init_repo_with_remote(tmp_path)
    branch = "agent/issue-1248-a1"
    info = create_worktree(repo, branch, base_ref="origin/main")

    ok, error = push_branch(repo, branch, worktree_path=info.path)
    assert ok, error
    old_remote_sha = _git(remote, "rev-parse", branch).stdout.strip()

    for name in ("first.txt", "second.txt"):
        (info.path / name).write_text(f"{name}\n", encoding="utf-8")
        _git(info.path, "add", name)
        _git(info.path, "commit", "-m", f"add {name}")
    local_tip = _git(info.path, "rev-parse", "HEAD").stdout.strip()

    result = salvage_push_stranded_commits(repo, branch, info.path)

    assert isinstance(result, SalvagePushResult)
    assert result.pushed is True
    assert result.skip_reason is None
    assert result.error is None
    assert result.old_remote_sha == old_remote_sha
    assert result.new_remote_sha == local_tip
    assert result.commit_count == 2

    assert _git(remote, "rev-parse", branch).stdout.strip() == local_tip


def test_salvage_push_diverged_leaves_remote_untouched(tmp_path: Path) -> None:
    """Remote advanced with a commit the worktree has fetched but is not an
    ancestor of the local tip -> skip_reason="diverged", remote unchanged.
    """
    remote, repo = _init_repo_with_remote(tmp_path)
    branch = "agent/issue-1248-a2"
    info = create_worktree(repo, branch, base_ref="origin/main")
    ok, error = push_branch(repo, branch, worktree_path=info.path)
    assert ok, error
    old_remote_sha = _git(remote, "rev-parse", branch).stdout.strip()

    # Local, unpushed commit.
    (info.path / "local.txt").write_text("local\n", encoding="utf-8")
    _git(info.path, "add", "local.txt")
    _git(info.path, "commit", "-m", "local unpushed commit")

    # A second clone pushes a sibling commit on the same branch, advancing
    # origin without the first worktree's knowledge.
    other_clone = tmp_path / "other-clone"
    _clone_repo(remote, other_clone)
    _git(other_clone, "checkout", branch)
    (other_clone / "remote.txt").write_text("remote\n", encoding="utf-8")
    _git(other_clone, "add", "remote.txt")
    _git(other_clone, "commit", "-m", "remote sibling commit")
    _git(other_clone, "push", "origin", branch)

    # Fetch the new remote tip into the worktree's object store so
    # `_object_exists` succeeds and the divergence (not "unknown object") is
    # what `_is_ancestor` actually detects.
    _git(info.path, "fetch", "origin", branch)

    result = salvage_push_stranded_commits(repo, branch, info.path)

    assert result.pushed is False
    assert result.skip_reason == "diverged"
    assert result.error is None
    assert result.old_remote_sha != old_remote_sha  # remote moved via other_clone
    assert _git(remote, "rev-parse", branch).stdout.strip() == result.old_remote_sha


def test_salvage_push_remote_head_not_local(tmp_path: Path) -> None:
    """Remote tip is unknown to the worktree's object store (never fetched)
    -> skip_reason="remote_head_not_local", distinct from "diverged".
    """
    remote, repo = _init_repo_with_remote(tmp_path)
    branch = "agent/issue-1248-a3"
    info = create_worktree(repo, branch, base_ref="origin/main")
    ok, error = push_branch(repo, branch, worktree_path=info.path)
    assert ok, error

    (info.path / "local.txt").write_text("local\n", encoding="utf-8")
    _git(info.path, "add", "local.txt")
    _git(info.path, "commit", "-m", "local unpushed commit")

    other_clone = tmp_path / "other-clone"
    _clone_repo(remote, other_clone)
    _git(other_clone, "checkout", branch)
    (other_clone / "remote.txt").write_text("remote\n", encoding="utf-8")
    _git(other_clone, "add", "remote.txt")
    _git(other_clone, "commit", "-m", "remote sibling commit")
    _git(other_clone, "push", "origin", branch)

    # Deliberately do NOT fetch -- the worktree has never heard of the new
    # remote tip.
    result = salvage_push_stranded_commits(repo, branch, info.path)

    assert result.pushed is False
    assert result.skip_reason == "remote_head_not_local"
    assert result.error is None


def test_salvage_push_up_to_date_skips(tmp_path: Path) -> None:
    """Local branch tip already equals the remote tip -> no push attempted."""
    remote, repo = _init_repo_with_remote(tmp_path)
    branch = "agent/issue-1248-a4"
    info = create_worktree(repo, branch, base_ref="origin/main")
    ok, error = push_branch(repo, branch, worktree_path=info.path)
    assert ok, error
    remote_sha = _git(remote, "rev-parse", branch).stdout.strip()

    result = salvage_push_stranded_commits(repo, branch, info.path)

    assert result.pushed is False
    assert result.skip_reason == "up_to_date"
    assert result.old_remote_sha == remote_sha
    assert _git(remote, "rev-parse", branch).stdout.strip() == remote_sha


def test_salvage_push_never_pushed_creates_branch(tmp_path: Path) -> None:
    """Branch never reached origin, worktree has commits beyond the base ->
    pushed=True, old_remote_sha=None, branch created on the bare repo.
    """
    remote, repo = _init_repo_with_remote(tmp_path)
    branch = "agent/issue-1248-a5"
    info = create_worktree(repo, branch, base_ref="origin/main")

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit, never pushed")
    local_tip = _git(info.path, "rev-parse", "HEAD").stdout.strip()

    # Confirm the branch is genuinely absent on origin before salvage.
    show_ref_before = _git(remote, "show-ref")
    assert branch not in show_ref_before.stdout

    result = salvage_push_stranded_commits(repo, branch, info.path)

    assert result.pushed is True
    assert result.skip_reason is None
    assert result.old_remote_sha is None
    assert result.new_remote_sha == local_tip
    assert result.commit_count == 1
    assert _git(remote, "rev-parse", branch).stdout.strip() == local_tip


def test_salvage_push_never_pushed_no_commits_beyond_base_skips(tmp_path: Path) -> None:
    """Branch never reached origin AND has no commits beyond the base ->
    skip_reason="no_commits_beyond_base", no branch created.
    """
    remote, repo = _init_repo_with_remote(tmp_path)
    branch = "agent/issue-1248-a6"
    info = create_worktree(repo, branch, base_ref="origin/main")
    # No commits made -- worktree HEAD is exactly origin/main's tip.

    result = salvage_push_stranded_commits(repo, branch, info.path)

    assert result.pushed is False
    assert result.skip_reason == "no_commits_beyond_base"
    show_ref_after = _git(remote, "show-ref")
    assert branch not in show_ref_after.stdout


def test_salvage_push_no_worktree_skips(tmp_path: Path) -> None:
    """Worktree directory does not exist -> skip_reason="no_worktree"."""
    remote, repo = _init_repo_with_remote(tmp_path)
    missing_path = tmp_path / "does-not-exist"

    result = salvage_push_stranded_commits(repo, "agent/issue-1248-a7", missing_path)

    assert result.pushed is False
    assert result.skip_reason == "no_worktree"


def test_salvage_push_live_writer_marker_skips(tmp_path: Path) -> None:
    """A live (this-process) worker writer marker refuses the push."""
    remote, repo = _init_repo_with_remote(tmp_path)
    branch = "agent/issue-1248-a8-worker"
    info = create_worktree(repo, branch, base_ref="origin/main")

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit, never pushed")

    # The current test process's own pid is alive by construction.
    write_worktree_marker(info.path, os.getpid(), "worker-session-1", kind="worker")

    result = salvage_push_stranded_commits(repo, branch, info.path)

    assert result.pushed is False
    assert result.skip_reason == "live_writer_marker"
    show_ref_after = _git(remote, "show-ref")
    assert branch not in show_ref_after.stdout


def test_salvage_push_operator_claimed_marker_skips(tmp_path: Path) -> None:
    """An operator-claim marker refuses the push regardless of pid liveness."""
    remote, repo = _init_repo_with_remote(tmp_path)
    branch = "agent/issue-1248-a8-operator"
    info = create_worktree(repo, branch, base_ref="origin/main")

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit, never pushed")

    write_worktree_marker(
        info.path, os.getpid(), OPERATOR_MARKER_SESSION_ID, kind=OPERATOR_MARKER_KIND
    )

    result = salvage_push_stranded_commits(repo, branch, info.path)

    assert result.pushed is False
    assert result.skip_reason == "operator_claimed"
    show_ref_after = _git(remote, "show-ref")
    assert branch not in show_ref_after.stdout


# ---------------------------------------------------------------------------
# Issue #1326: dry_run gate for push_branch / salvage_push_stranded_commits
# ---------------------------------------------------------------------------
#
# Under dry_run=True, push_branch must short-circuit before the ``git push``
# subprocess call -- no real push reaches origin. salvage_push_stranded_commits
# threads dry_run through to push_branch. Both tests use real local git repos
# (bare "origin" remote + clone with linked worktrees) and verify the remote
# branch tip is unchanged after the call.


def test_push_branch_dry_run_does_not_push(tmp_path: Path) -> None:
    """dry_run=True short-circuits before ``git push``; origin is untouched."""
    remote, repo = _init_repo_with_remote(tmp_path)
    branch = "agent/issue-1326-push-dry"
    info = create_worktree(repo, branch, base_ref="origin/main")

    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit, never pushed")
    local_tip = _git(info.path, "rev-parse", "HEAD").stdout.strip()

    # The branch must NOT exist on origin before the call.
    show_ref_before = _git(remote, "show-ref")
    assert branch not in show_ref_before.stdout

    ok, error = push_branch(repo, branch, worktree_path=info.path, dry_run=True)

    # Dry-run returns the natural "nothing happened" success shape.
    assert ok is True
    assert error is None

    # The branch must STILL NOT exist on origin -- no real push happened.
    show_ref_after = _git(remote, "show-ref")
    assert branch not in show_ref_after.stdout
    # And the local tip is unchanged (the commit is still only local).
    assert _git(info.path, "rev-parse", "HEAD").stdout.strip() == local_tip


def test_salvage_push_stranded_commits_dry_run_does_not_push(tmp_path: Path) -> None:
    """dry_run=True threads through to push_branch; origin is untouched.

    The read-only probing (rev-parse, ls-remote, merge-base, rev-list) still
    runs, but the mutating ``git push`` is suppressed. The salvage returns
    pushed=True (the "nothing happened" success shape that lets downstream
    classification proceed under dry-run), but the remote branch tip is
    unchanged.
    """
    remote, repo = _init_repo_with_remote(tmp_path)
    branch = "agent/issue-1326-salvage-dry"
    info = create_worktree(repo, branch, base_ref="origin/main")

    # Make a commit that would be stranded (never pushed).
    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "stranded commit, never pushed")
    local_tip = _git(info.path, "rev-parse", "HEAD").stdout.strip()

    # The branch must NOT exist on origin before the call.
    show_ref_before = _git(remote, "show-ref")
    assert branch not in show_ref_before.stdout

    result = salvage_push_stranded_commits(repo, branch, info.path, dry_run=True)

    assert isinstance(result, SalvagePushResult)
    # Dry-run: push_branch returned (True, None), so salvage reports pushed=True
    # with the local tip as the would-be new remote SHA -- but no real push
    # reached origin.
    assert result.pushed is True
    assert result.error is None
    assert result.new_remote_sha == local_tip

    # The branch must STILL NOT exist on origin -- no real push happened.
    show_ref_after = _git(remote, "show-ref")
    assert branch not in show_ref_after.stdout
