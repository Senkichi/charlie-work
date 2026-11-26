"""``create_review_checkout`` / ``remove_review_checkout`` and their input validation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _clone_repo, _git, _init_repo
from charlie_work.worktree import create_review_checkout, create_worktree, remove_review_checkout


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
