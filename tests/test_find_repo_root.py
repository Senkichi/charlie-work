"""Tests for find_repo_root, carved out of test_charlie_work.py (#1284)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from _helpers import _init_git_repo


def test_find_repo_root_explicit_raises_on_missing_path(tmp_path: Path) -> None:
    from charlie_work.paths import RepoNotFoundError, find_repo_root

    missing = tmp_path / "no-such-dir"

    try:
        find_repo_root(missing, explicit=True)
    except RepoNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RepoNotFoundError")


def test_find_repo_root_explicit_raises_when_not_git_repo(tmp_path: Path) -> None:
    from charlie_work.paths import RepoNotFoundError, find_repo_root

    non_git = tmp_path / "plain-dir"
    non_git.mkdir()

    try:
        find_repo_root(non_git, explicit=True)
    except RepoNotFoundError as exc:
        assert "git work tree" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RepoNotFoundError")


def test_find_repo_root_resolves_shared_root_from_linked_worktree(tmp_path: Path) -> None:
    """Issue #648: with no --repo, find_repo_root() invoked from inside a
    linked git worktree must resolve the *shared* (main) worktree root, not
    the worktree's own toplevel — otherwise runtime state silently targets a
    phantom, never-populated ``.var/charlie-work/`` directory."""
    from charlie_work.paths import find_repo_root, runtime_paths

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    # Seed a real state dir under the main root so we can distinguish it.
    main_state_dir = repo_root / ".var" / "charlie-work"
    main_state_dir.mkdir(parents=True)
    (main_state_dir / "state.json").write_text("{}", encoding="utf-8")

    branch = "agent/issue-648-linked"
    subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(tmp_path / "wt"), "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        resolved = find_repo_root(tmp_path / "wt")
        # Must resolve to the main worktree root, not the linked worktree.
        assert resolved == repo_root.resolve()
        paths = runtime_paths(resolved, ".var/charlie-work")
        assert paths.state_file.exists()
        assert paths.state_file == (repo_root / ".var" / "charlie-work" / "state.json").resolve()
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(tmp_path / "wt")],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )


def test_find_repo_root_explicit_main_worktree_returns_main_root(tmp_path: Path) -> None:
    """An explicit --repo pointing at the main checkout returns the main root.
    The shared-root resolution returns None in the main worktree (where
    --git-dir == --git-common-dir), so --show-toplevel is used and is correct."""
    from charlie_work.paths import find_repo_root

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    resolved = find_repo_root(repo_root, explicit=True)
    assert resolved == repo_root.resolve()


def test_find_repo_root_explicit_linked_worktree_resolves_main_root(tmp_path: Path) -> None:
    """Issue #648 review: an explicit --repo pointing at a linked worktree must
    also resolve to the shared main root, not the linked worktree's own
    toplevel.  The orchestrator's state is shared — there is no per-worktree
    state directory — so --repo <linked-worktree> would silently target a
    phantom state dir without this."""
    from charlie_work.paths import find_repo_root

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    branch = "agent/issue-648-explicit"
    linked_wt = tmp_path / "wt-explicit"
    subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(linked_wt), "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        resolved = find_repo_root(linked_wt, explicit=True)
        assert resolved == repo_root.resolve()
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(linked_wt)],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )


def test_find_repo_root_from_subdirectory_of_linked_worktree(tmp_path: Path) -> None:
    """Issue #648 review: find_repo_root from a *subdirectory* of a linked
    worktree must still resolve to the shared main root."""
    from charlie_work.paths import find_repo_root

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    linked_wt = tmp_path / "wt-subdir"
    subprocess.run(
        ["git", "worktree", "add", "-b", "agent/issue-648-subdir", str(linked_wt), "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subdir = linked_wt / "src" / "deep"
    subdir.mkdir(parents=True)
    try:
        resolved = find_repo_root(subdir)
        assert resolved == repo_root.resolve()
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(linked_wt)],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )


def test_find_repo_root_no_redirect_honors_linked_worktree(tmp_path: Path) -> None:
    """Issue #1600: ``find_repo_root(..., redirect_to_main_worktree=False)`` from
    inside a linked git worktree must return the linked worktree's own root, not
    the shared main root.  Read-only diagnostics (e.g. ``ast-equivalence-check``)
    need to inspect the worktree they were invoked from, not the main checkout.
    The default (``redirect_to_main_worktree=True``) still redirects for
    state-mutating commands — that behaviour is covered by
    ``test_find_repo_root_resolves_shared_root_from_linked_worktree`` above."""
    from charlie_work.paths import find_repo_root

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    linked_wt = tmp_path / "wt-no-redirect"
    subprocess.run(
        ["git", "worktree", "add", "-b", "agent/issue-1600", str(linked_wt), "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        resolved = find_repo_root(linked_wt, redirect_to_main_worktree=False)
        # Must resolve to the linked worktree's own root, NOT the main root.
        assert resolved == linked_wt.resolve()
        assert resolved != repo_root.resolve()
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(linked_wt)],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )


def test_find_repo_root_no_redirect_in_main_worktree_unchanged(tmp_path: Path) -> None:
    """Issue #1600: ``redirect_to_main_worktree=False`` in the *main* worktree
    must still return the main root — the opt-out only matters for linked
    worktrees, and must not perturb the main-worktree path."""
    from charlie_work.paths import find_repo_root

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    resolved = find_repo_root(repo_root, redirect_to_main_worktree=False)
    assert resolved == repo_root.resolve()


def test_find_repo_root_separate_git_dir_main_worktree(tmp_path: Path) -> None:
    """Issue #648 review MAJOR 1: a --separate-git-dir repo's main worktree
    must resolve to the *working tree* root (where the code lives), not the
    external git dir's container.  The shared-root resolution detects the main
    worktree (--git-dir == --git-common-dir) and returns None, so
    --show-toplevel is used and returns the working tree root."""
    from charlie_work.paths import find_repo_root

    repo_root = tmp_path / "repo"
    external_git = tmp_path / "external" / ".git"
    repo_root.mkdir(parents=True, exist_ok=True)
    external_git.parent.mkdir(parents=True, exist_ok=True)
    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )
    run(["git", "init", f"--separate-git-dir={external_git}", "--initial-branch=main"])
    run(["git", "config", "user.email", "test@example.test"])
    run(["git", "config", "user.name", "Test User"])
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    run(["git", "add", "README.md"])
    run(["git", "commit", "-m", "initial commit"])
    # The external dir's parent must NOT be returned as the repo root.
    resolved = find_repo_root(repo_root)
    assert resolved == repo_root.resolve()
    assert resolved != external_git.parent.resolve()
