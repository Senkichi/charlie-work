"""Tests for ``charlie_work.base_branch.resolve_base_branch_name`` (issue #1250).

Extracted out of ``tests/test_worktree.py`` by the file-size ratchet
(issue #1442): the #1250 regression tests grew ``test_worktree.py`` past its
recorded high-water mark, so they move here alongside the extracted
``base_branch.py`` module. The helpers ``_clone_repo`` / ``_git`` come from
``_worktree_fixtures`` (the shared git plumbing hoisted in #1284); the
``_init_repo_with_branch`` helper is local because it parametrizes the initial
branch name, which the stock ``_init_repo`` (hardcoded ``--initial-branch=main``)
does not. Where the moved tests previously called ``_init_repo(repo_root)``
(a ``main``-default repo with no remote), they now call
``_init_repo_with_branch(repo_root, "main")`` -- byte-identical behaviour --
so this module does not import a private helper out of the ``test_worktree.py``
monolith.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _clone_repo, _git
from charlie_work.base_branch import resolve_base_branch_name


def _init_repo_with_branch(repo_root: Path, initial_branch: str) -> None:
    """Init a non-bare repo with a custom initial branch name and one commit."""
    repo_root.mkdir(parents=True, exist_ok=True)
    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )
    run(["git", "init", f"--initial-branch={initial_branch}"])
    run(["git", "config", "user.email", "test@example.test"])
    run(["git", "config", "user.name", "Test User"])
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    run(["git", "add", "README.md"])
    run(["git", "commit", "-m", "initial commit"])


def test_resolve_base_branch_name_derives_trunk_default(tmp_path: Path) -> None:
    """Issue #1250: an unset base_ref must resolve to the repo's real default
    branch (here ``trunk``), not a hardcoded ``main``. Uses a local git fixture
    with no network: ``git clone`` sets ``refs/remotes/origin/HEAD`` to point at
    the remote's default branch, which ``resolve_base_branch_name`` reads via
    ``git symbolic-ref``.
    """
    remote_repo = tmp_path / "remote"
    _init_repo_with_branch(remote_repo, "trunk")
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # A fresh clone sets origin/HEAD -> origin/trunk automatically; verify the
    # precondition so the test fails loudly if the fixture regresses.
    symref = _git(repo_root, "symbolic-ref", "refs/remotes/origin/HEAD")
    assert symref.stdout.strip() == "refs/remotes/origin/trunk"

    assert resolve_base_branch_name(repo_root, "") == "trunk"


def test_resolve_base_branch_name_falls_back_to_main_when_repo_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Issue #1250: the ``"main"`` literal is reachable only when the repo
    itself provides no answer (no remote HEAD symref, no HEAD match), and its
    use is logged so the guess is visible. A bare ``git init`` with no remote
    has no ``refs/remotes/origin/HEAD`` to read.
    """
    repo_root = tmp_path / "repo"
    _init_repo_with_branch(repo_root, "main")

    with caplog.at_level("WARNING", logger="charlie_work.base_branch"):
        assert resolve_base_branch_name(repo_root, "") == "main"
    assert any("falling back to hardcoded 'main'" in record.message for record in caplog.records)


def test_resolve_base_branch_name_strips_origin_prefix(tmp_path: Path) -> None:
    """Prefix-stripping behavior is unchanged by the #1250 fix: an explicit
    ``origin/<branch>`` ref is returned as the bare branch name without
    consulting the remote HEAD.
    """
    repo_root = tmp_path / "repo"
    _init_repo_with_branch(repo_root, "main")
    assert resolve_base_branch_name(repo_root, "origin/develop") == "develop"
    assert resolve_base_branch_name(repo_root, "refs/remotes/origin/release") == "release"
    assert resolve_base_branch_name(repo_root, "refs/heads/feature") == "feature"


def test_resolve_base_branch_name_heals_missing_origin_head(tmp_path: Path) -> None:
    """Issue #1250 regression: a clone whose origin/HEAD symref is deleted must
    still resolve to the repo's real default branch, not a hardcoded ``main``.

    Mirrors ``test_resolve_default_branch_ref_heals_missing_origin_head`` but
    exercises the public ``resolve_base_branch_name`` fallback path. The remote
    uses a non-``main`` default branch (``trunk``) so the assertion discriminates
    a real heal from a hardcoded ``main`` fallback: against the unfixed code the
    deleted symref is not healed and the function returns ``"main"``, which is
    wrong for this repo.
    """
    remote_repo = tmp_path / "remote"
    _init_repo_with_branch(remote_repo, "trunk")
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Simulate the incident state from #239/#1250: origin remote present, but
    # the origin/HEAD symref is absent (deleted after clone).
    subprocess.run(
        ["git", "symbolic-ref", "--delete", "refs/remotes/origin/HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert resolve_base_branch_name(repo_root, "") == "trunk"

    # The heal must persist in-repo (set-head --auto), not just resolve
    # transiently — mirroring the assertion in the _resolve_default_branch_ref
    # healing test.
    symref = subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert symref.stdout.strip() == "refs/remotes/origin/trunk"


def test_resolve_base_branch_name_returns_main_when_origin_unhealable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Issue #1250 regression for the ``except RuntimeError`` branch in
    ``resolve_base_branch_name``: when an origin remote is present but its
    default branch cannot be healed (``_resolve_default_branch_ref`` raises),
    the function must return ``"main"`` without raising and log a warning
    carrying the underlying exception detail.

    Mirrors ``test_resolve_default_branch_ref_raises_when_unhealable``'s fixture
    (origin pointed at a nonexistent local path so ``set-head --auto`` fails
    fast without network), but exercises the public caller's contract that the
    RuntimeError is converted into a value plus a visible warning rather than
    propagated. Against an unfixed ``resolve_base_branch_name`` that did not
    catch the RuntimeError, this test would raise instead of returning ``main``.
    """
    repo_root = tmp_path / "repo"
    _init_repo_with_branch(repo_root, "main")
    _git(repo_root, "remote", "add", "origin", str(tmp_path / "does-not-exist"))

    with caplog.at_level("WARNING", logger="charlie_work.base_branch"):
        assert resolve_base_branch_name(repo_root, "") == "main"
    assert any(
        "falling back to hardcoded 'main'" in record.message and "issue #239" in record.message
        for record in caplog.records
    )
