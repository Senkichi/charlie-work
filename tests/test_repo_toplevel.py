"""Direct unit tests for :func:`charlie_work.repo_toplevel._repo_is_toplevel`.

Issue #1854: the predicate backs ``cross_repo_gate``'s refusal to trust git
answers (``check-ignore``, ``ls-files``) obtained for a ``repo_root`` that
is not itself a git toplevel — git resolves the repository by walking up
from the cwd, so a plain nested dir would answer with the enclosing repo's
data. The gate-level wiring is covered by
``test_cross_repo_gate_neutral_candidates.py`` and
``test_cross_repo_gate_sibling_repo.py``; these tests pin the predicate's
own truth table on real repositories.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from charlie_work.repo_toplevel import _repo_is_toplevel


def _git(repo: Path, *args: str) -> None:
    """Run a git command in *repo*, raising on failure."""
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_repo_is_toplevel_true_for_real_git_root(tmp_path: Path) -> None:
    """A ``git init``-ed directory is its own toplevel."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")

    assert _repo_is_toplevel(repo) is True


def test_repo_is_toplevel_false_for_plain_dir_inside_enclosing_repo(
    tmp_path: Path,
) -> None:
    """The discriminator case (issue #1854): a plain directory nested
    inside an enclosing checkout is NOT a toplevel — ``rev-parse
    --show-toplevel`` resolves to the enclosing repo's root, not the
    nested dir, even though ``ls-files``/``check-ignore`` run there would
    happily answer with that foreign repo's data."""
    enclosing = tmp_path / "enclosing"
    enclosing.mkdir()
    _git(enclosing, "init", "-q")
    nested = enclosing / "fixture" / "repo"
    nested.mkdir(parents=True)  # plain dir — deliberately not git-init'ed.

    assert _repo_is_toplevel(nested) is False
    # Positive control on the same tree: the enclosing root still is one.
    assert _repo_is_toplevel(enclosing) is True


def test_repo_is_toplevel_true_for_linked_worktree_root(tmp_path: Path) -> None:
    """A linked worktree's root is its own toplevel even though its git
    dir lives under the main checkout's ``.git/worktrees/`` — the reason
    the check compares ``rev-parse --show-toplevel`` rather than probing
    for a ``.git`` directory (a linked root has a ``.git`` *file*)."""
    main_repo = tmp_path / "main"
    main_repo.mkdir()
    _git(main_repo, "init", "-q")
    (main_repo / "f.py").write_text("# seed\n", encoding="utf-8")
    _git(main_repo, "add", "f.py")
    _git(
        main_repo,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "commit",
        "-q",
        "-m",
        "init",
    )
    linked = tmp_path / "linked"
    _git(main_repo, "worktree", "add", "--detach", str(linked))

    assert _repo_is_toplevel(linked) is True


def test_repo_is_toplevel_false_for_non_repo_dir(tmp_path: Path) -> None:
    """A plain directory outside any repository is not a toplevel —
    ``rev-parse`` fails outright and the predicate fails closed rather
    than trusting whatever an enclosing checkout might have answered."""
    plain = tmp_path / "plain"
    plain.mkdir()

    assert _repo_is_toplevel(plain) is False
