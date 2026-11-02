"""Issue #2195: foreign adoption refuses interactive operator session worktrees.

An interactive Claude Code session writes no ``.charlie-writer.json`` marker, so
the #1476 marker gate cannot see it. ``runtime.operator_worktree_roots``
(default ``[".claude/worktrees"]``) names directories whose worktrees are never
adopted.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _git, _init_repo

from charlie_work.config import OrchestratorConfig, RuntimeConfig
from charlie_work.worktree import WorktreeForeignWriterError, create_worktree

BRANCH = "agent/issue-2195-operator"


def _config(roots: tuple[str, ...]) -> OrchestratorConfig:
    return dataclasses.replace(
        OrchestratorConfig(), runtime=RuntimeConfig(operator_worktree_roots=roots)
    )


def _snapshot(path: Path) -> set[str]:
    return {str(p.relative_to(path)) for p in path.rglob("*") if ".git" not in p.parts[-1:]}


def test_default_root_is_claude_worktrees() -> None:
    assert RuntimeConfig().operator_worktree_roots == (".claude/worktrees",)


def test_refuses_foreign_worktree_under_operator_root_and_writes_nothing(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    foreign = repo_root / ".claude" / "worktrees" / "x"
    _git(repo_root, "worktree", "add", str(foreign), "-b", BRANCH)
    before = _snapshot(foreign)

    with pytest.raises(WorktreeForeignWriterError) as exc_info:
        create_worktree(repo_root, BRANCH, rework=True, worktrees_dir=tmp_path / "managed")

    assert "operator session worktree" in str(exc_info.value)
    assert _snapshot(foreign) == before
    assert not (foreign / ".orchestrator-prompt.md").exists()
    assert not (foreign / ".charlie-writer.json").exists()


def test_adopts_clean_foreign_worktree_outside_roots(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    foreign = tmp_path / "elsewhere"
    _git(repo_root, "worktree", "add", str(foreign), "-b", BRANCH)

    info = create_worktree(repo_root, BRANCH, rework=True, worktrees_dir=tmp_path / "managed")

    assert info.path == foreign
    assert info.foreign_adopted is True


def test_empty_roots_restores_adoption(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    foreign = repo_root / ".claude" / "worktrees" / "x"
    _git(repo_root, "worktree", "add", str(foreign), "-b", BRANCH)

    info = create_worktree(
        repo_root,
        BRANCH,
        rework=True,
        worktrees_dir=tmp_path / "managed",
        config=_config(()),
    )

    assert info.path == foreign
    assert info.foreign_adopted is True


def _link_dir(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def test_junction_under_root_pointing_elsewhere_is_judged_by_resolved_path(
    tmp_path: Path,
) -> None:
    """A link under the root whose target is outside it is not an operator
    worktree (resolved path wins), so it is adopted."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    real = tmp_path / "real-wt"
    _git(repo_root, "worktree", "add", str(real), "-b", BRANCH)
    # Register the worktree under its junction path by pointing the link at it.
    link = repo_root / ".claude" / "worktrees" / "link"
    _link_dir(link, real)

    from charlie_work.foreign_worktree import check_foreign_adoption

    assert (
        check_foreign_adoption(
            link,
            managed_path=tmp_path / "managed" / "m",
            repo_root=repo_root,
            registered=[{"worktree": str(repo_root)}],
            recovery=False,
            operator_worktree_roots=(".claude/worktrees",),
        )
        is True
    )


def test_junction_outside_root_pointing_into_root_is_refused(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    inside = repo_root / ".claude" / "worktrees" / "x"
    _git(repo_root, "worktree", "add", str(inside), "-b", BRANCH)
    link = tmp_path / "alias"
    _link_dir(link, inside)

    from charlie_work.foreign_worktree import check_foreign_adoption

    with pytest.raises(WorktreeForeignWriterError, match="operator session worktree"):
        check_foreign_adoption(
            link,
            managed_path=tmp_path / "managed" / "m",
            repo_root=repo_root,
            registered=[{"worktree": str(repo_root)}],
            recovery=False,
            operator_worktree_roots=(".claude/worktrees",),
        )
