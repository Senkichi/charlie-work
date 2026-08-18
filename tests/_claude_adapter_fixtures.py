"""Shared fixtures for ``test_claude_code_adapter.py``'s externally-imported
surface.

Hoisted out of ``test_claude_code_adapter.py`` (issue #1284): a
``monkeypatch``-based ``create_worktree`` stand-in, imported by other test
modules, plus the two small ``WorktreeInfo`` builders it closes over.
``test_claude_code_adapter.py`` is one of the three monoliths issue #1284
marks out of scope for a full split -- only these exported symbols move;
the rest of the file is untouched.

``_fake_worktree`` still has real internal callers in
``test_claude_code_adapter.py``, so that file imports it back (general
back-reference rule). ``_fake_worktree_with_venv`` had none -- its only
caller was always this fixture's ``with_venv=True`` branch -- so it moves
here outright with no back-reference needed. Both are plain module-level
imports now: unlike an earlier version of this module, there is no
circular-import hazard to defer around, since this module no longer needs
anything from ``test_claude_code_adapter.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.worktree import WorktreeInfo


def _fake_worktree(tmp_path: Path, branch: str) -> WorktreeInfo:
    worktree_path = tmp_path / "worktrees" / branch.replace("/", "-")
    worktree_path.mkdir(parents=True, exist_ok=True)
    return WorktreeInfo(path=worktree_path, branch=branch, venv_junction=None)


def _fake_worktree_with_venv(tmp_path: Path, branch: str) -> WorktreeInfo:
    """Create a fake worktree with a .venv directory.

    This makes sanitize_env actively SET VIRTUAL_ENV (instead of POP-ing it),
    which makes the merge order testable: if worker_env is merged first,
    sanitize_env will clobber the override.
    """
    worktree_path = tmp_path / "worktrees" / branch.replace("/", "-")
    worktree_path.mkdir(parents=True, exist_ok=True)
    (worktree_path / ".venv").mkdir()
    return WorktreeInfo(path=worktree_path, branch=branch, venv_junction=None)


def _install_fake_create_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    calls: list[dict] | None = None,
    with_venv: bool = False,
) -> None:
    from charlie_work import claude_code

    def fake_create_worktree(
        repo_root,
        branch,
        *,
        base_ref="HEAD",
        worktrees_dir=None,
        venv_source=None,
        materialize_dirs=(),
        rework=False,
        recovery=None,
        issue_number=None,
        config=None,
        sessions_dir=None,
    ):
        if calls is not None:
            calls.append(
                {
                    "repo_root": repo_root,
                    "branch": branch,
                    "base_ref": base_ref,
                    "worktrees_dir": worktrees_dir,
                    "venv_source": venv_source,
                    "materialize_dirs": materialize_dirs,
                    "rework": rework,
                    "recovery": recovery,
                    "issue_number": issue_number,
                    "config": config,
                    "sessions_dir": sessions_dir,
                }
            )
        if with_venv:
            return _fake_worktree_with_venv(tmp_path, branch)
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)
