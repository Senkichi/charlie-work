"""Shared fixtures/helpers for the claude-code adapter test modules.

Originally hoisted out of ``test_claude_code_adapter.py`` (issue #1284): a
``monkeypatch``-based ``create_worktree`` stand-in, imported by other test
modules, plus the two small ``WorktreeInfo`` builders it closes over.
``_fake_worktree`` still has real internal callers in the split-out test
modules, so they import it back (general back-reference rule).
``_fake_worktree_with_venv`` had none -- its only caller was always this
fixture's ``with_venv=True`` branch -- so it moved here outright with no
back-reference needed.

Issue #1560 (Track 1) split ``test_claude_code_adapter.py`` into seam-named
siblings; the helpers used by more than one sibling (``_fake_claude_script``,
``_init_real_repo``, ``_repo_head_sha``) moved here verbatim so the siblings
can share them -- the ``tests/_*.py`` hoisted-fixture convention sanctioned
by ``tests/test_zero_cross_test_import_guard.py``. All are plain
module-level imports: there is no circular-import hazard to defer around.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
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


def _fake_claude_script(tmp_path: Path) -> tuple[str, ...]:
    """A Python script standing in for the `claude` binary: reads stdin (the
    prompt), writes a marker file next to cwd, and exits 0."""
    script_path = tmp_path / "fake_claude.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path

            data = sys.stdin.read()
            Path("worker-ran.txt").write_text(data, encoding="utf-8")
            print("ok")
            """
        ),
        encoding="utf-8",
    )
    return (sys.executable, str(script_path))


def _init_real_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"], cwd=repo_root, check=True, capture_output=True
    )


def _repo_head_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
