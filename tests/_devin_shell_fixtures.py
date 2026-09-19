"""Shared fakes/helpers for the devin-shell adapter test modules.

Hoisted verbatim out of ``tests/test_devin_shell.py`` (issue #1542, Track-1
pilot) when that module was split into seam-named siblings -- the
``tests/_*.py`` hoisted-fixture convention is the sanctioned import target
for shared test helpers (see ``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from charlie_work import devin_shell
from charlie_work.worktree import WorktreeInfo

# A tiny fake "devin" CLI: writes its argv to stdout and exits 0. Launched via
# sys.executable to dodge PATH entirely (mirrors the sys.executable fake-binary
# pattern in tests/test_charlie_work.py).
_FAKE_DEVIN_SLEEP = """
import sys, time
sys.stdout.write("fake-devin argv=" + " ".join(sys.argv[1:]) + "\\n")
sys.stdout.flush()
time.sleep(0.2)
"""

_FAKE_DEVIN_VERSION = """
import sys
print("devin-fake 0.0.1")
sys.exit(0)
"""


def _write_fake_devin(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_devin.py"
    script.write_text(body, encoding="utf-8")
    return script


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

    monkeypatch.setattr(devin_shell, "create_worktree", fake_create_worktree)


def _init_repo(repo_root: Path) -> None:
    """Create a minimal git repo at ``repo_root`` for worktree tests."""
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main"], cwd=repo_root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=repo_root, check=True, capture_output=True
    )
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"], cwd=repo_root, check=True, capture_output=True
    )


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _make_session_sidecar(sessions_dir: Path, issue_number: int, log_path: Path) -> Path:
    """Write a minimal session sidecar for failure-classification tests."""
    sidecar_path = sessions_dir / f"issue-{issue_number}.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": issue_number,
                "branch": f"agent/issue-{issue_number}",
                "worktree_path": "/tmp/wt",
                "prompt_path": "p.md",
                "command": ["devin", "--print"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(log_path),
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    return sidecar_path
