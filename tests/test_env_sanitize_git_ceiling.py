"""Issue #2060: sanitize_env's GIT_CEILING_DIRECTORIES containment for worker temp space.

Split out of ``test_env_sanitize.py`` (which sits just under the 800-line cap
enforced by the file-size ratchet, issue #1442) so these tests do not grow that
file past its cap.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from charlie_work.env_sanitize import sanitize_env
from charlie_work.layout import worker_tmp_dir


def test_sanitize_env_ceilings_worker_tmp_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker env must bound repo discovery below the worktree tmp root.

    A `git config` run from a non-repo dir under TMP would otherwise ascend
    into the enclosing (linked) worktree and write the shared .git/config of
    the main repo.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv("GIT_CEILING_DIRECTORIES", raising=False)

    env = sanitize_env(worktree_path)

    tmp_parent = Path(os.path.realpath(str(worker_tmp_dir(worktree_path).parent)))
    entries = env["GIT_CEILING_DIRECTORIES"].split(os.pathsep)
    assert str(tmp_parent) in entries


def test_sanitize_env_preserves_ambient_git_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ambient ceiling entry is merged, never replaced."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", "/existing/ceiling")

    env = sanitize_env(worktree_path)

    entries = env["GIT_CEILING_DIRECTORIES"].split(os.pathsep)
    assert "/existing/ceiling" in entries


def test_sanitize_env_ceiling_stops_discovery_below_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a non-repo dir under the worker tmp root cannot reach the
    enclosing repo's config.

    The worktree root is a real repo so discovery *would* succeed without the
    ceiling — this fails on unfixed code, not just on a broken fixture."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    subprocess.run(["git", "init"], cwd=worktree_path, check=True, capture_output=True)
    monkeypatch.delenv("GIT_CEILING_DIRECTORIES", raising=False)

    env = sanitize_env(worktree_path)
    scratch = Path(env["TMPDIR"]) / "plain-dir"
    scratch.mkdir()

    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"], cwd=scratch, env=env, capture_output=True
    )

    assert result.returncode != 0
