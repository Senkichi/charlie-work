"""Session-scoped TMP/TEMP/TMPDIR isolation for worker subprocesses (issue
#1767). ``sanitize_env`` is the single chokepoint used by ``claude_code``,
``devin_shell``, and ``rescue_review``, so these tests exercise it directly
rather than duplicating coverage per adapter -- adapter-level "launch failure
returns an error value" tests live alongside each adapter's existing
sanitize_env coverage (``test_claude_code_adapter_launch_failures.py``,
``test_devin_shell_env.py``, ``test_rescue_review.py``).

Concurrent worker sessions on this host previously shared the ambient
TMP/TEMP, letting one session's scratch write collide with another's -- the
reported incident was a worker fetching a PR body to a literal
``/tmp/pr-body.md`` and reading back a different session's body. This module
proves: two concurrently constructed sessions get distinct TMP/TEMP/TMPDIR
dirs, the value genuinely reaches a real spawned subprocess (not just the env
dict), and the directory is reclaimed by the existing worktree-removal path
with no new cleanup code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from charlie_work.env_sanitize import sanitize_env
from charlie_work.worktree import remove_worktree


def _init_repo(repo_root: Path) -> None:
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


def test_sanitize_env_points_tmp_vars_at_worktree_local_dir(tmp_path: Path) -> None:
    """TMP/TEMP/TMPDIR must all point at a real, created worktree-local dir."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    env = sanitize_env(worktree_path)

    expected = worktree_path / ".var" / "worker-tmp"
    assert env.get("TMP") == str(expected)
    assert env.get("TEMP") == str(expected)
    assert env.get("TMPDIR") == str(expected)
    assert expected.is_dir()


def test_sanitize_env_two_concurrent_worktrees_get_distinct_tmp_dirs(tmp_path: Path) -> None:
    """Two sessions constructed via the same launcher code path (sanitize_env)
    must resolve to distinct TMP/TEMP/TMPDIR -- the core acceptance criterion
    for issue #1767: no two concurrent sessions may share a temp path."""
    worktree_a = tmp_path / "worktrees" / "issue-1"
    worktree_b = tmp_path / "worktrees" / "issue-2"
    worktree_a.mkdir(parents=True)
    worktree_b.mkdir(parents=True)

    env_a = sanitize_env(worktree_a)
    env_b = sanitize_env(worktree_b)

    assert env_a["TMP"] != env_b["TMP"]
    assert env_a["TEMP"] != env_b["TEMP"]
    assert env_a["TMPDIR"] != env_b["TMPDIR"]
    # And neither is the pre-fix shared host default -- each is genuinely
    # inside its own worktree.
    assert Path(env_a["TMP"]).is_relative_to(worktree_a)
    assert Path(env_b["TMP"]).is_relative_to(worktree_b)


def test_tmp_env_reaches_a_real_spawned_subprocess(tmp_path: Path) -> None:
    """The env dict is inert until a subprocess actually inherits it -- spawn
    a real Python child and have it report what it resolved via the stdlib
    tempfile module, rather than trusting the dict alone."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    env = sanitize_env(worktree_path)

    result = subprocess.run(
        [sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    expected = worktree_path / ".var" / "worker-tmp"
    assert Path(result.stdout.strip()) == expected


def test_worker_tmp_dir_reclaimed_by_worktree_removal(tmp_path: Path) -> None:
    """The existing worktree-removal path (remove_worktree) must reclaim the
    session temp dir -- no separate cleanup code is needed or wanted; this is
    a real git worktree, torn down via the same safe-removal path production
    uses (reparse-point-safe, never a raw recursive delete)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    worktree_path = tmp_path / "wt"
    subprocess.run(
        ["git", "worktree", "add", "-b", "agent/tmp-reclaim-test", str(worktree_path), "main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    env = sanitize_env(worktree_path)
    tmp_dir = Path(env["TMP"])
    (tmp_dir / "scratch.txt").write_text("scratch", encoding="utf-8")
    assert tmp_dir.is_dir()

    removed = remove_worktree(
        repo_root, worktree_path, force=True, branch="agent/tmp-reclaim-test"
    )

    assert removed is True
    assert not tmp_dir.exists()
    assert not worktree_path.exists()
