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
dict), the directory is reclaimed by the existing worktree-removal path with
no new cleanup code, a reused worktree's previous round does not leak into
the next one, the directory can never register as worker-authored dirt
regardless of the target repo's own .gitignore, and a lingering handle under
the directory alone cannot permanently wedge worktree removal.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from charlie_work import worktree
from charlie_work.env_sanitize import sanitize_env
from charlie_work.subprocess_runner import RunResult
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


def test_sanitize_env_clears_stale_content_from_a_previous_round(tmp_path: Path) -> None:
    """A rework dispatch reuses the same worktree round over round (issue
    #1767 finding #2): without clearing, round N's worker would start with
    round N-1's leftover scratch files still in place at the same
    predictable path -- the exact collision class #1767 exists to
    eliminate, just cross-round instead of cross-session."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    env_round_1 = sanitize_env(worktree_path)
    stale_file = Path(env_round_1["TMP"]) / "pr-body.md"
    stale_file.write_text("round 1's PR body", encoding="utf-8")
    assert stale_file.exists()

    env_round_2 = sanitize_env(worktree_path)

    # Same directory (still keyed on the worktree, per the isolation
    # contract) but round 1's leftover file must be gone.
    assert env_round_2["TMP"] == env_round_1["TMP"]
    assert not stale_file.exists()


def test_sanitize_env_seeds_worker_tmp_with_a_self_ignoring_gitignore(tmp_path: Path) -> None:
    """Issue #1767 finding #4: the directory must be self-ignoring at
    creation, independent of the target repo's own .gitignore."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    env = sanitize_env(worktree_path)
    gitignore = Path(env["TMP"]) / ".gitignore"

    assert gitignore.exists()
    assert gitignore.read_text(encoding="utf-8").strip() == "*"


def test_worker_tmp_dir_is_invisible_to_git_status_without_a_repo_level_gitignore_rule(
    tmp_path: Path,
) -> None:
    """Issue #1767 finding #4: onboarding a new repo whose own .gitignore
    does not happen to cover ``.var/`` must not turn worker scratch files
    into unexplained worktree dirt (a fifth trigger for
    ``worktree_unsafe_shim_dirt``). The directory must be self-ignoring
    regardless of the target repo's config -- proven end-to-end against a
    real repo with NO .gitignore at all, not just by inspecting the file
    sanitize_env writes."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)  # no .gitignore anywhere in this repo

    worktree_path = tmp_path / "wt"
    subprocess.run(
        ["git", "worktree", "add", "-b", "agent/gitignore-test", str(worktree_path), "main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    env = sanitize_env(worktree_path)
    (Path(env["TMP"]) / "scratch.txt").write_text("scratch", encoding="utf-8")

    result = subprocess.run(
        ["git", "status", "--porcelain=v2", "-z", "--untracked-files=all"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


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


def _make_removal_worktree(tmp_path: Path, branch: str) -> tuple[Path, Path, Path]:
    """Shared setup for the finding-#5 regression tests below: a real git
    worktree with a scratch file already written under its worker-tmp dir.
    Returns (repo_root, worktree_path, worker_tmp_dir)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    worktree_path = tmp_path / "wt"
    subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(worktree_path), "main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    env = sanitize_env(worktree_path)
    tmp_dir = Path(env["TMP"])
    (tmp_dir / "locked.bin").write_text("held open by a just-killed child", encoding="utf-8")
    return repo_root, worktree_path, tmp_dir


def test_remove_worktree_recovers_when_worker_tmp_handle_releases_after_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1767 finding #5 regression: relocating worker TMP/TEMP/TMPDIR
    inside the worktree means a lingering child-process file handle under
    that one subdirectory alone can block the whole-tree rmtree that
    ``git worktree remove --force`` falls back to. Without the retry-with-
    backoff fix, a single such transient failure permanently wedges the
    worktree; this simulates exactly one transient failure (mirroring a
    Windows handle that releases shortly after the owning process dies) and
    asserts removal recovers instead of giving up."""
    branch = "agent/tmp-lock-recovers"
    repo_root, worktree_path, _ = _make_removal_worktree(tmp_path, branch)

    # Force `git worktree remove --force` itself to fail (e.g. a locked file
    # under the tree at the moment git tries), driving remove_worktree into
    # its rmtree fallback path.
    real_run_captured = worktree.run_captured

    def failing_git_remove(args, **kwargs):
        if args[:3] == ["git", "worktree", "remove"]:
            return RunResult(returncode=1, stdout="", stderr="simulated lock")
        return real_run_captured(args, **kwargs)

    monkeypatch.setattr(worktree, "run_captured", failing_git_remove)

    # Simulate a lingering handle under worker-tmp specifically: the FIRST
    # whole-tree rmtree attempt fails (as if worker-tmp were still locked);
    # every other _robust_rmtree call (including the one inside the
    # worker-tmp-specific reclaim retry, and the second whole-tree attempt)
    # behaves for real.
    real_robust_rmtree = worktree._robust_rmtree
    whole_tree_attempts = {"n": 0}

    def flaky_robust_rmtree(path: Path) -> bool:
        if path == worktree_path:
            whole_tree_attempts["n"] += 1
            if whole_tree_attempts["n"] == 1:
                return False
        return real_robust_rmtree(path)

    monkeypatch.setattr(worktree, "_robust_rmtree", flaky_robust_rmtree)

    # No `branch=` here deliberately: forcing `git worktree remove` to fail
    # leaves a stale "prunable" worktree admin entry that a SEPARATE,
    # pre-existing `git branch -D` limitation (confirmed independently of
    # this change, unrelated to worker-tmp) refuses to delete through until
    # a second `git worktree prune` runs -- not something finding #5's
    # worker-tmp-reclaim retry is about. Omitting `branch` isolates this
    # test to the directory-removal property the fix actually changes.
    removed = remove_worktree(repo_root, worktree_path, force=True, sleep=lambda _seconds: None)

    assert removed is True
    assert not worktree_path.exists()
    # Proves the retry path actually ran (one failed attempt, one recovery),
    # not that the whole tree simply never needed a retry in the first place.
    assert whole_tree_attempts["n"] == 2


def test_remove_worktree_logs_worker_tmp_dir_when_removal_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Issue #1767 finding #5 ('at minimum'): when the worker-tmp reclaim
    retry does not unblock removal either, the failure must name the
    worker-tmp dir explicitly so the cause is diagnosable, rather than
    surfacing as a generic, opaque worktree-removal failure."""
    branch = "agent/tmp-lock-persists"
    repo_root, worktree_path, tmp_dir = _make_removal_worktree(tmp_path, branch)

    real_run_captured = worktree.run_captured

    def failing_git_remove(args, **kwargs):
        if args[:3] == ["git", "worktree", "remove"]:
            return RunResult(returncode=1, stdout="", stderr="simulated persistent lock")
        return real_run_captured(args, **kwargs)

    monkeypatch.setattr(worktree, "run_captured", failing_git_remove)
    # Every rmtree attempt fails -- the lock never releases within the
    # bounded retry budget.
    monkeypatch.setattr(worktree, "_robust_rmtree", lambda path: False)

    with caplog.at_level(logging.WARNING, logger="charlie_work.worktree"):
        removed = remove_worktree(
            repo_root, worktree_path, force=True, sleep=lambda _seconds: None
        )

    assert removed is False
    assert any(str(tmp_dir) in record.message for record in caplog.records)
