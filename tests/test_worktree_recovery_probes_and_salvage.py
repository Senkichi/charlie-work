"""Recovery when a probe fails (transient fetch, dirty-probe, local-tip probe,
``_is_confirmed_missing_ref``) and salvage of a worktree whose origin default
branch is not ``main`` or that has no origin."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _clone_repo, _git, _init_repo
from _worktree_shared_fixtures import _force_capture_failure
from charlie_work.subprocess_runner import RunResult
from charlie_work.worktree import (
    WorktreeProbeFailedError,
    WorktreeUnsafeError,
    _default_worktrees_dir,
    _has_origin_remote,
    _is_confirmed_missing_ref,
    _worktree_refuse_to_reset_reason,
    create_worktree,
    remove_worktree,
)


def test_recovery_transient_fetch_failure_via_probe_aborts(tmp_path: Path) -> None:
    """Issue #110: Transient fetch failure via _remote_branch_exists probe should abort, not fall through.

    When the _remote_branch_exists probe fails (network/auth error), the dispatch should abort
    with an error instead of falling through to the fetch-fallback path (which would delete
    the local worktree and branch). This test exercises the actual recovery= path, not the
    pre-existing rework=True fetch-raise branch.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Simulate a killed-before-push session: create branch locally but don't push
    branch_name = "agent/issue-110-transient-probe"
    recovery_record = {"branch_name": branch_name, "status": "dispatched"}
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Verify worktree exists locally
    assert info1.path.exists()
    assert (info1.path / "README.md").exists()

    # Break the origin remote to simulate a transient network error
    _git(repo_root, "remote", "set-url", "origin", "file:///nonexistent/path")

    # Recovery dispatch should abort with probe failure, NOT fall through to fetch-fallback
    with pytest.raises(
        RuntimeError, match="Failed to probe remote branch.*transient network or auth error"
    ):
        create_worktree(repo_root, branch_name, base_ref="HEAD", recovery=recovery_record)

    # Verify the local worktree and branch still exist (not deleted by fallback)
    assert info1.path.exists()
    assert (info1.path / "README.md").exists()

    # Verify the branch still exists
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Clean up
    _git(repo_root, "remote", "set-url", "origin", str(remote_repo))
    remove_worktree(repo_root, info1.path)


def test_fresh_dispatch_dirty_worktree_with_broken_remote_refuses_to_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #257: A dirty worktree must be refused even if the remote is broken.

    The salvage path is no longer used; the guard should refuse to reset before
    any push is attempted and leave the worktree intact.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create a worktree with dirty changes
    branch_name = "agent/issue-110-salvage-fail"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Add uncommitted changes
    (info1.path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    # Verify dirty state
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=info1.path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status_result.stdout.strip()

    # Break the origin remote to simulate push failure
    _git(repo_root, "remote", "set-url", "origin", "file:///nonexistent/path")

    # Fresh dispatch should refuse to reset the dirty worktree without trying to push.
    _force_capture_failure(monkeypatch)
    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted source work"):
        create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Verify the worktree still exists (not removed)
    assert info1.path.exists()
    # Verify the dirty file is still there
    assert (info1.path / "dirty.txt").read_text(encoding="utf-8") == "uncommitted\n"

    # Clean up
    _git(repo_root, "remote", "set-url", "origin", str(remote_repo))
    remove_worktree(repo_root, info1.path, force=True)


def test_dirty_probe_failure_refuses_to_reset(tmp_path: Path) -> None:
    """Issue #257: Failed dirty-probe should be treated as dirty and refused.

    When git status --porcelain fails (index lock, corruption, permissions),
    the guard should refuse to reset rather than risk discarding work. Per
    the issue #288 follow-up review (PR #314), this is raised as the distinct
    ``WorktreeProbeFailedError`` rather than ``WorktreeUnsafeError`` — a probe
    failure is transient contention, not a confirmed-dirty worktree, so the
    launch shim must classify it under a separate failure_kind that does not
    escalate to a human on first occurrence.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Create a worktree with dirty changes
    branch_name = "agent/issue-110-dirty-probe-fail"
    info1 = create_worktree(repo_root, branch_name, base_ref="HEAD")

    # Add uncommitted changes
    (info1.path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    # Monkeypatch run_captured to simulate a failed git status probe
    import charlie_work.worktree

    original_run_captured = charlie_work.worktree.run_captured

    def mock_run_captured(*args: object, **kwargs: object) -> object:
        # If this is a git status --porcelain=v2 call, return a failure. Matches
        # by prefix since the flag carries a mode suffix ("--porcelain=v2").
        if (
            isinstance(args[0], list)
            and "status" in args[0]
            and any(str(a).startswith("--porcelain") for a in args[0])
        ):
            from charlie_work.subprocess_runner import RunResult

            return RunResult(
                returncode=1,
                stdout="",
                stderr="index.lock: File exists",
                error="index.lock: File exists",
            )
        return original_run_captured(*args, **kwargs)

    charlie_work.worktree.run_captured = mock_run_captured

    try:
        # Fresh dispatch should refuse to reset because the probe could not confirm
        # the worktree is clean — but as a distinct exception type from a
        # confirmed-dirty worktree (see WorktreeProbeFailedError docstring).
        with pytest.raises(WorktreeProbeFailedError, match="worktree status probe failed"):
            create_worktree(repo_root, branch_name, base_ref="HEAD")

        # Verify the worktree still exists (not removed)
        assert info1.path.exists()
        # Verify the dirty file is still there
        assert (info1.path / "dirty.txt").read_text(encoding="utf-8") == "uncommitted\n"
    finally:
        charlie_work.worktree.run_captured = original_run_captured


def test_is_confirmed_missing_ref_true_for_quiet_verify_miss() -> None:
    """Issue #1011: with ``-q``, ``git rev-parse --verify`` reserves exit code
    1 exclusively for "the given ref does not resolve" (confirmed against git
    2.45 for both an unborn ``HEAD`` in an empty repo and a missing branch
    name -- both produce ``returncode=1`` with empty stdout/stderr). This is
    the one shape the discriminator must treat as "nothing to lose".
    """
    result = RunResult(returncode=1, stdout="", stderr="", error="command exited 1")
    assert _is_confirmed_missing_ref(result) is True


def test_is_confirmed_missing_ref_false_for_git_error_exit_128() -> None:
    """A broken probe -- not a git repository, corrupted refs, permissions --
    exits 128, not 1. Before this fix, ANY non-ok `rev-parse --verify` result
    (including this one) fell through to "nothing to lose"; this pins that
    the discriminator refuses instead of misreading a broken probe as a
    confirmed-absent ref.
    """
    result = RunResult(
        returncode=128,
        stdout="",
        stderr="fatal: not a git repository (or any of the parent directories): .git",
        error="command exited 128",
    )
    assert _is_confirmed_missing_ref(result) is False


def test_is_confirmed_missing_ref_false_for_timeout() -> None:
    """A timed-out probe must never be read as a confirmed-absent ref."""
    result = RunResult(
        returncode=None,
        stdout="",
        stderr="",
        timed_out=True,
        error="command timed out after 60s",
    )
    assert _is_confirmed_missing_ref(result) is False


def test_is_confirmed_missing_ref_false_for_missing_git_binary() -> None:
    """An ``OSError`` from a git binary that is not on PATH at all comes back
    as ``returncode=None`` with ``error`` set (not a timeout). This must
    refuse too, not be misread as a confirmed-absent ref -- the discriminator
    must survive the case where git cannot be found at all, not just where it
    runs and errors.
    """
    result = RunResult(
        returncode=None,
        stdout="",
        stderr="",
        error="[WinError 2] The system cannot find the file specified",
    )
    assert _is_confirmed_missing_ref(result) is False


def test_local_tip_probe_failure_refuses_to_reset(tmp_path: Path) -> None:
    """Issue #1011: the confirmed data-loss path. When the local-tip
    ``git rev-parse --verify -q`` probe fails for a reason OTHER than a
    confirmed-absent ref (index lock, AV-held handle, transient I/O error --
    simulated here as a "not a git repository" failure, exit 128), a fresh
    branch that was never created still has no worktree directory to trip
    the dirty check, so this is the pre-fix code's only way in: before this
    fix, ANY non-ok probe result fell through to ``return None`` --
    "nothing to lose" -- and authorized discarding unpushed commits. It must
    now raise ``WorktreeProbeFailedError`` (refuse) instead.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    branch_name = "agent/issue-1011-probe-fail"
    _git(repo_root, "branch", branch_name)

    import charlie_work.worktree

    original_run_captured = charlie_work.worktree.run_captured

    def mock_run_captured(*args: object, **kwargs: object) -> object:
        if (
            isinstance(args[0], list)
            and args[0][:4] == ["git", "rev-parse", "--verify", "-q"]
            and args[0][-1] == branch_name
        ):
            return RunResult(
                returncode=128,
                stdout="",
                stderr="fatal: not a git repository (or any of the parent directories): .git",
                error="command exited 128",
            )
        return original_run_captured(*args, **kwargs)

    charlie_work.worktree.run_captured = mock_run_captured
    try:
        with pytest.raises(WorktreeProbeFailedError, match="not a confirmed-missing ref"):
            _worktree_refuse_to_reset_reason(repo_root, branch_name, "HEAD")
    finally:
        charlie_work.worktree.run_captured = original_run_captured


def test_absent_ref_still_returns_none_not_a_permanent_refusal(tmp_path: Path) -> None:
    """Issue #1011's other failure direction: a branch that genuinely does
    not exist (never dispatched, or already cleaned up) must still resolve
    to ``None`` ("nothing to lose") through the real ``git`` binary -- not
    mocked, unlike the probe-failure test above. Getting this backwards (
    treating every non-ok ``rev-parse --verify`` as a refusal) would make
    every legitimately-absent branch permanently un-reclaimable, leaking a
    worktree/reclaim slot forever.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    reason = _worktree_refuse_to_reset_reason(repo_root, "agent/does-not-exist-1011", "HEAD")
    assert reason is None


def test_salvage_worktree_with_origin_non_main_default_with_commit(tmp_path: Path) -> None:
    """Issue #141: _salvage_worktree with origin, non-main default, worktree has commit.

    This is test 1a of the differentiating pair: origin exists, default branch is 'master'
    (not 'main'), and the worktree has one commit beyond origin/master. Salvage should
    return non-None because there is work to preserve.
    """
    from charlie_work.worktree import _salvage_worktree

    # Create a remote repo with 'master' as the default branch
    remote_repo = tmp_path / "remote"
    remote_repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=master"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (remote_repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    # Clone the remote repo
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Set up origin/HEAD to point to origin/master
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Create a worktree with a commit beyond origin/master
    branch_name = "agent/issue-141-salvage-with-commit"
    worktrees_dir = _default_worktrees_dir(repo_root)
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = worktrees_dir / branch_name.replace("/", "-")
    subprocess.run(
        ["git", "worktree", "add", "-b", branch_name, str(worktree_path), "origin/master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Add a commit to the worktree (so it has unpushed commits)
    (worktree_path / "file1.txt").write_text("partial work\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "file1.txt"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "partial work"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )

    # Verify the branch does NOT exist on origin (killed-before-push scenario)
    ls_remote_result = subprocess.run(
        ["git", "ls-remote", "origin", f"refs/heads/{branch_name}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not ls_remote_result.stdout.strip()

    # Call _salvage_worktree directly
    salvage_ref = _salvage_worktree(repo_root, worktree_path, branch_name)

    # Should return a salvage ref (not None) because there are unpushed commits
    assert salvage_ref is not None
    assert salvage_ref.startswith("salvage/")

    # Verify the salvage ref was created locally
    salvage_refs = subprocess.run(
        ["git", "show-ref"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert salvage_refs.returncode == 0
    assert "salvage/" in salvage_refs.stdout or "refs/salvage/" in salvage_refs.stdout

    # Clean up
    remove_worktree(repo_root, worktree_path, force=True, branch=branch_name)


def test_salvage_worktree_with_origin_non_main_default_clean(tmp_path: Path) -> None:
    """Issue #141: _salvage_worktree with origin, non-main default, clean worktree.

    This is test 1b of the differentiating pair: origin exists, default branch is 'master'
    (not 'main'), and the worktree is CLEAN (exactly at origin/master tip). Salvage should
    return None because there is nothing to preserve.

    This is the REAL mutation gate: a hardcoded "main" (or any wrong ref) makes merge-base
    fail → safe-default True → non-None → this test FAILS.
    """
    from charlie_work.worktree import _salvage_worktree

    # Create a remote repo with 'master' as the default branch
    remote_repo = tmp_path / "remote"
    remote_repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=master"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (remote_repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    # Clone the remote repo
    repo_root = tmp_path / "repo"
    _clone_repo(remote_repo, repo_root)

    # Set up origin/HEAD to point to origin/master
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Create a worktree exactly at origin/master tip (clean)
    branch_name = "agent/issue-141-salvage-clean"
    worktrees_dir = _default_worktrees_dir(repo_root)
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = worktrees_dir / branch_name.replace("/", "-")
    subprocess.run(
        ["git", "worktree", "add", "-b", branch_name, str(worktree_path), "origin/master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Verify the worktree is clean (no commits beyond origin/master)
    merge_base_result = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/master"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    merge_base = merge_base_result.stdout.strip()
    rev_list_result = subprocess.run(
        ["git", "rev-list", "--count", f"{merge_base}..HEAD"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    commit_count = int(rev_list_result.stdout.strip())
    assert commit_count == 0, (
        f"Expected clean worktree, got {commit_count} commits beyond merge-base"
    )

    # Verify that merge-base with "main" (the hardcoded bug) would FAIL
    # This is the mutation gate: if the code uses "main" instead of "origin/master",
    # merge-base will fail and the test should catch it
    merge_base_main_result = subprocess.run(
        ["git", "merge-base", "HEAD", "main"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    assert merge_base_main_result.returncode != 0, (
        "merge-base with 'main' should fail in a 'master' repo"
    )

    # Call _salvage_worktree directly
    salvage_ref = _salvage_worktree(repo_root, worktree_path, branch_name)

    # Should return None because the worktree is clean
    assert salvage_ref is None

    # Clean up
    remove_worktree(repo_root, worktree_path, force=True, branch=branch_name)


def test_salvage_worktree_no_origin_conservative(tmp_path: Path) -> None:
    """Issue #141: _salvage_worktree with no origin returns non-None conservatively.

    When there is no origin remote, there is no authoritative default-branch tip to compare
    against. The salvage logic conservatively assumes there are unpushed commits to preserve
    work. This is an intentional, documented trade-off.
    """
    from charlie_work.worktree import _salvage_worktree

    # Create a repo with 'master' as the default branch (no origin)
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Create a worktree with commits
    branch_name = "agent/issue-141-salvage-no-origin"
    worktrees_dir = _default_worktrees_dir(repo_root)
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = worktrees_dir / branch_name.replace("/", "-")
    subprocess.run(
        ["git", "worktree", "add", "-b", branch_name, str(worktree_path), "master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    # Add a commit to the worktree
    (worktree_path / "file1.txt").write_text("partial work\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "file1.txt"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "partial work"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )

    # Verify no origin remote exists
    assert not _has_origin_remote(repo_root)

    # Call _salvage_worktree directly
    salvage_ref = _salvage_worktree(repo_root, worktree_path, branch_name)

    # Should return a salvage ref (not None) due to conservative no-origin default
    assert salvage_ref is not None
    assert salvage_ref.startswith("salvage/")

    # Verify the salvage ref was created locally
    salvage_refs = subprocess.run(
        ["git", "show-ref"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert salvage_refs.returncode == 0
    assert "salvage/" in salvage_refs.stdout or "refs/salvage/" in salvage_refs.stdout

    # Clean up
    remove_worktree(repo_root, worktree_path, force=True, branch=branch_name)
