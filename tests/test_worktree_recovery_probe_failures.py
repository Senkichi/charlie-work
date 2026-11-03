"""Recovery probe failures: transient fetch, dirty-probe and local-tip probe errors
refuse the reset; ``_is_confirmed_missing_ref`` classification."""

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
