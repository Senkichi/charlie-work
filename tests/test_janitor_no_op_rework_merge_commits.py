"""Merge-commit git-path tests for the janitor no-op-rework gate.

Split out of ``tests/test_janitor_no_op_rework.py`` (issue #1558, Track 1;
itself split out of ``tests/test_janitor.py``): the real-git
merge-only-advance detection in ``_check_no_op_rework`` -- merges that
drag in non-merge commits clear the gate, merge-only advances fail it,
real worker commits clear it, and git failures degrade to a warning.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _janitor_fixtures import _config, _green_checks, _green_pr, _init_repo

from charlie_work.janitor import run_janitor


def test_no_op_rework_merge_with_non_merge_commit_clears_gate(tmp_path: Path) -> None:
    """Merge commits that bring in non-merge commits PLUS real worker commits clear the no-op gate (real git path)."""
    # Set up a local "remote" repo
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    # Create initial commit on main
    (remote_repo / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    initial_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Clone the remote repo to create a local repo
    local_repo = tmp_path / "local"
    subprocess.run(
        ["git", "clone", str(remote_repo), str(local_repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # Create agent branch and push it at the reviewed head
    subprocess.run(
        ["git", "checkout", "-b", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "push", "-u", "origin", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # Advance the branch with a merge that brings in a non-merge commit
    # Create a commit on main in the remote
    subprocess.run(["git", "checkout", "main"], cwd=remote_repo, check=True, capture_output=True)
    (remote_repo / "main-change.txt").write_text("main branch change")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "main branch change"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )

    # In the local repo, fetch and merge main into agent branch
    subprocess.run(
        ["git", "checkout", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "fetch", "origin"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "merge", "--no-ff", "origin/main"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # Add a REAL worker commit (non-merge) on the agent branch
    (local_repo / "worker-change.txt").write_text("real worker change")
    subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "real worker commit"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    final_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=local_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Push the merge commit and the worker commit
    subprocess.run(
        ["git", "push", "origin", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # Test with a PR that has advanced by a merge commit AND a real worker commit
    pr = _green_pr(headRefOid=final_sha, headRefName="agent/issue-123-test")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": initial_sha,
    }

    verdict = run_janitor(pr, _green_checks(), _config(), pr_state=pr_state, repo_root=local_repo)

    # Should PASS (the merge brings in a non-merge commit, AND there's a real worker commit)
    assert verdict.ok is True
    # Should NOT have a degradation warning (git succeeded, real path exercised)
    assert not any("git fetch/rev-list failed" in w for w in verdict.warnings)


def test_no_op_rework_merge_only_fails_gate(tmp_path: Path) -> None:
    """Merge-only advances (no real worker commits) fail the no-op gate (real git path)."""
    # Set up a local "remote" repo
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    # Create initial commit on main
    (remote_repo / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    initial_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Clone the remote repo to create a local repo
    local_repo = tmp_path / "local"
    subprocess.run(
        ["git", "clone", str(remote_repo), str(local_repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # Create agent branch and push it at the reviewed head
    subprocess.run(
        ["git", "checkout", "-b", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "push", "-u", "origin", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # Advance the branch with ONLY a merge commit (no real worker commits)
    # Create a commit on main in the remote
    subprocess.run(["git", "checkout", "main"], cwd=remote_repo, check=True, capture_output=True)
    (remote_repo / "main-change.txt").write_text("main branch change")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "main branch change"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )

    # In the local repo, fetch and merge main into agent branch (NO worker commit)
    subprocess.run(
        ["git", "checkout", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "fetch", "origin"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "merge", "--no-ff", "origin/main"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    merge_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=local_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Push the merge commit (no worker commit)
    subprocess.run(
        ["git", "push", "origin", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # Test with a PR that has advanced ONLY by a merge commit
    pr = _green_pr(headRefOid=merge_sha, headRefName="agent/issue-123-test")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": initial_sha,
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=local_repo,
        review_decision=pr_state,
    )

    # Should FAIL (merge-only advance is a no-op rework)
    assert verdict.ok is False
    # Should have the merge-only failure message
    assert any("only by merge commits" in f for f in verdict.failures)
    # Should NOT have a degradation warning (git succeeded, real path exercised)
    assert not any("git fetch/rev-list failed" in w for w in verdict.warnings)


def test_no_op_rework_real_commit_clears_gate(tmp_path: Path) -> None:
    """Real non-merge commits since verdict clear the no-op gate."""
    # Set up a local "remote" repo
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    # Create initial commit on main
    (remote_repo / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    initial_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Clone the remote repo to create a local repo
    local_repo = tmp_path / "local"
    subprocess.run(
        ["git", "clone", str(remote_repo), str(local_repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # Create agent branch and push it at the reviewed head
    subprocess.run(
        ["git", "checkout", "-b", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "push", "-u", "origin", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # Advance the branch with a real non-merge commit
    (local_repo / "test2.txt").write_text("real work")
    subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "real work"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    real_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=local_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Push the real commit
    subprocess.run(
        ["git", "push", "origin", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # Test with a PR that has advanced by a real commit
    pr = _green_pr(headRefOid=real_sha, headRefName="agent/issue-123-test")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": initial_sha,
    }

    verdict = run_janitor(pr, _green_checks(), _config(), pr_state=pr_state, repo_root=local_repo)

    # Should PASS (real non-merge commit clears the gate)
    assert verdict.ok is True
    # Should NOT have a degradation warning (git succeeded)
    assert not any("git fetch/rev-list failed" in w for w in verdict.warnings)


def test_no_op_rework_git_failure_degrades_to_warning(tmp_path: Path) -> None:
    """Git failures in criterion-2 detection degrade to warning, not failure."""
    # Set up a git repo WITHOUT origin (so git fetch will fail)
    _init_repo(tmp_path)
    # Create initial commit
    (tmp_path / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    initial_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Create a branch and advance it (no origin, so fetch will fail)
    subprocess.run(
        ["git", "checkout", "-b", "agent/issue-123-test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "test2.txt").write_text("some work")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "some work"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    advanced_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Test with a PR that has advanced but no origin (git fetch will fail)
    pr = _green_pr(headRefOid=advanced_sha, headRefName="agent/issue-123-test")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": initial_sha,
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        review_decision=pr_state,
    )

    # Should PASS (git failure degrades to warning, not failure)
    assert verdict.ok is True
    # Should have a warning about git failure
    assert any("git fetch/rev-list failed" in w for w in verdict.warnings)
