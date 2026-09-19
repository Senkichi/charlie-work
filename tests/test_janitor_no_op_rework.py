"""No-op-rework gate tests for the janitor.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1): the
``_check_no_op_rework`` verdict surface (patch-id comparison, SHA fallback,
merge-only detection, unpushed-commit enrichment, skip conditions, and
flag-like argv validation) plus the ``_get_unpushed_commit_info`` helper
tests.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _janitor_fixtures import (
    _config,
    _green_checks,
    _green_pr,
    _init_repo,
)

from charlie_work.janitor import (
    JanitorVerdict,
    _calculate_patch_id,
    _get_unpushed_commit_info,
    run_janitor,
)


def test_no_op_rework_offset_shift_still_blocks(tmp_path: Path) -> None:
    """No-op rework gate blocks when only hunk offsets shifted (base-update scenario).

    This is the integration-level proof of issue #222: the reviewed_patch_id was
    recorded from the unshifted diff; after a base-update merge shifts line numbers
    the current diff has different @@ headers but identical content — the janitor
    must still recognise it as a no-op and block re-review.

    MUTATION CHECK: this test MUST FAIL against the pre-fix implementation (without
    the @@ skip in _calculate_patch_id), because the shifted hunk header would make
    _calculate_patch_id return a different hash and the gate would incorrectly pass.
    """
    diff_at_review_time = """\
diff --git a/src/foo.py b/src/foo.py
index aaaaaaa..bbbbbbb 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,5 +10,6 @@
 context line
-old line
+new line
 another context
"""
    # Base-update merge shifted the hunk by 4 lines — same content, different header
    diff_after_base_update = """\
diff --git a/src/foo.py b/src/foo.py
index aaaaaaa..bbbbbbb 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -14,5 +14,6 @@
 context line
-old line
+new line
 another context
"""
    reviewed_patch_id = _calculate_patch_id(diff_at_review_time)

    pr = _green_pr(headRefOid="def456")  # Head SHA changed by base-update merge
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
        "reviewed_patch_id": reviewed_patch_id,
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=diff_after_base_update,
        review_decision=pr_state,
    )

    assert verdict.ok is False, (
        "Expected no-op block but got ok=True. "
        "Did the @@ skip get removed from _calculate_patch_id?"
    )
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures), (
        f"Expected patch-id no-op failure, got: {verdict.failures}"
    )


def test_no_op_rework_detects_unchanged_patch_id() -> None:
    """Detect no-op rework when PR patch-id is unchanged since request_changes verdict."""
    pr = _green_pr(headRefOid="def456")  # Head SHA changed (e.g., base-update merge)
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",  # Old head SHA
        "reviewed_patch_id": "test-patch-id-123",  # Old patch-id
    }
    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    # Calculate patch-id for the current diff
    current_patch_id = _calculate_patch_id(diff)
    # Set the state to have the same patch-id (simulating unchanged diff content)
    pr_state["reviewed_patch_id"] = current_patch_id

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        pr_diff=diff,
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)
    assert "patch-id" in verdict.failures[0]


def test_no_op_rework_patch_id_change_clears_gate() -> None:
    """Patch-id change clears the no-op gate even if head SHA is unchanged."""
    pr = _green_pr(headRefOid="abc123")  # Head SHA unchanged
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",  # Same head SHA
        "reviewed_patch_id": "old-patch-id",  # Different patch-id
    }
    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    # Current diff has a different patch-id than the state
    current_patch_id = _calculate_patch_id(diff)
    assert current_patch_id != "old-patch-id"

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(require_issue_link=False),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        pr_diff=diff,
    )

    # Should PASS because patch-id changed (actual content changed)
    assert verdict.ok is True, f"Expected ok=True but got {verdict.failures}"
    assert not any("PR diff unchanged" in f for f in verdict.failures)


def test_no_op_rework_fallback_to_sha_without_patch_id() -> None:
    """Fall back to SHA comparison when patch-id is not available (old verdicts)."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",  # Old verdict without patch-id
    }
    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        pr_diff=diff,
        review_decision=pr_state,
    )

    # Should FAIL because SHA matches (fallback behavior)
    assert verdict.ok is False
    assert any("PR head unchanged since request_changes verdict" in f for f in verdict.failures)


def test_no_op_rework_skips_patch_id_check_without_diff() -> None:
    """Skip patch-id check when diff is not provided (falls back to SHA)."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
        "reviewed_patch_id": "some-patch-id",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        pr_diff=None,
        review_decision=pr_state,
    )

    # Should FAIL because SHA matches (fallback behavior when diff is None)
    assert verdict.ok is False
    assert any("PR head unchanged since request_changes verdict" in f for f in verdict.failures)


def test_no_op_rework_detects_unchanged_head() -> None:
    """Detect no-op rework when PR head is unchanged since request_changes verdict."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("PR head unchanged since request_changes verdict" in f for f in verdict.failures)
    assert "abc123" in verdict.failures[0]


def test_no_op_rework_skips_when_no_verdict() -> None:
    """Skip no-op rework check when there's no request_changes verdict."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "approved",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(pr, _green_checks(), _config(), pr_state=pr_state, repo_root=Path.cwd())

    assert verdict.ok is True
    assert not any("PR head unchanged" in f for f in verdict.failures)


def test_no_op_rework_skips_when_head_advanced() -> None:
    """Skip no-op rework check when PR head has advanced since verdict."""
    pr = _green_pr(headRefOid="def456")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(pr, _green_checks(), _config(), pr_state=pr_state, repo_root=Path.cwd())

    assert verdict.ok is True
    assert not any("PR head unchanged" in f for f in verdict.failures)


def test_no_op_rework_skips_when_no_pr_state() -> None:
    """Skip no-op rework check when pr_state is None."""
    pr = _green_pr(headRefOid="abc123")

    verdict = run_janitor(pr, _green_checks(), _config(), repo_root=Path.cwd())

    assert verdict.ok is True
    assert not any("PR head unchanged" in f for f in verdict.failures)


def test_no_op_rework_skips_when_no_reviewed_sha() -> None:
    """Skip no-op rework check when reviewed_head_sha is missing."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
    }

    verdict = run_janitor(pr, _green_checks(), _config(), pr_state=pr_state, repo_root=Path.cwd())

    assert verdict.ok is True
    assert not any("PR head unchanged" in f for f in verdict.failures)


def test_no_op_rework_skips_when_no_current_sha() -> None:
    """Skip no-op rework check when current headRefOid is missing."""
    pr = _green_pr()
    pr.pop("headRefOid", None)
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(pr, _green_checks(), _config(), pr_state=pr_state, repo_root=Path.cwd())

    assert verdict.ok is True
    assert not any("PR head unchanged" in f for f in verdict.failures)


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


def test_no_op_rework_unpushed_commit_enrichment(tmp_path: Path) -> None:
    """Enrich failure message with unpushed commit count when worktree exists."""
    # Set up a git repo with a worktree
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

    # Create a worktree for the branch
    worktrees_dir = tmp_path / ".var" / "charlie-work" / "worktrees"
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = worktrees_dir / "agent-issue-123-test"
    subprocess.run(
        ["git", "worktree", "add", "-b", "agent/issue-123-test", str(worktree_path)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Add an unpushed commit in the worktree
    (worktree_path / "unpushed.txt").write_text("unpushed work")
    subprocess.run(["git", "add", "."], cwd=worktree_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "unpushed work"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
    )

    # Test with a PR that has unchanged head (no-op rework)
    pr = _green_pr(headRefOid=initial_sha, headRefName="agent/issue-123-test")
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

    # Should fail (no-op rework)
    assert verdict.ok is False, (
        f"Expected no-op rework to fail, but verdict.ok={verdict.ok}, failures={verdict.failures}"
    )
    # The unpushed commit enrichment is best-effort; if git fails, we still get the base failure message
    # Just check that we got SOME failure message about no-op rework
    assert any("no pushed commits" in f or "unpushed commit" in f for f in verdict.failures), (
        f"Expected no-op rework failure, got: {verdict.failures}"
    )


def test_get_unpushed_commit_info_excludes_base_update_merge_noise(tmp_path: Path) -> None:
    """A worktree whose only local-not-remote commits are base-update merges reports
    no unpushed content.

    Regression test: ``_get_unpushed_commit_info`` used to count with plain
    ``git rev-list --count origin/{branch}..HEAD``, which counts every commit a
    local ``git merge origin/main`` transitively drags in — including other PRs'
    already-landed squash-merge commits — as "unpushed". None of that is genuine
    unpushed work.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    (remote_repo / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=remote_repo, check=True, capture_output=True
    )

    # Clone the remote repo to create a local repo. A plain clone still reports
    # itself as a worktree via `git worktree list --porcelain`, so it doubles as
    # "the branch's worktree" for _get_unpushed_commit_info's lookup.
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

    # Create and push the agent branch (nothing unpushed yet)
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

    # Simulate two other PRs landing on main via squash-merge (each a single
    # non-merge commit, exactly like GitHub's squash-merge default)
    subprocess.run(["git", "checkout", "main"], cwd=remote_repo, check=True, capture_output=True)
    for i in range(2):
        (remote_repo / f"squash-{i}.txt").write_text(f"squashed PR #{i}")
        subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"squash-merged PR #{i}"],
            cwd=remote_repo,
            check=True,
            capture_output=True,
        )

    # In the local worktree, pull those in via a base-update merge — but don't push.
    # Old behavior (`rev-list --count origin/branch..HEAD`) would report 3
    # "unpushed" commits here (the 2 squashed commits + the merge commit itself),
    # all of them already on origin/main, none of them genuine unpushed work.
    subprocess.run(
        ["git", "checkout", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "fetch", "origin"], cwd=local_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "merge", "--no-ff", "origin/main"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    result = _get_unpushed_commit_info("agent/issue-123-test", local_repo, base_ref="main")
    assert result is None, f"expected no unpushed-commit message, got: {result!r}"


def test_get_unpushed_commit_info_reports_genuine_unpushed_commit(tmp_path: Path) -> None:
    """A worktree with a real unpushed non-merge commit still reports it, with the
    correct count, even when a base_ref is supplied for exclusion."""
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    (remote_repo / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=remote_repo, check=True, capture_output=True
    )

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

    # A genuine unpushed content commit
    (local_repo / "worker-change.txt").write_text("real unpushed work")
    subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "real unpushed work"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    result = _get_unpushed_commit_info("agent/issue-123-test", local_repo, base_ref="main")
    assert result is not None
    assert "1 unpushed commit(s)" in result
    assert "git push origin agent/issue-123-test" in result


def test_no_op_rework_merge_only_ignores_unpushed_base_merge_noise(tmp_path: Path) -> None:
    """Merge-only-advance no-op failures don't get a false "unpushed commit(s)"
    remediation when the worktree's only local-not-remote commits are further
    base-update merge noise.

    Regression test for the PR #680 false diagnostic: the janitor reported "26
    unpushed commit(s)" for a merge-only-advance failure when 25 of those were
    already-landed squash-merge commits pulled in transitively and the 26th was
    the merge commit itself — zero genuine unpushed content. The remediation text
    ("run 'git push origin <branch>'") could not have fixed anything, and risked
    inviting a no-op push that advances the PR head SHA without real rework.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    (remote_repo / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=remote_repo, check=True, capture_output=True
    )
    initial_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

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

    # First base-update: a merge-only advance that IS pushed. This becomes the
    # PR's recorded head (matches the merge-only-advance no-op path).
    subprocess.run(["git", "checkout", "main"], cwd=remote_repo, check=True, capture_output=True)
    (remote_repo / "main-change-1.txt").write_text("main branch change 1")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "main branch change 1"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "fetch", "origin"], cwd=local_repo, check=True, capture_output=True)
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
    subprocess.run(
        ["git", "push", "origin", "agent/issue-123-test"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )

    # Second base-update: another merge in the SAME local worktree, left unpushed.
    # Mirrors production: an operator/worker re-synced locally with main after the
    # last push, dragging in another already-landed commit, without pushing.
    (remote_repo / "main-change-2.txt").write_text("main branch change 2")
    subprocess.run(["git", "add", "."], cwd=remote_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "main branch change 2"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "fetch", "origin"], cwd=local_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "merge", "--no-ff", "origin/main"],
        cwd=local_repo,
        check=True,
        capture_output=True,
    )
    # Deliberately NOT pushed — local_repo's HEAD is now ahead of both the PR's
    # recorded headRefOid (merge_sha) and origin/agent/issue-123-test.

    # The PR still reports the earlier (pushed) merge-only SHA as its head
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

    # Should still FAIL as a merge-only no-op rework
    assert verdict.ok is False
    assert any("only by merge commits" in f for f in verdict.failures)
    # Should NOT falsely claim there are unpushed commits to push — the only
    # local-not-remote commits are base-update merge noise, not real content
    assert not any("unpushed commit(s)" in f for f in verdict.failures), (
        f"expected no false unpushed-commit claim, got: {verdict.failures}"
    )
    assert any(
        "check the branch worktree for unpushed work before re-reviewing" in f
        for f in verdict.failures
    )


def test_no_op_rework_warns_on_flag_like_reviewed_head_sha() -> None:
    """A flag-like reviewed_head_sha is caught as a warning, not a crash."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "--exec=foo",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert isinstance(verdict, JanitorVerdict)
    assert not verdict.is_no_op_rework
    assert any("_check_no_op_rework reviewed_head_sha" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_non_hex_reviewed_head_sha() -> None:
    """A non-hex reviewed_head_sha is caught as a warning, not a crash."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "not-a-sha!",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert isinstance(verdict, JanitorVerdict)
    assert not verdict.is_no_op_rework
    assert any("_check_no_op_rework reviewed_head_sha" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_current_head_sha() -> None:
    """A flag-like headRefOid is caught as a warning before reaching argv."""
    pr = _green_pr(headRefOid="--upload-pack=evil")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert isinstance(verdict, JanitorVerdict)
    assert not verdict.is_no_op_rework
    assert any("_check_no_op_rework current_head_sha" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_head_ref() -> None:
    """A flag-like headRefName is caught as a warning before ``git fetch origin``."""
    pr = _green_pr(headRefOid="def456", headRefName="--exec=foo")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert isinstance(verdict, JanitorVerdict)
    assert not verdict.is_no_op_rework
    assert any("_check_no_op_rework head_ref (merge-only)" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_base_ref() -> None:
    """A flag-like baseRefName is caught as a warning before ``git fetch origin``."""
    pr = _green_pr(headRefOid="def456", headRefName="agent/issue-1-fix", baseRefName="--exec=bar")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert isinstance(verdict, JanitorVerdict)
    assert not verdict.is_no_op_rework
    assert any("_check_no_op_rework base_ref (merge-only)" in w for w in verdict.warnings)


def test_no_op_rework_accepts_valid_sha_and_ref_values(tmp_path: Path) -> None:
    """Valid SHA and ref-name values pass validation and reach the normal path."""
    pr = _green_pr(headRefOid="def456", headRefName="agent/issue-1-fix", baseRefName="main")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    # Heads differ and no real git repo at tmp_path — the merge-only check will
    # fail gracefully (subprocess error → warning), but validation must pass.
    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        review_decision=pr_state,
    )

    assert isinstance(verdict, JanitorVerdict)
    assert not verdict.is_no_op_rework
    assert any("git fetch/rev-list failed" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_head_ref_patch_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag-like headRefName on the patch-id branch must not reach
    ``_get_unpushed_commit_info`` (issue #659).

    Pins the conflict-resolution hunk that merges 694's validation with PR
    #761's ``base_ref`` addition: a naive marker-only merge-conflict
    resolution dedents the ``_get_unpushed_commit_info`` call out of the
    ``except ValueError`` block, so it runs unconditionally with the raw,
    unvalidated ref even when validation just rejected it. This test fails
    against that regression.
    """
    from charlie_work import janitor as janitor_module

    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    current_patch_id = _calculate_patch_id(diff)
    pr = _green_pr(headRefName="--upload-pack=evil")
    pr_state = {
        "decision": "request_changes",
        "reviewed_patch_id": current_patch_id,
    }

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_get_unpushed_commit_info should not be called with an invalid ref")

    monkeypatch.setattr(janitor_module, "_get_unpushed_commit_info", _fail_if_called)

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        pr_diff=diff,
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("_check_no_op_rework head_ref (patch-id)" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_base_ref_patch_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag-like baseRefName on the patch-id branch must not reach
    ``_get_unpushed_commit_info`` (issue #659, same hunk as the head_ref
    variant above).
    """
    from charlie_work import janitor as janitor_module

    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    current_patch_id = _calculate_patch_id(diff)
    pr = _green_pr(headRefName="agent/issue-1-fix", baseRefName="--upload-pack=evil")
    pr_state = {
        "decision": "request_changes",
        "reviewed_patch_id": current_patch_id,
    }

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_get_unpushed_commit_info should not be called with an invalid ref")

    monkeypatch.setattr(janitor_module, "_get_unpushed_commit_info", _fail_if_called)

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        pr_diff=diff,
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("_check_no_op_rework base_ref (patch-id)" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_head_ref_sha_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag-like headRefName on the sha-match branch must not reach
    ``_get_unpushed_commit_info`` (issue #659). Pins the second
    conflict-resolution hunk against the same marker-strip dedent hazard as
    the patch-id variant above.
    """
    from charlie_work import janitor as janitor_module

    pr = _green_pr(headRefOid="abc123", headRefName="--upload-pack=evil")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_get_unpushed_commit_info should not be called with an invalid ref")

    monkeypatch.setattr(janitor_module, "_get_unpushed_commit_info", _fail_if_called)

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("_check_no_op_rework head_ref (sha-match)" in w for w in verdict.warnings)


def test_no_op_rework_warns_on_flag_like_base_ref_sha_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag-like baseRefName on the sha-match branch must not reach
    ``_get_unpushed_commit_info`` (issue #659, same hunk as the head_ref
    variant above).
    """
    from charlie_work import janitor as janitor_module

    pr = _green_pr(
        headRefOid="abc123", headRefName="agent/issue-1-fix", baseRefName="--upload-pack=evil"
    )
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_get_unpushed_commit_info should not be called with an invalid ref")

    monkeypatch.setattr(janitor_module, "_get_unpushed_commit_info", _fail_if_called)

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("_check_no_op_rework base_ref (sha-match)" in w for w in verdict.warnings)
