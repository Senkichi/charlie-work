"""No-op-rework gate tests for the janitor.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1): the
``_check_no_op_rework`` verdict surface's core detection logic -- patch-id
comparison, SHA fallback, and the skip conditions that leave the gate
alone. The merge-commit git path, unpushed-commit enrichment, and
flag-like argv guards live in the ``test_janitor_no_op_rework_*`` siblings.
"""

from __future__ import annotations

from pathlib import Path

from _janitor_fixtures import _config, _green_checks, _green_pr

from charlie_work.janitor import _calculate_patch_id, run_janitor


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
