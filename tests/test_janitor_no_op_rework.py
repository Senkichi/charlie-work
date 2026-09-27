"""No-op-rework gate tests for the janitor.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1): the
``_check_no_op_rework`` verdict surface's core detection logic -- patch-id
comparison, SHA fallback, and the skip conditions that leave the gate
alone. The merge-commit git path, unpushed-commit enrichment, and
flag-like argv guards live in the ``test_janitor_no_op_rework_*`` siblings.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from _janitor_fixtures import _config, _green_checks, _green_pr

from charlie_work.janitor import _calculate_patch_id, run_janitor


def _body_sha256(body: str | None) -> str:
    """The ``reviewed_body_sha256`` wire contract (issue #1939): SHA-256 of the
    PR body with CRLF/CR line endings normalized to LF, ``None`` treated as
    the empty string. Computed here rather than imported so the tests pin the
    contract, not the implementation that happens to produce it."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def test_no_op_rework_body_change_satisfies_patch_id_gate(tmp_path: Path) -> None:
    """Issue #1939: an unchanged patch-id plus a changed PR body is real
    rework, not a no-op -- the requested fix lived in the PR description
    (swole #198 / PR #348), which produces no code diff by construction.

    MUTATION CHECK: MUST FAIL against the pre-fix implementation, which
    only ever compared patch-ids and had no body-change signal."""
    diff = """\
diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    reviewed_patch_id = _calculate_patch_id(diff)
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
        "reviewed_patch_id": reviewed_patch_id,
        "reviewed_body_sha256": _body_sha256(
            "Closes #123.\n\nTests: body as it was at verdict time."
        ),
    }
    pr = _green_pr(
        headRefOid="def456",  # head moved by a base-update merge
        body="Closes #123.\n\nTests: corrected description after rework.",
    )

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=diff,
        review_decision=pr_state,
    )

    assert verdict.ok is True, (
        f"Expected the changed-body escape to satisfy the gate, got {verdict.failures}"
    )
    assert not verdict.is_no_op_rework
    assert not any("unchanged since request_changes verdict" in f for f in verdict.failures)


def test_no_op_rework_body_change_satisfies_head_sha_fallback() -> None:
    """Issue #1939: a verdict recorded without a patch-id (pre-#222 legacy
    shape, or a diff fetch that failed) still recognizes a body-only rework
    -- the escape precedes the head-SHA fallback comparison too."""
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",  # head identical to current -- no patch-id
        "reviewed_body_sha256": _body_sha256("Closes #123.\n\nOld description."),
    }
    pr = _green_pr(
        headRefOid="abc123",
        body="Closes #123.\n\nTests: added unit tests -- corrected description.",
    )

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=Path.cwd(),
        review_decision=pr_state,
    )

    assert verdict.ok is True, (
        f"Expected the changed-body escape to satisfy the head-SHA fallback, "
        f"got {verdict.failures}"
    )
    assert not any("PR head unchanged" in f for f in verdict.failures)


def test_no_op_rework_unchanged_body_still_blocks(tmp_path: Path) -> None:
    """Issue #1939 regression side: an unchanged body means the unchanged
    patch-id verdict stands -- the escape must not weaken the genuine
    no-op protection (no code change AND no body change is still a no-op)."""
    diff = """\
diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    body = "Closes #123.\n\nTests: added unit tests for the search path."
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
        "reviewed_patch_id": _calculate_patch_id(diff),
        "reviewed_body_sha256": _body_sha256(body),
    }
    pr = _green_pr(headRefOid="def456", body=body)

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=diff,
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)


def test_no_op_rework_missing_body_baseline_still_blocks(tmp_path: Path) -> None:
    """Fail closed on a legacy verdict: a request_changes recorded before
    issue #1939 carries no ``reviewed_body_sha256`` baseline, so the gate
    cannot tell a body-only rework from a genuine no-op -- it keeps
    blocking exactly as before (the worker's next verdict picks up a
    baseline and unlocks the escape from then on)."""
    diff = """\
diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
        "reviewed_patch_id": _calculate_patch_id(diff),
        # No reviewed_body_sha256 -- verdict predates the field.
    }
    pr = _green_pr(headRefOid="def456", body="Closes #123.\n\nEdited description.")

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=diff,
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)


def test_no_op_rework_line_ending_flip_is_not_a_body_change(tmp_path: Path) -> None:
    """A pure CRLF/LF serialization flip must not open the escape: the hash
    normalizes line endings on both sides, so a transport artifact (e.g.
    ``gh pr edit --body-file`` vs. the API's echo) cannot satisfy the gate."""
    diff = """\
diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    reviewed_body = "Closes #123.\r\n\r\nTests: added unit tests for the search path."
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
        "reviewed_patch_id": _calculate_patch_id(diff),
        "reviewed_body_sha256": _body_sha256(reviewed_body),
    }
    # Same content, LF-only line endings.
    pr = _green_pr(
        headRefOid="def456",
        body="Closes #123.\n\nTests: added unit tests for the search path.",
    )

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=diff,
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)


def test_no_op_rework_body_key_absent_fails_closed(tmp_path: Path) -> None:
    """A PR payload without a ``body`` key gives the gate nothing to compare
    against the baseline -- treat it as unknown and keep blocking, never as
    an implicit change."""
    diff = """\
diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
        "reviewed_patch_id": _calculate_patch_id(diff),
        "reviewed_body_sha256": _body_sha256("some body"),
    }
    pr = _green_pr(headRefOid="def456")
    pr.pop("body", None)

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=diff,
        review_decision=pr_state,
    )

    assert verdict.ok is False
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)
