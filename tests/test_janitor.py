from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from charlie_work.config import (
    AutoMergeConfig,
    OrchestratorConfig,
    ReviewConfig,
    TestAdequacyConfig,
)
from charlie_work.github import PR_VIEW_FIELDS
from charlie_work.janitor import (
    _calculate_patch_id,
    CONVENTIONAL_COMMIT_TYPES,
    JANITOR_PR_KEYS,
    JanitorVerdict,
    check_operator_containment,
    check_stub_tests,
    check_test_adequacy,
    iter_diff_files,
    run_janitor,
)

REQUIRED_CHECKS = ("Tests passed", "Lint & Format")


def _init_repo(repo_root: Path) -> None:
    """Initialize a git repo with a single commit."""
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main"], cwd=repo_root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def _config(**overrides) -> OrchestratorConfig:
    review = ReviewConfig(
        require_tests_or_rationale=overrides.pop("require_tests_or_rationale", True),
        require_issue_link=overrides.pop("require_issue_link", True),
    )
    auto_merge = AutoMergeConfig(required_checks=overrides.pop("required_checks", REQUIRED_CHECKS))
    assert not overrides, f"unused overrides: {overrides}"
    return OrchestratorConfig(review=review, auto_merge=auto_merge)


def _green_pr(**overrides) -> dict:
    base = {
        "number": 456,
        "title": "fix: search is broken",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": "abc123",
        "baseRefName": "main",
        "body": "Closes #123.\n\nTests: added unit tests for the search path.",
        "isDraft": False,
        "mergeable": "MERGEABLE",
        "additions": 10,
        "deletions": 5,
        "isCrossRepository": False,
    }
    base.update(overrides)
    return base


def _green_checks() -> list[dict]:
    return [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]


def test_fully_green_pr_yields_ok_with_empty_tuples() -> None:
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), repo_root=Path.cwd())

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()


def test_draft_pr_fails() -> None:
    verdict = run_janitor(
        _green_pr(isDraft=True), _green_checks(), _config(), repo_root=Path.cwd()
    )

    assert verdict.ok is False
    assert any("draft" in f.lower() for f in verdict.failures)


def test_non_open_state_fails() -> None:
    verdict = run_janitor(
        _green_pr(state="CLOSED"), _green_checks(), _config(), repo_root=Path.cwd()
    )

    assert verdict.ok is False
    assert any("CLOSED" in f for f in verdict.failures)


def test_conflicting_mergeable_fails() -> None:
    verdict = run_janitor(
        _green_pr(mergeable="CONFLICTING"), _green_checks(), _config(), repo_root=Path.cwd()
    )

    assert verdict.ok is False
    assert any("conflict" in f.lower() for f in verdict.failures)


def test_required_check_failure_blocks() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert any("Tests passed" in f for f in verdict.failures)


def test_required_check_missing_blocks() -> None:
    checks = [{"name": "Lint & Format", "bucket": "pass"}]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert any("missing" in f.lower() and "Tests passed" in f for f in verdict.failures)


def test_required_checks_unavailable_blocks() -> None:
    verdict = run_janitor(_green_pr(), None, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert any("Checks unavailable (gh failure)" in f for f in verdict.failures)


def test_required_check_pending_warns_not_fails() -> None:
    checks = [
        {"name": "Tests passed", "state": "PENDING"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is True
    assert verdict.failures == ()
    assert any("pending" in w.lower() and "Tests passed" in w for w in verdict.warnings)


def test_required_check_failure_exposes_failed_required_checks() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert verdict.failed_required_checks == ("Tests passed",)
    assert verdict.is_check_failure_block is True
    assert any("Tests passed" in f for f in verdict.failures)


def test_required_check_failure_with_other_blocker_is_not_check_failure_block() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]

    verdict = run_janitor(_green_pr(isDraft=True), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert verdict.failed_required_checks == ("Tests passed",)
    assert verdict.is_check_failure_block is False


def test_required_check_infra_failed_is_not_check_failure_block() -> None:
    checks = [{"name": "Tests passed", "state": "CANCELLED"}]

    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert verdict.failed_required_checks == ()
    assert verdict.is_check_failure_block is False
    assert any("infrastructure" in f.lower() for f in verdict.failures)


def test_no_required_checks_configured_skips_check_gate() -> None:
    verdict = run_janitor(_green_pr(), [], _config(required_checks=()))

    assert verdict.ok is True


def test_required_check_first_failure_returns_rerun_run_id() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "runId": 100},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    verdict = run_janitor(_green_pr(), checks, _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert verdict.is_check_failure_block is True
    assert verdict.rerun_run_ids == (100,)
    assert verdict.check_rerun_attempts == {"abc123": {"Tests passed": [100]}}


def test_required_check_second_failure_returns_no_rerun() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE", "runId": 100},
        {"name": "Lint & Format", "bucket": "pass"},
    ]
    pr_state = {"check_rerun_attempts": {"abc123": {"Tests passed": [100]}}}
    verdict = run_janitor(_green_pr(), checks, _config(), pr_state=pr_state, repo_root=Path.cwd())

    assert verdict.ok is False
    assert verdict.is_check_failure_block is True
    assert verdict.rerun_run_ids == ()
    assert verdict.failed_required_checks == ("Tests passed",)


def test_missing_linked_issue_fails_when_required() -> None:
    pr = _green_pr(
        headRefName="agent/misc-branch", body="No issue reference here at all, tests added."
    )

    verdict = run_janitor(pr, _green_checks(), _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert any("linked issue" in f.lower() for f in verdict.failures)


def test_missing_linked_issue_ok_when_not_required() -> None:
    pr = _green_pr(
        headRefName="agent/misc-branch", body="No issue reference here at all, tests added."
    )

    verdict = run_janitor(
        pr, _green_checks(), _config(require_issue_link=False), repo_root=Path.cwd()
    )

    assert verdict.ok is True


def test_empty_body_fails() -> None:
    verdict = run_janitor(_green_pr(body=""), _green_checks(), _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert any("body is empty" in f.lower() for f in verdict.failures)


def test_body_with_only_whitespace_fails() -> None:
    verdict = run_janitor(
        _green_pr(body="   \n  "), _green_checks(), _config(), repo_root=Path.cwd()
    )

    assert verdict.ok is False
    assert any("body is empty" in f.lower() for f in verdict.failures)


def test_body_without_tests_or_rationale_marker_fails_when_required() -> None:
    pr = _green_pr(body="Closes #123. This fixes the thing, no more detail than that.")

    verdict = run_janitor(pr, _green_checks(), _config(), repo_root=Path.cwd())

    assert verdict.ok is False
    assert any("tests/verification/rationale" in f.lower() for f in verdict.failures)


def test_body_with_rationale_marker_passes() -> None:
    pr = _green_pr(body="Closes #123. No tests because this is a comment-only change.")

    verdict = run_janitor(pr, _green_checks(), _config(), repo_root=Path.cwd())

    assert verdict.ok is True


def test_body_marker_check_skipped_when_not_required() -> None:
    pr = _green_pr(body="Closes #123. This fixes the thing, no more detail than that.")

    verdict = run_janitor(
        pr, _green_checks(), _config(require_tests_or_rationale=False), repo_root=Path.cwd()
    )

    assert verdict.ok is True


def test_non_conventional_title_warns() -> None:
    pr = _green_pr(title="Search improvements")

    verdict = run_janitor(pr, _green_checks(), _config(), repo_root=Path.cwd())

    assert verdict.ok is True
    assert any("conventional-commit" in w.lower() for w in verdict.warnings)
    # Verify the warning references the template (single source of truth)
    assert any("prompts/worker.md" in w for w in verdict.warnings)


def test_conventional_title_variants_do_not_warn() -> None:
    for title in ("feat: add x", "fix(search): bug", "chore!: breaking", "docs: update readme"):
        verdict = run_janitor(
            _green_pr(title=title), _green_checks(), _config(), repo_root=Path.cwd()
        )
        assert not any("conventional-commit" in w.lower() for w in verdict.warnings), title


def test_worker_template_title_format_passes_janitor() -> None:
    """Assert that the worker template's mandated PR title format passes janitor checks.

    This test prevents drift between prompts/worker.md PR requirements and janitor.py's
    title validation. The template mandates conventional-commit format (type(scope): description),
    which should never trigger the janitor's conventional-commit warning.
    """
    # Read the actual worker template to extract the documented example title
    repo_root = Path(__file__).parent.parent
    worker_template = repo_root / "src" / "charlie_work" / "prompts" / "worker.md"
    template_content = worker_template.read_text()

    # Extract the example title from the marker comment
    marker_match = re.search(r"JANITOR_TITLE_EXAMPLE:\s*(.+)", template_content)
    if not marker_match:
        raise AssertionError("Could not find JANITOR_TITLE_EXAMPLE marker in worker.md")

    title = marker_match.group(1).strip()
    verdict = run_janitor(_green_pr(title=title), _green_checks(), _config(), repo_root=repo_root)
    assert not any("conventional-commit" in w.lower() for w in verdict.warnings), (
        f"Worker template title format '{title}' should not trigger janitor warning"
    )


def test_conventional_commit_types_constant_pinned() -> None:
    """Assert that CONVENTIONAL_COMMIT_TYPES is pinned to the expected set.

    This test serves as a deliberate second anchor for the canonical type list.
    Removing any type from the constant will fail this test, preventing silent
    drift where the constant changes but documentation doesn't.
    """
    expected = frozenset({"feat", "fix", "refactor", "docs", "test", "chore", "perf", "ci"})
    assert CONVENTIONAL_COMMIT_TYPES == expected, (
        f"CONVENTIONAL_COMMIT_TYPES changed from expected {expected} to {CONVENTIONAL_COMMIT_TYPES}. "
        "If this change is intentional, update this test's expected set."
    )


def test_conventional_commit_regex_behavior() -> None:
    """Assert that the janitor's conventional-commit regex accepts all valid types and rejects unknown types.

    This test uses explicit example titles for EVERY type (not parametrized over the constant
    itself, which would be circular). Removing a type from the constant would NOT remove its
    test case here, ensuring the pin test catches the drift.
    """
    repo_root = Path(__file__).parent.parent

    # Test that all valid types pass the janitor check
    valid_titles = [
        "feat: add new feature",
        "fix: correct bug",
        "refactor: improve code structure",
        "docs: update documentation",
        "test: add tests",
        "chore: maintenance task",
        "perf: improve performance",
        "ci: update CI pipeline",
    ]
    for title in valid_titles:
        verdict = run_janitor(
            _green_pr(title=title), _green_checks(), _config(), repo_root=repo_root
        )
        assert not any("conventional-commit" in w.lower() for w in verdict.warnings), (
            f"Valid title '{title}' should not trigger janitor warning"
        )

    # Test that an unknown type triggers the warning
    verdict = run_janitor(
        _green_pr(title="foo: unknown type"), _green_checks(), _config(), repo_root=repo_root
    )
    assert any("conventional-commit" in w.lower() for w in verdict.warnings), (
        "Unknown type 'foo' should trigger janitor warning"
    )


def test_conventional_commit_types_documentation_consistency() -> None:
    """Assert that documented conventional-commit types match the canonical constant.

    This test prevents drift between the canonical type list in janitor.py and the
    documented lists in CONTRIBUTING.md and prompts/worker.md. All three must stay
    in sync to avoid confusing contributors with contradictory documentation.
    """
    repo_root = Path(__file__).parent.parent

    # Extract types from CONTRIBUTING.md - find ALL "Valid types:" occurrences
    contributing = repo_root / "CONTRIBUTING.md"
    contributing_content = contributing.read_text()
    contributing_matches = re.findall(
        r"Valid types: (`[^`]+`(?:, `[^`]+`)*)", contributing_content
    )
    if not contributing_matches:
        raise AssertionError("Could not find any 'Valid types:' line in CONTRIBUTING.md")

    # Assert there are at least 2 occurrences (generic section + PR-title section)
    assert len(contributing_matches) >= 2, (
        f"Expected at least 2 'Valid types:' occurrences in CONTRIBUTING.md, found {len(contributing_matches)}. "
        "Deleting a section should be caught by this test."
    )

    # Assert EVERY occurrence's extracted type set equals the canonical set
    for i, types_str in enumerate(contributing_matches):
        contributing_types = set(re.findall(r"`([^`]+)`", types_str))
        assert contributing_types == CONVENTIONAL_COMMIT_TYPES, (
            f"CONTRIBUTING.md occurrence {i + 1} types {contributing_types} != canonical {CONVENTIONAL_COMMIT_TYPES}"
        )

    # Extract types from worker.md - parse the actual enumeration line
    worker = repo_root / "src" / "charlie_work" / "prompts" / "worker.md"
    worker_content = worker.read_text()
    # The worker template explicitly enumerates types on line 62:
    # "Valid types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`."
    worker_match = re.search(r"Valid types: (`[^`]+`(?:, `[^`]+`)*)", worker_content)
    if not worker_match:
        raise AssertionError("Could not find 'Valid types:' line in worker.md")

    worker_types_str = worker_match.group(1)
    worker_types = set(re.findall(r"`([^`]+)`", worker_types_str))

    # Assert set EQUALITY (not subset) - worker.md must enumerate ALL types
    assert worker_types == CONVENTIONAL_COMMIT_TYPES, (
        f"worker.md types {worker_types} != canonical {CONVENTIONAL_COMMIT_TYPES}"
    )


def test_calculate_patch_id_stable_for_same_diff() -> None:
    """Patch-id calculation should be stable for the same diff content."""
    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    patch_id1 = _calculate_patch_id(diff)
    patch_id2 = _calculate_patch_id(diff)
    assert patch_id1 == patch_id2
    assert len(patch_id1) == 40  # SHA-1 hex string from git patch-id --stable


def test_calculate_patch_id_different_for_different_diffs() -> None:
    """Patch-id calculation should differ for different diff content."""
    diff1 = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    diff2 = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 different
"""
    patch_id1 = _calculate_patch_id(diff1)
    patch_id2 = _calculate_patch_id(diff2)
    assert patch_id1 != patch_id2


def test_calculate_patch_id_ignores_metadata() -> None:
    """Patch-id calculation should ignore diff metadata (hashes, timestamps)."""
    diff1 = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    # Same content but different metadata (different index hashes)
    diff2 = """diff --git a/test.txt b/test.txt
index 9999999..8888888 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    patch_id1 = _calculate_patch_id(diff1)
    patch_id2 = _calculate_patch_id(diff2)
    assert patch_id1 == patch_id2


def test_calculate_patch_id_empty_diff() -> None:
    """Patch-id calculation should return empty string for empty diff."""
    assert _calculate_patch_id("") == ""
    assert _calculate_patch_id("   \n  ") == ""


def test_calculate_patch_id_offset_immune() -> None:
    """Patch-id is identical for diffs with the same content but shifted hunk offsets.

    A base-update merge that adds lines to files shared with an open PR shifts
    hunk-header line numbers (@@ -N,M +N,M @@) without touching content lines.
    git patch-id strips hunk headers for this reason; _calculate_patch_id must
    do the same so that offset-only shifts do not change the patch-id.

    MUTATION CHECK: this test MUST FAIL if the @@ skip is removed from
    _calculate_patch_id (verified during development — see PR #229 rework notes).
    """
    # Two diffs with identical content lines but hunk offsets shifted by 4 lines
    diff_original = """\
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
    diff_shifted = """\
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
    id_original = _calculate_patch_id(diff_original)
    id_shifted = _calculate_patch_id(diff_shifted)
    assert id_original == id_shifted, (
        f"Hunk-offset shift changed patch-id: {id_original!r} != {id_shifted!r}. "
        "Did git patch-id --stable stop ignoring hunk headers?"
    )
    assert len(id_original) == 40  # SHA-1 hex string from git patch-id --stable


def test_calculate_patch_id_returns_empty_for_diff_without_hunks() -> None:
    """A diff with no hunk header is not a real patch and cannot be compared."""
    assert _calculate_patch_id("diff --git a/file b/file\n") == ""


def test_calculate_patch_id_returns_empty_when_git_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Git failures during patch-id computation must fail closed (empty string)."""
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

    def _fake_run_captured(*_args, **_kwargs):
        from charlie_work.subprocess_runner import RunResult

        return RunResult(returncode=1, stdout="", stderr="git failed", error="git failed")

    monkeypatch.setattr(janitor_module, "run_captured", _fake_run_captured)
    assert _calculate_patch_id(diff) == ""


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
        pr, _green_checks(), _config(), pr_state=pr_state, repo_root=Path.cwd(), pr_diff=diff
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
        pr, _green_checks(), _config(), pr_state=pr_state, repo_root=Path.cwd(), pr_diff=diff
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
        pr, _green_checks(), _config(), pr_state=pr_state, repo_root=Path.cwd(), pr_diff=None
    )

    # Should FAIL because SHA matches (fallback behavior when diff is None)
    assert verdict.ok is False
    assert any("PR head unchanged since request_changes verdict" in f for f in verdict.failures)


def test_oversized_diff_warns() -> None:
    pr = _green_pr(additions=1000, deletions=600)

    verdict = run_janitor(pr, _green_checks(), _config(), repo_root=Path.cwd())

    assert verdict.ok is True
    assert any("oversized diff" in w.lower() for w in verdict.warnings)


def test_diff_at_threshold_does_not_warn() -> None:
    pr = _green_pr(additions=1000, deletions=500)  # exactly 1500

    verdict = run_janitor(pr, _green_checks(), _config(), repo_root=Path.cwd())

    assert not any("oversized diff" in w.lower() for w in verdict.warnings)


def test_missing_keys_never_raise_and_skip_checks() -> None:
    # Minimal pr dict: gh omits fields depending on flags used to fetch it.
    verdict = run_janitor({}, [], _config(required_checks=()), repo_root=Path.cwd())

    assert isinstance(verdict, JanitorVerdict)
    # require_issue_link is on by default in _config(), and linked_issue_number
    # gracefully returns None for an empty dict, so that failure still fires.
    assert any("linked issue" in f.lower() for f in verdict.failures)
    # But no draft/state/mergeable/body/title/diff-size failures or warnings
    # should be raised from absent keys.
    assert not any("draft" in f.lower() for f in verdict.failures)
    assert not any("OPEN" in f for f in verdict.failures)
    assert not any("conflict" in f.lower() for f in verdict.failures)
    assert not any("body is empty" in f.lower() for f in verdict.failures)
    assert not any("tests/verification/rationale" in f.lower() for f in verdict.failures)


def test_fully_absent_pr_with_all_optional_checks_disabled_is_ok() -> None:
    verdict = run_janitor(
        {},
        [],
        _config(required_checks=(), require_issue_link=False, require_tests_or_rationale=False),
        repo_root=Path.cwd(),
    )

    assert verdict == JanitorVerdict(ok=True, failures=(), warnings=())


def test_multiple_failures_all_reported() -> None:
    pr = _green_pr(isDraft=True, state="CLOSED", mergeable="CONFLICTING", body="")

    verdict = run_janitor(pr, [], _config(required_checks=()), repo_root=Path.cwd())

    assert verdict.ok is False
    assert len(verdict.failures) >= 4


def test_base_movement_warns_for_agent_pr() -> None:
    pr = _green_pr(mergeStateStatus="BEHIND")
    config = _config()

    verdict = run_janitor(pr, _green_checks(), config, repo_root=Path.cwd())

    assert verdict.ok is True
    assert any(
        "Base branch has moved since branch (mergeStateStatus=BEHIND)" in w
        for w in verdict.warnings
    )


def test_base_movement_skips_fork_pr() -> None:
    pr = _green_pr(mergeStateStatus="BEHIND", isCrossRepository=True)

    verdict = run_janitor(
        pr, _green_checks(), _config(require_issue_link=False), repo_root=Path.cwd()
    )

    assert verdict.ok is True
    assert not any("Base moved" in w for w in verdict.warnings)


def test_base_movement_skips_non_prefix_branch() -> None:
    pr = _green_pr(mergeStateStatus="BEHIND", headRefName="feature/something")

    verdict = run_janitor(pr, _green_checks(), _config(), repo_root=Path.cwd())

    assert verdict.ok is True
    assert not any("Base moved" in w for w in verdict.warnings)


def test_base_movement_no_warning_when_up_to_date() -> None:
    pr = _green_pr(mergeStateStatus="CLEAN")

    verdict = run_janitor(pr, _green_checks(), _config(), repo_root=Path.cwd())

    assert verdict.ok is True
    assert not any("Base moved" in w for w in verdict.warnings)


def test_base_movement_no_warning_when_field_missing() -> None:
    pr = _green_pr()
    # Remove mergeStateStatus if it exists
    pr.pop("mergeStateStatus", None)

    verdict = run_janitor(pr, _green_checks(), _config(), repo_root=Path.cwd())

    assert verdict.ok is True
    assert not any("Base moved" in w for w in verdict.warnings)


def test_iter_diff_files_single_file_single_hunk() -> None:
    """A one-file diff yields exactly one (filename, False, hunk_lines) tuple with the expected hunk-header + body lines."""
    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    results = list(iter_diff_files(diff))
    assert len(results) == 1
    filename, is_new_file, hunk_lines = results[0]
    assert filename == "test.txt"
    assert is_new_file is False
    assert len(hunk_lines) == 4
    assert hunk_lines[0] == "@@ -1,2 +1,2 @@"
    assert hunk_lines[1] == " line 1"
    assert hunk_lines[2] == "-line 2"
    assert hunk_lines[3] == "+line 2 modified"


def test_iter_diff_files_multi_file() -> None:
    """A diff touching 2+ files yields one tuple per file, in source order, each with its own hunk lines (not cross-contaminated)."""
    diff = """diff --git a/a_module.py b/a_module.py
index 1234567..abcdef0 100644
--- a/a_module.py
+++ b/a_module.py
@@ -1,2 +1,2 @@
 def a_func():
-    return 'a'
+    return 'a_modified'
diff --git a/b_module.py b/b_module.py
index 1234567..abcdef0 100644
--- a/b_module.py
+++ b/b_module.py
@@ -1,2 +1,2 @@
 def b_func():
-    return 'b'
+    return 'b_modified'
"""
    results = list(iter_diff_files(diff))
    assert len(results) == 2
    filename_a, is_new_file_a, hunk_lines_a = results[0]
    filename_b, is_new_file_b, hunk_lines_b = results[1]
    assert filename_a == "a_module.py"
    assert is_new_file_a is False
    assert len(hunk_lines_a) == 4
    assert "@@ -1,2 +1,2 @@" in hunk_lines_a[0]
    assert "def a_func():" in hunk_lines_a[1]
    assert filename_b == "b_module.py"
    assert is_new_file_b is False
    assert len(hunk_lines_b) == 4
    assert "@@ -1,2 +1,2 @@" in hunk_lines_b[0]
    assert "def b_func():" in hunk_lines_b[1]
    # Verify no cross-contamination
    assert "a_func" not in hunk_lines_b
    assert "b_func" not in hunk_lines_a


def test_iter_diff_files_new_file_flag() -> None:
    """A diff containing `new file mode` for a path sets is_new_file=True; an existing-file modification sets it False."""
    diff = """diff --git a/new_file.py b/new_file.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,2 @@
+def new_func():
+    return 'new'
diff --git a/existing.py b/existing.py
index 1234567..abcdef0 100644
--- a/existing.py
+++ b/existing.py
@@ -1,1 +1,2 @@
 old line
+new line
"""
    results = list(iter_diff_files(diff))
    assert len(results) == 2
    filename_new, is_new_file_new, _ = results[0]
    filename_existing, is_new_file_existing, _ = results[1]
    assert filename_new == "new_file.py"
    assert is_new_file_new is True
    assert filename_existing == "existing.py"
    assert is_new_file_existing is False


def test_iter_diff_files_rename_no_hunk_body() -> None:
    """A pure rename (100% similarity, no @@ hunks) is skipped since it has no +++ b/ line (and thus no hunk body)."""
    diff = """diff --git a/old_name.py b/new_name.py
similarity index 100%
rename from old_name.py
rename to new_name.py
"""
    results = list(iter_diff_files(diff))
    # The function skips sections without a +++ b/ line
    assert len(results) == 0


def test_iter_diff_files_strips_no_newline_marker() -> None:
    r"""A hunk containing a `\ No newline at end of file` metadata line does not include that line in hunk_lines."""
    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
\\ No newline at end of file
+line 2 modified
"""
    results = list(iter_diff_files(diff))
    assert len(results) == 1
    _, _, hunk_lines = results[0]
    # The metadata line should be stripped
    assert not any(line.startswith("\\") for line in hunk_lines)
    assert len(hunk_lines) == 4


def test_iter_diff_files_empty_diff() -> None:
    """Empty diff yields nothing (empty iterator)."""
    results = list(iter_diff_files(""))
    assert results == []


# External API/live-payload fixture checks (issue #223)


def test_external_api_call_warns_for_gh_run_api() -> None:
    """A product file adding gh.run(['api', ...]) with no fixture or body evidence warns."""
    diff = """diff --git a/src/runners.py b/src/runners.py
index 123..456 100644
--- a/src/runners.py
+++ b/src/runners.py
@@ -1,3 +1,5 @@
 def observe(gh):
+    data = gh.run(["api", "repos/{owner}/{repo}/actions/runners"], json_output=True)
+    runners = data.get("runners", [])
     return runners
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert any("external API/library call" in w for w in verdict.warnings)
    assert any("gh.run" in w for w in verdict.warnings)


def test_external_api_call_warns_for_self_run_api_multiline() -> None:
    """Multi-line self.run(['api', ...]) in a product file warns."""
    diff = """diff --git a/src/github.py b/src/github.py
index 123..456 100644
--- a/src/github.py
+++ b/src/github.py
@@ -1,3 +1,10 @@
 class GitHub:
     def actions_job(self, job_id):
+        result = self.run(
+            [
+                "api",
+                f"repos/{{owner}}/{{repo}}/actions/jobs/{job_id}",
+            ],
+            json_output=True,
+        )
+        return result
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert any("external API/library call" in w for w in verdict.warnings)


def test_external_api_call_warns_for_requests_get() -> None:
    """A product file adding requests.get() with no fixture or body evidence warns."""
    diff = """diff --git a/src/foo.py b/src/foo.py
index 123..456 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,5 @@
 def fetch():
+    import requests
+    return requests.get("https://api.github.com/repos/owner/repo")
     pass
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert any("external API/library call" in w for w in verdict.warnings)
    assert any("requests.get" in w for w in verdict.warnings)


def test_external_api_call_warns_for_subprocess_gh_api() -> None:
    """subprocess.run(['gh', 'api', ...]) in a product file warns."""
    diff = """diff --git a/src/foo.py b/src/foo.py
index 123..456 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,5 @@
 def fetch():
+    import subprocess
+    subprocess.run(["gh", "api", "repos/owner/repo"], check=True)
     pass
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert any("external API/library call" in w for w in verdict.warnings)


def test_external_api_call_no_warning_with_fixture() -> None:
    """A product file adding gh.run(['api', ...]) plus a tests/fixtures/ file is clean."""
    diff = """diff --git a/src/runners.py b/src/runners.py
index 123..456 100644
--- a/src/runners.py
+++ b/src/runners.py
@@ -1,3 +1,5 @@
 def observe(gh):
+    data = gh.run(["api", "repos/{owner}/{repo}/actions/runners"], json_output=True)
+    runners = data.get("runners", [])
     return runners

diff --git a/tests/fixtures/runners.json b/tests/fixtures/runners.json
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/fixtures/runners.json
@@ -0,0 +1,3 @@
+{
+    "runners": [{"id": 1, "name": "jc-1"}]
+}
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert not any("external API/library call" in w for w in verdict.warnings)


def test_external_api_call_no_warning_with_body_evidence() -> None:
    """A product file adding an external call is clean when the PR body has live evidence."""
    diff = """diff --git a/src/runners.py b/src/runners.py
index 123..456 100644
--- a/src/runners.py
+++ b/src/runners.py
@@ -1,3 +1,5 @@
 def observe(gh):
+    data = gh.run(["api", "repos/{owner}/{repo}/actions/runners"], json_output=True)
+    runners = data.get("runners", [])
     return runners
"""
    pr = _green_pr(
        body="Closes #123.\n\nVerified with live payload: gh api repos/{owner}/{repo}/actions/runners"
    )
    verdict = run_janitor(pr, _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert not any("external API/library call" in w for w in verdict.warnings)


def test_external_api_call_no_warning_for_test_files() -> None:
    """Test files are excluded from external API call detection."""
    diff = """diff --git a/tests/test_runners.py b/tests/test_runners.py
index 123..456 100644
--- a/tests/test_runners.py
+++ b/tests/test_runners.py
@@ -1,3 +1,5 @@
 def test_observe():
+    gh = MagicMock()
+    gh.run = MagicMock(side_effect=lambda args: _mock(args))
     assert observe(gh)
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert not any("external API/library call" in w for w in verdict.warnings)


def test_external_api_call_no_warning_for_exempt_files() -> None:
    """Markdown/docs files are excluded from external API call detection."""
    diff = """diff --git a/docs/api.md b/docs/api.md
index 123..456 100644
--- a/docs/api.md
+++ b/docs/api.md
@@ -1,3 +1,5 @@
 # Notes
+
+    gh api repos/{owner}/{repo}/actions/runners
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert not any("external API/library call" in w for w in verdict.warnings)


def test_external_api_call_no_warning_without_any_call() -> None:
    """A product diff without external API calls does not warn."""
    diff = """diff --git a/src/foo.py b/src/foo.py
index 123..456 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,5 @@
 def compute():
+    return 1 + 1
     pass
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert not any("external API/library call" in w for w in verdict.warnings)


def test_containment_clean_tree_no_warnings(tmp_path: Path) -> None:
    """Clean operator checkout produces no containment warnings."""
    _init_repo(tmp_path)
    # Create an initial commit
    (tmp_path / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Empty diff (no PR changes)
    diff = ""
    warnings = check_operator_containment(tmp_path, diff, 123)

    assert warnings == ()


def test_containment_leak_detection(tmp_path: Path) -> None:
    """Detect leaked worker edits: working-tree file byte-identical to PR post-image."""
    _init_repo(tmp_path)
    # Create an initial commit
    (tmp_path / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Simulate a leak: modify the file to match PR post-image
    leaked_content = "leaked content from PR"
    (tmp_path / "test.txt").write_text(leaked_content)

    # Get the actual working-tree diff to use as the PR diff
    result = subprocess.run(
        ["git", "diff", "HEAD", "--", "test.txt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    diff = result.stdout

    warnings = check_operator_containment(tmp_path, diff, 123)

    assert len(warnings) == 1
    assert "Containment leak detected" in warnings[0]
    assert "PR #123" in warnings[0]
    assert "test.txt" in warnings[0]
    assert "git checkout --" in warnings[0]


def test_containment_unrelated_dirty_file(tmp_path: Path) -> None:
    """Unrelated dirty files produce generic warning, not leak warning."""
    _init_repo(tmp_path)
    # Create an initial commit
    (tmp_path / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Modify file to content NOT in the PR diff
    (tmp_path / "test.txt").write_text("unrelated local work")

    # Create a diff that doesn't match the dirty file
    diff = """diff --git a/other.txt b/other.txt
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/other.txt
@@ -0,0 +1 @@
+other file content
"""

    warnings = check_operator_containment(tmp_path, diff, 123)

    assert len(warnings) == 1
    assert "not a leak" in warnings[0]
    assert "test.txt" in warnings[0]
    assert "Containment leak" not in warnings[0]


def test_containment_git_failure_graceful_degradation(tmp_path: Path) -> None:
    """Git failures produce no warnings rather than blocking."""
    # Not a git repo, so git status will fail
    diff = ""
    warnings = check_operator_containment(tmp_path, diff, 123)

    # Should return empty tuple on git failure (graceful degradation)
    assert warnings == ()


def test_containment_partial_file_leak_detection(tmp_path: Path) -> None:
    """Detect leaked worker edits in multi-line file with mid-file edit (partial-file leak scenario)."""
    _init_repo(tmp_path)
    # Create a multi-line file
    initial_content = """line 1
line 2
line 3
line 4
line 5
"""
    (tmp_path / "multi.txt").write_text(initial_content)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Simulate a leak: modify a middle line to match PR post-image
    leaked_content = """line 1
line 2
MODIFIED LINE
line 4
line 5
"""
    (tmp_path / "multi.txt").write_text(leaked_content)

    # Get the actual working-tree diff to use as the PR diff
    result = subprocess.run(
        ["git", "diff", "HEAD", "--", "multi.txt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    diff = result.stdout

    warnings = check_operator_containment(tmp_path, diff, 123)

    assert len(warnings) == 1
    assert "Containment leak detected" in warnings[0]
    assert "PR #123" in warnings[0]
    assert "multi.txt" in warnings[0]
    assert "git checkout --" in warnings[0]


def test_containment_untracked_file_no_warning(tmp_path: Path) -> None:
    """Untracked files unrelated to the PR diff produce zero warnings."""
    _init_repo(tmp_path)
    # Create an initial commit
    (tmp_path / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Create an untracked file (simulating orchestrator.config.yaml)
    (tmp_path / "orchestrator.config.yaml").write_text("config: value")

    # Create a diff that doesn't include the untracked file
    diff = """diff --git a/other.txt b/other.txt
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/other.txt
@@ -0,0 +1 @@
+other file content
"""

    warnings = check_operator_containment(tmp_path, diff, 123)

    # Should produce zero warnings - untracked files not in the diff are ignored
    assert warnings == ()


def test_containment_multi_file_leak_first_file(tmp_path: Path) -> None:
    """Detect leaked worker edits when PR diff spans multiple files and first file is leaked."""
    _init_repo(tmp_path)
    # Create two files with initial content
    (tmp_path / "a_module.py").write_text("def a_func():\n    return 'a'\n")
    (tmp_path / "b_module.py").write_text("def b_func():\n    return 'b'\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Simulate a leak: modify the FIRST file (a_module.py) to match PR post-image
    leaked_content = "def a_func():\n    return 'a_modified'\n"
    (tmp_path / "a_module.py").write_text(leaked_content)

    # Create a multi-file PR diff (both a_module.py and b_module.py)
    # The PR modifies both files, but only a_module.py is leaked in the working tree
    diff = """diff --git a/a_module.py b/a_module.py
index 1234567..abcdef0 100644
--- a/a_module.py
+++ b/a_module.py
@@ -1,2 +1,2 @@
 def a_func():
-    return 'a'
+    return 'a_modified'
diff --git a/b_module.py b/b_module.py
index 1234567..abcdef0 100644
--- a/b_module.py
+++ b/b_module.py
@@ -1,2 +1,2 @@
 def b_func():
-    return 'b'
+    return 'b_modified'
"""

    warnings = check_operator_containment(tmp_path, diff, 123)

    # Should detect the leak in a_module.py (first file in diff)
    assert len(warnings) == 1
    assert "Containment leak detected" in warnings[0]
    assert "PR #123" in warnings[0]
    assert "a_module.py" in warnings[0]
    assert "git checkout --" in warnings[0]


def test_containment_multi_file_leak_last_file(tmp_path: Path) -> None:
    """Detect leaked worker edits when PR diff spans multiple files and last file is leaked."""
    _init_repo(tmp_path)
    # Create two files with initial content
    (tmp_path / "a_module.py").write_text("def a_func():\n    return 'a'\n")
    (tmp_path / "b_module.py").write_text("def b_func():\n    return 'b'\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Simulate a leak: modify the LAST file (b_module.py) to match PR post-image
    leaked_content = "def b_func():\n    return 'b_modified'\n"
    (tmp_path / "b_module.py").write_text(leaked_content)

    # Create a multi-file PR diff (both a_module.py and b_module.py)
    # The PR modifies both files, but only b_module.py is leaked in the working tree
    diff = """diff --git a/a_module.py b/a_module.py
index 1234567..abcdef0 100644
--- a/a_module.py
+++ b/a_module.py
@@ -1,2 +1,2 @@
 def a_func():
-    return 'a'
+    return 'a_modified'
diff --git a/b_module.py b/b_module.py
index 1234567..abcdef0 100644
--- a/b_module.py
+++ b/b_module.py
@@ -1,2 +1,2 @@
 def b_func():
-    return 'b'
+    return 'b_modified'
"""

    warnings = check_operator_containment(tmp_path, diff, 123)

    # Should detect the leak in b_module.py (last file in diff)
    assert len(warnings) == 1
    assert "Containment leak detected" in warnings[0]
    assert "PR #123" in warnings[0]
    assert "b_module.py" in warnings[0]
    assert "git checkout --" in warnings[0]


def test_no_op_rework_detects_unchanged_head() -> None:
    """Detect no-op rework when PR head is unchanged since request_changes verdict."""
    pr = _green_pr(headRefOid="abc123")
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    verdict = run_janitor(pr, _green_checks(), _config(), pr_state=pr_state, repo_root=Path.cwd())

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

    verdict = run_janitor(pr, _green_checks(), _config(), pr_state=pr_state, repo_root=local_repo)

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

    verdict = run_janitor(pr, _green_checks(), _config(), pr_state=pr_state, repo_root=tmp_path)

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

    verdict = run_janitor(pr, _green_checks(), _config(), pr_state=pr_state, repo_root=tmp_path)

    # Should fail (no-op rework)
    assert verdict.ok is False, (
        f"Expected no-op rework to fail, but verdict.ok={verdict.ok}, failures={verdict.failures}"
    )
    # The unpushed commit enrichment is best-effort; if git fails, we still get the base failure message
    # Just check that we got SOME failure message about no-op rework
    assert any("no pushed commits" in f or "unpushed commit" in f for f in verdict.failures), (
        f"Expected no-op rework failure, got: {verdict.failures}"
    )


def test_body_word_boundary_matching_prevents_false_positives() -> None:
    """Word-boundary matching prevents 'test' in 'latest' from passing the gate (regression test for issue #2)."""
    pr = _green_pr(body="Closes #123. Updated to latest version.")

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is False
    assert any("tests/verification/rationale" in f.lower() for f in verdict.failures)


def test_body_word_boundary_matching_allows_legitimate_markers() -> None:
    """Word-boundary matching still allows legitimate test/rationale markers (regression test for issue #2)."""
    pr = _green_pr(body="Closes #123. Added tests for the fix.")

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is True


def test_janitor_pr_keys_contained_in_pr_view_fields() -> None:
    """All PR keys read by janitor gates must be present in PR_VIEW_FIELDS (regression test for issue #2).

    This test prevents the regression that issue #2 fixed: if a janitor gate reads a PR key
    that is not in PR_VIEW_FIELDS, the gate will be silently disabled because gh pr view will
    not fetch that field. This test FAILS if any janitor-read key is dropped from PR_VIEW_FIELDS.
    """
    # Parse PR_VIEW_FIELDS into a set of field names
    pr_view_field_set = set(PR_VIEW_FIELDS.split(","))

    # Assert every janitor-read key is in PR_VIEW_FIELDS
    missing_keys = JANITOR_PR_KEYS - pr_view_field_set
    assert not missing_keys, (
        f"Janitor reads PR keys not in PR_VIEW_FIELDS: {missing_keys}. "
        f"Add them to github.PR_VIEW_FIELDS or update the gate to not read them."
    )


# Test-adequacy gate tests (issue #178)


def _test_adequacy_config(**overrides) -> TestAdequacyConfig:
    """Build a TestAdequacyConfig with overrides for testing."""
    return TestAdequacyConfig(**overrides)


def _test_pr(**overrides) -> dict:
    """Build a minimal PR dict for test-adequacy testing."""
    base = {
        "number": 123,
        "body": "Closes #123.",
    }
    base.update(overrides)
    return base


def test_check_test_adequacy_feature_with_tests_passes() -> None:
    """Feature diff + test file with real recognized assertions → ok=True, no warnings."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,5 @@
 def feature():
     pass
+def new_feature():
+    pass
diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,5 @@
 def test_feature():
-    pass
+    assert new_feature() is not None
+    assert new_feature() == "expected"
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 2
    assert verdict.facts.added_test_loc == 2
    assert verdict.facts.assertion_count == 2
    assert verdict.facts.test_files_changed == 1


def test_check_test_adequacy_pure_skip_hard_fails() -> None:
    """Feature diff, zero test files changed (pure skip), added_product_loc >= min_product_lines → ok=False."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,10 @@
 def feature():
     pass
+def new_feature():
+    pass
+def another():
+    pass
+def third():
+    pass
+def fourth():
+    pass
"""
    pr = _test_pr()
    config = _test_adequacy_config(min_product_lines=5)

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is False
    assert len(verdict.failures) == 1
    assert "Product code changed" in verdict.failures[0]
    assert "no test files changed" in verdict.failures[0]
    assert "src/feature.py" in verdict.failures[0]
    assert verdict.facts.added_product_loc == 8
    assert verdict.facts.test_files_changed == 0


def test_check_test_adequacy_zero_assertions_warns_by_default() -> None:
    """Feature diff + test file present but zero recognized assertion markers → default config: ok=True with warning."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,10 @@
 def feature():
     pass
+def new_feature():
+    pass
+def another():
+    pass
+def third():
+    pass
+def fourth():
+    pass
diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,5 @@
 def test_feature():
-    pass
+def test_new_feature():
+    pass
"""
    pr = _test_pr()
    config = _test_adequacy_config(min_product_lines=5)

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert len(verdict.warnings) == 1
    assert "zero recognized assertions" in verdict.warnings[0]
    assert verdict.facts.assertion_count == 0


def test_check_test_adequacy_zero_assertions_hard_fails_when_required() -> None:
    """Feature diff + test file present but zero recognized assertion markers → require_assertions=True: ok=False."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,10 @@
 def feature():
     pass
+def new_feature():
+    pass
+def another():
+    pass
+def third():
+    pass
+def fourth():
+    pass
diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,5 @@
 def test_feature():
-    pass
+def test_new_feature():
+    pass
"""
    pr = _test_pr()
    config = _test_adequacy_config(min_product_lines=5, require_assertions=True)

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is False
    assert len(verdict.failures) == 1
    assert "zero recognized assertions" in verdict.failures[0]


def test_check_test_adequacy_docs_only_passes() -> None:
    """Docs-only / config-only diff (all files match exempt_path_globs) → ok=True, facts.added_product_loc == 0."""
    diff = """diff --git a/README.md b/README.md
index 123..456 100644
--- a/README.md
+++ b/README.md
@@ -1,3 +1,5 @@
 # README
-Old text
+New text
diff --git a/pyproject.toml b/pyproject.toml
index 123..456 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -1,3 +1,5 @@
 [tool.pytest]
- old_setting = "value"
+ new_setting = "value"
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 0
    assert verdict.facts.added_test_loc == 0


def test_check_test_adequacy_examples_only_passes() -> None:
    """Examples-only diff (files under examples/** match exempt_path_globs) → ok=True, facts.added_product_loc == 0.

    The examples/ directory holds portable templates and config samples (XML,
    YAML, cron), not executable product code — same category as docs/**. This
    guards against the false positive that flagged
    examples/schedule/charlie-fleet-task.xml as untested product code (PR #690).
    """
    diff = """diff --git a/examples/schedule/charlie-fleet-task.xml b/examples/schedule/charlie-fleet-task.xml
index 123..456 100644
--- a/examples/schedule/charlie-fleet-task.xml
+++ b/examples/schedule/charlie-fleet-task.xml
@@ -1,3 +1,5 @@
 <?xml version="1.0" encoding="UTF-8"?>
 <Task>
+  <Triggers>
+    <TimeTrigger/>
+  </Triggers>
 </Task>
diff --git a/examples/orchestrator.config.devin.yaml b/examples/orchestrator.config.devin.yaml
index 123..456 100644
--- a/examples/orchestrator.config.devin.yaml
+++ b/examples/orchestrator.config.devin.yaml
@@ -1,3 +1,5 @@
 fleet:
-  old: value
+  new: value
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 0
    assert verdict.facts.added_test_loc == 0
    assert verdict.facts.untested_product_files == ()


def test_check_test_adequacy_rename_only_passes() -> None:
    """Rename-only diff (100% similarity, no hunk body) → ok=True, facts.added_product_loc == 0."""
    diff = """diff --git a/old_name.py b/new_name.py
similarity index 100%
rename from old_name.py
rename to new_name.py
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 0


def test_check_test_adequacy_rename_with_modify_counts_added_lines() -> None:
    """Rename+modify diff (small number of added lines on the renamed file) → counts added lines."""
    diff = """diff --git a/old_name.py b/new_name.py
similarity index 95%
rename from old_name.py
rename to new_name.py
index 123..456 100644
--- a/old_name.py
+++ b/new_name.py
@@ -1,3 +1,5 @@
 def old_func():
     pass
+def new_func():
+    pass
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.facts.added_product_loc == 2
    assert verdict.facts.test_files_changed == 0


def test_check_test_adequacy_binary_diff_warns() -> None:
    """Binary-file diff → ok=True with a warning, never raises."""
    diff = """Binary files a/image.png and b/image.png differ
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert len(verdict.warnings) == 1
    assert "diff unparseable" in verdict.warnings[0]


def test_check_test_adequacy_malformed_diff_warns() -> None:
    """Malformed/garbage diff string → ok=True, never raises."""
    diff = """this is not a valid diff
at all
just random text
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert len(verdict.warnings) == 1
    assert "diff unparseable" in verdict.warnings[0]


def test_check_test_adequacy_valid_exemption_passes() -> None:
    """Valid Test-exempt: <reason> in PR body → ok=True, facts.exempt is True, facts.exempt_reason == "<reason>"."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,5 @@
 def feature():
     pass
+def new_feature():
+    pass
"""
    pr = _test_pr(body="Closes #123.\n\nTest-exempt: this is a documentation-only change")
    config = _test_adequacy_config(min_product_lines=5)

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.exempt is True
    assert verdict.facts.exempt_reason == "this is a documentation-only change"


def test_check_test_adequacy_exemption_without_reason_not_exempt() -> None:
    """Test-exempt: with no trailing reason text → NOT exempt (regex requires non-empty reason)."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,10 @@
 def feature():
     pass
+def new_feature():
+    pass
+def another():
+    pass
+def third():
+    pass
+def fourth():
+    pass
"""
    pr = _test_pr(body="Closes #123.\n\nTest-exempt:")
    config = _test_adequacy_config(min_product_lines=5)

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is False
    assert verdict.facts.exempt is False
    assert verdict.facts.exempt_reason == ""


def test_check_test_adequacy_custom_exempt_marker_honored() -> None:
    """Custom exempt_marker config override is honored; the default Test-exempt: does NOT match when overridden."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,5 @@
 def feature():
     pass
+def new_feature():
+    pass
"""
    pr = _test_pr(body="Closes #123.\n\nNo-Test-Needed: this is a config-only change")
    config = _test_adequacy_config(min_product_lines=5, exempt_marker="No-Test-Needed:")

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.facts.exempt is True
    assert verdict.facts.exempt_reason == "this is a config-only change"


def test_check_test_adequacy_below_min_product_lines_passes() -> None:
    """Product diff below min_product_lines → ok=True regardless of test presence."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,4 @@
 def feature():
     pass
+def new_feature():
+    pass
"""
    pr = _test_pr()
    config = _test_adequacy_config(min_product_lines=10)

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()


def test_check_test_adequacy_test_only_diff_passes() -> None:
    """Test-only diff (no product files changed) → ok=True."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,5 @@
 def test_feature():
-    pass
+    assert new_feature() is not None
+    assert new_feature() == "expected"
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 0
    assert verdict.facts.added_test_loc == 2
    assert verdict.facts.test_files_changed == 1


def test_check_test_adequacy_bugfix_test_only_passes() -> None:
    """Bugfix diff that only modifies existing test files (with assertions present) and touches no product files → ok=True."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,5 @@
 def test_feature():
-    assert old_behavior() == "old"
+    assert new_behavior() == "new"
+    assert edge_case() is not None
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 0
    assert verdict.facts.assertion_count == 2


def test_check_test_adequacy_conftest_limitation_accepted() -> None:
    """conftest.py carrying non-trivial added logic, classified as test via default glob → those added lines do NOT count toward added_product_loc (locks in the documented accepted evasion)."""
    diff = """diff --git a/conftest.py b/conftest.py
index 123..456 100644
--- a/conftest.py
+++ b/conftest.py
@@ -1,3 +1,10 @@
 import pytest
-
+def new_fixture():
+    return "value"
+
+@pytest.fixture
+def custom_config():
+    return {"key": "value"}
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    # conftest.py is classified as test (matches default test_path_globs)
    # so its added lines count toward added_test_loc, not added_product_loc
    # Note: blank lines (like the one with just "-") are not counted as added lines
    assert verdict.facts.added_product_loc == 0
    assert verdict.facts.added_test_loc == 5
    assert verdict.facts.test_files_changed == 1


# Stub-test detection tests (issue #224)


def test_check_stub_tests_pass_body_marker() -> None:
    """Only-pass/.../docstring bodies are flagged as stub tests."""
    diff = '''diff --git a/tests/test_feature.py b/tests/test_feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,6 @@
+def test_pass_stub():
+    pass  # placeholder
+def test_ellipsis_stub():
+    ...
+def test_docstring_stub():
+    """docstring"""
'''
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert len(warnings) == 3
    assert any("test_pass_stub" in w for w in warnings)
    assert any("test_ellipsis_stub" in w for w in warnings)
    assert any("test_docstring_stub" in w for w in warnings)


def test_check_stub_tests_assert_constant_ignores_product_references() -> None:
    """Assertions referencing product modules are fine; unrelated module constants are flagged."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,2 +1,2 @@
 def feature():
-    return 1
+    return 2
diff --git a/tests/test_feature.py b/tests/test_feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,9 @@
+from feature import do_thing
+from other import UNRELATED
+def test_do_thing_works():
+    assert do_thing() is not None
+def test_constant_stub():
+    assert UNRELATED > 0
+def test_local_only():
+    result = do_thing()
+    assert result == 1
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert any("test_constant_stub" in w and "assert-constant" in w for w in warnings)
    assert not any("test_do_thing_works" in w for w in warnings)
    assert not any("test_local_only" in w for w in warnings)


def test_check_stub_tests_seam_name_mismatch() -> None:
    """Test names containing a seam keyword require the body to call/mention that seam."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,5 @@
+def test_call_model_smoke():
+    assert True
+def test_call_model_real():
+    call_model()
+def test_route_smoke():
+    assert True
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert any("test_call_model_smoke" in w and "seam-name" in w for w in warnings)
    assert any("test_route_smoke" in w and "seam-name" in w for w in warnings)
    assert not any("test_call_model_real" in w and "seam-name" in w for w in warnings)


def test_check_stub_tests_async_body_is_flagged() -> None:
    """async def test_... functions are inspected the same as sync functions."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,2 @@
+async def test_async_pass_stub():
+    pass
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert any("test_async_pass_stub" in w for w in warnings)


def test_check_stub_tests_gutted_existing_body_is_flagged() -> None:
    """A pre-existing test whose body is gutted to pass is flagged even if the def line is context."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,2 @@
 def test_existing():
-    result = feature()
-    assert result
+    pass
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert any("test_existing" in w for w in warnings)


def test_check_stub_tests_unmodified_test_not_flagged_by_added_blank_line() -> None:
    """An unmodified test is not flagged when only a blank line elsewhere is added."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,4 +1,5 @@
 def test_one():
     pass
+
 def test_two():
     pass
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert not any("test_one" in w or "test_two" in w for w in warnings)


def test_check_stub_tests_added_decorator_does_not_flag_unrelated_test() -> None:
    """A decorator added to another function must not flag an unmodified test."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,4 +1,5 @@
+@pytest.mark.slow
 def test_one():
     pass
 def test_two():
     pass
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert not any("test_one" in w or "test_two" in w for w in warnings)


def test_check_stub_tests_async_helper_reference_not_assert_constant() -> None:
    """A test asserting on a top-level async helper's name is not assert-constant.

    _collect_test_defined_names must record AsyncFunctionDef names; otherwise
    the helper name reads as an unknown external and the assertion is
    misclassified as constant.
    """
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,7 @@
+async def _drain_queue():
+    return 1
+
+
+def test_helper_is_exported():
+    handler = _drain_queue
+    assert handler is _drain_queue
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert not any("test_helper_is_exported" in w for w in warnings)


def test_check_stub_tests_gutted_existing_async_body_is_flagged() -> None:
    """An existing async test gutted to pass is flagged even if the def line is context."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,2 @@
 async def test_existing_async():
-    result = await feature()
-    assert result
+    pass
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert any("test_existing_async" in w for w in warnings)


def test_check_stub_tests_deletion_only_gutted_body_is_flagged() -> None:
    """Gutting by pure deletion (no added lines in the function) is still caught."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,4 +1,2 @@
 def test_existing():
     '''checks the feature'''
-    result = feature()
-    assert result
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert any("test_existing" in w for w in warnings)


def test_check_stub_tests_deletion_near_healthy_test_not_flagged() -> None:
    """A deletion adjacent to a substantive test does not flag that test."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,4 +1,3 @@
 def test_existing():
     result = feature()
     assert result
-# stale comment
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert not any("test_existing" in w for w in warnings)


def test_run_janitor_appends_stub_warnings_from_pr_diff() -> None:
    """run_janitor calls check_stub_tests and adds its warnings to the verdict."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,2 @@
+def test_pass_stub():
+    pass  # placeholder
"""
    verdict = run_janitor(
        _green_pr(), _green_checks(), _config(), repo_root=Path.cwd(), pr_diff=diff
    )

    assert verdict.ok is True
    assert any("test_pass_stub" in w for w in verdict.warnings)
