from __future__ import annotations

import subprocess
from pathlib import Path

from charlie_work.config import AutoMergeConfig, OrchestratorConfig, ReviewConfig
from charlie_work.janitor import JanitorVerdict, check_operator_containment, run_janitor

REQUIRED_CHECKS = ("Tests passed", "Lint & Format")


def _init_repo(repo_root: Path) -> None:
    """Initialize a git repo with a single commit."""
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
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
        "body": "Closes #123.\n\nTests: added unit tests for the search path.",
        "isDraft": False,
        "state": "OPEN",
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
    verdict = run_janitor(_green_pr(), _green_checks(), _config())

    assert verdict == JanitorVerdict(ok=True, failures=(), warnings=())


def test_draft_pr_fails() -> None:
    verdict = run_janitor(_green_pr(isDraft=True), _green_checks(), _config())

    assert verdict.ok is False
    assert any("draft" in f.lower() for f in verdict.failures)


def test_non_open_state_fails() -> None:
    verdict = run_janitor(_green_pr(state="CLOSED"), _green_checks(), _config())

    assert verdict.ok is False
    assert any("CLOSED" in f for f in verdict.failures)


def test_conflicting_mergeable_fails() -> None:
    verdict = run_janitor(_green_pr(mergeable="CONFLICTING"), _green_checks(), _config())

    assert verdict.ok is False
    assert any("conflict" in f.lower() for f in verdict.failures)


def test_required_check_failure_blocks() -> None:
    checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]

    verdict = run_janitor(_green_pr(), checks, _config())

    assert verdict.ok is False
    assert any("Tests passed" in f for f in verdict.failures)


def test_required_check_missing_blocks() -> None:
    checks = [{"name": "Lint & Format", "bucket": "pass"}]

    verdict = run_janitor(_green_pr(), checks, _config())

    assert verdict.ok is False
    assert any("missing" in f.lower() and "Tests passed" in f for f in verdict.failures)


def test_required_check_pending_warns_not_fails() -> None:
    checks = [
        {"name": "Tests passed", "state": "PENDING"},
        {"name": "Lint & Format", "bucket": "pass"},
    ]

    verdict = run_janitor(_green_pr(), checks, _config())

    assert verdict.ok is True
    assert verdict.failures == ()
    assert any("pending" in w.lower() and "Tests passed" in w for w in verdict.warnings)


def test_no_required_checks_configured_skips_check_gate() -> None:
    verdict = run_janitor(_green_pr(), [], _config(required_checks=()))

    assert verdict.ok is True


def test_missing_linked_issue_fails_when_required() -> None:
    pr = _green_pr(
        headRefName="agent/misc-branch", body="No issue reference here at all, tests added."
    )

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is False
    assert any("linked issue" in f.lower() for f in verdict.failures)


def test_missing_linked_issue_ok_when_not_required() -> None:
    pr = _green_pr(
        headRefName="agent/misc-branch", body="No issue reference here at all, tests added."
    )

    verdict = run_janitor(pr, _green_checks(), _config(require_issue_link=False))

    assert verdict.ok is True


def test_empty_body_fails() -> None:
    verdict = run_janitor(_green_pr(body=""), _green_checks(), _config())

    assert verdict.ok is False
    assert any("body is empty" in f.lower() for f in verdict.failures)


def test_body_with_only_whitespace_fails() -> None:
    verdict = run_janitor(_green_pr(body="   \n  "), _green_checks(), _config())

    assert verdict.ok is False
    assert any("body is empty" in f.lower() for f in verdict.failures)


def test_body_without_tests_or_rationale_marker_fails_when_required() -> None:
    pr = _green_pr(body="Closes #123. This fixes the thing, no more detail than that.")

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is False
    assert any("tests/verification/rationale" in f.lower() for f in verdict.failures)


def test_body_with_rationale_marker_passes() -> None:
    pr = _green_pr(body="Closes #123. No tests because this is a comment-only change.")

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is True


def test_body_marker_check_skipped_when_not_required() -> None:
    pr = _green_pr(body="Closes #123. This fixes the thing, no more detail than that.")

    verdict = run_janitor(pr, _green_checks(), _config(require_tests_or_rationale=False))

    assert verdict.ok is True


def test_non_conventional_title_warns() -> None:
    pr = _green_pr(title="Search improvements")

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is True
    assert any("conventional-commit" in w.lower() for w in verdict.warnings)


def test_conventional_title_variants_do_not_warn() -> None:
    for title in ("feat: add x", "fix(search): bug", "chore!: breaking", "docs: update readme"):
        verdict = run_janitor(_green_pr(title=title), _green_checks(), _config())
        assert not any("conventional-commit" in w.lower() for w in verdict.warnings), title


def test_oversized_diff_warns() -> None:
    pr = _green_pr(additions=1000, deletions=600)

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is True
    assert any("oversized diff" in w.lower() for w in verdict.warnings)


def test_diff_at_threshold_does_not_warn() -> None:
    pr = _green_pr(additions=1000, deletions=500)  # exactly 1500

    verdict = run_janitor(pr, _green_checks(), _config())

    assert not any("oversized diff" in w.lower() for w in verdict.warnings)


def test_missing_keys_never_raise_and_skip_checks() -> None:
    # Minimal pr dict: gh omits fields depending on flags used to fetch it.
    verdict = run_janitor({}, [], _config(required_checks=()))

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
    )

    assert verdict == JanitorVerdict(ok=True, failures=(), warnings=())


def test_multiple_failures_all_reported() -> None:
    pr = _green_pr(isDraft=True, state="CLOSED", mergeable="CONFLICTING", body="")

    verdict = run_janitor(pr, [], _config(required_checks=()))

    assert verdict.ok is False
    assert len(verdict.failures) >= 4


def test_base_movement_warns_for_agent_pr() -> None:
    pr = _green_pr(mergeStateStatus="BEHIND")
    config = _config()

    verdict = run_janitor(pr, _green_checks(), config)

    assert verdict.ok is True
    assert any(
        "Base branch has moved since branch (mergeStateStatus=BEHIND)" in w
        for w in verdict.warnings
    )


def test_base_movement_skips_fork_pr() -> None:
    pr = _green_pr(mergeStateStatus="BEHIND", isCrossRepository=True)

    verdict = run_janitor(pr, _green_checks(), _config(require_issue_link=False))

    assert verdict.ok is True
    assert not any("Base moved" in w for w in verdict.warnings)


def test_base_movement_skips_non_prefix_branch() -> None:
    pr = _green_pr(mergeStateStatus="BEHIND", headRefName="feature/something")

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is True
    assert not any("Base moved" in w for w in verdict.warnings)


def test_base_movement_no_warning_when_up_to_date() -> None:
    pr = _green_pr(mergeStateStatus="CLEAN")

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is True
    assert not any("Base moved" in w for w in verdict.warnings)


def test_base_movement_no_warning_when_field_missing() -> None:
    pr = _green_pr()
    # Remove mergeStateStatus if it exists
    pr.pop("mergeStateStatus", None)

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is True
    assert not any("Base moved" in w for w in verdict.warnings)


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


def test_closed_pr_blocked_with_real_pr_view_fields() -> None:
    """CLOSED PR is blocked when using real pr_view field set (regression test for issue #2)."""
    # This PR dict mirrors the real pr_view field set after the fix
    # Before the fix, state/mergeable/additions/deletions were missing
    pr = {
        "number": 456,
        "title": "fix: search is broken",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "baseRefName": "main",
        "body": "Closes #123.\n\nTests: added unit tests for the search path.",
        "isDraft": False,
        "labels": [],
        "author": {"login": "testuser"},
        "updatedAt": "2024-01-01T00:00:00Z",
        "reviewDecision": "APPROVED",
        "statusCheckRollup": [],
        "state": "CLOSED",  # This field was missing before the fix
        "mergeable": "MERGEABLE",  # This field was missing before the fix
        "additions": 10,  # This field was missing before the fix
        "deletions": 5,  # This field was missing before the fix
        "headRefOid": "abc123",
        "isCrossRepository": False,
        "mergeStateStatus": "CLEAN",
    }

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is False
    assert any("CLOSED" in f for f in verdict.failures)


def test_conflicting_pr_blocked_with_real_pr_view_fields() -> None:
    """CONFLICTING PR is blocked when using real pr_view field set (regression test for issue #2)."""
    # This PR dict mirrors the real pr_view field set after the fix
    pr = {
        "number": 456,
        "title": "fix: search is broken",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "baseRefName": "main",
        "body": "Closes #123.\n\nTests: added unit tests for the search path.",
        "isDraft": False,
        "labels": [],
        "author": {"login": "testuser"},
        "updatedAt": "2024-01-01T00:00:00Z",
        "reviewDecision": "APPROVED",
        "statusCheckRollup": [],
        "state": "OPEN",  # This field was missing before the fix
        "mergeable": "CONFLICTING",  # This field was missing before the fix
        "additions": 10,  # This field was missing before the fix
        "deletions": 5,  # This field was missing before the fix
        "headRefOid": "abc123",
        "isCrossRepository": False,
        "mergeStateStatus": "DIRTY",
    }

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is False
    assert any("conflict" in f.lower() for f in verdict.failures)


def test_oversized_diff_warning_with_real_pr_view_fields() -> None:
    """Oversized diff warning fires when using real pr_view field set (regression test for issue #2)."""
    # This PR dict mirrors the real pr_view field set after the fix
    pr = {
        "number": 456,
        "title": "fix: search is broken",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "baseRefName": "main",
        "body": "Closes #123.\n\nTests: added unit tests for the search path.",
        "isDraft": False,
        "labels": [],
        "author": {"login": "testuser"},
        "updatedAt": "2024-01-01T00:00:00Z",
        "reviewDecision": "APPROVED",
        "statusCheckRollup": [],
        "state": "OPEN",  # This field was missing before the fix
        "mergeable": "MERGEABLE",  # This field was missing before the fix
        "additions": 1000,  # This field was missing before the fix
        "deletions": 600,  # This field was missing before the fix
        "headRefOid": "abc123",
        "isCrossRepository": False,
        "mergeStateStatus": "CLEAN",
    }

    verdict = run_janitor(pr, _green_checks(), _config())

    assert verdict.ok is True
    assert any("oversized diff" in w.lower() for w in verdict.warnings)


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
