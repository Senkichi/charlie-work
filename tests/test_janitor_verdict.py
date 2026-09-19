"""``run_janitor`` verdict-composition tests.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1): the PR-shape
gates (draft / non-open / conflicting / fully-green), linked-issue and
body-marker requirements, conventional-commit title checks, oversized-diff
warning, missing-key tolerance, failure aggregation, body word-boundary
matching, the ``JANITOR_PR_KEYS`` / ``PR_VIEW_FIELDS`` containment check, and
the base-movement warning.
"""

from __future__ import annotations

import re
from pathlib import Path

from _janitor_fixtures import (
    _config,
    _green_checks,
    _green_pr,
)

from charlie_work.github import PR_VIEW_FIELDS
from charlie_work.janitor import (
    CONVENTIONAL_COMMIT_TYPES,
    JANITOR_PR_KEYS,
    JanitorVerdict,
    run_janitor,
)


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
    # Issue #818: draft is the ONLY failure here (checks/mergeable/issue-link/
    # body all pass), so this is the "otherwise ready" case workflow.review()
    # uses to decide whether auto-readying the PR via `gh pr ready` is safe.
    assert verdict.is_draft is True
    assert verdict.is_draft_only_block is True


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
