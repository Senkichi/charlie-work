"""Tests for linked_issue_number, carved out of test_charlie_work.py (#1284)."""

from __future__ import annotations

from charlie_work.github import linked_issue_number


def test_linked_issue_number_from_branch_body_or_title() -> None:
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-456-fix"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 456
    )
    assert (
        linked_issue_number(
            {"body": "Closes #789"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 789
    )
    assert (
        linked_issue_number(
            {"title": "Fix #321: thing"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 321
    )


def test_linked_issue_number_ignores_unqualified_body_references() -> None:
    body = "Bumps actions/checkout. See dependabot/dependabot-core#2454 for details."

    assert (
        linked_issue_number(
            {"body": body},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        is None
    )


def test_linked_issue_number_rejects_bare_hash_in_attacker_title() -> None:
    # A bare #N substring in an attacker-controlled title must NOT bind the PR
    # to issue N (label/merge hijack). Only a closing keyword counts.
    assert (
        linked_issue_number(
            {"title": "Refactor everything #1 nicely"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        is None
    )
    assert (
        linked_issue_number(
            {"title": "see #5 for context", "body": "no link"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        is None
    )
    # Closing-keyword forms still resolve.
    assert (
        linked_issue_number(
            {"title": "Fix #321: thing"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 321
    )
    assert (
        linked_issue_number(
            {"body": "Resolves #7"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 7
    )
    # Orchestrator's own branch convention is the trusted head-ref signal.
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-456-x", "title": "#999"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 456
    )


def test_linked_issue_number_fork_pr_branch_name_does_not_bind() -> None:
    # Issue #9: Fork PRs must not bind via branch name (attacker-controlled).
    # A fork PR with branch name "issue-42" should NOT bind to issue 42.
    assert (
        linked_issue_number(
            {"headRefName": "issue-42-fix"},
            is_cross_repository=True,
            branch_prefix="agent/issue",
        )
        is None
    )
    # Even with the orchestrator's prefix, fork PRs must not bind via branch.
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-42-fix"},
            is_cross_repository=True,
            branch_prefix="agent/issue",
        )
        is None
    )


def test_linked_issue_number_same_repo_branch_with_prefix_binds() -> None:
    # Issue #9: Same-repo PRs with correct branch prefix should still bind.
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-42-fix"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 42
    )
    # Same-repo PR with wrong prefix should not bind via branch.
    assert (
        linked_issue_number(
            {"headRefName": "issue-42-fix"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        is None
    )


def test_linked_issue_number_fork_pr_closing_keyword_does_not_bind() -> None:
    # Issue #9: Fork PRs must NOT bind via closing keywords for lifecycle purposes.
    # (GitHub's own auto-close on merge is GitHub's policy for issue state;
    # the orchestrator's label lifecycle is ours.)
    assert (
        linked_issue_number(
            {"body": "Closes #42"},
            is_cross_repository=True,
            branch_prefix="agent/issue",
        )
        is None
    )
    assert (
        linked_issue_number(
            {"title": "Fix #42: security issue"},
            is_cross_repository=True,
            branch_prefix="agent/issue",
        )
        is None
    )


def test_linked_issue_number_same_repo_closing_keyword_binds() -> None:
    # Same-repo PRs should still bind via closing keywords.
    assert (
        linked_issue_number(
            {"body": "Closes #42"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 42
    )
    assert (
        linked_issue_number(
            {"title": "Fix #42: security issue"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 42
    )


def test_linked_issue_number_none_treats_as_cross_repository() -> None:
    # When is_cross_repository is None (provenance unknown), treat as cross-repo
    # for trust purposes — bind nothing via branch name or closing keyword
    # (fail closed). This hardens against future call sites that omit the
    # parameter or pass a PR dict missing the isCrossRepository field.
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-42-fix"},
            is_cross_repository=None,
            branch_prefix="agent/issue",
        )
        is None
    )
    assert (
        linked_issue_number(
            {"body": "Closes #42"},
            is_cross_repository=None,
            branch_prefix="agent/issue",
        )
        is None
    )


def test_linked_issue_number_negation_guard_rejects_negated_keyword() -> None:
    # Issue #781 AC1: a closing keyword preceded by a negation must not
    # bind -- "does not fix #649" is the real-world PR #766 -> issue #649
    # text that produced a false LABEL TRANSITION (not a GitHub auto-close;
    # this guard only affects charlie-work's own binding). The non-negated
    # form of the identical keyword still binds.
    assert (
        linked_issue_number(
            {"body": "does not fix #649"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        is None
    )
    assert (
        linked_issue_number(
            {"body": "Fixes #649"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 649
    )


def test_linked_issue_number_negation_guard_table() -> None:
    # Issue #781 AC2: table-driven negation coverage (doesn't / never /
    # without / cannot / does not), plus non-negated controls proving the
    # guard doesn't over-trigger on unrelated text.
    negated_cases = [
        "doesn't close #1",
        "never resolves #2",
        # "without fixing #3" returns None via the BASE regex, not the
        # negation guard: _CLOSING_KEYWORD_REF only matches fix/fixes/fixed
        # (fix(?:e[sd])?) -- never the gerund "fixing". Included per the
        # issue's explicit list, but see the dedicated assertion below for
        # proof this is NOT evidence the negation guard itself fired.
        "without fixing #3",
        "cannot close #4",
        "does not fix #5",
    ]
    for body in negated_cases:
        assert (
            linked_issue_number(
                {"body": body},
                is_cross_repository=False,
                branch_prefix="agent/issue",
            )
            is None
        ), f"expected no binding for negated text: {body!r}"

    # Non-negated controls: the same keywords, without a preceding negation,
    # still bind -- proves the guard isn't simply matching nothing.
    non_negated_cases = [
        ("Closes #1", 1),
        ("This resolves #2", 2),
        ("Fixes #5", 5),
    ]
    for body, expected in non_negated_cases:
        assert (
            linked_issue_number(
                {"body": body},
                is_cross_repository=False,
                branch_prefix="agent/issue",
            )
            == expected
        ), f"expected binding to {expected} for non-negated text: {body!r}"

    # "fixing #3" alone (no negation at all) confirms the gerund truly never
    # matches the base regex -- so the None result above for "without
    # fixing #3" is not evidence the negation guard fired.
    assert (
        linked_issue_number(
            {"body": "fixing #3"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        is None
    )


def test_linked_issue_number_negation_does_not_shadow_later_genuine_match() -> None:
    # A negated match earlier in a text field must not prevent a later,
    # genuine closing keyword from binding -- finditer-continuation, not
    # search()-first-match-only. Monotone with respect to pre-#781 behavior:
    # no input that previously returned a number now returns None because of
    # this change; the negation guard can only turn a previously-binding
    # negated match into None, never suppress a separate genuine one.
    #
    # The negation lookback window (32 chars, chosen to cover "does not "
    # plus headroom) is deliberately wide enough to span a short clause, so
    # a genuine match must sit outside that window to prove continuation
    # rather than an accidental non-negation coincidence. Verified
    # empirically: without the padding, "does not fix #649. Fixes #700"
    # actually returns None (the 32-char window reaches back across the
    # sentence boundary to "not") -- that cross-sentence over-triggering is
    # the deliberate, sanctioned safe-direction tradeoff described on
    # `_NEGATION_LOOKBEHIND_CHARS`, not a bug. This test instead places the
    # genuine match far enough away to isolate the continuation behavior.
    padded_body = "does not fix #649. " + ("x" * 50) + " Fixes #700"
    assert (
        linked_issue_number(
            {"body": padded_body},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 700
    )


# ---------------------------------------------------------------------------
# Issue #1229: branch_issue_validator — reject stale branch-name bindings
# ---------------------------------------------------------------------------


def test_linked_issue_number_validator_rejects_stale_branch_binding() -> None:
    """A branch-name number with no matching open issue must not bind.

    This is the core of issue #1229: a branch ``agent/issue-709-…`` left over
    from a merged PR #709, reused by an unrelated issue-less PR, must not
    silently bind the PR to issue 709. The validator returns False for 709
    (not an open issue), so the branch-name binding is rejected.
    """
    open_issues = frozenset({123, 456})
    validator = lambda n: n in open_issues  # noqa: E731
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-709-stale-branch"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
            branch_issue_validator=validator,
        )
        is None
    )


def test_linked_issue_number_validator_accepts_valid_branch_binding() -> None:
    """A branch-name number that IS a real open issue still binds."""
    open_issues = frozenset({123, 456})
    validator = lambda n: n in open_issues  # noqa: E731
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-123-real-issue"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
            branch_issue_validator=validator,
        )
        == 123
    )


def test_linked_issue_number_validator_falls_through_to_closing_keyword() -> None:
    """When the branch-name binding is rejected, the closing-keyword path runs.

    A PR with a stale branch name ``agent/issue-709-…`` AND a genuine closing
    keyword ``Fixes #123`` in the body should bind to 123 (the closing
    keyword), not 709 (the rejected branch name) and not None.
    """
    open_issues = frozenset({123})
    validator = lambda n: n in open_issues  # noqa: E731
    assert (
        linked_issue_number(
            {
                "headRefName": "agent/issue-709-stale-branch",
                "body": "Fixes #123",
            },
            is_cross_repository=False,
            branch_prefix="agent/issue",
            branch_issue_validator=validator,
        )
        == 123
    )


def test_linked_issue_number_validator_none_preserves_existing_behavior() -> None:
    """When no validator is supplied, the branch-name binding is trusted.

    This is the backward-compatibility guarantee: existing callers that do not
    pass ``branch_issue_validator`` get the same behavior as before #1229.
    """
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-709-stale-branch"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        == 709
    )


def test_linked_issue_number_validator_does_not_affect_closing_keyword_only() -> None:
    """A PR that binds via closing keyword only (no branch prefix) is unaffected.

    The validator applies exclusively to the branch-name path; a closing-keyword
    binding with no branch-name match must still bind regardless of the
    validator's opinion about that number.
    """
    open_issues: frozenset[int] = frozenset()  # empty — nothing is "open"
    validator = lambda n: n in open_issues  # noqa: E731
    assert (
        linked_issue_number(
            {"body": "Fixes #789"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
            branch_issue_validator=validator,
        )
        == 789
    )


def test_linked_issue_number_validator_rejects_then_no_keyword_returns_none() -> None:
    """Stale branch binding + no closing keyword → None (issue-less PR).

    This is the exact scenario from issue #1229: PR #1660 with branch
    ``agent/issue-709-…`` and no closing keyword. The validator rejects 709,
    there is no closing keyword to fall through to, and the result is None —
    the PR is correctly treated as issue-less, preventing the collision with
    the unrelated issue/PR #709's state entry.
    """
    open_issues = frozenset({123})
    validator = lambda n: n in open_issues  # noqa: E731
    assert (
        linked_issue_number(
            {"headRefName": "agent/issue-709-stale-branch", "body": "docs update"},
            is_cross_repository=False,
            branch_prefix="agent/issue",
            branch_issue_validator=validator,
        )
        is None
    )
