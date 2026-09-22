"""Issue-reference text handling: issue_numbers_mentioned_by_pr, label_names, closing-keyword defang.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

from charlie_work.github import (
    issue_numbers_mentioned_by_pr,
    label_names,
)


def test_label_names_accepts_gh_shape() -> None:
    issue = {"labels": [{"name": "automated-ready"}, {"name": "agent:in-progress"}]}

    assert label_names(issue) == {"automated-ready", "agent:in-progress"}


def test_issue_numbers_mentioned_by_pr_matches_issue_reference() -> None:
    pr = {
        "title": "fix(scope): reap sidecar files on session exit (issue #113)",
        "body": "This PR addresses issue #113. PR #181 is an unrelated refactor.",
    }

    assert issue_numbers_mentioned_by_pr(pr) == {113}


def test_issue_numbers_mentioned_by_pr_ignores_fenced_code_blocks() -> None:
    # A code sample that happens to contain the literal text must not count
    # as a real reference — advisory-only matching per the function's
    # contract, but obviously-wrong matches are worth stripping.
    pr = {
        "title": "docs: add example",
        "body": "Example:\n```\n# see issue #113 for context\n```\nNo real reference here.",
    }

    assert issue_numbers_mentioned_by_pr(pr) == set()


def test_issue_numbers_mentioned_by_pr_ignores_blockquoted_lines() -> None:
    # Quoted reply text (e.g. an email-style blockquote) must not count.
    pr = {
        "title": "chore: reply to review",
        "body": "> unlike issue #113, this one is fine\n\nAddressed the other comments.",
    }

    assert issue_numbers_mentioned_by_pr(pr) == set()


def test_issue_numbers_mentioned_by_pr_suppresses_visibility_qualified_mentions() -> None:
    """Issue #1803: ``private issue #N`` / ``internal issue #N`` name another
    repo's tracker (the jobcannon docs rewrite that produced the #377/#391
    false escalations). Suppression needs no repo context -- visibility
    designators are unconditionally foreign."""
    pr = {
        "title": "docs: qualify bare issue refs",
        "body": "Rewrote to private issue #391; also internal issue #42.",
    }

    assert issue_numbers_mentioned_by_pr(pr) == set()


def test_issue_numbers_mentioned_by_pr_suppresses_slug_qualified_mentions() -> None:
    """Issue #1803: an ``owner/repo`` slug qualifier is GitHub's own
    cross-repo reference syntax. It is foreign unless it is exactly
    ``current_repo`` -- both qualifier positions count (before "issue" and
    between "issue" and the hash)."""
    pr = {"title": "", "body": "see issue owner/repo#7 and owner/repo issue #8"}

    assert issue_numbers_mentioned_by_pr(pr, current_repo="me/mine") == set()
    # Same text, self-slug: both mentions count.
    assert issue_numbers_mentioned_by_pr(pr, current_repo="owner/repo") == {7, 8}


def test_issue_numbers_mentioned_by_pr_suppresses_managed_repo_qualifier() -> None:
    """Issue #1803: a bare-word qualifier naming another *managed* repo
    (``other_repo_names``, fleet-registry derived) is foreign; the
    dispatching repo's own name is not."""
    pr = {"title": "", "body": "tracked as sibling-repo issue #10"}

    assert issue_numbers_mentioned_by_pr(pr, other_repo_names={"sibling-repo"}) == set()
    # Self-name beats the foreign set when both are supplied.
    assert issue_numbers_mentioned_by_pr(
        pr, current_repo="me/sibling-repo", other_repo_names={"sibling-repo"}
    ) == {10}


def test_issue_numbers_mentioned_by_pr_unqualified_and_generic_qualifiers_still_match() -> None:
    """Regression guard: the qualifier capture must not swallow ordinary
    prose -- unqualified mentions and non-repo qualifier words behave
    exactly as before."""
    pr = {
        "title": "fix(scope): reap sidecar files (issue #113)",
        "body": "This PR addresses issue #114 and the issue #115 too.",
    }

    assert issue_numbers_mentioned_by_pr(pr) == {113, 114, 115}


def test_defang_closing_keywords_strips_live_keyword_but_keeps_number_legible() -> None:
    # Issue #781 AC3: defang_closing_keywords rewrites `<keyword> #N` to
    # `<keyword> issue N` -- the rewritten text no longer matches the
    # closing-keyword regex (so it can no longer auto-close or falsely bind
    # via linked_issue_number), but the issue number stays legible to a
    # human reader.
    from charlie_work.github import defang_closing_keywords
    from charlie_work.issue_linking import _CLOSING_KEYWORD_REF

    text = "does not fix #649"
    defanged = defang_closing_keywords(text)

    assert _CLOSING_KEYWORD_REF.search(defanged) is None
    assert "649" in defanged, "issue number must remain legible to a human"
    assert defanged == "does not fix issue 649"


def test_defang_closing_keywords_preserves_non_keyword_hash_refs() -> None:
    # Bare `#N` (no preceding closing keyword) and `issue #N` mentions pass
    # through untouched -- defang targets only the live auto-close syntax.
    # Multiple keyword refs in one string are each independently defanged.
    from charlie_work.github import defang_closing_keywords

    assert defang_closing_keywords("see #5 for context") == "see #5 for context"
    assert defang_closing_keywords("related to issue #5") == "related to issue #5"
    assert (
        defang_closing_keywords("Closes #1 and also fixes #2")
        == "Closes issue 1 and also fixes issue 2"
    )
