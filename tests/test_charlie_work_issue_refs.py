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
