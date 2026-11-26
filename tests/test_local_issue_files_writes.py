"""Tests for local_issue_files.py's create/edit helpers (issue #3116).

Seam-named sibling of ``test_local_issue_files.py`` -- the same redirect the
attachment contract applied for ``test_local_issues.py`` ->
``test_local_issues_writes.py``: new coverage for the write helpers
(``new_issue_filename``, ``render_issue_text``, ``rewrite_body``) lives here
rather than growing the parse/scan module past its member ceiling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.local_issue_files import (
    _COMMENT_SEPARATOR,
    COMMENTS_MARKER,
    STATE_OPEN,
    IssueFileError,
    append_comment,
    new_issue_filename,
    parse_issue_text,
    render_issue_text,
    rewrite_body,
)


def _issue_text(
    *,
    title: str = "",
    state: str = "",
    labels: str = "",
    created: str = "",
    body: str = "Body text.",
) -> str:
    content: list[str] = []
    if title:
        content.append(f'title: "{title}"')
    if state:
        content.append(f"state: {state}")
    if labels:
        content.append(f"labels: {labels}")
    if created:
        content.append(f"created: {created}")
    if not content:
        # The frontmatter fence regex requires a (possibly blank) line
        # between the two "---" markers; an empty frontmatter block still
        # needs that blank line to be present.
        content.append("")
    lines = ["---", *content, "---", body]
    return "\n".join(lines) + "\n"


def _path(tmp_path: Path, name: str = "001_2026-09-17_a.md") -> Path:
    return tmp_path / name


def test_new_issue_filename_slugifies_the_title() -> None:
    assert new_issue_filename(3, "Fix the Thing: Now!") == "003_fix-the-thing-now.md"


def test_new_issue_filename_falls_back_when_title_has_no_slug_chars() -> None:
    assert new_issue_filename(7, "!!!") == "007_issue.md"


def test_render_issue_text_round_trips_through_parse(tmp_path: Path) -> None:
    text = render_issue_text(
        title='A "quoted" title: with colon',
        body="The body.",
        labels=("a", "needs: triage"),
        created="2026-11-25",
    )
    issue = parse_issue_text(text, _path(tmp_path))
    assert issue.title == 'A "quoted" title: with colon'
    assert issue.body == "The body."
    assert issue.labels == ("a", "needs: triage")
    assert issue.state == STATE_OPEN


def test_render_issue_text_normalizes_crlf_body(tmp_path: Path) -> None:
    """A CRLF body lands in the LF-rendered file as LF -- no stray ``\\r``."""
    text = render_issue_text(
        title="t",
        body="line one\r\nline two\r\n",
        labels=(),
        created="2026-11-25",
    )
    assert "\r" not in text
    assert parse_issue_text(text, _path(tmp_path)).body == "line one\nline two"


def test_rewrite_body_replaces_only_the_body_region(tmp_path: Path) -> None:
    text = _issue_text(title="T", state="open", labels="[a]", body="old body")
    rewritten = rewrite_body(text, "new body")

    issue = parse_issue_text(rewritten, _path(tmp_path))
    assert issue.body == "new body"
    assert issue.title == "T"
    assert issue.labels == ("a",)
    # Frontmatter block byte-identical: only the body region changed.
    assert rewritten[: rewritten.index("---", 4)] == text[: text.index("---", 4)]


def test_rewrite_body_preserves_comments_block(tmp_path: Path) -> None:
    text = append_comment(
        _issue_text(title="T", body="old"), "keep me", timestamp="2026-09-17T00:00:00Z"
    )
    rewritten = rewrite_body(text, "new body")
    issue = parse_issue_text(rewritten, _path(tmp_path))
    assert issue.body == "new body"
    assert issue.comments == ("**charlie-work** (2026-09-17T00:00:00Z)\n\nkeep me",)


def test_rewrite_body_normalizes_multiline_body_to_crlf(tmp_path: Path) -> None:
    """A multi-line LF body lands in a CRLF file as CRLF -- no bare ``\\n``."""
    text = _issue_text(title="T", body="old body").replace("\n", "\r\n")
    rewritten = rewrite_body(text, "line one\nline two\nline three")
    assert "\n" not in rewritten.replace("\r\n", "")
    issue = parse_issue_text(rewritten, _path(tmp_path))
    assert issue.body == "line one\r\nline two\r\nline three"


def test_rewrite_body_normalizes_multiline_body_to_crlf_with_comments(
    tmp_path: Path,
) -> None:
    """Same normalization when the file carries a comments block, which is
    preserved byte-for-byte below the marker."""
    text = append_comment(
        _issue_text(title="T", body="old"), "keep me", timestamp="2026-09-17T00:00:00Z"
    ).replace("\n", "\r\n")
    rewritten = rewrite_body(text, "line one\nline two")
    assert "\n" not in rewritten.replace("\r\n", "")
    issue = parse_issue_text(rewritten, _path(tmp_path))
    assert issue.body == "line one\r\nline two"
    assert len(issue.comments) == 1
    assert "keep me" in issue.comments[0]


def test_rewrite_body_normalizes_crlf_body_to_lf(tmp_path: Path) -> None:
    """The inverse direction: a CRLF body lands in an LF file as LF."""
    text = _issue_text(title="T", body="old body")
    rewritten = rewrite_body(text, "line one\r\nline two\r\n")
    assert "\r" not in rewritten
    assert parse_issue_text(rewritten, _path(tmp_path)).body == "line one\nline two"


@pytest.mark.parametrize("marker", [COMMENTS_MARKER, _COMMENT_SEPARATOR])
def test_render_issue_text_rejects_reserved_markup(marker: str) -> None:
    with pytest.raises(IssueFileError, match="reserved markup"):
        render_issue_text(title="t", body=f"junk {marker} junk", labels=(), created="2026-11-25")


@pytest.mark.parametrize("marker", [COMMENTS_MARKER, _COMMENT_SEPARATOR])
def test_rewrite_body_rejects_reserved_markup(marker: str) -> None:
    """A marker in the body would be stored corrupted -- everything below it
    parses as comments, and no later rewrite_body can remove it."""
    text = _issue_text(title="T", body="old body")
    with pytest.raises(IssueFileError, match="reserved markup"):
        rewrite_body(text, f"junk {marker} junk")


def test_rewrite_body_requires_frontmatter() -> None:
    with pytest.raises(IssueFileError):
        rewrite_body("no frontmatter here", "x")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_rewrite_body_fence_at_eof_inserts_the_files_newline(tmp_path: Path, newline: str) -> None:
    """A closing ``---`` at EOF carries no line ending; the new body must
    land on its own line in the file's convention. Gluing it onto the fence
    (``---new body``) stores a file that no longer parses -- while
    ``edit_issue_body`` reports success."""
    text = newline.join(("---", 'title: "T"', "state: open", "---"))
    rewritten = rewrite_body(text, "new body")
    assert rewritten.endswith(f"---{newline}new body{newline}")
    issue = parse_issue_text(rewritten, _path(tmp_path))
    assert issue.body == "new body"
    assert issue.title == "T"
