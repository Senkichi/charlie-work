"""``parse_blockers`` dependency-declaration parsing: extraction, dedup, quoted-phrase and fenced-block self-block guards.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


# --- Issue #108: dependency gate tests --------------------------------------


def test_parse_blockers_extracts_single_blocker() -> None:
    """Test that parse_blockers extracts a single blocker from issue body."""
    from charlie_work.github import parse_blockers

    body = "This issue is blocked by #743"
    blockers = parse_blockers(body)
    assert blockers == [743]


def test_parse_blockers_extracts_multiple_blockers() -> None:
    """Test that parse_blockers extracts multiple blockers from issue body."""
    from charlie_work.github import parse_blockers

    body = "Blocked by #743, #744"
    blockers = parse_blockers(body)
    assert blockers == [743, 744]


def test_parse_blockers_handles_various_patterns() -> None:
    """Test that parse_blockers handles different declaration patterns."""
    from charlie_work.github import parse_blockers

    # Test "Depends on" pattern
    assert parse_blockers("Depends on #123") == [123]

    # Test "Blocked-by:" pattern
    assert parse_blockers("Blocked-by: #456") == [456]

    # Test case insensitivity
    assert parse_blockers("BLOCKED BY #789") == [789]
    assert parse_blockers("depends on #100") == [100]


def test_parse_blockers_returns_empty_for_no_blockers() -> None:
    """Test that parse_blockers returns empty list when no blockers found."""
    from charlie_work.github import parse_blockers

    assert parse_blockers("No blockers here") == []
    assert parse_blockers("") == []
    assert parse_blockers(None) == []


def test_parse_blockers_deduplicates() -> None:
    """Test that parse_blockers deduplicates blocker numbers."""
    from charlie_work.github import parse_blockers

    body = "Blocked by #123, #123, #456"
    blockers = parse_blockers(body)
    assert blockers == [123, 456]


def test_parse_blockers_ignores_downstream_reference_to_self() -> None:
    """Issue #159 regression: prose describing OTHER issues as blocked by
    THIS issue must not be misread as a self-referencing blocker declaration.

    Real trip case from issue #159's "## Dependencies" section: the sentence
    describes #168/#169/#170 as blocked by #159, not #159 declaring its own
    blocker. Naively matching "blocked by #N" anywhere in the text extracted
    159 and treated it as #159 self-declaring a blocker on itself.
    """
    from charlie_work.github import parse_blockers

    body = (
        "## Dependencies\n\n"
        "None — greenfield, no blockers. Downstream: #168 (fleet status), "
        "#169 (global concurrency budget), and #170 (fleet dispatch) all "
        "build on this registry and are blocked by #159.\n\n"
        "_Filed from the fleet-management & worker-supervision design._\n"
    )
    assert parse_blockers(body) == []


def test_parse_blockers_quoted_backtick_phrase_does_not_self_block() -> None:
    """Issue #1454 regression: an issue whose body quotes ANOTHER issue's
    blocker declaration inside a Markdown backtick code span must not be
    classified as blocked by the quoted number.

    Reproduces the #1927 incident shape: a bug report ABOUT the parser
    flapping on #887/#888 quoted their trigger phrase on its own line, with
    no preceding issue ref in the clause, so the old backward-only guard
    could not suppress it and the describing issue self-gated on #886.
    """
    from charlie_work.github import parse_blockers

    body = (
        "## Symptom\n\n"
        "The parser flaps on #887 and #888.\n\n"
        "Their bodies contain the trigger phrase:\n\n"
        "`blocked by #886`\n\n"
        "which the parser reads as a self-declaration.\n"
    )
    assert parse_blockers(body) == []


def test_parse_blockers_quoted_double_quote_phrase_does_not_self_block() -> None:
    """Issue #1454: a trigger phrase inside straight double quotes is quoted
    prose, not a self-declaration."""
    from charlie_work.github import parse_blockers

    body = (
        "## Symptom\n\n"
        'The parser sees the literal phrase "blocked by #886" in #887\'s '
        "body and misreads it as a self-declaration.\n"
    )
    assert parse_blockers(body) == []


def test_parse_blockers_forward_foreign_ref_does_not_self_block() -> None:
    """Issue #1454: a match whose clause carries another #NNN AFTER it (e.g.
    an issue-referencing parenthetical) describes that other issue, not this
    one. The old guard only looked backward and missed this."""
    from charlie_work.github import parse_blockers

    body = (
        "## Symptom\n\n"
        "The trigger phrase blocked by #886 (see #887) appears verbatim in "
        "the upstream body.\n"
    )
    assert parse_blockers(body) == []


def test_parse_blockers_genuine_declaration_still_gates() -> None:
    """Issue #1454 regression: a genuine first-person blocker declaration
    (the #887/#888 shape) must still gate. The quoted-phrase fix must not
    suppress real declarations."""
    from charlie_work.github import parse_blockers

    assert parse_blockers("This issue is blocked by #743") == [743]
    assert parse_blockers("Blocked by #743, #744") == [743, 744]
    assert parse_blockers("Depends on #123") == [123]
    assert parse_blockers("Blocked-by: #456") == [456]
    # Genuine declaration with surrounding prose but no foreign issue ref.
    body = "## Summary\n\nFix the parser.\n\nBlocked by #886\n"
    assert parse_blockers(body) == [886]


def test_parse_blockers_stray_backtick_elsewhere_does_not_swallow_declaration() -> None:
    """Issue #1454 rework: an unrelated/unbalanced backtick ELSEWHERE in the
    body must not pair with a later backtick to form a code span that
    envelopes a genuine 'Blocked by #NNN' declaration and silently drop it.

    The body below has a stray opening backtick on the first line and a
    closing backtick on the last line. Against the whole-document span scan
    (the pre-rework guard 1) the regex ``(`+)(.+?)(\\1)`` with re.DOTALL
    matches one span whose content runs from "broken thing." through
    "Blocked by #159" through "See also ", so the declaration is
    misclassified as quoted prose and dropped -- a false negative. Scoping
    the span search to the containing clause (bounded by newlines) leaves
    the clause "Blocked by #159" with no backticks, so the declaration gates.
    """
    from charlie_work.github import parse_blockers

    body = "TODO: fix the `broken thing.\nBlocked by #159\nSee also `foo`.\n"
    assert parse_blockers(body) == [159]


def test_parse_blockers_stray_double_quote_elsewhere_does_not_swallow_declaration() -> None:
    """Issue #1454 rework: an unrelated/unbalanced straight double quote
    ELSEWHERE in the body must not pair with a later quote to envelope a
    genuine declaration. Same false-negative shape as the backtick case:
    ``"([^"]*)"`` matches from the first quote to the next, swallowing the
    declaration line in between when scanned over the whole document.
    Scoping to the clause leaves "Blocked by #743" with no quotes, so it
    gates.
    """
    from charlie_work.github import parse_blockers

    body = 'The error was "connection refused.\nBlocked by #743\nThen it said "done".\n'
    assert parse_blockers(body) == [743]


def test_parse_blockers_fenced_code_block_does_not_self_gate() -> None:
    """Issue #1454 rework round 2: a 'Blocked by #NNN' line inside a real
    multi-line triple-backtick fenced code block (fence markers on separate
    lines from the content) must NOT self-gate.

    The clause-scoped inline span guard (round 1) cannot detect this: clause
    bounds break on newlines, so the fenced content line ``blocked by #886``
    is its own clause with no fence markers in it, and the declaration is
    misclassified as a genuine self-declaration. The fenced-block check runs
    against the full document with absolute offsets and suppresses it.
    """
    from charlie_work.github import parse_blockers

    body = (
        "## Symptom\n\n"
        "The upstream issue's body contains:\n\n"
        "```python\n"
        "blocked by #886\n"
        "```\n\n"
        "which the parser used to misread as a self-declaration.\n"
    )
    assert parse_blockers(body) == []


def test_parse_blockers_fenced_code_block_tilde_fence_does_not_self_gate() -> None:
    """Issue #1454 rework round 2: ``~~~`` fences are equivalent to triple-
    backtick fences in CommonMark and must be detected the same way."""
    from charlie_work.github import parse_blockers

    body = "## Example\n\n~~~\nblocked by #886\n~~~\n"
    assert parse_blockers(body) == []


def test_parse_blockers_fenced_block_with_language_tag_does_not_self_gate() -> None:
    """Issue #1454 rework round 2: an opening fence carrying an info string
    (e.g. ```` ```bash ````) must still be recognized as a fence."""
    from charlie_work.github import parse_blockers

    body = "## Repro\n\n```bash\n$ echo 'blocked by #886'\n```\n"
    assert parse_blockers(body) == []


def test_parse_blockers_genuine_declaration_outside_fenced_block_still_gates() -> None:
    """Issue #1454 rework round 2: a genuine declaration on a line OUTSIDE a
    fenced block must still gate. The fenced-block guard must not over-suppress
    real declarations that merely share a document with a fenced block."""
    from charlie_work.github import parse_blockers

    body = "## Summary\n\nFix the parser.\n\n```python\nblocked by #886\n```\n\nBlocked by #743\n"
    assert parse_blockers(body) == [743]


# --- Issue #1847: heading-list blocker sections, colon inline form, ---------
# --- blockquote exclusion ---------------------------------------------------


def test_parse_blockers_heading_section_list_items() -> None:
    """Issue #1847: a '## Blocked by' heading opens a blocker section whose
    list items contribute the issue reference they start with. Later refs in
    the same item are ignored, not fatal — the foreign-issue-ref guard that
    voids an inline clause does not apply per item."""
    from charlie_work.github import parse_blockers

    body = "## Blocked by\n- #12 (schema migration)\n- #14 after #9 lands\n"
    assert parse_blockers(body) == [12, 14]


def test_parse_blockers_heading_section_multi_ref_item() -> None:
    """Issue #1847 rework round 1: a list item that leads with several issue
    refs joined by ',' or 'and' contributes ALL of them. Keeping only the
    first silently drops the rest — the issue is freed while those blockers
    are still open, and the prose-only park never sees it (the remaining
    refs made the item readable, so it was never flagged unreadable)."""
    from charlie_work.github import detect_prose_only_dependencies, parse_blockers

    assert parse_blockers("## Blocked by\n- #12, #13\n") == [12, 13]
    assert parse_blockers("## Blocked by\n- #12 and #13\n") == [12, 13]
    assert parse_blockers("## Blocked by\n- #12, #13, and #14\n") == [12, 13, 14]
    # A leading run may still carry annotation prose after it.
    assert parse_blockers("## Blocked by\n- #12, #13 (schema migrations)\n") == [12, 13]
    # A bare line that is only a run of references counts the same way.
    assert parse_blockers("## Blocked by\n#20, #21\n") == [20, 21]
    # A ref after non-separator prose is annotation, not a list member —
    # '- #14 after #9 lands' still yields 14 only.
    assert parse_blockers("## Blocked by\n- #14 after #9 lands\n") == [14]
    # A multi-ref item is readable: no prose-only park.
    assert detect_prose_only_dependencies("## Blocked by\n- #12, #13\n") is False


def test_parse_blockers_heading_section_level_3_with_colon() -> None:
    """Issue #1847: a level-3 'Depends on:' heading (optional trailing colon)
    opens a blocker section."""
    from charlie_work.github import parse_blockers

    body = "### Depends on:\n- #7\n"
    assert parse_blockers(body) == [7]


def test_parse_blockers_heading_section_all_levels_and_variants() -> None:
    """Issue #1847: heading levels 1-6, up to three leading spaces, a hyphen
    between the words, and case-insensitivity all open a blocker section."""
    from charlie_work.github import parse_blockers

    for hashes in ("#", "##", "###", "####", "#####", "######"):
        assert parse_blockers(f"{hashes} Blocked by\n- #5\n") == [5]
    assert parse_blockers("   ## Blocked-by\n- #6\n") == [6]
    assert parse_blockers("## blocked by\n- #7\n") == [7]
    # Four leading spaces is an indented code block, not a heading.
    assert parse_blockers("    ## Blocked by\n- #8\n") == []


def test_parse_blockers_heading_section_none_sentinel() -> None:
    """Issue #1847: an item or line starting with 'None'/'n/a' contributes
    nothing and is not unreadable."""
    from charlie_work.github import detect_prose_only_dependencies, parse_blockers

    body = "### Depends on:\nNone (can start immediately)\n- #8\n"
    assert parse_blockers(body) == [8]
    assert detect_prose_only_dependencies(body) is False

    body = "## Blocked by\n- n/a\n- #9\n"
    assert parse_blockers(body) == [9]
    assert detect_prose_only_dependencies(body) is False


def test_parse_blockers_heading_section_all_list_markers() -> None:
    """Issue #1847: '-', '*', '+' and ordered 'N.' list markers each start an
    item that contributes its leading issue reference."""
    from charlie_work.github import parse_blockers

    body = "## Blocked by\n- #1\n* #2\n+ #3\n1. #4\n"
    assert parse_blockers(body) == [1, 2, 3, 4]


def test_parse_blockers_heading_section_bare_ref_line() -> None:
    """Issue #1847: a bare line that is only an issue reference counts the
    same way as a list item."""
    from charlie_work.github import parse_blockers

    body = "## Blocked by\n#20\n"
    assert parse_blockers(body) == [20]


def test_parse_blockers_heading_section_inline_form_parses_normally() -> None:
    """Issue #1847: inline forms inside a heading section parse exactly as
    they do elsewhere, so a prose line carrying 'Depends on #N' contributes
    and is not unreadable."""
    from charlie_work.github import detect_prose_only_dependencies, parse_blockers

    body = "## Blocked by\n- #1\nSome prose. Depends on #9.\n"
    assert parse_blockers(body) == [1, 9]
    assert detect_prose_only_dependencies(body) is False


def test_parse_blockers_unreadable_section_item_flags_prose_only() -> None:
    """Issue #1847: a blocker-section item that is neither a same-repo issue
    reference nor a none-sentinel (a URL, an owner/repo reference, free
    prose) makes detect_prose_only_dependencies return True so the issue is
    parked for a human instead of silently freed."""
    from charlie_work.github import detect_prose_only_dependencies, parse_blockers

    body = "## Blocked by\n- https://github.com/other/repo/issues/7\n"
    assert parse_blockers(body) == []
    assert detect_prose_only_dependencies(body) is True

    body = "## Blocked by\n- owner/repo#7\n"
    assert parse_blockers(body) == []
    assert detect_prose_only_dependencies(body) is True

    body = "## Depends on\nSome free prose about dependencies.\n"
    assert parse_blockers(body) == []
    assert detect_prose_only_dependencies(body) is True


def test_parse_blockers_parent_section_never_contributes() -> None:
    """Issue #1847: a 'Parent' heading never opens a blocker section, so its
    contents contribute nothing and are not unreadable."""
    from charlie_work.github import detect_prose_only_dependencies, parse_blockers

    body = "## Blocked by\n- #3\n\n## Parent\n#20\n"
    assert parse_blockers(body) == [3]
    assert detect_prose_only_dependencies(body) is False


def test_parse_blockers_inline_colon_form() -> None:
    """Issue #1847: the inline 'Blocked by' pattern accepts an optional colon
    after 'by' (the shape Matt's GitHub fallback writes)."""
    from charlie_work.github import parse_blockers

    assert parse_blockers("Blocked by: #3, #4") == [3, 4]
    assert parse_blockers("blocked by:#8") == [8]


def test_parse_blockers_blockquoted_inline_ignored() -> None:
    """Issue #1847: a line whose first non-space character is '>' is a quoted
    reply and contributes no blockers, in any form."""
    from charlie_work.github import parse_blockers

    assert parse_blockers("> Blocked by #5") == []
    assert parse_blockers("  > Depends on #6") == []
    # A '>' later in the line is not a blockquote marker.
    assert parse_blockers("see docs > blocked by #7") == [7]


def test_parse_blockers_blockquote_lines_inside_section_contribute_nothing() -> None:
    """Issue #1847: blockquoted items inside a blocker section are quoted
    reply content — they contribute nothing and are not unreadable."""
    from charlie_work.github import detect_prose_only_dependencies, parse_blockers

    body = "## Blocked by\n- #1\n> - #2\n> #3\n"
    assert parse_blockers(body) == [1]
    assert detect_prose_only_dependencies(body) is False


def test_parse_blockers_heading_section_inside_fenced_block_ignored() -> None:
    """Issue #1847: a heading-form blocker section quoted inside a fenced
    code block opens no section and contributes nothing."""
    from charlie_work.github import detect_prose_only_dependencies, parse_blockers

    body = "## Blocked by\n- #1\n```text\n## Blocked by\n- #5\n```\n"
    assert parse_blockers(body) == [1]
    assert detect_prose_only_dependencies(body) is False


def test_parse_blockers_section_closed_by_next_heading() -> None:
    """Issue #1847: a blocker section runs to the next heading, so items
    after it are not parsed as section content."""
    from charlie_work.github import parse_blockers

    body = "## Blocked by\n- #5\n\n## Notes\n- #6\n"
    assert parse_blockers(body) == [5]


def test_parse_blockers_section_closed_by_fenced_block() -> None:
    """Issue #1847: a blocker section ends at the start of a fenced block;
    items after the fence are outside the section."""
    from charlie_work.github import parse_blockers

    body = "## Blocked by\n- #5\n```\n- #6\n```\n- #7\n"
    assert parse_blockers(body) == [5]


def test_parse_blockers_section_blank_line_does_not_close() -> None:
    """Issue #1847: only the next heading or a fenced block closes a blocker
    section — blank lines do not."""
    from charlie_work.github import parse_blockers

    body = "## Blocked by\n- #5\n\n- #6\n"
    assert parse_blockers(body) == [5, 6]


def test_parse_blockers_non_blocker_heading_does_not_open_section() -> None:
    """Issue #1847: a heading whose text merely CONTAINS 'blocked by' (rather
    than being exactly 'Blocked by'/'Depends on' plus an optional colon)
    opens no section."""
    from charlie_work.github import detect_prose_only_dependencies, parse_blockers

    body = "## Blocked by\n- #2\n\n## Not blocked by anything\n- #5\n"
    assert parse_blockers(body) == [2]
    assert detect_prose_only_dependencies(body) is False
