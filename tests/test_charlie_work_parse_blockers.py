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
