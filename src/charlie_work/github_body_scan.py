"""Issue/PR body-text scanning: mentions, blocker declarations, dependency prose.

Extracted from ``github.py`` (issue #1819 rework): ``github.py`` sits over the
800-line file-size cap and the ratchet (``tests/test_file_size_ratchet.py``)
fails any growth past its recorded high-water mark, so the body-scanning
domain lives here as its own module. ``github.py`` re-exports the public
surface (``issue_numbers_mentioned_by_pr``, ``parse_blockers``,
``detect_prose_only_dependencies``) so existing
``from charlie_work.github import ...`` callers and tests are unchanged.

Every scanner in this module shares ONE fenced-code-block model:
``_fenced_block_ranges`` — a line-based CommonMark-ish scan where a closing
fence must be its own line, of the same fence character and at least the same
length as the opening fence. ``_strip_fenced_blocks`` removes those blocks
(closing fence line included) before pattern matching; ``_inside_fenced_block``
answers point queries against the same ranges. A regex that pairs the two
nearest triple-backtick runs is the naive model issue #1819 removed — a
fenced block whose own content contains a triple-backtick substring desyncs
that pairing and leaks fenced text into the scan.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# GitHub repository-visibility designators that qualify an ``issue #N``
# mention as belonging to a different repo's tracker (issue #1803). The
# observed false positive was a jobcannon docs PR that rewrote bare
# ``issue #391`` references to ``private issue #391`` — "the private repo"
# is jobcannon's private source checkout, not the dispatching repo, so the
# qualified mention never referred to the flagged issue. ``internal`` is
# the other non-public GraphQL ``RepositoryVisibility`` value and reads
# the same way. ``public`` is deliberately absent: in a public repo "the
# public issue" can refer to this tracker itself, so counting it fails
# toward flagging — the safe direction for an advisory gate.
_FOREIGN_REPO_VISIBILITY_QUALIFIERS = frozenset({"private", "internal"})

# Explicit issue-reference pattern: the word "issue" or "issues" followed by
# an optional space and then "#N".  This is deliberately narrower than a bare
# ``#N`` so a merged PR that mentions a different PR (e.g. "PR #181") is not
# mistaken for an issue reference.  Closing-keyword binding is handled by
# ``linked_issue_number``.
#
# Issue #1803: the match also captures an *immediate qualifier* in either of
# two positions so a mention explicitly scoped to another repo can be
# suppressed:
#
# * a bare name or ``owner/repo`` slug directly before the word "issue"
#   (``private issue #N``, ``sibling-repo issue #N``, ``o/r issue #N``);
# * an ``owner/repo`` slug between "issue" and the hash (``issue o/r#N``)
#   — the same qualifier shape ``closing_reference._CLOSING_LINE_RE``
#   accepts on ``Closes`` lines.
#
# Group 1 captures an optional determiner immediately before the bare-word
# qualifier (``a private issue #N``, ``the internal issue #N``, ``this
# sibling-repo issue #N``). A determiner marks the word as an ordinary
# adjective in a noun phrase about THIS tracker — English determiners are a
# closed grammatical class, so a literal alternation is correct here, not a
# fleet-managed list. ``issue_numbers_mentioned_by_pr`` de-qualifies a
# bare-word qualifier under a determiner so organic prose that
# coincidentally names a managed repo or visibility designator cannot
# silently suppress a genuine same-repo mention. ``owner/repo`` slug
# qualifiers are canonical cross-repo syntax in any construction and are
# never de-qualified.
_ISSUE_MENTION_DETERMINERS_RE = (
    r"(an|a|the|this|that|these|those|our|my|its|their|your|any|each|every|another|no|such)"
)
_ISSUE_MENTION_RE = re.compile(
    rf"\b(?:{_ISSUE_MENTION_DETERMINERS_RE}\s+)?"
    r"(?:([\w.-]+(?:/[\w.-]+)?)\s+)?issues?\s*(?:([\w.-]+/[\w.-]+))?#(\d+)\b",
    flags=re.IGNORECASE,
)
# Stripped before matching to cut a concrete false-positive class: quoted
# reply text (e.g. an email-style ``> see issue #123`` blockquote). Fenced
# code samples are removed by ``_strip_fenced_blocks`` instead (issue
# #1819), which shares the line-based fence model ``parse_blockers`` uses
# rather than pairing the two nearest triple-backtick runs — a block whose
# own content contains a ``` substring desynced the naive regex and leaked
# fenced text into the scan.
_BLOCKQUOTE_LINE_RE = re.compile(r"^[ \t]*>.*$", flags=re.MULTILINE)


def _mention_qualifier_is_foreign(
    qualifier: str | None,
    *,
    current_repo: str | None,
    other_repo_names: Iterable[str],
) -> bool:
    """True when an ``issue #N`` mention's immediate qualifier names another repo.

    ``qualifier`` is whichever of ``_ISSUE_MENTION_RE``'s two qualifier
    groups matched (``None`` for an unqualified mention, which is never
    foreign — the pre-#1803 behaviour is preserved for plain ``issue #N``).
    The caller de-qualifies a bare-word qualifier sitting under a
    determiner (``a private issue`` reads as prose, not repo scoping) to
    ``None`` before calling, so this function never sees the determiner —
    by the time a bare word reaches here it is already in qualifier
    position.

    An ``owner/repo`` slug qualifier is GitHub's own cross-repo reference
    syntax — the writer explicitly named which repo the number belongs to.
    It is foreign unless it is exactly ``current_repo``; when our own slug
    cannot be resolved (``current_repo is None``) we cannot confirm it is
    ours, and a qualified reference failing toward *not* flagging is the
    safe direction for this advisory gate (an unqualified mention still
    flags).

    A bare-word qualifier is foreign when it names another *managed* repo
    (``other_repo_names``, derived from the fleet registry by the caller —
    never a hardcoded list) or is a GitHub repo-visibility designator
    (``private``/``internal`` — see
    ``_FOREIGN_REPO_VISIBILITY_QUALIFIERS``). The dispatching repo's own
    name is checked first so a self-named qualifier (``jobcannon issue
    #N`` inside jobcannon, or a repo literally named "private") still
    counts.
    """
    if not qualifier:
        return False
    q = qualifier.lower()
    if "/" in qualifier:
        return current_repo is None or q != current_repo.lower()
    current_name = current_repo.rsplit("/", 1)[-1].lower() if current_repo else None
    if current_name is not None and q == current_name:
        return False
    return q in _FOREIGN_REPO_VISIBILITY_QUALIFIERS or q in {
        str(n).lower() for n in other_repo_names
    }


def issue_numbers_mentioned_by_pr(
    pr: dict[str, Any],
    *,
    current_repo: str | None = None,
    other_repo_names: Iterable[str] = (),
) -> set[int]:
    """Return issue numbers loosely referenced by a PR's title/body — advisory only.

    Matches the literal phrase ``issue #N`` / ``issues #N`` (case-insensitive,
    with or without a space between the word and the hash), after stripping
    fenced code blocks and blockquoted lines, and skipping matches inside
    inline runs of 3+ backticks (`` ```...``` `` — the shape the pre-#1819
    regex incidentally stripped; single-backtick spans are not suppressed).
    The fenced-block exclusion runs on
    ``_fenced_block_ranges`` — the same line-based fence model
    ``parse_blockers`` uses (issue #1819). This is a strict subset of
    GitHub's issue-reference syntax: it does not treat a bare ``#N`` (which
    could be a PR number) as an issue reference, and it does not treat
    closing keywords like ``Fixes #N`` as any more than a reference.

    Issue #1803: a match whose *immediate qualifier* names another repo is
    suppressed — ``private issue #N`` / ``internal issue #N`` (GitHub
    repo-visibility designators), a qualifier naming another managed repo
    (``other_repo_names``), or an ``owner/repo`` slug qualifier
    (``owner/repo issue #N``, ``issue owner/repo#N``) that is not exactly
    ``current_repo``. Qualified mentions are explicit about which tracker
    the number belongs to; treating "not proven to be ours" as coverage
    was the absence-of-disproof defect behind the jobcannon #377/#391
    false escalations.

    This is looser than ``linked_issue_number``'s hijack-safety guarantee —
    phrases like "unlike issue #N", "follow-up to issue #N", or a collision
    with another repo's issue #N in the same text all still match, and there
    is no reliable lexical way to rule those out. Callers MUST treat a match
    as advisory only: it may be used to flag an issue for human review or
    exclude it from automation, but it must NEVER by itself authorize closing
    an issue or any other lifecycle-mutating action. Only ``linked_issue_number``
    (same-repo branch-prefix or closing-action verb) may authorize that.
    """
    text = f"{pr.get('title', '')}\n{pr.get('body', '')}"
    text = _strip_fenced_blocks(text)
    text = _BLOCKQUOTE_LINE_RE.sub("", text)
    found: set[int] = set()
    for match in _ISSUE_MENTION_RE.finditer(text):
        # A mention inside an inline `` ```...``` `` run (3+ backticks) is
        # quoted example text, not a reference — the incidental suppression
        # the old nearest-pair regex provided and the only inline shape
        # issue #1819's scope covers. Single-backtick code spans are NOT
        # suppressed (PR #1904 review): a ``.`` inside one is a clause
        # boundary, so the span's closing backtick lands inside a later
        # clause, pairs with the next span's opener, and envelopes a genuine
        # mention — 16 of 1,309 real bodies lost one. Clause-scoped like
        # ``parse_blockers``' guard 1b so a stray run elsewhere cannot pair
        # across the document and swallow a genuine mention.
        clause_start, clause_end = _clause_bounds(text, match.start(), match.end())
        if _inside_inline_fence_span(
            text[clause_start:clause_end],
            match.start() - clause_start,
            match.end() - clause_start,
        ):
            continue
        # The post-"issue" slug (group 3) wins over a bare-word qualifier
        # (group 2): it is GitHub's canonical ``owner/repo#N`` reference
        # syntax and sits closest to the number, so in ``see issue
        # owner/repo#7`` the scoping qualifier is ``owner/repo``, not the
        # prose word "see".
        qualifier = match.group(3) or match.group(2)
        # A bare-word qualifier under a determiner (group 1) reads as an
        # ordinary adjective phrase about THIS tracker — "a private
        # issue", "the internal issue", "this sibling-repo issue" — not a
        # repo-scoping noun adjunct. De-qualify it so an ordinary word
        # that coincidentally equals a managed repo name or visibility
        # designator cannot silently suppress a genuine same-repo mention.
        # ``owner/repo`` slug qualifiers are canonical cross-repo syntax
        # in any construction and are never de-qualified; a suppressed
        # mention fails toward flagging, the safe direction for this
        # advisory gate, so the boundary is drawn at the determiner.
        if match.group(1) is not None and qualifier and "/" not in qualifier:
            qualifier = None
        if _mention_qualifier_is_foreign(
            qualifier,
            current_repo=current_repo,
            other_repo_names=other_repo_names,
        ):
            continue
        found.add(int(match.group(4)))
    return found


# Blocker declaration patterns for dependency gate
# Case-insensitive patterns: "Blocked by #N", "Blocked by: #N", "Depends on #N",
# "Blocked-by: #N". Handles comma-separated lists like "Blocked by #743, #744".
# The optional colon after "blocked by" (issue #1847) covers the shape Matt's
# GitHub fallback writes.
_BLOCKER_PATTERNS = [
    re.compile(r"blocked\s+by\s*:?\s*#\d+(?:\s*,\s*#\d+)*", flags=re.IGNORECASE),
    re.compile(r"depends\s+on\s+#\d+(?:\s*,\s*#\d+)*", flags=re.IGNORECASE),
    re.compile(r"blocked-by:\s*#\d+(?:\s*,\s*#\d+)*", flags=re.IGNORECASE),
]

_CLAUSE_BOUNDARY_CHARS = ".!?\n"
_ISSUE_REF = re.compile(r"#\d+")

# Markdown backtick code span: an opening run of backticks, content, and a
# closing run of the SAME length. Capturing group 2 is the span content.
_CODE_SPAN_RE = re.compile(r"(`+)(.+?)(\1)", flags=re.DOTALL)
# Inline run of 3+ backticks (`` ```...``` ``) — the shape the pre-#1819
# nearest-pair `` ```.*?``` `` regex incidentally stripped along with real
# fenced blocks. ``issue_numbers_mentioned_by_pr`` keeps suppressing these
# but deliberately does NOT suppress single-backtick code spans (PR #1904
# review): a ``.`` inside an ordinary span (`` `worktree.py` ``) is a clause
# boundary, so the span's closing backtick lands inside the next clause and
# pairs with a later span's opener, enveloping a genuine prose mention.
_INLINE_FENCE_SPAN_RE = re.compile(r"(`{3,})(.+?)(\1)", flags=re.DOTALL)
# Balanced straight-double-quote span. Group 1 is the quoted content.
_DOUBLE_QUOTE_SPAN_RE = re.compile(r'"([^"]*)"')
# Opening fence of a fenced code block: a line beginning with a run of 3+
# backticks or tildes (optionally followed by an info string). CommonMark
# allows up to 3 leading spaces; we tolerate any leading whitespace.
_FENCE_OPEN_RE = re.compile(r"^[ \t]*([`~]{3,})")

# Issue #1847 — heading-list blocker sections ("## Blocked by\n- #N").
# ATX heading: up to three leading spaces, 1-6 '#' characters, then optional
# whitespace-separated text. CommonMark requires whitespace or end-of-line
# after the opening run, so "###foo" is not a heading.
_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+(.*?))?[ \t]*$")
# Optional CommonMark closing sequence of '#'s at the end of heading text
# ("## Blocked by ##" -> text "Blocked by").
_HEADING_CLOSING_RUN_RE = re.compile(r"[ \t]+#+[ \t]*$")
# Heading text that opens a blocker section: "Blocked by" or "Depends on"
# (case-insensitive, space or hyphen between the words, optional trailing colon).
_BLOCKER_HEADING_TEXT_RE = re.compile(
    r"(?:blocked[ \t-]+by|depends[ \t-]+on):?", flags=re.IGNORECASE
)
# List-item marker: '-', '*', '+' or an ordered 'N.' followed by whitespace.
_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+\.)[ \t]+(.*)$")
# An item or bare line that declares "no blockers" ("None", "n/a").
_NONE_SENTINEL_RE = re.compile(r"(?:none|n/a)\b", flags=re.IGNORECASE)
# A leading run of same-repo issue refs joined by ',' or 'and' separators
# ("#12, #13", "#12 and #13", "#12, #13, and #14"). A ref followed by
# non-separator prose ends the run ("#14 after #9 lands" -> "#14"), so
# trailing annotation text in an item is ignored rather than misparsed.
_BLOCKER_ITEM_REF_RUN_RE = re.compile(
    r"#\d+(?:[ \t]*(?:,[ \t]*(?:and[ \t]+)?|and[ \t]+)#\d+)*",
    flags=re.IGNORECASE,
)


def _inside_code_span(text: str, start: int, end: int) -> bool:
    """True if the [start, end) range falls inside a Markdown backtick code span."""
    for m in _CODE_SPAN_RE.finditer(text):
        if m.start(2) <= start and end <= m.end(2):
            return True
    return False


def _inside_inline_fence_span(text: str, start: int, end: int) -> bool:
    """True if the [start, end) range falls inside an inline run of 3+ backticks.

    Same shape as :func:`_inside_code_span` but restricted to
    ``_INLINE_FENCE_SPAN_RE`` (3+ backticks) — the only inline suppression
    ``issue_numbers_mentioned_by_pr`` keeps from the pre-#1819 regex.
    """
    for m in _INLINE_FENCE_SPAN_RE.finditer(text):
        if m.start(2) <= start and end <= m.end(2):
            return True
    return False


def _inside_quoted_span(text: str, start: int, end: int) -> bool:
    """True if the [start, end) range falls inside a straight-double-quote span."""
    for m in _DOUBLE_QUOTE_SPAN_RE.finditer(text):
        if m.start(1) <= start and end <= m.end(1):
            return True
    return False


def _fenced_block_ranges(text: str) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` char-offset ranges of fenced code blocks.

    A fenced block starts with a line beginning with a run of 3+ backticks or
    tildes (optionally followed by an info string, e.g. ```` ```python ````)
    and ends at the next line beginning with a closing fence of the same
    character and at least the same length. An unclosed fence runs to the end
    of the text. Each returned range spans from the start of the opening fence
    line up to (excluding) the closing fence line, so any content line between
    the fences is contained in the range.
    """
    ranges: list[tuple[int, int]] = []
    pos = 0
    in_fence = False
    fence_char = ""
    fence_len = 0
    block_start = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(" \t")
        if not in_fence:
            m = _FENCE_OPEN_RE.match(stripped)
            if m:
                fence_char = m.group(1)[0]
                fence_len = len(m.group(1))
                block_start = pos
                in_fence = True
        else:
            close_m = re.match(
                rf"{re.escape(fence_char)}{{{fence_len},}}[ \t]*$",
                stripped.rstrip("\r\n"),
            )
            if close_m:
                # Range covers opening fence line through last content line;
                # the closing fence line itself is excluded.
                ranges.append((block_start, pos))
                in_fence = False
                fence_char = ""
                fence_len = 0
        pos += len(line)
    if in_fence:
        # Unclosed fence runs to end of text.
        ranges.append((block_start, len(text)))
    return ranges


def _inside_fenced_block(text: str, start: int, end: int) -> bool:
    """True if the ``[start, end)`` range falls inside a fenced code block.

    Unlike inline code spans, fenced blocks are multi-line Markdown constructs
    whose opening and closing fence markers sit on separate lines from the
    content. Clause bounds (which break on newlines) therefore cannot detect
    them -- the content line is its own clause with no fence markers in it --
    so this check runs against the full document with absolute offsets, not
    the clause substring.
    """
    for r_start, r_end in _fenced_block_ranges(text):
        if r_start <= start and end <= r_end:
            return True
    return False


def _strip_fenced_blocks(text: str) -> str:
    """Return ``text`` with every fenced code block removed.

    Single fence model shared by the body scanners (issue #1819): the ranges
    come from :func:`_fenced_block_ranges` — the same line-based
    CommonMark-ish scanner :func:`parse_blockers` trusts — so a closing
    fence must be its own line of the same fence character and at least the
    same length, and a triple-backtick run inside a block's own content
    cannot desync the pairing the way a nearest-run regex does. The closing
    fence line is removed with the block so a fence re-scan of the result
    (e.g. ``_scan_blocker_sections`` inside ``detect_prose_only_dependencies``)
    cannot re-interpret an orphaned closer as a new opening fence.
    """
    out: list[str] = []
    prev = 0
    for start, end in _fenced_block_ranges(text):
        newline = text.find("\n", end)
        out.append(text[prev:start])
        prev = len(text) if newline == -1 else newline + 1
    out.append(text[prev:])
    return "".join(out)


def _clause_bounds(text: str, match_start: int, match_end: int) -> tuple[int, int]:
    """Return the (start, end) offsets of the sentence/line containing a match.

    Bounded by the closest preceding AND following sentence terminator
    (".", "!", "?") or line break, so each bullet/sentence is judged
    independently. The boundary characters themselves are excluded from the
    returned range.
    """
    start_boundary = max(text.rfind(ch, 0, match_start) for ch in _CLAUSE_BOUNDARY_CHARS)
    start = start_boundary + 1 if start_boundary != -1 else 0
    end_candidates = [text.find(ch, match_end) for ch in _CLAUSE_BOUNDARY_CHARS]
    end_candidates = [p for p in end_candidates if p != -1]
    end = min(end_candidates) if end_candidates else len(text)
    return start, end


def _is_blockquote_line(line: str) -> bool:
    """True if the line's first non-space character is ``>`` (a Markdown blockquote)."""
    return line.lstrip(" \t").startswith(">")


def _is_thematic_break(line: str) -> bool:
    """True if the line is a CommonMark thematic break (3+ of ``-``, ``_`` or ``*``)."""
    stripped = re.sub(r"[ \t]", "", line.strip())
    return len(stripped) >= 3 and stripped[0] in "-_*" and len(set(stripped)) == 1


def _scan_blocker_sections(text: str) -> tuple[list[int], bool]:
    """Scan ``Blocked by``/``Depends on`` heading sections for issue refs.

    Issue #1847: issue bodies written by the mattpocock-skills ``/to-tickets``
    skill (and some written by hand) declare blockers as a Markdown section —
    a heading plus a list — rather than an inline sentence. A heading of any
    level whose text is "Blocked by" or "Depends on" (case-insensitive, a
    space or hyphen between the words, optional trailing colon) opens a
    section that runs to the next heading or the start of a fenced code
    block. Inside it, each list item contributes the leading run of issue
    references it starts with — refs joined by ``,`` or ``and`` separators
    all count ("- #12, #13" -> [12, 13]), while refs after non-separator
    prose are ignored, not fatal ("- #14 after #9 lands" -> [14]; the
    foreign-issue-ref guard that voids an inline clause does not apply per
    item). A line that is only a run of issue references counts the same
    way.
    An item or line starting with "None"/"n/a" contributes nothing, and a
    ``Parent`` section never opens here so it never contributes. Inline
    forms inside the section are left to the caller's normal inline scan —
    they parse exactly as they do elsewhere. Blockquote lines contribute
    nothing in any form.

    Returns ``(refs, unreadable)`` where ``unreadable`` is True when a
    blocker section holds an item that is neither a same-repo issue
    reference nor a none-sentinel nor an inline declaration (a URL, an
    ``owner/repo`` reference, free prose) — the signal
    :func:`detect_prose_only_dependencies` uses to park the issue for a
    human instead of silently freeing it.
    """
    fenced = _fenced_block_ranges(text)
    refs: list[int] = []
    unreadable = False
    in_section = False
    fence_idx = 0
    pos = 0
    for line in text.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        while fence_idx < len(fenced) and fenced[fence_idx][1] <= line_start:
            fence_idx += 1
        if fence_idx < len(fenced) and fenced[fence_idx][0] <= line_start:
            # Inside a fenced block. The range starts at the opening fence
            # line, so reaching it closes any open blocker section; content
            # lines are quoted prose and never parsed.
            in_section = False
            continue
        heading = _HEADING_RE.match(line.rstrip("\r\n"))
        if heading is not None:
            heading_text = _HEADING_CLOSING_RUN_RE.sub("", heading.group(1) or "")
            in_section = bool(_BLOCKER_HEADING_TEXT_RE.fullmatch(heading_text.strip()))
            continue
        if not in_section or _is_blockquote_line(line) or _is_thematic_break(line):
            continue
        item = _LIST_ITEM_RE.match(line)
        candidate = item.group(1).strip() if item is not None else line.strip()
        if not candidate:
            continue
        ref_run = _BLOCKER_ITEM_REF_RUN_RE.match(candidate)
        if ref_run is not None and (item is not None or ref_run.end() == len(candidate)):
            # A list item contributes its leading run of references —
            # every ','/'and'-joined ref counts, so '- #12, #13' yields
            # both rather than silently dropping #13. A bare (non-list)
            # line counts only when it IS just a run of references.
            refs.extend(int(m.group(0)[1:]) for m in _ISSUE_REF.finditer(ref_run.group(0)))
            continue
        if _NONE_SENTINEL_RE.match(candidate) or any(
            p.search(candidate) for p in _BLOCKER_PATTERNS
        ):
            continue
        unreadable = True
    return refs, unreadable


def parse_blockers(text: str) -> list[int]:
    """Parse blocker issue numbers from issue body text.

    Returns a list of issue numbers declared as blockers using patterns like:
    - "Blocked by #N" (an optional colon after "by" is accepted, issue #1847)
    - "Depends on #N"
    - "Blocked-by: #N"
    - a "Blocked by"/"Depends on" Markdown heading section whose list items
      each start with ``#N`` — a leading run of ``,``/``and``-separated refs
      all count ("- #12, #13" -> [12, 13]) — (issue #1847; see
      :func:`_scan_blocker_sections`)

    Handles comma-separated lists (e.g., "Blocked by #743, #744").

    A match is only treated as the CURRENT issue declaring its own blocker
    when it reads as a first-person declaration about THIS issue. Three guards
    enforce that, in order of structural strength:

    1. **Quoted/code exclusion** — a match falling inside a Markdown backtick
       code span, a triple-backtick (or ``~~~``) fenced code block, or a
       straight-double-quote span is prose quoting another issue's blocker
       declaration, not a self-declaration. This is the fix for issue #1454:
       an issue describing another issue's blocker phrase (backticked, quoted,
       or parenthetically annotated) must not self-gate. The inline span
       search is scoped to the containing clause (see guard 2) so an unrelated
       stray backtick or quote ELSEWHERE in the body cannot pair with a later
       one to envelope a genuine declaration and silently drop it -- a real
       false-negative risk in this backtick-heavy codebase. Fenced code blocks
       are multi-line constructs whose fence markers sit on separate lines
       from the content, so clause bounds (which break on newlines) cannot
       detect them; the fenced-block check therefore runs against the full
       document with absolute offsets, not the clause substring.
    2. **Foreign-issue-ref exclusion** — a match whose containing
       sentence/line carries ANY other ``#NNN`` reference (before OR after
       the match, and not part of the match itself) describes those OTHER
       issues, not this one. This generalizes the original backward-only
       ``_clause_preceding`` guard (issue #159) to also look forward, so
       issue-referencing parentheticals after the match are excluded too.
    3. **Blockquote exclusion** — a match on a line whose first non-space
       character is ``>`` is a quoted reply, not a self-declaration (issue
       #1847). The rule applies to every form: an inline declaration, and any
       line scanned inside a heading-list blocker section.
    4. The remaining matches are honored as genuine self-declarations.

    Returns an empty list if no blockers are found.
    """
    if not text:
        return []

    blockers: set[int] = set()
    # Heading-list blocker sections (issue #1847) contribute refs through a
    # separate per-item scan; the unreadable flag is consumed by
    # detect_prose_only_dependencies, not here.
    section_refs, _section_unreadable = _scan_blocker_sections(text)
    blockers.update(section_refs)

    # Check if they appear in blocker context
    for pattern in _BLOCKER_PATTERNS:
        for match in pattern.finditer(text):
            match_start, match_end = match.start(), match.end()

            # Guard 1a: a match inside a fenced code block (triple-backtick or
            # ~~~) is quoted prose, not a self-declaration. Fenced blocks are
            # multi-line constructs whose fence markers sit on separate lines
            # from the content, so the clause-scoped inline span checks below
            # cannot detect them (the content line is its own clause with no
            # fence markers). This check therefore runs against the full
            # document with absolute offsets (issue #1454 rework round 2).
            if _inside_fenced_block(text, match_start, match_end):
                continue

            # Guard 1c (issue #1847): a match on a Markdown blockquote line —
            # first non-space character is ``>`` — is a quoted reply, not a
            # self-declaration. The ``>`` precedes the match, so checking the
            # line prefix is equivalent to checking the line.
            line_start = text.rfind("\n", 0, match_start) + 1
            if _is_blockquote_line(text[line_start:match_start]):
                continue

            # Both remaining guards judge the match against its containing
            # clause, so compute the clause window once and reuse it. Scoping
            # the inline span check to the clause is what prevents an unrelated
            # stray backtick/quote elsewhere in the body from swallowing a
            # genuine declaration (issue #1454 rework).
            clause_start, clause_end = _clause_bounds(text, match_start, match_end)
            clause = text[clause_start:clause_end]
            match_rel_start = match_start - clause_start
            match_rel_end = match_end - clause_start

            # Guard 1b: a match inside an inline code span or quoted span
            # WITHIN the clause is quoted prose describing another issue, not
            # a self-declaration. Searched on the clause substring so a span
            # opening outside this clause cannot envelope the match.
            if _inside_code_span(clause, match_rel_start, match_rel_end):
                continue
            if _inside_quoted_span(clause, match_rel_start, match_rel_end):
                continue

            # Guard 2: any OTHER #NNN in the containing clause (not part of
            # this match) means the clause is about a different issue.
            has_foreign_ref = False
            for ref in _ISSUE_REF.finditer(clause):
                if ref.start() >= match_rel_start and ref.end() <= match_rel_end:
                    continue  # part of the match itself
                has_foreign_ref = True
                break
            if has_foreign_ref:
                continue

            # Extract the full match and find all #N references within it
            match_text = match.group(0)
            numbers_in_match = re.findall(r"#(\d+)", match_text)
            for num_str in numbers_in_match:
                try:
                    blockers.add(int(num_str))
                except (ValueError, TypeError):
                    # Skip malformed numbers
                    continue

    return sorted(blockers)


def detect_prose_only_dependencies(text: str) -> bool:
    """Detect prose-only dependency declarations in issue body.

    Returns True if the issue body contains dependency-like prose without
    structured blocker declarations. This catches cases like "Do not dispatch
    before P2-T2/P2-T3 have landed" that lack corresponding "Blocked by #N" markers.

    Patterns detected:
    - "do not dispatch before" (case-insensitive)
    - "depends on <...> P\\d+-T\\d+" — task reference in dependency context
    - "wait for <...> P\\d+-T\\d+ <...> (complete|done|land|merge|ship)" — task
      reference with completion verb; covers "Wait for P1-T5 to complete first."
    - "before/until/after <...> P\\d+-T\\d+ <...> (land|merge|complete|done|ship)"
    - "wait for" before a PR or merge event (non-task dependency prose)

    Pattern 2 is intentionally scoped to dependency context only — bare task
    marker mentions like "implements P2-T4" or title suffixes "(P2-T4)" are
    NOT matched, to avoid flagging every plan-generated issue for human review.

    Additionally (issue #1847), a "Blocked by"/"Depends on" heading section
    that holds an item the parser cannot read — neither a same-repo issue
    reference nor a none-sentinel (a URL, an ``owner/repo`` reference, free
    prose) — returns True so the issue is parked for a human instead of
    silently freed.

    Dependency-shaped prose inside a fenced code block is quoted/example
    code, not the issue author's own declaration: the body is stripped via
    ``_strip_fenced_blocks`` (the ``_fenced_block_ranges`` model
    ``parse_blockers`` uses — issue #1819) before any pattern runs.

    Args:
        text: The issue body text to check

    Returns:
        True if prose-only dependencies are detected, False otherwise
    """
    if not text:
        return False

    # Fenced-code exclusion (issue #1819): a code sample quoting
    # dependency-shaped prose must not park the issue.
    text = _strip_fenced_blocks(text)

    # Pattern 1: "do not dispatch before" and variants
    if re.search(r"do\s+not\s+dispatch\s+before", text, flags=re.IGNORECASE):
        return True

    # Pattern 2: task references (P\d+-T\d+) only in dependency context.
    # "depends on ... P\d+-T\d+" — classic self-declaration
    if re.search(r"depends\s+on\s+[^.\n]*P\d+-T\d+", text, flags=re.IGNORECASE):
        return True
    # "wait for ... P\d+-T\d+ ... <completion verb>" — e.g. "Wait for P1-T5 to complete first."
    if re.search(
        r"wait\s+for\s+[^.\n]*P\d+-T\d+[^.\n]*(?:land|merge|complete|done|ship)",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    # "before/until/after ... P\d+-T\d+ ... <completion verb>"
    if re.search(
        r"(?:before|until|after)\s+[^.\n]*P\d+-T\d+[^.\n]*(?:land|merge|complete|done|ship)",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    # Pattern 3: "wait for" before a PR or merge event (non-task dependency prose)
    if re.search(
        r"wait\s+for\s+(?:this|that|these|those)?\s*(?:PR|merge|land)", text, flags=re.IGNORECASE
    ):
        return True

    # Issue #1847: an unreadable item inside a "Blocked by"/"Depends on"
    # heading section is a dependency declaration we cannot read.
    _section_refs, section_unreadable = _scan_blocker_sections(text)
    if section_unreadable:
        return True

    return False
