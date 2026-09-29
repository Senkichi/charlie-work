"""CommonMark-safe fencing for text embedded in a rendered prompt (issue #883).

A prompt template that writes its fence literally --

    ```md
    $issue_body
    ```

-- is only correct while the substituted value contains no fence of its own. On
a developer issue tracker that assumption fails most of the time: measured
against this repo, **61 of 100 open issue bodies contain a code fence**. When
one does, the body's own ``` closes the block early and everything after it
stops being quoted material. A heading in the issue body then becomes
indistinguishable from a heading the orchestrator wrote, and in the worker
templates ``$section_scope_contract`` follows immediately after the block, so
body text can merge visually with the scope contract.

That is a correctness bug first: the framing is lost for the *normal* case,
with no adversary involved. It is a prompt-injection vector second, and only
latently -- this repo is private, so issue authors are already trusted
collaborators. Worth noting that ``prompts.render_prompt`` deliberately
substitutes exactly once so that a supplied value is never re-scanned as a
template; a fence that the content can close reintroduces the same hazard one
layer down, at the formatting layer rather than the templating layer.

The fix is the CommonMark rule: a fenced block closes on the first line whose
fence is *at least as long* as the opener, so an opener longer than any
backtick run in the content cannot be terminated from inside it. Computing the
width means the fence has to move out of the template and into the substituted
value, which is why callers supply a pre-fenced ``*_block`` rather than a bare
value.

This lives in its own module rather than in ``issue_comments`` (where the rule
was first implemented, for #872) because it is a CommonMark concern with
several unrelated consumers, and each one would otherwise be tempted to
re-derive it. There are three already:

* ``$issue_body_block`` in the worker templates (this issue);
* the per-comment block in ``issue_comments`` (#872);
* ``$dispatch_note_block`` in ``rework.md`` -- reviewer prose quoting pytest
  output and shell commands, found by sweeping for the same defect class
  rather than assumed absent. 16 of 289 review summaries on disk carry a
  fence, and ``prs/pr-182/rework-prompt.md`` is a rendered instance of the
  break: the reviewer's own fence closed the wrapper early, so the template's
  intended *closing* fence opened a block that swallowed the brief's
  "Required behavior" and push-verification sections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "MIN_FENCE_LENGTH",
    "fence_for",
    "fenced_block",
    "FenceSpan",
    "Heading",
    "MarkdownStructure",
    "scan",
    "is_blockquote_marker",
    "split_lines",
]

# CommonMark's minimum fence. A shorter run is inline code, not a block.
MIN_FENCE_LENGTH = 3

_BACKTICK_RUN_RE = re.compile(r"`+")


def fence_for(text: str) -> str:
    """Return a backtick fence that ``text`` cannot terminate from inside.

    One longer than the longest backtick run present, never shorter than
    ``MIN_FENCE_LENGTH``.
    """
    longest = max((len(run) for run in _BACKTICK_RUN_RE.findall(text)), default=0)
    return "`" * max(MIN_FENCE_LENGTH, longest + 1)


def fenced_block(text: str, info: str = "") -> str:
    """Wrap ``text`` in a fence it cannot escape, tagged with ``info``.

    ``text`` is embedded verbatim -- no stripping, no normalisation. That is
    deliberate: it keeps the output byte-identical to a literal three-backtick
    fence for every value that contains no fence of its own, which is what makes
    this change auditable. Any prompt diff is then attributable to a body that
    genuinely needed a wider fence.
    """
    fence = fence_for(text)
    return f"{fence}{info}\n{text}\n{fence}"


# --- scan side (architecture-deepening candidate 3, "markdown structure") --
#
# Seven independent hand-rolled fence/blockquote/heading scanners exist in
# this repo (`github_body_scan.py`, `cross_repo_gate.py`,
# `outbound_body_guard.py`, `verdict_parsing.py` (twice), `rescue_review.py`,
# `attachment_contracts/hook_entry.py`), plus an eighth, deliberately
# stdlib-only copy in `scripts/heartbeat_check.py`. Full inventory and a
# CommonMark-deviation matrix: `md-recon.md` (wave A scratchpad). Every one
# of them deviates from CommonMark somewhere -- most from the pre-#1819
# "naive nearest-pair" defect `github_body_scan.py`'s own docstring above
# describes fixing in exactly one of the eight places that needed it.
#
# This is the scan side those seven consumers now delegate to (wiring
# happened in a later, separate commit; see the plan). It ports in the
# strongest existing model found in the recon: `outbound_body_guard.
# _FENCE_OPEN_RE`'s 0-3-COLUMN indent bound (tab stops of 4, not 0-3
# characters -- a single leading tab is already 4 columns, so `_strip_
# marker_indent` rejects it; see its docstring -- adversarial review finding
# B3, architecture-deepening candidate 3) and backtick-in-info-string
# exclusion, with `cross_repo_gate`'s same-char/length-aware closer search
# folded in as one shared subroutine (`_find_fence_close`), used only for a
# *valid* opener: a rejected opener (backtick fence, backtick in its info
# string) is not a fence at all under CommonMark, so it is treated as an
# ordinary line and scanning resumes immediately on the next line -- it does
# not search for or consume a "closer" (adversarial review finding B1;
# `_find_fence_close`'s docstring explains the narrower design this
# replaced).
#
# Deliberately out of scope (`tests/markdown_conformance_cases.py`'s own
# scope note): full CommonMark inline parsing (code spans, emphasis, links),
# list items, setext headings, thematic breaks, HTML blocks, and the
# lazy-paragraph-continuation nuance that lets an indented line's fence-ness
# depend on whether it interrupts a paragraph -- every construct here already
# bounds its own leading indent to 0-3 columns, which is sufficient on its own
# to reject every over-indented case in the conformance table without also
# needing to model paragraphs. Block-level fences, blockquotes, and ATX
# headings are what every surveyed consumer actually scans for.

# CommonMark line endings only: `\n`, `\r\n`, `\r`. Used by :func:`split_lines`
# instead of `str.splitlines()`, which also breaks on several separators
# (vertical tab, form feed, file/group/record separator, NEL, LINE/PARAGRAPH
# SEPARATOR) that neither CommonMark nor GitHub's renderer treat as line
# breaks (adversarial review finding B2, architecture-deepening candidate 3).
_LINE_ENDING_RE = re.compile(r"\r\n|\r|\n")

# The three markers below match against a line with its leading indent
# already stripped by :func:`_strip_marker_indent` (which enforces
# CommonMark's 0-3-*column* bound, tab stops of 4 -- see that function's
# docstring). They must NOT re-add an `[ \t]{0,3}` prefix: that pattern
# counts leading *characters*, not columns, so it would wrongly admit up to
# three tabs (12 columns) as if it were "0-3 spaces" (adversarial review
# finding B3).

# CommonMark fenced-code-block delimiter: 3+ of the same backtick or tilde
# character, then an optional info string.
_FENCE_DELIM_RE = re.compile(r"^(`{3,}|~{3,})[ \t]*(.*)$")

# CommonMark blockquote marker: `>`. (The optional single space CommonMark
# allows after the `>` is not captured here -- callers needing quoted
# *content*, not just the marker line, strip it themselves; every surveyed
# consumer only needs the boolean.)
_BLOCKQUOTE_RE = re.compile(r"^>")

# CommonMark ATX heading: 1-6 `#`, then either end-of-line or at least one
# space/tab before the heading text.
_ATX_HEADING_RE = re.compile(r"^(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")

# An ATX heading's optional closing sequence of `#`s (`## Text ##`): must be
# preceded by a space/tab (or be the whole remaining text) and followed only
# by trailing whitespace, which the caller has already stripped.
_ATX_CLOSING_RUN_RE = re.compile(r"(?:^|[ \t])#+$")


def split_lines(text: str, *, keepends: bool = False) -> list[str]:
    """Split ``text`` on CommonMark line endings only (``\\n``, ``\\r\\n``,
    ``\\r``) -- never ``str.splitlines()``, whose wider separator set
    (vertical tab, form feed, file/group/record separator, NEL, LINE/
    PARAGRAPH SEPARATOR) is not a CommonMark or GitHub line break.

    :func:`scan` uses this internally, and every consumer that indexes its
    own line list by a `FenceSpan`/`Heading` line number returned from
    `scan` (`outbound_body_guard._mask_example_secret_fences`,
    `verdict_parsing._extract_verdict_from_text` /
    `_strip_fenced_blocks`, `rescue_review._find_json_verdict`,
    `attachment_contracts.hook_entry.parse_advisories_comment`,
    `github_body_scan._line_start_offsets` / `_scan_blocker_sections`) must
    use it too, on the same ``text``, rather than re-deriving its own line
    list with ``str.splitlines()``. Otherwise a single invisible non-
    CommonMark separator (e.g. U+2028 LINE SEPARATOR) in the input desyncs
    the consumer's line indices from `scan`'s, which can silently shift a
    `FenceSpan` onto the wrong lines (adversarial review finding B2,
    architecture-deepening candidate 3). Matches ``str.splitlines()``'s
    output exactly for text containing only CommonMark line endings --
    including the "no trailing empty element after a final terminator"
    behaviour -- so this is a drop-in, narrower replacement.
    """
    if not text:
        return []
    lines: list[str] = []
    pos = 0
    for match in _LINE_ENDING_RE.finditer(text):
        end = match.end() if keepends else match.start()
        lines.append(text[pos:end])
        pos = match.end()
    if pos < len(text):
        lines.append(text[pos:])
    return lines


def _leading_indent_columns(line: str) -> tuple[int, int]:
    """Return ``(columns, chars)`` for ``line``'s leading run of spaces and
    tabs: the CommonMark *column* width of that run, and how many
    characters it takes up.

    A tab advances to the next multiple of 4 columns (CommonMark's tab
    handling), so a single leading tab is already 4 columns wide -- not the
    1 character that a naive ``[ \\t]{0,3}`` regex (matching characters, not
    columns) would attribute to it, wrongly admitting up to three tabs (12
    columns) as if they were "0-3 spaces" (adversarial review finding B3,
    architecture-deepening candidate 3). Stops at the first non-space/tab
    character, or as soon as the running total exceeds 3 columns -- nothing
    further can matter once CommonMark's 0-3-column bound is already blown.
    """
    columns = 0
    chars = 0
    for char in line:
        if char == " ":
            columns += 1
        elif char == "\t":
            columns += 4 - (columns % 4)
        else:
            break
        chars += 1
        if columns > 3:
            break
    return columns, chars


def _strip_marker_indent(line: str) -> str | None:
    """Strip ``line``'s leading indent and return the remainder, or
    ``None`` when that indent is 4+ CommonMark columns wide.

    A 4+-column indent makes the line an indented code block under
    CommonMark, never a fence/blockquote/heading marker, regardless of how
    many literal space/tab *characters* precede it -- see
    :func:`_leading_indent_columns`. Every marker check in :func:`scan`
    (fence delimiter, blockquote marker, ATX heading) and in
    :func:`_find_fence_close` goes through this first, so all four share
    one indent rule and cannot drift from each other.
    """
    columns, chars = _leading_indent_columns(line)
    if columns > 3:
        return None
    return line[chars:]


@dataclass(frozen=True)
class FenceSpan:
    """One fenced code block, as a half-open line-index range.

    ``end`` is one past the line with the CLOSING fence delimiter (the
    closer line IS part of the span) -- or, for an unclosed fence, one past
    the last line of the text. This is the same convention
    `tests/markdown_conformance_cases.py` documents; it differs from
    `github_body_scan._fenced_block_ranges`'s char-offset convention (which
    excludes the closer line), so a consumer wanting that shape converts.
    """

    start: int
    end: int
    char: str  # "`" or "~"
    length: int  # width of the delimiter run that actually opened this fence
    info: str  # the opening fence's info string, stripped
    closed: bool  # False when the fence ran to end-of-text unterminated

    def __contains__(self, line: int) -> bool:
        return self.start <= line < self.end


@dataclass(frozen=True)
class Heading:
    """One ATX heading line: its 0-indexed line number, level (1-6), and
    text with any optional closing `#`-run already stripped.
    """

    line: int
    level: int
    text: str


@dataclass(frozen=True)
class MarkdownStructure:
    """The block-level structure of one markdown document.

    Line-indexed against ``split_lines(text)`` (no keepends), 0-indexed --
    matching `tests/markdown_conformance_cases.py`. This is the one thing
    every surveyed consumer in `md-recon.md` actually needs: which lines/
    spans are fenced code, which are blockquote markers, which are ATX
    headings, and (by elimination) which are prose.
    """

    line_count: int
    fences: tuple[FenceSpan, ...]
    quoted_lines: frozenset[int]
    headings: tuple[Heading, ...]

    @property
    def fenced_line_spans(self) -> tuple[tuple[int, int], ...]:
        """``(start, end)`` half-open ranges, one per fenced block, in document order."""
        return tuple((fence.start, fence.end) for fence in self.fences)

    @property
    def heading_lines(self) -> tuple[int, ...]:
        """Line numbers of every ATX heading, in document order."""
        return tuple(heading.line for heading in self.headings)

    def is_fenced(self, line: int) -> bool:
        """True when ``line`` falls inside any fenced code block (open or closed)."""
        return any(line in fence for fence in self.fences)


def is_blockquote_marker(line: str) -> bool:
    """True if ``line`` opens with a CommonMark blockquote marker (0-3
    leading indent *columns*, tab stops of 4, then ``>``).

    A single-line primitive for consumers (e.g. ``github_prose_dependencies.
    _is_blockquote_line``) that only need the boolean for one line at a
    time -- not a multi-line document -- rather than paying for a full
    :func:`scan`. Shares ``_strip_marker_indent`` and ``_BLOCKQUOTE_RE``
    with :func:`scan` so the two can never drift on what counts as a
    blockquote marker.
    """
    stripped = _strip_marker_indent(line)
    return stripped is not None and _BLOCKQUOTE_RE.match(stripped) is not None


def _find_fence_close(lines: list[str], start: int, char: str, length: int) -> int | None:
    """Return the line index of the first valid closer for a ``char``-fence
    opened with a delimiter run of ``length``, searching from ``start``, or
    ``None`` if none exists before end-of-text.

    Only ever called for a *valid* opener (:func:`scan` treats a rejected
    one -- a backtick fence whose info string itself contains a backtick --
    as an ordinary line, not something with a "closer" to find; see
    :func:`scan`'s own comment). Each candidate closer line's indent is
    checked with the same :func:`_strip_marker_indent` fence/blockquote/
    heading share, so a closer over-indented past 0-3 columns (e.g. tab-
    indented) is correctly skipped rather than accepted.
    """
    close_re = re.compile(rf"^{re.escape(char)}{{{length},}}[ \t]*$")
    for index in range(start, len(lines)):
        stripped = _strip_marker_indent(lines[index])
        if stripped is not None and close_re.match(stripped):
            return index
    return None


def _strip_atx_closing_run(text: str) -> str:
    text = text.rstrip(" \t")
    match = _ATX_CLOSING_RUN_RE.search(text)
    if match:
        text = text[: match.start()].rstrip(" \t")
    return text


def scan(text: str) -> MarkdownStructure:
    """Scan ``text`` and return its CommonMark block-level structure.

    Implements: 0-3-column indent bound (tab stops of 4) on fence/
    blockquote/heading markers; backtick- and tilde-fence support; a closer
    that must be the same character and at least as long as its opener; a
    backtick fence's info string may not itself contain a backtick -- a
    line that fails this is not a fence opener at all, so it is ordinary
    text and scanning resumes on the very next line; an unclosed (valid)
    fence runs to end-of-text; an ATX heading requires whitespace (or
    end-of-line) after its `#` run, with an optional closing `#`-run
    stripped from its text.
    """
    lines = split_lines(text)
    fences: list[FenceSpan] = []
    quoted: set[int] = set()
    headings: list[Heading] = []

    line_count = len(lines)
    index = 0
    while index < line_count:
        stripped = _strip_marker_indent(lines[index])

        if stripped is not None:
            delimiter = _FENCE_DELIM_RE.match(stripped)
            if delimiter is not None:
                run, info = delimiter.group(1), delimiter.group(2).strip()
                char, length = run[0], len(run)
                if char == "`" and "`" in info:
                    # Not a fence opener under CommonMark (a backtick
                    # fence's info string may not itself contain a
                    # backtick) -- an ordinary line, not the start of a
                    # region to skip. The very next line is scanned fresh,
                    # so e.g. a real fence opener on it is still seen
                    # (adversarial review finding B1).
                    index += 1
                    continue
                close_at = _find_fence_close(lines, index + 1, char, length)
                end = close_at + 1 if close_at is not None else line_count
                fences.append(
                    FenceSpan(
                        start=index,
                        end=end,
                        char=char,
                        length=length,
                        info=info,
                        closed=close_at is not None,
                    )
                )
                index = end
                continue

            if _BLOCKQUOTE_RE.match(stripped):
                quoted.add(index)
                index += 1
                continue

            heading_match = _ATX_HEADING_RE.match(stripped)
            if heading_match is not None:
                level = len(heading_match.group(1))
                text_group = heading_match.group(2) or ""
                headings.append(
                    Heading(line=index, level=level, text=_strip_atx_closing_run(text_group))
                )
                index += 1
                continue

        index += 1

    return MarkdownStructure(
        line_count=line_count,
        fences=tuple(fences),
        quoted_lines=frozenset(quoted),
        headings=tuple(headings),
    )
