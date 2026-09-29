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
from enum import Enum

__all__ = [
    "MIN_FENCE_LENGTH",
    "fence_for",
    "fenced_block",
    "LineKind",
    "FenceSpan",
    "Heading",
    "MarkdownStructure",
    "scan",
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
# This is the scan side those seven consumers should eventually converge on
# (not wired here -- that is a later, separate step; see the plan). It ports
# in the strongest existing model found in the recon: `outbound_body_guard.
# _FENCE_OPEN_RE`'s 0-3-space indent bound and backtick-in-info-string
# exclusion (the only implementation that gets that rule right at all), with
# `cross_repo_gate`'s same-char/length-aware closer search folded in as one
# shared subroutine used for BOTH a valid opener and a rejected one -- see
# `_find_fence_close`'s docstring for why a rejected opener still searches
# for (and consumes) its matching closer rather than leaving the fence-shaped
# text open to be reinterpreted as fresh structure.
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

# CommonMark fenced-code-block delimiter: 0-3 leading spaces/tabs, then 3+
# of the same backtick or tilde character, then an optional info string.
_FENCE_DELIM_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*(.*)$")

# CommonMark blockquote marker: 0-3 leading spaces/tabs, then `>`. (The
# optional single space CommonMark allows after the `>` is not captured here
# -- callers needing quoted *content*, not just the marker line, strip it
# themselves; every surveyed consumer only needs the boolean.)
_BLOCKQUOTE_RE = re.compile(r"^[ \t]{0,3}>")

# CommonMark ATX heading: 0-3 leading spaces/tabs, 1-6 `#`, then either
# end-of-line or at least one space/tab before the heading text.
_ATX_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")

# An ATX heading's optional closing sequence of `#`s (`## Text ##`): must be
# preceded by a space/tab (or be the whole remaining text) and followed only
# by trailing whitespace, which the caller has already stripped.
_ATX_CLOSING_RUN_RE = re.compile(r"(?:^|[ \t])#+$")


class LineKind(Enum):
    """Which of the four (mutually exclusive) block-level buckets a line is
    in: fenced code, a blockquote marker, an ATX heading, or plain prose.
    """

    FENCE = "fence"
    QUOTE = "quote"
    HEADING = "heading"
    PROSE = "prose"


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

    Line-indexed against ``text.splitlines()`` (no keepends), 0-indexed --
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

    def nearest_heading_before(self, line: int) -> Heading | None:
        """The last heading whose line is strictly before ``line``, or ``None``.

        Headings are stored in document order, so this is the same "most
        recent heading regardless of level" rule `cross_repo_gate.
        _nearest_preceding_heading` implements by hand today.
        """
        nearest: Heading | None = None
        for heading in self.headings:
            if heading.line >= line:
                break
            nearest = heading
        return nearest

    def kind_of(self, line: int) -> LineKind:
        """Classify one line into its block-level bucket."""
        if self.is_fenced(line):
            return LineKind.FENCE
        if line in self.quoted_lines:
            return LineKind.QUOTE
        if line in self.heading_lines:
            return LineKind.HEADING
        return LineKind.PROSE


def _find_fence_close(lines: list[str], start: int, char: str, length: int) -> int | None:
    """Return the line index of the first valid closer for a ``char``-fence
    opened with a delimiter run of ``length``, searching from ``start``, or
    ``None`` if none exists before end-of-text.

    Called for a *rejected* opener too (one whose info string broke the
    backtick rule), not only a valid one: the text between a fence-shaped
    line and its nearest same-char, long-enough closer reads as an author's
    (malformed) attempt at a code block either way, and re-interpreting a
    line in the middle of it as a fresh heading, blockquote, or fence would
    be surprising -- so both cases consume the same region; they differ only
    in whether `scan` records a `FenceSpan` for it.
    """
    close_re = re.compile(rf"^[ \t]{{0,3}}{re.escape(char)}{{{length},}}[ \t]*$")
    for index in range(start, len(lines)):
        if close_re.match(lines[index]):
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

    Implements: 0-3 space/tab indent bound on fence/blockquote/heading
    markers; backtick- and tilde-fence support; a closer that must be the
    same character and at least as long as its opener; a backtick fence's
    info string may not itself contain a backtick; an unclosed fence runs to
    end-of-text; an ATX heading requires whitespace (or end-of-line) after
    its `#` run, with an optional closing `#`-run stripped from its text.
    """
    lines = text.splitlines()
    fences: list[FenceSpan] = []
    quoted: set[int] = set()
    headings: list[Heading] = []

    line_count = len(lines)
    index = 0
    while index < line_count:
        line = lines[index]

        delimiter = _FENCE_DELIM_RE.match(line)
        if delimiter is not None:
            run, info = delimiter.group(1), delimiter.group(2).strip()
            char, length = run[0], len(run)
            valid_opener = not (char == "`" and "`" in info)
            close_at = _find_fence_close(lines, index + 1, char, length)
            end = close_at + 1 if close_at is not None else line_count
            if valid_opener:
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

        if _BLOCKQUOTE_RE.match(line):
            quoted.add(index)
            index += 1
            continue

        heading_match = _ATX_HEADING_RE.match(line)
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
