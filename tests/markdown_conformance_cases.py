"""CommonMark ground-truth conformance cases for candidate 3 ("markdown
structure"), architecture-deepening plan
`docs/superpowers/plans/2026-09-29-architecture-deepening.md`.

This is a **data table**, not a test module (no `test_*` name, no assertions,
not collected by pytest). It records, per case, the input markdown and the
CommonMark-CORRECT block-level line spans for fenced code blocks, blockquote
lines, and ATX heading lines. A later step runs this table against the
unified scan side of `markdown_fence.py` *and* `scripts/heartbeat_check.py`'s
stdlib copy, once both exist, as their shared conformance suite.

It is deliberately NOT wired to a test yet: most cases here encode exactly
the deviations that `tests/test_markdown_structure_characterization.py`
pins as CURRENT (wrong) behavior, so asserting this table against any
existing consumer today would fail by design, not by accident.

## Conventions

* Lines are 0-indexed via `markdown.splitlines()` (no keepends).
* `fenced_line_spans`: tuple of `(start, end)` half-open line-index ranges.
  `start` is the line with the opening fence delimiter; `end` is one past
  the line with the CLOSING fence delimiter (i.e. the closer line IS part
  of the span) -- or, for an unclosed fence, one past the last line of the
  text. This differs from `github_body_scan._fenced_block_ranges`'s
  char-offset convention, which excludes the closer line; a consumer of
  this table converts as needed.
* `quoted_lines`: tuple of 0-indexed line numbers that are blockquote-marker
  lines at the block-structure level (a `>` that is genuinely blockquote
  syntax, not literal text inside a code block or over-indented out of
  blockquote range).
* `heading_lines`: tuple of 0-indexed line numbers that are valid ATX
  heading lines (level 1-6, `#` run followed by whitespace-or-EOL, not
  inside a fenced or indented code block).

## Scope

Block-level structure only (fences, blockquotes, ATX headings) -- matching
what every surveyed consumer in `md-recon.md` actually scans for. Full
CommonMark inline parsing (e.g. an inline code span that happens to span
multiple lines of a paragraph) is out of scope; see that recon's own scope
note for why the block-level cut is the right one here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConformanceCase:
    name: str
    markdown: str
    fenced_line_spans: tuple[tuple[int, int], ...] = ()
    quoted_lines: tuple[int, ...] = ()
    heading_lines: tuple[int, ...] = ()


CASES: tuple[ConformanceCase, ...] = (
    ConformanceCase(
        name="basic_backtick_fence",
        markdown="prose before\n```\ncode line\n```\nprose after\n",
        fenced_line_spans=((1, 4),),
    ),
    ConformanceCase(
        name="basic_tilde_fence",
        markdown="prose before\n~~~\ncode line\n~~~\nprose after\n",
        fenced_line_spans=((1, 4),),
    ),
    ConformanceCase(
        name="fence_with_language_tag",
        markdown="```python\nx = 1\n```\n",
        fenced_line_spans=((0, 3),),
    ),
    ConformanceCase(
        name="fence_indented_3_spaces_still_valid",
        markdown="prose\n\n   ```\n   code\n   ```\nprose after\n",
        fenced_line_spans=((2, 5),),
    ),
    ConformanceCase(
        name="fence_indented_4_spaces_is_not_a_fence",
        markdown="prose\n\n    ```\n    code\n    ```\nprose after\n",
        # 4+ leading spaces after a blank line is an indented CODE block
        # (CommonMark), not a fence -- the ``` characters are literal
        # content, not delimiters. No fenced span at all.
        fenced_line_spans=(),
    ),
    ConformanceCase(
        name="fence_backtick_in_info_string_is_not_a_fence",
        markdown="```code`sample\nprose, not code\n```\n",
        # A backtick-fence info string containing a backtick disqualifies
        # the opener under CommonMark -- no fence forms here at all
        # (block-level; see module scope note re: inline code spans).
        fenced_line_spans=(),
    ),
    ConformanceCase(
        name="fence_closer_must_be_at_least_as_long_as_opener",
        markdown="````json\ncontent with a decoy fence marker below\n```\nmore content after decoy\n````\n",
        # Opener is 4 backticks; the 3-backtick line is too short to close
        # it and is just literal content. Only the 4-backtick line closes.
        fenced_line_spans=((0, 5),),
    ),
    ConformanceCase(
        name="fence_unclosed_runs_to_eof",
        markdown="prose\n```\nunterminated code\nmore code, never closed\n",
        fenced_line_spans=((1, 4),),
    ),
    ConformanceCase(
        name="fence_desync_regression_embedded_backtick_run",
        markdown='```json\n{"note": "see the ``` marker above", "id": 2}\n```\nprose after, not code\n',
        # The mid-line ``` inside the JSON string is not a delimiter at all
        # (fences are line-anchored) -- one clean fence, lines 0-2.
        fenced_line_spans=((0, 3),),
    ),
    ConformanceCase(
        name="heading_basic_levels",
        markdown="# H1\n## H2\n### H3 with trailing ###\nprose\n",
        heading_lines=(0, 1, 2),
    ),
    ConformanceCase(
        name="heading_no_space_after_hash_is_not_a_heading",
        markdown="###notaheading\nprose\n",
        # ATX heading requires whitespace (or EOL) after the `#` run.
        heading_lines=(),
    ),
    ConformanceCase(
        name="heading_inside_fenced_block_is_not_a_heading",
        markdown="```\n# not a real heading, this is code\n```\n# real heading\n",
        fenced_line_spans=((0, 3),),
        heading_lines=(3,),
    ),
    ConformanceCase(
        name="blockquote_basic",
        markdown="> quoted line one\n> quoted line two\nprose, not quoted\n",
        quoted_lines=(0, 1),
    ),
    ConformanceCase(
        name="blockquote_indented_4_spaces_is_not_recognized",
        markdown="prose\n\n    > looks like a quote but is 4-space indented\nprose after\n",
        # 4+ leading spaces after a blank line makes this an indented code
        # block; the `>` is literal code text, not a blockquote marker.
        quoted_lines=(),
    ),
)


CASES_BY_NAME: dict[str, ConformanceCase] = {case.name: case for case in CASES}
