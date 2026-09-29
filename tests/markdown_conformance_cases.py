"""CommonMark ground-truth conformance cases for candidate 3 ("markdown
structure"), architecture-deepening plan
`docs/superpowers/plans/2026-09-29-architecture-deepening.md`.

This is a **data table**, not a test module (no `test_*` name, no assertions,
not collected by pytest). It records, per case, the input markdown and the
CommonMark-CORRECT block-level line spans for fenced code blocks, blockquote
lines, and ATX heading lines. `tests/test_markdown_conformance.py` runs this
table against both the unified scan side of `markdown_fence.py` and
`scripts/heartbeat_check.py`'s stdlib copy, as their shared conformance
suite.

## Conventions

* Lines are 0-indexed via `markdown_fence.split_lines()` (no keepends) --
  CommonMark line endings only (`\n`, `\r\n`, `\r`), NOT `str.splitlines()`,
  which also breaks on separators CommonMark doesn't (adversarial review
  finding B2, architecture-deepening candidate 3).
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
        # the opener under CommonMark -- that line is ordinary prose, not
        # the start of a "malformed fence" region to skip. The line-2 ```
        # is therefore free to open its OWN fence: unclosed (nothing
        # follows it), so it's an empty fence at (2, 3), not "no fence at
        # all". Verified against markdown-it-py in commonmark mode
        # (adversarial review finding B1, architecture-deepening
        # candidate 3): a prior version of this row asserted `()`, which
        # was the implementation's own (wrong) output, not CommonMark's.
        fenced_line_spans=((2, 3),),
    ),
    ConformanceCase(
        name="fence_after_unicode_line_separator_is_not_split",
        markdown="prose ```example-secret\nAKIAZ7Q3XK2PLM4RT5WN\n```\n",
        # U+2028 LINE SEPARATOR is not a CommonMark line ending (only \n,
        # \r\n, \r are), so line 0 is "prose ```example-secret" as ONE
        # line -- a `` ``` `` that doesn't open the line is not a fence
        # delimiter at all, just paragraph text. Line 2's ``` (the ONLY
        # real line break is the \n after "...RT5WN") then opens its own
        # unclosed fence. A splitter that also breaks on U+2028 (like
        # `str.splitlines()`) would instead see 3 "lines" before this one
        # and wrongly pair `` ```example-secret `` with the final ```,
        # masking prose as fenced content (adversarial review finding B2).
        fenced_line_spans=((2, 3),),
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
    ConformanceCase(
        name="heading_single_leading_tab_is_indented_code_not_heading",
        markdown="prose\n\n\t# heading\nprose after\n",
        # A tab advances to the next multiple of 4 columns, so a single
        # leading tab is already 4 columns -- 4+-column indent after a
        # blank line is an indented code block, not an ATX heading.
        # `[ \t]{0,3}` (a *character* count) would wrongly admit this one
        # tab as "0-3 spaces" (adversarial review finding B3,
        # architecture-deepening candidate 3). Verified against
        # markdown-it-py.
        heading_lines=(),
    ),
    ConformanceCase(
        name="fence_single_leading_tab_is_indented_code_not_fence",
        markdown="prose\n\n\t```\n\tcode\n\t```\nprose after\n",
        # Same column-counting rule as the heading case above, applied to
        # a fence delimiter: a single leading tab is 4 columns, so this
        # whole run is one indented code block, not a fence (adversarial
        # review finding B3).
        fenced_line_spans=(),
    ),
    ConformanceCase(
        name="blockquote_space_then_tab_is_indented_code_not_quote",
        markdown="prose\n\n \t> quote?\nprose after\n",
        # One space (1 column) then one tab: a tab advances to the NEXT
        # multiple of 4, so it consumes 3 more columns from column 1,
        # landing on column 4 -- 4+ columns total from just two
        # characters. Indented code, not a blockquote marker (adversarial
        # review finding B3).
        quoted_lines=(),
    ),
)


CASES_BY_NAME: dict[str, ConformanceCase] = {case.name: case for case in CASES}
