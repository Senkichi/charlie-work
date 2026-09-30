"""Container-tolerant scan mode (md-r2 re-review, findings B1 and B2).

`markdown_fence.scan` has no list-item container: it applies CommonMark's
top-level 0-3-column indent bound to fence openers, closers and blockquote
markers. Inside a list item CommonMark re-bases content to the item's content
column, so a fence or `>` indented 4+ columns under `1. ` / `- ` IS a fence /
blockquote. That makes one strict, container-blind model wrong for one side or
the other on nested input:

* *exclusion* guards (`github_body_scan`, `github_prose_dependencies`) must
  over-approximate quoted regions -> `max_indent=None`;
* *masking* guards (`outbound_body_guard`) must under-approximate the exempt
  region -> strict scan, plus a lenient closer bound (tested in
  `test_outbound_body_guard.py`).

Every "top-level control" below proves the harness can still report a blocker,
so the nested-input `[]` results are not an artifact of a dead parser.
"""

from __future__ import annotations

import pytest

from charlie_work import markdown_fence
from charlie_work.github_body_scan import parse_blockers

# --- B1: parse_blockers must not read quoted nested material as a blocker ---

_NESTED_FENCE_UNDER_NUMBERED_STEP = "Repro steps:\n\n1. Paste this into the issue:\n\n    ```\n    Blocked by #5\n    ```\n\n2. Done\n"
_TAB_FENCE_UNDER_BULLET = "- example:\n\t```\n\tDepends on #9\n\t```\n"
_SPACE_QUOTE_UNDER_BULLET = "Context\n\n- note from #7:\n    > Blocked by #5\n"
_TAB_QUOTE_UNDER_BULLET = "Context\n\n- note from #7:\n\t> Blocked by #5\n"


@pytest.mark.parametrize(
    "body",
    [
        _NESTED_FENCE_UNDER_NUMBERED_STEP,
        _TAB_FENCE_UNDER_BULLET,
        _SPACE_QUOTE_UNDER_BULLET,
        _TAB_QUOTE_UNDER_BULLET,
    ],
    ids=["numbered-step-fence", "tab-fence-under-bullet", "space-quote", "tab-quote"],
)
def test_parse_blockers_ignores_quoted_material_nested_in_list_items(body: str) -> None:
    assert parse_blockers(body) == []


def test_parse_blockers_control_top_level_indented_code_still_reports() -> None:
    """Positive control: the parser can report a blocker on this harness."""
    assert parse_blockers("Repro:\n\n    Blocked by #5\n") == [5]
    assert parse_blockers("Blocked by #5\n") == [5]


def test_parse_blockers_blocker_section_skips_nested_fence() -> None:
    """The heading-driven section parser (`_scan_blocker_sections`) must also
    treat a list-nested fenced block as quoted, not as a section entry."""
    body = "## Blocked by\n\n- #3\n\n1. step\n\n    ```\n    - #8\n    ```\n"
    assert parse_blockers(body) == [3]


# --- scan(max_indent=None) semantics -----------------------------------------


def test_scan_default_stays_strict_top_level() -> None:
    text = "1. step\n\n    ```\n    code\n    ```\n"
    assert markdown_fence.scan(text).fences == ()


def test_scan_tolerant_accepts_any_indent_fence_and_quote() -> None:
    text = "- item\n\n    ```\n    code\n    ```\n\n    > quoted\n"
    structure = markdown_fence.scan(text, max_indent=None)
    assert structure.fenced_line_spans == ((2, 5),)
    assert structure.quoted_lines == frozenset({6})


def test_scan_tolerant_keeps_other_fence_rules() -> None:
    """Only the indent bound is relaxed: closer char/length, info-string
    backtick exclusion and unclosed-to-EOF are unchanged."""
    # A shorter closer does not close a longer opener.
    short = markdown_fence.scan("    ````\n    x\n    ```\n", max_indent=None)
    assert [(f.start, f.end, f.closed) for f in short.fences] == [(0, 3, False)]
    # Backtick in a backtick fence's info string: not a fence.
    assert markdown_fence.scan("    ```a`b\n    x\n", max_indent=None).fences == ()
    # Tilde fences work at any indent, and are closed by the same char.
    tilde = markdown_fence.scan("\t~~~\n\tx\n\t~~~\n", max_indent=None)
    assert tilde.fenced_line_spans == ((0, 3),)


def test_scan_tolerant_keeps_headings_strict() -> None:
    """A 4+-column `#` line is never an ATX heading, even in tolerant mode:
    a spurious heading would open a phantom `Blocked by` section. (Known
    trade-off, not a regression: strict headings also miss a genuinely
    list-nested `## Blocked by`, which CommonMark would re-base into a real
    heading -- origin/main's `^ {0,3}` heading regex had the same gap.)"""
    structure = markdown_fence.scan("    # Blocked by\n# Real\n", max_indent=None)
    assert structure.heading_lines == (1,)


def test_is_blockquote_marker_max_indent() -> None:
    assert markdown_fence.is_blockquote_marker("    > q") is False
    assert markdown_fence.is_blockquote_marker("    > q", max_indent=None) is True
    assert markdown_fence.is_blockquote_marker("\t> q", max_indent=None) is True
    assert markdown_fence.is_blockquote_marker("x > q", max_indent=None) is False
