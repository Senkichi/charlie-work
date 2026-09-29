"""Characterization tests for candidate 3 ("markdown structure"), architecture-
deepening plan `docs/superpowers/plans/2026-09-29-architecture-deepening.md`.

Every hand-rolled fence/blockquote/heading scanner in the repo deviates from
CommonMark somewhere (full inventory: `md-recon.md`, wave A scratchpad). This
module pins each deviation's CURRENT (buggy) output, one test per (consumer,
deviation), so the unification pass that follows (scan side of
`markdown_fence.py`) has a regression net: every one of these tests is
expected to start FAILING the moment a consumer is migrated onto
CommonMark-correct scanning, at which point the flip is a deliberate,
named behavior change -- not a silent one.

Each test's docstring states the flip as "FLIP: current <x>; CommonMark gives
<y>." `outbound_body_guard.py` is the one surveyed consumer with **zero**
deviations (it already matches every CommonMark rule in the recon's matrix),
so it has no test here by design -- not an oversight.

Scope note: these are BLOCK-level fence/heading/blockquote deviations only,
matching what every surveyed consumer actually scans for. None of them do
full CommonMark inline parsing (e.g. inline code spans across paragraph
lines), so this suite doesn't characterize that either.
"""

from __future__ import annotations

import json

import pytest

from _heartbeat_check_fixtures import _load_heartbeat_check
from charlie_work.attachment_contracts.hook_entry import (
    ADVISORY_COMMENT_MARKER,
    parse_advisories_comment,
)
from charlie_work.attachment_contracts.model import AdvisoryRecord
from charlie_work.cross_repo_gate import _nearest_preceding_heading
from charlie_work.github_body_scan import _fenced_block_ranges
from charlie_work.github_prose_dependencies import _is_blockquote_line
from charlie_work.rescue_review import _VERDICT_RE, _find_json_verdict
from charlie_work.verdict_parsing import _extract_verdict_from_text


@pytest.fixture(scope="module")
def hb():
    return _load_heartbeat_check()


# --- github_body_scan.py -----------------------------------------------------


def test_flip_github_body_scan_unbounded_fence_indent() -> None:
    """FLIP: current `_FENCE_OPEN_RE` tolerates any amount of leading
    whitespace and treats a 4-space-indented ``` line as a fence opener,
    capturing an over-indented block; CommonMark bounds fence-open indent to
    0-3 spaces, so a 4+-space-indented ``` never opens a fence at all (it's
    either an indented code block or, lacking a preceding blank line, a lazy
    paragraph continuation) -- `_fenced_block_ranges` should return no range
    here.
    """
    text = "prose before\n    ```\n    #999 inside over-indented fence\n    ```\nprose after\n"
    opener_start = text.index("    ```\n")
    closer_line = "    ```\n"
    content_end = text.index(closer_line, opener_start + len("    ```\n"))

    ranges = _fenced_block_ranges(text)

    assert ranges == [(opener_start, content_end)]
    assert text[opener_start:content_end] == ("    ```\n    #999 inside over-indented fence\n")


def test_flip_github_body_scan_backtick_in_info_string_opens_fence() -> None:
    """FLIP: current `_FENCE_OPEN_RE` never inspects what follows the
    backtick run, so a backtick-fence opener whose info string itself
    contains a backtick (`` ```code`sample ``) is (incorrectly) treated as
    opening a fence; CommonMark requires a backtick fence's info string to
    contain no backtick, so this line should not open a fence at all and the
    two lines that follow should be ordinary prose, not fenced content.
    """
    text = "```code`sample\nshould be prose here, not code\n```\nafter\n"
    opener_start = 0
    content_end = text.index("```\nafter\n")

    ranges = _fenced_block_ranges(text)

    assert ranges == [(opener_start, content_end)]
    assert text[opener_start:content_end] == ("```code`sample\nshould be prose here, not code\n")


# --- cross_repo_gate.py -------------------------------------------------------


def test_flip_cross_repo_gate_backtick_info_string_swallows_heading() -> None:
    """FLIP: current `_nearest_preceding_heading` uses `_FENCE_DELIM_RE`,
    which (like `github_body_scan`'s opener regex) never checks the info
    string for an embedded backtick, so a `` ```code`sample `` line wrongly
    opens a fence and swallows every line after it -- including a real ATX
    heading -- as fence content up to end of text; CommonMark would not open
    a fence there at all, so the swallowed `# References` heading should be
    the nearest preceding heading instead of the one before the fake fence.
    """
    text = "## Real Heading\n```code`sample\n# References\nmore text\n"

    assert _nearest_preceding_heading(text) == "Real Heading"


# --- verdict_parsing.py -------------------------------------------------------


def test_flip_verdict_parsing_desync_embedded_backticks_drops_verdict() -> None:
    """FLIP: current `_VERDICT_FENCE_RE` is a non-line-anchored, non-greedy
    regex that pairs the nearest ``` runs regardless of what line they sit
    on, so a verdict whose `summary` field contains a triple-backtick
    citation desyncs the pairing and `_extract_verdict_from_text` returns
    `None` (the whole verdict is silently lost); CommonMark fences are
    line-anchored (a ``` run has to be the first thing on its line to be a
    delimiter at all), so a CommonMark-correct scanner would see one clean
    fence spanning the whole JSON line and extract the verdict successfully.
    """
    payload = {"decision": "approved", "summary": "see ```py\nx=1\n``` above"}
    text = "```json\n" + json.dumps(payload) + "\n```\n"

    assert _extract_verdict_from_text(text) is None


def test_flip_verdict_parsing_tilde_fence_unsupported() -> None:
    """FLIP: current `_VERDICT_FENCE_RE` only matches backtick runs, so a
    verdict wrapped in a ``~~~`` fence is invisible to
    `_extract_verdict_from_text` and it returns `None`; CommonMark treats
    tilde fences as fully equivalent to backtick fences, so a
    CommonMark-correct scanner would extract the same verdict either way.
    """
    payload = {"decision": "approved", "summary": "ok"}
    text = "~~~json\n" + json.dumps(payload) + "\n~~~\n"

    assert _extract_verdict_from_text(text) is None


def test_flip_verdict_parsing_unclosed_fence_drops_verdict() -> None:
    """FLIP: current `_VERDICT_FENCE_RE` requires a literal closing ``` to
    match at all, so a verdict fence that is never closed (session killed
    mid-output) has no match whatsoever and `_extract_verdict_from_text`
    returns `None`; CommonMark treats an unclosed fence as running to end of
    document, so a CommonMark-correct scanner would still recover the JSON
    verdict from the open fence's content.
    """
    payload = {"decision": "approved", "summary": "ok"}
    text = "```json\n" + json.dumps(payload) + "\n"

    assert _extract_verdict_from_text(text) is None


# --- rescue_review.py ---------------------------------------------------------


def test_flip_rescue_review_desync_embedded_backticks_drops_verdict() -> None:
    """FLIP: current `_VERDICT_FENCE_RE` (byte-identical to
    `verdict_parsing`'s, deliberately duplicated rather than imported) has
    the same non-line-anchored nearest-pair defect, so `_find_json_verdict`
    returns `None` when the `summary` field contains an embedded
    triple-backtick citation; CommonMark's line-anchored fence rule would
    let a correct scanner see one clean fence and extract the verdict.
    """
    payload = {"decision": "approved", "summary": "see ```py\nx=1\n``` above"}
    body = "```json\n" + json.dumps(payload) + "\n```\n"

    assert _find_json_verdict(body) is None


def test_flip_rescue_review_tilde_fence_unsupported() -> None:
    """FLIP: current `_VERDICT_FENCE_RE` only matches backtick runs, so a
    verdict wrapped in a ``~~~`` fence is invisible to `_find_json_verdict`
    and it returns `None`; CommonMark treats tilde fences as fully
    equivalent to backtick fences.
    """
    payload = {"decision": "approved", "summary": "ok"}
    body = "~~~json\n" + json.dumps(payload) + "\n~~~\n"

    assert _find_json_verdict(body) is None


def test_flip_rescue_review_verdict_heading_allows_no_space() -> None:
    """FLIP: current `_VERDICT_RE` accepts `#+\\s*verdict\\b` -- zero
    whitespace required between the `#` run and the text -- so `###verdict`
    (no space) matches as a verdict announcement; CommonMark requires at
    least one space/tab after an ATX heading's `#` run for the line to be a
    heading at all, so `###verdict` is not a heading under CommonMark (it's
    an ordinary paragraph starting with literal `#` characters). This
    looseness is a deliberate cross-family-model tolerance per the module's
    own comment, not an accidental bug, but it is still a CommonMark
    deviation worth pinning.
    """
    text = "###verdict\nApprove.\n"

    assert _VERDICT_RE.search(text) is not None


# --- attachment_contracts/hook_entry.py --------------------------------------


def test_flip_hook_entry_closer_length_mismatch_drops_second_record() -> None:
    """FLIP: current `parse_advisories_comment` treats ANY line starting
    with 3+ backticks as a closer, with no check that its length matches the
    opener's -- the most naive scanner in the recon. A payload opened with 4
    backticks (````json) that happens to contain a bare ``` line closes
    early there, silently truncating the captured fence body to whatever
    parses as valid JSON up to that point and dropping everything after;
    CommonMark requires a closer to be at least as long as its opener (a
    3-backtick line can never close a 4-backtick-opened fence), so a
    CommonMark-correct scanner would keep reading past the decoy line to the
    real (4-backtick) closer.
    """
    first_record = [
        {
            "severity": "info",
            "file": "x.py",
            "identity": "id1",
            "message": "first record",
            "redirect": None,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ]
    second_record = [
        {
            "severity": "warning",
            "file": "y.py",
            "identity": "id2",
            "message": "second record",
            "redirect": None,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ]
    body = (
        ADVISORY_COMMENT_MARKER
        + "\n````json\n"
        + json.dumps(first_record)
        + "\n```\n"
        + json.dumps(second_record)
        + "\n````\n"
    )

    result = parse_advisories_comment(body)

    assert result == (
        AdvisoryRecord(
            severity="info",
            file="x.py",
            identity="id1",
            message="first record",
            redirect=None,
            timestamp="2026-01-01T00:00:00Z",
        ),
    )


def test_flip_hook_entry_tilde_fence_unsupported() -> None:
    """FLIP: current `parse_advisories_comment` only recognizes a backtick
    (```` ``` ````) opener, so a payload wrapped in a ``~~~`` fence is never
    found and it returns `()` (the "marker present, fence missing/malformed"
    sentinel), discarding a well-formed record; CommonMark treats tilde
    fences as fully equivalent to backtick fences, so a CommonMark-correct
    scanner would find and parse the same payload.
    """
    records = [
        {
            "severity": "info",
            "file": "x.py",
            "identity": "id1",
            "message": "m",
            "redirect": None,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ]
    body = ADVISORY_COMMENT_MARKER + "\n~~~json\n" + json.dumps(records) + "\n~~~\n"

    assert parse_advisories_comment(body) == ()


# --- scripts/heartbeat_check.py -----------------------------------------------


def test_flip_heartbeat_check_desync_embedded_backticks_leaks_mention(hb) -> None:
    """FLIP: current `_FENCED_CODE_BLOCK_RE` (```` ```.*?``` ```` DOTALL,
    non-line-anchored) pairs the nearest two ``` runs, so a real fenced
    block whose own content contains a literal ``` substring desyncs the
    pairing -- the stripped text no longer removes the middle of the block,
    leaking `#111` (which is genuinely inside the fence) into
    `_mentioned_issue_numbers`'s result alongside the legitimately-outside
    `#222`; CommonMark's line-anchored fence rule would strip the whole
    block as one clean unit, leaving only `#222`.
    """
    text = '```json\n{"note": "```see #111 inside```", "id": 2}\n```\nAlso mentions #222 here.\n'

    assert hb._mentioned_issue_numbers(text) == {111, 222}


def test_flip_heartbeat_check_tilde_fence_unsupported_leaks_mention(hb) -> None:
    """FLIP: current `_FENCED_CODE_BLOCK_RE` only matches backtick runs, so
    a ``~~~``-fenced block is never stripped and `#123` (genuinely inside
    the tilde-fenced block) leaks into `_mentioned_issue_numbers`'s result
    alongside the legitimately-outside `#456`; CommonMark treats tilde
    fences as fully equivalent to backtick fences, so a CommonMark-correct
    scanner would strip the block and return only `#456`.
    """
    text = "~~~\n#123 mentioned inside a tilde-fenced block\n~~~\nalso #456\n"

    assert hb._mentioned_issue_numbers(text) == {123, 456}


# --- github_prose_dependencies.py --------------------------------------------


def test_flip_github_prose_dependencies_blockquote_unbounded_indent() -> None:
    """FLIP: current `_is_blockquote_line` strips all leading spaces/tabs
    before checking for `>`, so a `>` preceded by any amount of indentation
    -- including 4+ spaces -- still counts as a blockquote line; CommonMark
    bounds a blockquote marker's indent to 0-3 spaces (4+ makes it an
    indented code block instead, where `>` is just a literal character), so
    a 5-space-indented `>` line is not a blockquote marker under CommonMark.
    """
    line = "     > deeply indented quote (5 spaces)"

    assert _is_blockquote_line(line) is True
