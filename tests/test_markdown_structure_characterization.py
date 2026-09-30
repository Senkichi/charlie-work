"""Characterization tests for candidate 3 ("markdown structure"), architecture-
deepening plan `docs/superpowers/plans/2026-09-29-architecture-deepening.md`.

Every hand-rolled fence/blockquote/heading scanner in the repo deviated from
CommonMark somewhere (full inventory: `md-recon.md`, wave A scratchpad). This
module originally pinned each deviation's buggy output; now that every listed
consumer (`github_body_scan.py`, `cross_repo_gate.py`, `verdict_parsing.py`,
`rescue_review.py`, `attachment_contracts/hook_entry.py`,
`scripts/heartbeat_check.py`, `github_prose_dependencies.py`) is wired onto
`markdown_fence.scan`, these tests instead pin the CommonMark-correct output,
so a future regression back to ad hoc scanning fails loudly here.

Each flipped test's docstring states "FLIP: was <x>, now <y>." One test is
deliberately NOT flipped, and says why inline:

- `test_flip_rescue_review_verdict_heading_allows_no_space`: a deliberate
  cross-family-model tolerance, out of this wiring pass's scope.

`test_flip_cross_repo_gate_backtick_info_string_no_longer_swallows_heading`
WAS in that "not flipped" list (its info-string swallow was unaffected by
wiring alone), but a later fix to `markdown_fence.scan` itself -- a rejected
fence opener no longer swallows the region up to its nearest closer;
adversarial review finding B1, architecture-deepening candidate 3 -- flips
it too. See that test's own docstring.

`outbound_body_guard.py` is the one surveyed consumer with **zero**
deviations (it already matched every CommonMark rule in the recon's matrix
before this pass -- it was the module `markdown_fence.py` was ported from),
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
from charlie_work.rescue_review import CrossFamilyVerdict
from charlie_work.verdict_parsing import _extract_verdict_from_text


@pytest.fixture(scope="module")
def hb():
    return _load_heartbeat_check()


# --- github_body_scan.py -----------------------------------------------------


def test_github_body_scan_fence_indent_is_container_tolerant() -> None:
    """NOT a flip (md-r2 review B1): `_fenced_block_ranges` is an
    *exclusion* guard, so it keeps the old any-indent tolerance for fence
    openers/closers by scanning with `markdown_fence.scan(max_indent=None)`.
    `scan` has no list-item container, and inside a list item CommonMark
    re-bases content to the item's content column, so a 4+-space-indented
    ``` under `1. ` / `- ` IS a fence. The top-level 0-3 bound would
    under-approximate the quoted region and let `Blocked by #N` inside it
    become a live blocker. (For a top-level over-indented ``` this
    over-approximates -- the fail-safe direction for an exclusion guard.)
    """
    text = "prose before\n    ```\n    #999 inside over-indented fence\n    ```\nprose after\n"

    ((start, end),) = _fenced_block_ranges(text)
    assert text[start:end] == "    ```\n    #999 inside over-indented fence\n"


def test_flip_github_body_scan_backtick_in_info_string_opens_fence() -> None:
    """FLIP: was a captured range covering the ``` lines (the old
    `_FENCE_OPEN_RE` never inspected what follows the backtick run, so
    `` ```code`sample `` was treated as opening a fence); now, wired onto
    `markdown_fence.scan`, `_fenced_block_ranges` returns a DIFFERENT
    non-empty range: char offsets 46-56, covering the line-3 ``` and
    "after" -- not the empty range `[]` an earlier version of this test
    asserted.

    CommonMark requires a backtick fence's info string to contain no
    backtick, so `` ```code`sample `` (line 0) never opens a fence -- but
    that line is ordinary prose, NOT the start of a "malformed fence"
    region to skip (adversarial review finding B1, architecture-deepening
    candidate 3: `markdown_fence.scan` used to still swallow such a region
    without recording a span for it). The line-2 ``` is therefore free to
    open its own (unclosed) fence, running to end-of-text -- verified
    against markdown-it-py in commonmark mode.
    """
    text = "```code`sample\nshould be prose here, not code\n```\nafter\n"

    assert _fenced_block_ranges(text) == [(46, 56)]


# --- cross_repo_gate.py -------------------------------------------------------


def test_flip_cross_repo_gate_backtick_info_string_no_longer_swallows_heading() -> None:
    """FLIP: was `"Real Heading"`, now `"References"`.

    Immediately after wiring onto `markdown_fence.scan`, this did NOT flip
    (verified, not assumed): `markdown_fence.scan` at that point still
    *skipped* the lines between a rejected opener (the backtick in
    `` ```code`sample ``'s info string disqualifies it under CommonMark)
    and its nearest same-char, length-matching closer -- or end of text, if
    none exists -- without recording a `FenceSpan` for the skipped region.
    `_nearest_preceding_heading` only reads `structure.headings`, which was
    never populated for a skipped region regardless of whether the opener
    that triggered the skip was valid, so whether the region counted as a
    "real" fence was invisible to this function either way -- it returned
    `"Real Heading"`, matching the old hand-rolled `_FENCE_DELIM_RE`, which
    swallowed the same region for the same reason (any backtick run of
    length >= 3 opened a fence, no info-string check, then closed on the
    next same-length run or ran to EOF).

    That "still skips a rejected opener's region" design was itself a
    CommonMark deviation (adversarial review finding B1, architecture-
    deepening candidate 3): a rejected opener is not a fence at all, so the
    line right after it (`# References`) is ordinary text, eligible to be
    read as a heading like any other line -- confirmed against
    markdown-it-py in commonmark mode. Fixing B1 in `markdown_fence.scan`
    removes the skip entirely, so `# References` (line 2) IS now seen as a
    real heading, and being the last one before the end of the text, it
    becomes the "nearest preceding heading" for a candidate after it.
    """
    text = "## Real Heading\n```code`sample\n# References\nmore text\n"

    assert _nearest_preceding_heading(text) == "References"


# --- verdict_parsing.py -------------------------------------------------------


def test_flip_verdict_parsing_desync_embedded_backticks_drops_verdict() -> None:
    """FLIP: was `None` (the old `_VERDICT_FENCE_RE` is a non-line-anchored,
    non-greedy regex that pairs the nearest ``` runs regardless of what line
    they sit on, so a verdict whose `summary` field contains a
    triple-backtick citation desynced the pairing and the whole verdict was
    silently lost); now, wired onto `markdown_fence.scan`,
    `_extract_verdict_from_text` returns the recovered verdict dict.
    CommonMark fences are line-anchored (a ``` run has to be the first thing
    on its line to be a delimiter at all), so the embedded ```py citation --
    which sits mid-line inside the JSON string, not at line-start -- is never
    treated as a delimiter, and the real fence is read as one clean span.
    """
    payload = {"decision": "approved", "summary": "see ```py\nx=1\n``` above"}
    text = "```json\n" + json.dumps(payload) + "\n```\n"

    assert _extract_verdict_from_text(text) == {
        "decision": "approved",
        "summary": "see ```py\nx=1\n``` above",
        "required_changes": [],
    }


def test_flip_verdict_parsing_tilde_fence_unsupported() -> None:
    """FLIP: was `None` (the old `_VERDICT_FENCE_RE` only matched backtick
    runs, so a verdict wrapped in a ``~~~`` fence was invisible to
    `_extract_verdict_from_text`); now, wired onto `markdown_fence.scan`,
    it extracts the verdict. CommonMark treats tilde fences as fully
    equivalent to backtick fences.
    """
    payload = {"decision": "approved", "summary": "ok"}
    text = "~~~json\n" + json.dumps(payload) + "\n~~~\n"

    assert _extract_verdict_from_text(text) == {
        "decision": "approved",
        "summary": "ok",
        "required_changes": [],
    }


def test_flip_verdict_parsing_unclosed_fence_drops_verdict() -> None:
    """FLIP: was `None` (the old `_VERDICT_FENCE_RE` required a literal
    closing ``` to match at all, so a verdict fence never closed -- session
    killed mid-output -- had no match whatsoever); now, wired onto
    `markdown_fence.scan`, `_extract_verdict_from_text` recovers the verdict
    from the open fence's content. CommonMark treats an unclosed fence as
    running to end of document.
    """
    payload = {"decision": "approved", "summary": "ok"}
    text = "```json\n" + json.dumps(payload) + "\n"

    assert _extract_verdict_from_text(text) == {
        "decision": "approved",
        "summary": "ok",
        "required_changes": [],
    }


# --- rescue_review.py ---------------------------------------------------------


def test_flip_rescue_review_desync_embedded_backticks_drops_verdict() -> None:
    """FLIP: was `None` (the old `_VERDICT_FENCE_RE`, byte-identical to
    `verdict_parsing`'s pre-wiring regex, had the same non-line-anchored
    nearest-pair defect, so `_find_json_verdict` lost a verdict whose
    `summary` field contained an embedded triple-backtick citation); now,
    wired onto `markdown_fence.scan`, it returns the recovered
    `CrossFamilyVerdict`. CommonMark's line-anchored fence rule reads the
    real fence as one clean span, same as `verdict_parsing.py`'s flip above.
    """
    payload = {"decision": "approved", "summary": "see ```py\nx=1\n``` above"}
    body = "```json\n" + json.dumps(payload) + "\n```\n"

    assert _find_json_verdict(body) == CrossFamilyVerdict(
        decision="approved",
        summary="see ```py\nx=1\n``` above",
        required_changes=(),
    )


def test_flip_rescue_review_tilde_fence_unsupported() -> None:
    """FLIP: was `None` (the old `_VERDICT_FENCE_RE` only matched backtick
    runs, so a verdict wrapped in a ``~~~`` fence was invisible to
    `_find_json_verdict`); now, wired onto `markdown_fence.scan`, it returns
    the recovered `CrossFamilyVerdict`. CommonMark treats tilde fences as
    fully equivalent to backtick fences.
    """
    payload = {"decision": "approved", "summary": "ok"}
    body = "~~~json\n" + json.dumps(payload) + "\n~~~\n"

    assert _find_json_verdict(body) == CrossFamilyVerdict(
        decision="approved", summary="ok", required_changes=()
    )


def test_flip_rescue_review_verdict_heading_allows_no_space() -> None:
    """NOT FLIPPED, deliberately, and out of this wiring pass's scope:
    `_VERDICT_RE` still accepts `#+\\s*verdict\\b` -- zero whitespace
    required between the `#` run and the text -- so `###verdict` (no space)
    still matches as a verdict announcement. CommonMark requires at least
    one space/tab after an ATX heading's `#` run for the line to be a
    heading at all, so `###verdict` is not a CommonMark heading -- but
    `_VERDICT_RE`/`_SEVERITY_RE` are deliberate cross-family-model-tolerance
    detectors (per the module's own comment; some reviewer models, e.g.
    kimi-k3-style output, emit `###verdict` with no space), not accidental
    CommonMark bugs. The architecture-deepening plan's consumer citation for
    this file (`rescue_review.py:110`) points at the fence regex, which *is*
    wired above -- not at these heading regexes, which stay untouched.
    Wiring them to strict CommonMark heading rules would regress real
    verdict detection for those models, not fix a bug.
    """
    text = "###verdict\nApprove.\n"

    assert _VERDICT_RE.search(text) is not None


# --- attachment_contracts/hook_entry.py --------------------------------------


def test_flip_hook_entry_closer_length_mismatch_drops_second_record() -> None:
    """FLIP: was `(first_record,)` (the old `parse_advisories_comment`
    treated ANY line starting with 3+ backticks as a closer, with no check
    that its length matched the opener's, so a payload opened with 4
    backticks (````json) that happened to contain a bare ``` line closed
    early there, silently truncating the captured fence body to whatever
    parsed as valid JSON up to that point and dropping the second record
    with no signal); now, wired onto `markdown_fence.scan`,
    `parse_advisories_comment` returns `()`. CommonMark requires a closer to
    be at least as long as its opener (a 3-backtick line can never close a
    4-backtick-opened fence), so the decoy ``` line no longer ends the
    fence early -- the scanner correctly reads through to the real
    (4-backtick) closer. But the resulting fence body is then the first
    record's JSON, a literal ``` line, and the second record's JSON all
    concatenated -- not one valid JSON document -- so `json.loads` fails and
    `parse_advisories_comment` reports the "marker present, fence
    missing/malformed" sentinel rather than fabricating a partial result.
    That's the correct behavior change: silently returning one of two
    records with no error (the old behavior) is worse than a loud,
    unambiguous "malformed" signal, because a caller trusting `len(result)`
    as a count of advisories would silently under-count in the old
    behavior. The real bug this exposes -- a payload that legitimately needs
    a decoy-proof opener/closer length pair should not use content that
    itself contains a same-or-shorter backtick run -- belongs to whatever
    constructs the advisories comment, not to the scanner.
    """
    first_record = [
        {
            "severity": "advise",
            "file": "x.py",
            "identity": "id1",
            "message": "first record",
            "redirect": None,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ]
    second_record = [
        {
            "severity": "block",
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

    assert result == ()


def test_flip_hook_entry_tilde_fence_unsupported() -> None:
    """FLIP: was `()` (the old `parse_advisories_comment` only recognized a
    backtick (```` ``` ````) opener, so a payload wrapped in a ``~~~`` fence
    was never found, discarding a well-formed record); now, wired onto
    `markdown_fence.scan`, it finds and parses the record. CommonMark treats
    tilde fences as fully equivalent to backtick fences.
    """
    records = [
        {
            "severity": "advise",
            "file": "x.py",
            "identity": "id1",
            "message": "m",
            "redirect": None,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ]
    body = ADVISORY_COMMENT_MARKER + "\n~~~json\n" + json.dumps(records) + "\n~~~\n"

    assert parse_advisories_comment(body) == (
        AdvisoryRecord(
            severity="advise",
            file="x.py",
            identity="id1",
            message="m",
            redirect=None,
            timestamp="2026-01-01T00:00:00Z",
        ),
    )


# --- scripts/heartbeat_check.py -----------------------------------------------


def test_flip_heartbeat_check_desync_embedded_backticks_leaks_mention(hb) -> None:
    """FLIP: was `{111, 222}` (the old `_FENCED_CODE_BLOCK_RE`
    (```` ```.*?``` ```` DOTALL, non-line-anchored) paired the nearest two
    ``` runs, so a real fenced block whose own content contained a literal
    ``` substring desynced the pairing -- the stripped text no longer
    removed the middle of the block, leaking `#111` (genuinely inside the
    fence) into `_mentioned_issue_numbers`'s result alongside the
    legitimately-outside `#222`); now, wired onto the module's line-indexed
    scan (`_scan_markdown_structure`/`_strip_fenced_code_blocks`), the whole
    block strips as one clean unit and only `#222` remains. CommonMark's
    line-anchored fence rule is what makes the mid-line ``` inside the JSON
    string invisible as a delimiter, same as the `verdict_parsing.py`/
    `rescue_review.py` desync flips above.
    """
    text = '```json\n{"note": "```see #111 inside```", "id": 2}\n```\nAlso mentions #222 here.\n'

    assert hb._mentioned_issue_numbers(text) == {222}


def test_flip_heartbeat_check_tilde_fence_unsupported_leaks_mention(hb) -> None:
    """FLIP: was `{123, 456}` (the old `_FENCED_CODE_BLOCK_RE` only matched
    backtick runs, so a ``~~~``-fenced block was never stripped and `#123`
    (genuinely inside the tilde-fenced block) leaked into
    `_mentioned_issue_numbers`'s result alongside the legitimately-outside
    `#456`); now, wired onto the module's scan, the tilde-fenced block
    strips and only `#456` remains. CommonMark treats tilde fences as fully
    equivalent to backtick fences.
    """
    text = "~~~\n#123 mentioned inside a tilde-fenced block\n~~~\nalso #456\n"

    assert hb._mentioned_issue_numbers(text) == {456}


# --- github_prose_dependencies.py --------------------------------------------


def test_github_prose_dependencies_blockquote_is_container_tolerant() -> None:
    """NOT a flip (md-r2 review B1): `_is_blockquote_line` is an *exclusion*
    guard, so it keeps the old any-indent tolerance
    (`is_blockquote_marker(max_indent=None)`). A `>` indented 4+ columns
    under a list item is a real blockquote (CommonMark re-bases list-item
    content), and the shared scan has no list-item container to say so.
    """
    line = "     > deeply indented quote (5 spaces)"

    assert _is_blockquote_line(line) is True
    assert _is_blockquote_line("\t> tab-indented quote") is True
    assert _is_blockquote_line("not > a quote") is False
