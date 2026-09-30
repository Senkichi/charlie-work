"""Verdict fence extraction must never be more permissive than origin/main.

md-r2 re-review 2, B1: `_extract_verdict_from_text` and
`rescue_review._find_json_verdict` moved from the unanchored
`_VERDICT_FENCE_RE` to the line-anchored `markdown_fence.scan`, which misses
fence shapes the regex caught (opener mid-line, closer glued to the closing
brace, list-nested or tab-indented fence). With an earlier `approved` fence in
the same text, the final `request_changes` block was silently dropped and the
earlier approval won -- a verdict guard failing open. Both extractors now take
the union of both models, ordered by source position ("last verdict wins").
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from charlie_work import markdown_fence
from charlie_work.rescue_review import _find_json_verdict
from charlie_work.verdict_parsing import _VERDICT_FENCE_RE, _extract_verdict_from_text

_APPROVED = json.dumps({"decision": "approved", "summary": "looks fine"})
_REQUEST = json.dumps({"decision": "request_changes", "summary": "fix it"})
_DRAFT = f"```json\n{_APPROVED}\n```\nRevised:\n"


def _primary(text: str) -> str | None:
    verdict = _extract_verdict_from_text(text)
    return None if verdict is None else verdict["decision"]


def _cross_family(text: str) -> str | None:
    verdict = _find_json_verdict(text)
    return None if verdict is None else verdict.decision


_EXTRACTORS: list[Callable[[str], str | None]] = [_primary, _cross_family]

# Final-block shapes the old unanchored regex caught but scan() does not.
_FINAL_SHAPES: dict[str, str] = {
    "mid-line-opener": f"My verdict: ```json\n{_REQUEST}\n```",
    "glued-closer": f"```json\n{_REQUEST}```",
    "list-nested-4col": f"- verdict:\n\n    ```json\n    {_REQUEST}\n    ```",
    "tab-indented": f"\t```json\n{_REQUEST}\n\t```",
}


@pytest.mark.parametrize("extract", _EXTRACTORS, ids=["primary", "cross_family"])
@pytest.mark.parametrize("shape", sorted(_FINAL_SHAPES))
def test_final_block_in_legacy_only_shape_is_still_found(
    extract: Callable[[str], str | None], shape: str
) -> None:
    assert extract(_FINAL_SHAPES[shape]) == "request_changes"


@pytest.mark.parametrize("extract", _EXTRACTORS, ids=["primary", "cross_family"])
@pytest.mark.parametrize("shape", sorted(_FINAL_SHAPES))
def test_final_request_changes_beats_earlier_approved_fence(
    extract: Callable[[str], str | None], shape: str
) -> None:
    """The fail-open case: earlier approved fence + final legacy-only-shape
    request_changes must resolve to request_changes, never approved."""
    assert extract(_DRAFT + _FINAL_SHAPES[shape]) == "request_changes"


@pytest.mark.parametrize("extract", _EXTRACTORS, ids=["primary", "cross_family"])
def test_control_scan_and_legacy_agree_on_a_well_formed_pair(
    extract: Callable[[str], str | None],
) -> None:
    """Positive control: well-formed fences still resolve last-wins, and the
    reverse order (final approved) resolves to approved -- the harness can
    report either decision."""
    assert extract(f"{_DRAFT}```json\n{_REQUEST}\n```\n") == "request_changes"
    assert (
        extract(f"```json\n{_REQUEST}\n```\nRevised:\n```json\n{_APPROVED}\n```\n") == "approved"
    )


@pytest.mark.parametrize("extract", _EXTRACTORS, ids=["primary", "cross_family"])
def test_scan_only_shapes_are_still_found(extract: Callable[[str], str | None]) -> None:
    """The line-anchored gains survive: tilde fence and unclosed fence."""
    assert extract(f"~~~json\n{_REQUEST}\n~~~\n") == "request_changes"
    assert extract(f"```json\n{_REQUEST}\n") == "request_changes"


def test_legacy_regex_is_the_single_shared_definition() -> None:
    assert _VERDICT_FENCE_RE is markdown_fence.LEGACY_FENCE_RE


def test_fence_contents_are_ordered_latest_first() -> None:
    text = "```\nfirst\n```\ntext ```\nsecond```\n~~~\nthird\n~~~\n"
    bodies: list[Any] = [b.strip() for b in markdown_fence.fence_contents_latest_first(text)]
    assert bodies.index("third") < bodies.index("second") < bodies.index("first")


def test_strip_fenced_blocks_pins_line_anchored_behaviour() -> None:
    """md-r2 re-review 2, N2: `_strip_fenced_blocks` (plaintext-log summary
    fallback) is deliberately line-anchored -- cosmetic only, pinned so the
    delta vs origin/main's closer-requiring regex is visible. An unclosed
    fence drops the rest of the log; a mid-line opener is NOT a fence."""
    from charlie_work.verdict_parsing import _strip_fenced_blocks

    assert _strip_fenced_blocks("keep\n```json\ndrop\n```\nkeep2\n") == "keep\nkeep2\n"
    assert _strip_fenced_blocks("keep\n```json\ndrop to EOF\n") == "keep\n"
    assert _strip_fenced_blocks("say ```json\nkept\n").startswith("say ```json\nkept\n")


# md-r3 review B1: the mirror image of md-r2-2 B1. A later fence that only
# `scan` recognises (tilde block, unclosed trailing block) must not override a
# `request_changes` that origin/main's legacy regex already returned.
_SCAN_ONLY_AFTER_LEGACY: dict[str, str] = {
    "tilde-after": f"```json\n{_REQUEST}\n```\n~~~json\n{_APPROVED}\n~~~\n",
    "unclosed-after": f"```json\n{_REQUEST}\n```\n\n```json\n{_APPROVED}\n",
}


@pytest.mark.parametrize("extract", _EXTRACTORS, ids=["primary", "cross_family"])
@pytest.mark.parametrize("shape", sorted(_SCAN_ONLY_AFTER_LEGACY))
def test_scan_only_approval_after_legacy_request_changes_does_not_win(
    extract: Callable[[str], str | None], shape: str
) -> None:
    assert extract(_SCAN_ONLY_AFTER_LEGACY[shape]) == "request_changes"


@pytest.mark.parametrize("extract", _EXTRACTORS, ids=["primary", "cross_family"])
def test_scan_only_request_changes_after_legacy_approved_still_wins(
    extract: Callable[[str], str | None],
) -> None:
    """Positive control: the conservative clamp is one-directional -- a later
    scan-only `request_changes` still beats an earlier legacy `approved`."""
    text = f"```json\n{_APPROVED}\n```\n~~~json\n{_REQUEST}\n~~~\n"
    assert extract(text) == "request_changes"
