"""The composed markdown guards are monotone: never weaker than origin/main.

md-r4 replaced three rounds of point fixes to the verdict-fence and
example-secret guards with a composition that is monotone by construction:

* verdict extraction returns the MORE SEVERE of origin/main's exact extraction
  (kept as a private ``_legacy_*`` function) and the ``markdown_fence.scan``
  result (blocked > request_changes > no verdict > approved);
* example-secret masking masks a character only if BOTH origin/main's exact
  masking AND the scan-based masking mask it (intersection).

These tests are property-style: they enumerate deterministic combinations of
fence shapes (no ``hypothesis`` dependency) and assert the properties for every
one, instead of pinning individual inputs. They also pin that
``markdown_guard_disagreement`` is emitted on disagreement and only then.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from charlie_work import markdown_guard
from charlie_work.instrumentation import close_db, query_events
from charlie_work.markdown_guard import Disagreement, decision_severity
from charlie_work.outbound_body_guard import (
    _legacy_mask_ranges,
    _mask_example_secret_fences,
    _scan_mask_ranges,
    scan_outbound_text,
)
from charlie_work.rescue_review import (
    _find_json_verdict,
    _legacy_find_json_verdict,
    _scan_find_json_verdict,
)
from charlie_work.verdict_parsing import (
    _extract_verdict_from_text,
    _legacy_extract_verdict_from_text,
    _scan_extract_verdict_from_text,
)

_APPROVED = json.dumps({"decision": "approved", "summary": "looks fine"})
_REQUEST = json.dumps({"decision": "request_changes", "summary": "fix it"})
_BLOCKED = json.dumps({"decision": "blocked", "summary": "no"})
_DRAFT = f"```json\n{_APPROVED}\n```\nRevised:\n"

# Final-block shapes the old unanchored regex caught but scan() does not.
_FINAL_SHAPES: dict[str, str] = {
    "mid-line-opener": f"My verdict: ```json\n{_REQUEST}\n```",
    "glued-closer": f"```json\n{_REQUEST}```",
    "list-nested-4col": f"- verdict:\n\n    ```json\n    {_REQUEST}\n    ```",
    "tab-indented": f"\t```json\n{_REQUEST}\n\t```",
}

# md-r3 review B1 inputs: a fence only `scan` recognises after a legacy request_changes.
_SCAN_ONLY_AFTER_LEGACY: dict[str, str] = {
    "tilde-after": f"```json\n{_REQUEST}\n```\n~~~json\n{_APPROVED}\n~~~\n",
    "unclosed-after": f"```json\n{_REQUEST}\n```\n\n```json\n{_APPROVED}\n",
}


# --- verdict extractors under test ------------------------------------------------


def _label(verdict: object) -> str:
    if verdict is None:
        return markdown_guard.NO_VERDICT
    if isinstance(verdict, dict):
        return str(verdict["decision"])
    return str(verdict.decision)  # type: ignore[attr-defined]


_Extractor = Callable[..., object]
# (composed, legacy, scan-only) for the primary and the cross-family verdict guards.
_GUARDS: dict[str, tuple[_Extractor, _Extractor, _Extractor]] = {
    "primary": (
        _extract_verdict_from_text,
        _legacy_extract_verdict_from_text,
        _scan_extract_verdict_from_text,
    ),
    "cross_family": (_find_json_verdict, _legacy_find_json_verdict, _scan_find_json_verdict),
}
_GUARD_IDS = sorted(_GUARDS)


def _verdict_chunks() -> list[str]:
    bodies = {"approved": _APPROVED, "request_changes": _REQUEST, "blocked": _BLOCKED}
    chunks = ["no verdict here\n", _DRAFT]
    for body in bodies.values():
        chunks += [
            f"```json\n{body}\n```\n",
            f"~~~json\n{body}\n~~~\n",
            f"```json\n{body}\n",
            f"My verdict: ```json\n{body}\n```\n",
            f"```json\n{body}```\n",
            f"\t```json\n{body}\n\t```\n",
            f"- v:\n\n    ```json\n    {body}\n    ```\n",
        ]
    chunks += list(_FINAL_SHAPES.values()) + list(_SCAN_ONLY_AFTER_LEGACY.values())
    return chunks


def _verdict_corpus() -> list[str]:
    chunks = _verdict_chunks()
    corpus: list[str] = []
    for size in (1, 2, 3):
        corpus.extend("\n".join(combo) for combo in itertools.product(chunks, repeat=size))
    return corpus


_VERDICT_CORPUS = _verdict_corpus()


@pytest.mark.parametrize("guard", _GUARD_IDS)
def test_composed_verdict_is_never_less_severe_than_legacy(guard: str) -> None:
    composed, legacy, scan = _GUARDS[guard]
    seen_stricter = seen_equal = False
    for text in _VERDICT_CORPUS:
        c, lg, sc = _label(composed(text)), _label(legacy(text)), _label(scan(text))
        assert decision_severity(c) >= decision_severity(lg), (text, c, lg)
        assert decision_severity(c) >= decision_severity(sc), (text, c, sc)
        if c == "approved":
            assert lg == "approved" and sc == "approved", (text, lg, sc)
        seen_stricter |= decision_severity(c) > decision_severity(lg)
        seen_equal |= c == lg
    # Controls: the corpus contains inputs where the new side tightens the
    # result and inputs where both sides agree, so neither branch is vacuous.
    assert seen_stricter and seen_equal


@pytest.mark.parametrize("guard", _GUARD_IDS)
def test_disagreement_event_fires_only_on_disagreement(guard: str) -> None:
    composed, legacy, scan = _GUARDS[guard]
    fired = quiet = 0
    for text in _VERDICT_CORPUS:
        events: list[Disagreement] = []
        result = composed(text, on_disagreement=events.append)
        lg, sc = _label(legacy(text)), _label(scan(text))
        if lg == sc:
            assert events == [], text
            quiet += 1
        else:
            assert len(events) == 1, text
            event = events[0]
            assert (event.legacy, event.new) == (lg, sc)
            assert event.chosen == _label(result)
            fired += 1
    assert fired and quiet


@pytest.mark.parametrize("guard", _GUARD_IDS)
@pytest.mark.parametrize("shape", sorted(_FINAL_SHAPES))
def test_legacy_only_final_shape_beats_earlier_approved_fence(guard: str, shape: str) -> None:
    """md-r2-2 B1: the final legacy-only-shape request_changes must win."""
    composed = _GUARDS[guard][0]
    assert _label(composed(_FINAL_SHAPES[shape])) == "request_changes"
    assert _label(composed(_DRAFT + _FINAL_SHAPES[shape])) == "request_changes"


@pytest.mark.parametrize("guard", _GUARD_IDS)
@pytest.mark.parametrize("shape", sorted(_SCAN_ONLY_AFTER_LEGACY))
def test_scan_only_approval_after_legacy_request_changes_does_not_win(
    guard: str, shape: str
) -> None:
    """md-r3 B1 (the mirror image)."""
    assert _label(_GUARDS[guard][0](_SCAN_ONLY_AFTER_LEGACY[shape])) == "request_changes"


@pytest.mark.parametrize("guard", _GUARD_IDS)
def test_positive_controls_agreeing_pair_resolves_last_wins(guard: str) -> None:
    """The harness can report either decision on a well-formed pair, silently."""
    composed = _GUARDS[guard][0]
    events: list[Disagreement] = []
    text = f"{_DRAFT}```json\n{_REQUEST}\n```\n"
    assert _label(composed(text, on_disagreement=events.append)) == "request_changes"
    text = f"```json\n{_REQUEST}\n```\nRevised:\n```json\n{_APPROVED}\n```\n"
    assert _label(composed(text, on_disagreement=events.append)) == "approved"
    assert events == []


def test_unclosed_approved_fence_is_clamped_to_no_verdict() -> None:
    events: list[Disagreement] = []
    text = f"```json\n{_APPROVED}\n"
    assert _extract_verdict_from_text(text, on_disagreement=events.append) is None
    assert events == [Disagreement(markdown_guard.GUARD_VERDICT, "none", "approved", "none")]


def test_unclosed_blocked_fence_is_more_severe_than_legacy_and_wins() -> None:
    """Positive control for the 'new side is more severe' direction."""
    events: list[Disagreement] = []
    verdict = _extract_verdict_from_text(f"```json\n{_BLOCKED}\n", on_disagreement=events.append)
    assert verdict is not None and verdict["decision"] == "blocked"
    assert events == [Disagreement(markdown_guard.GUARD_VERDICT, "none", "blocked", "blocked")]


def test_tilde_approved_after_legacy_request_changes_reports_disagreement() -> None:
    events: list[Disagreement] = []
    text = _SCAN_ONLY_AFTER_LEGACY["tilde-after"]
    verdict = _extract_verdict_from_text(text, on_disagreement=events.append)
    assert verdict is not None and verdict["decision"] == "request_changes"
    assert events == [
        Disagreement(
            markdown_guard.GUARD_VERDICT, "request_changes", "approved", "request_changes"
        )
    ]


def test_rescue_verdict_disagreement_uses_its_own_guard_name() -> None:
    events: list[Disagreement] = []
    _find_json_verdict(f"~~~json\n{_APPROVED}\n~~~\n", on_disagreement=events.append)
    assert [e.guard for e in events] == [markdown_guard.GUARD_RESCUE_VERDICT]


def test_mid_line_opener_final_request_changes_after_earlier_approved() -> None:
    events: list[Disagreement] = []
    text = _DRAFT + _FINAL_SHAPES["mid-line-opener"]
    verdict = _extract_verdict_from_text(text, on_disagreement=events.append)
    assert verdict is not None and verdict["decision"] == "request_changes"
    assert [(e.legacy, e.new, e.chosen) for e in events] == [
        ("request_changes", "approved", "request_changes")
    ]


def test_strip_fenced_blocks_pins_line_anchored_behaviour() -> None:
    """`_strip_fenced_blocks` (plaintext-log summary fallback) is line-anchored;
    cosmetic only, pinned so the delta vs origin/main is visible."""
    from charlie_work.verdict_parsing import _strip_fenced_blocks

    assert _strip_fenced_blocks("keep\n```json\ndrop\n```\nkeep2\n") == "keep\nkeep2\n"
    assert _strip_fenced_blocks("keep\n```json\ndrop to EOF\n") == "keep\n"
    assert _strip_fenced_blocks("say ```json\nkept\n").startswith("say ```json\nkept\n")


# --- outbound example-secret mask ----------------------------------------------------

_KEY = "KEYKEYKEY"
_OUTBOUND_CHUNKS: list[str] = [
    "text\n",
    f"```example-secret\n{_KEY}\n```\n",
    f"~~~example-secret\n{_KEY}\n~~~\n",
    f"```example-secret\n{_KEY}\n\t```\n",
    f"```example-secret\n{_KEY}\n    ```\n",
    f"```example-secret\n{_KEY}\n",
    f"```\n{_KEY}\n```\n",
    f"\t```\n{_KEY}\n```\n",
    f"\t```example-secret\n{_KEY}\n```\n",
    f"```x`y\n```\n```example-secret\n{_KEY}\n```\n",
    f"x\r```example-secret\r{_KEY}\r```\r",
    f"prose\u2028```example-secret\n{_KEY}\n```\n",
    f"~~~\n\t\t~~~example-secret\n{_KEY}\n~~~\n",
    f"- ```example-secret\n  {_KEY}\n  ```\n",
    f"````example-secret\n{_KEY}\n```\n````\n",
]


def _outbound_corpus() -> list[str]:
    corpus: list[str] = []
    for size in (1, 2, 3):
        corpus.extend("".join(combo) for combo in itertools.product(_OUTBOUND_CHUNKS, repeat=size))
    return corpus


_OUTBOUND_CORPUS = _outbound_corpus()


def _remove(text: str, ranges: list[tuple[int, int]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for start, end in ranges:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _is_subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(ch in it for ch in needle)


def test_composed_mask_never_masks_more_than_legacy_or_scan() -> None:
    narrower = equal = 0
    for text in _OUTBOUND_CORPUS:
        composed = _mask_example_secret_fences(text, on_disagreement=lambda _d: None)
        legacy = _remove(text, _legacy_mask_ranges(text))
        scan = _remove(text, _scan_mask_ranges(text))
        assert _is_subsequence(legacy, composed), text
        assert _is_subsequence(scan, composed), text
        # Line breaks are always preserved, so line numbers stay stable.
        assert composed.count("\n") == text.count("\n"), text
        assert composed.count("\r") == text.count("\r"), text
        if composed == legacy:
            equal += 1
        else:
            narrower += 1
    # Controls: some inputs are masked identically, some strictly less.
    assert narrower and equal


def test_outbound_disagreement_event_fires_only_on_disagreement() -> None:
    fired = quiet = 0
    for text in _OUTBOUND_CORPUS:
        events: list[Disagreement] = []
        _mask_example_secret_fences(text, on_disagreement=events.append)
        if _legacy_mask_ranges(text) == _scan_mask_ranges(text):
            assert events == [], text
            quiet += 1
        else:
            assert [e.guard for e in events] == [markdown_guard.GUARD_OUTBOUND_MASK], text
            fired += 1
    assert fired and quiet


def test_outbound_agreeing_document_emits_no_event_and_masks() -> None:
    events: list[Disagreement] = []
    body = f"a\n```example-secret\n{_KEY}\n```\nb\n"
    masked = _mask_example_secret_fences(body, on_disagreement=events.append)
    assert _KEY not in masked and masked.count("\n") == body.count("\n")
    assert events == []


def test_outbound_bare_cr_document_reports_the_line_model_disagreement() -> None:
    events: list[Disagreement] = []
    body = f"x\r```example-secret\r{_KEY}\r```\r"
    assert _mask_example_secret_fences(body, on_disagreement=events.append) == body
    assert len(events) == 1 and events[0].chosen == "-"


_AWS = "AKIA" + "BC2D3E4F5G6H7JKL"


@pytest.mark.parametrize(
    "body",
    [
        f"\t```\n{_AWS}\n```\n",
        f"```x`y\n```\n```example-secret\n{_AWS}\n```\n",
        f"prose\u2028```example-secret\n{_AWS}\n```\n",
        f"~~~\n\t\t~~~example-secret\n{_AWS}\n~~~\n",
        f"x\r```example-secret\r{_AWS}\r```\r",
    ],
    ids=["tab-ordinary", "rejected-opener", "u2028", "tilde-inner", "bare-cr"],
)
def test_regression_inputs_keep_the_key_detectable(body: str) -> None:
    assert any(m.rule_id == "aws-access-token" for m in scan_outbound_text(body, part="body"))


# --- event plumbing ------------------------------------------------------------------


def test_default_emitter_writes_a_warning_event_to_the_bound_state_path(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    markdown_guard.bind_state_path(state)
    try:
        _extract_verdict_from_text(f"```json\n{_APPROVED}\n")  # legacy none / scan approved
        _extract_verdict_from_text(f"```json\n{_REQUEST}\n```\n")  # agreement -> no event
        rows = query_events(state, kind=markdown_guard.DISAGREEMENT_KIND)
    finally:
        markdown_guard.bind_state_path(None)
        close_db(state)
    assert len(rows) == 1
    assert rows[0]["level"] == "warning"
    payload = rows[0].get("payload", rows[0])
    assert payload["guard"] == markdown_guard.GUARD_VERDICT
    assert (payload["legacy"], payload["new"], payload["chosen"]) == ("none", "approved", "none")


def test_default_emitter_without_bound_state_path_only_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    markdown_guard.bind_state_path(None)
    with caplog.at_level("WARNING", logger=markdown_guard.__name__):
        assert _extract_verdict_from_text(f"```json\n{_APPROVED}\n") is None
    assert any(markdown_guard.DISAGREEMENT_KIND in r.getMessage() for r in caplog.records)


def test_a_failing_callback_never_changes_the_result() -> None:
    def boom(_d: Disagreement) -> None:
        raise RuntimeError("telemetry down")

    assert _extract_verdict_from_text(f"```json\n{_APPROVED}\n", on_disagreement=boom) is None
