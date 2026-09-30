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

from collections.abc import Callable
from pathlib import Path

import markdown_guard_corpus as corpus
import pytest

from charlie_work import markdown_guard
from charlie_work.instrumentation import close_db, query_events
from charlie_work.markdown_guard import Disagreement, decision_severity
from charlie_work.outbound_body_guard import (
    SecretMatch,
    _legacy_mask_ranges,
    _legacy_masked_text,
    _mask_example_secret_fences,
    _scan_mask_ranges,
    _scan_masked,
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

_APPROVED, _REQUEST, _BLOCKED, _DRAFT = (
    corpus.APPROVED,
    corpus.REQUEST,
    corpus.BLOCKED,
    corpus.DRAFT,
)
_FINAL_SHAPES = corpus.FINAL_SHAPES
_SCAN_ONLY_AFTER_LEGACY = corpus.SCAN_ONLY_AFTER_LEGACY


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


_VERDICT_CORPUS = corpus.verdict_corpus()


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

_KEY = corpus.KEY
_OUTBOUND_CORPUS = corpus.outbound_corpus()


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


def _scan_masked_body(masked: str) -> tuple[SecretMatch, ...]:
    return _scan_masked(masked, part="body")


_SPLIT_TOKEN_HEAD = corpus.SPLIT_TOKEN_HEAD
_SPLIT_TOKEN_TAIL = corpus.SPLIT_TOKEN_TAIL


def test_outbound_refuses_a_token_main_matches_across_a_legacy_only_masked_block() -> None:
    """md-r4 review B1: masking a subset of main's characters is not enough.

    Main blanks the tab-indented example-secret block (deleting its lines joins
    the base64 halves into one ``jwt-base64`` match); the strict scan reads it
    as indented code and keeps it, which breaks that multi-line match. The
    guard must still refuse, because main does.
    """
    body = f"{_SPLIT_TOKEN_HEAD}\n\t```example-secret\n\n```\n{_SPLIT_TOKEN_TAIL}\n"
    events: list[Disagreement] = []
    composed = _mask_example_secret_fences(body, on_disagreement=events.append)
    assert _scan_masked_body(composed) == (), "premise: the composed mask alone misses it"
    assert events, "premise: the two masks disagree on this body"
    assert [m.rule_id for m in _scan_masked_body(_legacy_masked_text(body))] == ["jwt-base64"]
    assert [m.rule_id for m in scan_outbound_text(body, part="body")] == ["jwt-base64"]


def test_outbound_agreeing_split_token_is_refused_control() -> None:
    body = f"{_SPLIT_TOKEN_HEAD}\r\n```example-secret\r\nnote\r\n```\r\n{_SPLIT_TOKEN_TAIL}\r\n"
    assert {m.rule_id for m in scan_outbound_text(body, part="body")} == {"jwt-base64"}


def test_outbound_scan_reports_every_match_main_reports_over_the_corpus() -> None:
    """Compared against origin/main's LITERAL match keys, not the branch's own legacy scan.

    The expected side is the golden table (``markdown_legacy_golden.json``,
    generated by running origin/main's code over the same corpus), so this
    cannot pass by construction: it fails if the union is dropped, if a legacy
    body is loosened, or if a legacy-only duplicate is collapsed.
    """
    golden = corpus.load_golden()
    texts = corpus.secret_corpus()
    assert golden["fingerprints"]["secret"] == corpus.corpus_fingerprint(texts)
    table = golden["secret"]
    with_main_matches = 0
    for text, index in zip(texts, table["index"], strict=True):
        main_keys = table["unique"][index]["keys"]
        got = [
            [m.rule_id, m.line, m.match_sha256[:12]] for m in scan_outbound_text(text, part="body")
        ]
        remaining = list(got)
        for key in main_keys:  # multiset containment: main's count is preserved too
            assert key in remaining, (text, key)
            remaining.remove(key)
        with_main_matches += bool(main_keys)
    # Control: the corpus has plenty of documents main refuses.
    assert with_main_matches > 1000


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


_AWS = corpus.AWS


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


# --- disagreement sink: per repo, never last-writer-wins ---------------------------------

_BARE_CR_BODY = f"x\r```example-secret\r{_KEY}\r```\r"  # legacy/scan masks disagree, no secret


def _state_file(repo_root: Path) -> Path:
    return repo_root / ".var" / "charlie-work" / "state.json"


def test_default_emitter_payload_carries_the_repo(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    markdown_guard.bind_state_path(state, "repo-x")
    try:
        _extract_verdict_from_text(f"```json\n{_APPROVED}\n")
        rows = query_events(state, kind=markdown_guard.DISAGREEMENT_KIND)
    finally:
        close_db(state)
    assert len(rows) == 1
    assert rows[0]["payload"]["repo"] == "repo-x"
    assert rows[0]["repo"] == "repo-x"


def test_outbound_disagreement_goes_to_the_writing_clients_repo_not_the_ambient_one(
    tmp_path: Path,
) -> None:
    from charlie_work.outbound_body_guard import check_outbound_write

    other, mine = tmp_path / "other-repo", tmp_path / "my-repo"
    ambient = _state_file(other)
    markdown_guard.bind_state_path(ambient, other.name)  # the last-constructed app
    try:
        assert (
            check_outbound_write(
                surface="pr_comment", parts=(("body", _BARE_CR_BODY),), repo_root=mine
            )
            == ()
        )
        mine_rows = query_events(_state_file(mine), kind=markdown_guard.DISAGREEMENT_KIND)
        other_rows = query_events(ambient, kind=markdown_guard.DISAGREEMENT_KIND)
    finally:
        close_db(_state_file(mine))
        close_db(ambient)
    assert [r["payload"]["repo"] for r in mine_rows] == ["my-repo"]
    assert mine_rows[0]["payload"]["guard"] == markdown_guard.GUARD_OUTBOUND_MASK
    assert other_rows == []


def test_ambient_sinks_are_isolated_per_thread(tmp_path: Path) -> None:
    """Two fleet lanes on two pool threads each write to their own repo."""
    import threading

    states = {name: tmp_path / name / "state.json" for name in ("a", "b")}
    barrier = threading.Barrier(2)

    def lane(name: str) -> None:
        token = markdown_guard.bind_sink(states[name], name)
        try:
            barrier.wait(timeout=10)  # both bound before either emits
            _extract_verdict_from_text(f"```json\n{_APPROVED}\n")
        finally:
            markdown_guard.unbind_sink(token)

    threads = [threading.Thread(target=lane, args=(name,)) for name in states]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    try:
        for name, state in states.items():
            rows = query_events(state, kind=markdown_guard.DISAGREEMENT_KIND)
            assert [r["payload"]["repo"] for r in rows] == [name]
    finally:
        for state in states.values():
            close_db(state)


def test_the_autouse_fixture_leaves_no_ambient_sink_bound() -> None:
    """tests/conftest.py resets the binding around every test (no cross-test leakage)."""
    assert markdown_guard._sink.get() is None
    markdown_guard.bind_state_path(Path("leak") / "state.json", "leak")  # reset after this test


# --- N4: legacy-only duplicates keep main's count ----------------------------------------


def test_legacy_only_duplicate_matches_keep_mains_count(monkeypatch: pytest.MonkeyPatch) -> None:
    dup = SecretMatch(rule_id="r", part="body", line=2, match_sha256="h")
    other = SecretMatch(rule_id="r", part="body", line=3, match_sha256="h")
    from charlie_work import outbound_body_guard as guard

    by_text = {"composed": (other,), "legacy": (dup, dup, other)}
    monkeypatch.setattr(guard, "_mask_example_secret_fences", lambda _t, **_k: "composed")
    monkeypatch.setattr(guard, "_legacy_masked_text", lambda _t: "legacy")
    monkeypatch.setattr(guard, "_scan_masked", lambda masked, *, part: by_text[masked])
    result = scan_outbound_text("anything", part="body")
    # main reported 3 matches; the composed side already covers `other`, so the two
    # legacy-only `dup` matches must both survive (not be collapsed to one).
    assert result == (other, dup, dup)
