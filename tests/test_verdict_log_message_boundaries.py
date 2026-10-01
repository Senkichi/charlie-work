"""Plaintext devin logs concatenate messages with no separator (issue #2143)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work import markdown_guard
from charlie_work.verdict_parsing import (
    _extract_verdict_from_text,
    _parse_review_verdict_from_log,
)

_VERDICT = json.dumps({"decision": "approved", "summary": "Looks good.", "required_changes": []})


@pytest.fixture
def disagreements(monkeypatch: pytest.MonkeyPatch) -> list[markdown_guard.Disagreement]:
    seen: list[markdown_guard.Disagreement] = []
    monkeypatch.setattr(markdown_guard, "emit_disagreement", seen.append)
    return seen


def _log(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "issue-41.log"
    path.write_text(text, encoding="utf-8")
    return path


def test_fence_glued_to_prior_message_is_recovered_without_disagreement(
    tmp_path: Path, disagreements: list[markdown_guard.Disagreement]
) -> None:
    text = (
        "Let me look at the `run()` method.\n"
        "every behavior has a genuine regression test."
        f"```json\n{_VERDICT}\n```\n"
    )
    verdict = _parse_review_verdict_from_log(_log(tmp_path, text))
    assert verdict is not None and verdict["decision"] == "approved"
    assert disagreements == []


def test_line_start_fence_behaves_as_before(
    tmp_path: Path, disagreements: list[markdown_guard.Disagreement]
) -> None:
    verdict = _parse_review_verdict_from_log(_log(tmp_path, f"prose\n```json\n{_VERDICT}\n```\n"))
    assert verdict is not None and verdict["decision"] == "approved"
    assert disagreements == []


@pytest.mark.parametrize(
    "text",
    [
        "Use `code` and ```json\nnot a verdict\n``` in prose.\n",
        "inline ```js\nconsole.log(1)\n``` fence-looking prose\n",
        "no fence at all, just prose.Let me continue.\n",
    ],
)
def test_normalization_does_not_invent_a_verdict(
    tmp_path: Path, text: str, disagreements: list[markdown_guard.Disagreement]
) -> None:
    assert _parse_review_verdict_from_log(_log(tmp_path, text)) is None
    assert disagreements == []


_REJECT = json.dumps(
    {"decision": "request_changes", "summary": "Needs work.", "required_changes": ["fix it"]}
)


def test_glued_unclosed_fence_earlier_does_not_hide_real_verdict(
    tmp_path: Path, disagreements: list[markdown_guard.Disagreement]
) -> None:
    text = (
        f"Example: here is the format.```json\n{_VERDICT}\n"
        "more prose that never closes the example above\n"
        f"\n```json\n{_REJECT}\n```\n"
    )
    verdict = _parse_review_verdict_from_log(_log(tmp_path, text))
    assert verdict is not None and verdict["decision"] == "request_changes"
    assert disagreements == []


def test_final_block_wins_over_example_with_nested_approved_and_unbalanced_fences(
    tmp_path: Path,
) -> None:
    text = (
        "Reviewing the diff. The format is:\n"
        "```\n"
        "```json\n"
        f"{_VERDICT}\n"
        "```\n"
        "```\n"
        "```text\n"
        "some stray tagged fence\n"
        "\n"
        "Now my actual verdict.\n"
        f"```json\n{_REJECT}\n```\n"
    )
    verdict = _parse_review_verdict_from_log(_log(tmp_path, text))
    assert verdict is not None
    assert verdict["decision"] == "request_changes"


def test_invalid_trailing_block_falls_back_to_whole_text_scan(tmp_path: Path) -> None:
    """A malformed trailing block yields no trailing verdict; the whole-text scan decides.

    The trailing path returns ``None`` (its body is not JSON), so extraction
    falls through to the origin/main whole-text scan, which skips the
    undecodable block and selects the only valid verdict present -- the earlier
    ``approved`` one. Pinned so a change to that fallback is a visible decision.
    """
    text = f"```json\n{_VERDICT}\n```\nprose\n```json\n{{not json}}\n```\n"
    verdict = _parse_review_verdict_from_log(_log(tmp_path, text))
    assert verdict is not None
    assert verdict["decision"] == "approved"


def test_confirmed_glued_request_changes_suppresses_raw_disagreement(
    tmp_path: Path, disagreements: list[markdown_guard.Disagreement]
) -> None:
    """Raw legacy finds a glued verdict that scan refuses; the normalized text confirms it.

    The raw pass records a legacy/scan disagreement (legacy=request_changes,
    scan=none). Because the boundary-restored text yields the same decision with
    no disagreement, the disagreement was an artifact of the lost boundary and
    is suppressed -- the verdict is still returned.
    """
    text = f"Done reviewing.```json\n{_REJECT}\n```\n"
    raw_events: list[markdown_guard.Disagreement] = []
    assert _extract_verdict_from_text(text, on_disagreement=raw_events.append) is not None
    assert len(raw_events) == 1, "premise: the raw pass must record a disagreement"

    verdict = _parse_review_verdict_from_log(_log(tmp_path, text))
    assert verdict is not None and verdict["decision"] == "request_changes"
    assert disagreements == []


@pytest.mark.parametrize(
    "text",
    [
        # Normalizing makes a glued approved block visible, so the decision differs.
        f"```json\n{_REJECT}\n```\np.```json\n{_VERDICT}\n```\n",
        # Normalized text still has a legacy/scan disagreement of its own.
        f"```text\nstray\n```text\nstray\np.```json\n{_REJECT}\n```\n",
    ],
    ids=["normalized-decision-differs", "normalized-still-disagrees"],
)
def test_unconfirmed_raw_disagreement_is_still_emitted(
    tmp_path: Path, disagreements: list[markdown_guard.Disagreement], text: str
) -> None:
    """When the normalized pass does not cleanly confirm, the raw disagreement is emitted."""
    raw_events: list[markdown_guard.Disagreement] = []
    assert _extract_verdict_from_text(text, on_disagreement=raw_events.append) is not None
    assert raw_events, "premise: the raw pass must record a disagreement"

    verdict = _parse_review_verdict_from_log(_log(tmp_path, text))
    assert verdict is not None and verdict["decision"] == "request_changes"
    assert disagreements == raw_events
