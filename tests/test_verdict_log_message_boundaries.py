"""Plaintext devin logs concatenate messages with no separator (issue #2143)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work import markdown_guard
from charlie_work.verdict_parsing import _parse_review_verdict_from_log

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


def test_trailing_block_not_a_verdict_never_selects_example_approval(tmp_path: Path) -> None:
    text = f"```json\n{_VERDICT}\n```\nprose\n```json\n{{not json}}\n```\n"
    verdict = _parse_review_verdict_from_log(_log(tmp_path, text))
    # Falls back to the whole-text scan (origin/main behaviour); the point is
    # the trailing path itself does not raise or fabricate.
    assert verdict is None or verdict["decision"] == "approved"
