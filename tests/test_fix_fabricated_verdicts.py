"""Tests for issue #597: a missing reviewer verdict must never become an approval.

Live incident (2026-07-25, Senkichi/charlie-work + Senkichi/job-cannon). The
#566 verdict-recovery fallback globbed ``*.md`` in the PR's packet directory
looking for a review summary the reviewer had written. Review sessions launch
with a hard-pinned ``--permission-mode plan``, so reviewers cannot write files
at all -- every file in that directory is authored by the orchestrator. One of
them is ``review-prompt.md``, which embedded an example verdict block reading
``"decision": "approved"``.

So a reviewer that finished without emitting a fenced JSON block had the
orchestrator parse its own instructions and record an approval on the
reviewer's behalf. The PR then took the merge label. Ten PRs across two repos
merged unreviewed, including one whose reviewer had explicitly concluded
``request_changes`` after 34 tool calls and said so in its final message.

The failure was fail-open: absence of a verdict produced the one decision that
leads directly to a merge.
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.workflow import (
    _extract_verdict_from_text,
    _parse_review_verdict_from_files,
    _validate_review_verdict,
)

STARTED_AT = "2026-07-25T17:00:00Z"

# The exact example block that was live in prompts/review.md during the
# incident. Kept verbatim: these tests assert it can no longer become a
# verdict, so it must stay byte-faithful to what actually shipped.
INCIDENT_PROMPT_EXAMPLE = """## Decision output

Write your review summary to a Markdown file, then emit your final verdict as
a fenced JSON object.

```json
{
  "decision": "approved",
  "summary": "<concise summary of the review>",
  "required_changes": []
}
```
"""


def _packet(tmp_path: Path) -> Path:
    packet_dir = tmp_path / "prs" / "pr-1408"
    packet_dir.mkdir(parents=True)
    return packet_dir


def _log(tmp_path: Path, text: str = "") -> Path:
    log_path = tmp_path / "reviewer.log"
    log_path.write_text(text, encoding="utf-8")
    return log_path


def test_packet_prompt_is_never_a_verdict_source(tmp_path: Path) -> None:
    """The exact incident: reviewer emits nothing, prompt sits in the packet."""
    packet_dir = _packet(tmp_path)
    (packet_dir / "review-prompt.md").write_text(INCIDENT_PROMPT_EXAMPLE, encoding="utf-8")
    log_path = _log(tmp_path, "I have staged the verdict rather than emitting it.\n")

    assert _parse_review_verdict_from_files(log_path, packet_dir, STARTED_AT) is None


def test_packet_path_mentioned_by_the_reviewer_is_still_excluded(tmp_path: Path) -> None:
    """Excluding the glob but honouring mentions would reopen the same hole:
    a reviewer that merely *names* the prompt would pull it back in."""
    packet_dir = _packet(tmp_path)
    prompt_path = packet_dir / "review-prompt.md"
    prompt_path.write_text(INCIDENT_PROMPT_EXAMPLE, encoding="utf-8")
    log_path = _log(tmp_path, f"My instructions are in {prompt_path}\n")

    assert _parse_review_verdict_from_files(log_path, packet_dir, STARTED_AT) is None


def test_reviewer_written_file_outside_the_packet_still_recovers(tmp_path: Path) -> None:
    """#566's real purpose must survive: a genuine review file the reviewer
    wrote and referenced is still a valid recovery source."""
    packet_dir = _packet(tmp_path)
    review_file = tmp_path / "my-review.md"
    review_file.write_text(
        'Full review below.\n\n```json\n{"decision": "request_changes", '
        '"summary": "Unbounded serial retries in the fallback path.", '
        '"required_changes": ["Bound the retry count"]}\n```\n',
        encoding="utf-8",
    )
    log_path = _log(tmp_path, f"I wrote the full review to {review_file}\n")

    result = _parse_review_verdict_from_files(log_path, packet_dir, STARTED_AT)
    assert result is not None
    verdict, source = result
    assert verdict["decision"] == "request_changes"
    assert source == str(review_file)


def test_approved_with_no_summary_is_rejected() -> None:
    """An approval with no stated reason is indistinguishable from a reviewer
    that never formed an opinion -- and approvals merge."""
    assert _validate_review_verdict({"decision": "approved", "summary": ""}) is None
    assert _validate_review_verdict({"decision": "approved", "summary": "   "}) is None


def test_unfilled_template_placeholder_summary_is_rejected() -> None:
    """The literal string that shipped on all ten fabricated approvals."""
    assert (
        _validate_review_verdict(
            {"decision": "approved", "summary": "<concise summary of the review>"}
        )
        is None
    )


def test_a_real_approval_still_validates() -> None:
    verdict = _validate_review_verdict(
        {"decision": "approved", "summary": "Clean fix, tests cover the regression."}
    )
    assert verdict is not None
    assert verdict["decision"] == "approved"
    assert verdict["required_changes"] == []


def test_incident_example_block_no_longer_parses_as_a_verdict() -> None:
    """Belt-and-braces: even if this text reaches an extractor by some other
    route, it must not yield a verdict."""
    assert _extract_verdict_from_text(INCIDENT_PROMPT_EXAMPLE) is None


def test_no_orchestrator_authored_text_contains_a_parseable_verdict() -> None:
    """Regression guard, derived rather than enumerated.

    Orchestrator-authored text can end up in a reviewer's context or on disk
    next to a review. None of it may contain a fenced JSON block that validates
    as a verdict -- that is what turned an illustrative example into ten
    unreviewed merges.

    Two corpora, because prompts are authored two ways: packaged ``.md`` files
    and prompts built in Python (``rescue.py`` constructs its own). Sweeping
    both by glob rather than naming files means a prompt added later is covered
    whichever way it is written.
    """
    package_root = Path(__file__).resolve().parents[1] / "src" / "charlie_work"
    sources = sorted(package_root.joinpath("prompts").rglob("*.md")) + sorted(
        package_root.rglob("*.py")
    )
    assert sources, f"no orchestrator-authored sources found under {package_root}"

    offenders = [
        source.relative_to(package_root).as_posix()
        for source in sources
        if _extract_verdict_from_text(source.read_text(encoding="utf-8")) is not None
    ]

    assert not offenders, (
        f"orchestrator-authored file(s) contain a parseable verdict block: {offenders}. "
        "Use a non-parseable schema form instead, e.g. "
        '\'"decision": "approved" | "request_changes" | "blocked"\'.'
    )


def test_rescue_prompt_example_is_already_non_parseable() -> None:
    """rescue.py used the safe form all along; lock it in so a future edit
    doesn't 'tidy' it into valid JSON."""
    from charlie_work import rescue

    source = Path(rescue.__file__).read_text(encoding="utf-8")
    assert '"approved" | "request_changes" | "blocked"' in source
    assert _extract_verdict_from_text(source) is None


def test_validate_rejects_non_dict_and_bad_decision() -> None:
    assert _validate_review_verdict("approved") is None
    assert _validate_review_verdict({"decision": "lgtm", "summary": "fine"}) is None
    assert _validate_review_verdict(json.loads('{"summary": "no decision"}')) is None
