"""usage_from_events stream-json ingestion for the spend ledger (issue #480).

Includes the ``parse_claude_events`` regression pair kept beside the
accumulator that reuses its ``iter_claude_events`` extraction.

Split out of ``tests/test_api_budget.py`` (issue #1571, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_api_budget_unit_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.api_budget import Usage, usage_from_events
from charlie_work.claude_code import iter_claude_events, parse_claude_events

from _api_budget_unit_fixtures import _assistant_event, _result_event


def test_usage_from_events_prefers_result_cumulative() -> None:
    """The terminal result event's cumulative usage is authoritative."""
    events = [
        _assistant_event("msg_1", input_tokens=100, output_tokens=50),
        _assistant_event("msg_2", input_tokens=200, output_tokens=80, cache_read=40),
        _result_event(input_tokens=300, output_tokens=130, cache_creation=10, cache_read=60),
    ]
    usage = usage_from_events(events)
    # input folds in cache_creation: 300 + 10 = 310
    assert usage == Usage(input_tokens=310, output_tokens=130, cached_tokens=60)


def test_usage_from_events_falls_back_to_assistant_sum_when_no_result() -> None:
    """Killed session with no result event: sum per-turn assistant usage."""
    events = [
        _assistant_event("msg_1", input_tokens=100, output_tokens=50),
        _assistant_event("msg_2", input_tokens=200, output_tokens=80, cache_read=40),
    ]
    usage = usage_from_events(events)
    assert usage == Usage(input_tokens=300, output_tokens=130, cached_tokens=40)


def test_usage_from_events_dedups_multiline_assistant_by_message_id() -> None:
    """Claude Code writes one JSONL line per content block; input/cache are
    identical across lines, output grows. Counting per-line over-counts; the
    accumulator dedups by message.id keeping the max output_tokens."""
    events = [
        # One API call (msg_1) split across two content-block lines.
        _assistant_event("msg_1", input_tokens=100, output_tokens=30),
        _assistant_event("msg_1", input_tokens=100, output_tokens=50),  # final output
        # A second API call without a message.id (falls into the no-id bucket).
        {
            "type": "assistant",
            "message": {
                "usage": {
                    "input_tokens": 40,
                    "output_tokens": 10,
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 0,
                }
            },
        },
    ]
    usage = usage_from_events(events)
    # msg_1: input 100, output 50 (max). no-id: input 40+5=45, output 10.
    assert usage == Usage(input_tokens=145, output_tokens=60, cached_tokens=0)


def test_usage_from_events_empty() -> None:
    assert usage_from_events([]) == Usage()


def test_usage_from_events_skips_malformed_and_non_dict() -> None:
    events = [
        {"type": "assistant"},  # no message.usage
        {"type": "assistant", "message": "not-a-dict"},
        {"type": "assistant", "message": {"usage": "not-a-dict"}},
        {"type": "result"},  # no usage
        {"type": "result", "usage": "not-a-dict"},
        "not-a-dict",
        _result_event(input_tokens=42, output_tokens=7),
    ]
    usage = usage_from_events(events)
    assert usage == Usage(input_tokens=42, output_tokens=7, cached_tokens=0)


def test_usage_from_events_reuses_iter_claude_events_from_file(tmp_path: Path) -> None:
    """usage_from_events consumes iter_claude_events output (no re-implemented parsing)."""
    events_file = tmp_path / "issue-1.events.jsonl"
    events_file.write_text(
        json.dumps(_assistant_event("msg_1", input_tokens=100, output_tokens=50))
        + "\n"
        + json.dumps(_result_event(input_tokens=100, output_tokens=50))
        + "\n",
        encoding="utf-8",
    )
    usage = usage_from_events(iter_claude_events(events_file))
    assert usage == Usage(input_tokens=100, output_tokens=50, cached_tokens=0)


def test_parse_claude_events_still_works_after_refactor(tmp_path: Path) -> None:
    """parse_claude_events preserves behavior after extracting iter_claude_events."""
    events_file = tmp_path / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 1000, "cost_usd": 0.01}\n'
        '{"type": "user_message", "tokens": 2000, "cost_usd": 0.02}\n'
        '{"type": "assistant_message", "tokens": 3000, "cost_usd": 0.03}\n',
        encoding="utf-8",
    )
    progress = parse_claude_events(events_file)
    assert progress is not None
    assert progress.tool_call_count == 1
    assert progress.turn_count == 2
    assert progress.tokens == 3000
    assert progress.cost_usd == 0.03


def test_parse_claude_events_missing_file_returns_none(tmp_path: Path) -> None:
    assert parse_claude_events(tmp_path / "nope.events.jsonl") is None
