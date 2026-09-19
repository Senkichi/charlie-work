"""``parse_claude_events`` stream-JSON parsing: wellformed,
truncated, and malformed-line handling.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.claude_code import (
    ClaudeProgress,
    parse_claude_events,
)

# ---------------------------------------------------------------------------
# Tests for parse_claude_events (issue #160)
# ---------------------------------------------------------------------------


def test_parse_claude_events_file_not_exists(tmp_path: Path) -> None:
    """parse_claude_events returns None when the events file doesn't exist."""
    events_path = tmp_path / "issue-1.events.jsonl"
    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is None


def test_parse_claude_events_empty_file(tmp_path: Path) -> None:
    """parse_claude_events returns None when the events file is empty."""
    events_path = tmp_path / "issue-1.events.jsonl"
    events_path.write_text("", encoding="utf-8")
    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is None


def test_parse_claude_events_wellformed(tmp_path: Path) -> None:
    """parse_claude_events correctly accumulates counts and usage from well-formed JSONL."""
    events_path = tmp_path / "issue-1.events.jsonl"
    events = [
        '{"type": "user_message"}',
        '{"type": "assistant_message"}',
        '{"type": "tool_call"}',
        '{"type": "tool_call"}',
        '{"type": "assistant_message", "tokens": 1000, "cost_usd": 0.01}',
        '{"type": "tool_call"}',
        '{"type": "assistant_message", "tokens": 1500, "cost_usd": 0.015}',
    ]
    events_path.write_text("\n".join(events), encoding="utf-8")

    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is not None
    assert result.tool_call_count == 3
    assert result.turn_count == 4  # 4 user/assistant messages total
    assert result.tokens == 1500  # Last-seen value
    assert result.cost_usd == 0.015  # Last-seen value


def test_parse_claude_events_truncated_final_line(tmp_path: Path) -> None:
    """parse_claude_events tolerates a truncated final line (live-appending file)."""
    events_path = tmp_path / "issue-1.events.jsonl"
    events = [
        '{"type": "user_message"}',
        '{"type": "tool_call"}',
        '{"type": "assistant_message", "tokens": 500',  # Truncated JSON
    ]
    events_path.write_text("\n".join(events), encoding="utf-8")

    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is not None
    assert result.tool_call_count == 1
    assert result.turn_count == 1
    assert result.tokens is None  # Truncated line is skipped, so no tokens field


def test_parse_claude_events_malformed_lines_skipped(tmp_path: Path) -> None:
    """parse_claude_events skips malformed JSON lines without raising."""
    events_path = tmp_path / "issue-1.events.jsonl"
    events = [
        '{"type": "user_message"}',
        "not valid json",
        '{"type": "tool_call"}',
        '{"type": "assistant_message"}',
        "also not json",
        '{"type": "tool_call"}',
    ]
    events_path.write_text("\n".join(events), encoding="utf-8")

    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is not None
    assert result.tool_call_count == 2
    assert result.turn_count == 2


def test_parse_claude_events_only_malformed_returns_none(tmp_path: Path) -> None:
    """parse_claude_events returns None when the file contains only malformed JSON."""
    events_path = tmp_path / "issue-1.events.jsonl"
    events_path.write_text("not json\nalso not json", encoding="utf-8")

    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is None


def test_parse_claude_events_non_dict_events_skipped(tmp_path: Path) -> None:
    """parse_claude_events skips non-dict JSON values (arrays, strings, etc)."""
    events_path = tmp_path / "issue-1.events.jsonl"
    events = [
        '{"type": "user_message"}',
        '["array", "value"]',
        '{"type": "tool_call"}',
        '"string value"',
        '{"type": "assistant_message"}',
    ]
    events_path.write_text("\n".join(events), encoding="utf-8")

    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is not None
    assert result.tool_call_count == 1
    assert result.turn_count == 2
