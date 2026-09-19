"""``parse_cumulative_usage`` tests for ``charlie_work.worker``.

Split out of ``tests/test_worker_health.py`` (issue #1568, Track-1):
events.jsonl cumulative-usage parsing -- missing/empty/malformed
files, truncated trailing lines, and absent usage fields (issue #163).
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.worker import parse_cumulative_usage


# Tests for parse_cumulative_usage (issue #163)


def test_parse_cumulative_usage_missing_file(tmp_path: Path) -> None:
    """parse_cumulative_usage returns None when the events file doesn't exist."""
    events_file = tmp_path / "issue-1.events.jsonl"
    usage = parse_cumulative_usage(events_file)
    assert usage is None


def test_parse_cumulative_usage_wellformed_jsonl(tmp_path: Path) -> None:
    """parse_cumulative_usage returns the latest cumulative values from well-formed JSONL."""
    events_file = tmp_path / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 1000, "cost_usd": 0.01}\n'
        '{"type": "user_message", "tokens": 2000, "cost_usd": 0.02}\n'
        '{"type": "assistant_message", "tokens": 3000, "cost_usd": 0.03}\n',
        encoding="utf-8",
    )

    usage = parse_cumulative_usage(events_file)
    assert usage is not None
    assert usage.tokens == 3000
    assert usage.cost_usd == 0.03


def test_parse_cumulative_usage_truncated_trailing_line(tmp_path: Path) -> None:
    """parse_cumulative_usage ignores a truncated trailing line and returns prior valid values."""
    events_file = tmp_path / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 1000, "cost_usd": 0.01}\n'
        '{"type": "user_message", "tokens": 2000, "cost_usd": 0.02}\n'
        '{"type": "assistant_message", "tokens": 3000, "cost_usd": 0.03',  # Truncated JSON
        encoding="utf-8",
    )

    usage = parse_cumulative_usage(events_file)
    assert usage is not None
    assert usage.tokens == 2000
    assert usage.cost_usd == 0.02


def test_parse_cumulative_usage_empty_file(tmp_path: Path) -> None:
    """parse_cumulative_usage returns None for an empty file."""
    events_file = tmp_path / "issue-1.events.jsonl"
    events_file.write_text("", encoding="utf-8")

    usage = parse_cumulative_usage(events_file)
    assert usage is None


def test_parse_cumulative_usage_no_usage_fields(tmp_path: Path) -> None:
    """parse_cumulative_usage returns None when events have no tokens/cost_usd fields."""
    events_file = tmp_path / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call"}\n{"type": "user_message"}',
        encoding="utf-8",
    )

    usage = parse_cumulative_usage(events_file)
    assert usage is None
