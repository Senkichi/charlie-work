"""Regression tests for review-verdict extraction from stream-json logs.

Issue #566: since ``tee_stream_json`` was force-enabled for reviewers, the
sidecar log is JSONL where every fenced verdict block lives *inside* a JSON
string (``\\n`` as two-character escape sequences). The old parser regex
required literal newlines around the fence, so it could never match — and the
events-file fallback scanned for a fictional ``assistant_message`` event type
the real stream never emits (real types: ``assistant``/``result`` with a
nested ``message.content``). Every completed review therefore counted as a
failed attempt, and the 3-attempt cap escalated PRs whose reviewers had done
their job perfectly (observed live on PRs #540/#503, 2026-07-24).

Fixes under test:

1. ``_parse_review_verdict_from_log`` decodes stream-json events and scans
   the decoded texts (result event first) alongside the plain-text path.
2. ``_parse_review_verdict_from_events`` understands the real event schema.
3. A last-resort file fallback recovers verdicts that reviewers wrote to a
   referenced Markdown file, mtime-gated to the session's ``started_at`` so a
   stale file from a prior round can never resurrect an old verdict.
4. ``parse_claude_events`` extracts progress metrics from the real schema.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.claude_code import parse_claude_events
from charlie_work.state import load_state
from charlie_work.workflow import (
    _parse_review_verdict_from_events,
    _parse_review_verdict_from_files,
    _parse_review_verdict_from_log,
)
from test_charlie_work import (
    _dispatch_reviews_app,
    _make_dead_review_sidecar,
    _set_review_dispatched_state,
    _write_review_packet,
)

VERDICT_TEXT = (
    "Review complete. Final verdict:\n\n```json\n"
    '{\n  "decision": "approved",\n  "summary": "lgtm",\n  "required_changes": []\n}\n```\n'
)


def _assistant_event(*blocks: dict) -> str:
    return json.dumps(
        {"type": "assistant", "message": {"role": "assistant", "content": list(blocks)}}
    )


def _result_event(result_text: str) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "result": result_text,
            "num_turns": 20,
            "total_cost_usd": 1.51,
            "usage": {"input_tokens": 1200, "output_tokens": 900},
        }
    )


def _stream_json_log(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def test_log_parser_finds_verdict_in_stream_json_result_event(tmp_path: Path) -> None:
    """The canonical live failure: a compliant reviewer's verdict sits inside
    the result event's JSON-escaped string (PR #503, 2026-07-24)."""
    log = tmp_path / "review.claude.log"
    log.write_text(
        _stream_json_log(
            json.dumps({"type": "system", "subtype": "init"}),
            _assistant_event({"type": "text", "text": "Analyzing the diff..."}),
            _result_event(VERDICT_TEXT),
        ),
        encoding="utf-8",
    )

    verdict = _parse_review_verdict_from_log(log)

    assert verdict is not None
    assert verdict["decision"] == "approved"
    assert verdict["summary"] == "lgtm"
    assert verdict["required_changes"] == []


def test_log_parser_finds_verdict_in_stream_json_assistant_text(tmp_path: Path) -> None:
    """A reviewer killed before the result event still has its verdict in the
    last assistant text block."""
    log = tmp_path / "review.claude.log"
    log.write_text(
        _stream_json_log(
            _assistant_event({"type": "text", "text": "Looking at tests..."}),
            _assistant_event({"type": "text", "text": VERDICT_TEXT}),
        ),
        encoding="utf-8",
    )

    verdict = _parse_review_verdict_from_log(log)

    assert verdict is not None
    assert verdict["decision"] == "approved"


def test_log_parser_prefers_final_output_over_earlier_draft(tmp_path: Path) -> None:
    """The result event (last line) wins over an earlier assistant draft."""
    draft = VERDICT_TEXT.replace('"approved"', '"blocked"').replace(
        '"lgtm"', '"draft — do not use"'
    )
    log = tmp_path / "review.claude.log"
    log.write_text(
        _stream_json_log(
            _assistant_event({"type": "text", "text": draft}),
            _result_event(VERDICT_TEXT),
        ),
        encoding="utf-8",
    )

    verdict = _parse_review_verdict_from_log(log)

    assert verdict is not None
    assert verdict["decision"] == "approved"


def test_log_parser_ignores_verdict_in_thinking_blocks(tmp_path: Path) -> None:
    """Draft verdicts inside thinking blocks must not be treated as output."""
    log = tmp_path / "review.claude.log"
    log.write_text(
        _stream_json_log(
            _assistant_event({"type": "thinking", "thinking": VERDICT_TEXT, "signature": "x"}),
        ),
        encoding="utf-8",
    )

    assert _parse_review_verdict_from_log(log) is None


def test_log_parser_plain_text_log_unchanged(tmp_path: Path) -> None:
    """Plain-text logs (no stream-json tee) keep working exactly as before."""
    log = tmp_path / "review.claude.log"
    log.write_text(VERDICT_TEXT, encoding="utf-8")

    verdict = _parse_review_verdict_from_log(log)

    assert verdict is not None
    assert verdict["decision"] == "approved"


def test_events_parser_handles_real_stream_json_schema(tmp_path: Path) -> None:
    events = tmp_path / "review.events.jsonl"
    events.write_text(
        _stream_json_log(
            json.dumps({"type": "system", "subtype": "init"}),
            _result_event(VERDICT_TEXT),
        ),
        encoding="utf-8",
    )

    verdict = _parse_review_verdict_from_events(events)

    assert verdict is not None
    assert verdict["decision"] == "approved"


def test_events_parser_still_handles_legacy_assistant_message(tmp_path: Path) -> None:
    events = tmp_path / "review.events.jsonl"
    events.write_text(
        json.dumps({"type": "assistant_message", "content": VERDICT_TEXT}) + "\n",
        encoding="utf-8",
    )

    verdict = _parse_review_verdict_from_events(events)

    assert verdict is not None
    assert verdict["decision"] == "approved"


# --- File fallback (issue #566) ---------------------------------------------


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def test_file_fallback_recovers_verdict_from_referenced_md(tmp_path: Path) -> None:
    """A verdict written to a Markdown file referenced in final output is
    recovered when the file is fresher than the session start."""
    md = tmp_path / "adversarial-review-packet.md"
    md.write_text("# Review\n\nDetails...\n\n" + VERDICT_TEXT, encoding="utf-8")

    log = tmp_path / "review.claude.log"
    log.write_text(
        _stream_json_log(_result_event(f"Full review written to `{md}` with the verdict.")),
        encoding="utf-8",
    )

    started_at = _utc_iso(datetime.now(UTC) - timedelta(minutes=10))
    hit = _parse_review_verdict_from_files(log, tmp_path / "no-packet-dir", started_at)

    assert hit is not None
    verdict, source = hit
    assert verdict["decision"] == "approved"
    assert source == str(md)


def test_file_fallback_ignores_stale_md(tmp_path: Path) -> None:
    """A file older than the session start (previous review round) must never
    resurrect an old verdict for a new head."""
    md = tmp_path / "adversarial-review-packet.md"
    md.write_text(VERDICT_TEXT, encoding="utf-8")
    stale = (datetime.now(UTC) - timedelta(hours=6)).timestamp()
    os.utime(md, (stale, stale))

    log = tmp_path / "review.claude.log"
    log.write_text(
        _stream_json_log(_result_event(f"Full review written to `{md}`.")),
        encoding="utf-8",
    )

    started_at = _utc_iso(datetime.now(UTC) - timedelta(minutes=10))
    assert _parse_review_verdict_from_files(log, tmp_path / "no-packet-dir", started_at) is None


def test_file_fallback_scans_packet_dir(tmp_path: Path) -> None:
    """A review.md dropped in the PR packet dir is found even when the log
    never mentions a path."""
    packet_dir = tmp_path / "pr-100"
    packet_dir.mkdir()
    (packet_dir / "review.md").write_text(VERDICT_TEXT, encoding="utf-8")

    log = tmp_path / "review.claude.log"
    log.write_text(
        _stream_json_log(_result_event("Review done, see the packet directory.")),
        encoding="utf-8",
    )

    started_at = _utc_iso(datetime.now(UTC) - timedelta(minutes=10))
    hit = _parse_review_verdict_from_files(log, packet_dir, started_at)

    assert hit is not None
    verdict, source = hit
    assert verdict["decision"] == "approved"
    assert source == str(packet_dir / "review.md")


def test_file_fallback_requires_started_at(tmp_path: Path) -> None:
    """Without a parseable started_at there is no safe mtime gate — no fallback."""
    md = tmp_path / "review.md"
    md.write_text(VERDICT_TEXT, encoding="utf-8")
    log = tmp_path / "review.claude.log"
    log.write_text(
        _stream_json_log(_result_event(f"see `{md}`")),
        encoding="utf-8",
    )

    assert _parse_review_verdict_from_files(log, tmp_path, None) is None
    assert _parse_review_verdict_from_files(log, tmp_path, "not-a-timestamp") is None


# --- End-to-end through the reaper ------------------------------------------


def test_reap_records_verdict_from_stream_json_log(monkeypatch, tmp_path: Path) -> None:
    """A dead reviewer whose stream-json log carries the verdict in its result
    event has the verdict recorded (previously: counted as a failed attempt)."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    reviews_dir = app._resolve(app.config.review_dispatch.reviews_dir)

    log_text = _stream_json_log(
        json.dumps({"type": "system", "subtype": "init"}),
        _assistant_event({"type": "text", "text": "Reviewing..."}),
        _result_event(VERDICT_TEXT),
    )
    _make_dead_review_sidecar(reviews_dir, 100, log_text)
    _set_review_dispatched_state(app, 100, 10, "2026-07-06T12:00:00Z")
    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    assert result["recorded"] == [
        {"pr": 100, "issue": 10, "decision": "approved", "verdict_source": "log"}
    ]
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_completed"


def test_reap_records_verdict_via_file_fallback(monkeypatch, tmp_path: Path) -> None:
    """A dead reviewer that only *referenced* its verdict file (the PR #540
    live failure) has the verdict recovered through the file fallback."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    reviews_dir = app._resolve(app.config.review_dispatch.reviews_dir)

    md = tmp_path / "plans" / "adversarial-review-notes.md"
    md.parent.mkdir(parents=True)
    md.write_text("# Full review\n\n...analysis...\n\n" + VERDICT_TEXT, encoding="utf-8")

    log_text = _stream_json_log(
        _result_event(f"Full review written to `{md}` with the verdict JSON block at the end.")
    )
    started_at = _utc_iso(datetime.now(UTC) - timedelta(minutes=5))
    _make_dead_review_sidecar(reviews_dir, 100, log_text, started_at=started_at)
    _set_review_dispatched_state(app, 100, 10, started_at)
    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    assert result["recorded"] == [
        {"pr": 100, "issue": 10, "decision": "approved", "verdict_source": f"file:{md}"}
    ]
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_completed"


# --- parse_claude_events on the real schema ---------------------------------


def test_parse_claude_events_real_stream_json_metrics(tmp_path: Path) -> None:
    events = tmp_path / "review.events.jsonl"
    events.write_text(
        _stream_json_log(
            json.dumps({"type": "system", "subtype": "init"}),
            _assistant_event(
                {"type": "text", "text": "Working..."},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "Grep", "input": {}},
            ),
            _assistant_event({"type": "text", "text": "Done."}),
            _result_event("final"),
        ),
        encoding="utf-8",
    )

    progress = parse_claude_events(events)

    assert progress is not None
    assert progress.tool_call_count == 2
    # result.num_turns (20) is authoritative over the 2 assistant events seen
    assert progress.turn_count == 20
    assert progress.tokens == 2100
    assert progress.cost_usd == 1.51
