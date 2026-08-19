"""Regression tests for issue #1354: reviewer sessions that die mid-session
at 14-28 turns must capture the terminating cause in ``review_verdict_missed``
payloads, and dead claim directories must be released within one loop pass.

Three acceptance criteria under test:

1. **Cause capture**: ``_extract_terminating_cause`` extracts the
   stream-json ``result`` event's terminal fields (``is_error``,
   ``api_error_status``, ``terminal_reason``, ``stop_reason``, ``subtype``)
   when present, falls back to ``exit_code`` from the durable terminal-status
   record, and returns ``{"cause": "unknown"}`` when neither source yields
   any signal. The cause propagates through ``ReviewSessionOutcome`` into the
   ``review_verdict_missed`` event payload and the fleet attention event.

2. **Load-correlation verdict**: documented in the PR body with data from
   ``events.db`` (not testable here -- the verdict is a written conclusion,
   not code).

3. **Same-pass cleanup**: ``_reap_review_verdicts`` releases the dead
   reviewer's isolated review checkout in the same pass that detects the
   death, rather than waiting for ``_detect_and_handle_stalled_reviews``'s
   5-minute stale-claim timeout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from charlie_work.state import load_state, state_lock
from charlie_work.verdict_parsing import (
    CAUSE_UNKNOWN,
    _extract_terminating_cause,
    _extract_review_session_summary,
)

from _review_fixtures import (
    _dispatch_reviews_app,
    _make_dead_review_sidecar,
    _set_review_dispatched_state,
    _write_review_events,
    _write_review_packet,
)

_PR = {
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


# --- AC1: _extract_terminating_cause ---


def test_extract_terminating_cause_from_result_event(tmp_path: Path) -> None:
    """When the stream-json ``result`` event is present, its terminal fields
    are the authoritative cause."""
    result_event = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "api_error_status": None,
        "terminal_reason": "completed",
        "stop_reason": "end_turn",
        "num_turns": 14,
        "total_cost_usd": 0.96,
        "result": "## Decision: Approved",
    }
    events_path = tmp_path / "issue-100-review.events.jsonl"
    events_path.write_text(json.dumps(result_event) + "\n", encoding="utf-8")

    cause = _extract_terminating_cause(events_path, events_path)

    assert cause["result_event"] == "present"
    assert cause["is_error"] is False
    assert cause["subtype"] == "success"
    assert cause["terminal_reason"] == "completed"
    assert cause["stop_reason"] == "end_turn"
    assert cause["api_error_status"] is None
    assert cause["cause"] == "result_ok:completed"


def test_extract_terminating_cause_from_error_result_event(tmp_path: Path) -> None:
    """A result event with ``is_error=True`` is classified as a result error."""
    result_event = {
        "type": "result",
        "subtype": "error",
        "is_error": True,
        "api_error_status": 529,
        "terminal_reason": "api_error",
        "stop_reason": "api_error",
        "result": "API overloaded",
    }
    events_path = tmp_path / "issue-100-review.events.jsonl"
    events_path.write_text(json.dumps(result_event) + "\n", encoding="utf-8")

    cause = _extract_terminating_cause(events_path, events_path)

    assert cause["result_event"] == "present"
    assert cause["is_error"] is True
    assert cause["api_error_status"] == 529
    assert cause["cause"] == "result_error:error"


def test_extract_terminating_cause_absent_result_event_stream_cut(
    tmp_path: Path,
) -> None:
    """When the stream was cut before the ``result`` event (the observed death
    mode for ``died_mid_session``), ``result_event`` is ``"absent"`` and the
    last event type is recorded."""
    # A session that died mid-turn: last event is a system event, no result.
    lines = [
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Analyzing..."}]},
            }
        ),
        json.dumps({"type": "system", "subtype": "thinking_tokens"}),
    ]
    events_path = tmp_path / "issue-100-review.events.jsonl"
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cause = _extract_terminating_cause(events_path, events_path)

    assert cause["result_event"] == "absent"
    assert cause["last_event_type"] == "system"
    assert cause["cause"] == "stream_cut_no_result_event"


def test_extract_terminating_cause_with_exit_code_only(tmp_path: Path) -> None:
    """When no events file exists but an exit code is available, the cause is
    the exit code."""
    events_path = tmp_path / "nonexistent.events.jsonl"
    log_path = tmp_path / "nonexistent.claude.log"

    cause = _extract_terminating_cause(events_path, log_path, exit_code=137)

    assert cause["exit_code"] == 137
    assert cause["cause"] == "exit_code:137"


def test_extract_terminating_cause_unknown_when_nothing_available(
    tmp_path: Path,
) -> None:
    """When neither events nor exit code yield any signal, return the explicit
    ``{"cause": "unknown"}`` sentinel."""
    events_path = tmp_path / "nonexistent.events.jsonl"
    log_path = tmp_path / "nonexistent.claude.log"

    cause = _extract_terminating_cause(events_path, log_path)

    assert cause == CAUSE_UNKNOWN
    assert cause["cause"] == "unknown"


def test_extract_terminating_cause_combines_result_event_and_exit_code(
    tmp_path: Path,
) -> None:
    """When both a result event and an exit code are available, both are
    recorded and the cause summary reflects the result event."""
    result_event = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "terminal_reason": "completed",
        "stop_reason": "end_turn",
    }
    events_path = tmp_path / "issue-100-review.events.jsonl"
    events_path.write_text(json.dumps(result_event) + "\n", encoding="utf-8")

    cause = _extract_terminating_cause(events_path, events_path, exit_code=0)

    assert cause["result_event"] == "present"
    assert cause["exit_code"] == 0
    assert cause["cause"] == "result_ok:completed"


# --- AC1: ReviewSessionOutcome.terminating_cause ---


def test_review_session_summary_carries_terminating_cause(
    tmp_path: Path,
) -> None:
    """``_extract_review_session_summary`` populates ``terminating_cause``
    from the events file."""
    # A session that did substantial work (3 turns, 2 tool calls) then died
    # without a result event.
    lines: list[str] = []
    for index in range(3):
        content: list[dict[str, Any]] = [{"type": "text", "text": f"Analysis step {index + 1}."}]
        if index < 2:
            content.append({"type": "tool_use", "id": f"t{index}", "name": "Read", "input": {}})
        lines.append(json.dumps({"type": "assistant", "message": {"content": content}}))
    jsonl = "\n".join(lines) + "\n"
    events_path = tmp_path / "issue-100-review.events.jsonl"
    log_path = tmp_path / "issue-100-review.claude.log"
    events_path.write_text(jsonl, encoding="utf-8")
    log_path.write_text(jsonl, encoding="utf-8")

    outcome = _extract_review_session_summary(events_path, log_path, max_turns=20)

    assert outcome is not None
    assert outcome.reason == "died_mid_session"
    assert outcome.terminating_cause["result_event"] == "absent"
    assert outcome.terminating_cause["last_event_type"] == "assistant"
    assert outcome.terminating_cause["cause"] == "stream_cut_no_result_event"


def test_review_session_summary_with_exit_code(tmp_path: Path) -> None:
    """``_extract_review_session_summary`` folds the exit code into the
    terminating cause."""
    lines = [
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Analyzing..."}]},
            }
        ),
    ]
    jsonl = "\n".join(lines) + "\n"
    events_path = tmp_path / "issue-100-review.events.jsonl"
    log_path = tmp_path / "issue-100-review.claude.log"
    events_path.write_text(jsonl, encoding="utf-8")
    log_path.write_text(jsonl, encoding="utf-8")

    outcome = _extract_review_session_summary(events_path, log_path, max_turns=20, exit_code=1)

    assert outcome is not None
    assert outcome.terminating_cause["exit_code"] == 1
    assert outcome.terminating_cause["result_event"] == "absent"


# --- AC1: review_verdict_missed event payload carries cause ---


def _events(state: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


def test_review_verdict_missed_payload_carries_cause(monkeypatch, tmp_path: Path) -> None:
    """``_reap_review_verdicts`` emits ``review_verdict_missed`` with a
    ``cause`` field containing the terminating-cause dict."""
    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    _set_review_dispatched_state(app, 100, 10, "2026-07-06T12:00:00Z")

    reviews_dir = app._layout.reviews_dir
    # Write a dead reviewer sidecar with events that show substantial work
    # but no result event (stream cut = died_mid_session).
    _write_review_events(reviews_dir, 100, turns=3, tool_calls=2)
    _make_dead_review_sidecar(reviews_dir, 100, "log text")

    # Prevent actual PR comments from being posted.
    monkeypatch.setattr(app, "_comment_pr", lambda *a, **kw: None)

    result = app._reap_review_verdicts(reviews_dir)

    missed = result.get("missed", [])
    assert len(missed) == 1
    assert missed[0]["reason"] == "died_mid_session"
    assert "cause" in missed[0]
    assert missed[0]["cause"]["result_event"] == "absent"
    assert missed[0]["cause"]["cause"] == "stream_cut_no_result_event"

    # Verify the event was emitted with the cause field.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    events = _events(state, "review_verdict_missed")
    assert len(events) == 1
    payload = events[0].get("payload", {})
    assert payload["reason"] == "died_mid_session"
    assert "cause" in payload
    assert payload["cause"]["result_event"] == "absent"


# --- AC1: cause survives fleet attention-event conversion ---


def test_fleet_attention_event_preserves_cause(tmp_path: Path) -> None:
    """``_add_review_verdict_events`` preserves the ``cause`` field from
    missed verdicts in fleet-level attention events."""
    from charlie_work.fleet_dispatch import _add_review_verdict_events

    data = {
        "missed_verdicts": [
            {
                "pr": 100,
                "issue": 10,
                "reason": "died_mid_session",
                "cause": {
                    "cause": "stream_cut_no_result_event",
                    "result_event": "absent",
                    "last_event_type": "system",
                },
            }
        ]
    }
    events: list[dict[str, Any]] = []
    _add_review_verdict_events("repo-key", data, events)

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "review_verdict_missed"
    assert event["reason"] == "died_mid_session"
    assert "cause" in event
    assert event["cause"]["cause"] == "stream_cut_no_result_event"
    assert event["cause"]["result_event"] == "absent"


def test_fleet_attention_event_without_cause_still_works(tmp_path: Path) -> None:
    """Missed verdicts without a ``cause`` field (e.g. from the
    record_review-failure path or older records) still produce attention
    events -- the ``cause`` field is simply absent, not ``unknown``."""
    from charlie_work.fleet_dispatch import _add_review_verdict_events

    data = {
        "missed_verdicts": [
            {
                "pr": 100,
                "issue": 10,
                "reason": "record_review failed: escalated",
            }
        ]
    }
    events: list[dict[str, Any]] = []
    _add_review_verdict_events("repo-key", data, events)

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "review_verdict_missed"
    assert event["reason"] == "record_review failed: escalated"
    assert "cause" not in event


# --- AC3: same-pass dead-claim cleanup ---


def test_reap_review_verdicts_releases_checkout_same_pass(monkeypatch, tmp_path: Path) -> None:
    """``_reap_review_verdicts`` releases the dead reviewer's isolated review
    checkout in the same pass that detects the death (issue #1354 AC3).

    Before the fix, the checkout lingered until
    ``_detect_and_handle_stalled_reviews``'s 5-minute stale-claim timeout
    elapsed, which could be the next loop pass. Now the checkout is removed
    immediately after the ``review_verdict_missed`` event is emitted.
    """
    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    _set_review_dispatched_state(app, 100, 10, "2026-07-06T12:00:00Z")

    reviews_dir = app._layout.reviews_dir
    _write_review_events(reviews_dir, 100, turns=3, tool_calls=2)
    _make_dead_review_sidecar(reviews_dir, 100, "log text")

    # Create a fake review checkout directory (the git worktree that
    # remove_review_checkout would remove). In production this is a real
    # git worktree at reviews_dir/pr-100.
    checkout_path = reviews_dir / "pr-100"
    checkout_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(app, "_comment_pr", lambda *a, **kw: None)

    # Track whether remove_review_checkout was called for this PR.
    removed_prs: list[int] = []
    original_remove = None
    from charlie_work import workflow as wf_module

    original_remove = wf_module.remove_review_checkout

    def tracking_remove(repo_root, pr_number, *, reviews_dir):
        removed_prs.append(pr_number)
        return original_remove(repo_root, pr_number, reviews_dir=reviews_dir)

    monkeypatch.setattr(wf_module, "remove_review_checkout", tracking_remove)

    app._reap_review_verdicts(reviews_dir)

    # The checkout should have been removed in the same pass.
    assert 100 in removed_prs


def test_reap_review_verdicts_does_not_reap_sidecar_same_pass(monkeypatch, tmp_path: Path) -> None:
    """``_reap_review_verdicts`` does NOT reap the sidecar in the same pass
    (issue #1354): the stalled-review sweep needs the sidecar's log tail to
    classify provider-throttle signatures and arm fleet-wide backoff.
    """
    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    _set_review_dispatched_state(app, 100, 10, "2026-07-06T12:00:00Z")

    reviews_dir = app._layout.reviews_dir
    _write_review_events(reviews_dir, 100, turns=3, tool_calls=2)
    sidecar_path = _make_dead_review_sidecar(reviews_dir, 100, "log text")

    monkeypatch.setattr(app, "_comment_pr", lambda *a, **kw: None)

    app._reap_review_verdicts(reviews_dir)

    # The sidecar must still exist for the stalled-review sweep.
    assert sidecar_path.exists(), (
        "Sidecar was reaped in _reap_review_verdicts -- the stalled-review "
        "sweep needs it for throttle classification"
    )


# --- AC1 regression: exit-code fallback reads the reviewer's terminal-status
# record, not the original coding worker's stale one (PR #1356 round-2 review) ---


def test_reap_review_verdicts_exit_code_from_reviewer_terminal_record(
    monkeypatch, tmp_path: Path
) -> None:
    """``_reap_review_verdicts`` reads the reviewer's exit code from the
    terminal-status record written under ``reviews_dir`` keyed by ``pr_number``
    -- not from the layout's default sessions dir keyed by the linked issue
    number (PR #1356 round-2 review).

    The review launch site (``dispatch_reviews``) calls
    ``launch_claude_worker(sessions_dir=reviews_dir, issue_number=pr_number,
    review=True)``, and ``start_terminal_status_watcher`` writes the
    terminal-status file to ``worker_terminal_status_path(sessions_dir,
    issue_number, ...)`` = ``reviews_dir/issue-<pr_number>.claude.terminal.json``.

    Before the fix, ``_reap_review_verdicts`` called
    ``find_worker_terminal_status(self._layout.sessions_dir, issue_number)``,
    which looked under the default sessions dir keyed by the linked issue
    number -- silently returning the original coding worker's stale exit code
    instead of the reviewer's. This test plants both records with distinct
    exit codes and asserts the reviewer's wins.
    """
    from charlie_work.process_utils import (
        worker_terminal_status_path,
        write_worker_terminal_status,
    )

    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    # PR #100 is linked to issue #10.
    _set_review_dispatched_state(app, 100, 10, "2026-07-06T12:00:00Z")

    reviews_dir = app._layout.reviews_dir
    # A dead reviewer that did substantial work but had no result event
    # (stream cut = died_mid_session), so the exit code is the only
    # terminal signal.
    _write_review_events(reviews_dir, 100, turns=3, tool_calls=2)
    _make_dead_review_sidecar(reviews_dir, 100, "log text")

    # The reviewer's real terminal-status record, written under reviews_dir
    # keyed by pr_number (the actual production write path).
    write_worker_terminal_status(
        worker_terminal_status_path(reviews_dir, 100, "claude"),
        pid=99999,
        exit_code=137,
        started_at="2026-07-06T12:00:00Z",
        ended_at="2026-07-06T12:05:00Z",
        duration_seconds=300.0,
    )

    # A STALE terminal-status record for the original coding worker, written
    # under the layout's default sessions dir keyed by the linked issue
    # number. Before the fix, _reap_review_verdicts read THIS record and
    # silently returned exit_code=0 (the coding worker's clean exit) instead
    # of the reviewer's crash (137).
    sessions_dir = app._layout.sessions_dir
    write_worker_terminal_status(
        worker_terminal_status_path(sessions_dir, 10, "claude"),
        pid=88888,
        exit_code=0,
        started_at="2026-07-06T11:00:00Z",
        ended_at="2026-07-06T11:30:00Z",
        duration_seconds=1800.0,
    )

    monkeypatch.setattr(app, "_comment_pr", lambda *a, **kw: None)

    result = app._reap_review_verdicts(reviews_dir)

    missed = result.get("missed", [])
    assert len(missed) == 1
    assert missed[0]["reason"] == "died_mid_session"
    cause = missed[0]["cause"]
    # The reviewer's crash exit code (137), not the coding worker's clean
    # exit (0).
    assert cause["exit_code"] == 137, (
        f"expected reviewer's exit_code=137 from reviews_dir terminal record, "
        f"got {cause.get('exit_code')!r} -- _reap_review_verdicts is reading "
        f"from the wrong directory/key"
    )

    # Verify the event payload carries the reviewer's exit code too.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
    events = _events(state, "review_verdict_missed")
    assert len(events) == 1
    payload = events[0].get("payload", {})
    assert payload["cause"]["exit_code"] == 137
