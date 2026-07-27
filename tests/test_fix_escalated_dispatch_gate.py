"""Tests for issue #575: escalated PRs/issues must never be re-dispatched to
a reviewer, and any verdict ``_reap_review_verdicts`` cannot land must be
visible as an event.

Live incident (2026-07-25): issue #480's state status was ``escalated``
while its PR #540 was still ``review_dispatch_status == review_dispatch_dispatched``
(``reviewing``). ``dispatch_reviews`` kept selecting it as a candidate (the
escalation guard lives only in ``record_review``), so three reviewer sessions
were launched at real provider-quota cost, each producing a verdict that
``record_review``'s escalation guard (see
``test_fix_escalation_paths.py::test_record_review_escalated_pr_blocks_and_writes_no_decision``)
then refused. The refusal only ever landed in ``_reap_review_verdicts``'s
``missed`` list, which ``dispatch_reviews`` returns in its result dict but
never emits as an event -- so the loss was invisible in ``events.db``, and
the PR went on to hit a bogus attempt-cap re-escalation.

Two fixes under test:

1. ``dispatch_reviews``'s candidate-selection loop now skips any PR whose
   pr-state ``status`` is ``"escalated"``, or whose linked issue's state
   ``status`` is ``"escalated"``, BEFORE writing a ``review_dispatch_pending``
   claim and BEFORE incrementing ``review_dispatch_attempt_count``. Skipped
   PRs are reported in ``result.data["escalated_skipped"]`` but do not emit a
   per-pass event (it would fire every pass while the PR awaits a human).
2. Every append to ``_reap_review_verdicts``'s ``missed`` list now also
   emits a ``review_verdict_missed`` event with
   ``{pr_number, issue_number, reason}``.

Reuses ``_dispatch_reviews_app``/``_write_review_packet``/
``_make_dead_review_sidecar``/``_set_review_dispatched_state`` from
test_charlie_work.py (PR #566/#507 review-reaper fixtures).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from charlie_work.state import load_state, save_state, state_lock

from test_charlie_work import (
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


def _events(state: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


def _seed_pr_state(paths, **fields: Any) -> None:
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["100"] = {"number": 100, "issue_number": 10, **fields}
        save_state(paths.state_file, state)


def _seed_issue_state(paths, **fields: Any) -> None:
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["10"] = {"number": 10, **fields}
        save_state(paths.state_file, state)


def _no_launch(monkeypatch) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Fail the test loudly if a launch is attempted -- these tests assert
    the escalated candidate is never dispatched at all."""
    launched: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> Any:
        launched.append((args, kwargs))
        raise AssertionError("launch_claude_worker must not be called for an escalated PR")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    return launched


# --- Fix 1: dispatch_reviews skips escalated candidates ---


def test_dispatch_reviews_skips_pr_state_escalated(monkeypatch, tmp_path: Path) -> None:
    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    _seed_pr_state(app.paths, status="escalated")
    launched = _no_launch(monkeypatch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert launched == []
    assert result.data["escalated_skipped"] == [100]
    assert result.data["launched_count"] == 0
    assert result.data["selected_count"] == 0

    state = load_state(app.paths.state_file)
    pr_state = state["prs"]["100"]
    # No pending claim written and no attempt-count increment.
    assert pr_state.get("review_dispatch_status") is None
    assert int(pr_state.get("review_dispatch_attempt_count", 0)) == 0
    assert pr_state["status"] == "escalated"
    assert _events(state, "review_dispatch_claim") == []


def test_dispatch_reviews_skips_linked_issue_escalated(monkeypatch, tmp_path: Path) -> None:
    """The PR itself is clean (open/reviewing) but its linked issue carries
    the escalation. record_review's guard checks both sides; the dispatch
    gate must too."""
    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    _seed_issue_state(app.paths, status="escalated")
    launched = _no_launch(monkeypatch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert launched == []
    assert result.data["escalated_skipped"] == [100]
    assert result.data["launched_count"] == 0

    state = load_state(app.paths.state_file)
    # No PR entry (or an untouched one) -- no claim written, no dispatch.
    pr_state = state["prs"].get("100", {})
    assert pr_state.get("review_dispatch_status") is None
    assert int(pr_state.get("review_dispatch_attempt_count", 0)) == 0
    assert state["issues"]["10"]["status"] == "escalated"


def test_dispatch_reviews_skips_issue_escalated_pr_at_attempt_cap_without_re_escalating(
    monkeypatch, tmp_path: Path
) -> None:
    """The escalation gate must run BEFORE the attempt-cap escalation branch,
    not just before the final dispatch selection. If an issue is escalated
    via an independent path (not this attempt-cap branch) while its PR
    happens to already be sitting at review_dispatch_attempt_count ==
    max_attempts, the attempt-cap branch must not fire a second, bogus
    "max_review_dispatch_attempts_exceeded" review_dispatch_escalated event
    on top of it -- that duplicate/contradictory escalation reason is exactly
    the "bogus attempt-cap re-escalation" this issue names as a harm."""
    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    max_attempts = app.config.review_dispatch.max_review_dispatch_attempts
    _seed_pr_state(app.paths, review_dispatch_attempt_count=max_attempts)
    _seed_issue_state(app.paths, status="escalated")
    launched = _no_launch(monkeypatch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert launched == []
    assert result.data["escalated_skipped"] == [100]

    state = load_state(app.paths.state_file)
    assert _events(state, "review_dispatch_escalated") == []
    # The PR-side status is untouched by this pass -- the issue-side
    # escalation is the sole source of truth here.
    assert state["prs"]["100"].get("status") != "escalated"
    assert int(state["prs"]["100"]["review_dispatch_attempt_count"]) == max_attempts


def test_dispatch_reviews_still_dispatches_non_escalated_pr(monkeypatch, tmp_path: Path) -> None:
    """Control: a PR with no escalation anywhere is unaffected by the gate."""
    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")

    from test_charlie_work import _fake_claude_worker_record

    launched: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> Any:
        launched.append((args, kwargs))
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["escalated_skipped"] == []
    assert result.data["launched_count"] == 1
    assert len(launched) == 1

    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert int(state["prs"]["100"]["review_dispatch_attempt_count"]) == 1


# --- Fix 2: _reap_review_verdicts emits review_verdict_missed for every miss ---


def test_reap_review_verdicts_emits_event_when_record_review_refuses_escalated(
    monkeypatch, tmp_path: Path
) -> None:
    """The live-incident scenario: a dead reviewer produced a perfectly valid
    verdict, but by the time the reaper runs, the issue has been escalated
    (e.g. by a concurrent attempt-cap sweep). record_review refuses via its
    escalation guard, and the refusal must be recorded as a
    review_verdict_missed event -- not silently dropped."""
    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    reviews_dir = app._resolve(app.config.review_dispatch.reviews_dir)

    verdict_log = (
        "Final verdict:\n```json\n{\n"
        '  "decision": "approved",\n'
        '  "summary": "lgtm",\n'
        '  "required_changes": []\n'
        "}\n```\n"
    )
    _make_dead_review_sidecar(reviews_dir, 100, verdict_log)
    _set_review_dispatched_state(app, 100, 10, "2026-07-06T12:00:00Z")
    _seed_issue_state(app.paths, status="escalated")

    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    assert result["recorded"] == []
    assert len(result["missed"]) == 1
    missed = result["missed"][0]
    assert missed["pr"] == 100
    assert missed["issue"] == 10
    assert "escalated" in missed["reason"]

    state = load_state(app.paths.state_file)
    missed_events = _events(state, "review_verdict_missed")
    assert len(missed_events) == 1
    payload = missed_events[0]["payload"]
    assert payload["pr_number"] == 100
    assert payload["issue_number"] == 10
    assert "escalated" in payload["reason"]

    # The escalation guard is a hard block: no decision file, no state churn
    # beyond the event -- the claim stays dispatched so the stalled-review
    # sweep can disposition it via the existing path.
    decision_path = app.paths.prs / "pr-100" / "review-decision.json"
    assert not decision_path.exists()
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"


def test_reap_review_verdicts_launch_failure_is_not_reported_as_turn_limit(
    monkeypatch, tmp_path: Path
) -> None:
    """A reviewer that died before its first turn must not be labelled a
    turn-limit death (issue #588).

    The fixture is the real shape of the 2026-07-22 outage: no events sidecar
    at all, and a log holding only the process's own error text. Zero turns and
    zero tool calls means the session never ran, so its death is environmental
    and says nothing about this PR. Reporting it as ``turn_limit_summary_posted``
    pointed every diagnostic at ``review_max_turns`` for 25 hours while the
    actual cause was a rejected argv.
    """
    from datetime import UTC, datetime, timedelta

    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    reviews_dir = app._resolve(app.config.review_dispatch.reviews_dir)

    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    _make_dead_review_sidecar(
        reviews_dir,
        100,
        "Error: When using --print, --output-format=stream-json requires --verbose",
    )
    _set_review_dispatched_state(app, 100, 10, old_dispatched)

    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    # The miss is still reported -- only its classification changes.
    assert len(result["missed"]) == 1
    assert result["missed"][0]["reason"] == "launch_failed"

    state = load_state(app.paths.state_file)
    missed_events = _events(state, "review_verdict_missed")
    assert len(missed_events) == 1
    assert missed_events[0]["payload"] == {
        "pr_number": 100,
        "issue_number": 10,
        "reason": "launch_failed",
        "turn_count": 0,
        "tool_call_count": 0,
    }

    # The turn-limit marker gates two downstream behaviours: #584 counts it as
    # a turn-limit death, and the provider-throttle sweep reads it to deny a
    # rollback (#583). A session that never ran must set neither.
    pr_state = state["prs"]["100"]
    assert pr_state.get("review_turn_limit_summary_posted") is False
    assert pr_state.get("review_miss_summary_posted") is True


def test_reap_review_verdicts_turn_limit_miss_emits_event(monkeypatch, tmp_path: Path) -> None:
    """Regression guard for the other missed-append site: a dead reviewer with
    no parseable verdict (turn-limit path) must also emit
    review_verdict_missed, matching test_charlie_work.py's
    test_reap_review_verdicts_leaves_invalid_verdict_for_stalled_reaper.

    Unlike the launch-failure case above, this session genuinely exhausted its
    turn budget, so it keeps the ``turn_limit_summary_posted`` reason and sets
    the marker that denies the provider-throttle rollback."""
    from datetime import UTC, datetime, timedelta

    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    reviews_dir = app._resolve(app.config.review_dispatch.reviews_dir)

    max_turns = app.config.review_dispatch.review_max_turns
    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    _make_dead_review_sidecar(reviews_dir, 100, "Truncated log with no verdict block")
    _write_review_events(reviews_dir, 100, turns=max_turns, tool_calls=3)
    _set_review_dispatched_state(app, 100, 10, old_dispatched)

    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    assert len(result["missed"]) == 1
    assert result["missed"][0]["reason"] == "turn_limit_summary_posted"

    state = load_state(app.paths.state_file)
    missed_events = _events(state, "review_verdict_missed")
    assert len(missed_events) == 1
    assert missed_events[0]["payload"] == {
        "pr_number": 100,
        "issue_number": 10,
        "reason": "turn_limit_summary_posted",
        "turn_count": max_turns,
        "tool_call_count": 3,
    }
    assert state["prs"]["100"].get("review_turn_limit_summary_posted") is True


def test_reap_review_verdicts_death_after_partial_work_is_its_own_bucket(
    monkeypatch, tmp_path: Path
) -> None:
    """A reviewer that did real work but died short of the turn cap is neither
    a launch failure nor a turn-limit death (issue #588)."""
    from datetime import UTC, datetime, timedelta

    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    reviews_dir = app._resolve(app.config.review_dispatch.reviews_dir)

    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    _make_dead_review_sidecar(reviews_dir, 100, "Truncated log with no verdict block")
    _write_review_events(reviews_dir, 100, turns=3, tool_calls=2)
    _set_review_dispatched_state(app, 100, 10, old_dispatched)

    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    assert result["missed"][0]["reason"] == "died_mid_session"
    state = load_state(app.paths.state_file)
    # It did substantial work, so the throttle-rollback denial still applies.
    assert state["prs"]["100"].get("review_turn_limit_summary_posted") is True


# Claude Code's account-level session-limit notice, verbatim as observed
# 2026-07-21 (see RuntimeConfig.throttle_error_markers). Distinct from the
# "rate limit"/"usage limit" phrasings and the --verbose argv crash covered by
# test_reap_review_verdicts_launch_failure_is_not_reported_as_turn_limit above.
_SESSION_LIMIT_NOTICE = "You've hit your session limit \u00b7 resets 4:40pm (America/Los_Angeles)"


def _write_session_limit_events(reviews_dir: Path, pr_number: int) -> Path:
    """Write an events sidecar for a reviewer that died on the session-limit notice.

    ``parse_claude_events`` counts the notice's assistant event as one turn but
    zero tool calls -- the exact shape that defeated the #583 rollback guard
    before issue #651 (the notice is counted as a turn, so it failed the
    ``turn_count == 0 and tool_call_count == 0`` launch-failure check and fell
    through to ``died_mid_session``).
    """
    reviews_dir.mkdir(parents=True, exist_ok=True)
    events_path = reviews_dir / f"issue-{pr_number}-review.events.jsonl"
    line = json.dumps(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": _SESSION_LIMIT_NOTICE}]},
        }
    )
    events_path.write_text(line + "\n", encoding="utf-8")
    return events_path


def test_reap_review_verdicts_session_limit_notice_is_launch_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """A reviewer whose only output is the session-limit notice must be classified
    as a launch failure, not ``died_mid_session`` (issue #651).

    The notice gets counted as one turn by ``parse_claude_events`` (turn_count=1,
    tool_call_count=0), so before this fix it failed the
    ``turn_count == 0 and tool_call_count == 0`` check and fell through to
    ``REVIEW_MISS_DIED_MID_SESSION``. That made ``did_substantial_work`` True,
    persisting ``review_turn_limit_summary_posted=True`` and silently defeating
    the #583 throttle-rollback guard: a global session-limit outage burned a PR's
    ``review_dispatch_attempt_count`` budget with zero actual review work, eventually
    tripping ``max_review_dispatch_attempts_exceeded`` on a PR that was never really
    reviewed.
    """
    from datetime import UTC, datetime, timedelta

    app = _dispatch_reviews_app(tmp_path, prs=[_PR])
    _write_review_packet(tmp_path, 100, "sha-100")
    reviews_dir = app._resolve(app.config.review_dispatch.reviews_dir)

    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    _make_dead_review_sidecar(reviews_dir, 100, _SESSION_LIMIT_NOTICE)
    _write_session_limit_events(reviews_dir, 100)
    _set_review_dispatched_state(app, 100, 10, old_dispatched)

    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    assert len(result["missed"]) == 1
    assert result["missed"][0]["reason"] == "launch_failed"

    state = load_state(app.paths.state_file)
    missed_events = _events(state, "review_verdict_missed")
    assert len(missed_events) == 1
    assert missed_events[0]["payload"] == {
        "pr_number": 100,
        "issue_number": 10,
        "reason": "launch_failed",
        "turn_count": 1,
        "tool_call_count": 0,
    }

    # did_substantial_work is False, so the #583 rollback guard actually fires:
    # the turn-limit marker must NOT be set, otherwise the throttle sweep takes
    # the counted-failure path and consumes the attempt budget for zero work.
    pr_state = state["prs"]["100"]
    assert pr_state.get("review_turn_limit_summary_posted") is False
    assert pr_state.get("review_miss_summary_posted") is True


def test_extract_review_session_summary_session_limit_notice_no_substantial_work(
    tmp_path: Path,
) -> None:
    """Direct unit test for issue #651: a session-limit-notice session with
    turn_count=1, tool_call_count=0 must report ``did_substantial_work=False``
    so the #583 rollback guard fires, regardless of turn_count."""
    from charlie_work.workflow import _extract_review_session_summary

    events_path = tmp_path / "issue-100-review.events.jsonl"
    line = json.dumps(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": _SESSION_LIMIT_NOTICE}]},
        }
    )
    events_path.write_text(line + "\n", encoding="utf-8")
    log_path = tmp_path / "issue-100-review.claude.log"
    log_path.write_text(_SESSION_LIMIT_NOTICE + "\n", encoding="utf-8")

    outcome = _extract_review_session_summary(events_path, log_path, max_turns=20)

    assert outcome is not None
    assert outcome.turn_count == 1
    assert outcome.tool_call_count == 0
    assert outcome.reason == "launch_failed"
    assert outcome.did_substantial_work is False
