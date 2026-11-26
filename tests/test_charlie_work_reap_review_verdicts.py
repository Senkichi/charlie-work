"""Reap of review-verdict sessions: verdict recording, session metrics folding, missing-events-file handling, and invalid-verdict leave-for-stalled-reaper.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

import json
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _helpers import _init_git_repo
from _review_fixtures import (
    _dispatch_reviews_app,
    _make_dead_opencode_review_sidecar,
    _make_dead_review_sidecar,
    _set_review_dispatched_state,
    _write_review_events,
    _write_review_packet,
)
from _rework_dispatch_fixtures import _wg
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import _detect_and_handle_stalled_reviews


def test_reap_review_verdicts_records_valid_verdict(monkeypatch, tmp_path: Path) -> None:
    """Issue #507: a dead reviewer with a valid verdict block has its verdict recorded."""
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
    reviews_dir = app._layout.reviews_dir

    verdict_log = (
        "Final verdict:\n```json\n{\n"
        '  "decision": "request_changes",\n'
        '  "summary": "fix the edge case in validate()",\n'
        '  "required_changes": ["add null check"]\n'
        "}\n```\n"
    )
    _make_dead_review_sidecar(reviews_dir, 100, verdict_log)
    _set_review_dispatched_state(app, 100, 10, "2026-07-06T12:00:00Z")

    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    assert result["recorded"] == [
        {"pr": 100, "issue": 10, "decision": "request_changes", "verdict_source": "log"}
    ]
    assert result["missed"] == []

    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_completed"
    assert state["prs"]["100"]["status"] == "request_changes"
    assert state["issues"]["10"]["status"] == "rework_requested"

    decision_path = app.paths.prs / "pr-100" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "request_changes"
    assert decision["summary"] == "fix the edge case in validate()"
    assert decision["required_changes"] == ["add null check"]
    assert decision["reviewed_head_sha"] == "sha-100"


def test_reap_review_verdicts_records_session_metrics(monkeypatch, tmp_path: Path) -> None:
    """A dead reviewer's events.jsonl telemetry (tokens/cost/turns/tool-calls) must
    flow into the record_review event payload and the PR's state entry."""
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
    reviews_dir = app._layout.reviews_dir

    verdict_log = (
        "Final verdict:\n```json\n{\n"
        '  "decision": "approved",\n'
        '  "summary": "lgtm",\n'
        '  "required_changes": []\n'
        "}\n```\n"
    )
    _make_dead_review_sidecar(reviews_dir, 100, verdict_log)
    _set_review_dispatched_state(app, 100, 10, "2026-07-06T12:00:00Z")

    events_path = reviews_dir / "issue-100-review.events.jsonl"
    events = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use"}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use"}]}},
        {
            "type": "result",
            "num_turns": 4,
            "total_cost_usd": 1.25,
            "usage": {"input_tokens": 900, "output_tokens": 100},
        },
    ]
    events_path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )

    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    assert result["recorded"] == [
        {"pr": 100, "issue": 10, "decision": "approved", "verdict_source": "log"}
    ]

    state = load_state(app.paths.state_file)
    metrics = state["prs"]["100"]["review_session_metrics"]
    assert metrics == {
        "tokens": 1000,
        "cost_usd": 1.25,
        "turn_count": 4,
        "tool_call_count": 2,
        "verdict_source": "log",
    }

    record_events = [e for e in state["events"] if e["kind"] == "record_review"]
    assert record_events, "expected a record_review event"
    assert record_events[-1]["payload"]["session_metrics"] == metrics


def test_reap_review_verdicts_folds_review_effort_arm_into_session_metrics(
    monkeypatch, tmp_path: Path
) -> None:
    """The review_effort experiment's arm/effort (recorded on pr_state at
    dispatch/claim time) must be folded into session_metrics at reap time, so
    the record_review event alone is enough to split spend/quality by arm."""
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
    reviews_dir = app._layout.reviews_dir

    verdict_log = (
        "Final verdict:\n```json\n{\n"
        '  "decision": "approved",\n'
        '  "summary": "lgtm",\n'
        '  "required_changes": []\n'
        "}\n```\n"
    )
    _make_dead_review_sidecar(reviews_dir, 100, verdict_log)
    _set_review_dispatched_state(app, 100, 10, "2026-07-06T12:00:00Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"] = {
            **state["prs"]["100"],
            "review_effort_arm": "treatment",
            "review_effort_used": "high",
        }
        save_state(app.paths.state_file, state)

    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    assert result["recorded"] == [
        {"pr": 100, "issue": 10, "decision": "approved", "verdict_source": "log"}
    ]

    state = load_state(app.paths.state_file)
    metrics = state["prs"]["100"]["review_session_metrics"]
    assert metrics["review_effort_arm"] == "treatment"
    assert metrics["review_effort_used"] == "high"

    record_events = [e for e in state["events"] if e["kind"] == "record_review"]
    assert record_events, "expected a record_review event"
    assert record_events[-1]["payload"]["session_metrics"]["review_effort_arm"] == "treatment"
    assert record_events[-1]["payload"]["session_metrics"]["review_effort_used"] == "high"


def test_reap_review_verdicts_missing_events_file_records_verdict_with_no_metrics(
    monkeypatch, tmp_path: Path
) -> None:
    """A missing/unparseable events.jsonl sidecar must never block verdict
    recording — session_metrics is simply absent."""
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
    reviews_dir = app._layout.reviews_dir

    verdict_log = (
        "Final verdict:\n```json\n{\n"
        '  "decision": "approved",\n'
        '  "summary": "lgtm",\n'
        '  "required_changes": []\n'
        "}\n```\n"
    )
    _make_dead_review_sidecar(reviews_dir, 100, verdict_log)
    _set_review_dispatched_state(app, 100, 10, "2026-07-06T12:00:00Z")
    # Deliberately do not create issue-100-review.events.jsonl.

    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    assert result["recorded"] == [
        {"pr": 100, "issue": 10, "decision": "approved", "verdict_source": "log"}
    ]
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["status"] == "approved"
    assert state["prs"]["100"].get("review_session_metrics") is None
    record_events = [e for e in state["events"] if e["kind"] == "record_review"]
    assert record_events
    assert "session_metrics" not in record_events[-1]["payload"]


def test_reap_review_verdicts_leaves_invalid_verdict_for_stalled_reaper(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #507: a dead reviewer whose log has no parseable verdict falls through
    to the existing stale-claim failed path unchanged."""
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
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
    reviews_dir = app._layout.reviews_dir

    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    _make_dead_review_sidecar(reviews_dir, 100, "Truncated log with no verdict block")
    # Seed a real turn-limit session: without event telemetry the reviewer has
    # zero turns, which since issue #588 classifies as a launch failure rather
    # than a turn-limit death.
    _write_review_events(
        reviews_dir, 100, turns=app.config.review_dispatch.review_max_turns, tool_calls=3
    )
    _set_review_dispatched_state(app, 100, 10, old_dispatched)

    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: False)

    reap = app._reap_review_verdicts(reviews_dir)
    assert reap["recorded"] == []
    # The reviewer had prose in its log (no verdict), so a turn-limit summary
    # comment is posted and the miss is recorded with that reason.
    assert len(reap["missed"]) == 1
    assert reap["missed"][0]["reason"] == "turn_limit_summary_posted"

    # State should still be dispatched (reaper does not mark failed) and
    # _detect_and_handle_stalled_reviews should then move it to failed.
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir,
        app.paths.state_file,
        app.config,
        repo_root,
        write_gate=_wg(app.paths.state_file),
    )
    assert any(entry.get("pr") == 100 for entry in stalled)

    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_failed"


def test_reap_sweeps_opencode_quota_death_rolls_back_claim(monkeypatch, tmp_path: Path) -> None:
    """End-to-end ordering regression for the 0-tool-call opencode quota death.

    _run_review_reap_sweeps runs _reap_review_verdicts BEFORE
    _detect_and_handle_stalled_reviews. A reviewer that emitted step_start
    then died on a provider quota error with zero tool calls must classify as
    launch_failed (not died_mid_session), so the miss pass does NOT set
    review_turn_limit_summary_posted -- and the stalled sweep that follows in
    the SAME pass takes the provider_throttled rollback path instead of
    failing the PR as provider_throttled_turn_limit_counted.
    """
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
    reviews_dir = app._layout.reviews_dir

    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    # step_start + provider-quota error, zero tool calls: the reviewer died
    # inside its first provider call (GoUsageLimitError in the response body,
    # the shape opencode's opencode-go provider emits).
    _make_dead_opencode_review_sidecar(
        reviews_dir,
        100,
        '{"type":"step_start","part":{"id":"s1"}}\n'
        '{"type":"error","error":{"name":"APIError","data":{"message":"Usage limit reached",'
        '"statusCode":429,"isRetryable":true,"responseBody":"{\\"type\\":\\"error\\",'
        '\\"error\\":{\\"type\\":\\"GoUsageLimitError\\"}}"}}}\n',
        started_at=old_dispatched,
    )
    _set_review_dispatched_state(app, 100, 10, old_dispatched)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"]["review_dispatch_attempt_count"] = 1
        save_state(app.paths.state_file, state)

    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: False)

    result = app._run_review_reap_sweeps(datetime.now(UTC))

    # The miss pass classified the death environmental (no substantial work),
    # so no turn-limit summary flag exists to deny the rollback.
    assert [entry["reason"] for entry in result["verdict_result"]["missed"]] == ["launch_failed"]
    # The stalled sweep in the same pass took the rollback path, not the
    # counted-failure path.
    assert [entry["reason"] for entry in result["stalled"]] == ["provider_throttled"]

    state = load_state(app.paths.state_file)
    pr = state["prs"]["100"]
    assert not pr.get("review_turn_limit_summary_posted")
    # Claim rolled back (re-dispatchable once quota clears), NOT failed.
    assert pr["review_dispatch_status"] is None
    assert pr["review_dispatch_status"] != "review_dispatch_failed"
    # The quota death must not burn the per-PR attempt budget.
    assert pr["review_dispatch_attempt_count"] == 0
    # The throttle signal itself is still armed fleet-wide.
    assert state.get("reviewer_quota", {}).get("throttled_until")


def test_reap_sweeps_opencode_v2_retry_death_rolls_back_claim(monkeypatch, tmp_path: Path) -> None:
    """The 2.x wire shape of the same e2e ordering invariant: opencode 2.3.0
    emits {"type":"retry"} events (carrying the structured quota error) while
    it sleeps out retry-after, and a watchdog kill inside that sleep may never
    produce a terminal error event. The retry records must classify the death
    provider_throttled exactly like the 1.x error-event fixture above."""
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
    reviews_dir = app._layout.reviews_dir

    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    # The captured 2.3.0 shape: structured error is flat, retry-after at the
    # event top level, no terminal error event -- the retry record is the
    # only provider-error evidence.
    _make_dead_opencode_review_sidecar(
        reviews_dir,
        100,
        '{"type":"step_start","part":{"id":"s1"}}\n'
        '{"type":"retry","messageID":"msg_1","attempt":1,'
        '"nextRetryAt":1763473200000,"error":{"type":"provider.quota",'
        '"message":"Usage limit reached -- you have used your monthly budget",'
        '"status":429,"response":{"body":"{\\"error\\":{\\"message\\":'
        '\\"Usage limit reached\\",\\"type\\":\\"GoUsageLimitError\\"}}"}},"retryAfterMs":3600000}\n',
        started_at=old_dispatched,
    )
    _set_review_dispatched_state(app, 100, 10, old_dispatched)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"]["review_dispatch_attempt_count"] = 1
        save_state(app.paths.state_file, state)

    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: False)

    result = app._run_review_reap_sweeps(datetime.now(UTC))

    assert [entry["reason"] for entry in result["verdict_result"]["missed"]] == ["launch_failed"]
    assert [entry["reason"] for entry in result["stalled"]] == ["provider_throttled"]

    state = load_state(app.paths.state_file)
    pr = state["prs"]["100"]
    assert not pr.get("review_turn_limit_summary_posted")
    assert pr["review_dispatch_status"] is None
    assert pr["review_dispatch_attempt_count"] == 0
    assert state.get("reviewer_quota", {}).get("throttled_until")
