"""Issue #1340: review-decision.json never resynced after a log-extracted
verdict -- file says pending while state.json carries the decision.

Regression tests for two coordinated fixes:

1. ``_reap_review_verdicts`` now threads ``verdict_source`` (the parser-level
   provenance -- "log", "events", "file:<source>") into ``record_review``, which
   persists it into ``review-decision.json`` so a later reader can distinguish
   a log-extracted verdict (dead reviewer) from a clean structured completion.

2. ``review()``'s pending-reset decision-file write is now inside the
   ``state_lock`` and, when a stale-head terminal verdict is voided back to
   "pending", the state.json ``decision``/``reviewed_head_sha`` fields are
   cleared in the same locked section. Without this, a head advance after a
   log-extracted approval left the file at "pending" while state.json retained
   ``decision: "approved"`` -- a file-trusting consumer (the packet-current
   skip in ``loop()``) saw "pending" while state-trusting paths acted on the
   stale approval.
"""

from __future__ import annotations

import json
from pathlib import Path

from _fakes_github import FakeGitHub
from _review_fixtures import (
    _dispatch_reviews_app,
    _make_dead_review_sidecar,
    _set_review_dispatched_state,
    _write_review_packet,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp


def test_log_extracted_verdict_persists_verdict_source_in_decision_file(
    monkeypatch, tmp_path: Path
) -> None:
    """A log-extracted verdict (dead reviewer) must carry
    ``verdict_source: "log"`` in review-decision.json, and the file and
    state.json must agree on the decision."""
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

    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    assert result["recorded"] == [
        {"pr": 100, "issue": 10, "decision": "approved", "verdict_source": "log"}
    ]

    decision_path = app.paths.prs / "pr-100" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "approved"
    assert decision["verdict_source"] == "log", (
        "issue #1340: a log-extracted verdict must persist verdict_source='log' "
        "in the decision file so a later reader can distinguish it from a clean "
        "structured completion"
    )
    assert decision["verdict_provenance"] == "fresh_llm_review"

    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["decision"] == "approved", (
        "issue #1340: state.json and review-decision.json must agree on the "
        "decision after a log-extracted verdict"
    )


def test_review_pending_reset_keeps_file_and_state_in_agreement_after_log_verdict(
    monkeypatch, tmp_path: Path
) -> None:
    """After a log-extracted approval is recorded, a subsequent ``review()``
    call with a moved head must void the decision file back to "pending" AND
    clear state.json's ``decision``/``reviewed_head_sha`` in the same locked
    section, so the two stores never diverge (file "pending" while state
    carries "approved")."""
    prs = [
        {
            "number": 100,
            "title": "fix: search bug",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10\n\nTests: regression coverage added.",
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

    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)

    # Step 1: reap the log-extracted verdict -- both stores agree on "approved".
    app._reap_review_verdicts(reviews_dir)

    decision_path = app.paths.prs / "pr-100" / "review-decision.json"
    state = load_state(app.paths.state_file)
    assert json.loads(decision_path.read_text(encoding="utf-8"))["decision"] == "approved"
    assert state["prs"]["100"]["decision"] == "approved"

    # Step 2: the PR head advances (e.g. a rework push or carry-forward). The
    # next loop pass calls review() for the new head. The stale approval
    # pinned to sha-100 must be voided -- in BOTH stores, not just the file.
    # FakeGitHub needs issue #10 to exist for review()'s issue_view call.
    app.gh.issues.append(
        {
            "number": 10,
            "title": "Fix the bug",
            "url": "https://example.test/issues/10",
            "body": "Bug description",
            "labels": [],
            "state": "OPEN",
        }
    )
    app.gh.pr_head_shas[100] = "sha-200"
    result = app.review(100)
    assert result.ok, result.message

    after_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    after_state = load_state(app.paths.state_file)

    assert after_decision["decision"] == "pending", (
        "the stale approval must be voided back to pending in the decision file"
    )
    assert after_state["prs"]["100"]["decision"] == "pending", (
        "issue #1340: state.json must also be reset to 'pending' when the "
        "decision file is voided -- without this, the file says 'pending' "
        "while state.json carries the stale 'approved', and the packet-current "
        "skip in loop() (file-trusting) diverges from state-trusting consumers"
    )


def test_review_fresh_packet_does_not_clear_existing_state_decision(tmp_path: Path) -> None:
    """When ``review()`` writes a fresh pending template (no prior decision
    file), it must NOT inject a ``decision: "pending"`` key into state.json --
    that would clobber a state-only decision from a concurrent path. The
    state-clear only fires when a stale terminal verdict is actually voided."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    # No prior review-decision.json -- the fresh-packet path, not the
    # stale-void path.
    decision_path = paths.prs / "pr-456" / "review-decision.json"
    assert not decision_path.exists()

    result = app.review(456)
    assert result.ok, result.message

    after_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert after_decision["decision"] == "pending"

    state = load_state(app.paths.state_file)
    # The fresh-packet path must not inject a "decision" key -- only the
    # stale-void path does, and only when a terminal verdict was actually
    # voided. Injecting "pending" here would mask a state-only decision
    # written by a concurrent record_review that hasn't written its file yet.
    assert "decision" not in state["prs"]["456"], (
        "the fresh-packet pending write must not inject a decision key into "
        "state.json -- only the stale-void path clears state, and only when "
        "a terminal verdict was actually voided"
    )
