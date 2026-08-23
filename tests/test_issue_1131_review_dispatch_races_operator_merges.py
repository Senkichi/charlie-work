"""Tests for issue #1131: review_dispatch races operator merges.

The observed race (2026-08-10, PRs 1538/1541): an operator
unescalate -> approve -> merge interleaved with a dispatch pass, so a
reviewer's ``request_changes`` landed ~8 minutes after the merge and:

1. clobbered ``prs/pr-1538/review-decision.json`` (approved ->
   request_changes), destroying the audit record that authorized the merge;
2. fired the unauthorized-merge tripwire as a false positive;
3. applied ``agent:needs-rework`` to an already-CLOSED issue.

Three single-enforcement-point fixes:

- ``record_review`` refuses when the PR is already MERGED/CLOSED (a
  terminal-state PR must not have its decision file overwritten or rework
  routed).
- ``dispatch_reviews`` re-checks the decision file at claim time and skips
  any PR that already records an approval for the head this pass would
  review (``review_queue``'s snapshot can be stale when a pass interleaves
  with an operator approve -> merge).
- ``record_review``'s rework label transition checks the linked issue's
  GitHub state and never applies ``agent:needs-rework`` to a CLOSED issue.

These tests reuse ``FakeGitHub`` (PR #456 <-> issue #123) and the
packet-seeding helper pattern from test_fix_1497_review_dispatch_merge_conflict.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import CommandResult, OrchestratorApp

from _fakes_github import FakeGitHub


def _write_review_packet(paths, pr_number: int, head_sha: str) -> None:
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        f'{{"number": {pr_number}, "headRefOid": "{head_sha}"}}', encoding="utf-8"
    )
    (pr_dir / "review-prompt.md").write_text("review prompt", encoding="utf-8")


def _write_decision(paths, pr_number: int, payload: dict[str, Any]) -> None:
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(json.dumps(payload), encoding="utf-8")


def _seed_reviewing_pr(paths, pr_number: int, issue_number: int) -> None:
    """Seed state so the PR looks reviewable: no prior dispatch claim, not
    escalated, issue in ``reviewing``."""
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"][str(issue_number)] = {
            "number": issue_number,
            "status": "reviewing",
        }
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
        }
        save_state(paths.state_file, state)


# --- Fix #1: record_review refuses on a terminal-state PR -------------------


def test_record_review_refuses_on_merged_pr_preserving_decision_file(
    tmp_path: Path,
) -> None:
    """Issue #1131 fix #1: a MERGED PR must not have its decision file
    overwritten. A late ``request_changes`` arriving after the merge must be
    refused so the approved verdict that authorized the merge survives as
    the durable audit record."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_number = 456
    issue_number = 123
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True)
    decision_path = pr_dir / "review-decision.json"
    # The merge-authorizing approved verdict.
    approved_decision = {
        "pr_number": pr_number,
        "issue_number": issue_number,
        "decision": "approved",
        "summary": "lgtm",
        "required_changes": [],
        "reviewed_head_sha": "sha-abc123",
    }
    decision_path.write_text(json.dumps(approved_decision), encoding="utf-8")

    # The PR has now been merged.
    fake_gh.prs[0]["state"] = "MERGED"

    # A late request_changes from a reviewer whose dispatch interleaved with
    # the operator merge.
    result = app.record_review(
        pr_number,
        "request_changes",
        summary="please fix the naming",
        reviewed_head="sha-abc123",
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is False
    assert result.data["terminal_state"] == "MERGED"
    # The merge-authorizing decision file is untouched.
    after = json.loads(decision_path.read_text(encoding="utf-8"))
    assert after["decision"] == "approved"
    assert after["reviewed_head_sha"] == "sha-abc123"


def test_record_review_refuses_on_closed_pr(tmp_path: Path) -> None:
    """Issue #1131 fix #1: a CLOSED (non-merged) PR is equally terminal --
    recording a verdict on it is meaningless and must not overwrite any
    prior decision file."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True)
    decision_path = pr_dir / "review-decision.json"
    decision_path.write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    fake_gh.prs[0]["state"] = "CLOSED"

    result = app.record_review(
        456,
        "request_changes",
        summary="rework needed",
        reviewed_head="sha-abc123",
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is False
    assert result.data["terminal_state"] == "CLOSED"
    assert json.loads(decision_path.read_text(encoding="utf-8"))["decision"] == "approved"


def test_record_review_records_normally_on_open_pr(tmp_path: Path) -> None:
    """Issue #1131 fix #1: the terminal-state guard must not fire on an OPEN
    PR -- normal verdict recording proceeds. Guards against an over-broad
    refusal that would break the happy path."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = "diff --git a/file b/file\n+packet diff"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Generate a packet at the live head so record_review can resolve it.
    assert app.review(456).ok is True

    result = app.record_review(
        456,
        "approved",
        summary="lgtm",
        reviewed_head="sha-abc123",
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["decision"] == "approved"


# --- Fix #2: dispatch_reviews skips an already-approved PR at claim time -----


def _stale_queue_review_queue(pr_number: int, issue_number: int, head_sha: str):
    """Return a callable mimicking ``review_queue`` that reports a stale
    ``pending`` snapshot for a PR whose decision file has *actually* already
    been recorded as approved at ``head_sha``.

    This is the exact race in issue #1131: ``review_queue`` built its
    snapshot before the operator recorded the approved verdict, so the queue
    still sees ``pending`` while the decision file on disk says ``approved``.
    The claim-time re-check reads the file fresh and must skip the PR.
    """

    def _queue(self) -> CommandResult:
        return CommandResult(
            True,
            "review queue: 1 PR(s) awaiting verdict",
            {
                "queue": [
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "packet_head_sha": head_sha,
                        "decision": "pending",
                        "reviewed_head_sha": None,
                        "mergeable": "MERGEABLE",
                        "mergeStateStatus": "CLEAN",
                    }
                ]
            },
        )

    return _queue


def test_dispatch_reviews_skips_already_approved_pr_at_claim_time(
    tmp_path: Path,
) -> None:
    """Issue #1131 fix #2: when a dispatch pass interleaves with an operator
    approve -> merge, ``review_queue``'s snapshot is stale (it still sees the
    PR as ``pending``). The claim-time re-check reads the decision file fresh
    and skips any PR that already records an approval for the head this pass
    would review -- no redundant paid reviewer is launched, and no late
    ``request_changes`` can clobber the merge-authorizing decision file."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_number = 456
    issue_number = 123
    head_sha = "sha-abc123"
    _write_review_packet(paths, pr_number, head_sha)
    _seed_reviewing_pr(paths, pr_number, issue_number)
    # The operator recorded an approved verdict AFTER review_queue's snapshot
    # was built -- the file on disk now says approved at the live head.
    _write_decision(
        paths,
        pr_number,
        {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "decision": "approved",
            "summary": "lgtm",
            "required_changes": [],
            "reviewed_head_sha": head_sha,
        },
    )

    # Simulate the stale review_queue snapshot (still sees ``pending``).
    app.review_queue = _stale_queue_review_queue(pr_number, issue_number, head_sha).__get__(  # type: ignore[method-assign]
        app
    )

    result = app.dispatch_reviews()

    assert result.ok is True
    # The PR must be reported as skipped for already-approved.
    assert result.data["skipped_already_approved"] == [pr_number]
    # No claim was written -- the PR is not pending/dispatched.
    state = load_state(paths.state_file)
    pr_state = state["prs"].get(str(pr_number), {})
    assert pr_state.get("review_dispatch_status") != "review_dispatch_pending"
    assert pr_state.get("review_dispatch_status") != "review_dispatch_dispatched"
    # The attempt counter must not advance (an already-approved PR is not a
    # review attempt, mirroring the empty-diff gate).
    assert pr_state.get("review_dispatch_attempt_count", 0) == 0
    # An event must be emitted for observability.
    skip_events = [
        e
        for e in state.get("events", [])
        if e["kind"] == "review_dispatch_skipped_already_approved"
    ]
    assert len(skip_events) == 1
    assert skip_events[0]["payload"]["pr_number"] == pr_number
    assert skip_events[0]["payload"]["reviewed_head_sha"] == head_sha


def test_dispatch_reviews_dispatches_pending_pr_without_approved_decision(
    tmp_path: Path,
) -> None:
    """Issue #1131 fix #2: a PR with no recorded approval must NOT be skipped
    by the already-approved gate -- it proceeds to normal claim. Guards
    against an over-broad skip that would starve the review lane."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_number = 456
    issue_number = 123
    head_sha = "sha-abc123"
    _write_review_packet(paths, pr_number, head_sha)
    _seed_reviewing_pr(paths, pr_number, issue_number)
    # No decision file -- genuinely pending.

    app.review_queue = _stale_queue_review_queue(pr_number, issue_number, head_sha).__get__(  # type: ignore[method-assign]
        app
    )

    result = app.dispatch_reviews()

    # The launch itself fails in the test env (no real reviewer binary);
    # that is unrelated to the already-approved gate, which is what this
    # test asserts about -- the PR must NOT have been skipped by it.
    assert result.data.get("skipped_already_approved", []) == []
    # The PR must have been selected for normal dispatch (proceeded past the
    # already-approved gate to a real claim).
    assert result.data["selected_count"] >= 1
    state = load_state(paths.state_file)
    claim_events = [e for e in state.get("events", []) if e["kind"] == "review_dispatch_claim"]
    assert any(456 in c["payload"]["pr_numbers"] for c in claim_events)


def test_dispatch_reviews_dry_run_skips_already_approved_pr(
    tmp_path: Path,
) -> None:
    """Issue #1131 fix #2 (dry-run mirror): the already-approved pre-claim
    gate added to ``dispatch_reviews`` must be reflected in the dry-run
    preview so the two paths cannot diverge. A dry-run pass against a PR
    whose decision file already records an approval for the head this pass
    would review must report that PR under ``skipped_already_approved`` and
    exclude it from the dry-run selected/dispatch count -- and, because the
    dry-run branch is read-only, must not emit the
    ``review_dispatch_skipped_already_approved`` event the real path emits."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    pr_number = 456
    issue_number = 123
    head_sha = "sha-abc123"
    _write_review_packet(paths, pr_number, head_sha)
    _seed_reviewing_pr(paths, pr_number, issue_number)
    # The operator recorded an approved verdict AFTER review_queue's snapshot
    # was built -- the file on disk now says approved at the live head.
    _write_decision(
        paths,
        pr_number,
        {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "decision": "approved",
            "summary": "lgtm",
            "required_changes": [],
            "reviewed_head_sha": head_sha,
        },
    )

    # Simulate the stale review_queue snapshot (still sees ``pending``).
    app.review_queue = _stale_queue_review_queue(pr_number, issue_number, head_sha).__get__(  # type: ignore[method-assign]
        app
    )

    result = app.dispatch_reviews()

    assert result.ok is True
    # The PR must be reported under the dry-run skip field.
    assert result.data["skipped_already_approved"] == [pr_number]
    # The PR must be excluded from the dry-run selected/dispatch count.
    assert result.data["selected_count"] == 0
    assert result.data["attempted_count"] == 0
    # The dry-run branch is read-only: no state mutation, no event emitted
    # (the real path emits ``review_dispatch_skipped_already_approved``;
    # the dry-run mirror must not, per the gate's "no events emitted, no
    # state mutation" contract).
    state = load_state(paths.state_file)
    pr_state = state["prs"].get(str(pr_number), {})
    assert pr_state.get("review_dispatch_status") != "review_dispatch_pending"
    assert pr_state.get("review_dispatch_status") != "review_dispatch_dispatched"
    assert pr_state.get("review_dispatch_attempt_count", 0) == 0
    skip_events = [
        e
        for e in state.get("events", [])
        if e["kind"] == "review_dispatch_skipped_already_approved"
    ]
    assert skip_events == []


# --- Fix #3: record_review never labels a CLOSED issue agent:needs-rework ---


def test_record_review_does_not_label_closed_issue_needs_rework(
    tmp_path: Path,
) -> None:
    """Issue #1131 fix #3: a ``request_changes`` verdict whose linked issue
    is already CLOSED must not apply ``agent:needs-rework`` to it. The PR is
    still open (fix #1 only refuses on a merged/closed PR), but the issue was
    closed independently -- routing rework onto a closed issue pollutes
    roll-call/metrics and requires a manual label strip."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = "diff --git a/file b/file\n+packet diff"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # The linked issue is already CLOSED on GitHub.
    fake_gh.issues[0]["state"] = "CLOSED"

    # Generate a packet at the live head so record_review can resolve it.
    assert app.review(456).ok is True

    result = app.record_review(
        456,
        "request_changes",
        summary="please fix the naming",
        reviewed_head="sha-abc123",
        verdict_provenance="fresh_llm_review",
    )

    # The verdict itself is recorded (the PR is open, so fix #1 does not
    # refuse) -- only the rework LABEL is suppressed for the closed issue.
    assert result.ok is True
    # The needs_rework label must NOT have been applied to the closed issue.
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    # An observability event must be emitted.
    state = load_state(paths.state_file)
    skip_events = [
        e for e in state.get("events", []) if e["kind"] == "rework_label_skipped_issue_closed"
    ]
    assert len(skip_events) == 1
    assert skip_events[0]["payload"]["issue_number"] == 123


def test_record_review_labels_open_issue_needs_rework(tmp_path: Path) -> None:
    """Issue #1131 fix #3: an OPEN issue still receives ``agent:needs-rework``
    on a ``request_changes`` verdict. Guards against an over-broad skip that
    would break normal rework routing."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = "diff --git a/file b/file\n+packet diff"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Issue is OPEN (the default).
    assert app.review(456).ok is True

    result = app.record_review(
        456,
        "request_changes",
        summary="please fix the naming",
        reviewed_head="sha-abc123",
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    assert (123, config.labels.needs_rework) in fake_gh.labels_added
