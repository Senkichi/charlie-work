"""Regression tests for stranded-verdict reconciliation (issue #736).

Issue #736: a completed review verdict can sit on disk in
``review-decision.json`` but never be ingested into the PR's state record.
The sole ingestion path that reads an existing ``review-decision.json``
(``_reap_review_verdicts``) requires a live reviewer sidecar, and the
stale-claim recovery branch in ``_detect_and_handle_stalled_reviews``
explicitly *skips* PRs whose on-disk verdict is already completed (issue
#734's ``decision_already_recorded`` skip). With ``review_dispatch.enabled``
disabled fleet-wide, the PR is stranded in ``status=reviewing`` forever.

``_reconcile_stranded_verdicts`` closes the gap by scanning non-terminal PRs
for on-disk verdicts not reflected in ``state.decision`` and ingesting them
via ``record_review``. It runs in ``dispatch_reviews`` ahead of the
``review_dispatch.enabled`` gate (issue #868), so it executes even when
dispatch is disabled.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from charlie_work.instrumentation import close_db, query_events
from charlie_work.state import load_state, save_state, state_lock

from test_charlie_work import (
    _dispatch_reviews_app,
    _write_review_packet,
)


def _seed_stranded_pr(
    app: Any,
    tmp_path: Path,
    pr_number: int,
    issue_number: int,
    *,
    head_sha: str = "sha-100",
    decision: dict[str, Any] | None = None,
    pr_status: str = "reviewing",
    state_decision: str | None = None,
) -> Path:
    """Seed a PR with a review packet and a completed on-disk verdict that is
    NOT reflected in ``state.decision`` (the #736 stranded-verdict shape).

    Returns the PR directory path.
    """
    pr_dir = _write_review_packet(tmp_path, pr_number, head_sha)
    decision_path = pr_dir / "review-decision.json"
    if decision is not None:
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        pr_entry: dict[str, Any] = {
            "number": pr_number,
            "issue_number": issue_number,
            "status": pr_status,
            "prompt_path": str(pr_dir / "review-prompt.md"),
            "decision_path": str(decision_path),
        }
        if state_decision is not None:
            pr_entry["decision"] = state_decision
        state["prs"][str(pr_number)] = pr_entry
        save_state(app.paths.state_file, state)
    return pr_dir


def test_stranded_request_changes_verdict_ingested_when_dispatch_disabled(
    tmp_path: Path,
) -> None:
    """Issue #736: a completed ``request_changes`` verdict on disk but absent
    from state is ingested by ``_reconcile_stranded_verdicts`` even when
    ``review_dispatch.enabled`` is False (the fleet-wide disabled condition
    that stranded PR 1343)."""
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
    app = _dispatch_reviews_app(tmp_path, prs=prs, enabled=False)

    verdict = {
        "decision": "request_changes",
        "issue_number": 10,
        "pr_number": 100,
        "reviewed_at": (datetime.now(UTC) - timedelta(days=7)).isoformat(),
        "reviewed_head_sha": "sha-100",
        "summary": "Address issue #10 acceptance criterion #2",
        "required_changes": ["Fix the salary_floor rows"],
    }
    _seed_stranded_pr(app, tmp_path, 100, 10, decision=verdict)

    result = app.dispatch_reviews()

    # dispatch is disabled, so the gate returns early — but the reconciliation
    # ran above the gate and ingested the stranded verdict.
    assert result.ok is True
    assert result.data["disabled"] is True
    assert len(result.data["reconciled_verdicts"]) == 1
    assert result.data["reconciled_verdicts"][0]["pr_number"] == 100
    assert result.data["reconciled_verdicts"][0]["decision"] == "request_changes"
    assert result.data["reconciled_verdicts"][0]["ok"] is True

    state = load_state(app.paths.state_file)
    # The verdict was ingested into the PR state record.
    pr_state = state["prs"]["100"]
    assert pr_state["decision"] == "request_changes"
    assert pr_state["status"] == "request_changes"
    assert pr_state["review_dispatch_status"] == "review_dispatch_completed"
    # The issue was moved to rework_requested so dispatch_rework can select it.
    assert state["issues"]["10"]["status"] == "rework_requested"

    # A reconciliation event was emitted to events.db.
    close_db(app.paths.state_file)
    events = query_events(app.paths.state_file, kind="review_verdict_reconciled")
    assert len(events) == 1
    assert events[0]["payload"]["pr_number"] == 100
    assert events[0]["payload"]["decision"] == "request_changes"


def test_stranded_verdict_not_reingested_when_state_already_has_decision(
    tmp_path: Path,
) -> None:
    """Issue #736 control: a PR whose ``state.decision`` already matches the
    on-disk verdict (the normal case — PRs 1443 and 1535 from the issue's
    census) must NOT be re-reconciled. Ingestion already happened."""
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
    app = _dispatch_reviews_app(tmp_path, prs=prs, enabled=False)

    verdict = {
        "decision": "request_changes",
        "issue_number": 10,
        "pr_number": 100,
        "reviewed_at": (datetime.now(UTC) - timedelta(days=7)).isoformat(),
        "reviewed_head_sha": "sha-100",
        "summary": "fix A",
        "required_changes": ["fix A"],
    }
    # state.decision is already set — the verdict was already ingested.
    _seed_stranded_pr(app, tmp_path, 100, 10, decision=verdict, state_decision="request_changes")

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["reconciled_verdicts"] == []

    close_db(app.paths.state_file)
    events = query_events(app.paths.state_file, kind="review_verdict_reconciled")
    assert len(events) == 0


def test_pending_on_disk_verdict_not_reconciled(tmp_path: Path) -> None:
    """Issue #736: a ``pending`` on-disk verdict is normal in-flight (16 of 17
    reviewing PRs in the issue's census). Reconciliation must not touch it —
    the reviewer has not yet rendered a verdict."""
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
    app = _dispatch_reviews_app(tmp_path, prs=prs, enabled=False)

    verdict = {
        "decision": "pending",
        "issue_number": 10,
        "pr_number": 100,
        "reviewed_head_sha": "sha-100",
    }
    _seed_stranded_pr(app, tmp_path, 100, 10, decision=verdict)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["reconciled_verdicts"] == []

    state = load_state(app.paths.state_file)
    # PR state is unchanged — no decision key was added.
    assert "decision" not in state["prs"]["100"]
    assert state["prs"]["100"]["status"] == "reviewing"


def test_terminal_pr_not_reconciled(tmp_path: Path) -> None:
    """Issue #736: a merged or closed PR is lifecycle-terminal and needs no
    verdict ingestion. Reconciliation must skip it."""
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
    app = _dispatch_reviews_app(tmp_path, prs=prs, enabled=False)

    verdict = {
        "decision": "approved",
        "issue_number": 10,
        "pr_number": 100,
        "reviewed_head_sha": "sha-100",
        "summary": "looks good",
    }
    _seed_stranded_pr(app, tmp_path, 100, 10, decision=verdict, pr_status="merged")

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["reconciled_verdicts"] == []

    state = load_state(app.paths.state_file)
    assert "decision" not in state["prs"]["100"]
    assert state["prs"]["100"]["status"] == "merged"


def test_stranded_verdict_ingested_when_dispatch_enabled(
    tmp_path: Path,
) -> None:
    """Issue #736: reconciliation also runs when dispatch is enabled — the
    stranded verdict is ingested before the stale-claim sweep, so the
    ``decision_already_recorded`` skip (issue #734) never fires for a PR this
    reconciliation just ingested."""
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
    app = _dispatch_reviews_app(tmp_path, prs=prs, enabled=True)

    verdict = {
        "decision": "request_changes",
        "issue_number": 10,
        "pr_number": 100,
        "reviewed_at": (datetime.now(UTC) - timedelta(days=7)).isoformat(),
        "reviewed_head_sha": "sha-100",
        "summary": "fix A",
        "required_changes": ["fix A"],
    }
    _seed_stranded_pr(app, tmp_path, 100, 10, decision=verdict)

    result = app.dispatch_reviews()

    assert result.ok is True
    # The stranded verdict was reconciled.
    assert len(result.data["reconciled_verdicts"]) == 1
    assert result.data["reconciled_verdicts"][0]["ok"] is True

    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["decision"] == "request_changes"
    assert state["prs"]["100"]["status"] == "request_changes"
    assert state["issues"]["10"]["status"] == "rework_requested"

    # The stale-claim sweep's decision_already_recorded skip must NOT have
    # fired — the reconciliation ran first and moved the PR to
    # review_dispatch_completed, so the stale-claim branch (which only
    # matches review_dispatch_status is None) no longer applies.
    close_db(app.paths.state_file)
    skip_events = query_events(app.paths.state_file, kind="review_stale_claim_recovery_skipped")
    decision_skips = [
        e for e in skip_events if e["payload"].get("reason") == "decision_already_recorded"
    ]
    assert len(decision_skips) == 0


def test_stranded_verdict_not_ingested_when_live_head_advanced(tmp_path: Path) -> None:
    """Issue #736 review finding: when the on-disk verdict's
    ``reviewed_head_sha`` matches the review packet on disk, but the PR's live
    ``headRefOid`` has since advanced (new commit pushed while status stayed
    'reviewing'), the verdict is legitimately stale and must NOT be ingested.

    ``record_review``'s #467/#1072 guard refuses to pin a verdict to a
    superseded head for automated callers (``allow_stale_head`` defaults to
    ``False``). ``_reconcile_stranded_verdicts`` is an automated caller and
    must not bypass that guard — doing so would finalize a decision against a
    diff nobody re-reviewed. The stranded verdict is correctly skipped this
    pass and left for the stale-claim sweep / a fresh review dispatch.
    """
    # The live PR head has advanced to sha-200 (new commit pushed while the
    # PR sat in 'reviewing'). The review packet and on-disk verdict were
    # written against sha-100.
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-200",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs, enabled=False)

    verdict = {
        "decision": "request_changes",
        "issue_number": 10,
        "pr_number": 100,
        "reviewed_at": (datetime.now(UTC) - timedelta(days=7)).isoformat(),
        "reviewed_head_sha": "sha-100",
        "summary": "fix A",
        "required_changes": ["fix A"],
    }
    # _seed_stranded_pr writes the review packet with head_sha="sha-100"
    # (matching the verdict's reviewed_head_sha), while the live PR head
    # is "sha-200" — the head-drift shape.
    _seed_stranded_pr(app, tmp_path, 100, 10, head_sha="sha-100", decision=verdict)

    result = app.dispatch_reviews()

    assert result.ok is True
    # The reconciliation attempted the verdict but record_review refused
    # because the live head has moved past the packet head.
    assert len(result.data["reconciled_verdicts"]) == 1
    assert result.data["reconciled_verdicts"][0]["pr_number"] == 100
    assert result.data["reconciled_verdicts"][0]["ok"] is False

    state = load_state(app.paths.state_file)
    # The verdict was NOT ingested — state.decision is absent and the PR
    # stays in 'reviewing'.
    assert "decision" not in state["prs"]["100"]
    assert state["prs"]["100"]["status"] == "reviewing"

    # A reconcile-failed event was emitted.
    close_db(app.paths.state_file)
    failed_events = query_events(app.paths.state_file, kind="review_verdict_reconcile_failed")
    assert len(failed_events) == 1
    assert failed_events[0]["payload"]["pr_number"] == 100
    assert failed_events[0]["payload"]["decision"] == "request_changes"
