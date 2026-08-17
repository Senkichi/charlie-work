"""Tests for issue #1497: review pipeline is blind to merge-conflicting PRs.

``review_queue()`` and ``_is_review_dispatchable()`` had no
``mergeable == CONFLICTING`` check, so a PR sitting in ``agent:reviewing``
with a current review packet and a merge conflict would be dispatched a
reviewer every pass — but the reviewer's verdict cannot merge a CONFLICTING
branch, and if the PR never reaches the merge lane (e.g. no verdict is ever
recorded), ``review()``'s own janitor-gate conflict route never fires either.
The PR sits in ``agent:reviewing`` forever with zero dispatch events.

The fix routes CONFLICTING candidates discovered by ``dispatch_reviews()``
through the same ``_route_janitor_gate_failure_to_rework`` path ``review()``
uses for its janitor-gate merge-conflict block — same ``reason``,
``attempts_key``, ``max_attempts`` cap, and escalation to
``agent:human-needed`` when the cap is exceeded.

These tests reuse ``FakeGitHub`` from test_charlie_work.py (PR #456 <->
issue #123) and the packet-seeding helper from test_fix_escalation_paths.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import (
    OrchestratorConfig,
    ReviewConfig,
    ReviewDispatchConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHub


def _write_review_packet(paths, pr_number: int, head_sha: str) -> None:
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        f'{{"number": {pr_number}, "headRefOid": "{head_sha}"}}', encoding="utf-8"
    )
    (pr_dir / "review-prompt.md").write_text("review prompt", encoding="utf-8")


def _seed_reviewing_issue(paths, pr_number: int, issue_number: int) -> None:
    """Seed state so the PR looks like it's in ``agent:reviewing`` with a
    current packet but no recorded verdict — the exact shape issue #1497
    describes: a PR that got a review packet, then drifted into conflict
    before any verdict was recorded."""
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


def _conflicting_gh() -> FakeGitHub:
    gh = FakeGitHub()
    gh.prs[0]["mergeable"] = "CONFLICTING"
    gh.prs[0]["mergeStateStatus"] = "DIRTY"
    return gh


def test_dispatch_reviews_routes_conflicting_pr_to_rework(tmp_path: Path) -> None:
    """A CONFLICTING PR in the review queue must be routed to rework, not
    dispatched to a reviewer. The routing mirrors ``review()``'s janitor-gate
    merge-conflict path: same ``reason="merge_conflict"``, same
    ``conflict_rework_attempts`` counter, same ``needs_rework`` label."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        review=ReviewConfig(max_conflict_rework_attempts=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _conflicting_gh()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_reviewing_issue(paths, 456, 123)

    result = app.dispatch_reviews()

    assert result.ok is True
    # The PR must NOT have been dispatched — no review_dispatch_pending claim.
    state = load_state(paths.state_file)
    assert state["prs"]["456"].get("review_dispatch_status") != "review_dispatch_pending"
    # The PR must have been routed to rework.
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["status"] == "rework_requested"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    # The needs_rework label must have been applied.
    assert (123, config.labels.needs_rework) in fake_gh.labels_added
    # The result payload must report the merge-conflict routing.
    assert len(result.data["merge_conflict_results"]) == 1
    assert result.data["merge_conflict_results"][0]["pr"] == 456
    assert result.data["merge_conflict_results"][0]["routed_to_rework"] is True


def test_dispatch_reviews_does_not_route_clean_pr(tmp_path: Path) -> None:
    """A non-conflicting PR in the review queue must NOT be routed to rework
    by the merge-conflict check — it should proceed through the normal
    dispatch path as before. (The launch itself may fail in the test env
    because there is no real reviewer binary; that is unrelated to the
    merge-conflict check, which is what this test asserts about.)"""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # default: mergeStateStatus=CLEAN, no mergeable
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_reviewing_issue(paths, 456, 123)

    result = app.dispatch_reviews()

    state = load_state(paths.state_file)
    # The PR must NOT have been routed to rework by the merge-conflict path.
    assert state["issues"]["123"].get("status") != "rework_requested"
    assert result.data["merge_conflict_results"] == []
    # The PR must have been selected for normal dispatch (not skipped by the
    # merge-conflict filter).
    assert result.data["selected_count"] >= 1


def test_dispatch_reviews_conflict_skipped_when_reviewer_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A CONFLICTING PR with a live in-flight reviewer must NOT be routed to
    rework — the verdict may still clear the conflict (e.g. a request_changes
    that prompts a rebase). Mirrors the attempt-cap path's live-reviewer
    protection (issue #573)."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _conflicting_gh()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    # Seed a live dispatched reviewer claim.
    from datetime import UTC, datetime

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "reviewing"}
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "reviewer_pid": 424242,
            "reviewer_process_start_time": 1.0,
        }
        save_state(paths.state_file, state)
    # issue #1283 Phase A: the merge-conflict routing branch this test
    # exercises lives inside `_select_review_dispatch_candidates`, which
    # moved to `charlie_work.dispatch_selection`. That function's own
    # `_reviewer_pid_alive(pr_state)` call resolves the name from
    # dispatch_selection.py's module globals (where the function is
    # defined), not workflow.py's -- patching only the workflow-side
    # attribute rebinds workflow.py's own facade re-export and is a no-op
    # for this call site (confirmed empirically: patching only the
    # workflow side leaves this test failing with status ==
    # "rework_requested"). Dual-patch both module paths: the
    # dispatch_selection side is what this test's merge-conflict branch
    # actually calls; the workflow side is kept too since
    # `OrchestratorApp.dispatch_reviews`'s own attempt-cap-escalation
    # branch (tests/test_fix_escalation_paths.py) still calls the
    # facade-reexported bare name in workflow.py's module globals.
    monkeypatch.setattr("charlie_work.workflow._reviewer_pid_alive", lambda *_: True)
    monkeypatch.setattr("charlie_work.dispatch_selection._reviewer_pid_alive", lambda *_: True)

    result = app.dispatch_reviews()

    assert result.ok is True
    state = load_state(paths.state_file)
    # The PR must NOT have been routed to rework while the reviewer is live.
    assert state["issues"]["123"].get("status") != "rework_requested"
    assert result.data["merge_conflict_results"] == []


def test_dispatch_reviews_conflict_skipped_when_escalated(tmp_path: Path) -> None:
    """An already-escalated CONFLICTING PR must NOT be re-routed to rework —
    the escalation gate (issue #575) runs before the merge-conflict check."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _conflicting_gh()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "escalated"}
        state["prs"]["456"] = {"number": 456, "issue_number": 123, "status": "escalated"}
        save_state(paths.state_file, state)

    result = app.dispatch_reviews()

    assert result.ok is True
    state = load_state(paths.state_file)
    # The PR must NOT have been re-routed to rework.
    assert state["issues"]["123"]["status"] == "escalated"
    assert result.data["merge_conflict_results"] == []


def test_dispatch_reviews_conflict_dry_run_no_mutation(tmp_path: Path) -> None:
    """In dry-run mode, a CONFLICTING PR must be reported in the result
    payload but NOT routed to rework — no state mutations or GitHub API
    calls (same gate as the escalation label edge)."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _conflicting_gh()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_reviewing_issue(paths, 456, 123)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert 456 in result.data["merge_conflict_routed"]
    # No state mutation in dry-run mode.
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "reviewing"
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
