"""Merge-ready --dry-run mode.

Split out of ``tests/test_charlie_work.py`` (issue #1550, Track-1
wave 4/8).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from _fakes_github import FakeGitHub
from _review_fixtures import (
    _approved_automerge,
    _required_checks_config,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from _merge_ready_fixtures import _mergequeue_automerge
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_merge_ready_dry_run_does_not_merge_or_persist(tmp_path: Path) -> None:
    """Issue #614: under --dry-run, merge_ready must not call merge_pr, must
    not write status='merged' to state.json, and must return the computed
    readiness verdict (can_merge=True) without persisting."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    # The verdict is computed (can_merge=True) but nothing happened.
    assert result.data["can_merge"] is True
    assert result.data["merged"] is False
    assert result.data["dry_run"] is True
    assert "would merge" in result.message
    # No merge was attempted.
    assert fake_gh.merged == []
    assert fake_gh.deleted_branches == []
    # State was NOT written — no 'merged' status, no prs entry at all.
    state = load_state(paths.state_file)
    pr_state = state["prs"].get("456", {})
    assert pr_state.get("status") != "merged"
    assert pr_state.get("merged") is not True


def test_merge_ready_dry_run_mergequeue_does_not_label_or_persist(tmp_path: Path) -> None:
    """Issue #614: under --dry-run with mergequeue_label set, merge_ready must
    not call add_pr_label, must not write status='mergequeue' to state.json,
    and must return the computed readiness verdict without persisting.

    This is the highest-severity blast radius: without the gate, the dry-run
    stub's _run_bool returns True, state records status='mergequeue', and the
    PR is silently stranded — the orchestrator believes Aviator owns it but
    Aviator never received the label."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is True
    assert result.data["merged"] is False
    assert result.data["dry_run"] is True
    assert result.data["mergequeue_label_applied"] is None
    assert "would hand off to mergequeue" in result.message
    # No label was added, no merge was attempted.
    assert fake_gh.pr_labels_added == []
    assert fake_gh.merged == []
    # State was NOT written — no 'mergequeue' status.
    state = load_state(paths.state_file)
    pr_state = state["prs"].get("456", {})
    assert pr_state.get("status") != "mergequeue"
    assert pr_state.get("status") != "merged"


def test_merge_ready_dry_run_preserves_existing_state(tmp_path: Path) -> None:
    """Issue #614: under --dry-run, merge_ready must not increment
    consecutive_failed_merge_attempts or advance the state machine in any
    way.  A pre-existing PR state entry must be byte-for-byte unchanged
    after a dry-run pass."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    # Plant a pre-existing PR state entry with a non-zero counter.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "approved",
            "consecutive_failed_merge_attempts": 3,
            "consecutive_stale_base_deferrals": 0,
        }
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=True)

    assert result.data["dry_run"] is True
    # The counter is reported from the existing state, not incremented.
    assert result.data["consecutive_failed_merge_attempts"] == 3
    # State on disk is unchanged.
    persisted = load_state(paths.state_file)["prs"]["456"]
    assert persisted["consecutive_failed_merge_attempts"] == 3
    assert persisted["status"] == "approved"


def test_merge_ready_dry_run_already_merged_is_noop(tmp_path: Path) -> None:
    """Issue #614: under --dry-run, an already-merged PR returns the
    idempotency no-op without writing state (the real path's idempotency
    block conditionally clears merge_alert — a write that must be skipped)."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Plant an already-merged PR state entry with a non-OK merge_alert.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "merged",
            "merged": True,
        }
        state["issues"]["123"] = {
            "number": 123,
            "merge_alert": "DEGRADED",
        }
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=True)

    assert result.data["already_merged"] is True
    assert result.data["merged"] is True
    assert result.data["dry_run"] is True
    # The merge_alert was NOT cleared (the real path would clear it).
    persisted_issue = load_state(paths.state_file)["issues"]["123"]
    assert persisted_issue["merge_alert"] == "DEGRADED"


def test_merge_ready_dry_run_unapproved_reports_can_merge_false(tmp_path: Path) -> None:
    """Issue #614: under --dry-run, an unapproved PR reports can_merge=False
    without persisting any state.

    Issue #1060: the dry-run return payload must also surface the four gate
    inputs (``summary_ready``, ``approved``, ``require_approved_review``,
    ``sync_failed``) so the preview has diagnostic parity with the persisted
    ``merge_ready`` event.  This scenario — checks green but no recorded
    approval — exercises the ``approved=False`` / ``require_approved_review=True``
    branch that makes ``can_merge`` False.
    """
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is False
    assert result.data["merged"] is False
    assert result.data["dry_run"] is True
    # Issue #1060: the four gate inputs are spread into the dry-run payload
    # from the same dict the gate reads, so the preview explains *why*
    # can_merge is False without needing events.db.
    assert result.data["summary_ready"] is True
    assert result.data["approved"] is False
    assert result.data["require_approved_review"] is True
    assert result.data["sync_failed"] is False
    # No state was written for this PR.
    assert "456" not in load_state(paths.state_file)["prs"]


# --------------------------------------------------------------------------- #
# Issue #2743 (review round 2): the dry-run preview's requeue projection.
#
# The preview projects what decide_merge's ``requeue_gate`` would see. These
# pin the same contract the live lane's tests
# (``tests/test_charlie_work_merge_ready_requeue.py``) pin, through the
# ``--dry-run`` driver: the ledger only climbs when the pass would re-add.
# --------------------------------------------------------------------------- #

_PR = 456
_ISSUE = 123


def _mergequeue_dry_run_app(
    tmp_path: Path, cap: int = 3
) -> tuple[OrchestratorApp, FakeGitHub, object]:
    """A dry-run app under mergequeue automerge with an inline cap override."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    if cap != config.auto_merge.mergequeue_requeue_cap:
        config = replace(
            config,
            auto_merge=replace(config.auto_merge, mergequeue_requeue_cap=cap),
        )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    app.record_review(_PR, "approved", summary="ok", verdict_provenance="fresh_llm_review")
    return app, fake_gh, paths


def _seed_reverted_requeue(paths) -> None:
    """A PR parked in the queue two counted reverts under the cap of 3."""
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"][str(_PR)] = {
            **state["prs"].get(str(_PR), {}),
            "number": _PR,
            "status": "mergequeue",
            "consecutive_mergequeue_requeues": 2,
            "mergequeue_requeues_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)


def _assert_dry_run_never_climbs(app_result, paths) -> None:
    """The payload stays at the persisted count and disk state is untouched."""
    assert app_result.data["dry_run"] is True
    assert app_result.data["mergequeue_requeue_capped"] is False
    assert app_result.data["consecutive_mergequeue_requeues"] == 2
    assert "would hold and route to rework" not in app_result.message
    persisted = load_state(paths.state_file)["prs"][str(_PR)]
    assert persisted["consecutive_mergequeue_requeues"] == 2
    assert persisted["status"] == "mergequeue"


def test_merge_ready_dry_run_capped_requeue_reports_cap_and_rework_route(
    tmp_path: Path,
) -> None:
    """A dry run over a PR whose same-head revert count reaches the cap must
    report ``mergequeue_requeue_capped`` and the rework route instead of
    "would hand off", with the projected count in the payload — and since the
    preview runs no accounting, nothing may be written."""
    app, fake_gh, paths = _mergequeue_dry_run_app(tmp_path)
    _seed_reverted_requeue(paths)

    result = app.merge_ready(_PR, merge=True)

    assert result.data["dry_run"] is True
    assert "capped (3/3)" in result.message
    assert "would hold and route to rework" in result.message
    assert "would hand off" not in result.message
    assert result.data["mergequeue_requeue_capped"] is True
    assert result.data["consecutive_mergequeue_requeues"] == 3
    # Nothing was applied: no label POST and no state write (the preview
    # never enters the accounting stage), so the projected count is not
    # persisted and the cap event cannot have been recorded.
    assert fake_gh.pr_labels_added == []
    persisted = load_state(paths.state_file)["prs"][str(_PR)]
    assert persisted["consecutive_mergequeue_requeues"] == 2
    assert persisted["status"] == "mergequeue"


def test_merge_ready_dry_run_merge_hold_keeps_requeues_flat(tmp_path: Path) -> None:
    """A merge-hold PR parked in the queue observes the reverted label pass
    after pass; the dry run must project only the persisted count (no re-add
    is attempted) and never report the cap."""
    app, fake_gh, paths = _mergequeue_dry_run_app(tmp_path)
    _seed_reverted_requeue(paths)
    fake_gh.prs[0]["labels"] = [{"name": "agent:merge-hold"}]

    result = app.merge_ready(_PR, merge=True)

    assert result.data["merge_hold"] is True
    _assert_dry_run_never_climbs(result, paths)


def test_merge_ready_dry_run_should_merge_false_keeps_requeues_flat(
    tmp_path: Path,
) -> None:
    """A dry run with ``should_merge=False`` observes the reverted label but
    attempts no re-add: the projection stays at the persisted count and the
    cap never engages."""
    app, fake_gh, paths = _mergequeue_dry_run_app(tmp_path)
    _seed_reverted_requeue(paths)

    result = app.merge_ready(_PR, merge=False)

    assert result.data["auto_merge_enabled"] is True
    _assert_dry_run_never_climbs(result, paths)


def test_merge_ready_dry_run_escalated_hold_never_reports_capped(
    tmp_path: Path,
) -> None:
    """An escalated linked issue holds the pass; the dry run must mirror the
    live lane — flat projection, no cap verdict, the escalation message."""
    app, fake_gh, paths = _mergequeue_dry_run_app(tmp_path)
    _seed_reverted_requeue(paths)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"][str(_ISSUE)] = {"number": _ISSUE, "status": "escalated"}
        save_state(paths.state_file, state)

    result = app.merge_ready(_PR, merge=True)

    assert result.data["escalated_merge_hold"] is True
    assert "escalated" in result.message
    _assert_dry_run_never_climbs(result, paths)
