"""Rework cap accounting: startup-death vs cap, unescalate re-arm, pre-review rework candidate, no-op rework repair.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8; sub-split for the PR #1750 review's 800-line cap).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import (
    AutoMergeConfig,
    DevinConfig,
    OrchestratorConfig,
    ReviewConfig,
    WorkerRoleConfig,
)
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    PASSIVE_OPEN_STATUS,
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_startup_death_does_not_consume_conflict_rework_cap(
    tmp_path: Path,
) -> None:
    """Issue #1106: a rework session that dies at CLI startup (before the
    worker's first tool action) must NOT consume the PR's no-op/conflict
    rework cap.  The cap counters should only count sessions that actually
    ran and produced no useful change.

    This test seeds a PR state with ``last_rework_was_startup_death=True``
    (the flag _reap_restore_rework_requested sets when a dead session is
    classified as a startup death) and verifies that
    _route_janitor_gate_failure_to_rework requeues without incrementing
    ``conflict_rework_attempts``.

    Mutation gate: removing the startup-death check in
    _route_janitor_gate_failure_to_rework makes this test fail (the counter
    increments to 1 instead of staying at 0).
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        ),
        review=ReviewConfig(max_conflict_rework_attempts=2),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # Seed the PR state with the startup-death flag, simulating a dead
    # rework session that was reaped by _reap_restore_rework_requested.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            **state.get("prs", {}).get("456", {}),
            "number": 456,
            "issue_number": 123,
            "last_rework_failure_kind": "launch_failed",
            "last_rework_was_startup_death": True,
        }
        # Issue must NOT be in a pending rework state, so the wrapper
        # reaches the counter-increment path (the startup-death check
        # is right before it).
        state["issues"]["123"] = {
            **state.get("issues", {}).get("123", {}),
            "number": 123,
            "status": "needs_review",
        }
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=False)
    assert result.ok is True
    assert result.data["merge_conflict"] is True

    state = load_state(paths.state_file)
    # The startup-death requeue must NOT have incremented the cap.
    assert state["prs"]["456"].get("conflict_rework_attempts", 0) == 0
    # The issue must have been routed back to rework_requested.
    assert state["issues"]["123"]["status"] == "rework_requested"
    # The startup-death flags must have been cleared.
    assert state["prs"]["456"]["last_rework_was_startup_death"] is False
    assert state["prs"]["456"]["last_rework_failure_kind"] is None


def test_startup_death_does_not_consume_no_op_rework_cap(
    tmp_path: Path,
) -> None:
    """Issue #1106: same as the conflict-rework variant, but for the no-op
    rework cap.  A startup-dead session requeued via the no-op-rework path
    must not increment ``no_op_rework_attempts``.
    """
    config = OrchestratorConfig(
        review=ReviewConfig(max_no_op_rework_attempts=2),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Force issue status to "reviewing" (the orphaned/stuck shape the
    # no-op route exists for — same as test_janitor_no_op_rework_routes_
    # to_rework in test_fix_janitor_routing.py).
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            **state.get("issues", {}).get("123", {}),
            "number": 123,
            "status": "reviewing",
        }
        # Seed the startup-death flag so the janitor gate's startup-death
        # check fires before the counter increment.
        state["prs"]["456"] = {
            **state.get("prs", {}).get("456", {}),
            "last_rework_failure_kind": "launch_failed",
            "last_rework_was_startup_death": True,
        }
        save_state(paths.state_file, state)

    # Same head, same diff as the recorded verdict: no actual content change,
    # so the janitor's no-op-rework signal fires and routes through
    # _route_janitor_gate_failure_to_rework with attempts_key=
    # "no_op_rework_attempts".
    result = app.review(456)
    assert result is not None
    assert result.ok is True
    assert result.data["routed_to_rework"] is True
    assert result.data.get("startup_death_requeue") is True

    state = load_state(paths.state_file)
    # The startup-death requeue must NOT have incremented the no-op cap.
    assert state["prs"]["456"].get("no_op_rework_attempts", 0) == 0
    # The issue must have been routed back to rework_requested.
    assert state["issues"]["123"]["status"] == "rework_requested"
    # The startup-death flags must have been cleared.
    assert state["prs"]["456"]["last_rework_was_startup_death"] is False
    assert state["prs"]["456"]["last_rework_failure_kind"] is None


def test_non_startup_death_still_consumes_conflict_rework_cap(
    tmp_path: Path,
) -> None:
    """Issue #1106 regression guard: a session that genuinely ran and died
    (NOT a startup death — e.g. ``stalled`` with a long runtime) must STILL
    consume the conflict rework cap.  The startup-death exemption must not
    be over-broad.
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        ),
        review=ReviewConfig(max_conflict_rework_attempts=2),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # Seed the PR state with a NON-startup death (stalled, but the flag
    # is False — the session ran long enough to be a genuine no-op).
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            **state.get("prs", {}).get("456", {}),
            "number": 456,
            "issue_number": 123,
            "last_rework_failure_kind": "stalled",
            "last_rework_was_startup_death": False,
        }
        state["issues"]["123"] = {
            **state.get("issues", {}).get("123", {}),
            "number": 123,
            "status": "needs_review",
        }
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=False)
    assert result.ok is True
    assert result.data["merge_conflict"] is True

    state = load_state(paths.state_file)
    # A non-startup death MUST still increment the cap.
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1


def test_is_startup_death_classification() -> None:
    """Issue #1106: unit test for the _is_startup_death classifier itself.

    ``launch_failed`` is always a startup death (the process never
    launched).  ``stalled`` is a startup death only under the threshold
    (the CLI exited before the worker did real work); a longer runtime
    means the worker genuinely ran and got stuck.  Unknown/None failure
    kinds are never startup deaths.
    """
    from charlie_work.workflow import (
        STARTUP_DEATH_THRESHOLD_SECONDS,
        _is_startup_death,
    )

    assert _is_startup_death("launch_failed", 0.0) is True
    assert _is_startup_death("launch_failed", 999.0) is True
    assert _is_startup_death("stalled", 1.0) is True
    assert _is_startup_death("stalled", float(STARTUP_DEATH_THRESHOLD_SECONDS)) is False
    assert _is_startup_death("stalled", float(STARTUP_DEATH_THRESHOLD_SECONDS) + 1) is False
    assert _is_startup_death(None, 0.0) is False
    assert _is_startup_death("worker_blocked", 0.0) is False
    assert _is_startup_death("rate_limited", 1.0) is False


def test_unescalate_clears_conflict_cap_escalation_and_merge_ready_redispatches(
    tmp_path: Path,
) -> None:
    """Issue #776 follow-up: the new same-reason guard in
    _route_janitor_gate_failure_to_rework (which refuses to re-route once
    escalation_reason == f"{attempts_key}_cap_exceeded" is already recorded)
    must not become a NEW one-way door of its own. ``charlie unescalate`` is
    the sanctioned re-arm: it clears ``escalation_reason`` on both the issue
    and PR records (``_UNESCALATE_ISSUE_RESET_FIELDS`` /
    ``_UNESCALATE_PR_RESET_FIELDS`` both list it) and zeros
    ``conflict_rework_attempts`` on the PR record, so a PR that is STILL
    conflicting after a human re-arms it gets a genuinely fresh attempts
    budget rather than being silently re-refused by the guard or picking up
    where the exhausted counter left off.
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        ),
        review=ReviewConfig(max_conflict_rework_attempts=2),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # Directly construct the "this lane's own cap already exhausted,
    # escalated" state that _route_janitor_gate_failure_to_rework's
    # cap-exceeded branch produces (workflow.py ~11862-11878), merged over
    # whatever record_review() already wrote -- mirrors the construction
    # convention test_fix_unescalate.py uses rather than re-deriving the
    # escalation via a repeated merge_ready() loop.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            **state["prs"]["456"],
            "status": "escalated",
            "escalation_reason": "conflict_rework_attempts_cap_exceeded",
            "escalation_reasons_seen": ["conflict_rework_attempts_cap_exceeded"],
            "conflict_rework_attempts": 3,
        }
        state["issues"]["123"] = {
            **state["issues"]["123"],
            "status": "escalated",
            "escalation_reason": "conflict_rework_attempts_cap_exceeded",
            "escalation_reasons_seen": ["conflict_rework_attempts_cap_exceeded"],
        }
        save_state(paths.state_file, state)

    # Sanity check: while escalated for THIS lane's own reason, the new guard
    # refuses to re-route at all (the property test C already covers directly
    # -- reconfirmed here as a precondition for what unescalate() is about to
    # undo).
    precheck = app.merge_ready(456, merge=False)
    assert precheck.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 3

    unescalate_result = app.unescalate(issue_number=123)
    assert unescalate_result.ok is True
    assert unescalate_result.data["changed"] is True

    state = load_state(paths.state_file)
    assert "escalation_reason" not in state["prs"]["456"]
    assert "escalation_reason" not in state["issues"]["123"]
    assert "conflict_rework_attempts" not in state["prs"]["456"]
    assert state["prs"]["456"]["status"] == PASSIVE_OPEN_STATUS

    # The conflict is still present (PR still CONFLICTING/DIRTY on GitHub) --
    # a fresh merge_ready() pass must dispatch rework again with a genuinely
    # fresh attempts counter, not pick up where the exhausted counter (3)
    # left off and not be silently refused by the same-reason guard (which no
    # longer matches now that escalation_reason has been cleared).
    result = app.merge_ready(456, merge=False)
    assert result.ok is True
    assert result.data["merge_conflict"] is True

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["status"] == "rework_requested"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    dispatch_events = [
        e for e in state["events"] if e["kind"] == "merge_conflict_rework_requested"
    ]
    assert len(dispatch_events) == 1


def test_is_pre_review_rework_candidate_detects_merge_conflict_and_stale_empty_checks() -> None:
    """Issue #439: the two pre-review rework predicates are detected independently."""
    from datetime import UTC, datetime, timedelta

    from charlie_work.workflow import _is_pre_review_rework_candidate

    config = OrchestratorConfig()
    now = datetime.now(UTC)

    # Merge conflict is an immediate rework trigger.
    assert _is_pre_review_rework_candidate({"mergeable": "CONFLICTING"}, config, now) == (
        True,
        "merge_conflict",
    )

    # mergeStateStatus DIRTY is also an immediate trigger with a distinct reason.
    assert _is_pre_review_rework_candidate({"mergeStateStatus": "DIRTY"}, config, now) == (
        True,
        "rework_branch_conflict",
    )

    old = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    stale_pr = {"statusCheckRollup": [], "updatedAt": old}
    assert _is_pre_review_rework_candidate(stale_pr, config, now) == (
        True,
        "stale_empty_checks",
    )

    # A fresh empty-rollup PR is not yet stale.
    fresh = (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    fresh_pr = {"statusCheckRollup": [], "updatedAt": fresh}
    assert _is_pre_review_rework_candidate(fresh_pr, config, now) == (False, "")

    # Any present check disqualifies the stale predicate.
    checks_pr = {"statusCheckRollup": [{"name": "Tests passed"}], "updatedAt": old}
    assert _is_pre_review_rework_candidate(checks_pr, config, now) == (False, "")


def test_no_op_rework_repair_brief_preserves_reviewer_summary(tmp_path: Path) -> None:
    """F3: _request_no_op_rework_repair's hardcoded no-op note must not
    displace the reviewer's findings. It routes through _route_to_rework,
    which calls the shared _write_rework_prompt -- the same single point of
    enforcement (issue #632) every other rework route uses -- so the
    reviewer's on-disk verdict is read fresh from review-decision.json,
    independent of the `summary` argument this method builds. Verified
    concretely here rather than assumed: this test fails if that separation
    is ever broken (e.g. a future edit threads the no-op text into
    review-decision.json's own `summary`, or bypasses _write_rework_prompt)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    reviewer_summary = (
        "BLOCKER - does not fix #649. The pin is a no-op over uv's resolver; "
        "the underlying version conflict is still unresolved."
    )
    decision = {
        "decision": "request_changes",
        "summary": reviewer_summary,
        "required_changes": [],
    }
    (pr_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")
    pr = fake_gh.pr_view(456)

    result = app._request_no_op_rework_repair(pr, 123, decision)

    assert result is None, f"expected a clean label transition, got {result!r}"
    brief = (pr_dir / "rework-prompt.md").read_text(encoding="utf-8")
    # The no-op operational note (the "dispatch_note") is present...
    assert "no actual content change" in brief
    # ...and the reviewer's original findings are still present alongside it,
    # not replaced by it -- defanged of the live closing keyword (issue
    # #781 outbound fix) the same way as the summary-fallback test above.
    assert "does not fix issue 649" in brief
    assert "does not fix #649" not in brief, "live closing keyword leaked into brief"


def test_no_op_rework_repair_note_survives_dispatch_rework_regeneration(tmp_path: Path) -> None:
    """Issue #887: the operational note from _route_to_rework must survive
    a dispatch_rework re-render unchanged.

    _request_no_op_rework_repair routes through _route_to_rework, which uses
    the shared _write_rework_prompt. That writer also writes the
    rework-dispatch-note.txt sidecar. dispatch_rework re-renders stale briefs
    and replays the note from the sidecar; this test extends the same no-op
    repair setup through a full dispatch_rework call and a newer-verdict
    re-render, asserting the operational note is still present in the
    regenerated brief. A future writer that bypasses _write_rework_prompt and
    leaves the sidecar absent or stale would fail this test, because the
    re-render would either drop the note (mtime-gate empty note) or replay a
    stale one."""
    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    reviewer_summary = (
        "BLOCKER - does not fix #649. The pin is a no-op over uv's resolver; "
        "the underlying version conflict is still unresolved."
    )
    decision = {
        "decision": "request_changes",
        "summary": reviewer_summary,
        "required_changes": [],
    }
    (pr_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")
    pr = fake_gh.pr_view(456)

    result = app._request_no_op_rework_repair(pr, 123, decision)
    assert result is None, f"expected a clean label transition, got {result!r}"

    brief_path = pr_dir / "rework-prompt.md"
    note_path = pr_dir / "rework-dispatch-note.txt"
    before = brief_path.read_text(encoding="utf-8")
    # The no-op operational note (the "dispatch_note") is present...
    assert "no actual content change" in before
    # ...and the reviewer's original findings are present alongside it.
    assert "does not fix issue 649" in before
    # _write_rework_prompt produced a sidecar for dispatch-time re-render.
    operational_note = (
        "The previous rework cycle produced no actual content change (the diff or head "
        "matches the last request_changes verdict). Check the branch worktree for "
        "unpushed commits and push the real fix, or explain in the PR body why no "
        "further change was needed."
    )
    assert note_path.read_text(encoding="utf-8") == operational_note

    # The operator / reviewer updates the verdict with a new finding and makes
    # it newer than the brief. This is the #632 axis that forces a re-render.
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": reviewer_summary,
                "required_changes": ["add a regression test for the empty-list case"],
            }
        ),
        encoding="utf-8",
    )
    now = time.time()
    os.utime(brief_path, (now, now))
    os.utime(note_path, (now, now))
    os.utime(pr_dir / "review-decision.json", (now + 10, now + 10))

    result = app.dispatch_rework()

    assert result.ok is True, result.message
    assert result.data["selected_count"] == 1
    after = brief_path.read_text(encoding="utf-8")
    # The regenerated brief picked up the new verdict finding.
    assert "add a regression test for the empty-list case" in after
    # The operational note from _route_to_rework was replayed from the sidecar
    # and survived the re-render.
    assert "no actual content change" in after
    # The sidecar itself is unchanged.
    assert note_path.read_text(encoding="utf-8") == operational_note
    # The regeneration is recorded with the correct axis.
    events = query_events(paths.state_file, kind="rework_brief_regenerated")
    assert len(events) == 1, events
    assert events[0]["payload"]["reason"] == "verdict_newer"
    assert events[0]["payload"]["pr_number"] == 456
