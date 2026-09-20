"""Merge-ready escalation holds and readiness-gate escalation.

Split out of ``tests/test_charlie_work.py`` (issue #1550, Track-1
wave 4/8).
"""

from __future__ import annotations

from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from _fakes_github import (
    FakeGitHub,
    FakeGitHubWithMissingRequired,
)
from _review_fixtures import _approved_automerge
from charlie_work.config import (
    AutoMergeConfig,
    OrchestratorConfig,
    ReviewConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from _merge_ready_fixtures import _mergequeue_automerge
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_merge_ready_readiness_gate_escalates_no_ci_stall(tmp_path: Path) -> None:
    """Issue #474: an approved PR with no CI check runs after the configured
    readiness timeout is routed to rework instead of waiting silently.

    Expected checks are derived from ``auto_merge.required_checks``; no check
    names are hard-coded in the assertion.
    """
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
            readiness_no_ci_minutes=15,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequired()
    # Simulate a head push well beyond the no-CI timeout.
    stale_updated = (datetime.now(UTC) - timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    fake_gh.prs[0]["updatedAt"] = stale_updated
    fake_gh.prs[0]["mergeStateStatus"] = "CLEAN"
    fake_gh.prs[0]["mergeable"] = "MERGEABLE"

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=False)

    assert result.data["can_merge"] is False
    assert result.data.get("readiness_no_ci_stall") is True
    assert result.data.get("merge_attempt_alarm") is False
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    no_ci_events = [e for e in state["events"] if e["kind"] == "readiness_no_ci_rework_requested"]
    assert len(no_ci_events) == 1
    missing = no_ci_events[0]["payload"]["missing_checks"]
    assert set(missing) == set(required)
    prompt_path = paths.prs / "pr-456" / "rework-prompt.md"
    assert prompt_path.exists()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    for check in required:
        assert check in prompt_text


def test_merge_ready_readiness_gate_escalates_dirty_pr(tmp_path: Path) -> None:
    """Issue #474: a PR reporting mergeStateStatus=DIRTY escalates to rework."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            failed_attempt_alarm=1,
        )
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
            "mergeable": "UNKNOWN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    fake_gh._record_pr_heads(fake_gh.prs)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=False)

    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data.get("merge_attempt_alarm") is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    conflict_events = [
        e for e in state["events"] if e["kind"] == "merge_conflict_rework_requested"
    ]
    assert len(conflict_events) == 1
    assert (paths.prs / "pr-456" / "rework-prompt.md").exists()


def test_merge_ready_reads_escalated_from_persisted_decision(tmp_path: Path) -> None:
    """Issue #407: merge_ready (via _review_decision and merge-train
    eligibility) must see the escalated flag from the persisted decision file.
    """
    config = OrchestratorConfig(review=ReviewConfig(max_rework_cycles=1))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # First request_changes is not escalated (count 0 -> 1).
    fake_gh.pr_head_shas[456] = "sha-1"
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )
    # Second request_changes hits the max_rework_cycles cap and escalates.
    fake_gh.pr_head_shas[456] = "sha-2"
    app.record_review(
        456, "request_changes", summary="fix B", verdict_provenance="fresh_llm_review"
    )

    result = app.merge_ready(456)
    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["review_decision"]["decision"] == "request_changes"
    assert result.data["review_decision"]["escalated"] is True


def test_merge_ready_escalated_head_moved_makes_no_label_or_status_mutations(
    tmp_path: Path,
) -> None:
    """Issue #833: merge_ready()'s head_moved branch must not keep re-asserting
    itself once the PR/issue is escalated.

    review() has always had an entry-level escalation gate (issue #384); this
    is the twin gate for merge_ready(), scoped to the specific head_moved
    mutation the reported bug is about rather than the whole function --
    merge_ready() also owns the merge-conflict-rework dispatch lane, which
    issue #776 requires to keep running while escalated for an unrelated
    reason (see test_merge_ready_conflict_escalated_for_unrelated_reason_
    routes_to_rework below), so a blanket top-of-function gate would silently
    reintroduce #776's bug while fixing this one.

    Real corpus: issue #602 / PR #679 -- once the issue was escalated
    (agent:human-needed), merge_ready() kept re-adding agent:reviewing and
    rewriting state status to "reviewing" on every ~30 min loop pass forever,
    because this branch had no escalation awareness at all. Nothing was
    waiting on the re-review it kept re-requesting, so it never resolved --
    a livelock.

    Positive control: test_merge_ready_refuses_when_head_moved_after_approval
    (immediately above) is the byte-identical scenario MINUS the escalated
    status write, and it DOES assert the label add and the status write --
    proving this test's negative assertions would fail on unpatched code.
    """
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    fake_gh.prs[0] = {**fake_gh.prs[0], "headRefOid": "sha-new-head"}
    fake_gh.pr_head_shas[456] = "sha-new-head"
    # Simulate the fleet state a livelocked issue #602/PR #679 was actually
    # in: escalated with the human-needed / reviewing labels already applied
    # from a prior pass.
    fake_gh.issues[0] = {
        **fake_gh.issues[0],
        "labels": [{"name": "agent:human-needed"}, {"name": "agent:reviewing"}],
    }
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            **state["issues"].get("123", {}),
            "status": "escalated",
            "escalation_reason": "redispatch_cap_exceeded",
        }
        save_state(paths.state_file, state)

    labels_added_before = list(fake_gh.labels_added)
    labels_removed_before = list(fake_gh.labels_removed)

    result = app.merge_ready(456, merge=True)

    assert result.ok is False
    assert "PR head moved since approval" in result.message
    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert result.data["head_moved"] is True
    assert result.data["escalated"] is True
    assert fake_gh.merged == []
    assert fake_gh.merged_merge_flags == []
    # The actual reported bug: zero new label mutations on this pass.
    assert fake_gh.labels_added == labels_added_before
    assert fake_gh.labels_removed == labels_removed_before
    assert (123, "agent:reviewing") not in fake_gh.labels_added[len(labels_added_before) :]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] != "reviewing"
    assert state["issues"]["123"]["status"] == "escalated"


def test_merge_ready_escalated_issue_blocks_merge_of_otherwise_green_pr(
    tmp_path: Path,
) -> None:
    """Issue #840: an approved, green, conflict-free PR whose linked issue is
    escalated (status == "escalated" / agent:human-needed) for a reason
    unrelated to this PR's mergeability must NOT be actually merged while
    that flag is up.

    This is the byte-identical scenario as
    test_merge_ready_merges_when_head_unchanged_after_approval (immediately
    above) PLUS an escalated linked issue — proving the escalation gate
    suppresses the merge that the positive control proves would otherwise
    fire. The escalation reason is deliberately unrelated to mergeability
    (redispatch_cap_exceeded — a dead request-changes-fix worker exhausting
    the watchdog's redispatch cap; real corpus: issues #592/#648/#606).
    """
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    # Escalate the linked issue for an unrelated reason.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            **state["issues"].get("123", {}),
            "status": "escalated",
            "escalation_reason": "redispatch_cap_exceeded",
        }
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=True)

    # The irreversible step must not have happened — assert this FIRST so a
    # mutation that removes the gate fails on the behavioral bug (the PR
    # getting merged), not merely on a missing return-data key.
    assert result.data["merged"] is False
    assert fake_gh.merged == []
    # can_merge is still True — the PR is genuinely mergeable. The gate is
    # on the merge-execution block, not on can_merge itself (see the issue
    # for why modifying can_merge would cause a counter/alarm regression).
    assert result.data["can_merge"] is True
    assert result.data["escalated_merge_hold"] is True
    # The linked issue's escalated status must be preserved.
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"


def test_merge_ready_escalated_pr_blocks_merge_of_otherwise_green_pr(
    tmp_path: Path,
) -> None:
    """Issue #840 (PR-level escalation variant): the escalation gate also
    fires when the PR's own state entry is escalated, not just the linked
    issue's."""
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            **state["prs"].get("456", {}),
            "status": "escalated",
        }
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=True)

    assert result.data["merged"] is False
    assert fake_gh.merged == []
    assert result.data["can_merge"] is True
    assert result.data["escalated_merge_hold"] is True


def test_merge_ready_escalated_issue_counter_does_not_climb(
    tmp_path: Path,
) -> None:
    """Issue #840: the failed-attempt-alarm counter must NOT spuriously climb
    for an escalated-but-otherwise-mergeable PR. Because the escalation gate
    leaves ``can_merge`` True (it gates the merge-execution block, not
    can_merge), the counter block's ``elif can_merge:`` branch zeroes the
    streak on every pass — a green pass held by escalation is not a failed
    pass. A spurious climb would eventually cross ``failed_attempt_alarm``
    and fire a ``merge_attempt_alarm`` digest for a PR that isn't failing to
    merge for a mergeability reason (the diagnostic regression the issue
    calls out as the complication with naively adding an escalation term to
    can_merge).
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            **state["issues"].get("123", {}),
            "status": "escalated",
            "escalation_reason": "redispatch_cap_exceeded",
        }
        save_state(paths.state_file, state)

    # Run enough passes to cross the alarm threshold IF the counter were
    # climbing (which it must not).
    for _ in range(5):
        result = app.merge_ready(456, merge=True)
        # Behavioral assertions first: the PR must not be merged and the
        # counter must not climb — these are the actual bugs the fix prevents.
        assert result.data["merged"] is False
        assert result.data["consecutive_failed_merge_attempts"] == 0
        assert result.data["merge_attempt_alarm"] is False
        assert result.data["merge_attempt_warning"] is None
        assert result.data["escalated_merge_hold"] is True

    assert fake_gh.merged == []


def test_merge_ready_escalated_issue_blocks_mergequeue_handoff(
    tmp_path: Path,
) -> None:
    """Issue #840 (mergequeue mode): an escalated PR must not be handed off
    to the mergequeue either — the mergequeue label add IS the handoff
    (task #10), and handing off while escalated is the same class of
    silently-completing-an-action-while-a-human-is-asked-to-intervene as a
    direct merge."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            **state["issues"].get("123", {}),
            "status": "escalated",
            "escalation_reason": "redispatch_cap_exceeded",
        }
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=True)

    # Behavioral assertions first: the handoff must not have happened.
    assert result.data["merged"] is False
    assert result.data["mergequeue_label_applied"] is None
    # The mergequeue label must not have been added.
    assert "mergequeue" not in [label for _, label in fake_gh.labels_added]
    state = load_state(paths.state_file)
    assert state["prs"]["456"].get("status") != "mergequeue"
    assert result.data["can_merge"] is True
    assert result.data["escalated_merge_hold"] is True


def test_merge_ready_dry_run_escalated_issue_reports_hold(tmp_path: Path) -> None:
    """Issue #840 (dry-run): the dry-run preview must accurately report
    "would hold" instead of "would merge" when the linked issue is
    escalated, mirroring the real path's gate."""
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            require_approved_review=True,
            enabled=False,  # dry-run never merges, but the gate must still report
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.dry_run = True

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            **state["issues"].get("123", {}),
            "status": "escalated",
            "escalation_reason": "redispatch_cap_exceeded",
        }
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=True)

    # Behavioral assertion first: the preview must not say "would merge".
    assert "would merge" not in result.message
    assert "escalated" in result.message
    assert result.data["dry_run"] is True
    assert result.data["can_merge"] is True
    assert result.data["escalated_merge_hold"] is True
