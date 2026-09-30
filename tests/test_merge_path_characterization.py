"""Characterization of ``merge_ready`` end-to-end outcomes (live and dry-run).

Pins CURRENT behaviour -- not desired behaviour -- for every decision path in
``OrchestratorApp.merge_ready`` and its hand-copied read-only twin
``_merge_ready_dry_run``, BEFORE the merge-path decision is extracted into a
pure module.  Every assertion here passes against the pre-refactor tree; a
failure after the refactor means an outcome moved and the move must be either
reverted or acknowledged by editing the pin in the same commit with a stated
reason.

Shape of the pins, per scenario:

* the live result (``ok``, message, ``data`` keys that identify the outcome)
  plus the GitHub-side effects (labels, merges, branch updates, comments);
* the persisted state (PR / issue status, counters, event kinds);
* the dry-run result for the same facts, which must also leave ``state.json``
  byte-for-byte unchanged and call nothing mutating on GitHub.

The ``DIVERGENCE`` tests make the known live-vs-dry-run drift explicit so the
refactor can remove it deliberately instead of by accident.  Key-set
snapshots (``_*_KEYS``) pin the exact ``data`` contract of each return shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


from _fakes_github import FakeGitHub
from charlie_work.config import (
    DispatchConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
)
from charlie_work.github import GitHubError
from charlie_work.workflow import OrchestratorApp

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _merge_path_characterization_harness import (
    _DRY_FINAL_KEYS,
    _ISSUE,
    _LIVE_FINAL_KEYS,
    _PR,
    _REQUIRED,
    _assert_dry_run_inert,
    _cfg,
    _pair,
    _run,
    _seed_issue_entry,
    _seed_pr_entry,
)


# ===========================================================================
# Short-circuits: already merged, not found
# ===========================================================================


def test_already_merged_short_circuit_live_converges_issue_dry_run_does_not(
    tmp_path: Path,
) -> None:
    def seed(_app: OrchestratorApp, paths: Any, _gh: FakeGitHub) -> None:
        _seed_pr_entry(paths, status="merged", merged=True)
        _seed_issue_entry(paths, status="approved", merge_alert="DEGRADED")

    pair = _pair(tmp_path, _cfg(), approve=False, pre=seed)

    live, dry = pair.live, pair.dry
    assert live.result.ok is True
    assert live.result.message == f"PR #{_PR} already merged"
    assert set(live.data) == {"pr", "issue", "already_merged", "merged"}
    assert live.data == {"pr": _PR, "issue": _ISSUE, "already_merged": True, "merged": True}
    # Live path converges the half-finalized issue record (#1493); no merge call.
    assert live.issue_entry["status"] == "closed"
    assert live.gh.merged == []

    assert dry.result.ok is True
    assert dry.result.message == f"PR #{_PR} already merged"
    assert dry.data == {
        "pr": _PR,
        "issue": _ISSUE,
        "already_merged": True,
        "merged": True,
        "dry_run": True,
    }
    assert dry.issue_entry["status"] == "approved"
    assert dry.issue_entry["merge_alert"] == "DEGRADED"
    _assert_dry_run_inert(dry)


class _NoPrGitHub(FakeGitHub):
    def pr_view(self, number: int):  # type: ignore[override]
        return None


def test_pr_not_found_returns_empty_data_both_paths(tmp_path: Path) -> None:
    pair = _pair(tmp_path, _cfg(), _NoPrGitHub, approve=False)

    for out in (pair.live, pair.dry):
        assert out.result.ok is False
        assert out.result.message == f"PR #{_PR} was not found"
        # Neither path carries a dry_run marker on this shape.
        assert out.data == {}
        assert out.state_after == out.state_before
        assert out.gh.merged == []


# ===========================================================================
# Verdict: unapproved, head moved
# ===========================================================================


def test_unapproved_pr_cannot_merge_and_writes_nothing_to_the_pr_entry_in_dry_run(
    tmp_path: Path,
) -> None:
    pair = _pair(tmp_path, _cfg(required=_REQUIRED), approve=False)

    live, dry = pair.live, pair.dry
    for out in (live, dry):
        assert out.result.ok is True
        assert out.data["can_merge"] is False
        assert out.data["merged"] is False
        assert out.data["approved"] is False
        assert out.data["summary_ready"] is True
        assert out.data["require_approved_review"] is True
        assert out.data["sync_failed"] is False
        assert out.gh.merged == []
        assert out.gh.pr_labels_added == []
    assert set(live.data) == _LIVE_FINAL_KEYS
    assert set(dry.data) == _DRY_FINAL_KEYS
    assert dry.data["dry_run"] is True
    # Live records a failed attempt only when approved; unapproved is not one.
    assert live.pr_entry.get("consecutive_failed_merge_attempts", 0) == 0
    assert "merge_ready" in live.new_event_kinds()
    _assert_dry_run_inert(dry)


def _move_head(_app: OrchestratorApp, _paths: Any, gh: FakeGitHub) -> None:
    gh.prs[0] = {**gh.prs[0], "headRefOid": "sha-new-head"}
    gh.pr_head_shas[_PR] = "sha-new-head"


_HEAD_MOVED_MSG = "PR head moved since approval — re-review required"


def test_head_moved_no_carry_forward_requests_re_review_live_and_dry(tmp_path: Path) -> None:
    config = _cfg(review_dispatch=ReviewDispatchConfig(enabled=True))
    pair = _pair(tmp_path, config, pre=_move_head)

    live, dry = pair.live, pair.dry
    assert live.result.ok is False
    assert live.result.message == _HEAD_MOVED_MSG
    assert set(live.data) == {
        "pr",
        "issue",
        "can_merge",
        "merged",
        "head_moved",
        "reviewed_head_sha",
        "live_head_sha",
        "review_decision",
        "label_error",
        "escalated",
    }
    assert live.data["can_merge"] is False
    assert live.data["merged"] is False
    assert live.data["head_moved"] is True
    assert live.data["reviewed_head_sha"] == "sha-abc123"
    assert live.data["live_head_sha"] == "sha-new-head"
    assert live.data["escalated"] is False
    assert (_ISSUE, "agent:reviewing") in live.gh.labels_added
    assert live.pr_entry["status"] == "reviewing"
    assert live.pr_entry["head_moved"] is True
    assert "head_moved" in live.new_event_kinds()
    assert live.gh.merged == []

    assert dry.result.ok is False
    assert dry.result.message == _HEAD_MOVED_MSG
    assert set(dry.data) == {
        "pr",
        "issue",
        "can_merge",
        "merged",
        "head_moved",
        "reviewed_head_sha",
        "live_head_sha",
        "review_decision",
        "dry_run",
    }
    assert dry.data["head_moved"] is True
    assert dry.data["dry_run"] is True
    _assert_dry_run_inert(dry)


def test_head_moved_with_dispatch_disabled_does_not_stamp_reviewing(tmp_path: Path) -> None:
    pair = _pair(tmp_path, _cfg(), pre=_move_head)

    live = pair.live
    assert live.result.ok is False
    assert live.data["head_moved"] is True
    assert (_ISSUE, "agent:reviewing") not in live.gh.labels_added
    assert live.pr_entry["status"] != "reviewing"
    assert live.pr_entry["head_moved"] is True
    assert "head_moved" in live.new_event_kinds()
    _assert_dry_run_inert(pair.dry)
    assert pair.dry.result.ok is False


def test_head_moved_escalated_issue_skips_reviewing_stamp_and_reports_escalated(
    tmp_path: Path,
) -> None:
    def pre(app: OrchestratorApp, paths: Any, gh: FakeGitHub) -> None:
        _move_head(app, paths, gh)
        _seed_issue_entry(paths, status="escalated", escalation_reason="redispatch_cap_exceeded")

    config = _cfg(review_dispatch=ReviewDispatchConfig(enabled=True))
    live = _run(tmp_path / "live", config, FakeGitHub, dry_run=False, pre=pre).result

    assert live.ok is False
    assert live.data["escalated"] is True
    assert live.data["head_moved"] is True


_CARRY_DIFF = (
    "diff --git a/file b/file\n"
    "index 123..456 100644\n"
    "--- a/file\n"
    "+++ b/file\n"
    "@@ -1,3 +1,4 @@\n"
    " line1\n"
    " line2\n"
    "+line3\n"
    " line4\n"
)


def test_head_moved_with_identical_patch_carries_verdict_forward_then_merges_live(
    tmp_path: Path,
) -> None:
    def gh_factory() -> FakeGitHub:
        gh = FakeGitHub()
        gh.diffs[_PR] = _CARRY_DIFF
        return gh

    def pre(_app: OrchestratorApp, _paths: Any, gh: FakeGitHub) -> None:
        gh.prs[0] = {**gh.prs[0], "headRefOid": "sha-rebased"}
        gh.pr_head_shas[_PR] = "sha-rebased"

    pair = _pair(tmp_path, _cfg(), gh_factory, pre=pre)

    live, dry = pair.live, pair.dry
    assert live.result.ok is True
    assert live.data["merged"] is True
    assert live.data["approved"] is True
    assert live.gh.merged == [(_PR, "squash")]
    assert live.pr_entry["status"] == "merged"
    # The decision now records the carried-forward head.
    assert live.data["review_decision"]["reviewed_head_sha"] == "sha-rebased"

    # Dry-run evaluates the same carry-forward but persists nothing.
    assert dry.result.ok is True
    assert dry.data["merged"] is False
    assert dry.data["can_merge"] is True
    assert "would merge" in dry.result.message
    _assert_dry_run_inert(dry)


# ===========================================================================
# The happy paths: self-merge and mergequeue hand-off
# ===========================================================================


def test_self_merge_live_merges_and_finalizes_dry_run_previews(tmp_path: Path) -> None:
    pair = _pair(tmp_path, _cfg())

    live, dry = pair.live, pair.dry
    assert live.result.ok is True
    assert set(live.data) == _LIVE_FINAL_KEYS
    assert live.data["can_merge"] is True
    assert live.data["merged"] is True
    assert live.data["merge_output"] == "merged"
    assert live.data["mergequeue_label_applied"] is None
    assert live.data["consecutive_failed_merge_attempts"] == 0
    assert live.data["merge_attempt_alarm"] is False
    assert live.data["merge_attempt_warning"] is None
    assert live.data["label_error"] is None
    assert live.data["human_merge_label_error"] is None
    assert live.gh.merged == [(_PR, "squash")]
    assert live.gh.closed_issues == [_ISSUE]
    assert live.gh.deleted_branches == ["agent/issue-123-fix-search"]
    assert live.gh.pr_labels_added == []
    assert live.pr_entry["status"] == "merged"
    assert live.pr_entry["consecutive_failed_merge_attempts"] == 0
    assert live.issue_entry["status"] == "closed"
    # merge_succeeded is emitted ONLY for a fleet self-merge (#502 tripwire).
    assert {"merge_ready", "merge_succeeded"} <= set(live.new_event_kinds())

    assert dry.result.ok is True
    assert dry.result.message == "dry-run: merge readiness evaluated (would merge)"
    assert set(dry.data) == _DRY_FINAL_KEYS
    assert dry.data["can_merge"] is True
    assert dry.data["merged"] is False
    assert dry.data["merge_output"] is None
    assert dry.data["branch_deleted"] is None
    assert dry.data["mergequeue_label_applied"] is None
    assert dry.data["merge_attempt_alarm"] is False
    assert dry.data["merge_attempt_warning"] is None
    assert dry.data["cross_pr_revert_routed"] is False
    assert dry.data["dry_run"] is True
    _assert_dry_run_inert(dry)


def test_self_merge_with_merge_false_evaluates_without_merging(tmp_path: Path) -> None:
    pair = _pair(tmp_path, _cfg(), merge=False)

    live = pair.live
    assert live.data["can_merge"] is True
    assert live.data["merged"] is False
    assert live.gh.merged == []
    assert live.pr_entry["status"] != "merged"
    assert live.pr_entry["consecutive_failed_merge_attempts"] == 0
    assert live.new_event_kinds().count("merge_ready") == 1
    assert "merge_succeeded" not in live.new_event_kinds()
    # merge=False overrides auto_merge.enabled: the dry-run preview does not
    # promise a merge either.
    assert "would merge" not in pair.dry.result.message
    _assert_dry_run_inert(pair.dry)


def test_mergequeue_handoff_labels_but_never_merges(tmp_path: Path) -> None:
    pair = _pair(tmp_path, _cfg(mergequeue="mergequeue"))

    live, dry = pair.live, pair.dry
    assert live.result.ok is True
    assert live.data["can_merge"] is True
    assert live.data["merged"] is False
    assert live.data["mergequeue_label_applied"] is True
    assert live.gh.pr_labels_added == [(_PR, "mergequeue")]
    assert live.gh.merged == []
    assert live.pr_entry["status"] == "mergequeue"
    assert live.pr_entry["mergequeue_head_sha"] == "sha-abc123"
    assert live.pr_entry["mergequeue_since"]
    assert live.pr_entry["consecutive_failed_merge_attempts"] == 0
    assert "merge_succeeded" not in live.new_event_kinds()

    assert dry.result.message == (
        "dry-run: merge readiness evaluated (would hand off to mergequeue label 'mergequeue')"
    )
    assert dry.data["mergequeue_label_applied"] is None
    assert dry.data["merged"] is False
    _assert_dry_run_inert(dry)


def test_mergequeue_label_apply_failure_counts_as_failed_handoff(tmp_path: Path) -> None:
    def gh_factory() -> FakeGitHub:
        gh = FakeGitHub()
        gh.add_pr_label_ok = False
        return gh

    live = _run(
        tmp_path / "live",
        _cfg(mergequeue="mergequeue"),
        gh_factory,
        dry_run=False,
    )

    assert live.data["can_merge"] is True
    assert live.data["merged"] is False
    assert live.data["mergequeue_label_applied"] is False
    assert live.data["consecutive_failed_merge_attempts"] == 1
    assert live.pr_entry["status"] != "mergequeue"
    assert live.pr_entry["consecutive_failed_merge_attempts"] == 1
    assert live.gh.merged == []
    assert live.data["merge_attempt_warning"] is None


def test_mergequeue_label_reverted_cross_pass_counts_failed_attempt(tmp_path: Path) -> None:
    """#823: prior status 'mergequeue' + label absent on the live PR is a
    silent Aviator revert: the counter climbs even though the re-apply works,
    and the counter is NOT reset by the successful re-apply."""

    def pre(_app: OrchestratorApp, paths: Any, _gh: FakeGitHub) -> None:
        _seed_pr_entry(paths, status="mergequeue", consecutive_failed_merge_attempts=0)

    live = _run(
        tmp_path / "live", _cfg(mergequeue="mergequeue"), FakeGitHub, dry_run=False, pre=pre
    )

    assert live.data["can_merge"] is True
    assert live.data["mergequeue_label_applied"] is True
    assert live.data["consecutive_failed_merge_attempts"] == 1
    assert live.gh.pr_labels_added == [(_PR, "mergequeue")]
    assert live.gh.merged == []


def test_mergequeue_self_revoked_stale_head_reapply_is_not_a_failed_attempt(
    tmp_path: Path,
) -> None:
    def pre(_app: OrchestratorApp, paths: Any, _gh: FakeGitHub) -> None:
        _seed_pr_entry(
            paths,
            status="mergequeue",
            mergequeue_revoked_reason="stale_head_pending_carry_forward",
        )

    live = _run(
        tmp_path / "live", _cfg(mergequeue="mergequeue"), FakeGitHub, dry_run=False, pre=pre
    )

    assert live.data["mergequeue_label_applied"] is True
    assert live.data["consecutive_failed_merge_attempts"] == 0
    assert live.data["merge_attempt_alarm"] is False
    assert live.pr_entry["status"] == "mergequeue"
    assert live.pr_entry.get("mergequeue_revoked_reason") is None


def test_already_queued_pr_is_not_synced_by_us(tmp_path: Path) -> None:
    """ADR-0003 #5: while truly queued (status mergequeue AND label present)
    our own ``pr_update_branch`` never fires; the freshness read still runs."""

    def gh_factory() -> FakeGitHub:
        gh = FakeGitHub()
        gh.prs[0]["labels"] = [{"name": "mergequeue"}]
        gh.prs[0]["mergeStateStatus"] = "BEHIND"
        return gh

    def pre(_app: OrchestratorApp, paths: Any, _gh: FakeGitHub) -> None:
        _seed_pr_entry(paths, status="mergequeue")

    config = _cfg(mergequeue="mergequeue", strategy="front_of_train")
    live = _run(tmp_path / "live", config, gh_factory, dry_run=False, pre=pre)

    assert live.gh.pr_update_branch_calls == []
    assert live.gh.merged == []


# ===========================================================================
# Holds: escalated, human-merge, merge-hold
# ===========================================================================


def _escalate_issue(_app: OrchestratorApp, paths: Any, _gh: FakeGitHub) -> None:
    _seed_issue_entry(paths, status="escalated", escalation_reason="redispatch_cap_exceeded")


def test_escalated_issue_holds_merge_without_touching_can_merge_or_counter(
    tmp_path: Path,
) -> None:
    pair = _pair(tmp_path, _cfg(), pre=_escalate_issue)

    live, dry = pair.live, pair.dry
    for out in (live, dry):
        assert out.data["can_merge"] is True
        assert out.data["escalated_merge_hold"] is True
        assert out.data["merged"] is False
        assert out.gh.merged == []
    assert live.data["consecutive_failed_merge_attempts"] == 0
    assert live.issue_entry["status"] == "escalated"
    assert live.pr_entry["consecutive_failed_merge_attempts"] == 0

    assert dry.result.message == (
        "dry-run: merge readiness evaluated "
        "(escalated — would hold merge while agent:human-needed is up)"
    )
    _assert_dry_run_inert(dry)


def test_escalated_pr_entry_also_holds_mergequeue_handoff(tmp_path: Path) -> None:
    def pre(_app: OrchestratorApp, paths: Any, _gh: FakeGitHub) -> None:
        _seed_pr_entry(paths, status="escalated")

    pair = _pair(tmp_path, _cfg(mergequeue="mergequeue"), pre=pre)

    for out in (pair.live, pair.dry):
        assert out.data["escalated_merge_hold"] is True
        assert out.data["mergequeue_label_applied"] is None
        assert out.gh.pr_labels_added == []


def _human_cfg(*, mergequeue: bool = True) -> OrchestratorConfig:
    return _cfg(
        mergequeue="mergequeue" if mergequeue else None,
        dispatch=DispatchConfig(human_merge_labels=("needs-design",)),
    )


def _human_label_gh() -> FakeGitHub:
    gh = FakeGitHub()
    gh.issues[0]["labels"] = [{"name": "automated-ready"}, {"name": "needs-design"}]
    return gh


def test_human_merge_label_live_escalates_issue_dry_run_only_reports(tmp_path: Path) -> None:
    pair = _pair(tmp_path, _human_cfg(), _human_label_gh)

    live, dry = pair.live, pair.dry
    for out in (live, dry):
        assert out.data["can_merge"] is True
        assert out.data["human_merge_hold"] is True
        assert out.data["human_merge_check_unavailable"] is False
        assert out.data["merged"] is False
        assert out.data["mergequeue_label_applied"] is None
        assert out.gh.merged == []
        assert out.gh.pr_labels_added == []
    # Live: hand-off to the operator queue, policy class, once-only comment.
    assert live.issue_entry["status"] == "escalated"
    assert live.issue_entry["reason_class"] == "policy"
    assert live.pr_entry["human_merge_comment_posted"] is True
    assert "human_merge_required" in live.new_event_kinds()
    assert len(live.gh.pr_comments_posted) == 1
    assert live.data["consecutive_failed_merge_attempts"] == 0

    assert dry.result.message == (
        "dry-run: merge readiness evaluated (human-merge label on issue — would not auto-merge)"
    )
    _assert_dry_run_inert(dry)


class _IssueViewRaises(FakeGitHub):
    def __init__(self) -> None:
        super().__init__()
        self.issues[0]["labels"] = [{"name": "automated-ready"}, {"name": "needs-design"}]

    def issue_view(self, number: int):  # type: ignore[override]
        raise GitHubError("simulated gh outage")


def test_human_merge_check_unavailable_fails_closed_ok_differs_live_vs_dry(
    tmp_path: Path,
) -> None:
    """DIVERGENCE (recon a.3 #4): dry-run ``ok`` includes
    human_merge_check_unavailable; the live ``ok`` does not."""
    pair = _pair(tmp_path, _human_cfg(), _IssueViewRaises)

    live, dry = pair.live, pair.dry
    for out in (live, dry):
        assert out.data["human_merge_check_unavailable"] is True
        assert out.data["human_merge_hold"] is False
        assert out.data["can_merge"] is True
        assert out.data["merged"] is False
        assert out.data["mergequeue_label_applied"] is None
        assert out.gh.merged == []
        assert out.gh.pr_labels_added == []
    assert live.result.ok is True
    assert dry.result.ok is False
    assert dry.result.message == (
        "dry-run: merge readiness evaluated "
        f"(human-merge label check unavailable for issue #{_ISSUE})"
    )
    _assert_dry_run_inert(dry)


def test_human_merge_label_removed_deescalates_policy_escalation_live(tmp_path: Path) -> None:
    def pre(_app: OrchestratorApp, paths: Any, _gh: FakeGitHub) -> None:
        _seed_issue_entry(
            paths,
            status="escalated",
            reason_class="policy",
            escalation_reason="human_merge_required",
        )

    # Label no longer on the issue; labels are configured -> de-escalate, then
    # the pass (which re-reads state) proceeds to hand off.
    live = _run(tmp_path / "live", _human_cfg(), FakeGitHub, dry_run=False, pre=pre)

    assert live.data["human_merge_hold"] is False
    assert live.data["escalated_merge_hold"] is False
    assert live.data["mergequeue_label_applied"] is True
    assert "human_merge_label_removed" in live.new_event_kinds()
    assert live.issue_entry["status"] != "escalated"


def test_merge_hold_label_on_pr_blocks_mergequeue_handoff(tmp_path: Path) -> None:
    def gh_factory() -> FakeGitHub:
        gh = FakeGitHub()
        gh.prs[0]["labels"] = [{"name": "agent:merge-hold"}]
        return gh

    pair = _pair(tmp_path, _cfg(mergequeue="mergequeue"), gh_factory)

    for out in (pair.live, pair.dry):
        assert out.data["can_merge"] is True
        assert out.data["merge_hold"] is True
        assert out.data["merge_hold_check_unavailable"] is False
        assert out.data["mergequeue_label_applied"] is None
        assert out.gh.pr_labels_added == []
        assert out.gh.merged == []
    assert pair.dry.result.message == (
        "dry-run: merge readiness evaluated "
        "(merge-hold label 'agent:merge-hold' present — would be left alone)"
    )
    assert pair.live.pr_entry.get("status") != "mergequeue"
    _assert_dry_run_inert(pair.dry)


def test_merge_hold_label_on_issue_blocks_mergequeue_handoff(tmp_path: Path) -> None:
    def gh_factory() -> FakeGitHub:
        gh = FakeGitHub()
        gh.issues[0]["labels"] = [{"name": "automated-ready"}, {"name": "agent:merge-hold"}]
        return gh

    pair = _pair(tmp_path, _cfg(mergequeue="mergequeue"), gh_factory)

    for out in (pair.live, pair.dry):
        assert out.data["merge_hold"] is True
        assert out.data["mergequeue_label_applied"] is None
        assert out.gh.pr_labels_added == []


class _MergeHoldIssueViewRaises(FakeGitHub):
    def issue_view(self, number: int):  # type: ignore[override]
        raise GitHubError("simulated gh outage")


def test_merge_hold_check_unavailable_fails_closed_and_ok_false_both_paths(
    tmp_path: Path,
) -> None:
    pair = _pair(tmp_path, _cfg(mergequeue="mergequeue"), _MergeHoldIssueViewRaises)

    for out in (pair.live, pair.dry):
        assert out.result.ok is False
        assert out.data["merge_hold"] is False
        assert out.data["merge_hold_check_unavailable"] is True
        assert out.data["mergequeue_label_applied"] is None
        assert out.gh.pr_labels_added == []
        assert out.gh.merged == []
    # The dry-run message does NOT mention the unavailable hold check (it
    # still says "would hand off"), even though ok is False: message drift.
    assert pair.dry.result.message == (
        "dry-run: merge readiness evaluated (would hand off to mergequeue label 'mergequeue')"
    )
    _assert_dry_run_inert(pair.dry)


def test_merge_hold_is_read_only_in_mergequeue_mode_on_the_live_path(tmp_path: Path) -> None:
    """DIVERGENCE (recon a.3 #3): the live path reads merge-hold only when
    ``mergequeue_label`` is set; the dry-run reads it whenever can_merge and
    should_merge.  With self-merge and a merge-hold label, live still merges;
    dry-run reports the hold."""

    def gh_factory() -> FakeGitHub:
        gh = FakeGitHub()
        gh.prs[0]["labels"] = [{"name": "agent:merge-hold"}]
        return gh

    pair = _pair(tmp_path, _cfg(), gh_factory)

    assert pair.live.data["merge_hold"] is False
    assert pair.live.data["merged"] is True
    assert pair.live.gh.merged == [(_PR, "squash")]
    assert pair.dry.data["merge_hold"] is True
    assert "would be left alone" in pair.dry.result.message
    _assert_dry_run_inert(pair.dry)
