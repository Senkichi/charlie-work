"""Merge-ready merge-queue (mergequeue) mode and merge train.

Split out of ``tests/test_charlie_work.py`` (issue #1550, Track-1
wave 4/8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest
from _fakes_github import FakeGitHub
from _helpers import _second_mergequeue_pr
from _review_fixtures import _approved_automerge
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
)
from charlie_work.workflow import OrchestratorApp
from _merge_ready_fixtures import _mergequeue_automerge
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_merge_ready_mergequeue_mode_labels_instead_of_merging(tmp_path: Path) -> None:
    """Aviator MergeQueue handoff (task #10): when auto_merge.mergequeue_label
    is set, an approved+green PR is labeled for the queue INSTEAD of being
    self-merged, and state records a distinct 'mergequeue' status — never
    'merged', so the merge_ready idempotency short-circuit does not fire while
    Aviator's async merge is still pending."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is True
    assert (456, "mergequeue") in fake_gh.pr_labels_added
    assert fake_gh.merged == []
    assert result.data["merged"] is False
    assert result.data["mergequeue_label_applied"] is True
    persisted = load_state(paths.state_file)["prs"]["456"]
    assert persisted["status"] == "mergequeue"
    assert persisted["status"] != "merged"


def test_merge_ready_mergequeue_stamps_and_preserves_dwell_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1401: merge_ready stamps mergequeue_since/mergequeue_head_sha when
    a PR enters Aviator's queue, preserves them across passes while the head is
    frozen (so the wedge watchdog can measure true no-progress dwell), and
    resets them when the head advances (Aviator rebased -> progress, not a wedge)."""
    # utc_now() strips microseconds, so two merge_ready calls within the same
    # wall-clock second produce identical timestamps. Mock it with a per-call
    # counter so the "head advanced -> fresh dwell window" assertion is
    # deterministic, not a race against the clock.
    _utc_call = 0

    def _fake_utc_now() -> str:
        nonlocal _utc_call
        _utc_call += 1
        return f"2026-01-01T00:00:{_utc_call:02d}Z"

    monkeypatch.setattr("charlie_work.workflow.utc_now", _fake_utc_now)
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    app.merge_ready(456, merge=True)
    first = load_state(paths.state_file)["prs"]["456"]
    assert first["status"] == "mergequeue"
    assert first["mergequeue_head_sha"] == "sha-abc123"
    assert first["mergequeue_since"]
    first_since = first["mergequeue_since"]

    # Second pass, head unchanged: the queue-enter time is preserved so the
    # watchdog's dwell measurement is not reset by a no-op re-evaluation pass.
    app.merge_ready(456, merge=True)
    second = load_state(paths.state_file)["prs"]["456"]
    assert second["mergequeue_head_sha"] == "sha-abc123"
    assert second["mergequeue_since"] == first_since

    # Aviator rebases the PR -> head advances. merge_ready returns early
    # ("head moved, re-review required") without persisting; a fresh
    # record_review at the new head re-approves it, and the next merge_ready
    # stamps a fresh dwell window (progress, not a wedge).
    fake_gh.pr_head_shas[456] = "sha-abc123-rebased"
    head_moved = app.merge_ready(456, merge=True)
    assert head_moved.data["head_moved"] is True
    app.record_review(
        456,
        "approved",
        summary="ok",
        verdict_provenance="fresh_llm_review",
        reviewed_head="sha-abc123-rebased",
    )
    app.merge_ready(456, merge=True)
    third = load_state(paths.state_file)["prs"]["456"]
    assert third["mergequeue_head_sha"] == "sha-abc123-rebased"
    assert third["mergequeue_since"] != first_since


def test_merge_ready_mergequeue_hold_label_on_pr_prevents_re_add(tmp_path: Path) -> None:
    """Issue #496: an approved PR carrying the configured merge-hold label
    must not be swept back into the mergequeue on subsequent passes."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["labels"] = [{"name": config.labels.merge_hold}]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is True
    assert result.data["merge_hold"] is True
    assert result.data["mergequeue_label_applied"] is None
    assert fake_gh.pr_labels_added == []
    assert fake_gh.merged == []
    assert "left alone" in result.message
    assert load_state(paths.state_file)["prs"]["456"].get("status") != "mergequeue"


def test_merge_ready_mergequeue_hold_label_on_issue_prevents_re_add(tmp_path: Path) -> None:
    """Issue #496: the merge-hold label on the linked issue is also a
    valid operator signal and must keep the PR out of the mergequeue."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [
        {"name": config.labels.ready},
        {"name": config.labels.merge_hold},
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is True
    assert result.data["merge_hold"] is True
    assert fake_gh.pr_labels_added == []
    assert fake_gh.merged == []


def test_merge_ready_mergequeue_hold_label_removed_resumes_re_add(tmp_path: Path) -> None:
    """Issue #496: removing the merge-hold label restores normal
    auto-merge behavior on the next pass."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    hold_label = config.labels.merge_hold
    fake_gh.prs[0]["labels"] = [{"name": hold_label}]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    held = app.merge_ready(456, merge=True)
    assert held.data["merge_hold"] is True
    assert fake_gh.pr_labels_added == []

    # Operator removes the hold label
    fake_gh.prs[0]["labels"] = []
    resumed = app.merge_ready(456, merge=True)

    assert resumed.data["merge_hold"] is False
    assert resumed.data["mergequeue_label_applied"] is True
    assert (456, "mergequeue") in fake_gh.pr_labels_added


def test_merge_ready_mergequeue_hold_issue_check_unavailable_fails_closed(tmp_path: Path) -> None:
    """Issue #496 regression: if issue_view fails while checking for the
    merge-hold label on the linked issue, the PR must not be handed to the
    mergequeue. The failure is reported as merge_hold_check_unavailable and
    must not be treated as a mergequeue handoff failure (no failed-attempt
    alarm side effects)."""
    from charlie_work.github import GitHubError as _GitHubError

    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class IssueViewFailGitHub(FakeGitHub):
        def issue_view(self, number: int):
            if number == 123:
                raise _GitHubError("transient gh issue view failure")
            return super().issue_view(number)

    fake_gh = IssueViewFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is True
    assert result.data["merge_hold"] is False
    assert result.data["merge_hold_check_unavailable"] is True
    assert result.data["mergequeue_label_applied"] is None
    assert result.data["merge_attempt_alarm"] is False
    assert result.data["merge_attempt_warning"] is None
    assert result.ok is False
    assert fake_gh.pr_labels_added == []
    assert fake_gh.merged == []
    assert "merge-hold check unavailable" in result.message
    assert "not handed off to Aviator" in result.message
    pr_state = load_state(paths.state_file)["prs"]["456"]
    assert pr_state.get("status") != "mergequeue"
    assert pr_state.get("consecutive_failed_merge_attempts", 0) == 0


@pytest.mark.parametrize("degraded_payload", [{}, {"number": 123}])
def test_merge_ready_mergequeue_hold_issue_degraded_payload_fails_closed(
    tmp_path: Path,
    degraded_payload: dict[str, Any],
) -> None:
    """Issue #496 regression: a degraded gh issue view payload (empty dict or
    missing labels) must be treated as unavailable, not as "no hold"."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class IssueViewDegradedGitHub(FakeGitHub):
        def issue_view(self, number: int):
            if number == 123:
                return degraded_payload
            return super().issue_view(number)

    fake_gh = IssueViewDegradedGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is True
    assert result.data["merge_hold"] is False
    assert result.data["merge_hold_check_unavailable"] is True
    assert result.data["mergequeue_label_applied"] is None
    assert result.data["merge_attempt_alarm"] is False
    assert result.data["merge_attempt_warning"] is None
    assert result.ok is False
    assert fake_gh.pr_labels_added == []
    assert fake_gh.merged == []


def test_merge_ready_mergequeue_mode_unapproved_pr_not_labeled(tmp_path: Path) -> None:
    """An unapproved PR must never be labeled for the merge queue — the
    approval gate (can_merge) is upstream of the mergequeue branch, exactly as
    it is upstream of the self-merge branch today."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is False
    assert fake_gh.pr_labels_added == []
    assert fake_gh.merged == []
    assert result.data.get("mergequeue_label_applied") is None


def test_merge_ready_mergequeue_parked_pr_excluded_from_merge_train_head(
    tmp_path: Path,
) -> None:
    """Adversarial review finding #1a: once PR #456 is parked in Aviator's
    queue (state status 'mergequeue'), it must not keep winning
    front-of-train's merge-train head on every subsequent poll — Aviator now
    owns its serialization. Without excluding 'mergequeue'-status PRs from
    _merge_train_candidates, #456 (reviewed first, still approved, still
    head-SHA-matching) would win merge-train head forever and PR #789 would
    never be attempted, even though #789 is independently approved and green."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    _second_mergequeue_pr(fake_gh)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")
    app.record_review(789, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    # First poll: 456 was reviewed first (and sorts first on a tie), so it
    # wins merge-train head and gets parked into the mergequeue.
    first = app.merge_ready(456, merge=True)
    assert first.data["can_merge"] is True
    assert first.data["mergequeue_label_applied"] is True
    assert (456, "mergequeue") in fake_gh.pr_labels_added

    # Second poll: 789 must now become merge-train head. Before the fix, 456
    # (still "approved" + head-SHA-matching from _merge_train_candidates'
    # point of view) keeps winning head, so 789 gets bounced as "not the
    # head of the merge-train queue" (can_merge False) forever.
    second = app.merge_ready(789, merge=True)
    assert second.data["can_merge"] is True
    assert second.data["mergequeue_label_applied"] is True
    assert (789, "mergequeue") in fake_gh.pr_labels_added


def test_merge_train_candidates_no_state_read_when_mergequeue_label_unset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Issue #421: when auto_merge.mergequeue_label is None (the default),
    _merge_train_candidates must not call load_state_locked. The mergequeue
    handoff feature is disabled, so no PR can have status 'mergequeue' and the
    state read is pure hot-path overhead that widens the StateLockBusy window.
    """
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("load_state_locked called with mergequeue_label unset")

    monkeypatch.setattr("charlie_work.workflow.load_state_locked", _fail_if_called)

    candidates = app._merge_train_candidates(prs=fake_gh.prs)

    pr_numbers = [pr_number for _sort_key, pr_number, _pr, _decision, _head in candidates]
    assert 456 in pr_numbers


def test_merge_ready_mergequeue_parked_pr_skips_charlie_branch_sync(
    tmp_path: Path,
) -> None:
    """Adversarial review finding #1b: once a PR is parked in Aviator's queue
    (state status 'mergequeue'), charlie must stop calling pr_update_branch
    for it on every subsequent poll — Aviator now owns rebasing queued PRs.
    This repo's live orchestrator.config.yaml sets update_open_prs: true
    (broadcast), so without this fix charlie would race Aviator's own rebase
    as a second writer on the same ref on every poll while the base is stale."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    # First poll parks the PR.
    first = app.merge_ready(456, merge=True)
    assert first.data["mergequeue_label_applied"] is True
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "mergequeue"

    # "Parked in Aviator's queue" means the queue label is actually still on
    # the live PR — FakeGitHub.add_pr_label only logs the call, so the label
    # must be placed on the PR explicitly. Without it the next pass observes
    # a stripped label (mergequeue_label_reverted), which is the abandoned-
    # handoff case issue #1873 un-deadlocks by re-syncing — the opposite of
    # the skip this test pins.
    fake_gh.prs[0]["labels"] = [{"name": "mergequeue"}]

    # Simulate main having advanced past this PR's merge-base — a stale base,
    # exactly as would happen once Aviator (or anything else) merges another
    # PR into main while #456 sits in the queue.
    fake_gh.compare_overrides[("main", "sha-abc123")] = {
        "base_commit": {"sha": "new-main-tip"},
        "merge_base_commit": {"sha": "stale-ancestor"},
    }

    second = app.merge_ready(456, merge=True)

    assert second.data["can_merge"] is False
    assert fake_gh.pr_update_branch_calls == []


def test_merge_ready_mergequeue_reverted_handoff_syncs_stale_base(
    tmp_path: Path,
) -> None:
    """Issue #1873: an Aviator-abandoned mergequeue handoff must not deadlock
    on a stale base.

    Once a PR is handed off (state status 'mergequeue'), the base-currency
    sync is deliberately skipped so Aviator owns the rebase — but the skip
    is keyed on the persisted status, not on the label still being present.
    When Aviator silently strips the label (the #823 revert shape) while the
    base is stale, the stale-base early return fired every pass BEFORE the
    mergequeue_handoff_failed recovery could run, and the sync skip was the
    only thing that could make the base current — swole PR #321 looped
    merge_deferred_stale_base 23+ passes until an operator ran
    `gh pr update-branch` by hand.

    A reverted handoff is void: the PR is not in Aviator's queue, so charlie
    runs its own sync (pr_update_branch), the approval carries forward to
    the verified sync head, the handoff is retried this same pass, and the
    revert is still accounted as a failed handoff."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    # Pass 1: fresh handoff, succeeds. Status becomes "mergequeue".
    first = app.merge_ready(456, merge=True)
    assert first.data["mergequeue_label_applied"] is True
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "mergequeue"

    # Aviator silently bounces the PR out of its queue — the mergequeue label
    # is stripped. FakeGitHub.add_pr_label only logs the call, so the live PR
    # the next pass observes already shows the label absent (same #823 revert
    # shape as the silent-revert tests). Meanwhile main advanced past the
    # PR's merge-base, exactly what the stale-base gate checks.
    fake_gh.compare_overrides[("main", "sha-abc123")] = {
        "base_commit": {"sha": "new-main-tip"},
        "merge_base_commit": {"sha": "stale-ancestor"},
    }

    # Pass 2: the abandoned handoff no longer suppresses charlie's own sync,
    # so the stale-base deferral never fires — the deadlock's two halves
    # (skip-the-sync + early-return-before-recovery) are both broken.
    second = app.merge_ready(456, merge=True)

    assert fake_gh.pr_update_branch_calls == [456]
    assert second.data.get("stale_base") is not True
    assert second.data["consecutive_stale_base_deferrals"] == 0
    # The verified sync advanced the head, the approval carried forward, and
    # the mergequeue handoff was retried in the same pass. The revert is
    # still detected cross-pass and counted as a failed handoff.
    assert second.data["can_merge"] is True
    assert second.data["mergequeue_label_applied"] is True
    assert second.data["consecutive_failed_merge_attempts"] == 1
    persisted = load_state(paths.state_file)
    assert persisted["prs"]["456"]["status"] == "mergequeue"
    assert not any(event["kind"] == "merge_deferred_stale_base" for event in persisted["events"])

    # Pass 3: still reverted (the fake never mutates labels) but the base is
    # now current, so no further sync is attempted — only the handoff retry
    # and its accounting continue. This is the steady-state the bug denied:
    # the counter climbs toward the alarm threshold instead of the stale-base
    # deferral looping forever.
    third = app.merge_ready(456, merge=True)
    assert fake_gh.pr_update_branch_calls == [456]
    assert third.data["mergequeue_label_applied"] is True
    assert third.data["consecutive_failed_merge_attempts"] == 2


def test_merge_ready_mergequeue_reverted_handoff_sync_failure_still_recovers(
    tmp_path: Path,
) -> None:
    """Issue #1873 companion: even when charlie's own sync write fails, a
    reverted handoff must escape the stale-base deferral loop.

    pr_update_branch failing sets sync_failed, which bypasses the stale-base
    early return entirely — so the pass reaches the shared failed-attempt
    accounting instead of looping merge_deferred_stale_base with a counter
    nothing escalates on. consecutive_failed_merge_attempts increments every
    pass toward failed_attempt_alarm; consecutive_stale_base_deferrals stays
    at 0 because no deferral was recorded."""

    class UpdateBranchFailGitHub(FakeGitHub):
        """gh pr update-branch fails without moving the head."""

        def pr_update_branch(self, pr_number: int) -> bool:
            self.pr_update_branch_calls.append(pr_number)
            return False

    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = UpdateBranchFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    # Pass 1: fresh handoff, succeeds. Status becomes "mergequeue".
    first = app.merge_ready(456, merge=True)
    assert first.data["mergequeue_label_applied"] is True
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "mergequeue"

    # Label stripped by Aviator (absent on the live PR) + stale base.
    fake_gh.compare_overrides[("main", "sha-abc123")] = {
        "base_commit": {"sha": "new-main-tip"},
        "merge_base_commit": {"sha": "stale-ancestor"},
    }

    # Pass 2: the sync is attempted (not skipped) and fails; the pass falls
    # through to the generic approved-but-unmergeable accounting rather than
    # the stale-base deferral, so the failure counter advances.
    second = app.merge_ready(456, merge=True)

    assert fake_gh.pr_update_branch_calls == [456]
    assert second.data.get("stale_base") is not True
    assert second.data["can_merge"] is False
    assert second.data["mergequeue_label_applied"] is None
    assert second.data["consecutive_failed_merge_attempts"] == 1
    assert second.data["consecutive_stale_base_deferrals"] == 0
    persisted = load_state(paths.state_file)
    assert not any(event["kind"] == "merge_deferred_stale_base" for event in persisted["events"])


def test_merge_ready_mergequeue_label_add_failure_does_not_advance_status(
    tmp_path: Path,
) -> None:
    """Adversarial review finding #2: add_pr_label IS the entire handoff. A
    failed label add must not be silently treated like a best-effort cleanup
    step — it must not advance state to 'mergequeue' (that would orphan the
    PR: never self-merged, never picked up by Aviator, and nothing would ever
    look wrong to state). It must instead increment
    consecutive_failed_merge_attempts (so the failure retries and can
    escalate) and surface the failure in the result message."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.add_pr_label_ok = False
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["can_merge"] is True
    assert result.data["mergequeue_label_applied"] is False
    assert (456, "mergequeue") in fake_gh.pr_labels_added  # attempted
    assert fake_gh.merged == []
    assert "FAILED to apply" in result.message
    persisted = load_state(paths.state_file)["prs"]["456"]
    assert persisted["status"] == "approved"
    assert persisted["status"] != "mergequeue"
    assert persisted["consecutive_failed_merge_attempts"] == 1


def test_merge_ready_mergequeue_label_add_failure_alarm_fires_at_threshold(
    tmp_path: Path,
) -> None:
    """The failed-attempt alarm must be able to fire for a handoff-label
    failure too. Before this fix, can_merge is True whenever checks are green
    and the PR is approved (that is the whole point of reaching the
    mergequeue branch), so the pre-existing 'approved and not can_merge'
    alarm gate could never trigger for a persistently failing label add — a
    typo'd label or a missing repo label would retry forever with zero
    escalation."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            require_approved_review=True,
            mergequeue_label="mergequeue",
            failed_attempt_alarm=2,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.add_pr_label_ok = False
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    first = app.merge_ready(456, merge=True)
    assert first.data["merge_attempt_alarm"] is False
    assert first.data["consecutive_failed_merge_attempts"] == 1

    second = app.merge_ready(456, merge=True)
    assert second.data["merge_attempt_alarm"] is True
    assert second.data["consecutive_failed_merge_attempts"] == 2
    assert "mergequeue" in (second.data["merge_attempt_warning"] or "")


def test_merge_ready_mergequeue_silent_revert_increments_counter_across_passes(
    tmp_path: Path,
) -> None:
    """Issue #823: Aviator can accept the mergequeue label POST and then
    asynchronously strip it 2-3 seconds later (draft PR, failing required
    check, base mismatch, paused queue, ...). add_pr_label's boolean return
    only proves the POST succeeded, so a silent revert must be detected
    cross-pass (existing_pr_state's prior status vs. this pass's live
    labels) and must INCREMENT consecutive_failed_merge_attempts, never
    reset it. A single-pass test would still pass throughout the entire live
    incident this issue documents (PRs #690/#700, 2026-07-31) and is
    explicitly insufficient per the issue's acceptance criteria -- this
    drives three consecutive passes and asserts the counter actually climbs
    (0, 1, 2), which also empirically exercises the ordering hazard between
    the handoff-success zeroing write and the failed-attempt-alarm
    increment: without guarding the zeroing on mergequeue_label_reverted,
    every reverted pass would read a false 0 baseline and the sequence would
    be 0, 1, 1 instead."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    # Pass 1: fresh handoff, succeeds. Counter starts/stays at 0.
    first = app.merge_ready(456, merge=True)
    assert first.data["mergequeue_label_applied"] is True
    assert first.data["consecutive_failed_merge_attempts"] == 0
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "mergequeue"

    # FakeGitHub.add_pr_label only logs the call -- it never mutates
    # fake_gh.prs[0]["labels"] -- so the live PR the next pass observes
    # already shows the label absent, exactly like a real Aviator revert.
    #
    # Pass 2: existing_pr_state.status == "mergequeue" from pass 1, live
    # label absent -> reverted. The re-add attempt succeeds again this pass
    # (mergequeue_label_applied True), but the counter must climb to 1, not
    # reset to 0.
    second = app.merge_ready(456, merge=True)
    assert second.data["mergequeue_label_applied"] is True
    assert second.data["consecutive_failed_merge_attempts"] == 1
    assert second.data["merge_attempt_alarm"] is False

    # Pass 3: reverted again -> counter keeps climbing, not stuck at 1.
    third = app.merge_ready(456, merge=True)
    assert third.data["mergequeue_label_applied"] is True
    assert third.data["consecutive_failed_merge_attempts"] == 2
    assert third.data["merge_attempt_alarm"] is False


def test_merge_ready_mergequeue_silent_revert_escalation_reachable(tmp_path: Path) -> None:
    """Issue #823 acceptance criterion 3: after the configured number of
    consecutive reverted mergequeue handoffs, the PR must escalate exactly
    as an outright add_pr_label failure already does -- the same shared
    consecutive_failed_merge_attempts counter crossing the same
    failed_attempt_alarm threshold, no new parallel alarm mechanism."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            require_approved_review=True,
            mergequeue_label="mergequeue",
            failed_attempt_alarm=2,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    first = app.merge_ready(456, merge=True)
    assert first.data["merge_attempt_alarm"] is False
    assert first.data["consecutive_failed_merge_attempts"] == 0

    second = app.merge_ready(456, merge=True)
    assert second.data["merge_attempt_alarm"] is False
    assert second.data["consecutive_failed_merge_attempts"] == 1

    third = app.merge_ready(456, merge=True)
    assert third.data["merge_attempt_alarm"] is True
    assert third.data["consecutive_failed_merge_attempts"] == 2


def test_merge_ready_mergequeue_label_present_next_pass_not_treated_as_reverted(
    tmp_path: Path,
) -> None:
    """Regression guard: a PR whose mergequeue label IS still present on the
    live PR the next pass is the normal healthy path and must not be
    misdetected as a silent revert -- the counter must stay at 0 and no
    alarm must fire."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    first = app.merge_ready(456, merge=True)
    assert first.data["consecutive_failed_merge_attempts"] == 0
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "mergequeue"

    # The label genuinely survived to the next pass -- simulate the live PR
    # actually carrying it (unlike add_pr_label, which is a call log, not a
    # label mutation on the fake).
    fake_gh.prs[0]["labels"] = [{"name": "mergequeue"}]

    second = app.merge_ready(456, merge=True)
    assert second.data["mergequeue_label_applied"] is True
    assert second.data["consecutive_failed_merge_attempts"] == 0
    assert second.data["merge_attempt_alarm"] is False
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "mergequeue"


def test_merge_ready_mergequeue_revert_detector_ignores_non_mergequeue_prior_status(
    tmp_path: Path,
) -> None:
    """Issue #823 acceptance criterion 4: existing_pr_state empty, or with a
    status other than 'mergequeue', must never trip the revert detector --
    only a PRIOR pass that itself recorded status == 'mergequeue' is
    eligible. Seeds a plausible non-mergequeue prior status (freshly
    approved, never yet handed off) to prove the detector keys off
    status == 'mergequeue' specifically, not merely 'some prior state
    exists'."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    seed = load_state(paths.state_file)
    seed["prs"]["456"] = {
        "status": "approved",
        "issue_number": 123,
        "consecutive_failed_merge_attempts": 0,
    }
    save_state(paths.state_file, seed)

    result = app.merge_ready(456, merge=True)

    assert result.data["mergequeue_label_applied"] is True
    assert result.data["consecutive_failed_merge_attempts"] == 0
    assert result.data["merge_attempt_alarm"] is False
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "mergequeue"


def test_merge_ready_mergequeue_handoff_does_not_emit_merge_succeeded(
    tmp_path: Path,
) -> None:
    """Issue #747 negative control: the Aviator mergequeue handoff is a skip of
    the fleet's own merge (the fleet applies a label; Aviator merges
    asynchronously). ``merge_succeeded`` with actor='fleet' must NOT fire,
    because the fleet did not perform the merge -- this is what makes
    fleet-merged and externally-merged PRs distinguishable by the event."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.data["merged"] is False
    assert result.data["mergequeue_label_applied"] is True
    state = load_state(paths.state_file)
    success_events = [e for e in state["events"] if e["kind"] == "merge_succeeded"]
    assert success_events == []
