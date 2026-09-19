"""Merge-ready failed-attempt and stale-base alarms.

Split out of ``tests/test_charlie_work.py`` (issue #1550, Track-1
wave 4/8).
"""

from __future__ import annotations

from pathlib import Path
import pytest
from _fakes_github import (
    FakeGitHub,
    FakeGitHubWithChecks,
    FakeGitHubWithMissingRequired,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_merge_ready_failed_attempt_alarm_fires_once_at_threshold(tmp_path: Path) -> None:
    """Issue #254: after N approved-but-unmergeable passes, emit an alarm once."""
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # First two failed attempts do not alarm.
    result1 = app.merge_ready(456, merge=False)
    assert result1.data["can_merge"] is False
    assert result1.data["consecutive_failed_merge_attempts"] == 1
    assert result1.data["merge_attempt_alarm"] is False
    assert result1.data["merge_attempt_warning"] is None

    result2 = app.merge_ready(456, merge=False)
    assert result2.data["consecutive_failed_merge_attempts"] == 2
    assert result2.data["merge_attempt_alarm"] is False

    # Third attempt crosses the threshold.
    result3 = app.merge_ready(456, merge=False)
    assert result3.data["consecutive_failed_merge_attempts"] == 3
    assert result3.data["merge_attempt_alarm"] is True
    warning = result3.data["merge_attempt_warning"]
    assert warning is not None
    assert "PR #456 approved but unmergeable for 3 passes" in warning
    assert "required checks missing while GitHub shows the PR open" in warning

    # Fourth attempt is still unmergeable but does not re-alarm.
    result4 = app.merge_ready(456, merge=False)
    assert result4.data["consecutive_failed_merge_attempts"] == 4
    assert result4.data["merge_attempt_alarm"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 4
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert len(alarm_events) == 1
    assert alarm_events[0]["payload"]["pr_number"] == 456
    assert alarm_events[0]["payload"]["attempts"] == 3
    assert set(alarm_events[0]["payload"]["checks_summary"].keys()) == {
        "required",
        "passed",
        "pending",
        "failed",
        "missing",
        "infra_failed",
        "infra_blocked",
        "unavailable",
    }


def test_merge_ready_failed_attempt_alarm_reports_merge_state_not_unknown(
    tmp_path: Path,
) -> None:
    """Issue #751: when every check-summary bucket is empty (all required
    checks pass) but the PR is still unmergeable for a reason none of the
    explicitly modelled branches (merge_conflict, cross_pr_revert_detected,
    mergequeue_handoff_failed, summary.failed) name, the terminal ``else``
    branch must report GitHub's own ``mergeable``/``mergeStateStatus``
    instead of discarding it as "check summary unknown" (the real #679
    payload: all required checks green, alarm text still uninformative).

    The scenario is built by forcing a genuine merge-base staleness
    (compare_overrides) combined with a failed ``pr_update_branch`` — this
    sets ``sync_failed`` (and therefore ``can_merge=False``) without ever
    setting ``merge_conflict`` (which only looks at CONFLICTING/DIRTY), so
    the alarm chain falls through every explicit branch into the ``else``.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            # No required checks -> the check-summary buckets are vacuously
            # empty, isolating the terminal `else` branch under test.
            required_checks=(),
            require_approved_review=True,
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["mergeable"] = "MERGEABLE"
    fake_gh.prs[0]["mergeStateStatus"] = "BLOCKED"
    # Force the merge-base freshness check to see a stale base (independent of
    # mergeStateStatus, which the real _is_base_current path never consults),
    # then fail the resulting update-branch attempt so sync_failed=True without
    # ever tripping the CONFLICTING/DIRTY-only merge_conflict detector.
    fake_gh.compare_overrides[("main", "sha-abc123")] = {
        "base_commit": {"sha": "main-tip"},
        "merge_base_commit": {"sha": "main-tip-stale"},
    }
    fake_gh.update_branch_ok = False
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=False)
    assert result.data["can_merge"] is False
    assert result.data["merge_attempt_alarm"] is True
    warning = result.data["merge_attempt_warning"]
    assert warning is not None
    assert "mergeable=MERGEABLE" in warning
    assert "mergeStateStatus=BLOCKED" in warning
    # The negative control: without this assertion the test would also pass
    # against the pre-fix code, which always appended the literal fallback
    # regardless of what pr_view returned.
    assert "check summary unknown" not in warning

    state = load_state(paths.state_file)
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert len(alarm_events) == 1
    assert alarm_events[0]["payload"]["mergeable"] == "MERGEABLE"
    assert alarm_events[0]["payload"]["merge_state_status"] == "BLOCKED"


def test_merge_ready_failed_attempt_alarm_falls_back_when_merge_state_unknown(
    tmp_path: Path,
) -> None:
    """Issue #751: the "check summary unknown" fallback must survive for the
    genuinely-unknown case where GitHub reports neither ``mergeable`` nor
    ``mergeStateStatus`` as a usable signal (both ``"UNKNOWN"``), so a later
    refactor of the merge-state bucket can't silently delete the fallback.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            require_approved_review=True,
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["mergeable"] = "UNKNOWN"
    fake_gh.prs[0]["mergeStateStatus"] = "UNKNOWN"
    fake_gh.compare_overrides[("main", "sha-abc123")] = {
        "base_commit": {"sha": "main-tip"},
        "merge_base_commit": {"sha": "main-tip-stale"},
    }
    fake_gh.update_branch_ok = False
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=False)
    assert result.data["can_merge"] is False
    assert result.data["merge_attempt_alarm"] is True
    warning = result.data["merge_attempt_warning"]
    assert warning is not None
    assert "check summary unknown" in warning

    state = load_state(paths.state_file)
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert len(alarm_events) == 1
    assert alarm_events[0]["payload"]["mergeable"] == "UNKNOWN"
    assert alarm_events[0]["payload"]["merge_state_status"] == "UNKNOWN"


def test_merge_ready_failed_attempt_alarm_reports_unavailable_not_passed(
    tmp_path: Path,
) -> None:
    """When ``gh pr checks`` itself fails, summarize_checks(None, required)
    puts every required check into ``summary.unavailable`` — missing,
    pending, failed, and infra_failed all stay empty. The terminal ``else``
    branch's bucket chain checked those four buckets but never
    ``unavailable``, so it fell through to the "every bucket empty" fallback
    and (after the mergeable/mergeStateStatus fix above) would have claimed
    "all required checks passed" — an actively false statement when the
    check status could not be fetched at all. This must report the
    unavailable checks instead of asserting they passed.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests",),
            require_approved_review=True,
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubWithChecksUnavailable(FakeGitHub):
        def pr_checks(self, number: int):
            return None

    fake_gh = FakeGitHubWithChecksUnavailable()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=False)
    assert result.data["checks_unavailable"] is True
    assert result.data["can_merge"] is False
    assert result.data["merge_attempt_alarm"] is True
    warning = result.data["merge_attempt_warning"]
    assert warning is not None
    assert "unavailable: Tests" in warning
    # The negative control: an unavailable check is not a passed check.
    assert "all required checks passed" not in warning
    assert "check summary unknown" not in warning

    state = load_state(paths.state_file)
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert len(alarm_events) == 1
    assert alarm_events[0]["payload"]["checks_summary"]["unavailable"] == ["Tests"]


def test_merge_ready_failed_attempt_alarm_resets_on_merge(tmp_path: Path) -> None:
    """Issue #254: a successful merge resets the failed attempt counter."""
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    missing_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, missing_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    for _ in range(3):
        app.merge_ready(456, merge=False)

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 3

    # Now the checks turn green and merge succeeds.
    passing_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, passing_gh)
    result = app.merge_ready(456, merge=True)
    assert result.data["merged"] is True

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 0
    assert state["issues"]["123"]["merge_alert"] == "OK"


def test_merge_ready_failed_attempt_alarm_resets_on_head_move(tmp_path: Path) -> None:
    """Issue #254: a head move after approval resets the failed attempt counter."""
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.merge_ready(456, merge=False)
    app.merge_ready(456, merge=False)

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 2

    # Simulate the PR head advancing on GitHub.
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    result = app.merge_ready(456, merge=False)
    assert result.data["head_moved"] is True

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 0
    assert state["issues"]["123"]["merge_alert"] == "OK"


def test_merge_ready_failed_attempt_alarm_resets_on_decision_change(tmp_path: Path) -> None:
    """Issue #254: a decision change resets the failed attempt counter."""
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.merge_ready(456, merge=False)
    app.merge_ready(456, merge=False)

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 2

    # Operator changes decision to request_changes.
    app.record_review(
        456, "request_changes", summary="needs work", verdict_provenance="fresh_llm_review"
    )
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 0
    assert state["issues"]["123"]["merge_alert"] == "OK"


def test_merge_ready_failed_attempt_alarm_skips_pending_only_checks(tmp_path: Path) -> None:
    """Issue #254: pending-only checks must not count toward failed merge attempts."""
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    pending_checks = [
        {"name": "Tests passed", "state": "PENDING"},
        {"name": "Lint & Format", "state": "PENDING"},
        {"name": "Pre-commit", "state": "PENDING"},
    ]
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(checks=pending_checks)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    for _ in range(3):
        result = app.merge_ready(456, merge=False)
        assert result.data["can_merge"] is False
        assert result.data["merge_attempt_alarm"] is False
        assert result.data["merge_attempt_warning"] is None
        assert result.data["consecutive_failed_merge_attempts"] == 0

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 0
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert len(alarm_events) == 0


def test_merge_ready_failed_attempt_alarm_preserves_count_across_pending_only_pass(
    tmp_path: Path,
) -> None:
    """Issue #861: a pending-only pass must PRESERVE an already-accumulated
    failed-attempt count, not clobber it back to 0.

    ``test_merge_ready_failed_attempt_alarm_skips_pending_only_checks`` above
    only proves the counter stays 0 when it *starts* at 0 -- that also passed
    on the pre-#861 code, since the old default was an unconditional 0 every
    pass. This test seeds a nonzero count first, so it actually distinguishes
    "reset to 0" from "preserve the existing value."
    """
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.merge_ready(456, merge=False)
    app.merge_ready(456, merge=False)

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 2

    # Checks are now merely still running (pending-only) -- not failed, not
    # passed, not new information worth resetting the streak.
    pending_checks = [
        {"name": "Tests passed", "state": "PENDING"},
        {"name": "Lint & Format", "state": "PENDING"},
        {"name": "Pre-commit", "state": "PENDING"},
    ]
    pending_gh = FakeGitHubWithChecks(checks=pending_checks)
    app = OrchestratorApp(tmp_path, paths, config, pending_gh)
    result = app.merge_ready(456, merge=False)

    assert result.data["can_merge"] is False
    assert result.data["consecutive_failed_merge_attempts"] == 2
    assert result.data["merge_attempt_alarm"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 2
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert len(alarm_events) == 0


def test_merge_ready_failed_attempt_alarm_pending_passes_do_not_delay_threshold(
    tmp_path: Path,
) -> None:
    """Issue #861: interleaved pending-only passes must not reset progress
    toward the alarm threshold. Replays fail, pending, fail, pending, fail
    with threshold=3 and asserts the counter climbs 1, 1, 2, 2, 3 -- i.e. the
    pending passes hold steady rather than either incrementing or clobbering
    -- with exactly one alarm firing, on the final structural failure.

    Before the #861 fix this sequence never crossed the threshold: every
    pending-only pass reset the counter to 0, so the structural failures
    never accumulated past 1.
    """
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    pending_checks = [
        {"name": "Tests passed", "state": "PENDING"},
        {"name": "Lint & Format", "state": "PENDING"},
        {"name": "Pre-commit", "state": "PENDING"},
    ]
    fail_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fail_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    expected_sequence = [1, 1, 2, 2, 3]
    is_pending_step = [False, True, False, True, False]
    for expected, pending_step in zip(expected_sequence, is_pending_step, strict=True):
        if pending_step:
            app = OrchestratorApp(
                tmp_path, paths, config, FakeGitHubWithChecks(checks=pending_checks)
            )
        else:
            app = OrchestratorApp(tmp_path, paths, config, FakeGitHubWithMissingRequired())
        result = app.merge_ready(456, merge=False)
        assert result.data["consecutive_failed_merge_attempts"] == expected

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 3
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert len(alarm_events) == 1
    assert alarm_events[0]["payload"]["attempts"] == 3


def test_merge_ready_failed_attempt_alarm_clamps_at_threshold_plus_one(
    tmp_path: Path,
) -> None:
    """Issue #777(b), pinned against regression by #861: many consecutive
    structural failures must clamp the counter at ``threshold + 1`` (not grow
    unbounded, and not clamp AT threshold -- see the comment above the clamp
    in ``merge_ready`` for why threshold+1 is required for the alarm's
    one-shot semantics) and the alarm must fire exactly once, not on every
    pass once clamped.
    """
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    threshold = 3
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=threshold,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    results = [app.merge_ready(456, merge=False) for _ in range(6)]

    assert results[0].data["consecutive_failed_merge_attempts"] == 1
    assert results[1].data["consecutive_failed_merge_attempts"] == 2
    assert results[2].data["consecutive_failed_merge_attempts"] == threshold
    for result in results[3:]:
        assert result.data["consecutive_failed_merge_attempts"] == threshold + 1

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == threshold + 1
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert len(alarm_events) == 1
    assert alarm_events[0]["payload"]["attempts"] == threshold


def test_merge_ready_failed_attempt_alarm_resets_when_can_merge_without_merging(
    tmp_path: Path,
) -> None:
    """Issue #861: when an evaluation-only pass (``merge=False``) finds the PR
    genuinely mergeable, the counter must reset to 0 even though no merge (or
    mergequeue handoff) actually ran to reset it via those separate write
    paths. This is the one scenario that exercises the ``elif can_merge:``
    reset in ``merge_ready`` directly -- every other passing test that resets
    the counter does so through an earlier, unconditional write (a completed
    merge or mergequeue handoff), which zeroes ``existing`` before this
    method's shared write block re-reads it.
    """
    from charlie_work.config import AutoMergeConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fail_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fail_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.merge_ready(456, merge=False)
    app.merge_ready(456, merge=False)

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 2

    passing_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, passing_gh)
    result = app.merge_ready(456, merge=False)

    assert result.data["can_merge"] is True
    assert result.data["merged"] is False
    assert result.data["consecutive_failed_merge_attempts"] == 0

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 0


def test_merge_ready_stale_base_alarm_fires_after_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #368: an operator alarm is emitted after N consecutive base_stale
    deferrals for the same PR.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # The default base_head_sha is base-sha, which is already the merge-base of
    # sha-abc123. Advance it to a post-merge tip whose merge-base with sha-abc123
    # is still base-sha, so the freshness gate sees a stale base.
    post_merge_base = "main-merged-sha-abc123"
    fake_gh.base_head_sha = post_merge_base
    fake_gh.commits[post_merge_base] = {"parents": [{"sha": "base-sha"}, {"sha": "sha-abc123"}]}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Simulate a base-sync that reports success but does not advance the head.
    monkeypatch.setattr(fake_gh, "pr_update_branch", lambda pr_number: True)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    for attempt in range(1, 4):
        result = app.merge_ready(456, merge=False)
        assert result.data["can_merge"] is False
        assert result.data["merged"] is False
        assert result.data.get("stale_base") is True
        assert result.data["consecutive_stale_base_deferrals"] == attempt
        if attempt < 3:
            assert result.data["merge_attempt_alarm"] is False
            assert result.data["merge_attempt_warning"] is None
        else:
            assert result.data["merge_attempt_alarm"] is True
            assert result.data["merge_attempt_warning"] is not None
            assert "base is stale" in result.data["merge_attempt_warning"]

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_stale_base_deferrals"] == 3
    stale_events = [e for e in state["events"] if e["kind"] == "merge_deferred_stale_base"]
    assert len(stale_events) == 3
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_deferred_stale_base_alarm"]
    assert len(alarm_events) == 1
    assert alarm_events[0]["payload"]["pr_number"] == 456
    assert alarm_events[0]["payload"]["reason"] == "base_stale"
    assert alarm_events[0]["payload"]["attempts"] == 3
    assert alarm_events[0]["payload"]["threshold"] == 3

    # A fourth deferral is still counted but does not re-fire the alarm.
    result = app.merge_ready(456, merge=False)
    assert result.data["consecutive_stale_base_deferrals"] == 4
    assert result.data["merge_attempt_alarm"] is False
    assert result.data["merge_attempt_warning"] is None


def test_merge_ready_merge_alert_refires_after_can_merge_recovery(tmp_path: Path) -> None:
    """Issue #254: merge=False recovery resets merge_alert so a second degradation
    can re-fire the notify digest.
    """
    from charlie_work.config import AutoMergeConfig
    from charlie_work.workflow import _build_attention_digest

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    failing_checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "state": "FAILURE"},
        {"name": "Pre-commit", "state": "FAILURE"},
    ]
    passing_checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "state": "SUCCESS"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            failed_attempt_alarm=3,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    failing_gh = FakeGitHubWithChecks(checks=failing_checks)
    app = OrchestratorApp(tmp_path, paths, config, failing_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # First degradation to threshold.
    for _ in range(3):
        result = app.merge_ready(456, merge=False)
    warning = result.data["merge_attempt_warning"]
    assert warning is not None

    # Simulate the loop digest that would set merge_alert to MERGE_BLOCKED.
    _build_attention_digest(
        paths.state_file,
        {
            123: {
                "adapter_kind": "unknown",
                "health": "MERGE_BLOCKED",
                "last_log_line": None,
                "pid": None,
                "terminal_tool": None,
                "terminal_reason": warning,
            }
        },
        repo="test-repo",
        state_field="merge_alert",
    )
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["merge_alert"] == "MERGE_BLOCKED"

    # Recovery: can_merge=True but no merge attempted (merge=False).
    passing_gh = FakeGitHubWithChecks(checks=passing_checks)
    app = OrchestratorApp(tmp_path, paths, config, passing_gh)
    result = app.merge_ready(456, merge=False)
    assert result.data["can_merge"] is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["merge_alert"] == "OK"

    # Second degradation to threshold.
    app = OrchestratorApp(tmp_path, paths, config, failing_gh)
    for _ in range(3):
        result = app.merge_ready(456, merge=False)
    warning = result.data["merge_attempt_warning"]
    assert warning is not None

    # The digest should fire again because merge_alert moved OK -> MERGE_BLOCKED.
    digest = _build_attention_digest(
        paths.state_file,
        {
            123: {
                "adapter_kind": "unknown",
                "health": "MERGE_BLOCKED",
                "last_log_line": None,
                "pid": None,
                "terminal_tool": None,
                "terminal_reason": warning,
            }
        },
        repo="test-repo",
        state_field="merge_alert",
    )
    assert digest is not None
    assert len(digest.transitions) == 1
    assert digest.transitions[0].health == "MERGE_BLOCKED"
    assert digest.transitions[0].previous_health == "OK"
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["merge_alert"] == "MERGE_BLOCKED"
