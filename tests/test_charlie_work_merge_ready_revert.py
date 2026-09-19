"""Merge-ready silent-revert and racing-update defenses.

Split out of ``tests/test_charlie_work.py`` (issue #1550, Track-1
wave 4/8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest
from _fakes_github import FakeGitHub
from charlie_work.config import (
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
)
from charlie_work.workflow import OrchestratorApp
from _merge_ready_fixtures import (
    _init_cross_pr_revert_repo,
    _make_racing_merge_ready_app,
    _mergequeue_automerge,
)
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_merge_ready_self_revoked_stale_head_no_false_alarm(tmp_path: Path) -> None:
    """Issue #1402: when reconcile's own ``detect_mergequeue_not_approved``
    (#819 gap fix) revokes ``mergequeue`` because the approval is pinned to a
    stale head, and ``merge_ready``'s carry-forward then re-validates and
    re-applies the label in the same pass, the cross-pass
    ``mergequeue_label_reverted`` signal must NOT be folded into
    ``mergequeue_handoff_failed``. The label was removed by our own code
    (cooperative self-revocation), not by Aviator's #823 silent rejection --
    so the counter must stay at 0 and no alarm must fire.

    Simulates the exact PR #1843 sequence: pass 1 hands off to Aviator
    (status="mergequeue"), Aviator rebases the branch (head moves), reconcile
    revokes the label (stale-head, records ``mergequeue_revoked_reason``),
    pass 2 carry-forwards the approval and re-applies the label."""
    diff_text = (
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
    new_head = "sha-rebased-1843"
    pr_number = 456

    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[pr_number] = diff_text
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(pr_number, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    # Pass 1: fresh handoff, succeeds. Status becomes "mergequeue".
    first = app.merge_ready(pr_number, merge=True)
    assert first.data["mergequeue_label_applied"] is True
    assert first.data["consecutive_failed_merge_attempts"] == 0
    assert load_state(paths.state_file)["prs"][str(pr_number)]["status"] == "mergequeue"

    # Simulate reconcile's cooperative self-revocation: Aviator rebased the
    # branch (head moves), the approval is now at a stale head, reconcile
    # revokes the label and records the reason in state.
    state = load_state(paths.state_file)
    state["prs"][str(pr_number)]["mergequeue_revoked_reason"] = "stale_head_pending_carry_forward"
    save_state(paths.state_file, state)
    fake_gh.prs[0]["headRefOid"] = new_head
    fake_gh.diffs[pr_number] = diff_text  # same diff -> patch-id matches -> carry-forward

    # Pass 2: carry-forward re-validates the approval at the new head and
    # re-applies the mergequeue label. The label was absent (reconcile removed
    # it) and prior status was "mergequeue", so mergequeue_label_reverted is
    # True -- but mergequeue_self_revoked_stale_head is also True, so
    # mergequeue_handoff_failed must be False. No counter increment, no alarm.
    second = app.merge_ready(pr_number, merge=True)
    assert second.data["mergequeue_label_applied"] is True
    assert second.data["consecutive_failed_merge_attempts"] == 0
    assert second.data["merge_attempt_alarm"] is False

    # The self-revocation reason must be cleared so a subsequent Aviator #823
    # rejection is not misread as a stale self-revocation.
    state = load_state(paths.state_file)
    assert state["prs"][str(pr_number)].get("mergequeue_revoked_reason") is None


def test_merge_ready_aviator_revert_without_self_revocation_reason_still_counts(
    tmp_path: Path,
) -> None:
    """Issue #1402 regression guard: a genuine Aviator #823 silent rejection
    (no ``mergequeue_revoked_reason`` recorded -- the label was stripped by
    Aviator, not by reconcile) must still increment the counter and eventually
    fire the alarm. This is the exact same scenario as the pre-#1402
    ``test_merge_ready_mergequeue_silent_revert_increments_counter_across_passes``
    test, but explicitly seeds an ABSENT ``mergequeue_revoked_reason`` to prove
    the self-revocation exclusion does not swallow the #823 path."""
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    # Pass 1: fresh handoff, succeeds.
    first = app.merge_ready(456, merge=True)
    assert first.data["consecutive_failed_merge_attempts"] == 0

    # Explicitly ensure no self-revocation reason is set (Aviator stripped the
    # label, not reconcile). The key may be present as None from pass 1's
    # label re-add write, but it must NOT be "stale_head_pending_carry_forward".
    state = load_state(paths.state_file)
    assert (
        state["prs"]["456"].get("mergequeue_revoked_reason") != "stale_head_pending_carry_forward"
    )

    # Pass 2: label absent (FakeGitHub.add_pr_label doesn't mutate labels),
    # no self-revocation reason -> genuine #823 path -> counter increments.
    second = app.merge_ready(456, merge=True)
    assert second.data["mergequeue_label_applied"] is True
    assert second.data["consecutive_failed_merge_attempts"] == 1
    assert second.data["merge_attempt_alarm"] is False


def test_merge_ready_silent_cross_pr_revert_blocks_and_routes_to_rework(
    tmp_path: Path,
) -> None:
    """Issue #390: a branch that merges a base commit and reverts it must not merge."""
    from charlie_work.config import AutoMergeConfig, DevinConfig

    _base_sha, _feature_sha, agent_sha = _init_cross_pr_revert_repo(tmp_path)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        ),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: revert cross-pr",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-revert",
            "baseRefName": "main",
            "headRefOid": agent_sha,
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["cross_pr_revert_detected"] is True
    assert result.data["cross_pr_revert_routed"] is True
    assert result.data["merge_conflict"] is False
    assert "feature C" in result.data.get("cross_pr_revert_reason", "")

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["status"] == "rework_requested"
    assert any(e["kind"] == "cross_pr_revert_rework_requested" for e in state["events"])
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])

    prompt_path = paths.prs / "pr-456" / "rework-prompt.md"
    assert prompt_path.exists()
    assert (123, config.labels.needs_rework) in fake_gh.labels_added


def test_merge_ready_silent_cross_pr_revert_allows_explicit_marker(
    tmp_path: Path,
) -> None:
    """Issue #390: an explicit 'allow-revert:' marker line in the PR body suppresses the block."""
    from charlie_work.config import AutoMergeConfig, DevinConfig

    _base_sha, _feature_sha, agent_sha = _init_cross_pr_revert_repo(tmp_path)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        ),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: intentional revert",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-revert",
            "baseRefName": "main",
            "headRefOid": agent_sha,
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nallow-revert: intentional revert of feature C",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["can_merge"] is True
    assert result.data["cross_pr_revert_detected"] is False
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]


def test_merge_ready_silent_cross_pr_revert_prompt_echo_does_not_bypass(
    tmp_path: Path,
) -> None:
    """Issue #390: a bare 'allow-revert' word (e.g. quoting the rework prompt) must not bypass the gate."""
    from charlie_work.config import AutoMergeConfig, DevinConfig

    _base_sha, _feature_sha, agent_sha = _init_cross_pr_revert_repo(tmp_path)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        ),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: revert cross-pr",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-revert",
            "baseRefName": "main",
            "headRefOid": agent_sha,
            "mergeStateStatus": "CLEAN",
            "body": (
                "Closes #123\n\n"
                "...or add an explicit 'allow-revert' marker to the PR body if the revert "
                "is intentional. Then push the corrected branch and re-request review."
            ),
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["cross_pr_revert_detected"] is True
    assert result.data["cross_pr_revert_routed"] is True
    assert result.data["merge_conflict"] is False
    assert "feature C" in result.data.get("cross_pr_revert_reason", "")


def test_merge_ready_cross_pr_revert_undetermined_blocks_merge_without_rework(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1068: an undetermined cross-PR revert gate fails closed.

    When the gate cannot verify (transient local-git failure), the merge must
    be blocked (can_merge=False) but the PR must NOT be routed to rework —
    undetermined is not a detected revert, only a refusal to merge on an
    unverified gate. Previously the unverifiable state folded into the same
    None as "verified clean", silently disabling the gate.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work import workflow as workflow_module
    from charlie_work.cross_pr_revert import CrossPrRevertResult, CrossPrRevertStatus

    _base_sha, _feature_sha, agent_sha = _init_cross_pr_revert_repo(tmp_path)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        ),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: revert cross-pr",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-revert",
            "baseRefName": "main",
            "headRefOid": agent_sha,
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    def _undetermined(*_args: object, **_kwargs: object) -> CrossPrRevertResult:
        return CrossPrRevertResult(
            CrossPrRevertStatus.UNDETERMINED, "git fetch failed (simulated)"
        )

    monkeypatch.setattr(workflow_module, "detect_cross_pr_revert", _undetermined)

    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    # Undetermined blocks but is NOT a detected revert and is NOT routed.
    assert result.data["cross_pr_revert_detected"] is False
    assert result.data["cross_pr_revert_undetermined"] is True
    assert result.data["cross_pr_revert_routed"] is False
    assert result.data["merge_conflict"] is False
    assert "undetermined" in result.message
    assert "fail-closed" in result.message

    state = load_state(paths.state_file)
    # No rework routing happened for the linked issue.
    assert state["issues"].get("123", {}).get("status") != "rework_requested"
    assert state["prs"].get("456", {}).get("status") != "rework_requested"
    assert not any(e["kind"] == "cross_pr_revert_rework_requested" for e in state["events"])
    # The merge was never attempted.
    assert fake_gh.merged == []


def test_merge_ready_dry_run_cross_pr_revert_undetermined_blocks_without_rework(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1068: under --dry-run, an undetermined cross-PR revert gate fails closed.

    The dry-run path (``_merge_ready_dry_run``) must exercise the same
    fail-closed contract as the live ``merge_ready`` path: an UNDETERMINED
    verdict blocks the merge (``can_merge=False``) without routing to rework.
    No state is persisted and no merge is attempted. This is the regression
    test for the dry-run UNDETERMINED branch that previously had no coverage.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work import workflow as workflow_module
    from charlie_work.cross_pr_revert import CrossPrRevertResult, CrossPrRevertStatus

    _base_sha, _feature_sha, agent_sha = _init_cross_pr_revert_repo(tmp_path)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        ),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: revert cross-pr",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-revert",
            "baseRefName": "main",
            "headRefOid": agent_sha,
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    def _undetermined(*_args: object, **_kwargs: object) -> CrossPrRevertResult:
        return CrossPrRevertResult(
            CrossPrRevertStatus.UNDETERMINED, "git fetch failed (simulated)"
        )

    monkeypatch.setattr(workflow_module, "detect_cross_pr_revert", _undetermined)

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["dry_run"] is True
    # Undetermined blocks but is NOT a detected revert and is NOT routed.
    assert result.data["cross_pr_revert_detected"] is False
    assert result.data["cross_pr_revert_undetermined"] is True
    assert result.data["cross_pr_revert_routed"] is False
    assert result.data["merge_conflict"] is False
    assert "undetermined" in result.message
    assert "fail-closed" in result.message

    # No state was persisted — no rework routing under dry-run.
    state = load_state(paths.state_file)
    assert state["issues"].get("123", {}).get("status") != "rework_requested"
    assert state["prs"].get("456", {}).get("status") != "rework_requested"
    assert not any(e["kind"] == "cross_pr_revert_rework_requested" for e in state["events"])
    # The merge was never attempted.
    assert fake_gh.merged == []


def test_merge_ready_compare_unavailable_fail_closed(tmp_path: Path) -> None:
    """Issue #333: a failed compare API returns ``None`` and merge_ready fails closed.

    Mutating the gate to fail-open (treating ``base_current is None`` as current)
    causes this test to fail because the PR is merged instead of deferred.
    """
    from charlie_work.config import AutoMergeConfig

    class FakeGitHubCompareUnavailable(FakeGitHub):
        """compare() returns None, simulating an unavailable GitHub compare API."""

        def compare(self, base: str, head: str) -> dict[str, Any] | None:
            return None

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubCompareUnavailable()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merged"] is False
    assert result.data.get("stale_base") is True
    assert fake_gh.merged == []

    state = json.loads(paths.state_file.read_text())
    stale_events = [
        event for event in state["events"] if event["kind"] == "merge_deferred_stale_base"
    ]
    assert len(stale_events) == 1
    assert stale_events[0]["payload"]["pr_number"] == 456
    assert stale_events[0]["payload"]["reason"] == "compare_unavailable"


def test_merge_ready_merge_train_post_sync_head_race_rejected(tmp_path: Path) -> None:
    """If pr_view returns a non-qualifying head after update-branch, do not merge.

    Regression test for the TOCTOU described in issue #258: a racing push to the
    PR branch in the update-window must not be blessed as the approved head.
    """
    from charlie_work.config import AutoMergeConfig

    class FakeGitHubRacingUpdate(FakeGitHub):
        def pr_update_branch(self, pr_number: int) -> bool:
            ok = super().pr_update_branch(pr_number)
            for pr in self.prs:
                if pr["number"] == pr_number:
                    racing = "racing-sha"
                    self.pr_head_shas[pr_number] = racing
                    self.commits[racing] = {
                        "parents": [{"sha": "other-sha"}],
                        "committer": {"login": "not-web-flow"},
                        "commit": {"committer": {"name": "Not GitHub"}},
                    }
            return ok

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubRacingUpdate()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert fake_gh.merged == []
    # The approved head must not be migrated to the racing SHA.
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-abc123"


def test_merge_ready_race_with_spoofed_committer_name_rejected(tmp_path: Path) -> None:
    """A racing push whose git metadata claims name 'GitHub' must still be rejected.

    The commit.committer.name field is settable by any pusher; only the
    web-flow account login together with the GitHub name identifies a real
    base-sync merge. Structural parent checks are satisfied on purpose.
    """
    app, fake_gh, paths = _make_racing_merge_ready_app(
        tmp_path,
        {
            "parents": [{"sha": "sha-abc123"}, {"sha": "main-tip-sha"}],
            "committer": {"login": "attacker"},
            "commit": {"committer": {"name": "GitHub"}},
        },
    )

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=True)

    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert fake_gh.merged == []
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-abc123"


def test_merge_ready_race_with_spoofed_webflow_login_rejected(tmp_path: Path) -> None:
    """A racing push attributed to web-flow but with a non-GitHub name is rejected.

    Login attribution follows the committer email, which a pusher can set to
    noreply@github.com; the git metadata name must corroborate it.
    """
    app, fake_gh, paths = _make_racing_merge_ready_app(
        tmp_path,
        {
            "parents": [{"sha": "sha-abc123"}, {"sha": "main-tip-sha"}],
            "committer": {"login": "web-flow"},
            "commit": {"committer": {"name": "Devin Worker"}},
        },
    )

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=True)

    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert fake_gh.merged == []
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-abc123"
