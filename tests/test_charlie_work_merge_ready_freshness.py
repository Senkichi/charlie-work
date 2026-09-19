"""Merge-ready base-freshness gate.

Split out of ``tests/test_charlie_work.py`` (issue #1550, Track-1
wave 4/8).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from _fakes_github import FakeGitHub
from charlie_work.config import (
    AutoMergeConfig,
    OrchestratorConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from _merge_ready_fixtures import _stale_base_prs
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_merge_ready_stale_base_deferred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #316: a PR whose merge-base is not the current base tip is deferred."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
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
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "baseRefName": "main",
            "headRefOid": "sha-def456",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    # Create review decision files for both PRs (approved state)
    for pr_number, head_sha in [(456, "sha-abc123"), (789, "sha-def456")]:
        decision_dir = paths.prs / f"pr-{pr_number}"
        decision_dir.mkdir(parents=True, exist_ok=True)
        (decision_dir / "review-decision.json").write_text(
            json.dumps(
                {"decision": "approved", "reviewed_head_sha": head_sha},
                indent=2,
            ),
            encoding="utf-8",
        )

    # Merging PR 456 advances the fake base tip. PR 456 is then up-to-date with
    # the new base tip, but PR 789's merge-base is still the old base tip, so
    # the merge-base freshness gate defers it organically.

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Simulate a base-sync that reports success but does not advance the head, so
    # the merge-base freshness gate still defers the PR after the first PR merges.
    monkeypatch.setattr(fake_gh, "pr_update_branch", lambda pr_number: True)

    result_456 = app.merge_ready(456, merge=True)
    assert result_456.ok is True
    assert result_456.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]

    result_789 = app.merge_ready(789, merge=True, merge_train_head=789)
    assert result_789.ok is True
    assert result_789.data["can_merge"] is False
    assert result_789.data["merged"] is False
    assert result_789.data.get("stale_base") is True
    assert fake_gh.merged == [(456, "squash")]

    state = json.loads(paths.state_file.read_text())
    stale_events = [
        event for event in state["events"] if event["kind"] == "merge_deferred_stale_base"
    ]
    assert len(stale_events) == 1
    assert stale_events[0]["payload"]["pr_number"] == 789
    assert stale_events[0]["payload"]["reason"] == "base_stale"


def test_merge_ready_require_current_base_false_allows_stale_base(tmp_path: Path) -> None:
    """Operators may opt out of the merge-base freshness gate."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="off",
            require_current_base=False,
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
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "baseRefName": "main",
            "headRefOid": "sha-def456",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    for pr_number, head_sha in [(456, "sha-abc123"), (789, "sha-def456")]:
        decision_dir = paths.prs / f"pr-{pr_number}"
        decision_dir.mkdir(parents=True, exist_ok=True)
        (decision_dir / "review-decision.json").write_text(
            json.dumps(
                {"decision": "approved", "reviewed_head_sha": head_sha},
                indent=2,
            ),
            encoding="utf-8",
        )

    # With require_current_base=False the gate is disabled, so the second PR
    # ships even though merge_pr(456) has advanced the fake base tip.

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result_456 = app.merge_ready(456, merge=True)
    assert result_456.data["merged"] is True

    result_789 = app.merge_ready(789, merge=True)
    assert result_789.data["can_merge"] is True
    assert result_789.data["merged"] is True
    assert result_789.data.get("stale_base") is not True
    assert fake_gh.merged == [(456, "squash"), (789, "squash")]

    state = json.loads(paths.state_file.read_text())
    assert not any(e["kind"] == "merge_deferred_stale_base" for e in state["events"])


def test_merge_ready_protection_strict_true_defers_even_when_config_disables_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #812: base freshness is DERIVED from branch protection, not merely
    defaulted from config. Prove the direction that matters most for adoption --
    protection strict:true still enforces the gate even though the operator has
    explicitly set require_current_base=False in config. If this regressed to
    "config wins", a repo relying on protection-derived enforcement while having
    require_current_base=False would silently stop deferring stale merges.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,  # broadcast
            require_current_base=False,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = _stale_base_prs()
    fake_gh.branch_protection_overrides["main"] = {"required_status_checks": {"strict": True}}

    for pr_number, head_sha in [(456, "sha-abc123"), (789, "sha-def456")]:
        decision_dir = paths.prs / f"pr-{pr_number}"
        decision_dir.mkdir(parents=True, exist_ok=True)
        (decision_dir / "review-decision.json").write_text(
            json.dumps({"decision": "approved", "reviewed_head_sha": head_sha}, indent=2),
            encoding="utf-8",
        )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    monkeypatch.setattr(fake_gh, "pr_update_branch", lambda pr_number: True)

    result_456 = app.merge_ready(456, merge=True)
    assert result_456.data["merged"] is True

    result_789 = app.merge_ready(789, merge=True)
    assert result_789.data["can_merge"] is False
    assert result_789.data["merged"] is False
    assert result_789.data.get("stale_base") is True
    assert fake_gh.merged == [(456, "squash")]
    assert "main" in fake_gh.branch_protection_calls

    state = json.loads(paths.state_file.read_text())
    stale_events = [e for e in state["events"] if e["kind"] == "merge_deferred_stale_base"]
    assert len(stale_events) == 1
    assert stale_events[0]["payload"]["pr_number"] == 789


def test_merge_ready_protection_strict_false_still_syncs_stale_base_before_merge(
    tmp_path: Path,
) -> None:
    """Issue #875: protection ``strict: false`` must NOT disable the merge gate.

    This test previously pinned the opposite (issue #812): with ``strict:
    false``, PR 789 merged on a stale base and ``pr_update_branch`` was never
    called. That is the defect. This repo runs ``strict: false`` deliberately
    (two self-hosted runners cannot sustain strict mode), so the gate derived
    "currency not required" and disabled itself -- an approved PR whose base had
    advanced merged without its merged tree ever being tested, which is how
    ``main`` went red.

    The corrected policy is ``require_current_base OR protection.strict``:
    protection may raise the requirement, never lower it below what the operator
    configured. So PR 789 is now brought current *first* and merges on the
    rebased tree.

    The write firing here is also what makes the gate safe: gating merges while
    leaving the write suppressed would deadlock a stale PR forever.
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,  # broadcast
        )
    )
    assert config.auto_merge.require_current_base is True  # sanity: default unchanged
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = _stale_base_prs()
    fake_gh.branch_protection_overrides["main"] = {"required_status_checks": {"strict": False}}

    for pr_number, head_sha in [(456, "sha-abc123"), (789, "sha-def456")]:
        decision_dir = paths.prs / f"pr-{pr_number}"
        decision_dir.mkdir(parents=True, exist_ok=True)
        (decision_dir / "review-decision.json").write_text(
            json.dumps({"decision": "approved", "reviewed_head_sha": head_sha}, indent=2),
            encoding="utf-8",
        )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result_456 = app.merge_ready(456, merge=True)
    assert result_456.data["merged"] is True
    # 456's base is still current when it merges, so nothing to sync.
    assert fake_gh.pr_update_branch_calls == []

    result_789 = app.merge_ready(789, merge=True)
    # 456's merge advanced the base tip, leaving 789 organically stale. The gate
    # is now live despite strict:false, so 789 is synced before it merges --
    # pre-#875 this list stayed empty and 789 merged stale.
    assert fake_gh.pr_update_branch_calls == [789]
    assert result_789.data["merged"] is True
    assert fake_gh.merged == [(456, "squash"), (789, "squash")]


def test_merge_ready_strict_false_defers_when_sync_cannot_make_base_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #875, the refusal half: when the just-in-time sync cannot make the
    PR current, the merge must be refused rather than proceeding on a stale
    base. Distinct from the test above, which proves the happy path (sync
    succeeds -> merge on the rebased tree); this proves the gate actually
    blocks, so a green sync is not the only reason merges stop being stale.

    ``pr_update_branch`` returning False is a *different* failure mode than
    the one this test needs: it makes ``merge_ready`` set ``sync_failed``
    (workflow.py's ``else: sync_failed = True`` branch right after the
    ``self.gh.pr_update_branch(pr_number)`` call), which short-circuits the
    base-currency check entirely (guarded by ``not sync_failed``) and routes
    through the generic "approved but unmergeable" bookkeeping instead --
    bumping ``consecutive_failed_merge_attempts`` with no ``stale_base``/
    ``reason`` in the result and no ``merge_deferred_stale_base`` event. That
    is real, correct behavior for an outright write failure, but it is not
    the stale-base refusal path this test is meant to exercise.

    The failure mode that actually reaches ``_merge_deferred_stale_base_result``
    is a sync that *reports* success without moving the branch: mirroring the
    sibling test above, ``pr_update_branch`` is monkeypatched to return True
    (so ``sync_failed`` stays False) while never performing the fake's normal
    side effect of advancing ``headRefOid``. ``_verify_synced_head`` then sees
    the head SHA unchanged, treats it as "already up-to-date; nothing to do",
    and the subsequent base-currency check finds the (still organically
    stale) merge-base is not the current tip -- so the gate defers rather
    than merging. This models a real GitHub race: the update-branch call is
    accepted but the branch does not actually advance before the next read
    (e.g. lost a race with another writer, or a conflict GitHub resolves as a
    no-op).
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,  # broadcast
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = _stale_base_prs()
    fake_gh.branch_protection_overrides["main"] = {"required_status_checks": {"strict": False}}

    for pr_number, head_sha in [(456, "sha-abc123"), (789, "sha-def456")]:
        decision_dir = paths.prs / f"pr-{pr_number}"
        decision_dir.mkdir(parents=True, exist_ok=True)
        (decision_dir / "review-decision.json").write_text(
            json.dumps({"decision": "approved", "reviewed_head_sha": head_sha}, indent=2),
            encoding="utf-8",
        )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    assert app.merge_ready(456, merge=True).data["merged"] is True

    # The sync reports success but never actually advances 789's head (see
    # the docstring above): this is what makes the JIT sync fail to bring the
    # PR current without tripping the unrelated sync_failed short-circuit.
    # Record calls directly (the monkeypatch replaces the whole method, so
    # FakeGitHub's own pr_update_branch_calls bookkeeping never runs) so the
    # test proves the sync was actually ATTEMPTED and merely didn't help --
    # not just that the PR was already stale before any sync was tried.
    update_calls: list[int] = []

    def _sync_reports_success_without_moving_head(pr_number: int) -> bool:
        update_calls.append(pr_number)
        return True

    monkeypatch.setattr(fake_gh, "pr_update_branch", _sync_reports_success_without_moving_head)

    result_789 = app.merge_ready(789, merge=True)
    assert update_calls == [789]
    assert result_789.data["can_merge"] is False
    assert result_789.data["merged"] is False
    assert result_789.data.get("stale_base") is True
    assert fake_gh.merged == [(456, "squash")]

    # _merge_deferred_stale_base_result's CommandResult.data does not carry a
    # "reason" key (only "stale_base": True) -- the reason is recorded on the
    # merge_deferred_stale_base event payload instead, so verify it there.
    state = json.loads(paths.state_file.read_text())
    stale_events = [e for e in state["events"] if e["kind"] == "merge_deferred_stale_base"]
    assert len(stale_events) == 1
    assert stale_events[0]["payload"]["pr_number"] == 789
    assert stale_events[0]["payload"]["reason"] == "base_stale"


def test_merge_ready_strategy_off_skips_freshness_check_despite_protection_strict_true(
    tmp_path: Path,
) -> None:
    """Regression pin for a deadlock this fix could otherwise introduce:
    update_branch_strategy="off" means there is no sync mechanism at all, so
    requiring base currency would be an inescapable deferral loop -- the same
    hazard AutoMergeConfig.__post_init__ already blocks for the config-only
    case (require_current_base=True + strategy=off). Since base_freshness_required
    can now become True purely from protection (independent of config), the
    strategy=="off" guard in merge_ready must short-circuit it before the
    protection read is ever consulted.
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="off",
            require_current_base=False,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = _stale_base_prs()
    # If the "off" guard were missing, this would force freshness_required=True
    # for both PRs and PR 789 would be deferred below.
    fake_gh.branch_protection_overrides["main"] = {"required_status_checks": {"strict": True}}

    for pr_number, head_sha in [(456, "sha-abc123"), (789, "sha-def456")]:
        decision_dir = paths.prs / f"pr-{pr_number}"
        decision_dir.mkdir(parents=True, exist_ok=True)
        (decision_dir / "review-decision.json").write_text(
            json.dumps({"decision": "approved", "reviewed_head_sha": head_sha}, indent=2),
            encoding="utf-8",
        )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result_456 = app.merge_ready(456, merge=True)
    assert result_456.data["merged"] is True

    result_789 = app.merge_ready(789, merge=True)
    assert result_789.data["merged"] is True
    assert result_789.data.get("stale_base") is not True
    assert fake_gh.merged == [(456, "squash"), (789, "squash")]
    # strategy=="off" short-circuits before the protection read ever happens.
    assert fake_gh.branch_protection_calls == []


def test_merge_ready_next_mode_syncs_head_before_merge(tmp_path: Path) -> None:
    """Merge-train head with a BEHIND mergeStateStatus is base-synced before merge."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["mergeStateStatus"] = "BEHIND"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    # The branch was synced before the merge
    assert fake_gh.prs[0]["headRefOid"] == "sha-abc123-updated"
    # The approved head was updated to match the new base
    decision = json.loads((paths.prs / "pr-456" / "review-decision.json").read_text())
    assert decision["reviewed_head_sha"] == "sha-abc123-updated"


def test_merge_ready_next_mode_skips_non_head(tmp_path: Path) -> None:
    """In merge-train mode, a non-head approved PR cannot be merged."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
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
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.record_review(789, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    # Ensure 456 is the head of the queue regardless of when approvals occurred.
    for idx, pr_number in enumerate((456, 789)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    result = app.merge_ready(789, merge=True)

    assert result.ok is True
    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert "not the head" in result.message
    assert fake_gh.merged == []


def test_merge_ready_clean_stale_base_syncs_and_merges(tmp_path: Path) -> None:
    """Issue #334: approved PR with mergeStateStatus CLEAN but stale merge-base is synced and merged."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # mergeStateStatus is CLEAN, but the compare API says the merge-base is stale.
    fake_gh.compare_overrides[("main", "sha-abc123")] = {
        "base_commit": {"sha": "base-sha"},
        "merge_base_commit": {"sha": "base-sha-old"},
    }
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    # The branch was synced despite mergeStateStatus CLEAN because the base was stale.
    assert fake_gh.prs[0]["headRefOid"] == "sha-abc123-updated"
    # The approved head was updated to match the new base.
    decision = json.loads((paths.prs / "pr-456" / "review-decision.json").read_text())
    assert decision["reviewed_head_sha"] == "sha-abc123-updated"


def test_merge_ready_current_base_no_sync(tmp_path: Path) -> None:
    """Issue #334 negative control: an already-current approved PR should not be synced."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # mergeStateStatus CLEAN and the compare API agrees the base is current.
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    # No update-branch should have been attempted.
    assert fake_gh.prs[0]["headRefOid"] == "sha-abc123"
    decision = json.loads((paths.prs / "pr-456" / "review-decision.json").read_text())
    assert decision["reviewed_head_sha"] == "sha-abc123"
