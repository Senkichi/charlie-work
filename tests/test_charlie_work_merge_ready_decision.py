"""Merge-ready decision/verdict verification.

Split out of ``tests/test_charlie_work.py`` (issue #1550, Track-1
wave 4/8).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any
import pytest
from _fakes_github import (
    FakeGitHub,
    FakeGitHubWithChecks,
)
from _review_fixtures import (
    _approved_automerge,
    _required_checks_config,
)
from charlie_work.config import (
    AutoMergeConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_merge_ready_carries_forward_approved_verdict_on_tier2_line_content(
    tmp_path: Path,
) -> None:
    """Issue #414: the ship-it merge gate also carries forward via tier 2
    when patch-ids differ due to main-advance context drift but the ordered
    +/- lines and changed-file set are identical."""
    from charlie_work.janitor import _calculate_patch_id, _diff_content_signature

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    reviewed_diff = (
        "diff --git a/file b/file\n"
        "index 123..456 78910\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " alpha\n"
        " beta\n"
        "+gamma\n"
        " delta\n"
    )
    live_diff = (
        "diff --git a/file b/file\n"
        "index 123..789 78910\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " alpha\n"
        " beta-updated\n"
        "+gamma\n"
        " delta\n"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    live_patch_id = _calculate_patch_id(live_diff)
    assert reviewed_patch_id != live_patch_id, "test fixture must reproduce patch-id drift"
    reviewed_signature = _diff_content_signature(reviewed_diff)

    old_head = "sha-abc123"
    new_head = "sha-rebased123"

    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": old_head,
                "reviewed_patch_id": reviewed_patch_id,
                "reviewed_changed_lines": list(reviewed_signature.changed_lines),
                "reviewed_changed_files": sorted(reviewed_signature.changed_files),
                "summary": "lgtm",
            }
        ),
        encoding="utf-8",
    )

    # Simulate a rebase-style head move (not a 2-parent web-flow merge commit,
    # so _verify_synced_head would reject it) with genuine patch-id drift
    # from an intervening main advance, but content-identical +/- lines.
    fake_gh.pr_head_shas[456] = new_head
    fake_gh.diffs[456] = live_diff
    fake_gh.compare_overrides[("main", new_head)] = {
        "base_commit": {"sha": fake_gh.base_head_sha},
        "merge_base_commit": {"sha": fake_gh.base_head_sha},
    }
    fake_gh.commits[new_head] = {
        "parents": [{"sha": old_head}],
        "committer": {"login": "someone"},
        "commit": {"committer": {"name": "Not GitHub"}},
    }

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["can_merge"] is True
    assert result.data["merged"] is True
    assert result.data.get("head_moved") is not True
    assert fake_gh.merged == [(456, "squash")]

    decision = json.loads((decision_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert decision["reviewed_head_sha"] == new_head
    assert decision["carry_forward_tier"] == "line-content"
    assert decision["reviewed_patch_id"] == live_patch_id
    assert decision["carried_forward_from"] == [old_head]

    state = load_state(paths.state_file)
    pr_state = state["prs"]["456"]
    assert pr_state["reviewed_head_sha"] == new_head
    assert pr_state["carry_forward_tier"] == "line-content"
    assert pr_state["carried_forward_from"] == [old_head]
    assert pr_state["status"] != "reviewing"

    carry_events = [
        e for e in state["events"] if e["kind"] == "verdict_carried_forward_line_content"
    ]
    assert len(carry_events) == 1
    payload = carry_events[0]["payload"]
    assert payload["pr_number"] == 456
    assert payload["old_reviewed_head_sha"] == old_head
    assert payload["new_head_sha"] == new_head
    assert payload["patch_id"] == live_patch_id
    assert payload["carry_forward_tier"] == "line-content"
    assert payload["carried_forward_from"] == [old_head]

    # No review_started transition should fire for a tier-2 carry-forward.
    assert (123, "agent:reviewing") not in fake_gh.labels_added


def test_merge_ready_post_update_branch_records_verified_sync_event(
    tmp_path: Path,
) -> None:
    """Issue #638: the post-``pr_update_branch`` + ``_verify_synced_head``
    carry-forward inside ``merge_ready`` (a previously-silent
    ``verified-sync`` call site) must record a
    ``verdict_carried_forward_verified_sync`` event."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_branch_strategy="broadcast",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    pr_number = 456
    issue_number = 123
    old_head = "sha-abc123"
    fake_gh.prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": old_head,
            "mergeStateStatus": "BEHIND",
            "mergeable": "MERGEABLE",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(pr_number, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # Advance the base tip so the PR is stale and triggers pr_update_branch.
    post_merge_base = "main-merged-sha"
    fake_gh.base_head_sha = post_merge_base
    fake_gh.commits[post_merge_base] = {"parents": [{"sha": "base-sha"}, {"sha": old_head}]}

    result = app.merge_ready(pr_number, merge=False)
    assert result.ok is True

    state = load_state(paths.state_file)
    sync_events = [
        e for e in state["events"] if e["kind"] == "verdict_carried_forward_verified_sync"
    ]
    assert len(sync_events) == 1
    payload = sync_events[0]["payload"]
    assert payload["pr_number"] == pr_number
    assert payload["issue_number"] == issue_number
    assert payload["old_reviewed_head_sha"] == old_head
    assert payload["carry_forward_tier"] == "verified-sync"


def test_merge_ready_head_sync_verification_none_sets_sync_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A degraded post-sync ``pr.get("headRefOid")`` (resolving to ``None``)
    must be treated as a FAILED sync verification, not silently as
    "already up-to-date". ``_verify_synced_head`` always returns ``None``
    when passed ``old_head_sha=None`` (its sentinel for "verification
    failed"), but the buggy call site in ``merge_ready`` checked
    ``new_head == live_head_sha`` before checking ``new_head is None``, so
    ``None == None`` took the up-to-date no-op branch and ``sync_failed``
    stayed False despite a real, unverified ``pr_update_branch`` mutation
    having just happened on GitHub."""
    from charlie_work.config import AutoMergeConfig

    class FakeGitHubDegradedSecondFetch(FakeGitHub):
        """Returns the real PR on the first two ``pr_view`` calls (consumed
        by ``record_review`` and ``merge_ready``'s initial fetch), then a
        degraded ``headRefOid: None`` on every call after that -- modeling a
        GitHub API response missing the field on the re-fetch that follows
        a carry-forward."""

        def __init__(self) -> None:
            super().__init__()
            self._pr_view_calls = 0

        def pr_view(self, number: int):
            self._pr_view_calls += 1
            pr_copy = super().pr_view(number)
            if self._pr_view_calls > 2:
                pr_copy["headRefOid"] = None
            return pr_copy

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_branch_strategy="broadcast",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubDegradedSecondFetch()
    pr_number = 456
    issue_number = 123
    old_head = "sha-abc123"
    new_head = "sha-v2-rebased"
    original_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+line3\n"
        " line4\n"
    )
    fake_gh.prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": old_head,
            "mergeStateStatus": "BEHIND",
            "mergeable": "MERGEABLE",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    fake_gh.diffs[pr_number] = original_diff

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(pr_number, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # Simulate the branch moving (e.g. a rebase) with unchanged cumulative
    # content, so the patch-id carry-forward tier fires naturally and the
    # head_moved gate does not short-circuit before reaching the
    # pr_update_branch sync path under test.
    fake_gh.prs[0]["headRefOid"] = new_head

    # The compare API reports the branch as stale, so merge_ready attempts a
    # sync regardless of what the (degraded) post-carry-forward pr_view call
    # reports for headRefOid -- isolating this test to the
    # _verify_synced_head call-site bug rather than the unrelated
    # base-freshness signal.
    monkeypatch.setattr(app, "_is_base_current", lambda pr: False)

    result = app.merge_ready(pr_number, merge=True)

    assert result.ok is True
    # All required checks are green and the verdict is approved -- the ONLY
    # thing that can make can_merge False here is sync_failed having been
    # set from the None-verification result. This also rules out the
    # merge-base-freshness deferral gate as a false-positive explanation for
    # can_merge being False: that gate returns before ever fetching real
    # checks (an empty/synthetic "checks" summary), so seeing the real,
    # all-passed checks proves execution reached the sync_failed-gated
    # can_merge computation instead.
    assert result.data["checks"]["passed"] == (
        "Tests passed",
        "Lint & Format",
        "Pre-commit",
    )
    assert result.data["checks"]["missing"] == ()
    assert result.data["review_decision"]["decision"] == "approved"
    assert result.data.get("stale_base") is not True
    assert result.data["can_merge"] is False
    assert result.data.get("merged") is not True
    assert fake_gh.merged == []


def test_merge_ready_event_persists_gate_inputs_distinguishing_false_causes(
    tmp_path: Path,
) -> None:
    """Issue #1060: the ``merge_ready`` event must persist the three gate
    inputs (``summary_ready``, ``approved``, ``require_approved_review``,
    ``sync_failed``) alongside ``can_merge``, plus ``mergequeue_label_applied``,
    so a ``can_merge=False`` can be diagnosed from events.db alone.

    The three distinct false-causes -- CI not green, no recorded approval, base
    sync failed -- must produce three *different* payloads, not three identical
    ones. Mutating any single input must change the recorded event; if it does
    not, the record is not load-bearing.
    """

    def _last_merge_ready_payload(state_file: Path) -> dict[str, Any]:
        state = load_state(state_file)
        events = [e for e in state["events"] if e["kind"] == "merge_ready"]
        assert events, "no merge_ready event was recorded"
        return events[-1]["payload"]

    gate_keys = {"summary_ready", "approved", "require_approved_review", "sync_failed"}

    # --- Scenario 1: CI not green (summary_ready=False) -------------------
    # require_current_base=False keeps the base-freshness deferral gate off so
    # the pass reaches the merge_ready event instead of short-circuiting on a
    # stale-base deferral (which records a different event kind).
    config_ci = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed",),
            require_approved_review=True,
            require_current_base=False,
        )
    )
    paths_ci = runtime_paths(tmp_path / "ci", config_ci.runtime.state_dir)
    fake_gh_ci = FakeGitHubWithChecks(checks=[{"name": "Tests passed", "state": "FAILURE"}])
    app_ci = OrchestratorApp(tmp_path / "ci", paths_ci, config_ci, fake_gh_ci)
    app_ci.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")
    app_ci.merge_ready(456, merge=False)
    payload_ci = _last_merge_ready_payload(paths_ci.state_file)

    # --- Scenario 2: no recorded approval (approved=False) ----------------
    config_noappr = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=(),
            require_approved_review=True,
            require_current_base=False,
        )
    )
    paths_noappr = runtime_paths(tmp_path / "noappr", config_noappr.runtime.state_dir)
    fake_gh_noappr = FakeGitHub()
    app_noappr = OrchestratorApp(tmp_path / "noappr", paths_noappr, config_noappr, fake_gh_noappr)
    # Deliberately do NOT record a review -> approved=False.
    app_noappr.merge_ready(456, merge=False)
    payload_noappr = _last_merge_ready_payload(paths_noappr.state_file)

    # --- Scenario 3: base sync failed (sync_failed=True) ------------------
    # A genuine merge conflict (mergeable=CONFLICTING) sets sync_failed=True
    # before the gate, while the default passing checks keep summary_ready=True.
    config_sync = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            require_approved_review=True,
        )
    )
    paths_sync = runtime_paths(tmp_path / "sync", config_sync.runtime.state_dir)
    fake_gh_sync = FakeGitHub()
    fake_gh_sync.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "BEHIND",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app_sync = OrchestratorApp(tmp_path / "sync", paths_sync, config_sync, fake_gh_sync)
    app_sync.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")
    app_sync.merge_ready(456, merge=False)
    payload_sync = _last_merge_ready_payload(paths_sync.state_file)

    # All three are can_merge=False but for three distinct reasons, and every
    # payload carries the gate inputs plus the persisted handoff outcome.
    for payload in (payload_ci, payload_noappr, payload_sync):
        assert payload["can_merge"] is False
        assert gate_keys <= set(payload)
        assert "mergequeue_label_applied" in payload

    # Scenario 1: CI not green.
    assert payload_ci["summary_ready"] is False
    assert payload_ci["approved"] is True
    assert payload_ci["require_approved_review"] is True
    assert payload_ci["sync_failed"] is False

    # Scenario 2: no recorded approval.
    assert payload_noappr["summary_ready"] is True
    assert payload_noappr["approved"] is False
    assert payload_noappr["require_approved_review"] is True
    assert payload_noappr["sync_failed"] is False

    # Scenario 3: base sync failed (merge conflict).
    assert payload_sync["summary_ready"] is True
    assert payload_sync["approved"] is True
    assert payload_sync["require_approved_review"] is True
    assert payload_sync["sync_failed"] is True

    # The three payloads are distinguishable on the gate-input sub-dict: no two
    # share the same (summary_ready, approved, sync_failed) triple. Mutating any
    # single input changes the recorded event -- the record is load-bearing.
    triples = {
        (p["summary_ready"], p["approved"], p["sync_failed"])
        for p in (payload_ci, payload_noappr, payload_sync)
    }
    assert len(triples) == 3


def test_merge_ready_head_moved_transition_failure_recorded(tmp_path: Path) -> None:
    """Issue #135: PARTIAL_FAILURE during merge_ready head-moved transition must be recorded."""
    from charlie_work.labels import TransitionOutcome

    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            # Return False to simulate add failure (error-as-value)
            return False

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # First approve the PR to set reviewed_head_sha
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # Then simulate head moved by updating the PR head SHA
    fake_gh.pr_head_shas[456] = "sha-different"

    # Now merge_ready should trigger head-moved re-review path
    result = app.merge_ready(456)

    # Head-moved returns ok=False (cannot merge), but label_error is still recorded
    assert result.ok is False
    assert result.data["head_moved"] is True
    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "review_started"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert len(label_error["add_failures"]) > 0


def test_merge_ready_carries_forward_approved_verdict_on_clean_rebase(tmp_path: Path) -> None:
    """Issue #375: a clean rebase with unchanged cumulative patch-id carries the
    approved verdict forward and lets the PR merge once CI/base checks pass."""
    from charlie_work.janitor import _calculate_patch_id

    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    original_diff = (
        "diff --git a/file b/file\n"
        "index 123..456 78910\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+line3\n"
        " line4\n"
    )
    patch_id = _calculate_patch_id(original_diff)
    old_head = "sha-abc123"
    new_head = "sha-rebased123"

    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": old_head,
                "reviewed_patch_id": patch_id,
                "summary": "lgtm",
            }
        ),
        encoding="utf-8",
    )

    # Simulate a rebase-style head move: the new head is not a 2-parent web-flow
    # merge commit, so _verify_synced_head would reject it. The cumulative diff
    # is unchanged, so patch-id carry-forward should keep the approval valid.
    fake_gh.pr_head_shas[456] = new_head
    fake_gh.diffs[456] = original_diff
    fake_gh.compare_overrides[("main", new_head)] = {
        "base_commit": {"sha": fake_gh.base_head_sha},
        "merge_base_commit": {"sha": fake_gh.base_head_sha},
    }
    fake_gh.commits[new_head] = {
        "parents": [{"sha": old_head}],
        "committer": {"login": "someone"},
        "commit": {"committer": {"name": "Not GitHub"}},
    }

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["can_merge"] is True
    assert result.data["merged"] is True
    assert result.data.get("head_moved") is not True
    assert fake_gh.merged == [(456, "squash")]

    decision = json.loads((decision_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert decision["reviewed_head_sha"] == new_head
    assert decision["reviewed_patch_id"] == patch_id
    # Issue #414 (d): the tier-1 fast path is unchanged and tags its own
    # carry-forwards distinctly from tier 2.
    assert decision["carry_forward_tier"] == "patch-id"
    assert decision["carried_forward_from"] == [old_head]

    state = load_state(paths.state_file)
    pr_state = state["prs"]["456"]
    assert pr_state["reviewed_head_sha"] == new_head
    assert pr_state["carry_forward_tier"] == "patch-id"
    assert pr_state["carried_forward_from"] == [old_head]
    # The approval was carried forward (not reset to "reviewing") and the PR
    # proceeded to merge on the same poll.
    assert pr_state["status"] != "reviewing"

    carry_events = [
        e for e in state["events"] if e["kind"] == "verdict_carried_forward_clean_rebase"
    ]
    assert len(carry_events) == 1
    payload = carry_events[0]["payload"]
    assert payload["pr_number"] == 456
    assert payload["issue_number"] == 123
    assert payload["old_reviewed_head_sha"] == old_head
    assert payload["new_head_sha"] == new_head
    assert payload["patch_id"] == patch_id
    assert payload["carried_forward_from"] == [old_head]

    # No review_started transition should fire for a clean rebase.
    assert (123, "agent:reviewing") not in fake_gh.labels_added


def test_merge_ready_changed_patch_id_resets_to_pending(tmp_path: Path) -> None:
    """Issue #375: if the cumulative diff changes, the approval is voided."""
    from charlie_work.janitor import _calculate_patch_id

    config = dataclasses.replace(
        _required_checks_config(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    original_diff = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+original\n"
    )
    changed_diff = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+changed\n"
    )
    patch_id = _calculate_patch_id(original_diff)
    old_head = "sha-abc123"
    new_head = "sha-new-head"

    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": old_head,
                "reviewed_patch_id": patch_id,
                "summary": "lgtm",
            }
        ),
        encoding="utf-8",
    )

    fake_gh.pr_head_shas[456] = new_head
    fake_gh.diffs[456] = changed_diff

    result = app.merge_ready(456)

    assert result.ok is False
    assert result.data["head_moved"] is True
    assert result.data["can_merge"] is False
    assert result.data["merged"] is False
    assert fake_gh.merged == []
    assert (123, "agent:reviewing") in fake_gh.labels_added

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "reviewing"
    assert not any(e["kind"] == "verdict_carried_forward_clean_rebase" for e in state["events"])


def test_merge_ready_missing_patch_id_falls_back_to_pending(tmp_path: Path) -> None:
    """Issue #375: an old approved decision without reviewed_patch_id falls back to
    the legacy head-SHA reset."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    old_head = "sha-abc123"
    new_head = "sha-new-head"

    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": old_head,
                "summary": "lgtm",
            }
        ),
        encoding="utf-8",
    )

    fake_gh.pr_head_shas[456] = new_head
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new\n"
    )

    result = app.merge_ready(456)

    assert result.ok is False
    assert result.data["head_moved"] is True
    assert result.data["can_merge"] is False
    assert fake_gh.merged == []


def test_merge_ready_refuses_when_head_moved_after_approval(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        auto_merge=_approved_automerge(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    fake_gh.prs[0] = {**fake_gh.prs[0], "headRefOid": "sha-new-head"}
    fake_gh.pr_head_shas[456] = "sha-new-head"

    result = app.merge_ready(456, merge=True)

    assert result.ok is False
    assert "PR head moved since approval" in result.message
    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert result.data["head_moved"] is True
    assert fake_gh.merged == []
    assert (123, "agent:reviewing") in fake_gh.labels_added
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "reviewing"
    assert fake_gh.merged_merge_flags == []


def test_merge_ready_head_moved_does_not_stamp_reviewing_when_dispatch_disabled(
    tmp_path: Path,
) -> None:
    """Issue #868: merge_ready()'s head-moved re-review branch is a second,
    independent call site that stamps ``reviewing``/``agent:reviewing`` --
    separate from review()'s own packet-generation path. It must be gated
    the same way: byte-identical scenario to
    test_merge_ready_refuses_when_head_moved_after_approval above, minus the
    enabled flag. The head_moved bookkeeping (state field + event) must
    still happen -- only the reviewing state/label stamp is gated, since
    nothing else keys off head_moved together with status=="reviewing".
    """
    config = OrchestratorConfig(auto_merge=_approved_automerge())  # review_dispatch defaults False
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    fake_gh.prs[0] = {**fake_gh.prs[0], "headRefOid": "sha-new-head"}
    fake_gh.pr_head_shas[456] = "sha-new-head"

    result = app.merge_ready(456, merge=True)

    assert result.ok is False
    assert result.data["merged"] is False
    assert result.data["can_merge"] is False
    assert result.data["head_moved"] is True
    assert fake_gh.merged == []
    assert (123, "agent:reviewing") not in fake_gh.labels_added
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] != "reviewing"
    assert state["prs"]["456"]["head_moved"] is True


def test_merge_ready_merges_when_head_unchanged_after_approval(tmp_path: Path) -> None:
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    # Default config: no --admin
    assert fake_gh.merged_admin_flags == [False]
    assert fake_gh.merged_merge_flags == [()]


def test_merge_ready_legacy_approved_decision_without_head_sha_is_refused(
    tmp_path: Path,
) -> None:
    config = OrchestratorConfig(auto_merge=_approved_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved"}), encoding="utf-8"
    )

    result = app.merge_ready(456, merge=True)

    assert result.ok is False
    assert "PR head moved since approval" in result.message
    assert result.data["head_moved"] is True
    assert result.data["merged"] is False
    assert fake_gh.merged == []
