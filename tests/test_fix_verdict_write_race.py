"""Tests for issues #1036, #1038, and #1072: defects in workflow.py's
writes to ``review-decision.json``.

Both sites read a snapshot at time T0 (a PR-head snapshot for #1036, a
``decision`` dict for #1038), do slow work including network round-trips, then
write outside the state lock at time T1. Anything recorded on disk in the
T0..T1 window is silently clobbered.

- #1036 (``review()``'s stale-verdict reset): fails CLOSED. A verdict pinned
  to the genuinely current head was voided because the comparison used a
  stale build-start snapshot instead of the live head. Fixed by re-reading
  the live head immediately before committing the packet's outputs and
  discarding the WHOLE packet (neither prompt nor decision written) if the
  head moved -- not by sparing the reset alone, which would leave a
  current-head verdict sitting next to a prompt describing an older diff
  (see the issue's own corrected analysis, comment 2).

- #1038 (``_update_approval_head``): fails OPEN. A whole-dict replace from
  the caller's pre-network-round-trip copy could resurrect a superseded
  ``approved`` over a concurrent ``request_changes`` and re-pin it to the
  current head, authorizing a merge that should not happen. Fixed by
  re-reading ``review-decision.json`` from disk inside the same
  ``state_lock`` that guards the state.json half of the transition, and
  refusing to write when the on-disk verdict's identity no longer matches
  what the caller observed.

- #1072 (``record_review()``'s compare-and-swap guard): the #1036 guard
  covered only ``review()``'s packet-commit tail exit. Two earlier exits
  (CI-required-check-failure and Tier-1 test-adequacy hard gate) returned
  through ``record_review()`` and bypassed it, pinning verdicts to heads
  that moved mid-build. Fixed by pushing the guard into ``record_review()``
  itself so every caller gets it by construction. The operator CLI
  (``charlie verdict``) is the one exempt caller (issue #467's
  explicit-choice design) and passes ``allow_stale_head=True``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from charlie_work.config import OrchestratorConfig, TestAdequacyConfig
from charlie_work.paths import runtime_paths

from charlie_work.workflow import OrchestratorApp

# Reuse the shared FakeGitHub whose default PR #456 is janitor-green and
# linked to issue #123.
from _fakes_github import FakeGitHub, FakeGitHubWithChecksAndAnnotations


class FakeGitHubHeadMovesOnSecondView(FakeGitHub):
    """Returns ``first_head`` on the first ``pr_view`` call and ``second_head``
    on every call after that.

    ``review()`` calls ``pr_view`` exactly twice on the packet-commit path:
    once at the top of the method (the build-start snapshot) and once
    immediately before committing the packet (the compare-and-swap re-read
    added for issue #1036). This simulates a push landing in between --
    exactly the window the fix closes.
    """

    def __init__(self, *args: Any, first_head: str, second_head: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._first_head = first_head
        self._second_head = second_head
        self._pr_view_calls = 0

    def pr_view(self, number: int):  # type: ignore[override]
        self._pr_view_calls += 1
        pr = super().pr_view(number)
        pr["headRefOid"] = self._first_head if self._pr_view_calls == 1 else self._second_head
        return pr


def test_review_packet_discarded_when_head_moves_during_build(tmp_path: Path) -> None:
    """Issue #1036: a verdict recorded mid-pass at the live head must survive a
    packet build whose snapshot has gone stale.

    ``review()`` snapshots the PR head once, then does several minutes of
    work (diff fetch, janitor, containment check, cross-family review --
    385s observed in production, issue #1036) before committing the packet.
    If the head moves during that window, the packet the build is about to
    commit describes a diff that is no longer current. The fix discards the
    whole packet (neither the prompt nor the decision reset is written) and
    lets the next pass rebuild against the live head -- which, as a
    consequence, leaves an existing verdict pinned to that live head
    completely untouched.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubHeadMovesOnSecondView(first_head="sha-1", second_head="sha-2")
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True)
    decision_path = pr_dir / "review-decision.json"
    # Recorded "mid-build": pinned to sha-2, the head that only becomes live
    # partway through this review() call.
    live_decision = {
        "pr_number": 456,
        "issue_number": 123,
        "decision": "approved",
        "summary": "looks good",
        "required_changes": [],
        "reviewed_head_sha": "sha-2",
    }
    decision_path.write_text(json.dumps(live_decision), encoding="utf-8")

    result = app.review(456)

    assert result.ok is False
    assert result.data["reason"] == "head_moved_during_build"
    assert result.data["snapshot_head_sha"] == "sha-1"
    assert result.data["live_head_sha"] == "sha-2"

    after = json.loads(decision_path.read_text(encoding="utf-8"))
    assert after == live_decision, "a verdict pinned to the live head must not be touched"

    state = json.loads(paths.state_file.read_text(encoding="utf-8"))
    discard_events = [
        e for e in state["events"] if e["kind"] == "review_packet_discarded_head_moved"
    ]
    assert len(discard_events) == 1
    assert discard_events[0]["payload"]["snapshot_head_sha"] == "sha-1"
    assert discard_events[0]["payload"]["live_head_sha"] == "sha-2"
    assert discard_events[0]["payload"]["pr_number"] == 456


def test_update_approval_head_refuses_to_clobber_concurrent_request_changes(
    tmp_path: Path,
) -> None:
    """Issue #1038: a concurrent request_changes landing in the T0..T1 window
    (between a caller reading ``decision`` and ``_update_approval_head``
    writing it back) must survive the carry-forward write, not be silently
    replaced by the caller's stale ``approved`` copy re-pinned to the new
    head.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_number = 456
    issue_number = 123
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True)
    decision_path = pr_dir / "review-decision.json"

    # T0: the caller reads this "approved" decision before its network
    # round-trips (pr_update_branch / _verify_synced_head / diff fetch).
    stale_caller_copy = {
        "pr_number": pr_number,
        "issue_number": issue_number,
        "decision": "approved",
        "summary": "lgtm",
        "required_changes": [],
        "reviewed_head_sha": "old-head",
    }
    decision_path.write_text(json.dumps(stale_caller_copy), encoding="utf-8")

    # T0..T1: a concurrent record_review lands a fresh rejection at the same
    # (still-pinned) head, superseding what the caller read.
    concurrent_rejection = {
        "pr_number": pr_number,
        "issue_number": issue_number,
        "decision": "request_changes",
        "summary": "actually this breaks X",
        "required_changes": ["fix X"],
        "reviewed_head_sha": "old-head",
    }
    decision_path.write_text(json.dumps(concurrent_rejection), encoding="utf-8")

    # T1: the carry-forward call itself, operating on the T0 (pre-rejection)
    # copy it captured before the round-trips.
    applied = app._update_approval_head(
        pr_number,
        stale_caller_copy,
        "new-head",
        old_head="old-head",
        issue_number=issue_number,
        tier="verified-sync",
    )

    assert applied is False, "carry-forward must refuse when the on-disk verdict changed"

    on_disk = json.loads(decision_path.read_text(encoding="utf-8"))
    assert on_disk["decision"] == "request_changes"
    assert on_disk["reviewed_head_sha"] == "old-head"
    assert on_disk["required_changes"] == ["fix X"]

    state = json.loads(paths.state_file.read_text(encoding="utf-8"))
    skip_events = [
        e for e in state["events"] if e["kind"] == "verdict_carry_forward_skipped_stale"
    ]
    assert len(skip_events) == 1
    assert skip_events[0]["payload"]["on_disk_decision"] == "request_changes"
    assert skip_events[0]["payload"]["expected_decision"] == "approved"


def test_update_approval_head_still_applies_when_decision_unchanged(tmp_path: Path) -> None:
    """Sanity check: the identity guard must not block the ordinary case where
    nothing raced. When the on-disk decision still matches what the caller
    read, the carry-forward proceeds exactly as before.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_number = 456
    issue_number = 123
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True)
    decision_path = pr_dir / "review-decision.json"

    decision = {
        "pr_number": pr_number,
        "issue_number": issue_number,
        "decision": "approved",
        "summary": "lgtm",
        "required_changes": [],
        "reviewed_head_sha": "old-head",
    }
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    applied = app._update_approval_head(
        pr_number,
        decision,
        "new-head",
        old_head="old-head",
        issue_number=issue_number,
        tier="verified-sync",
    )

    assert applied is True
    on_disk = json.loads(decision_path.read_text(encoding="utf-8"))
    assert on_disk["decision"] == "approved"
    assert on_disk["reviewed_head_sha"] == "new-head"
    assert on_disk["carried_forward_from"] == ["old-head"]


# --- Issue #1072: record_review() compare-and-swap guard ---------------------


class FakeGitHubHeadMovesWithChecks(FakeGitHubWithChecksAndAnnotations):
    """Like FakeGitHubHeadMovesOnSecondView but with configurable checks.

    Used to test the CI-failure exit in review(), which needs failing
    required checks AND a head that moves between the build-start snapshot
    and record_review()'s live fetch.
    """

    def __init__(
        self,
        *args: Any,
        first_head: str,
        second_head: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._first_head = first_head
        self._second_head = second_head
        self._pr_view_calls = 0

    def pr_view(self, number: int):  # type: ignore[override]
        self._pr_view_calls += 1
        pr = super().pr_view(number)
        pr["headRefOid"] = self._first_head if self._pr_view_calls == 1 else self._second_head
        return pr


def test_record_review_rejects_stale_packet_head_without_allow_stale_head(
    tmp_path: Path,
) -> None:
    """Issue #1072: record_review() must refuse to pin a verdict to a packet
    head that has been superseded by a live head, when the caller is an
    automated path (allow_stale_head defaults to False).

    This is the core fix: pushing the compare-and-swap guard into
    record_review() itself so every automated caller gets it by
    construction, not just the packet-commit exit that #1036 happened to
    guard.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = "diff --git a/file b/file\n+packet diff"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Generate a packet at head "sha-abc123".
    review_result = app.review(456)
    assert review_result.ok is True

    # Simulate a new commit landing after the packet was generated.
    fake_gh.pr_head_shas[456] = "sha-new789"

    decision_path = paths.prs / "pr-456" / "review-decision.json"

    # An automated caller passing the stale packet head must be rejected.
    result = app.record_review(
        456,
        "approved",
        summary="lgtm",
        reviewed_head="sha-abc123",
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok is False
    assert result.data["reason"] == "head_moved_during_build"
    assert result.data["packet_head_sha"] == "sha-abc123"
    assert result.data["live_head_sha"] == "sha-new789"
    # The pending decision written by review() must be untouched.
    assert json.loads(decision_path.read_text(encoding="utf-8")).get("decision") == "pending"


def test_record_review_allows_stale_packet_head_with_allow_stale_head(
    tmp_path: Path,
) -> None:
    """Issue #1072: the operator CLI exemption. allow_stale_head=True
    preserves issue #467's explicit-choice design — a human deliberately
    pinning a verdict to a superseded head is allowed, because the failure
    direction is toward redundant work (re-review next pass), never toward
    authorizing a merge on the wrong head.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = "diff --git a/file b/file\n+packet diff"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    review_result = app.review(456)
    assert review_result.ok is True

    fake_gh.pr_head_shas[456] = "sha-new789"

    # The operator CLI passes allow_stale_head=True and is allowed to proceed.
    result = app.record_review(
        456,
        "approved",
        summary="lgtm",
        reviewed_head="sha-abc123",
        allow_stale_head=True,
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-abc123"
    assert decision["reviewed_head_source"] == "packet"


def test_record_review_allows_live_head_when_packet_stale(tmp_path: Path) -> None:
    """Issue #1072: choosing the live head (which has moved past the packet)
    is always allowed — the verdict is pinned to the current head, not a
    stale one. The guard only rejects pinning to a *superseded* head.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = "diff --git a/file b/file\n+packet diff"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    review_result = app.review(456)
    assert review_result.ok is True

    fake_gh.pr_head_shas[456] = "sha-new789"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+unreviewed change"

    # Choosing the live head is fine — no allow_stale_head needed.
    result = app.record_review(
        456,
        "approved",
        summary="lgtm",
        reviewed_head="sha-new789",
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-new789"
    assert decision["reviewed_head_source"] == "live"


def test_review_ci_failure_exit_rejects_verdict_when_head_moved(
    tmp_path: Path,
) -> None:
    """Issue #1072: the CI-required-check-failure exit in review() routes
    through record_review() before the #1036 packet-commit guard. When the
    head moves between the build-start snapshot and record_review()'s live
    fetch, the verdict must not be recorded.

    This exit fires before the packet is written in the current pass, so
    packet_head_sha comes from a *previous* packet. We plant one with the
    snapshot head so the new guard in record_review() can detect the
    mismatch.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            enabled=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubHeadMovesWithChecks(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "databaseId": 9002},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        annotations_by_check_run_id={},
        first_head="sha-1",
        second_head="sha-2",
    )
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Plant a previous packet at the snapshot head so packet_head_sha == "sha-1".
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps({"headRefOid": "sha-1", "number": 456}), encoding="utf-8"
    )

    result = app.review(456)

    assert result.ok is False
    assert result.data["reason"] == "head_moved_during_build"
    assert result.data["packet_head_sha"] == "sha-1"
    assert result.data["live_head_sha"] == "sha-2"

    # No verdict was recorded.
    decision_path = pr_dir / "review-decision.json"
    assert not decision_path.exists()


def test_review_test_adequacy_exit_rejects_verdict_when_head_moved(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Issue #1072: the Tier-1 test-adequacy hard gate exit in review() also
    routes through record_review() before the #1036 guard. When the head
    moves between the build-start snapshot and record_review()'s live fetch,
    the verdict must not be recorded.

    Unlike the CI-failure exit, this exit fires AFTER the packet is written
    in the current pass, so packet_head_sha is the snapshot head from the
    just-written pr.json — no manual packet planting needed.
    """
    from charlie_work.config import ReviewConfig
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict

    config = OrchestratorConfig(
        test_adequacy=TestAdequacyConfig(enabled=True, exempt_marker="Test-exempt:"),
        review=ReviewConfig(max_rework_cycles=2),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubHeadMovesOnSecondView(first_head="sha-1", second_head="sha-2")
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    hard_fail_verdict = TestAdequacyVerdict(
        ok=False,
        failures=("Product code changed (15 LOC added) but no test files changed.",),
        warnings=(),
        facts=TestAdequacyFacts(
            added_product_loc=15,
            added_test_loc=0,
            assertion_count=0,
            test_files_changed=0,
            untested_product_files=("src/foo.py",),
            exempt=False,
            exempt_reason="",
        ),
    )
    monkeypatch.setattr(
        "charlie_work.workflow.check_test_adequacy",
        lambda diff, pr, cfg: hard_fail_verdict,
    )

    result = app.review(456)

    assert result.ok is False
    assert result.data["reason"] == "head_moved_during_build"
    assert result.data["packet_head_sha"] == "sha-1"
    assert result.data["live_head_sha"] == "sha-2"

    # The packet was written (pr.json exists) but no verdict was recorded.
    pr_dir = paths.prs / "pr-456"
    assert (pr_dir / "pr.json").exists()
    decision_path = pr_dir / "review-decision.json"
    if decision_path.exists():
        assert json.loads(decision_path.read_text(encoding="utf-8")).get("decision") == "pending"
