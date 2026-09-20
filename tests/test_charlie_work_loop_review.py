"""Loop-pass review dispatch and review-decision caching.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from _review_fixtures import (
    _approved_automerge,
    _fake_claude_worker_record,
    _make_loop_app,
    _required_checks_config,
    _write_review_packet,
)
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import AutoMergeConfig, OrchestratorConfig, ReviewDispatchConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_loop_dispatches_reviews_and_evaluates_merge(monkeypatch, tmp_path: Path) -> None:
    """Issue #370: loop() runs dispatch_reviews() and the per-PR merge lane uses the verdict."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        auto_merge=AutoMergeConfig(
            enabled=True,
            strategy="squash",
            delete_branch=True,
            require_approved_review=True,
            required_checks=(),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    prs = [
        {
            "number": 456,
            "title": "Fix #456",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-456-fix",
            "baseRefName": "main",
            "headRefOid": "sha-456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #456",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(tmp_path, 456, "sha-456")

    def fake_launch(
        issue_number: int, branch: str, prompt_text: str, **kwargs: Any
    ) -> ClaudeWorkerRecord:
        # Simulate the reviewer agent writing an approved verdict.
        pr_dir = app.paths.prs / f"pr-{issue_number}"
        decision = {
            "decision": "approved",
            "summary": "lgtm",
            "required_changes": [],
            "reviewed_head_sha": "sha-456",
            "reviewed_patch_id": "",
            "reviewed_at": "2026-07-06T12:00:00Z",
            "pr_number": issue_number,
            "issue_number": 456,
            "escalated": False,
        }
        (pr_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")
        return _fake_claude_worker_record(issue_number, branch)

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.loop(merge=False)

    assert result.ok is True
    assert result.data["dispatch_reviews"]["launched_count"] == 1
    assert len(result.data["merges"]) == 1
    assert result.data["merges"][0]["can_merge"] is True


def test_loop_checks_unavailable_review_lands_in_errors_bucket(tmp_path: Path) -> None:
    """A PR whose review is blocked by checks unavailable must be recorded as an error, not reviewed or merged."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubWithChecksUnavailable(FakeGitHub):
        def pr_checks(self, number: int):
            return None

    fake_gh = FakeGitHubWithChecksUnavailable()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.loop(merge=False)

    assert result.ok is False
    assert result.data["reviews"] == []
    assert result.data["merges"] == []
    assert len(result.data["errors"]) == 1
    assert result.data["errors"][0]["pr"] == 456
    assert "checks unavailable" in result.data["errors"][0]["error"].lower()


def test_loop_zero_checks_pr_does_not_land_in_errors_bucket(tmp_path: Path) -> None:
    """Issue #846: a PR with zero checks (pr_checks() == []) must not be
    counted as a loop-pass error the way checks_unavailable (pr_checks() is
    None) is. This mirrors test_loop_checks_unavailable_review_lands_in_errors_bucket
    above but exercises the other outcome the client boundary must now
    produce: [] means "genuinely no checks yet" (e.g. a conflicted/dirty PR
    CI never ran on), which is a normal review outcome, not an
    infrastructure failure.

    This is a downstream-contract characterization test: it stubs
    GitHub.pr_checks() directly, the same way the sibling test above does, so
    it does not exercise GitHubClient.pr_checks()'s own gh-subprocess
    disambiguation logic -- that regression coverage lives in
    tests/test_github.py (test_pr_checks_zero_checks_returns_empty_list_not_none
    and friends). What this test proves is that once the client returns []
    (as it now correctly does for a zero-check PR), workflow.py's loop() does
    not misclassify it as an error the way it misclassifies None.
    """
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubWithZeroChecks(FakeGitHub):
        def pr_checks(self, number: int):
            return []

    fake_gh = FakeGitHubWithZeroChecks()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.loop(merge=False)

    assert result.data["errors"] == []
    assert result.data["merges"] == []
    assert len(result.data["reviews"]) == 1
    assert result.data["reviews"][0]["pr"] == 456


def test_loop_isolates_per_pr_errors(tmp_path: Path) -> None:
    from charlie_work.github import GitHubError as _GitHubError

    class ExplodingGitHub(FakeGitHub):
        def pr_view(self, number: int):
            raise _GitHubError("merge conflict boom")

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, ExplodingGitHub())

    result = app.loop(limit=0)

    assert result.data["errors"] == [{"pr": 456, "error": "merge conflict boom"}]
    assert result.ok is False


def test_loop_corrupt_review_decision_does_not_crash_or_merge(tmp_path: Path) -> None:
    """A corrupt review-decision.json on the loop path must be treated as a
    non-approval: the loop re-reviews the PR and never attempts to merge."""
    config = OrchestratorConfig(
        auto_merge=_approved_automerge(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text("{truncated", encoding="utf-8")

    result = app.loop(limit=0)

    assert result.ok is True
    assert result.data["merges"] == []
    assert fake_gh.merged == []
    assert len(result.data["reviews"]) == 1


# --- Issue #18: idempotence of ship-it and loop --------------------------------


def test_loop_skips_review_for_approved_unmerged_pr(tmp_path: Path) -> None:
    """A second loop() pass over an approved-but-unmerged PR must NOT rewrite
    the review packet or re-fire label transitions — it should go straight to
    merge_ready."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubListingPRs(FakeGitHub):
        """loop() now runs a reconcile pass (merge-lane-recovery §6-B) that
        calls gh.run(["pr", "list", ...]) via reconcile._fetch_prs. The base
        FakeGitHub.run() generic fallback always returns [] for that query
        regardless of self.prs, which makes detect_drift see an empty GitHub
        snapshot against a non-empty tracked-PR state and misreport PR 456 as
        missing on GitHub. Reflect self.prs for real here so the reconcile
        pass sees the same PR the rest of this fake already knows about."""

        def run(self, args, *, json_output=False, allow_failure=False):
            if args[:2] == ["pr", "list"]:
                return list(self.prs) if json_output else ""
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubListingPRs()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    # Record an approved decision in state (as record_review would).
    state = load_state(paths.state_file)
    state["prs"]["456"] = {
        "number": 456,
        "issue_number": 123,
        "decision": "approved",
        "status": "approved",
        "reviewed_head_sha": "sha-abc123",
    }
    save_state(paths.state_file, state)
    # Also write the decision file so merge_ready can read it.
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.loop(limit=0)

    # review() was skipped — no review packet written, no reviewing label fired.
    assert result.data["reviews"] == []
    # merge_ready was attempted (straight to merge evaluation).
    assert len(result.data["merges"]) == 1
    # The reviewing label must NOT have been re-added (would indicate review() ran).
    assert (123, "agent:reviewing") not in fake_gh.labels_added


def test_loop_pass_refreshes_pr_decision_cache_to_file_value(tmp_path: Path) -> None:
    """Issue #1386: pin the _refresh_pr_decision_cache CALL SITE in loop()'s
    per-PR dispatch block.

    The three behavioral tests above cover the method itself (updates a
    disagreeing tracked PR, no-ops when the cache agrees, skips untracked
    PRs), but none of them exercise the call site -- deleting the
    ``self._refresh_pr_decision_cache(...)`` invocation from ``loop()``'s
    dispatch block would leave the whole suite green. This test closes that
    gap: it seeds a tracked PR whose state-side decision disagrees with its
    flat ``review-decision.json`` (the #1340 divergence shape: state lags a
    concurrent void/record_review), runs one ``loop()`` pass, and asserts
    ``state["prs"][N]`` was reconciled to the file value.

    The test runs in live (non-dry-run) mode. Dry-run would seem isolating
    but is not: ``_refresh_pr_decision_cache`` writes through
    ``self.write_gate.save_state``, which is a no-op under dry-run (the
    WriteGate's strict "zero writes under dry-run" invariant), so the
    refresh's disk write is invisible there. In live mode the merge success
    path (``merge_ready``) DOES write state, but it carries forward the
    existing decision cache fields via ``**state["prs"].get(...)``
    (workflow.py merge_success block) -- it does NOT re-derive
    ``decision``/``reviewed_head_sha``/``decision_path`` from the file. So
    the final state's cache fields are exactly what the refresh wrote: the
    file value if the refresh ran, the stale seed value if it did not.
    Deleting the call site leaves the seeded divergence unreconciled and
    this test fails.
    """
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubListingPRs(FakeGitHub):
        """loop()'s reconcile pass (merge-lane-recovery §6-B) queries
        gh.run(["pr", "list", ...]); the base fake's generic run() fallback
        returns [] regardless of self.prs, which misreports PR 456 as
        missing on GitHub. Reflect self.prs so the reconcile pass sees the
        same PR the rest of this fake knows about. Same override as
        test_loop_skips_review_for_approved_unmerged_pr."""

        def run(self, args, *, json_output=False, allow_failure=False):
            if args[:2] == ["pr", "list"]:
                return list(self.prs) if json_output else ""
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubListingPRs()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Seed a tracked PR whose state-side decision DISAGREES with the file
    # (the #1340 divergence shape: state says "pending" at a stale head while
    # the file has already been reset to "approved" at the live head).
    state = load_state(paths.state_file)
    state["prs"]["456"] = {
        "number": 456,
        "issue_number": 123,
        "status": "reviewing",
        "decision": "pending",
        "reviewed_head_sha": "stale-sha",
        "decision_path": "stale-path",
    }
    save_state(paths.state_file, state)

    # The flat file is authoritative: it says "approved" at the live head
    # (sha-abc123, matching FakeGitHub's default PR head). The file's
    # reviewed_head_sha matches the live head so the already_approved /
    # head_matches gates fire and merge_ready's merge-success path runs --
    # which carries forward the existing cache fields rather than
    # re-deriving them, isolating the refresh as the sole reconciler.
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    decision_path = decision_dir / "review-decision.json"
    decision_path.write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.loop(limit=0)

    # The merge must have run (confirms the per-PR dispatch block reached
    # merge_ready, which is downstream of the refresh call site).
    assert len(result.data["merges"]) == 1
    assert result.data["merges"][0]["merged"] is True

    refreshed = load_state(paths.state_file)["prs"]["456"]
    # The three cache fields must be reconciled to the file value by the
    # refresh -- merge_ready's carry-forward preserves whatever the refresh
    # wrote, so these fail if the refresh call site is deleted.
    assert refreshed["decision"] == "approved"
    assert refreshed["reviewed_head_sha"] == "sha-abc123"
    assert refreshed["decision_path"] == str(decision_path)
    # Non-decision fields survive: issue_number is carried forward by both
    # the refresh and merge_ready's spread. (status becomes "merged" after
    # the merge, which is expected and not a cache field.)
    assert refreshed["issue_number"] == 123


def test_loop_re_reviews_when_head_moved_after_approval(tmp_path: Path) -> None:
    config = dataclasses.replace(
        _required_checks_config(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    # Seed an approved decision pinned to the old head.
    state = load_state(paths.state_file)
    state["prs"]["456"] = {
        "number": 456,
        "issue_number": 123,
        "decision": "approved",
        "status": "approved",
        "reviewed_head_sha": "sha-abc123",
    }
    save_state(paths.state_file, state)
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )
    # New commit pushed after approval.
    fake_gh.prs[0] = {**fake_gh.prs[0], "headRefOid": "sha-new-head"}
    fake_gh.pr_head_shas[456] = "sha-new-head"

    result = app.loop(limit=0)

    assert len(result.data["reviews"]) == 1
    assert result.data["merges"] == []
    assert (123, "agent:reviewing") in fake_gh.labels_added
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "reviewing"


def test_loop_skips_review_and_merges_when_head_unchanged_after_approval(
    tmp_path: Path,
) -> None:
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubListingPRs(FakeGitHub):
        """See the identical override in
        test_loop_skips_review_for_approved_unmerged_pr: loop()'s reconcile
        pass (merge-lane-recovery §6-B) queries gh.run(["pr", "list", ...]),
        and the base fake's generic run() fallback returns [] regardless of
        self.prs, which misreports PR 456 as missing on GitHub."""

        def run(self, args, *, json_output=False, allow_failure=False):
            if args[:2] == ["pr", "list"]:
                return list(self.prs) if json_output else ""
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubListingPRs()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    state = load_state(paths.state_file)
    state["prs"]["456"] = {
        "number": 456,
        "issue_number": 123,
        "decision": "approved",
        "status": "approved",
        "reviewed_head_sha": "sha-abc123",
    }
    save_state(paths.state_file, state)
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.loop(limit=0)

    assert result.data["reviews"] == []
    assert len(result.data["merges"]) == 1
    assert result.data["merges"][0]["merged"] is True
    assert (123, "agent:reviewing") not in fake_gh.labels_added


def test_loop_no_merge_evaluates_readiness_but_skips_gh_merge(tmp_path: Path) -> None:
    """bash-rats --no-merge: the pass reviews and evaluates merge readiness but
    never calls `gh pr merge` — operators sequencing same-surface cascades by
    hand rely on this to dispatch reworks without out-of-order merges."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    state = load_state(paths.state_file)
    state["prs"]["456"] = {
        "number": 456,
        "issue_number": 123,
        "decision": "approved",
        "status": "approved",
        "reviewed_head_sha": "sha-abc123",
    }
    save_state(paths.state_file, state)
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    result = app.loop(limit=0, merge=False)

    assert len(result.data["merges"]) == 1
    assert result.data["merges"][0]["merged"] is False
    assert fake_gh.merged == []


def test_loop_undecided_same_head_skips_review(tmp_path: Path) -> None:
    """Undecided PR with a same-head packet does NOT re-invoke review()."""
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": "sha-same",
        "body": "Closes #123",
        "labels": [],
        "isCrossRepository": False,
    }
    app, fake_gh = _make_loop_app(tmp_path, prs=[pr])

    # Pre-plant a pr.json packet with the same headRefOid as the live PR
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    import json as _json

    (pr_dir / "pr.json").write_text(
        _json.dumps({"number": 456, "headRefOid": "sha-same"}), encoding="utf-8"
    )
    # No review-decision.json → undecided

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    result = app.loop(limit=0)

    assert result.data["skipped_reviews"] == 1
    assert 456 not in review_calls


def test_loop_undecided_head_moved_invokes_review(tmp_path: Path) -> None:
    """Undecided PR whose head has advanced past the packet re-invokes review()."""
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": "sha-new",
        "body": "Closes #123",
        "labels": [],
        "isCrossRepository": False,
    }
    app, fake_gh = _make_loop_app(tmp_path, prs=[pr])

    # Packet has OLD sha — head has moved
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    import json as _json

    (pr_dir / "pr.json").write_text(
        _json.dumps({"number": 456, "headRefOid": "sha-old"}), encoding="utf-8"
    )

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    result = app.loop(limit=0)

    assert result.data["skipped_reviews"] == 0
    assert 456 in review_calls


def test_loop_undecided_same_head_skip_still_merges_on_approved_decision_file(
    tmp_path: Path,
) -> None:
    """Regression for review finding #7: an operator-written decision file
    must not stay invisible until the head moves, even when state.json
    hasn't caught up -- it should proceed straight to merge_ready(), same as
    the decided path.

    Pre-#1362 this was reached via the same-head packet-skip branch (which
    re-checked the decision file directly as a fallback, incrementing
    ``skipped_reviews``). Issue #1362 Stage 1 made ``already_approved``
    itself file-first, so the file-only approval is now caught one branch
    earlier -- ``already_approved`` is True immediately and routes straight
    to ``merge_ready()`` without ever reaching the packet-skip branch, so
    ``skipped_reviews`` stays 0. The observable guarantee this test protects
    (the approval is not invisible; the PR merges) is unchanged; only which
    internal branch reaches it is.
    """
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": "sha-abc123",
        "body": "Closes #123\n\nTests: regression coverage added.",
        "labels": [],
        "isCrossRepository": False,
    }
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [pr]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # State has NO decision recorded yet (undecided from state's perspective).
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    import json as _json

    # Packet head matches the live PR head → same-head skip branch fires.
    (pr_dir / "pr.json").write_text(
        _json.dumps({"number": 456, "headRefOid": "sha-abc123"}), encoding="utf-8"
    )
    # Operator wrote the decision file directly; state.json wasn't updated.
    (pr_dir / "review-decision.json").write_text(
        _json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    result = app.loop(limit=0)

    # Packet regeneration is still skipped (review() never called)...
    assert 456 not in review_calls
    # Issue #1362 Stage 1: already_approved is file-first, so this approval
    # is caught before the packet-skip branch runs at all -- skipped_reviews
    # stays 0 rather than incrementing (see docstring above).
    assert result.data["skipped_reviews"] == 0
    # ...but the approval is not left invisible: merge_ready() fires.
    assert len(result.data["merges"]) == 1
    assert result.data["merges"][0]["merged"] is True


def test_loop_undecided_no_packet_invokes_review(tmp_path: Path) -> None:
    """Undecided PR with no existing packet still invokes review()."""
    pr = {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": "sha-abc",
        "body": "Closes #123",
        "labels": [],
        "isCrossRepository": False,
    }
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    # No packet at all

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]
    result = app.loop(limit=0)

    assert result.data["skipped_reviews"] == 0
    assert 456 in review_calls
