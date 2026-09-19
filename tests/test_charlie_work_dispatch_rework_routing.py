"""Rework-dispatch head-moved routing: relaunch vs review on PR head drift.

Split out of ``tests/test_charlie_work.py`` (issue #1547, Track-1 wave 1/8):
the ``test_dispatch_rework_*`` seam's routing half -- when a rework PR's
live head moved since the reviewed SHA, the dispatcher routes to review or
relaunches by mergeable state, sync-merge provenance, janitor blocks, and
missing head/patch-id signals. Shared fakes/helpers in
``tests/_rework_dispatch_fixtures.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import (
    _dispatch_rework_config,
)
from charlie_work.config import ReviewDispatchConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp


def test_dispatch_rework_routes_to_review_instead_of_relaunch_when_head_moved(
    tmp_path: Path,
) -> None:
    """Issue #339: a rework worker relaunched onto a PR whose rework was
    already pushed (head moved past the last request_changes verdict) finds
    nothing to do, idles, and gets watchdog-reaped, burning a session and a
    concurrency slot. dispatch_rework must detect the head-moved-with-real-
    content-change case and route the issue to the review lane instead of
    launching a redundant worker (acceptance criterion 1).
    """
    config = dataclasses.replace(
        _dispatch_rework_config(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record a request_changes decision: this puts the issue into
    # rework_requested and pins reviewed_head_sha/reviewed_patch_id to the
    # current (pre-rework) head/diff.
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+first"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        assert state["issues"]["123"]["status"] == "rework_requested"
        assert state["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"

    # Simulate the rework already having been pushed: head advances AND the
    # diff content genuinely changes (not just a sync-merge).
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+second"
    )

    # A rework prompt exists — absent the fix, this is exactly what lets
    # dispatch proceed and launch a redundant worker.
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["routed_to_review"] == [123]
    assert result.data["skipped_head_indeterminate"] == []
    # No rework worker was launched for issue 123.
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    # Routed to the review lane instead: needs_rework cleared, reviewing/pr_open added.
    assert (123, "agent:needs-rework") in fake_gh.labels_removed
    assert (123, "agent:reviewing") in fake_gh.labels_added

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "reviewing"
    assert any(e["kind"] == "rework_already_pushed" for e in state["events"])


def test_dispatch_rework_launches_for_conflicted_pr_with_unknown_mergeable_pr_view(
    tmp_path: Path,
) -> None:
    """Issue #1349 pr_view fallback: pr_list's ``mergeable`` can be UNKNOWN
    (GitHub computes it asynchronously). When pr_list is indeterminate, the
    #339 filter must re-check with a fresh pr_view before routing to review,
    so a persistently-conflicting PR whose conflict only shows on pr_view is
    still dispatched rather than deadlocked.
    """

    class PrViewConflictingGitHub(FakeGitHub):
        def pr_view(self, number: int):
            pr_copy = dict(super().pr_view(number))
            # pr_list reports UNKNOWN; the authoritative pr_view reveals
            # the conflict.
            pr_copy["mergeable"] = "CONFLICTING"
            pr_copy["mergeStateStatus"] = "DIRTY"
            return pr_copy

    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = PrViewConflictingGitHub()
    # pr_list's mergeable is UNKNOWN (not a definite CONFLICTING/MERGEABLE)
    # and mergeStateStatus is CLEAN, so the pr_view fallback path is taken.
    fake_gh.prs[0]["mergeable"] = "UNKNOWN"
    fake_gh.prs[0]["mergeStateStatus"] = "CLEAN"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+first"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+second"
    )

    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["routed_to_review"] == []
    assert result.data["sessions"][0]["issue_number"] == 123
    state = load_state(paths.state_file)
    assert not any(e["kind"] == "rework_already_pushed" for e in state["events"])


def test_dispatch_rework_launches_when_head_matches_reviewed_sha(tmp_path: Path) -> None:
    """Regression pin (issue #339 acceptance criterion 2): dispatch_rework
    must still launch exactly as before when the PR head is unchanged since
    the request_changes verdict — the rework is genuinely outstanding.
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Head is unchanged (still the default "sha-abc123") — genuinely outstanding.
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["routed_to_review"] == []
    assert result.data["sessions"][0]["issue_number"] == 123
    assert (123, "agent:in-progress") in fake_gh.labels_added


def test_dispatch_rework_skips_without_stranding_when_head_indeterminate(
    tmp_path: Path,
) -> None:
    """Issue #339 fail-safe direction: if content identity can't be
    established after a head change (diff fetch fails), dispatch_rework must
    not launch a redundant worker, but must also not strand the issue — it
    stays rework_requested so the next pass retries (acceptance: fail-closed
    without permanent stranding).
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Head moves, but the diff fetch now fails — GitHub.pr_diff's real
    # allow_failure=True contract returns "" on failure, so an empty diff is
    # the correct fake-adapter stand-in for "gh pr diff failed".
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = ""

    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["routed_to_review"] == []
    assert result.data["skipped_head_indeterminate"] == [123]
    # Not stranded: still rework_requested so the next pass retries.
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    # No launch, and no review-lane relabeling either — genuinely indeterminate.
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    assert (123, "agent:reviewing") not in fake_gh.labels_added


def test_dispatch_rework_launches_when_head_moved_by_sync_merge_only(tmp_path: Path) -> None:
    """Issue #339 acceptance: a sync-merge-only head advance (base merged into
    the PR branch moves headRefOid, but the PR's own patch content is
    unchanged) must NOT be treated as "already reworked" — the same patch
    still needs a genuine rework cycle, so dispatch_rework must still launch.

    Regression coverage for a reviewer-caught gap: mutating away the
    same-patch-id carve-out (routing to review on ANY head mismatch,
    regardless of patch-id) left all pre-existing dispatch_rework tests
    green, because none of them exercised a moved-head/same-patch-id PR —
    exactly the common case on a fleet where every open PR gets synced with
    main constantly.
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    diff_text = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    fake_gh.diffs[456] = diff_text
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        assert state["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"
        assert state["prs"]["456"]["reviewed_patch_id"]

    # Simulate a sync-merge: base merged into the branch moves the head SHA
    # (FakeGitHub.pr_update_branch models this exactly), but the PR's diff
    # content is unchanged — a real sync merge does not touch the patch.
    fake_gh.pr_update_branch(456)
    new_head = fake_gh.prs[0]["headRefOid"]
    assert new_head != "sha-abc123"
    fake_gh.pr_head_shas[456] = new_head
    fake_gh.diffs[456] = diff_text

    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    fake_gh.labels_added.clear()

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["routed_to_review"] == []
    assert result.data["review_blocked_retry"] == []
    assert result.data["sessions"][0]["issue_number"] == 123
    assert (123, "agent:in-progress") in fake_gh.labels_added


def test_dispatch_rework_dispatches_conflicted_pr_with_advanced_head_no_deadlock(
    tmp_path: Path,
) -> None:
    """Issue #1349: a candidate whose PR head moved with a real content
    change (live_patch_id != reviewed_patch_id) AND whose PR is
    CONFLICTING/DIRTY must receive a rework worker, not be routed to
    review(). Previously the #339 "already pushed" filter routed such a PR
    to review(), whose janitor gate bounced it back to rework_requested
    without writing a packet or touching reviewed_head_sha -- deadlocking
    the issue between dispatch_rework and review() forever (the only exit
    being the #765 stall escalation to a human, not a dispatch). The
    "already pushed" inference is unsound in exactly this state: whatever
    was pushed since the last verdict did NOT resolve the conflict the
    rework was requested for, so the rework is still outstanding.

    The desync guard the original #339 finding 1 test exercised (don't
    force-flip status to "reviewing" when review() blocks without a packet)
    is covered for the non-conflict case by
    test_dispatch_rework_head_moved_but_review_blocked_by_janitor_does_not_flip_to_reviewing.
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+first"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Head advances with a real content change (the #339 "already pushed"
    # condition) AND the PR is now conflicting: the advance did not resolve
    # the conflict, so the rework is still outstanding.
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.prs[0]["mergeable"] = "CONFLICTING"
    fake_gh.prs[0]["mergeStateStatus"] = "DIRTY"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+second"
    )

    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    result = app.dispatch_rework()

    assert result.ok is True
    # A rework worker is dispatched -- the conflicted PR is kept as a
    # legitimate launch candidate, not routed to the review lane.
    assert result.data["selected_count"] == 1
    assert result.data["routed_to_review"] == []
    assert result.data["review_blocked_retry"] == []
    assert result.data["sessions"][0]["issue_number"] == 123
    assert (123, "agent:in-progress") in fake_gh.labels_added

    # The rework_already_pushed -> janitor-blocked -> rework_already_pushed
    # cycle cannot recur: no rework_already_pushed event fires because the
    # issue was never misrouted to review().
    state = load_state(paths.state_file)
    assert not any(e["kind"] == "rework_already_pushed" for e in state["events"])


def test_dispatch_rework_head_moved_but_review_blocked_by_janitor_does_not_flip_to_reviewing(
    tmp_path: Path,
) -> None:
    """Issue #339 finding 1 (non-conflict regression guard): a candidate whose
    PR head moved with a real content change gets routed to review() — but if
    review()'s deterministic janitor gate returns ok=False *before* writing a
    packet or firing the review_started transition (here: a draft PR whose
    ``gh pr ready`` fails), the routing helper must NOT force-flip the issue's
    status to "reviewing". Doing so would desync state.json from GitHub
    reality (labels still say needs-rework, no packet exists) and strand the
    issue outside dispatch_rework's own candidate pool forever, with no
    automated recovery path. The issue must stay rework_requested so the next
    pass retries.

    This is the non-conflict counterpart to
    test_dispatch_rework_dispatches_conflicted_pr_with_advanced_head_no_deadlock:
    that test covers the #1349 conflict-bypass (the PR never reaches
    _route_rework_candidate_to_review); this test covers the #339 finding 1
    desync guard (the PR DOES reach _route_rework_candidate_to_review, but
    review() blocks without a packet). A draft PR is used as the non-conflict
    janitor-block trigger so the #1349 conflict-bypass does not intercept it.
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # ``gh pr ready`` must fail so the draft parks as janitor_blocked and
    # review() returns ok=False without writing a packet. If it succeeded,
    # the PR would be readied and the block would be transient (deferred to
    # the next pass), not a durable janitor_blocked state.
    fake_gh.pr_ready_ok = False
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+first"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Head advances with a real content change (routes to review) AND the PR
    # is now a draft (janitor blocks review() before any packet/label write).
    # mergeable=MERGEABLE so the #1349 conflict-bypass does not intercept
    # this candidate — it must reach _route_rework_candidate_to_review.
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.prs[0]["isDraft"] = True
    fake_gh.prs[0]["mergeable"] = "MERGEABLE"
    fake_gh.prs[0]["mergeStateStatus"] = "CLEAN"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+second"
    )

    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    # Not reported as successfully routed — review() never produced a packet.
    assert result.data["routed_to_review"] == []
    assert result.data["review_blocked_retry"] == [123]

    # No label churn at all: neither a launch nor a review_started transition.
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    assert (123, "agent:reviewing") not in fake_gh.labels_added
    assert (123, "agent:needs-rework") not in fake_gh.labels_removed

    # Not stranded: still rework_requested so the next pass retries.
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    # PR record shows the janitor block, not a flipped "reviewing" state.
    assert state["prs"]["456"]["status"] == "janitor_blocked"
    assert any(
        e["kind"] == "rework_already_pushed" and e["payload"].get("routed") is False
        for e in state["events"]
    )


def test_dispatch_rework_skips_when_live_head_ref_oid_missing(tmp_path: Path) -> None:
    """Non-blocking coverage: live_head_sha itself (not just a diff-fetch
    failure) can be unavailable — pr_list() returning a record with no
    headRefOid. Must fail closed the same as any other indeterminate case.
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Live head is unavailable from the PR list response.
    fake_gh.prs[0]["headRefOid"] = None

    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["skipped_head_indeterminate"] == [123]
    assert result.data["routed_to_review"] == []
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"


def test_dispatch_rework_skips_when_reviewed_patch_id_missing(tmp_path: Path) -> None:
    """Non-blocking coverage: an older/malformed pr_state that recorded
    reviewed_head_sha without reviewed_patch_id must also fail closed on a
    head mismatch rather than guessing at content identity.
    """
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
            # reviewed_patch_id deliberately absent (older/malformed record).
        }
        save_state(paths.state_file, state)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    # Head has moved relative to the recorded reviewed_head_sha.
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert result.data["skipped_head_indeterminate"] == [123]
    assert result.data["routed_to_review"] == []
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
