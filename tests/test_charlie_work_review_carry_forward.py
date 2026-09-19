"""Review-queue carry-forward: identical-patch-id verdict carry-forward.

Split out of ``tests/test_charlie_work.py`` (issue #1549, Track-1 wave 3/8):
the patch-id carry-forward half of the ``test_review_queue_*`` seam -- identical-patch-id verdict carry-forward and the tier-2 line-content comparisons. Shared fakes and helpers in ``tests/_review_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from _review_fixtures import _write_review_packet, _review_queue_carry_forward_app
from charlie_work.state import load_state
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_review_queue_carries_forward_approved_on_identical_patch_id(tmp_path: Path) -> None:
    """Issue #411: an approved verdict whose cumulative patch-id is unchanged
    should be carried forward to the new head and not reported as stale."""
    from charlie_work.janitor import _calculate_patch_id

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
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-abc123"
    new_head = "sha-rebased123"
    pr_number = 456
    issue_number = 123

    prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": new_head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _review_queue_carry_forward_app(tmp_path, prs=prs)
    fake_gh = app.gh
    fake_gh.diffs[pr_number] = diff_text

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []

    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == new_head
    assert decision["reviewed_patch_id"] == patch_id
    # Issue #414 (d): the tier-1 fast path is unchanged and tags its own
    # carry-forwards distinctly from tier 2.
    assert decision["carry_forward_tier"] == "patch-id"
    assert old_head in decision["carried_forward_from"]

    state = load_state(app.paths.state_file)
    assert state["prs"][str(pr_number)]["reviewed_head_sha"] == new_head
    assert state["prs"][str(pr_number)]["carried_forward_from"] == [old_head]


def test_review_queue_carries_forward_request_changes_on_identical_patch_id(
    tmp_path: Path,
) -> None:
    """Issue #411: a request_changes verdict is also valid when the patch is identical."""
    from charlie_work.janitor import _calculate_patch_id

    diff_text = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 modified\n"
    )
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-old-head"
    new_head = "sha-sync-merge-head"
    pr_number = 456
    issue_number = 123

    prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": new_head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _review_queue_carry_forward_app(tmp_path, prs=prs)
    fake_gh = app.gh
    fake_gh.diffs[pr_number] = diff_text

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "request_changes",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "carried_forward_from": [],
            # Real (non-placeholder) summary: this test exercises identical-
            # patch-id carry-forward (issue #411), not issue #784's
            # content-free "vacuous" detection -- a content-free verdict
            # would be ineligible for carry-forward entirely, which is
            # covered by its own dedicated tests.
            "summary": "some prior finding that needs a rework brief",
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []

    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == new_head
    assert decision["carried_forward_from"] == [old_head]


def test_review_queue_carries_forward_blocked_on_identical_patch_id(tmp_path: Path) -> None:
    """Issue #413: a blocked verdict whose cumulative patch-id is unchanged
    should be carried forward to the new head and not reported as stale."""
    from charlie_work.janitor import _calculate_patch_id

    diff_text = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 blocked\n"
    )
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-old-head"
    new_head = "sha-sync-merge-head"
    pr_number = 456
    issue_number = 123

    prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": new_head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _review_queue_carry_forward_app(tmp_path, prs=prs)
    fake_gh = app.gh
    fake_gh.diffs[pr_number] = diff_text

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "blocked",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []

    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == new_head
    assert decision["carried_forward_from"] == [old_head]

    state = load_state(app.paths.state_file)
    assert state["prs"][str(pr_number)]["reviewed_head_sha"] == new_head
    assert state["prs"][str(pr_number)]["carried_forward_from"] == [old_head]


def test_review_queue_reports_stale_on_different_patch_id(tmp_path: Path) -> None:
    """Issue #411: a head move that changes the cumulative diff is still stale."""
    from charlie_work.janitor import _calculate_patch_id

    old_diff = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 old\n"
    )
    new_diff = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 new\n"
    )
    old_patch_id = _calculate_patch_id(old_diff)
    new_head = "sha-new-head"
    pr_number = 456
    issue_number = 123

    prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": new_head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _review_queue_carry_forward_app(tmp_path, prs=prs)
    fake_gh = app.gh
    fake_gh.diffs[pr_number] = new_diff

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": "sha-old-head",
            "reviewed_patch_id": old_patch_id,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": "sha-old-head",
            "mergeable": None,
            "mergeStateStatus": "CLEAN",
        }
    ]

    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-old-head"


def test_review_queue_git_failure_falls_back_to_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #411: if git patch-id computation fails, treat the verdict as stale."""
    from charlie_work import workflow as workflow_module

    old_head = "sha-old-head"
    new_head = "sha-new-head"
    pr_number = 456
    issue_number = 123

    prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": new_head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _review_queue_carry_forward_app(tmp_path, prs=prs)
    fake_gh = app.gh
    fake_gh.diffs[pr_number] = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 new\n"
    )

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": "known-patch-id",
            "carried_forward_from": [],
        },
    )

    monkeypatch.setattr(workflow_module, "_calculate_patch_id", lambda _diff: "")

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": old_head,
            "mergeable": None,
            "mergeStateStatus": "CLEAN",
        }
    ]


def test_review_queue_dry_run_skips_carry_forward_write_but_not_stale_check(
    tmp_path: Path,
) -> None:
    """Issue #411: dry-run review-queue must not write but still hide stale verdicts
    when the cumulative patch-id is unchanged."""
    from charlie_work.janitor import _calculate_patch_id

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
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-abc123"
    new_head = "sha-rebased123"
    pr_number = 456
    issue_number = 123

    prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": new_head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _review_queue_carry_forward_app(tmp_path, prs=prs, dry_run=True)
    fake_gh = app.gh
    fake_gh.diffs[pr_number] = diff_text

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "carried_forward_from": [],
        },
    )
    before_decision = (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(
        encoding="utf-8"
    )
    before_state = app.paths.state_file.read_text(encoding="utf-8")

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []
    assert (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(
        encoding="utf-8"
    ) == before_decision
    assert app.paths.state_file.read_text(encoding="utf-8") == before_state


def test_review_queue_carries_forward_on_tier2_line_content_after_main_advance(
    tmp_path: Path,
) -> None:
    """Issue #414: patch-id is unstable across every main advance because the
    merge-base moves, which can shift a hunk's CONTEXT lines even when the
    PR's own +/- content is untouched. Tier 2 recognizes this via the
    ordered +/- line stream and changed-file set, ignoring context drift."""
    from charlie_work.janitor import _calculate_patch_id, _diff_content_signature

    # Reviewed diff: the PR added "+gamma" between unchanged "beta"/"delta".
    reviewed_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " alpha\n"
        " beta\n"
        "+gamma\n"
        " delta\n"
    )
    # Live diff: main advanced and changed the *context* line "beta" ->
    # "beta-updated" between the old and new merge-base. The PR's own
    # "+gamma" contribution is byte-identical and in the same position, but
    # git patch-id --stable hashes context text too, so the cumulative
    # patch-id differs even though nothing the PR authored changed.
    live_diff = (
        "diff --git a/file b/file\n"
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
    live_signature = _diff_content_signature(live_diff)
    assert reviewed_signature == live_signature, "tier-2 signature must ignore context drift"

    old_head = "sha-old-head"
    new_head = "sha-new-head-after-main-advance"
    pr_number = 456
    issue_number = 123

    prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": new_head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _review_queue_carry_forward_app(tmp_path, prs=prs)
    fake_gh = app.gh
    fake_gh.diffs[pr_number] = live_diff

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": reviewed_patch_id,
            "reviewed_changed_lines": list(reviewed_signature.changed_lines),
            "reviewed_changed_files": sorted(reviewed_signature.changed_files),
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []

    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == new_head
    assert decision["carry_forward_tier"] == "line-content"
    assert decision["reviewed_patch_id"] == live_patch_id
    assert old_head in decision["carried_forward_from"]

    state = load_state(app.paths.state_file)
    assert state["prs"][str(pr_number)]["reviewed_head_sha"] == new_head
    assert state["prs"][str(pr_number)]["carry_forward_tier"] == "line-content"
    assert state["prs"][str(pr_number)]["carried_forward_from"] == [old_head]


def test_review_queue_reports_stale_on_reordered_changed_lines(tmp_path: Path) -> None:
    """Issue #414: the same +/- lines in a different ORDER is a real semantic
    change and must not carry forward via tier 2 (ordered, not sorted)."""
    from charlie_work.janitor import _calculate_patch_id

    reviewed_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+first\n"
        "+second\n"
        " delta\n"
    )
    live_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+second\n"
        "+first\n"
        " delta\n"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    old_head = "sha-old-head"
    new_head = "sha-new-head"
    pr_number = 456
    issue_number = 123

    prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": new_head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _review_queue_carry_forward_app(tmp_path, prs=prs)
    fake_gh = app.gh
    fake_gh.diffs[pr_number] = live_diff

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": reviewed_patch_id,
            "reviewed_changed_lines": ["+first", "+second"],
            "reviewed_changed_files": ["file"],
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": old_head,
            "mergeable": None,
            "mergeStateStatus": "CLEAN",
        }
    ]
    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == old_head
    assert "carry_forward_tier" not in decision
