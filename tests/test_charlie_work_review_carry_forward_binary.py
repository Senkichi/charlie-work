"""Review-queue carry-forward: file-set, binary, and rename edges.

Split out of ``tests/test_charlie_work.py`` (issue #1549, Track-1 wave 3/8):
the file-set/binary half of the ``test_review_queue_*`` carry-forward seam -- changed-file-set, binary-content, and rename edge cases, plus the carry-forward event payload. Shared fakes and helpers in ``tests/_review_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from _fakes_github import FakeGitHub
from _review_fixtures import _write_review_packet, _review_queue_carry_forward_app
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_review_queue_reports_stale_on_changed_file_set(tmp_path: Path) -> None:
    """Issue #414: an identical line stream but a different set of changed
    files (a file added/removed) is a real change and must not carry
    forward via tier 2."""
    from charlie_work.janitor import _calculate_patch_id

    reviewed_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+gamma\n"
        " delta\n"
    )
    live_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+gamma\n"
        " delta\n"
        "diff --git a/other b/other\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/other\n"
        "@@ -0,0 +1,1 @@\n"
        "+extra file\n"
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
            "reviewed_changed_lines": ["+gamma"],
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


def test_review_queue_git_failure_in_tier2_falls_back_to_stale(tmp_path: Path) -> None:
    """Issue #414: if the live diff cannot be fetched at all (gh/git failure),
    tier 2 must fail closed to stale even when a tier-2 baseline is recorded
    — never carry forward on uncertainty."""
    from charlie_work.janitor import _calculate_patch_id

    class FakeGitHubEmptyDiff(FakeGitHub):
        """Simulates a gh/git failure: pr_diff always returns empty."""

        def pr_diff(self, number: int) -> str:
            return ""

    reviewed_diff = (
        "diff --git a/file b/file\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+gamma\n"
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
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHubEmptyDiff()
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": reviewed_patch_id,
            "reviewed_changed_lines": ["+gamma"],
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


def test_review_queue_reports_stale_on_mixed_binary_content_change(tmp_path: Path) -> None:
    """Issue #414 (review follow-up): a binary file's payload emits no +/-
    lines, so a text hunk that's byte-identical alongside a binary asset
    whose content genuinely changed (different git index blob hashes) must
    NOT carry forward via tier 2 — the signature can't see binary content
    and must fail closed instead of silently ignoring it."""
    from charlie_work.janitor import _calculate_patch_id, _diff_content_signature

    reviewed_diff = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+gamma\n"
        " delta\n"
        "diff --git a/logo.png b/logo.png\n"
        "index aaa1111..bbb2222 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    live_diff = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+gamma\n"
        " delta\n"
        "diff --git a/logo.png b/logo.png\n"
        "index ccc3333..ddd4444 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    live_patch_id = _calculate_patch_id(live_diff)
    assert reviewed_patch_id != live_patch_id, "test fixture must have differing patch-ids"
    reviewed_signature = _diff_content_signature(reviewed_diff)
    live_signature = _diff_content_signature(live_diff)
    # The bug this test guards against: without the has_binary gate, these
    # signatures compare equal despite genuinely different binary content.
    assert reviewed_signature.changed_lines == live_signature.changed_lines
    assert reviewed_signature.changed_files == live_signature.changed_files
    assert reviewed_signature.has_binary is True
    assert live_signature.has_binary is True

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
            "reviewed_changed_lines": list(reviewed_signature.changed_lines),
            "reviewed_changed_files": sorted(reviewed_signature.changed_files),
            "reviewed_has_binary": reviewed_signature.has_binary,
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


def test_review_queue_reports_stale_on_binary_only_content_change(tmp_path: Path) -> None:
    """Issue #414 (review follow-up): a pure binary-only diff (no hunks at
    all, so patch-id is empty on both sides) whose binary content genuinely
    changed must still report stale, not silently carry forward through the
    tier-2 signature fields (which never had any content to compare)."""
    from charlie_work.janitor import _calculate_patch_id, _diff_content_signature

    reviewed_diff = (
        "diff --git a/logo.png b/logo.png\n"
        "index aaa1111..bbb2222 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    live_diff = (
        "diff --git a/logo.png b/logo.png\n"
        "index ccc3333..ddd4444 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    assert reviewed_patch_id == "", "a hunk-less binary diff must not produce a patch-id"
    reviewed_signature = _diff_content_signature(reviewed_diff)

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
            "reviewed_changed_lines": list(reviewed_signature.changed_lines),
            "reviewed_changed_files": sorted(reviewed_signature.changed_files),
            "reviewed_has_binary": reviewed_signature.has_binary,
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


def test_review_queue_carries_forward_identical_mixed_binary_and_text_via_tier1(
    tmp_path: Path,
) -> None:
    """Issue #414 (review follow-up): a byte-identical mixed text+binary diff
    (same index hash, same text) still carries forward — via tier 1's
    patch-id match, since tier 2 is never reached when tier 1 already
    succeeds. Confirms the has_binary gate does not regress the ordinary
    identical-content case."""
    from charlie_work.janitor import _calculate_patch_id

    diff_text = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,2 +1,3 @@\n"
        " alpha\n"
        "+gamma\n"
        " delta\n"
        "diff --git a/logo.png b/logo.png\n"
        "index aaa1111..bbb2222 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    patch_id = _calculate_patch_id(diff_text)
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
    assert decision["carry_forward_tier"] == "patch-id"


def test_review_queue_stays_stale_with_empty_patch_id_from_rename(
    tmp_path: Path,
) -> None:
    """Issue #414 (review follow-up, deliberately NOT fixed): a pure-rename
    (100% similarity, no hunk) diff has no patch-id at all, even though it
    has a valid tier-2 signature on file. Eligibility for both tiers gates
    on ``reviewed_patch_id`` being recorded (matching #412 exactly) — an
    earlier attempt to gate on the signature fields' presence instead was
    reverted because ``record_review`` unconditionally records a signature
    (possibly trivially empty) for every approved/request_changes decision,
    which made unrelated no-op/placeholder diffs look like valid tier-2
    baselines and wrongly carried forward verdicts whose head had actually
    moved to different content (test_merge_ready_refuses_when_head_moved_
    after_approval and siblings). This case stays conservatively stale;
    tracked as a narrow follow-up rather than fixed here."""
    from charlie_work.janitor import _calculate_patch_id, _diff_content_signature

    rename_diff = (
        "diff --git a/old.py b/new.py\n"
        "similarity index 100%\n"
        "rename from old.py\n"
        "rename to new.py\n"
    )
    reviewed_patch_id = _calculate_patch_id(rename_diff)
    assert reviewed_patch_id == "", "a hunk-less rename diff must not produce a patch-id"
    reviewed_signature = _diff_content_signature(rename_diff)
    assert reviewed_signature.changed_files == frozenset({"new.py"})

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
    fake_gh.diffs[pr_number] = rename_diff

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
            "reviewed_has_binary": reviewed_signature.has_binary,
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


def test_review_queue_carry_forward_records_event(tmp_path: Path) -> None:
    """Issue #638: the ``review_queue()`` carry-forward path (one of the three
    previously-silent call sites) must record a carry-forward event, not just
    mutate ``review-decision.json`` and ``state.json``."""
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
    app.gh.diffs[pr_number] = diff_text

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

    state = load_state(app.paths.state_file)
    carry_events = [
        e for e in state["events"] if e["kind"] == "verdict_carried_forward_clean_rebase"
    ]
    assert len(carry_events) == 1
    payload = carry_events[0]["payload"]
    assert payload["pr_number"] == pr_number
    assert payload["issue_number"] == issue_number
    assert payload["old_reviewed_head_sha"] == old_head
    assert payload["new_head_sha"] == new_head
    assert payload["carry_forward_tier"] == "patch-id"
