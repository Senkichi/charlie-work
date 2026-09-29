"""Review-queue body-drift guard for ``request_changes`` verdicts (issue #1983).

The carry-forward / stranded-reroute half of the #1939 body-change signal:
when a ``request_changes`` verdict's required fix lived in the PR body and
the live body now differs from the verdict's ``reviewed_body_sha256``
baseline, ``review_queue()`` must queue a FRESH review (never auto-approve)
instead of carrying the verdict forward onto a content-identical head or
re-routing it to rework. Before this fix a body-only rework looped through
``dispatch_rework`` until the redispatch cap escalated the issue (job-cannon
PR #2201). Shared fakes and helpers in ``tests/_review_fixtures.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _review_fixtures import _stale_ci_pr, _write_review_packet, _review_queue_app
from charlie_work.instrumentation import query_events
from charlie_work.state import load_state, save_state
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def _body_sha256(body: str | None) -> str:
    """The ``reviewed_body_sha256`` wire contract (issue #1939): SHA-256 of the
    PR body with CRLF/CR line endings normalized to LF, ``None`` treated as
    the empty string."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Same patch-id-carry-forward diff shape as
# test_charlie_work_review_stale_ci.py: tier-1 matches when ``reviewed_patch_id``
# equals ``_calculate_patch_id`` of this text and no tier-2 signature fields are
# recorded (the legacy-decision path in ``_check_carry_forward``).
_DIFF = (
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

_PR_NUMBER = 456
_ISSUE_NUMBER = 123
_OLD_BODY = "Closes #123.\n\nStale description the verdict was recorded against."
_NEW_BODY = "Closes #123.\n\nTests: corrected description after body-only rework."


def _request_changes_decision(
    reviewed_head_sha: str,
    *,
    reviewed_body_sha256: str | None = _body_sha256(_OLD_BODY),
    patch_id: str | None = None,
) -> dict:
    decision = {
        "decision": "request_changes",
        "escalated": False,
        "reviewed_head_sha": reviewed_head_sha,
        "carried_forward_from": [],
        "required_changes": ["Update the PR body to reflect the current head."],
        "summary": "the PR description misstates the change; fix the body text",
    }
    if patch_id is not None:
        decision["reviewed_patch_id"] = patch_id
    if reviewed_body_sha256 is not None:
        decision["reviewed_body_sha256"] = reviewed_body_sha256
    return decision


def test_carry_forward_skipped_when_request_changes_body_drifted(tmp_path: Path) -> None:
    """Issue #1983 AC1: a request_changes verdict, a patch-id-identical new
    head, and a changed live body must NOT carry forward -- the verdict's
    only finding was already satisfied by the body edit, which no diff-based
    tier can see. The PR falls through to the stale-queue path (a fresh
    review supersedes the verdict, never auto-approved), no
    ``verdict_carried_forward_*`` event fires, and no
    ``stranded_request_changes_rework_requested`` re-routing happens."""
    from charlie_work.janitor import _calculate_patch_id

    old_head = "sha-abc123"
    new_head = "sha-rebased123"
    patch_id = _calculate_patch_id(_DIFF)
    pr = _stale_ci_pr(_PR_NUMBER, _ISSUE_NUMBER, new_head)
    pr["body"] = _NEW_BODY

    app = _review_queue_app(tmp_path, prs=[pr], dry_run=False)
    app.gh.diffs[_PR_NUMBER] = _DIFF
    _write_review_packet(
        tmp_path,
        _PR_NUMBER,
        new_head,
        _request_changes_decision(old_head, patch_id=patch_id),
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": _PR_NUMBER,
            "issue": _ISSUE_NUMBER,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": old_head,
            "mergeable": None,
            "mergeStateStatus": "CLEAN",
        }
    ]

    # The decision file must NOT have been carried forward onto the new head.
    decision = json.loads(
        (app.paths.prs / f"pr-{_PR_NUMBER}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == old_head
    assert "carry_forward_tier" not in decision

    assert query_events(app.paths.state_file, kind="verdict_carried_forward_clean_rebase") == []
    assert query_events(app.paths.state_file, kind="verdict_carried_forward_line_content") == []
    assert (
        query_events(app.paths.state_file, kind="stranded_request_changes_rework_requested") == []
    )
    requeue_events = query_events(
        app.paths.state_file, kind="request_changes_body_changed_requeued"
    )
    assert len(requeue_events) == 1
    assert requeue_events[0]["payload"]["pr_number"] == _PR_NUMBER
    assert requeue_events[0]["payload"]["issue_number"] == _ISSUE_NUMBER
    assert requeue_events[0]["payload"]["reviewed_head_sha"] == old_head


def test_same_head_body_drift_queues_review_not_rework(tmp_path: Path) -> None:
    """Issue #1983 AC2: a body-only rework with no push leaves the head
    unchanged, so the verdict is "reviewed at live head". The same-head
    branch must queue a fresh review rather than fire
    ``_reroute_stranded_request_changes`` -- re-driving rework on a verdict
    whose body finding is already fixed is exactly the loop the issue
    reports."""
    head = "sha-live-head"
    pr = _stale_ci_pr(_PR_NUMBER, _ISSUE_NUMBER, head)
    pr["body"] = _NEW_BODY

    app = _review_queue_app(tmp_path, prs=[pr], dry_run=False)
    _write_review_packet(tmp_path, _PR_NUMBER, head, _request_changes_decision(head))

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": _PR_NUMBER,
            "issue": _ISSUE_NUMBER,
            "packet_head_sha": head,
            "decision": "stale",
            "reviewed_head_sha": head,
            "mergeable": None,
            "mergeStateStatus": "CLEAN",
        }
    ]

    state = load_state(app.paths.state_file)
    assert state["issues"].get(str(_ISSUE_NUMBER), {}).get("status") != "rework_requested"
    assert (
        query_events(app.paths.state_file, kind="stranded_request_changes_rework_requested") == []
    )
    assert not (app.paths.prs / f"pr-{_PR_NUMBER}" / "rework-prompt.md").exists()
    assert (_ISSUE_NUMBER, app.config.labels.needs_rework) not in app.gh.labels_added

    requeue_events = query_events(
        app.paths.state_file, kind="request_changes_body_changed_requeued"
    )
    assert len(requeue_events) == 1


def test_same_head_body_drift_requeue_deduped(tmp_path: Path) -> None:
    """``review_queue()`` runs multiple times per loop pass (issue #1120), so
    the ``request_changes_body_changed_requeued`` emission must dedup like
    ``stale_ci_verdict_requeued``: queued on every call (dispatch dedup is
    separate) but emitted once per live head+body combination."""
    head = "sha-live-head"
    pr = _stale_ci_pr(_PR_NUMBER, _ISSUE_NUMBER, head)
    pr["body"] = _NEW_BODY

    app = _review_queue_app(tmp_path, prs=[pr], dry_run=False)
    _write_review_packet(tmp_path, _PR_NUMBER, head, _request_changes_decision(head))

    first = app.review_queue()
    second = app.review_queue()

    assert len(first.data["queue"]) == 1
    assert len(second.data["queue"]) == 1
    requeue_events = query_events(
        app.paths.state_file, kind="request_changes_body_changed_requeued"
    )
    assert len(requeue_events) == 1


def test_same_head_unchanged_body_still_reroutes_to_rework(tmp_path: Path) -> None:
    """Issue #1983 AC3 (same-head regression): an unchanged body means the
    verdict still stands -- the stranded-verdict repair fires exactly as
    before, routing the issue to ``rework_requested`` and never queueing a
    re-review."""
    head = "sha-live-head"
    pr = _stale_ci_pr(_PR_NUMBER, _ISSUE_NUMBER, head)
    pr["body"] = _NEW_BODY

    app = _review_queue_app(tmp_path, prs=[pr], dry_run=False)
    _write_review_packet(
        tmp_path,
        _PR_NUMBER,
        head,
        _request_changes_decision(head, reviewed_body_sha256=_body_sha256(_NEW_BODY)),
    )
    state = load_state(app.paths.state_file)
    state["issues"][str(_ISSUE_NUMBER)] = {"number": _ISSUE_NUMBER, "status": "reviewing"}
    save_state(app.paths.state_file, state)

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []
    state_after = load_state(app.paths.state_file)
    assert state_after["issues"][str(_ISSUE_NUMBER)]["status"] == "rework_requested"
    assert (
        len(query_events(app.paths.state_file, kind="stranded_request_changes_rework_requested"))
        == 1
    )
    assert query_events(app.paths.state_file, kind="request_changes_body_changed_requeued") == []


def test_carry_forward_applies_when_body_unchanged(tmp_path: Path) -> None:
    """Issue #1983 AC3 (carry-forward regression): a patch-id-identical head
    with an UNCHANGED body keeps the existing carry-forward -- the verdict
    re-pins to the new head and no re-review is queued."""
    from charlie_work.janitor import _calculate_patch_id

    old_head = "sha-abc123"
    new_head = "sha-rebased123"
    patch_id = _calculate_patch_id(_DIFF)
    pr = _stale_ci_pr(_PR_NUMBER, _ISSUE_NUMBER, new_head)
    pr["body"] = _NEW_BODY

    app = _review_queue_app(tmp_path, prs=[pr], dry_run=False)
    app.gh.diffs[_PR_NUMBER] = _DIFF
    _write_review_packet(
        tmp_path,
        _PR_NUMBER,
        new_head,
        _request_changes_decision(
            old_head,
            patch_id=patch_id,
            reviewed_body_sha256=_body_sha256(_NEW_BODY),
        ),
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []
    decision = json.loads(
        (app.paths.prs / f"pr-{_PR_NUMBER}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == new_head
    assert decision["carry_forward_tier"] == "patch-id"
    carry_events = query_events(app.paths.state_file, kind="verdict_carried_forward_clean_rebase")
    assert len(carry_events) == 1
    assert query_events(app.paths.state_file, kind="request_changes_body_changed_requeued") == []


def test_carry_forward_applies_when_body_baseline_missing(tmp_path: Path) -> None:
    """Issue #1983 AC4 (legacy verdict): a request_changes recorded before
    ``reviewed_body_sha256`` existed has no baseline to compare, so the
    drift check fails closed -- carry-forward behaves exactly as today."""
    from charlie_work.janitor import _calculate_patch_id

    old_head = "sha-abc123"
    new_head = "sha-rebased123"
    patch_id = _calculate_patch_id(_DIFF)
    pr = _stale_ci_pr(_PR_NUMBER, _ISSUE_NUMBER, new_head)
    pr["body"] = _NEW_BODY

    app = _review_queue_app(tmp_path, prs=[pr], dry_run=False)
    app.gh.diffs[_PR_NUMBER] = _DIFF
    _write_review_packet(
        tmp_path,
        _PR_NUMBER,
        new_head,
        _request_changes_decision(old_head, patch_id=patch_id, reviewed_body_sha256=None),
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []
    decision = json.loads(
        (app.paths.prs / f"pr-{_PR_NUMBER}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == new_head
    assert decision["carry_forward_tier"] == "patch-id"
    assert (
        len(query_events(app.paths.state_file, kind="verdict_carried_forward_clean_rebase")) == 1
    )


def test_approved_verdict_carries_forward_across_body_change(tmp_path: Path) -> None:
    """Issue #1983 AC5: an ``approved`` verdict is unaffected -- it still
    carries forward across a body edit (only ``request_changes`` gains the
    drift guard)."""
    from charlie_work.janitor import _calculate_patch_id

    old_head = "sha-abc123"
    new_head = "sha-rebased123"
    patch_id = _calculate_patch_id(_DIFF)
    pr = _stale_ci_pr(_PR_NUMBER, _ISSUE_NUMBER, new_head)
    pr["body"] = _NEW_BODY

    app = _review_queue_app(tmp_path, prs=[pr], dry_run=False)
    app.gh.diffs[_PR_NUMBER] = _DIFF
    _write_review_packet(
        tmp_path,
        _PR_NUMBER,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "reviewed_body_sha256": _body_sha256(_OLD_BODY),
            "carried_forward_from": [],
            "summary": "looks good",
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []
    decision = json.loads(
        (app.paths.prs / f"pr-{_PR_NUMBER}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == new_head
    assert decision["carry_forward_tier"] == "patch-id"
    assert (
        len(query_events(app.paths.state_file, kind="verdict_carried_forward_clean_rebase")) == 1
    )
    assert query_events(app.paths.state_file, kind="request_changes_body_changed_requeued") == []


def test_same_head_legacy_verdict_still_reroutes_to_rework(tmp_path: Path) -> None:
    """Issue #1983 AC4 (same-head legacy half): a request_changes verdict
    with no ``reviewed_body_sha256`` keeps today's behavior -- the
    stranded-verdict repair still routes it to rework."""
    head = "sha-live-head"
    pr = _stale_ci_pr(_PR_NUMBER, _ISSUE_NUMBER, head)
    pr["body"] = _NEW_BODY

    app = _review_queue_app(tmp_path, prs=[pr], dry_run=False)
    _write_review_packet(
        tmp_path,
        _PR_NUMBER,
        head,
        _request_changes_decision(head, reviewed_body_sha256=None),
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []
    state_after = load_state(app.paths.state_file)
    assert state_after["issues"][str(_ISSUE_NUMBER)]["status"] == "rework_requested"
