"""Issue #2899: the zero-diff skip's operator-queue exit route.

Split out of ``tests/test_charlie_work_dispatch_reviews_stalled.py``: adding
these tests there pushed that file past the 800-line cap
(``tests/test_file_size_ratchet.py``, issue #1442). The pre-existing empty-diff
tests stay in the stalled-review file, updated for the new behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _dispatch_fixtures import _fail_if_launched
from _review_fixtures import (
    _dispatch_reviews_app,
    _fake_claude_worker_record,
    _write_review_packet,
)
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.instrumentation import query_events
from charlie_work.state import (
    clear_escalation,
    clear_escalation_on_issue_prs,
    load_state,
    save_state,
    state_lock,
)


def _empty_diff_issue_linked_app(tmp_path: Path) -> Any:
    """An app whose PR 100 is linked to open issue 10 and whose packet diff is
    empty at head ``sha-100`` -- the #2899 fixture shape."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    app.gh.issues = [
        {
            "number": 10,
            "title": "Fix search",
            "url": "https://example.test/issues/10",
            "body": "Search is broken",
            "labels": [],
            "state": "OPEN",
        }
    ]
    _write_review_packet(tmp_path, 100, "sha-100")
    (app.paths.prs / "pr-100" / "diff.patch").write_text("", encoding="utf-8")
    return app


def _clear_empty_diff_escalation(app: Any) -> None:
    """Reset the #2899 escalation fields the way a clear does (the mechanical
    de-escalation sweep's per-issue clear shape, mirrored on the PR record),
    leaving the head-scoped dedup marker in place."""
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        issue_entry = state["issues"]["10"]
        issue_entry["status"] = "open_passive"
        clear_escalation(issue_entry)
        clear_escalation_on_issue_prs(state, 10)
        state["issues"]["10"] = issue_entry
        pr_entry = state["prs"]["100"]
        pr_entry["status"] = "open_passive"
        clear_escalation(pr_entry)
        state["prs"]["100"] = pr_entry
        save_state(app.paths.state_file, state)


def test_dispatch_reviews_empty_diff_route_is_once_per_head(monkeypatch, tmp_path: Path) -> None:
    """Issue #2899 ACs 1-3: repeated passes over one zero-diff head produce
    exactly one transition and one event (first pass: escalated-skipped;
    later same-head passes: silent even after the escalation is cleared --
    the head-scoped marker deduplicates) while a new empty head re-routes
    once."""
    app = _empty_diff_issue_linked_app(tmp_path)

    _fail_if_launched(monkeypatch)

    # Pass 1: the route fires.
    result1 = app.dispatch_reviews()
    assert result1.data["empty_diff_operator_queued"] == [100]
    escalated_after_pass1 = query_events(app.paths.state_file, kind="review_dispatch_escalated")
    assert len(escalated_after_pass1) == 1
    assert app.gh.labels_added.count((10, app.config.labels.operator_queue)) == 1

    # Pass 2: escalated status blinds selection -- nothing new fires.
    result2 = app.dispatch_reviews()
    assert 100 in result2.data["escalated_skipped"]
    assert result2.data["empty_diff_operator_queued"] == []
    assert len(query_events(app.paths.state_file, kind="review_dispatch_escalated")) == 1
    assert app.gh.labels_added.count((10, app.config.labels.operator_queue)) == 1

    # Simulate the escalation being cleared while the head is STILL empty
    # (the mechanical de-escalation sweep's clear shape; the marker stays).
    _clear_empty_diff_escalation(app)

    # Pass 3: same head, still empty -- the marker keeps it silent, no
    # re-escalation ping-pong.
    result3 = app.dispatch_reviews()
    assert result3.data["empty_diff_operator_queued"] == []
    assert result3.data["skipped_empty_diff"] == [100]
    assert len(query_events(app.paths.state_file, kind="review_dispatch_escalated")) == 1
    assert app.gh.labels_added.count((10, app.config.labels.operator_queue)) == 1
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["status"] != "escalated"
    assert state["prs"]["100"]["empty_diff_operator_queued_head_sha"] == "sha-100"
    assert int(state["prs"]["100"].get("review_dispatch_attempt_count", 0)) == 0

    # A NEW head with an empty diff re-routes (once per head). Move the real
    # PR head (pr_list's headRefOid) -- review_queue reads pr_list, not
    # pr_view's pr_head_shas override.
    app.gh.prs[0]["headRefOid"] = "sha-200"
    _write_review_packet(tmp_path, 100, "sha-200")
    (app.paths.prs / "pr-100" / "diff.patch").write_text("", encoding="utf-8")

    result4 = app.dispatch_reviews()
    assert result4.data["empty_diff_operator_queued"] == [100]
    escalated_after_pass4 = query_events(app.paths.state_file, kind="review_dispatch_escalated")
    assert len(escalated_after_pass4) == 2
    assert escalated_after_pass4[1]["payload"]["reason"] == "empty_diff_zero_file"
    assert escalated_after_pass4[1]["payload"]["packet_head_sha"] == "sha-200"
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["empty_diff_operator_queued_head_sha"] == "sha-200"
    assert int(state["prs"]["100"].get("review_dispatch_attempt_count", 0)) == 0
    assert state["issues"]["10"]["status"] == "escalated"


def test_dispatch_reviews_missing_diff_patch_is_not_empty(monkeypatch, tmp_path: Path) -> None:
    """Issue #2899 AC4 (the #1807 behaviour): a MISSING diff.patch is never
    read as empty -- no operator-queue route, no skipped_empty_diff entry,
    no escalation event -- so a packet whose diff file is absent (a missing
    packet, a different problem from a zero-delta PR) still goes through the
    pre-existing normal gate chain instead of being escalated for a zero
    diff it does not have."""
    app = _empty_diff_issue_linked_app(tmp_path)
    (app.paths.prs / "pr-100" / "diff.patch").unlink()

    recorded: list[int] = []

    def recording_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        recorded.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", recording_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["skipped_empty_diff"] == []
    assert result.data["empty_diff_operator_queued"] == []
    assert query_events(app.paths.state_file, kind="review_dispatch_escalated") == []
    # The missing packet is not a zero-delta PR: the dispatch proceeds under
    # the normal gate chain (claim + launch), exactly as before #2899.
    assert result.data["launched_count"] == 1
    assert recorded == [100]


def test_dispatch_reviews_empty_diff_new_head_regained_diff_is_dispatched(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #2899 AC3: a PR that gains a non-empty diff on a new head is
    dispatched for review as usual once the escalation is cleared -- the
    head-scoped marker must never stand between a plus-diff head and the
    reviewer."""
    app = _empty_diff_issue_linked_app(tmp_path)

    _fail_if_launched(monkeypatch)

    # Pass 1: route to the operator queue.
    result1 = app.dispatch_reviews()
    assert result1.data["empty_diff_operator_queued"] == [100]

    # Simulate the escalation being cleared (the mechanical de-escalation
    # sweep's clear shape) while the head is still empty.
    _clear_empty_diff_escalation(app)

    # Pass 2: same head, still empty -- the marker keeps it silent and no
    # reviewer is claimed.
    result2 = app.dispatch_reviews()
    assert result2.data["skipped_empty_diff"] == [100]
    assert result2.data["empty_diff_operator_queued"] == []
    assert len(query_events(app.paths.state_file, kind="review_dispatch_escalated")) == 1

    # Now the PR regains a diff on the new head: normal dispatch. Move the
    # real PR head (pr_list's headRefOid) -- review_queue reads pr_list.
    app.gh.prs[0]["headRefOid"] = "sha-200"
    _write_review_packet(tmp_path, 100, "sha-200")
    (app.paths.prs / "pr-100" / "diff.patch").write_text(
        "diff --git a/foo.py b/foo.py\n+print('hello')\n", encoding="utf-8"
    )

    recorded: list[int] = []

    def recording_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        recorded.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", recording_launch)

    result3 = app.dispatch_reviews()

    assert result3.ok is True
    assert recorded == [100]
    assert result3.data["launched_count"] == 1
    assert result3.data["empty_diff_operator_queued"] == []
    # Still exactly one escalation event -- the regained-diff pass raised
    # no new one.
    assert len(query_events(app.paths.state_file, kind="review_dispatch_escalated")) == 1
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert int(state["prs"]["100"].get("review_dispatch_attempt_count", 0)) == 1
