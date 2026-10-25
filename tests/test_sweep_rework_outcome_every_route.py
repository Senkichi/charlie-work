"""A rework worker's ``pr_body``/``pr_comment`` is applied on EVERY dead-worker-with-PR route (#2826).

The rework brief promises "the orchestrator applies them to the PR for you", but the
sweep queued the apply only inside ``_completed_outcome`` -- reached solely from the
same-head branches. The head-changed route, the unreviewed route and the drift
fallback never read the outcome, so the edit was dropped without an event and the
reviewer read the stale body (the sibling-repo incident that motivated #2826).

Beds are the shared ``_dead_worker_rework_bed`` family; the helper below only adds the
head-changed / unreviewed / drift variants the existing characterization module keeps
private. Dates are derived from ``datetime.now`` -- never literals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from _dead_session_fixtures import _write_flat_review_decision
from _dead_worker_sweep_characterization_fixtures import (
    events_of,
    iso,
    issue_entry,
    run_sweep,
    seed_issue,
)
from _orphan_sweep_fixtures import _dead_worker_rework_bed, _write_outcome

from charlie_work import rework_outcome
from charlie_work.command_result import CommandResult
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.host.clock import format_utc
from charlie_work.host.fakes import FakeClock
from charlie_work.instrumentation import query_events
from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state

OLD_HEAD = "abc123"  # the head the bed's verdict was recorded at
NEW_HEAD = "def456"
BODY = "Closes #207\n\nCorrected PR body per review."
COMMENT = "Reviewer: the body now documents the verification."

ROUTES = ("head_changed", "same_head_approved", "same_head_request_changes", "unreviewed", "drift")


def _config(*, every_route: bool = True) -> OrchestratorConfig:
    return OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(
            enabled=True, stall_minutes=20, apply_rework_outcome_on_every_route=every_route
        ),
    )


def _payload(head: str, **fields: Any) -> dict[str, Any]:
    return {
        "push_succeeded": True,
        "pr_created": False,
        "head_sha": head,
        "pr_body": BODY,
        "pr_comment": COMMENT,
        **fields,
    }


@dataclass
class ReviewSpy:
    """A ``review_callback`` that records what the PR looked like when it was called."""

    gh: Any
    paths: Any
    ok: bool = True
    calls: list[int] = field(default_factory=list)
    seen: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, pr_number: int) -> CommandResult:
        applied = load_state(self.paths.state_file).get("rework_outcome_applied_heads", {})
        self.calls.append(pr_number)
        self.seen.append(
            {
                "pr_edits": len(self.gh.pr_edits),
                "pr_comments": len(self.gh.pr_comments_posted),
                "applied_head": applied.get("207"),
            }
        )
        return CommandResult(self.ok, "review", {"pr_number": pr_number})


def _unreviewed(tmp_path: Path, config: OrchestratorConfig, *, decision: str = "absent"):
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path, config=config)
    state = load_state(paths.state_file)
    state["prs"]["100"] = {"reviewed_head_sha": None}
    save_state(paths.state_file, state)
    if decision == "absent":
        (paths.prs / "pr-100" / "review-decision.json").unlink()
    else:
        _write_flat_review_decision(paths, 100, decision, None)
    gh.issues[-1]["labels"] = [{"name": config.labels.in_progress}]
    gh.prs[0].update({"mergeStateStatus": "CLEAN", "state": "OPEN"})
    return config, paths, gh


def _bed(tmp_path: Path, route: str, *, every_route: bool = True):
    """``(config, paths, gh, live_head)`` for one terminal branch of ``with_pr_flow``."""
    config = _config(every_route=every_route)
    if route == "head_changed":
        config, paths, gh, _ = _dead_worker_rework_bed(tmp_path, config=config)
        gh.prs[0]["headRefOid"] = NEW_HEAD
        return config, paths, gh, NEW_HEAD
    if route == "same_head_approved":
        config, paths, gh, _ = _dead_worker_rework_bed(
            tmp_path, decision="approved", pr_state_status="rework_requested", config=config
        )
        return config, paths, gh, OLD_HEAD
    if route == "same_head_request_changes":
        config, paths, gh, _ = _dead_worker_rework_bed(tmp_path, config=config)
        return config, paths, gh, OLD_HEAD
    config, paths, gh = _unreviewed(tmp_path, config)
    if route == "drift":
        # The label edge is refused, so the unreviewed advance falls back to the
        # ``dead_worker_unsafe_to_auto_reset`` drift -- the flow's last branch.
        gh.remove_issue_label = lambda number, label: False  # type: ignore[method-assign]
    return config, paths, gh, OLD_HEAD


def _sweep(
    tmp_path: Path,
    paths: Any,
    config: OrchestratorConfig,
    gh: Any,
    monkeypatch,
    *,
    remote_head: str,
    review: Any = None,
) -> None:
    probe = patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: remote_head)
    run_sweep(
        tmp_path,
        paths,
        config,
        gh,
        review_callback=review,
        monkeypatch=monkeypatch,
        patches=(probe,),
    )


def _applied_events(paths: Any) -> list[dict[str, Any]]:
    return events_of(paths, "rework_outcome_applied")


def _assert_applied_once(paths: Any, gh: Any, head: str) -> None:
    assert [n for n, _ in gh.pr_edits] == [100]
    assert BODY in gh.pr_edits[0][1]
    assert [n for n, _ in gh.pr_comments_posted] == [100]
    assert COMMENT in gh.pr_comments_posted[0][1]
    (event,) = _applied_events(paths)
    assert event["payload"]["body_updated"] is True
    assert event["payload"]["comment_posted"] is True
    assert event["payload"]["head_sha"] == head
    assert load_state(paths.state_file)["rework_outcome_applied_heads"]["207"] == head


# ---------------------------------------------------------------------------
# Criterion 1: head-changed route applies BEFORE review()
# ---------------------------------------------------------------------------


def test_head_changed_route_applies_once_before_review(tmp_path: Path, monkeypatch) -> None:
    config, paths, gh, head = _bed(tmp_path, "head_changed")
    _write_outcome(paths, tmp_path, _payload(head))
    spy = ReviewSpy(gh, paths)

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=spy)

    _assert_applied_once(paths, gh, head)
    assert spy.calls == [100]
    # Call order: the body edit, the comment and the head record all precede review().
    assert spy.seen == [{"pr_edits": 1, "pr_comments": 1, "applied_head": head}]
    assert issue_entry(paths, 207)["status"] == "reviewing"
    (routed,) = events_of(paths, "orphaned_worker_routed_to_review")
    assert routed["payload"]["reason"] == "dead_worker_with_head_change"


# ---------------------------------------------------------------------------
# Criterion 2: unreviewed route applies; advance-to-pr-open is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decision", ["absent", "pending"])
def test_unreviewed_route_applies_and_still_advances_to_pr_open(
    tmp_path: Path, monkeypatch, decision: str
) -> None:
    config = _config()
    config, paths, gh = _unreviewed(tmp_path, config, decision=decision)
    _write_outcome(paths, tmp_path, _payload(OLD_HEAD))

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=OLD_HEAD)

    _assert_applied_once(paths, gh, OLD_HEAD)
    entry = issue_entry(paths, 207)
    assert entry["status"] == PASSIVE_OPEN_STATUS
    assert entry.get("dispatched_at") is None
    (advanced,) = events_of(paths, "orphaned_worker_advanced_to_pr_open")
    assert advanced["payload"]["reason"] == "dead_worker_unsafe_to_auto_reset_open_unreviewed_pr"
    assert (207, config.labels.in_progress) in gh.labels_removed
    assert (207, config.labels.pr_open) in gh.labels_added


# ---------------------------------------------------------------------------
# Criterion 3: same-head regression -- one apply, not two
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["same_head_request_changes", "same_head_approved"])
def test_same_head_route_still_applies_exactly_once(
    tmp_path: Path, monkeypatch, route: str
) -> None:
    config, paths, gh, head = _bed(tmp_path, route)
    _write_outcome(paths, tmp_path, _payload(head))
    spy = ReviewSpy(gh, paths)

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=spy)

    _assert_applied_once(paths, gh, head)
    assert spy.seen == [{"pr_edits": 1, "pr_comments": 1, "applied_head": head}]
    (routed,) = events_of(paths, "orphaned_worker_routed_to_review")
    assert routed["payload"]["reason"] == "dead_worker_completed_outcome"


# ---------------------------------------------------------------------------
# Criterion 4: a second pass over the same state neither re-edits nor re-posts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["head_changed", "drift"])
def test_second_pass_does_not_reapply(tmp_path: Path, monkeypatch, route: str) -> None:
    config, paths, gh, head = _bed(tmp_path, route)
    _write_outcome(paths, tmp_path, _payload(head))
    # A refused review (head-changed) / refused label edge (drift) leaves the issue
    # ``dispatched`` so the second sweep pass really sees the same state again.
    spy = ReviewSpy(gh, paths, ok=False)

    for _ in range(2):
        _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=spy)

    assert issue_entry(paths, 207)["status"] == "dispatched"
    _assert_applied_once(paths, gh, head)


# ---------------------------------------------------------------------------
# Criterion 5: guards are unchanged on the newly covered routes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["head_changed", "unreviewed"])
def test_remote_head_mismatch_skips_with_no_pr_mutation(
    tmp_path: Path, monkeypatch, route: str
) -> None:
    config, paths, gh, head = _bed(tmp_path, route)
    _write_outcome(paths, tmp_path, _payload(head))

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head="9999zz")

    assert gh.pr_edits == []
    assert gh.pr_comments_posted == []
    assert _applied_events(paths) == []
    (skipped,) = events_of(paths, "rework_outcome_skipped")
    assert skipped["payload"]["reason"] == "head_mismatch"
    assert skipped["payload"]["reported_head_sha"] == head
    assert skipped["payload"]["remote_head_sha"] == "9999zz"


@pytest.mark.parametrize("route", ["head_changed", "unreviewed"])
def test_outcome_for_a_different_head_than_the_pr_is_never_applied(
    tmp_path: Path, monkeypatch, route: str
) -> None:
    config, paths, gh, head = _bed(tmp_path, route)
    _write_outcome(paths, tmp_path, _payload("0ther0"))

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head="0ther0")

    assert gh.pr_edits == []
    assert gh.pr_comments_posted == []
    assert _applied_events(paths) == []
    assert "207" not in load_state(paths.state_file).get("rework_outcome_applied_heads", {})


@pytest.mark.parametrize("route", ["head_changed", "unreviewed"])
def test_blocked_outcome_is_never_applied(tmp_path: Path, monkeypatch, route: str) -> None:
    config, paths, gh, head = _bed(tmp_path, route)
    _write_outcome(paths, tmp_path, _payload(head, outcome="blocked"))

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head)

    assert gh.pr_edits == []
    assert gh.pr_comments_posted == []
    assert _applied_events(paths) == []


@pytest.mark.parametrize("route", ["head_changed", "unreviewed"])
def test_outcome_written_before_the_dispatch_is_never_applied(
    tmp_path: Path, monkeypatch, route: str
) -> None:
    config, paths, gh, head = _bed(tmp_path, route)
    # The bed dispatched an hour ago; a file from three hours ago is a prior round's.
    _write_outcome(paths, tmp_path, _payload(head), mtime=datetime.now(UTC) - timedelta(hours=3))

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head)

    assert gh.pr_edits == []
    assert gh.pr_comments_posted == []
    assert _applied_events(paths) == []


@pytest.mark.parametrize("route", ["head_changed", "unreviewed"])
def test_outcome_with_neither_field_mutates_nothing_but_records_the_head(
    tmp_path: Path, monkeypatch, route: str
) -> None:
    config, paths, gh, head = _bed(tmp_path, route)
    _write_outcome(
        paths,
        tmp_path,
        {"push_succeeded": True, "pr_created": False, "head_sha": head},
    )

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=ReviewSpy(gh, paths))

    assert gh.pr_edits == []
    assert gh.pr_comments_posted == []
    (event,) = _applied_events(paths)
    assert event["payload"]["body_updated"] is False
    assert event["payload"]["comment_posted"] is False
    assert load_state(paths.state_file)["rework_outcome_applied_heads"]["207"] == head


# ---------------------------------------------------------------------------
# Criterion 6: a transient gh failure holds the head-changed review route
# ---------------------------------------------------------------------------


def test_transient_edit_failure_holds_the_review_route_until_the_apply_lands(
    tmp_path: Path, monkeypatch
) -> None:
    config, paths, gh, head = _bed(tmp_path, "head_changed")
    _write_outcome(paths, tmp_path, _payload(head))
    spy = ReviewSpy(gh, paths)

    real_edit = gh.pr_edit
    attempts: list[int] = []

    def flaky_edit(number: int, body_file: Path) -> None:
        attempts.append(number)
        if len(attempts) == 1:
            raise RuntimeError("gh: 502 Bad Gateway")
        real_edit(number, body_file)

    gh.pr_edit = flaky_edit  # type: ignore[method-assign]

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=spy)

    (failed,) = events_of(paths, "rework_outcome_apply_failed")
    assert "502" in failed["payload"]["error"]
    (failed_row,) = query_events(paths.state_file, kind="rework_outcome_apply_failed")
    assert failed_row["level"] == "warning"
    assert spy.calls == []  # held: the reviewer must not see the stale body
    assert issue_entry(paths, 207)["status"] == "dispatched"
    assert _applied_events(paths) == []
    assert gh.pr_edits == []

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=spy)

    _assert_applied_once(paths, gh, head)
    assert spy.seen == [{"pr_edits": 1, "pr_comments": 1, "applied_head": head}]
    assert issue_entry(paths, 207)["status"] == "reviewing"
    assert len(events_of(paths, "rework_outcome_apply_failed")) == 1


def _fail_every_pr_edit(gh: Any) -> None:
    def failing_edit(number: int, body_file: Path) -> None:
        raise RuntimeError("gh: 502 Bad Gateway")

    gh.pr_edit = failing_edit  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Criterion 6b: a held head-changed route is bounded by the #654 backstop
# ---------------------------------------------------------------------------


def test_held_head_changed_route_anchors_the_dead_dispatched_backstop(
    tmp_path: Path, monkeypatch, fake_host
) -> None:
    """An apply that can never land ends in the reap; it does not hold the issue forever.

    A held route writes no drift of its own, so ``_head_changed`` anchors
    ``orphan_drift_at`` itself. Time moves only through the clock port: the anchor is
    written by the sweep (never seeded here), is not re-armed by a later pass, and the
    reap fires once ``dead_dispatched_reap_minutes`` have passed since it.
    """
    clock = FakeClock(datetime.now(UTC))
    fake_host(clock=clock)
    anchored_at = format_utc(clock.now())
    config, paths, gh, head = _bed(tmp_path, "head_changed")
    _write_outcome(paths, tmp_path, _payload(head))
    _fail_every_pr_edit(gh)
    spy = ReviewSpy(gh, paths)

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=spy)

    assert spy.calls == []  # held: the apply failed, the reviewer must not see the stale body
    entry = issue_entry(paths, 207)
    assert entry["status"] == "dispatched"
    assert entry["orphan_drift_at"] == anchored_at

    clock.advance(5 * 60)
    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=spy)

    assert spy.calls == []
    assert len(events_of(paths, "rework_outcome_apply_failed")) == 2  # still held, retried
    assert issue_entry(paths, 207)["orphan_drift_at"] == anchored_at  # not re-stamped
    assert events_of(paths, "dead_dispatched_worker_reaped") == []  # inside the window

    clock.advance(config.watchdog.dead_dispatched_reap_minutes * 60)
    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=spy)

    (reaped,) = events_of(paths, "dead_dispatched_worker_reaped")
    assert reaped["payload"]["issue_number"] == 207
    assert reaped["payload"]["orphan_drift_at"] == anchored_at
    entry = issue_entry(paths, 207)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "dead_dispatched_worker_reap"
    assert spy.calls == []  # never reviewed the stale body


@pytest.mark.parametrize("control", ["kill_switch_off", "no_outcome"])
def test_head_changed_route_that_is_not_held_stamps_no_anchor(
    tmp_path: Path, monkeypatch, control: str
) -> None:
    """The anchor belongs to a HELD route only: a review that proceeds writes no drift."""
    config, paths, gh, head = _bed(
        tmp_path, "head_changed", every_route=control != "kill_switch_off"
    )
    if control == "kill_switch_off":
        _write_outcome(paths, tmp_path, _payload(head))
    spy = ReviewSpy(gh, paths)

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=spy)

    assert spy.calls == [100]  # not held
    entry = issue_entry(paths, 207)
    assert entry["status"] == "reviewing"
    assert entry.get("orphan_drift_at") is None
    assert gh.pr_edits == []


# ---------------------------------------------------------------------------
# Criterion 8: branch-coverage guard -- every terminal branch carries an apply
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
def test_every_terminal_branch_of_with_pr_flow_applies_the_outcome(
    tmp_path: Path, monkeypatch, route: str
) -> None:
    config, paths, gh, head = _bed(tmp_path, route)
    _write_outcome(paths, tmp_path, _payload(head))

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=ReviewSpy(gh, paths))

    _assert_applied_once(paths, gh, head)


# ---------------------------------------------------------------------------
# Criterion 9: kill switch -- False reproduces the pre-#2826 routing exactly
# ---------------------------------------------------------------------------


def test_default_applies_the_outcome_on_every_route() -> None:
    assert WatchdogConfig().apply_rework_outcome_on_every_route is True
    assert OrchestratorConfig().watchdog.apply_rework_outcome_on_every_route is True


def test_kill_switch_off_head_changed_route_reviews_without_applying(
    tmp_path: Path, monkeypatch
) -> None:
    config, paths, gh, head = _bed(tmp_path, "head_changed", every_route=False)
    _write_outcome(paths, tmp_path, _payload(head))
    spy = ReviewSpy(gh, paths)

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=spy)

    assert spy.calls == [100]  # not held: today's behaviour
    assert gh.pr_edits == []
    assert gh.pr_comments_posted == []
    assert _applied_events(paths) == []
    assert issue_entry(paths, 207)["status"] == "reviewing"


def test_kill_switch_off_unreviewed_route_advances_without_applying(
    tmp_path: Path, monkeypatch
) -> None:
    config, paths, gh, head = _bed(tmp_path, "unreviewed", every_route=False)
    _write_outcome(paths, tmp_path, _payload(head))

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head)

    assert gh.pr_edits == []
    assert _applied_events(paths) == []
    assert issue_entry(paths, 207)["status"] == PASSIVE_OPEN_STATUS
    assert len(events_of(paths, "orphaned_worker_advanced_to_pr_open")) == 1


@pytest.mark.parametrize("route", ["same_head_request_changes", "same_head_approved"])
def test_kill_switch_off_same_head_route_still_applies_once(
    tmp_path: Path, monkeypatch, route: str
) -> None:
    config, paths, gh, head = _bed(tmp_path, route, every_route=False)
    _write_outcome(paths, tmp_path, _payload(head))
    spy = ReviewSpy(gh, paths)

    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=spy)

    _assert_applied_once(paths, gh, head)
    assert spy.seen == [{"pr_edits": 1, "pr_comments": 1, "applied_head": head}]


# ---------------------------------------------------------------------------
# Criterion 12: replay of the #2826 incident shapes on one issue, in order
# ---------------------------------------------------------------------------


def test_replay_incident_three_rounds_apply_once_per_distinct_head(
    tmp_path: Path, monkeypatch
) -> None:
    """Round 1 head-changed, rounds 2 and 3 unreviewed: three applies, one per head."""
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path, config=_config())
    gh.issues[-1]["labels"] = [{"name": config.labels.in_progress}]
    gh.prs[0].update({"mergeStateStatus": "CLEAN", "state": "OPEN"})
    spy = ReviewSpy(gh, paths)

    heads = ("c0ffee1", "c0ffee2", "c0ffee3")

    # Round 1: the verdict is at abc123, the worker pushed c0ffee1 -> head-changed route.
    gh.prs[0]["headRefOid"] = heads[0]
    _write_outcome(paths, tmp_path, _payload(heads[0], pr_body=f"{BODY}\n\nround 1"))
    _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=heads[0], review=spy)
    assert issue_entry(paths, 207)["status"] == "reviewing"
    assert spy.seen == [{"pr_edits": 1, "pr_comments": 1, "applied_head": heads[0]}]

    # Rounds 2 and 3: no verdict on the new head (the review packet was voided), so the
    # next dead worker lands on the unreviewed route.
    for round_no, head in ((2, heads[1]), (3, heads[2])):
        state = load_state(paths.state_file)
        state["prs"]["100"] = {"reviewed_head_sha": None}
        save_state(paths.state_file, state)
        decision_file = paths.prs / "pr-100" / "review-decision.json"
        if decision_file.exists():
            decision_file.unlink()
        seed_issue(paths, 207, status="dispatched", dispatched_at=iso(minutes_ago=30))
        gh.prs[0]["headRefOid"] = head
        _write_outcome(paths, tmp_path, _payload(head, pr_body=f"{BODY}\n\nround {round_no}"))
        _sweep(tmp_path, paths, config, gh, monkeypatch, remote_head=head, review=spy)
        assert issue_entry(paths, 207)["status"] == PASSIVE_OPEN_STATUS

    applied = _applied_events(paths)
    assert [e["payload"]["head_sha"] for e in applied] == list(heads)
    assert all(e["payload"]["body_updated"] and e["payload"]["comment_posted"] for e in applied)
    assert [n for n, _ in gh.pr_edits] == [100, 100, 100]
    assert [text.rsplit("round ", 1)[1] for _, text in gh.pr_edits] == ["1", "2", "3"]
    assert load_state(paths.state_file)["rework_outcome_applied_heads"]["207"] == heads[2]
