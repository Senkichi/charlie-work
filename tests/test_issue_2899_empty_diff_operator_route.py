"""Issue #2899: the zero-diff skip's operator-queue exit route.

Split out of ``tests/test_charlie_work_dispatch_reviews_stalled.py``: adding
these tests there pushed that file past the 800-line cap
(``tests/test_file_size_ratchet.py``, issue #1442). The pre-existing empty-diff
tests stay in the stalled-review file, updated for the new behaviour.

Two layers:

* Flow tests drive the REAL ``dispatch_reviews`` and the REAL mechanical
  de-escalation sweep (``_deescalate_mechanical_issue`` /
  ``_maybe_deescalate_mechanical``) and the real operator ``unescalate`` --
  no hand-rolled stand-in for a clear. The fixture PR is open, mergeable and
  janitor-green, i.e. exactly the shape the sweep would otherwise clear.
* Direct tests call ``route_empty_diff_prs`` with a crafted state to pin its
  silent paths and its PR-only path.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from _dispatch_fixtures import _fail_if_launched
from _fakes_github import FakeGitHub
from _review_fixtures import _fake_claude_worker_record, _write_review_packet
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import OrchestratorConfig, PostMortemConfig, ReviewDispatchConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.review_empty_diff_route import (
    EMPTY_DIFF_ESCALATION_REASON,
    EMPTY_DIFF_ROUTED_HEAD_FIELD,
    EmptyDiffSkip,
    route_empty_diff_prs,
)
from charlie_work.state import (
    PASSIVE_OPEN_STATUS,
    SINK_STATUSES,
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from charlie_work.write_gate import WriteGate


def _empty_diff_issue_linked_app(tmp_path: Path) -> OrchestratorApp:
    """An app whose PR 100 is linked to open issue 10 and whose packet diff is
    empty at head ``sha-100`` -- the #2899 fixture shape.

    Janitor-green on purpose (closing keyword + tests mention, FakeGitHub's
    green checks, open, not conflicting) and isolated from the host's real
    ``sessions.db``, so the real de-escalation sweep reaches its clear decision
    instead of stopping on an unrelated gate.
    """
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    gh = FakeGitHub()
    gh.prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "mergeable": "MERGEABLE",
            "body": "Closes #10\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    gh.issues = [
        {
            "number": 10,
            "title": "Fix search",
            "url": "https://example.test/issues/10",
            "body": "Search is broken",
            "labels": [],
            "state": "OPEN",
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _write_review_packet(tmp_path, 100, "sha-100")
    (app.paths.prs / "pr-100" / "diff.patch").write_text("", encoding="utf-8")
    return app


def _move_head(app: OrchestratorApp, tmp_path: Path, head: str, *, diff: str = "") -> None:
    """Move PR 100's real head (``pr_list``/``pr_view`` both read it) and
    rebuild the packet for it, with ``diff`` as its diff.patch."""
    app.gh.prs[0]["headRefOid"] = head
    _write_review_packet(tmp_path, 100, head)
    (app.paths.prs / "pr-100" / "diff.patch").write_text(diff, encoding="utf-8")


def _routed_events(app: OrchestratorApp) -> list[dict[str, Any]]:
    return query_events(app.paths.state_file, kind="review_dispatch_escalated")


def _all_event_ids(app: OrchestratorApp) -> list[tuple[str, str]]:
    return [(e["kind"], str(e.get("ts"))) for e in query_events(app.paths.state_file)]


def _recording_launch(recorded: list[int]) -> Any:
    def launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        recorded.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    return launch


def test_dispatch_reviews_empty_diff_route_is_once_per_head(monkeypatch, tmp_path: Path) -> None:
    """Issue #2899 ACs 1-3: repeated passes over one zero-diff head produce
    exactly one transition and one event (first pass: escalated-skipped;
    later same-head passes: blinded by the escalation) while a NEW empty head
    re-routes once -- after the real sweep clears the old head's escalation."""
    app = _empty_diff_issue_linked_app(tmp_path)
    _fail_if_launched(monkeypatch)

    result1 = app.dispatch_reviews()
    assert result1.data["empty_diff_operator_queued"] == [100]
    assert len(_routed_events(app)) == 1
    assert app.gh.labels_added.count((10, app.config.labels.operator_queue)) == 1

    # Pass 2: escalated status blinds selection -- nothing new fires.
    result2 = app.dispatch_reviews()
    assert 100 in result2.data["escalated_skipped"]
    assert result2.data["empty_diff_operator_queued"] == []
    assert len(_routed_events(app)) == 1
    assert app.gh.labels_added.count((10, app.config.labels.operator_queue)) == 1

    # The head moves (still empty): the real sweep clears, the next pass
    # re-routes exactly once for the new head.
    _move_head(app, tmp_path, "sha-200")
    assert app._deescalate_mechanical_issue(10)["cleared"] is True

    result3 = app.dispatch_reviews()
    assert result3.data["empty_diff_operator_queued"] == [100]
    routed = _routed_events(app)
    assert len(routed) == 2
    assert routed[1]["payload"]["reason"] == EMPTY_DIFF_ESCALATION_REASON
    assert routed[1]["payload"]["packet_head_sha"] == "sha-200"
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"][EMPTY_DIFF_ROUTED_HEAD_FIELD] == "sha-200"
    assert int(state["prs"]["100"].get("review_dispatch_attempt_count", 0)) == 0
    assert state["issues"]["10"]["status"] == "escalated"

    # And that head is itself once-only.
    result4 = app.dispatch_reviews()
    assert result4.data["empty_diff_operator_queued"] == []
    assert len(_routed_events(app)) == 2


def test_sweep_skips_empty_diff_escalation_while_head_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #2899 (durable exit), case (a): the issue's own scenario -- an
    open, mergeable, janitor-green zero-diff PR -- must NOT be auto-cleared by
    the mechanical de-escalation sweep while the live head still equals the
    head the route fired for. The issue stays escalated and operator-queued,
    and the sweep writes no state and no event for it."""
    app = _empty_diff_issue_linked_app(tmp_path)
    _fail_if_launched(monkeypatch)
    app.dispatch_reviews()
    assert len(_routed_events(app)) == 1

    state_before = load_state(app.paths.state_file)
    events_before = _all_event_ids(app)
    added_before = list(app.gh.labels_added)
    removed_before = list(app.gh.labels_removed)

    outcome = app._deescalate_mechanical_issue(10)
    assert outcome == {"skipped": "empty_diff_head_unchanged", "issue_number": 10}
    assert app._deescalate_mechanical_issue(10) == outcome  # stable, not one-shot

    state = load_state(app.paths.state_file)
    assert state["issues"]["10"] == state_before["issues"]["10"]
    assert state["prs"]["100"] == state_before["prs"]["100"]
    assert state["issues"]["10"]["status"] == "escalated"
    assert state["issues"]["10"]["escalation_reason"] == EMPTY_DIFF_ESCALATION_REASON
    assert state["issues"]["10"]["reason_class"] == "mechanical"
    assert "auto_deescalation_count" not in state["issues"]["10"]
    assert _all_event_ids(app) == events_before  # no new events
    assert app.gh.labels_added == added_before
    assert app.gh.labels_removed == removed_before
    assert (10, app.config.labels.operator_queue) in app.gh.labels_added
    assert (10, app.config.labels.human_needed) not in app.gh.labels_added

    # The periodic sweep reports the skip under its own histogram bucket and
    # clears nothing.
    app._maybe_deescalate_mechanical()
    (completed,) = query_events(app.paths.state_file, kind="deescalation_pass_completed")
    assert completed["payload"]["cleared"] == []
    assert completed["payload"]["skipped"] == {"empty_diff_head_unchanged": 1}

    # Dispatch stays blinded by the still-standing escalation: no re-route.
    result = app.dispatch_reviews()
    assert 100 in result.data["escalated_skipped"]
    assert result.data["empty_diff_operator_queued"] == []
    assert len(_routed_events(app)) == 1


def test_sweep_clears_empty_diff_escalation_once_head_moves_and_regained_diff_is_dispatched(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #2899 AC3, case (b): once the head moves the sweep clears as
    usual, and a PR that regained a diff on the new head is dispatched for
    review -- the head-scoped marker never stands between a plus-diff head and
    the reviewer."""
    app = _empty_diff_issue_linked_app(tmp_path)
    _fail_if_launched(monkeypatch)
    app.dispatch_reviews()
    assert app._deescalate_mechanical_issue(10)["skipped"] == "empty_diff_head_unchanged"

    _move_head(app, tmp_path, "sha-200", diff="diff --git a/foo.py b/foo.py\n+print('hello')\n")
    cleared = app._deescalate_mechanical_issue(10)
    assert cleared["cleared"] is True
    assert cleared["cleared_condition"] == EMPTY_DIFF_ESCALATION_REASON
    state = load_state(app.paths.state_file)
    assert state["issues"]["10"]["status"] == PASSIVE_OPEN_STATUS
    assert "escalation_reason" not in state["issues"]["10"]
    assert (10, app.config.labels.pr_open) in app.gh.labels_added
    assert state["prs"]["100"]["status"] == PASSIVE_OPEN_STATUS

    recorded: list[int] = []
    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", _recording_launch(recorded))
    result = app.dispatch_reviews()

    assert result.ok is True
    assert recorded == [100]
    assert result.data["launched_count"] == 1
    assert result.data["empty_diff_operator_queued"] == []
    # Still exactly one escalation event -- the regained-diff pass raised no new one.
    assert len(_routed_events(app)) == 1
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert int(state["prs"]["100"].get("review_dispatch_attempt_count", 0)) == 1


def test_operator_rearm_of_still_empty_head_does_not_reroute(monkeypatch, tmp_path: Path) -> None:
    """The marker's reason to exist: a real ``charlie unescalate`` of a
    still-empty head clears the escalation, and the next pass must not re-fire
    the escalation the operator just cleared for the same bytes (the marker is
    deliberately outside UNESCALATE_PR_RESET_FIELDS)."""
    app = _empty_diff_issue_linked_app(tmp_path)
    _fail_if_launched(monkeypatch)
    app.dispatch_reviews()

    result = app.unescalate(pr_number=100)
    assert result.ok is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["status"] != "escalated"
    assert state["prs"]["100"][EMPTY_DIFF_ROUTED_HEAD_FIELD] == "sha-100"

    after = app.dispatch_reviews()
    assert after.data["skipped_empty_diff"] == [100]
    assert after.data["empty_diff_operator_queued"] == []
    assert len(_routed_events(app)) == 1
    assert app.gh.labels_added.count((10, app.config.labels.operator_queue)) == 1


def test_sweep_clears_when_the_escalation_reason_is_not_the_empty_diff_route(
    monkeypatch, tmp_path: Path
) -> None:
    """The sweep's skip is scoped to ``empty_diff_zero_file``: the same PR with
    the marker still equal to the live head clears normally when the issue's
    escalation reason is a different mechanical one."""
    app = _empty_diff_issue_linked_app(tmp_path)
    _fail_if_launched(monkeypatch)
    app.dispatch_reviews()

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["10"]["escalation_reason"] = "session_failed_escalated"
        save_state(app.paths.state_file, state)

    assert app._deescalate_mechanical_issue(10)["cleared"] is True


def test_sweep_clears_when_no_route_marker_names_a_head(monkeypatch, tmp_path: Path) -> None:
    """An escalation with the empty-diff reason but no usable marker (e.g. the
    packet head was unreadable at route time) names no head to be unchanged
    against: the sweep falls through and clears, bounded by the sweep's own
    ``max_auto_deescalations`` cap instead of parking silently forever.

    Run twice: with a live head (marker absent != live head), and with the live
    head ALSO absent from ``pr_view`` -- ``None == None`` must not read as "head
    unchanged", which is what the truthy-marker guard is for."""
    app = _empty_diff_issue_linked_app(tmp_path)
    _fail_if_launched(monkeypatch)
    app.dispatch_reviews()

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"].pop(EMPTY_DIFF_ROUTED_HEAD_FIELD)
        save_state(app.paths.state_file, state)

    assert app._deescalate_mechanical_issue(10)["cleared"] is True

    # Re-escalate (marker still absent), then drop the live head too.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["10"].update(
            status="escalated", escalation_reason=EMPTY_DIFF_ESCALATION_REASON
        )
        state["issues"]["10"]["reason_class"] = "mechanical"
        state["prs"]["100"]["status"] = "escalated"
        save_state(app.paths.state_file, state)
    app.gh.prs[0].pop("headRefOid")

    assert app._deescalate_mechanical_issue(10).get("skipped") != "empty_diff_head_unchanged"


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
    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", _recording_launch(recorded))

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["skipped_empty_diff"] == []
    assert result.data["empty_diff_operator_queued"] == []
    assert _routed_events(app) == []
    # The missing packet is not a zero-delta PR: the dispatch proceeds under
    # the normal gate chain (claim + launch), exactly as before #2899.
    assert result.data["launched_count"] == 1
    assert recorded == [100]


# --- route_empty_diff_prs: direct tests on a crafted state ---------------------


def _gate(tmp_path: Path) -> WriteGate:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    return WriteGate(dry_run=False, state_path=state_path, repo="repo")


def _state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {"version": 1, "issues": {}, "prs": {}, "events": []}
    state.update(overrides)
    return state


@pytest.mark.parametrize("sink_status", sorted(SINK_STATUSES))
@pytest.mark.parametrize("sink_on", ["pr", "issue"])
def test_route_is_silent_when_pr_or_issue_already_in_a_sink_status(
    tmp_path: Path, sink_on: str, sink_status: str
) -> None:
    """The PR or its issue was escalated/blocked since the selection snapshot
    (another lane won the race): stay silent -- no state mutation, no event, no
    label edge -- rather than clobbering that lane's escalation reason."""
    state = _state(
        prs={
            "100": {
                "number": 100,
                "issue_number": 10,
                "status": sink_status if sink_on == "pr" else "open_passive",
            }
        },
        issues={
            "10": {
                "number": 10,
                "status": sink_status if sink_on == "issue" else "open_passive",
                "escalation_reason": "other_lane_reason",
            }
        },
    )
    before = copy.deepcopy(state)

    outcome = route_empty_diff_prs(
        state, [EmptyDiffSkip(100, 10, "sha-100")], write_gate=_gate(tmp_path)
    )

    assert outcome.state == before
    assert outcome.routed_prs == ()
    assert outcome.label_edges == ()
    assert query_events(tmp_path / "state.json", kind="review_dispatch_escalated") == []


def test_route_without_a_linked_issue_escalates_the_pr_only(tmp_path: Path) -> None:
    """``issue_num is None`` (cross-repo/fork PR): the PR is escalated and
    marked, the event is written with ``issue_number=None``, no issue record is
    created, there is NO label edge, and the PR is still reported as routed
    (it feeds ``empty_diff_operator_queued``)."""
    state = _state(prs={"100": {"number": 100, "status": "open_passive"}})

    outcome = route_empty_diff_prs(
        state, [EmptyDiffSkip(100, None, "sha-100")], write_gate=_gate(tmp_path)
    )

    assert outcome.label_edges == ()
    assert outcome.routed_prs == (100,)
    pr = outcome.state["prs"]["100"]
    assert pr["status"] == "escalated"
    assert pr["escalation_reason"] == EMPTY_DIFF_ESCALATION_REASON
    assert pr[EMPTY_DIFF_ROUTED_HEAD_FIELD] == "sha-100"
    assert outcome.state["issues"] == {}
    (event,) = query_events(tmp_path / "state.json", kind="review_dispatch_escalated")
    assert event["payload"] == {
        "pr_number": 100,
        "issue_number": None,
        "attempt_count": 0,
        "reason": EMPTY_DIFF_ESCALATION_REASON,
        "packet_head_sha": "sha-100",
    }


def test_route_with_a_linked_issue_reports_the_label_edge_and_leaves_attempts_alone(
    tmp_path: Path,
) -> None:
    state = _state(
        prs={"100": {"number": 100, "status": "open_passive", "review_dispatch_attempt_count": 2}}
    )

    outcome = route_empty_diff_prs(
        state, [EmptyDiffSkip(100, 10, "sha-100")], write_gate=_gate(tmp_path)
    )

    assert outcome.routed_prs == (100,)
    assert outcome.label_edges == ((100, 10),)
    assert outcome.state["issues"]["10"]["status"] == "escalated"
    assert outcome.state["issues"]["10"]["reason_class"] == "mechanical"
    assert outcome.state["prs"]["100"]["review_dispatch_attempt_count"] == 2
    (event,) = query_events(tmp_path / "state.json", kind="review_dispatch_escalated")
    assert event["payload"]["attempt_count"] == 2


def test_route_dedups_by_head_and_reroutes_a_new_head(tmp_path: Path) -> None:
    """A PR record already carrying the marker for this exact head is silent
    (no mutation, no event); a different head routes."""
    gate = _gate(tmp_path)
    state = _state(
        prs={
            "100": {
                "number": 100,
                "issue_number": 10,
                "status": "open_passive",
                EMPTY_DIFF_ROUTED_HEAD_FIELD: "sha-100",
            }
        },
        issues={"10": {"number": 10, "status": "open_passive"}},
    )
    before = copy.deepcopy(state)

    same = route_empty_diff_prs(state, [EmptyDiffSkip(100, 10, "sha-100")], write_gate=gate)
    assert same.state == before
    assert same.routed_prs == ()

    moved = route_empty_diff_prs(state, [EmptyDiffSkip(100, 10, "sha-200")], write_gate=gate)
    assert moved.routed_prs == (100,)
    assert moved.state["prs"]["100"][EMPTY_DIFF_ROUTED_HEAD_FIELD] == "sha-200"


def test_route_with_an_unreadable_head_escalates_but_records_no_marker(tmp_path: Path) -> None:
    """No head to scope by: the route still fires, but writes no marker -- an
    unscoped marker could never be told apart from a new head, the silent-forever
    shape the route exists to remove."""
    state = _state(prs={"100": {"number": 100, "status": "open_passive"}})

    outcome = route_empty_diff_prs(
        state, [EmptyDiffSkip(100, 10, None)], write_gate=_gate(tmp_path)
    )

    assert outcome.routed_prs == (100,)
    assert EMPTY_DIFF_ROUTED_HEAD_FIELD not in outcome.state["prs"]["100"]


def test_route_handles_a_mixed_batch_in_input_order(tmp_path: Path) -> None:
    state = _state(
        prs={
            "100": {"number": 100, "status": "open_passive"},
            "101": {"number": 101, "status": "escalated"},
            "102": {"number": 102, "status": "open_passive"},
        }
    )

    outcome = route_empty_diff_prs(
        state,
        [
            EmptyDiffSkip(100, 10, "sha-a"),
            EmptyDiffSkip(101, 11, "sha-b"),
            EmptyDiffSkip(102, None, "sha-c"),
        ],
        write_gate=_gate(tmp_path),
    )

    assert outcome.routed_prs == (100, 102)
    assert outcome.label_edges == ((100, 10),)
