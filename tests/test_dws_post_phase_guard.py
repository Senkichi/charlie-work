"""Review-drain guard atomicity (dws-6 review fixes).

The old review drain checked ``status == "dispatched"`` and the unchanged
reviewed head, flipped the status, and recorded the event in ONE ``state_lock``;
the label edge ran only for flips that committed. The post phase re-checks the
same guards under a fresh lock, so a concurrent writer that lands between the
``review()`` result and the guarded write must suppress the dependent event,
recovery and label edge too -- never leave the label and state.json disagreeing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from _dead_worker_sweep_characterization_fixtures import events_of, issue_entry, run_sweep
from _orphan_sweep_fixtures import _dead_worker_rework_bed, _write_outcome
from charlie_work import rework_outcome
from charlie_work.dead_worker_sweep import apply as sweep_apply
from charlie_work.dead_worker_sweep.model import Review
from charlie_work.state import load_state, save_state
from charlie_work.command_result import CommandResult

_COMPLETED_OUTCOME = {
    "push_succeeded": True,
    "pr_created": False,
    "head_sha": "abc123",
    "pr_body": "Closes #207\n\nCorrected PR body per review.",
}


def _refused(pr_number: int) -> CommandResult:
    return CommandResult(False, "transient refusal", {"pr_number": pr_number})


def _concurrent_writer(paths: Any, mutate: Any) -> Any:
    """A ``serve`` wrapper that lands ``mutate`` right after ``Review`` is served."""
    real_serve = sweep_apply.serve

    def serve(ctx: Any, request: Any) -> Any:
        result = real_serve(ctx, request)
        if isinstance(request, Review):
            state = load_state(paths.state_file)
            mutate(state)
            save_state(paths.state_file, state)
        return result

    return serve


def _run_with_gap(tmp_path: Path, mutate: Any, monkeypatch) -> tuple[Any, Any, Any]:
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)
    _write_outcome(paths, tmp_path, _COMPLETED_OUTCOME)
    run_sweep(
        tmp_path,
        paths,
        config,
        gh,
        review_callback=_refused,
        patches=(
            patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"),
            patch.object(sweep_apply, "serve", _concurrent_writer(paths, mutate)),
        ),
        monkeypatch=monkeypatch,
    )
    return config, paths, gh


def test_status_moved_in_gap_suppresses_rework_label_and_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    def escalate(state: dict[str, Any]) -> None:
        state["issues"]["207"]["status"] = "escalated"

    config, paths, gh = _run_with_gap(tmp_path, escalate, monkeypatch=monkeypatch)

    assert issue_entry(paths, 207)["status"] == "escalated"
    assert (207, config.labels.needs_rework) not in gh.labels_added
    assert events_of(paths, "orphaned_worker_recovered") == []


def test_reviewed_head_moved_in_gap_suppresses_rework_label_and_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    def new_verdict(state: dict[str, Any]) -> None:
        state["prs"]["100"]["reviewed_head_sha"] = "zzz999"

    config, paths, gh = _run_with_gap(tmp_path, new_verdict, monkeypatch=monkeypatch)

    assert issue_entry(paths, 207)["status"] == "dispatched"
    assert (207, config.labels.needs_rework) not in gh.labels_added
    assert events_of(paths, "orphaned_worker_recovered") == []


# ---------------------------------------------------------------- nits from the review


def _advance_head(gh: Any) -> None:
    gh.prs[0]["headRefOid"] = "def456"


def test_review_route_failure_is_audit_only_not_a_ring_event(tmp_path: Path, monkeypatch) -> None:
    """N1: the original wrote it with ``log_event`` (events.db), never the state ring."""
    from charlie_work.instrumentation import query_events

    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)
    _advance_head(gh)

    def boom(pr_number: int) -> CommandResult:
        raise RuntimeError("reviewer down")

    run_sweep(tmp_path, paths, config, gh, review_callback=boom, monkeypatch=monkeypatch)

    assert events_of(paths, "orphaned_worker_review_route_failed") == []
    rows = query_events(paths.state_file, kind="orphaned_worker_review_route_failed")
    assert len(rows) == 1


def test_drift_anchor_is_stamped_after_review_returns(tmp_path: Path, monkeypatch) -> None:
    """N2: ``orphan_drift_at`` feeds the #654 backstop; it must not predate review()."""
    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)
    _advance_head(gh)
    clock = {"now": "2026-09-30T10:00:00Z"}

    def slow_refusal(pr_number: int) -> CommandResult:
        clock["now"] = "2026-09-30T10:10:00Z"  # the reviewer took ten minutes
        return _refused(pr_number)

    run_sweep(
        tmp_path,
        paths,
        config,
        gh,
        review_callback=slow_refusal,
        patches=(patch("charlie_work.workflow.utc_now", lambda: clock["now"]),),
        monkeypatch=monkeypatch,
    )

    assert issue_entry(paths, 207)["orphan_drift_at"] == "2026-09-30T10:10:00Z"


def test_later_lane_refetches_an_empty_issue_listing() -> None:
    """N4: a transient empty listing must not blank every later consumer."""
    from types import SimpleNamespace

    from charlie_work.dead_worker_sweep.apply_requests_pre import fetch_open_issues
    from charlie_work.dead_worker_sweep.model import FetchOpenIssues

    listings = [[], [{"number": 7, "labels": []}]]
    gh = SimpleNamespace(issue_list=lambda state: listings.pop(0))
    ctx = SimpleNamespace(gh=gh, issues=None)

    assert fetch_open_issues(ctx, FetchOpenIssues("no_pr")).issues_by_number == {}
    assert fetch_open_issues(ctx, FetchOpenIssues("no_pr")).issues_by_number == {}  # no_pr: once
    assert list(fetch_open_issues(ctx, FetchOpenIssues("unreviewed")).issues_by_number) == [7]
    assert not listings  # and a populated cache is not fetched again
    assert list(fetch_open_issues(ctx, FetchOpenIssues("live_handoff")).issues_by_number) == [7]


def test_applied_heads_are_read_per_route() -> None:
    """N3: the original re-read the map before each completed-outcome route.

    Drive ``_route_flow`` for two completed-outcome routes against an applied-heads
    map that matches only the first: the flow must ask for heads once per route
    (keyed by that route's issue) and route ONLY the route whose live head matches.
    """
    from charlie_work.dead_worker_sweep.decide_post import _route_flow
    from charlie_work.dead_worker_sweep.model import ReadAppliedHeads, ReviewRoute

    def route(issue: int, pr: int, live: str) -> ReviewRoute:
        return ReviewRoute(issue, pr, None, live, f"fp{issue}", "dead_worker_completed_outcome")

    applied_map = {"1": "head-1", "2": "stale-head"}
    requests: list[Any] = []
    for r in (route(1, 11, "head-1"), route(2, 22, "head-2")):
        flow = _route_flow(None, r, [], [])  # type: ignore[arg-type]  # facts is unused
        request = next(flow)
        requests.append(request)
        if isinstance(request, ReadAppliedHeads):
            try:
                requests.append(flow.send(applied_map))
            except StopIteration:
                pass
    assert requests == [
        ReadAppliedHeads(1),
        Review(1, 11, "dead_worker_completed_outcome"),
        ReadAppliedHeads(2),
    ]


# ------------------------------------------------- #2113: status and event share a lock window


def test_append_event_failure_leaves_neither_status_nor_event(tmp_path: Path, monkeypatch) -> None:
    """The audit event rides the guarded write: if ``append_event`` raises, the status
    flip is not saved, there is no recovery row and no label edge, and the sweep
    does not swallow it."""
    import pytest

    from charlie_work.write_gate import WriteGate

    real_append = WriteGate.append_event

    def failing_append(self: Any, data: Any, kind: str, *args: Any, **kwargs: Any) -> Any:
        if kind == "orphaned_worker_recovered":
            raise OSError("disk full")
        return real_append(self, data, kind, *args, **kwargs)

    config, paths, gh, _ = _dead_worker_rework_bed(tmp_path)
    _write_outcome(paths, tmp_path, _COMPLETED_OUTCOME)
    with pytest.raises(OSError, match="disk full"):
        run_sweep(
            tmp_path,
            paths,
            config,
            gh,
            review_callback=_refused,
            patches=(
                patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"),
                patch.object(WriteGate, "append_event", failing_append),
            ),
            monkeypatch=monkeypatch,
        )

    assert issue_entry(paths, 207)["status"] == "dispatched"
    assert events_of(paths, "orphaned_worker_recovered") == []
    assert (207, config.labels.needs_rework) not in gh.labels_added


def test_recovery_event_and_status_land_together(tmp_path: Path, monkeypatch) -> None:
    """Positive control for the test above: the same bed, unpatched, records both."""
    config, paths, gh = _run_with_gap(tmp_path, lambda state: None, monkeypatch=monkeypatch)

    assert issue_entry(paths, 207)["status"] == "rework_requested"
    assert len(events_of(paths, "orphaned_worker_recovered")) == 1
