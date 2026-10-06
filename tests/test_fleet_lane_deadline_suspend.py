"""Deadline refusal semantics at the irreversible/cancellation seams (#1948).

``test_fleet_lane_deadline_refusal.py`` covers the refusal *contract*; this
module covers the places a refusal must NOT behave like an ordinary abort:

* ``_loop_impl`` -- the armed hook is disarmed in a ``finally`` before the
  post-pass epilogue, so observability calls (queue-impact's cold
  ``gh.issue_list``, the secret-refusal digest) cannot raise a
  ``BaseException`` refusal past their ``except Exception`` containment.
* ``merge_ready`` -- the whole post-``merge_pr`` region (label transition,
  close_issue, delete_branch, plus the ``_update_open_agent_prs`` /
  ``cancel_superseded_runs`` deferral tail) runs under
  ``pass_deadline_suspended``: refusing there cannot undo a landed merge --
  it can only strand durable bookkeeping or silently drop the merge's
  events/``merges[]`` entry. The ``add_pr_label`` mergequeue handoff is NOT
  suspended -- it is the reversible step itself, so a refusal there
  propagates cleanly.
* ``dispatch`` / ``dispatch_rework`` -- a refusal between the durable
  ``dispatch_pending`` claim write and ``dispatch_sessions`` must strand
  nothing: no in-progress label is applied without a worker, and the claim
  is recoverable via ``is_claim_stale`` / reconcile's stale-claim sweep.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any

import pytest
from _deadline_fixtures import (
    _assert_merge_finalized,
    _build_app,
    _DeadlineAwareGitHub,
    _dispatch_pending_claim,
    _merge_ready_app,
    _ok_dispatch_sessions,
)
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    RunnersConfig,
    WorkerRoleConfig,
)
from charlie_work.host.fakes import FakeWorkerLauncher
from charlie_work.instrumentation import query_events
from charlie_work.pass_deadline import (
    PassDeadlineExceeded,
    pass_deadline_spent,
    set_pass_deadline_exceeded,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import empty_state, load_state, save_state
from charlie_work.workflow import OrchestratorApp


# ---------------------------------------------------------------------------
# _loop_impl epilogue: the hook is disarmed before post-pass observability
# ---------------------------------------------------------------------------


def test_loop_impl_epilogue_runs_cold_issue_list_after_spent_deadline(
    tmp_path: Path,
) -> None:
    """A spent hook must not survive into _loop_impl's post-pass epilogue.

    The epilogue's operator-queue impact check calls ``gh.issue_list`` under
    ``except Exception`` containment -- which cannot catch a ``BaseException``
    refusal. Without the ``finally`` disarm, an armed+spent hook makes that
    cold fetch raise ``PassDeadlineExceeded`` clean past the containment, and
    ``loop_completed`` / ``record_loop_pass`` / the status snapshot never run.
    Here the pass budget is spent before the body even starts (``preflight``
    defers), so the *only* gh call the pass makes is the epilogue's.
    """
    fake_gh = _DeadlineAwareGitHub()
    app, paths, fake_gh = _build_app(tmp_path / "repo", gh=fake_gh)
    # A sink member so the queue-impact check reaches its gh.issue_list fetch
    # (empty roots short-circuit before the call).
    state = load_state(paths.state_file)
    state["issues"]["99"] = {"number": 99, "status": "escalated"}
    save_state(paths.state_file, state)

    # Arm the hook spent, the way _run_fleet_repo_lane leaves a lane's client
    # when the pass budget runs out mid-lane.
    set_pass_deadline_exceeded(fake_gh, lambda: True)

    result = app.loop(0, merge=False, deadline_exceeded=lambda: True)

    assert result.ok is True
    assert result.data["deadline_deferred"] is True
    # The cold issue_list ran instead of being refused -- the hook was
    # disarmed by _loop_impl's finally before the epilogue.
    assert "issue_list" in fake_gh.gh_calls
    assert fake_gh.deadline_refusals == []
    assert pass_deadline_spent(fake_gh) is False, "hook must be disarmed after the pass"
    completed = query_events(paths.state_file, kind="loop_completed")
    assert len(completed) == 1


# ---------------------------------------------------------------------------
# merge_ready: the post-merge_pr finalize trio is refusal-suspended
# ---------------------------------------------------------------------------


def test_merge_ready_transition_runs_despite_deadline_spent_at_merge(
    tmp_path: Path,
) -> None:
    """Deadline trips on merge_pr's success: the 'merged' transition still runs.

    ``merge_pr`` is the irreversible step -- once it lands, refusing the
    label transition cannot undo anything; it would only strand durable
    bookkeeping for reconcile to redo. ``pass_deadline_suspended`` disarms
    the hook for exactly that window, then re-arms it.
    """
    app, paths, fake_gh = _merge_ready_app(tmp_path)
    # Spent iff merge_pr has already landed: every pre-merge check runs on a
    # live budget; the first post-merge gh call (transition's add_issue_label)
    # is where the deadline is observed.
    set_pass_deadline_exceeded(fake_gh, lambda: bool(fake_gh.merged))

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    _assert_merge_finalized(app, paths, fake_gh)
    assert pass_deadline_spent(fake_gh) is True, "suspend must re-arm the hook on exit"


def test_merge_ready_close_issue_runs_despite_deadline_spent_at_transition(
    tmp_path: Path,
) -> None:
    """Deadline trips after the label transition: close_issue still runs."""
    app, paths, fake_gh = _merge_ready_app(tmp_path)
    # Trip on the "merged" transition's own marker -- the ``agent:done`` add
    # is unique to that edge, so the deadline is spent exactly when the
    # transition has landed and close_issue is the next gh call.
    set_pass_deadline_exceeded(
        fake_gh,
        lambda: any(label == app.config.labels.done for _, label in fake_gh.labels_added),
    )

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    _assert_merge_finalized(app, paths, fake_gh)


def test_merge_ready_delete_branch_runs_despite_deadline_spent_at_close(
    tmp_path: Path,
) -> None:
    """Deadline trips after close_issue: delete_branch still runs."""
    app, paths, fake_gh = _merge_ready_app(tmp_path)
    set_pass_deadline_exceeded(fake_gh, lambda: bool(fake_gh.closed_issues))

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    _assert_merge_finalized(app, paths, fake_gh)


def test_merge_ready_post_merge_tail_runs_despite_deadline_spent(
    tmp_path: Path,
) -> None:
    """Deadline trips on merge_pr: the deferral tail still runs to completion.

    Under ``update_branch_strategy="front_of_train"`` (the default) the
    post-``merge_pr`` tail -- ``_update_open_agent_prs`` and
    ``cancel_superseded_runs`` -- sits past the irreversible step. A refusal
    there propagates out of merge_ready before the ``merge_ready`` /
    ``merge_succeeded`` events and the ``merges[]`` result are produced, and
    the idempotency short-circuit makes that loss permanent. The tail shares
    the finalize trio's ``pass_deadline_suspended`` window: the calls run,
    the records land, and the armed predicate still reports spent for the
    merge-counter guard and the pass's deferred marker.
    """
    app, paths, fake_gh = _merge_ready_app(
        tmp_path,
        update_branch_strategy="front_of_train",
        runners=RunnersConfig(
            enabled=True,
            cancel_superseded_main_runs=True,
            workflow_name="CI",
        ),
    )
    # A second approved agent PR riding the merge train: merging 456
    # advances the base tip, leaving 789 stale -- so the tail's
    # front_of_train update does real work (pr_update_branch), not a no-op.
    fake_gh.issues.append(
        {
            "number": 789,
            "title": "Another fix",
            "url": "https://example.test/issues/789",
            "labels": [{"name": "automated-ready"}],
            "state": "OPEN",
        }
    )
    fake_gh.prs = [
        *fake_gh.prs,
        {
            "number": 789,
            "title": "Fix #789: other",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-789-another-fix",
            "baseRefName": "main",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #789\n\nTests: covered.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    assert app.record_review(
        789, "approved", summary="lgtm", verdict_provenance="fresh_llm_review"
    ).ok
    # Spent the instant merge_pr lands: every pre-merge check runs on a live
    # budget; the tail's first gh call (the train sweep's pr_list) is where
    # the deadline is observed.
    set_pass_deadline_exceeded(fake_gh, lambda: bool(fake_gh.merged))

    result = app.merge_ready(456, merge=True)

    assert result.ok is True
    _assert_merge_finalized(app, paths, fake_gh)
    # The tail ran to completion past the spent deadline -- no refusal
    # anywhere in the post-merge sequence.
    assert fake_gh.deadline_refusals == []
    assert fake_gh.pr_update_branch_calls == [789]
    assert result.data["update_open_prs_results"][0]["pr_number"] == 789
    assert result.data["update_open_prs_results"][0]["updated"] is True
    assert result.data["cancel_superseded_runs_results"]["total_queued"] == 0
    assert pass_deadline_spent(fake_gh) is True, "suspend must re-arm the hook on exit"
    # merge_succeeded is recorded exactly once -- and stays once across the
    # idempotent re-entry _assert_merge_finalized already performed (the
    # permanent-loss regression this fix closes).
    assert len(query_events(paths.state_file, kind="merge_succeeded")) == 1
    assert len(query_events(paths.state_file, kind="merge_ready")) == 1
    # The deferred tail is not misread as a merge failure: no counter bump,
    # no alarm, no escalation.
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 0
    assert state["issues"]["123"]["status"] == "closed"
    assert "escalation_reason" not in state["issues"]["123"]
    assert state["prs"]["456"].get("status") == "merged"
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert alarm_events == []


def test_merge_ready_mergequeue_handoff_refusal_propagates_cleanly(
    tmp_path: Path,
) -> None:
    """A refusal AT the add_pr_label handoff propagates; nothing is half-done.

    The mergequeue handoff is NOT inside the suspend window -- no
    irreversible step has happened yet, so refusal is correct there: the
    label add is the handoff itself, it never landed, and the next pass
    re-derives readiness from scratch. Assert the refusal escapes merge_ready
    without touching the merge counters, the alarm, or the mergequeue status.
    """
    # Skip-line off: its issue read would otherwise be the first refusal site
    # (pinned by the TIS-CW-7 test below).
    app, paths, fake_gh = _merge_ready_app(
        tmp_path, mergequeue_label="mergequeue", mergequeue_skip_line_label=None
    )
    spent = {"hit": False}
    real_issue_view = fake_gh.issue_view

    def _issue_view_then_spend(number: int) -> Any:
        result = real_issue_view(number)
        spent["hit"] = True
        return result

    fake_gh.issue_view = _issue_view_then_spend  # type: ignore[method-assign]
    set_pass_deadline_exceeded(fake_gh, lambda: spent["hit"])

    with pytest.raises(PassDeadlineExceeded):
        app.merge_ready(456, merge=True)

    assert fake_gh.deadline_refusals == ["add_pr_label"]
    assert fake_gh.pr_labels_added == [], "the refused handoff must not appear applied"
    assert fake_gh.merged == []
    state = load_state(paths.state_file)
    pr_state = state["prs"].get("456", {})
    assert pr_state.get("consecutive_failed_merge_attempts", 0) == 0
    assert pr_state.get("merge_attempt_alarm", False) is False
    assert pr_state.get("status") != "mergequeue"
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert alarm_events == []


def test_merge_ready_skip_line_read_refusal_propagates_cleanly(tmp_path: Path) -> None:
    """TIS-CW-7: with skip-line on, the hand-off's issue read is the refusal site.

    It runs before the queue label, so a refusal there leaves nothing applied,
    exactly like a refusal at the label add.
    """
    app, paths, fake_gh = _merge_ready_app(tmp_path, mergequeue_label="mergequeue")
    fake_gh.issues[0]["labels"] = [{"name": "automated-ready"}, {"name": "priority:critical"}]
    spent = {"hit": False}
    real_issue_view = fake_gh.issue_view

    def _issue_view_then_spend(number: int) -> Any:
        result = real_issue_view(number)
        spent["hit"] = True
        return result

    fake_gh.issue_view = _issue_view_then_spend  # type: ignore[method-assign]
    set_pass_deadline_exceeded(fake_gh, lambda: spent["hit"])

    with pytest.raises(PassDeadlineExceeded):
        app.merge_ready(456, merge=True)

    assert fake_gh.deadline_refusals == ["issue_view"]
    assert fake_gh.pr_labels_added == []
    assert fake_gh.merged == []
    pr_state = load_state(paths.state_file)["prs"].get("456", {})
    assert pr_state.get("status") != "mergequeue"


# ---------------------------------------------------------------------------
# dispatch / dispatch_rework: a refusal between claim and launch must not
# strand the issue -- the claim is durable state, the in-progress label is
# only applied after a successful launch, and the claim is sweepable.
# ---------------------------------------------------------------------------


def test_dispatch_refusal_between_claim_and_launch_leaves_recoverable_claim(
    tmp_path: Path, fake_host
) -> None:
    """A BaseException refusal after the claim write strands nothing unsafe.

    The claim is ``dispatch_pending`` in state.json -- durable, and
    recoverable two ways: a fresh claim makes the issue invisible to
    re-dispatch until ``is_claim_stale`` ages it out (~30 min), and
    reconcile's ``stale_dispatch_pending_claim`` sweep clears it. Critically
    the ``agent:in-progress`` label is only applied in the second lock AFTER
    a successful ``dispatch_sessions`` -- a refusal in the claim->launch
    window can never leave an in-progress label on an issue with no worker.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(0)")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    save_state(paths.state_file, empty_state())
    fake_gh = _DeadlineAwareGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    dispatch_calls: list[list[Any]] = []

    def _spy_dispatch_sessions(*args: Any, **kwargs: Any) -> list[Any]:
        requests = list(args[-1])
        dispatch_calls.append(requests)
        return _ok_dispatch_sessions(requests)

    fake_host(worker_launch=FakeWorkerLauncher([_spy_dispatch_sessions]))

    # Spent exactly when the claim exists: the refusal lands between the
    # claim write (first lock) and the worker launch (dispatch_sessions).
    set_pass_deadline_exceeded(fake_gh, lambda: _dispatch_pending_claim(paths))

    with pytest.raises(PassDeadlineExceeded):
        app.dispatch(limit=1)

    assert fake_gh.deadline_refusals, "a post-claim gh call must be the refusal site"
    assert dispatch_calls == [], "dispatch_sessions must never run"
    # No label mutation at all: nothing stranded as in-progress.
    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []
    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry["status"] == "dispatch_pending"
    assert entry["dispatch_pending_at"]

    # Recovery: the stale claim is re-dispatchable (dispatch_pending_at aged
    # past is_claim_stale's 30-minute window; derived from the wall clock).
    entry["dispatch_pending_at"] = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=31)
    ).isoformat()
    save_state(paths.state_file, state)
    set_pass_deadline_exceeded(fake_gh, None)

    app.dispatch(limit=1)

    assert len(dispatch_calls) == 1, "worker launch must run on the retry"
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"
    assert (123, config.labels.in_progress) in fake_gh.labels_added


def test_dispatch_rework_refusal_between_claim_and_launch_leaves_recoverable_claim(
    tmp_path: Path, fake_host
) -> None:
    """Same claim->launch contract for the rework lane.

    Rework candidates are selected by ``status == "rework_requested"``, so a
    stranded ``dispatch_pending`` claim is not re-selected by
    ``dispatch_rework`` itself -- recovery is reconcile's
    ``stale_dispatch_pending_claim`` sweep (which clears the entry, letting
    the open-PR label-heal lane converge the issue back to the review lane).
    Assert that detect_drift actually sees the stranded claim, and again
    that no ``agent:in-progress`` label was applied without a worker.
    """
    from charlie_work.reconcile import detect_drift

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(0)")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    state = empty_state()
    state["issues"]["123"] = {
        "number": 123,
        "title": "Fix search",
        "url": "https://example.test/issues/123",
        "status": "rework_requested",
    }
    save_state(paths.state_file, state)

    fake_gh = _DeadlineAwareGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.needs_rework}]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")

    dispatch_calls: list[list[Any]] = []

    def _spy_dispatch_sessions(*args: Any, **kwargs: Any) -> list[Any]:
        requests = list(args[-1])
        dispatch_calls.append(requests)
        return _ok_dispatch_sessions(requests)

    fake_host(worker_launch=FakeWorkerLauncher([_spy_dispatch_sessions]))

    set_pass_deadline_exceeded(fake_gh, lambda: _dispatch_pending_claim(paths))

    with pytest.raises(PassDeadlineExceeded):
        app.dispatch_rework()

    assert fake_gh.deadline_refusals, "a post-claim gh call must be the refusal site"
    assert dispatch_calls == [], "dispatch_sessions must never run"
    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []
    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry["status"] == "dispatch_pending"
    assert entry["dispatch_pending_at"]

    # A still-fresh claim is not re-dispatchable -- the claim wins.
    set_pass_deadline_exceeded(fake_gh, None)
    result = app.dispatch_rework()
    assert result.data["selected_count"] == 0
    assert dispatch_calls == []

    # Once stale, reconcile's drift sweep surfaces the stranded claim for
    # clearing -- the documented recovery path for this lane.
    state = load_state(paths.state_file)
    state["issues"]["123"]["dispatch_pending_at"] = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=31)
    ).isoformat()
    save_state(paths.state_file, state)
    drift = detect_drift(fake_gh, state, config)
    assert any(d.kind == "stale_dispatch_pending_claim" and d.issue_number == 123 for d in drift)
