"""Real ``OrchestratorApp.review()`` coverage for the #1915 orphan-sweep drain.

``test_orphaned_worker_completed_outcome_sweep.py`` pins the drain's routing
contract with stub review callbacks; this module pins what the drain does
when the callback IS production ``OrchestratorApp.review()`` -- the reviewer
finding that motivated this file was that the motivating
request_changes/unchanged-head case never reaches ``reviewing`` under the
real entry point, and that the rework_requested fallback's downstream bound
was unverified.

Under real ``review()`` an unchanged-head ``request_changes`` verdict splits
two ways, and this file pins both:

- stale-CI (issue #1111): every finding cites a configured required check
  that is green now -- the janitor's ``no_op_check_skipped_stale_ci`` lane
  passes the gate, the packet rebuilds, and the issue flips ``dispatched``
  -> ``reviewing``. This is the only unchanged-head shape production can
  actually promote to review.
- substantive findings: the janitor's unchanged-head no-op gate fires, the
  janitor-gate rework wrapper returns ``None`` (a rework is already pending
  for the still-``dispatched`` issue), ``review()`` returns not-ok, and the
  drain's issue #1915 fallback hands the issue to the ordinary dispatch
  loop as ``rework_requested`` -- bounded by ``dispatch_rework``'s own
  ``redispatch_at`` accounting (``max_auto_redispatch`` -> mechanical
  ``redispatch_cap_exceeded``), which the cap-pin half of the test asserts.

The remaining tests are regression coverage for the drain's guard rails:
the per-route exception guard (one throwing ``review()`` must not starve
later routes), ``label_error`` persistence on a failed ``rework_requested``
label edge, the ``orphan_drift_at`` backstop arming on a deferred route and
the #654 timed reap converging when the apply never lands, and completed-
outcome routing reached from the approved-variant call site
(``orphaned_worker_sweep.handle_dead_worker_with_pr``'s #1109 branch).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _helpers import _STALE_CI_REQUIRED
from _orphan_sweep_fixtures import (
    _dead_worker_rework_bed,
    _run_orphan_sweep,
    _write_outcome,
)
from charlie_work import rework_outcome
from charlie_work.config import (
    AutoMergeConfig,
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.state import (
    load_state,
    save_state,
)
from charlie_work.workflow import CommandResult, OrchestratorApp


def _outcome_payload() -> dict[str, Any]:
    return {
        "push_succeeded": True,
        "pr_created": False,
        "head_sha": "abc123",
        "pr_body": (
            "Closes #207\n\nCorrected PR body per review.\n\n"
            "## Tests\nuv run --extra dev pytest -q"
        ),
    }


def test_completed_outcome_stale_ci_verdict_reaches_reviewing_via_real_review(
    tmp_path: Path,
) -> None:
    """request_changes + unchanged head CAN reach ``reviewing`` under the real
    ``review()`` -- but only through the issue #1111 stale-CI lane.

    The recorded verdict's sole finding cites a configured required check
    that is green now, so the janitor's ``no_op_check_skipped_stale_ci``
    suppression lets the packet rebuild proceed. The drain then flips the
    still-``dispatched`` issue to ``reviewing`` -- pinned against the real
    entry point, not a stub, per the #1915 rework finding that the
    unchanged-head path had never been driven through ``review()`` itself.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        auto_merge=AutoMergeConfig(required_checks=_STALE_CI_REQUIRED),
    )
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(
        tmp_path,
        config=config,
        janitor_green=True,
        flat_decision_extra={
            "required_changes": ["Tests passed: .github:18 - Process completed with exit code 1."],
            "verdict_provenance": "fresh_llm_review",
        },
    )
    # The janitor's body check needs the live PR body to mention
    # tests/verification/rationale before it will let a packet build.
    fake_gh.prs[0]["body"] = "Closes #207\n\n## Tests\nuv run --extra dev pytest -q"
    _write_outcome(paths, tmp_path, _outcome_payload())

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh, review_callback=app.review)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    # The outcome applied (the drafted PR body landed) AND the issue left
    # dispatched through the real review path -- no death credit.
    assert len(fake_gh.pr_edits) == 1
    assert entry.get("status") == "reviewing"
    assert entry.get("worker_death_at") is None
    # review()'s own bookkeeping confirms a real packet, not a janitor-gate
    # route or a drift marker. ``review_dispatch.enabled`` is False in this
    # harness (and in the shipped devin profile), so issue #868's gate
    # correctly withholds the PR-state ``status:"reviewing"`` stamp and the
    # review_started label edge -- the packet files + janitor_ok carry the
    # terminal state.
    pr_state = state["prs"]["100"]
    assert pr_state.get("status") is None
    assert pr_state.get("janitor_ok") is True
    assert (paths.prs / "pr-100" / "review-prompt.md").exists()
    # The recorded request_changes verdict is on the live head, so it is
    # NOT voided -- the decision file and state keep it (only a stale-head
    # verdict resets to "pending").
    assert pr_state.get("decision") == "request_changes"
    assert (
        json.loads((paths.prs / "pr-100" / "review-decision.json").read_text(encoding="utf-8"))[
            "decision"
        ]
        == "request_changes"
    )
    events = state.get("events", [])
    # The stale-CI lane specifically -- the event the janitor emits when the
    # skipped no-op check lets an unchanged-head verdict through the gate.
    assert [e for e in events if e.get("kind") == "stale_ci_verdict_gate_pass"]
    routed = [e for e in events if e.get("kind") == "orphaned_worker_routed_to_review"]
    assert len(routed) == 1
    assert routed[0]["payload"]["reason"] == "dead_worker_completed_outcome"
    assert routed[0]["payload"]["routed"] is True


def test_completed_outcome_substantive_verdict_falls_back_and_redispatch_caps(
    tmp_path: Path,
) -> None:
    """Substantive request_changes + unchanged head under real ``review()``:
    the janitor no-op gate blocks the packet, the drain returns the issue to
    ``rework_requested``, and ``dispatch_rework``'s own ``redispatch_at``
    accounting -- not a new counter -- bounds the re-dispatch loop.

    Below ``max_auto_redispatch`` windowed redispatches the issue is
    redispatched normally (the recorded findings still need a worker even
    though the recovered body edit already landed); at the cap the issue
    escalates mechanically with ``redispatch_cap_exceeded`` instead of
    burning another session.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(
        tmp_path,
        config=config,
        janitor_green=True,
        flat_decision_extra={
            "required_changes": ["src/foo.py: the retry loop drops the last batch"],
            "verdict_provenance": "fresh_llm_review",
        },
    )
    _write_outcome(paths, tmp_path, _outcome_payload())

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh, review_callback=app.review)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    # review() could not produce a packet for the unchanged substantive
    # verdict, so the drain's #1915 fallback requeued the issue -- with the
    # armed drift marker cleared, the needs_rework edge applied, and still
    # no death credit.
    assert len(fake_gh.pr_edits) == 1
    assert entry.get("status") == "rework_requested"
    assert entry.get("dispatched_at") is None
    assert entry.get("orphan_drift_at") is None
    assert entry.get("worker_death_at") is None
    assert (207, config.labels.needs_rework) in fake_gh.labels_added
    recovered = [
        e for e in state.get("events", []) if e.get("kind") == "orphaned_worker_recovered"
    ]
    assert len(recovered) == 1
    assert recovered[0]["payload"]["reason"] == "dead_worker_completed_outcome"

    # Bound pin, part 1: below the cap, dispatch_rework treats the fallback
    # issue as a normal rework candidate and redispatches it -- the
    # redispatch is explicitly counted in ``redispatch_at``.
    (paths.prs / "pr-100" / "rework-prompt.md").write_text(
        "Address the recorded findings", encoding="utf-8"
    )
    result = app.dispatch_rework()
    assert result.ok is True
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "dispatched"
    assert len(entry.get("redispatch_at") or []) == 1

    # Bound pin, part 2: once the windowed redispatch count reaches
    # max_auto_redispatch (3) with no credited worker deaths to subtract,
    # the head-unchanged pre-filter escalates mechanically instead of
    # launching another worker on the identical head.
    state["issues"]["207"]["status"] = "rework_requested"
    state["issues"]["207"]["redispatch_at"] = [
        (datetime.now(UTC) - timedelta(minutes=30 - i)).isoformat().replace("+00:00", "Z")
        for i in range(3)
    ]
    save_state(paths.state_file, state)

    result = app.dispatch_rework()
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "redispatch_cap_exceeded"
    assert result.data.get("no_op_rework_escalated") == [207]
    escalated = [e for e in state.get("events", []) if e.get("kind") == "session_failed_escalated"]
    assert escalated
    assert escalated[-1]["payload"]["reason"] == "no_op_rework_cap_exceeded"
    # Mechanical escalations land the operator-queue label, not human-needed.
    assert (207, config.labels.operator_queue) in fake_gh.labels_added


def test_approved_rework_variant_completed_outcome_routes_via_real_review(
    tmp_path: Path,
) -> None:
    """``approved`` + PR-state ``rework_requested`` + unchanged head: the
    #1109 classified branch reaches ``handle_dead_worker_completed_outcome``
    from its own call site, and the drain's route behaves identically --
    apply, then ``review()`` flips the issue to ``reviewing``.

    For an ``approved`` verdict the janitor's no-op check (which only fires
    on ``request_changes``) does not run, so a janitor-green PR produces a
    fresh packet directly. Drives the real ``review()`` entry point, not a
    stub -- the approved-variant call site had no route coverage at all.
    """
    from unittest.mock import patch

    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(
        tmp_path,
        decision="approved",
        pr_state_status="rework_requested",
        janitor_green=True,
        flat_decision_extra={"verdict_provenance": "fresh_llm_review"},
    )
    fake_gh.prs[0]["body"] = "Closes #207\n\n## Tests\nuv run --extra dev pytest -q"
    _write_outcome(paths, tmp_path, _outcome_payload())

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh, review_callback=app.review)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert len(fake_gh.pr_edits) == 1
    assert entry.get("status") == "reviewing"
    assert entry.get("worker_death_at") is None
    events = state.get("events", [])
    routed = [e for e in events if e.get("kind") == "orphaned_worker_routed_to_review"]
    assert len(routed) == 1
    assert routed[0]["payload"]["reason"] == "dead_worker_completed_outcome"
    assert routed[0]["payload"]["routed"] is True
    # The approved verdict stays recorded (unchanged head -> not voided);
    # the PR-state ``rework_requested`` lane marker persists because
    # issue #868's review_dispatch-disabled gate withholds the
    # ``status:"reviewing"`` stamp -- the issue-level flip to ``reviewing``
    # above is the drain's own routing write.
    pr_state = state["prs"]["100"]
    assert pr_state.get("decision") == "approved"
    assert pr_state.get("status") == "rework_requested"
    assert pr_state.get("janitor_ok") is True
    assert (paths.prs / "pr-100" / "review-prompt.md").exists()


def test_review_route_exception_does_not_starve_remaining_routes(tmp_path: Path) -> None:
    """Per-route guard: a ``review()`` that raises (network blip, malformed
    response, any unexpected escape) must not starve later routes or the
    post-drain transition loops. The failed route re-collects next pass --
    its drift fingerprint is deliberately left unmarked.
    """
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)

    # A second dead dispatched worker with a head-advanced PR: the
    # ``dead_worker_with_head_change`` branch emits the same route shape.
    state = load_state(paths.state_file)
    state["issues"]["208"] = {
        "status": "dispatched",
        "worker_pid": 88888,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": _dispatched_at,
        "branch_name": "agent/issue-208",
    }
    state["prs"]["101"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)
    pr101_dir = paths.prs / "pr-101"
    pr101_dir.mkdir(parents=True, exist_ok=True)
    (pr101_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "abc123"}),
        encoding="utf-8",
    )
    fake_gh.prs.append(
        {
            "number": 101,
            "headRefOid": "def456",  # Advanced past the recorded verdict
            "isCrossRepository": False,
            "headRepository": {"owner": {"login": "test"}, "name": "repo"},
            "headRefName": "agent/issue-208",
        }
    )
    fake_gh.prs[0]["headRefOid"] = "aaa111"  # PR 100 also head-advanced
    fake_gh.issues.append(
        {
            "number": 208,
            "title": "Second issue",
            "url": "https://example.test/issues/208",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    calls: list[int] = []

    def flaky_review(pr_number: int) -> Any:
        calls.append(pr_number)
        if pr_number == 100:
            raise RuntimeError("simulated gh api outage")
        return CommandResult(True, "review packet generated", {"pr_number": pr_number})

    _run_orphan_sweep(tmp_path, paths, config, fake_gh, review_callback=flaky_review)

    state = load_state(paths.state_file)
    # Route order is state-file order: issue 207's route throws, issue 208's
    # still runs to completion.
    assert calls == [100, 101]
    # The per-route failure event goes through log_event -> events.db.
    from charlie_work.instrumentation import query_events

    failed = query_events(paths.state_file, kind="orphaned_worker_review_route_failed")
    assert len(failed) == 1
    assert failed[0]["payload"]["issue_number"] == 207
    assert failed[0]["payload"]["pr_number"] == 100
    assert "RuntimeError" in failed[0]["payload"]["error"]
    entry207 = state["issues"]["207"]
    assert entry207.get("status") == "dispatched"
    # Unresolved, not fingerprinted away -- the route re-collects next pass.
    assert entry207.get("orphan_drift_fingerprint") is None
    # The surviving route ran to completion: issue 208 is in the review lane.
    assert state["issues"]["208"]["status"] == "reviewing"
    events = state.get("events", [])
    routed = [e for e in events if e.get("kind") == "orphaned_worker_routed_to_review"]
    assert len(routed) == 1
    assert routed[0]["payload"]["issue_number"] == 208
    assert routed[0]["payload"]["routed"] is True


def test_rework_requested_label_failure_persists_label_error(tmp_path: Path) -> None:
    """The drain's ``rework_requested`` label edge is best-effort: a failed
    transition must persist ``label_error`` on the issue entry (the marker
    ``dead_worker_reap`` uses everywhere else) instead of silently leaving
    GitHub labels stale against the committed status flip.
    """
    from unittest.mock import patch

    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    _write_outcome(paths, tmp_path, _outcome_payload())

    # Wrap the live fake's add_issue_label so the needs_rework edge fails
    # while every other GitHub call keeps working.
    fake_gh.add_issue_label = lambda _n, _l: False  # type: ignore[method-assign]

    def blocked_review(pr_number: int) -> Any:
        return CommandResult(False, "janitor gate blocked review", {"pr_number": pr_number})

    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh, review_callback=blocked_review)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    # The status flip committed despite the label failure...
    assert entry.get("status") == "rework_requested"
    assert entry.get("worker_death_at") is None
    # ...and the failure is recorded on the issue, inspectable.
    label_error = entry.get("label_error")
    assert label_error is not None
    assert label_error["edge"] == "rework_requested"
    assert label_error["outcome"] != "applied"
    assert label_error["add_failures"] or label_error["remove_failures"]


def test_deferred_completed_outcome_route_reaps_via_orphan_drift_backstop(
    tmp_path: Path,
) -> None:
    """``orphan_drift_at`` backstop: when the outcome apply can never land
    (here the remote head permanently mismatches the outcome's pin), the
    deferred route re-collects every pass -- and after
    ``dead_dispatched_reap_minutes`` the #654 timed reap escalates the
    issue mechanically rather than leaving it ``dispatched`` forever.
    """
    from unittest.mock import patch

    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    _write_outcome(paths, tmp_path, _outcome_payload())

    review_calls: list[int] = []

    def never_called(pr_number: int) -> Any:
        review_calls.append(pr_number)
        return CommandResult(True, "review packet generated", {})

    # Pass 1: the remote head disagrees with the outcome's pin, so the apply
    # is deferred (head_mismatch) and the review route stays pending -- but
    # ``orphan_drift_at`` is armed so the finding cannot hold forever.
    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "other-head"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh, review_callback=never_called)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "dispatched"
    assert entry.get("worker_death_at") is None
    assert review_calls == []
    assert entry.get("orphan_drift_fingerprint") is None
    drift_at = entry.get("orphan_drift_at")
    assert drift_at is not None

    # Backdate the armed marker past the 60-minute reap window, as if the
    # apply had been deferred (and re-collected) for over an hour.
    state["issues"]["207"]["orphan_drift_at"] = (
        (datetime.now(UTC) - timedelta(minutes=61)).isoformat().replace("+00:00", "Z")
    )
    save_state(paths.state_file, state)

    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "other-head"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh, review_callback=never_called)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "dead_dispatched_worker_reap"
    # The reap clears the armed markers as part of the escalation.
    assert entry.get("orphan_drift_at") is None
    assert entry.get("orphan_drift_fingerprint") is None
    reaped = [
        e for e in state.get("events", []) if e.get("kind") == "dead_dispatched_worker_reaped"
    ]
    assert len(reaped) == 1
    assert reaped[0]["payload"]["reason"] == "dead_dispatched_worker_reap"
    # Mechanical escalations land the operator-queue label.
    assert (207, config.labels.operator_queue) in fake_gh.labels_added
    assert review_calls == []
