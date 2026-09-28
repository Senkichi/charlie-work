"""Orphan-sweep recovery for a dead worker's completed ``.worker-outcome.json`` (issue #1911).

Split out of ``tests/test_charlie_work_orphaned_worker_sweep.py`` during the
#1911 rework: that file crossed the 800-line file-size cap. A dead devin
rework session never gets a terminal-status record (only claude_code's
launch path runs ``start_terminal_status_watcher``), so
``terminal_exit_code`` is None even when the session completed, pushed, and
wrote a fresh, on-target ``.worker-outcome.json``. Before crediting a worker
death, the sweep must route that outcome into the #1877 apply path
(``rework_outcome.apply_collected_rework_outcomes``).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _orphan_sweep_fixtures import (
    _dead_worker_rework_bed,
    _run_orphan_sweep,
    _write_outcome,
)
from charlie_work import rework_outcome
from charlie_work.state import (
    load_state,
)


# ---------------------------------------------------------------------------
# Issue #1911: a dead devin rework session never gets a terminal-status record
# (only claude_code's launch path runs start_terminal_status_watcher), so
# ``terminal_exit_code`` is None even when the session completed, pushed, and
# wrote a fresh, on-target .worker-outcome.json. Before crediting a worker
# death, the sweep must route that outcome into the #1877 apply path.
# ---------------------------------------------------------------------------


def test_orphaned_worker_fresh_outcome_without_terminal_record_is_not_a_death(
    tmp_path: Path,
) -> None:
    """Acceptance: dead worker, no terminal record, fresh on-target outcome.

    The sweep must apply the outcome through the #1877 path (the PR body the
    worker drafted lands on the PR) and surface once via fingerprinted drift --
    not reset to ``rework_requested`` and credit a worker death, which is what
    drove the swole#198 no_op_rework_attempts_cap_exceeded loop.
    """
    from unittest.mock import patch

    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",
            "pr_body": "Closes #207\n\nCorrected PR body per review.",
        },
    )
    # No terminal-status file in sessions_dir -- the devin launch path never
    # writes one.

    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    # Not credited as a death: no reset, no worker_death_at.
    assert entry.get("status") == "dispatched"
    assert entry.get("worker_death_at") is None
    events = state.get("events", [])
    assert [e for e in events if e.get("kind") == "orphaned_worker_recovered"] == []
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    payload = drift_events[0]["payload"]
    assert payload["reason"] == "dead_worker_completed_outcome"
    assert payload["exit_code"] is None
    assert payload["worker_outcome_head_sha"] == "abc123"

    # The outcome reached the #1877 apply path: the drafted PR body landed.
    assert len(fake_gh.pr_edits) == 1
    assert fake_gh.pr_edits[0][0] == 100
    assert "Corrected PR body per review" in fake_gh.pr_edits[0][1]
    applied = [e for e in events if e.get("kind") == "rework_outcome_applied"]
    assert len(applied) == 1
    assert state.get("rework_outcome_applied_heads", {}).get("207") == "abc123"


def test_orphaned_worker_fresh_outcome_approved_rework_is_not_a_death(
    tmp_path: Path,
) -> None:
    """Approved + PR-state ``rework_requested`` branch of the same recovery.

    The post-approval rework lane (#1109) dispatches workers on PRs whose
    ``decision`` is ``approved`` and whose PR-state ``status`` is
    ``rework_requested``. A dead devin worker there gets the identical
    #1911 recovery -- no death credit, no reset -- and the drift payload
    records the branch's decision/status provenance.
    """
    from unittest.mock import patch

    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="approved", pr_state_status="rework_requested"
    )
    _write_outcome(
        paths,
        tmp_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",
            "pr_body": "Closes #207\n\nCorrected PR body per review.",
        },
    )

    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "dispatched"
    assert entry.get("worker_death_at") is None
    events = state.get("events", [])
    assert [e for e in events if e.get("kind") == "orphaned_worker_recovered"] == []
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    payload = drift_events[0]["payload"]
    assert payload["reason"] == "dead_worker_completed_outcome"
    assert payload["decision"] == "approved"
    assert payload["pr_state_status"] == "rework_requested"
    assert payload["exit_code"] is None
    assert payload["worker_outcome_head_sha"] == "abc123"
    assert len(fake_gh.pr_edits) == 1
    applied = [e for e in events if e.get("kind") == "rework_outcome_applied"]
    assert len(applied) == 1


def test_orphaned_worker_stale_outcome_without_terminal_record_is_a_death(
    tmp_path: Path,
) -> None:
    """An outcome file from a PREVIOUS dispatch (mtime before dispatched_at)
    must not suppress the death credit -- otherwise a redispatch would read
    the prior round's leftover outcome as proof of completion forever."""
    config, paths, fake_gh, dispatched_at = _dead_worker_rework_bed(tmp_path)
    stale_mtime = datetime.fromisoformat(dispatched_at.replace("Z", "+00:00")) - timedelta(
        minutes=5
    )
    _write_outcome(
        paths,
        tmp_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",
            "pr_body": "stale leftover",
        },
        mtime=stale_mtime,
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "rework_requested"
    assert isinstance(entry.get("worker_death_at"), list)
    events = state.get("events", [])
    recovered = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered) == 1
    assert recovered[0]["payload"]["reason"] == "dead_worker_with_request_changes"
    assert fake_gh.pr_edits == []


def test_orphaned_worker_outcome_at_other_head_is_a_death(tmp_path: Path) -> None:
    """A fresh outcome pinning a head other than the live head describes a
    different remote state -- not proof this dispatch completed."""
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "other-head-sha",
            "pr_body": "different head",
        },
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "rework_requested"
    assert isinstance(entry.get("worker_death_at"), list)
    events = state.get("events", [])
    recovered = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered) == 1
    assert fake_gh.pr_edits == []


def test_orphaned_worker_recorded_crash_with_fresh_outcome_is_a_death(
    tmp_path: Path,
) -> None:
    """Boundary pin: a terminal record carrying a confirmed non-zero exit is
    still a crash even when a fresh, on-target outcome file exists -- the
    #1911 recovery only fills the *missing* terminal-record case
    (``terminal_exit_code is None``), matching the issue's suggested fix."""
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",
            "pr_body": "finished then crashed",
        },
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "issue-207.devin.terminal.json").write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 1,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:05:00Z",
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "rework_requested"
    assert isinstance(entry.get("worker_death_at"), list)
    events = state.get("events", [])
    recovered = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered) == 1
    assert recovered[0]["payload"]["exit_code"] == 1
    assert fake_gh.pr_edits == []


# ---------------------------------------------------------------------------
# Issue #1915: applying the recovered outcome is only half the recovery. The
# issue must also leave ``dispatched`` -- routed through review() like the
# ``dead_worker_with_head_change`` sibling branch -- instead of dead-ending
# into the 60-minute ``dead_dispatched_worker_reap`` backstop (swole#198).
# ---------------------------------------------------------------------------


def test_orphaned_worker_completed_outcome_routes_to_review(tmp_path: Path) -> None:
    """A fresh, applied completed outcome must route the still-``dispatched``
    issue through ``review()`` and flip it to ``reviewing`` on a fresh packet
    -- the same ``review_routes`` machinery ``dead_worker_with_head_change``
    uses. Before the fix the issue stayed ``dispatched`` with no live worker
    until the dead-dispatched reap force-escalated it."""
    from unittest.mock import patch

    from charlie_work.workflow import CommandResult

    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",
            "pr_body": "Closes #207\n\nCorrected PR body per review.",
        },
    )

    review_calls: list[int] = []

    def fake_review(pr_number: int) -> Any:
        review_calls.append(pr_number)
        return CommandResult(True, "review packet generated", {"pr_number": pr_number})

    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh, review_callback=fake_review)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    # The outcome landed AND the issue routed to review -- out of dispatched.
    assert len(fake_gh.pr_edits) == 1
    assert entry.get("status") == "reviewing"
    assert entry.get("worker_death_at") is None
    assert review_calls == [100]
    events = state.get("events", [])
    applied = [e for e in events if e.get("kind") == "rework_outcome_applied"]
    assert len(applied) == 1
    routed = [e for e in events if e.get("kind") == "orphaned_worker_routed_to_review"]
    assert len(routed) == 1
    payload = routed[0]["payload"]
    assert payload["issue_number"] == 207
    assert payload["pr_number"] == 100
    assert payload["reason"] == "dead_worker_completed_outcome"
    assert payload["routed"] is True

    # A second pass must not re-review or re-emit: the issue is no longer
    # dispatched, so the sweep leaves it alone.
    _run_orphan_sweep(tmp_path, paths, config, fake_gh, review_callback=fake_review)
    state = load_state(paths.state_file)
    assert state["issues"]["207"]["status"] == "reviewing"
    assert review_calls == [100]
    routed = [
        e for e in state.get("events", []) if e.get("kind") == "orphaned_worker_routed_to_review"
    ]
    assert len(routed) == 1


def test_orphaned_worker_completed_outcome_blocked_review_returns_to_rework(
    tmp_path: Path,
) -> None:
    """Issue #1915 fallback: when ``review()`` cannot produce a fresh packet
    for the applied outcome (e.g. the janitor's unchanged-head no-op gate for
    a substantive request_changes verdict), the issue must still leave
    ``dispatched`` -- handed to the ordinary dispatch loop as
    ``rework_requested`` WITHOUT crediting the dead worker's death (the #1911
    invariant the fix exists to preserve)."""
    from unittest.mock import patch

    from charlie_work.workflow import CommandResult

    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",
            "pr_body": "Closes #207\n\nCorrected PR body per review.",
        },
    )

    def fake_review(pr_number: int) -> Any:
        return CommandResult(False, "janitor gate blocked review", {"pr_number": pr_number})

    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh, review_callback=fake_review)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    # The outcome still applied, but the issue is requeued -- not left
    # dispatched for the 60-minute reap.
    assert len(fake_gh.pr_edits) == 1
    assert entry.get("status") == "rework_requested"
    assert entry.get("dispatched_at") is None
    # No death credit (issue #1911 invariant).
    assert entry.get("worker_death_at") is None
    # The rework_requested label edge fired so GitHub labels match state.
    assert (207, config.labels.needs_rework) in fake_gh.labels_added
    events = state.get("events", [])
    applied = [e for e in events if e.get("kind") == "rework_outcome_applied"]
    assert len(applied) == 1
    recovered = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert len(recovered) == 1
    payload = recovered[0]["payload"]
    assert payload["issue_number"] == 207
    assert payload["reason"] == "dead_worker_completed_outcome"
    assert payload["previous_status"] == "dispatched"
    assert payload["new_status"] == "rework_requested"


def test_orphaned_worker_completed_outcome_defers_review_until_applied(
    tmp_path: Path,
) -> None:
    """Issue #1915: routing to review is gated on the outcome actually being
    applied this pass or earlier. A transient apply failure (here the live
    remote head no longer matches the outcome's reported ``head_sha``) must
    defer the review -- not strand the worker's edits by flipping the issue
    out of ``dispatched`` before they land. The next pass retries."""
    from unittest.mock import patch

    from charlie_work.workflow import CommandResult

    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",
            "pr_body": "Closes #207\n\nCorrected PR body per review.",
        },
    )

    review_calls: list[int] = []

    def fake_review(pr_number: int) -> Any:
        review_calls.append(pr_number)
        return CommandResult(True, "review packet generated", {"pr_number": pr_number})

    # Pass 1: the remote head no longer matches the reported outcome head, so
    # the apply is skipped (head_mismatch) and no review may run.
    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "other-head"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh, review_callback=fake_review)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert review_calls == []
    assert entry.get("status") == "dispatched"
    assert fake_gh.pr_edits == []
    # Not resolved and not fingerprinted away: retriable next pass.
    assert entry.get("orphan_drift_fingerprint") is None

    # Pass 2: the remote head agrees again, the outcome applies, and the
    # deferred review route finally runs.
    with patch.object(rework_outcome, "remote_branch_head_sha", lambda *_a: "abc123"):
        _run_orphan_sweep(tmp_path, paths, config, fake_gh, review_callback=fake_review)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert review_calls == [100]
    assert len(fake_gh.pr_edits) == 1
    assert entry.get("status") == "reviewing"
