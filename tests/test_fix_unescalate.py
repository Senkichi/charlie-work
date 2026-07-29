"""Tests for PR #550's operator re-arm command: ``OrchestratorApp.unescalate``
and its ``charlie unescalate`` CLI wiring.

Escalation is deliberately terminal for every automated path (``review()``
and ``record_review()`` both hard-stop on ``status == "escalated"``) --
before this command existed the only recovery was hand-editing state.json
and labels, which is exactly how status/label desyncs kept happening in
production (pr-lifecycle.md). ``unescalate`` is the sanctioned door back:

- PR still open on GitHub: reset to the passive pr-open state, zero every
  attempt counter / frozen janitor-cache field, apply the
  ``unescalated_pr_open`` label edge.
- PR merged/closed on GitHub: normalize the record to that terminal state;
  no label edit (finalization/reconcile own the rest).
- Issue with no live PR anywhere: drop back to the never-dispatched
  baseline and apply ``unescalated_requeued`` (strip every workflow label,
  add nothing -- adding ``queued`` would be an ACTIVE label and exclude the
  issue from dispatch, the exact trap this command exists to avoid).
- Idempotent on a non-escalated record; ``dry_run`` computes the transition
  map without touching state, labels, or events.

These tests reuse ``FakeGitHub`` from test_charlie_work.py (PR #456 <->
issue #123, OPEN by default) and ``_FakeGitHub``/``_make_repo`` from
test_cli.py for the CLI-level check.
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work import cli
from charlie_work.config import OrchestratorConfig, PostMortemConfig
from charlie_work.labels import TransitionOutcome, transition
from charlie_work.paths import runtime_paths
from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from test_charlie_work import FakeGitHub
from test_cli import _FakeGitHub, _make_repo


def _events(state, kind: str) -> list[dict]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


def _app(tmp_path: Path) -> OrchestratorApp:
    # Isolate post_mortem.db_path from the real Devin sessions.db. The default
    # (db_path="") resolves to %APPDATA%\devin\cli\sessions.db at read time;
    # on a self-hosted CI runner that file exists with real session data, so
    # issue_worker_liveness's real-activity probe could surface a stale
    # timestamp for the test PID and flip the verdict from inconclusive-defer
    # (live=True, refuse) to conclusive-stale (live=False, proceed) -- dropping
    # the ``issue_worker_alive`` key the refusal branch sets. Pointing at a
    # nonexistent path under tmp_path makes every probe source error out
    # (inconclusive), which is the condition both #625 tests depend on.
    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    return OrchestratorApp(tmp_path, paths, config, fake_gh)


# --- labels.py: the two new label edges, tested directly via transition() ---


def test_unescalated_pr_open_edge_adds_pr_open_removes_the_rest() -> None:
    config = OrchestratorConfig()
    fake_gh = FakeGitHub()

    result = transition(fake_gh, config.labels, 123, "unescalated_pr_open")

    assert result.outcome == TransitionOutcome.APPLIED
    assert fake_gh.labels_added == [(123, config.labels.pr_open)]
    removed = {label for (_, label) in fake_gh.labels_removed}
    assert removed == config.labels.workflow_labels - {config.labels.pr_open}
    assert config.labels.human_needed in removed


def test_unescalated_requeued_edge_adds_nothing_removes_all_workflow_labels() -> None:
    config = OrchestratorConfig()
    fake_gh = FakeGitHub()

    result = transition(fake_gh, config.labels, 123, "unescalated_requeued")

    assert result.outcome == TransitionOutcome.APPLIED
    assert fake_gh.labels_added == []
    removed = {label for (_, label) in fake_gh.labels_removed}
    assert removed == config.labels.workflow_labels
    # Never queued: queued is an ACTIVE label and would exclude the issue
    # from dispatch -- the exact trap this edge exists to avoid.
    assert config.labels.queued in removed


# --- OrchestratorApp.unescalate() ---


def test_unescalate_escalated_open_pr_resets_and_relabels(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
            "review_dispatch_attempt_count": 3,
            "request_changes_count": 2,
            "conflict_rework_attempts": 1,
            "no_op_rework_attempts": 1,
            "janitor_ok": False,
            "janitor_failures": ["some failure"],
            "janitor_warnings": ["some warning"],
            "review_dispatch_status": "review_dispatch_failed",
            "escalation_reason": "max_review_dispatch_attempts_exceeded",
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "max_review_dispatch_attempts_exceeded",
            "label_error": {"edge": "escalated", "outcome": "partial_failure"},
            "worker_pid": 12345,
            "dispatched_at": "2024-01-01T00:00:00Z",
        }
        save_state(app.paths.state_file, state)

    result = app.unescalate(pr_number=456)

    assert result.ok is True
    assert result.data["changed"] is True
    state = load_state(app.paths.state_file)
    pr_entry = state["prs"]["456"]
    assert pr_entry["status"] == PASSIVE_OPEN_STATUS == "reviewing"
    assert pr_entry["review_dispatch_attempt_count"] == 0
    assert pr_entry["request_changes_count"] == 0
    for stale_field in (
        "conflict_rework_attempts",
        "no_op_rework_attempts",
        "janitor_ok",
        "janitor_failures",
        "janitor_warnings",
        "review_dispatch_status",
        "escalation_reason",
    ):
        assert stale_field not in pr_entry, stale_field

    issue_entry = state["issues"]["123"]
    assert issue_entry["status"] == PASSIVE_OPEN_STATUS
    for stale_field in ("escalation_reason", "label_error", "worker_pid", "dispatched_at"):
        assert stale_field not in issue_entry, stale_field

    assert (123, app.config.labels.pr_open) in app.gh.labels_added
    assert (123, app.config.labels.human_needed) in app.gh.labels_removed
    assert len(_events(state, "unescalate")) == 1


def test_unescalate_janitor_blocked_pr_uses_same_rearm_path(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "janitor_blocked",
            "janitor_failures": ["merge conflict"],
        }
        state["issues"]["123"] = {"number": 123, "status": "rework_requested"}
        save_state(app.paths.state_file, state)

    result = app.unescalate(pr_number=456)

    assert result.ok is True
    assert result.data["changed"] is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["status"] == PASSIVE_OPEN_STATUS
    assert "janitor_failures" not in state["prs"]["456"]
    assert state["issues"]["123"]["status"] == PASSIVE_OPEN_STATUS
    assert (123, app.config.labels.pr_open) in app.gh.labels_added


def test_unescalate_merged_pr_normalizes_status_without_label_edge(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.gh.prs[0]["state"] = "MERGED"
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
            "review_dispatch_attempt_count": 3,
        }
        save_state(app.paths.state_file, state)

    result = app.unescalate(pr_number=456)

    assert result.ok is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["status"] == "merged"
    # Counters are only reset on the passive-open re-arm branch; a terminal
    # PR is left for finalization/reconcile, so no label edge must fire.
    assert app.gh.labels_added == []
    assert app.gh.labels_removed == []


def test_unescalate_escalated_issue_with_no_pr_drops_status_and_requeues(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "escalated"}
        # No entry in state["prs"] references issue 123 at all.
        save_state(app.paths.state_file, state)

    result = app.unescalate(issue_number=123)

    assert result.ok is True
    assert result.data["pr"] is None
    state = load_state(app.paths.state_file)
    assert "status" not in state["issues"]["123"]
    assert app.gh.labels_added == []
    removed = {label for (num, label) in app.gh.labels_removed if num == 123}
    assert removed == app.config.labels.workflow_labels


def test_unescalate_non_escalated_record_is_idempotent_noop(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "reviewing",
            "review_dispatch_attempt_count": 1,
        }
        state["issues"]["123"] = {"number": 123, "status": "reviewing"}
        save_state(app.paths.state_file, state)
    before = load_state(app.paths.state_file)

    result = app.unescalate(pr_number=456)

    assert result.ok is True
    assert result.data["changed"] is False
    assert "nothing to unescalate" in result.message
    after = load_state(app.paths.state_file)
    assert after["prs"] == before["prs"]
    assert after["issues"] == before["issues"]
    assert after["events"] == before["events"]
    assert app.gh.labels_added == []
    assert app.gh.labels_removed == []


def test_unescalate_dry_run_reports_transitions_without_mutating_anything(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
            "review_dispatch_attempt_count": 3,
        }
        state["issues"]["123"] = {"number": 123, "status": "escalated"}
        save_state(app.paths.state_file, state)
    before = load_state(app.paths.state_file)

    result = app.unescalate(pr_number=456, dry_run=True)

    assert result.ok is True
    assert result.data["changed"] is False
    assert result.data["transitions"]["pr.status"] == ["escalated", PASSIVE_OPEN_STATUS]
    assert result.data["transitions"]["issue.status"] == ["escalated", PASSIVE_OPEN_STATUS]
    assert result.data["label_edge"] == "unescalated_pr_open"

    after = load_state(app.paths.state_file)
    assert after["prs"] == before["prs"]
    assert after["issues"] == before["issues"]
    assert after["events"] == before["events"]
    # No label-mutating gh calls; pr_view (read-only) is allowed since the
    # live PR state is what decides the re-entry point even in a dry run.
    assert app.gh.labels_added == []
    assert app.gh.labels_removed == []


# --- CLI wiring ---


def test_cli_unescalate_dry_run_parses_and_dispatches(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    state_path = repo / ".var" / "charlie-work" / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {"1": {"number": 1, "status": "escalated"}},
                "prs": {"1": {"number": 1, "issue_number": 1, "status": "escalated"}},
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(["--repo", str(repo), "unescalate", "--pr", "1", "--dry-run"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "dry-run" in captured.out
    # dry_run must not touch the state file.
    on_disk = json.loads(state_path.read_text(encoding="utf-8"))
    assert on_disk["prs"]["1"]["status"] == "escalated"


def test_unescalate_refuses_entirely_when_worker_session_alive(tmp_path: Path) -> None:
    """Issue #214 precedent (reconcile's live_session_issue_numbers guard):
    PR "janitor_blocked" + issue "dispatched" with a LIVE worker is the
    NORMAL mid-rework steady state, not a wedge -- unescalate must refuse
    entirely. Popping issue-side worker_pid/dispatched_at would blind
    orphan-worker detection; resetting the PR side would zero the
    conflict/no-op attempt caps for a rework cycle still in flight and flip
    the PR to the passive reviewing status, inviting a concurrent review()
    against the worker's in-progress push.

    Issue #625: "live" now means PID-alive AND not stalled. This test process
    is the live worker; with no real activity sources for the test PID the
    probe is inconclusive, so a FRESH dispatched_at (within the wall-clock
    deadline) keeps the verdict deferred to live=True (refuse). The
    wedged-but-alive case is covered by
    ``test_unescalate_proceeds_when_worker_alive_but_wedged`` below.
    """
    import os
    from datetime import UTC, datetime

    app = _app(tmp_path)
    fresh_dispatched_at = datetime.now(UTC).isoformat()
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "janitor_blocked",
            "janitor_ok": False,
            "janitor_failures": ["merge conflict"],
            "conflict_rework_attempts": 1,
            "conflict_rework_attempts_last_head": "sha-mid-cycle",
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            # This test process itself is the "live worker": is_pid_alive
            # passes for a real live pid when no start time is recorded.
            "worker_pid": os.getpid(),
            # Fresh timestamp: the inconclusive-activity probe defers to
            # live=True only while the session is within the wall-clock
            # deadline, so the refusal branch is exercised here.
            "dispatched_at": fresh_dispatched_at,
        }
        save_state(app.paths.state_file, state)

    result = app.unescalate(456, None, dry_run=False)

    assert result.ok is True
    assert result.data["issue_worker_alive"] is True
    assert result.data["changed"] is False
    # Issue #625 direction 4: the refusal must be diagnosable.
    assert result.data["issue_worker_source"] == "state"
    assert result.data["issue_worker_pid"] == os.getpid()
    assert result.data["issue_worker_session_started_at"] is not None
    assert "inconclusive" in result.message
    assert app.gh.labels_added == []
    assert app.gh.labels_removed == []

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "dispatched"
    assert issue["worker_pid"] == os.getpid()
    assert issue["dispatched_at"] == fresh_dispatched_at
    # PR side untouched too: mid-epoch attempt caps and janitor caches must
    # survive, or the anti-infinite-loop bound resets around a live worker.
    pr = state["prs"]["456"]
    assert pr["status"] == "janitor_blocked"
    assert pr["conflict_rework_attempts"] == 1
    assert pr["conflict_rework_attempts_last_head"] == "sha-mid-cycle"
    assert pr["janitor_failures"] == ["merge conflict"]


def test_unescalate_proceeds_when_worker_alive_but_wedged(tmp_path: Path) -> None:
    """Issue #625 regression: a worker process that finishes its work but
    never exits held its slot forever and blocked ``charlie unescalate``
    indefinitely, because the state-side check asked only "is the PID alive?"
    with no staleness bound. The session's sidecar had been reaped, so the
    watchdog (which knows how to time it out) never saw it; only the
    timeout-less state-side check remained.

    Now both checks route through ``issue_worker_liveness``: an alive PID
    whose session is past the wall-clock deadline with no fresh activity is
    wedged, not live, so unescalate proceeds and re-arms the record. This
    test uses the test process as the alive PID with a stale dispatched_at
    and no sidecar, mirroring the observed job-cannon #1268/#1392 wedge.
    """
    import os

    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "janitor_blocked",
            "janitor_ok": False,
            "janitor_failures": ["merge conflict"],
            "conflict_rework_attempts": 1,
            "conflict_rework_attempts_last_head": "sha-mid-cycle",
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            # The test process is alive, but the session is days old -- past
            # the wall-clock deadline and with no activity source for the
            # test PID, the probe is inconclusive and the wall-clock backstop
            # classifies it as wedged.
            "worker_pid": os.getpid(),
            "dispatched_at": "2026-07-23T19:44:05+00:00",
        }
        save_state(app.paths.state_file, state)

    result = app.unescalate(456, None, dry_run=False)

    assert result.ok is True
    # The wedged worker no longer blocks unescalate: the record is re-armed.
    assert result.data["changed"] is True
    assert result.data.get("issue_worker_alive") is not True
    state = load_state(app.paths.state_file)
    pr = state["prs"]["456"]
    assert pr["status"] == PASSIVE_OPEN_STATUS
    issue = state["issues"]["123"]
    assert issue["status"] == PASSIVE_OPEN_STATUS
    # The wedged worker_pid / dispatched_at are cleared on re-arm.
    assert "worker_pid" not in issue
    assert "dispatched_at" not in issue


def test_unescalate_concurrent_writer_fields_survive_the_write(tmp_path: Path) -> None:
    """unescalate computes its transformation from a pre-fetch snapshot but
    must re-apply it to freshly-loaded entries inside the write lock -- a
    field written by a concurrent writer between the snapshot and the write
    (simulated via the FakeGitHub pr_view hook) must survive.
    """
    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
        }
        state["issues"]["123"] = {"number": 123, "status": "escalated"}
        save_state(app.paths.state_file, state)

    original_pr_view = app.gh.pr_view

    def _pr_view_with_concurrent_write(number: int):
        # A concurrent writer (e.g. a reconcile pass) lands a field while
        # unescalate is off doing its network fetch.
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["prs"]["456"] = {
                **state["prs"]["456"],
                "concurrent_field": "must-survive",
            }
            state["issues"]["123"] = {
                **state["issues"]["123"],
                "concurrent_issue_field": "must-survive",
            }
            save_state(app.paths.state_file, state)
        return original_pr_view(number)

    app.gh.pr_view = _pr_view_with_concurrent_write  # type: ignore[method-assign]

    result = app.unescalate(456, None, dry_run=False)

    assert result.ok is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["concurrent_field"] == "must-survive"
    assert state["issues"]["123"]["concurrent_issue_field"] == "must-survive"
    assert state["prs"]["456"]["status"] == PASSIVE_OPEN_STATUS
