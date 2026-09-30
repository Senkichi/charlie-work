"""Characterization of the dead-worker sweep's end-to-end routing (wave B, dws-1).

Pins CURRENT behaviour of ``workflow._detect_and_handle_orphaned_workers`` --
one test per routing decision path -- so the dead-worker-sweep consolidation
can move the code without changing where a dead ``dispatched`` Worker's issue
ends up. Every test drives the real sweep (real ``state.json`` + events, a
``FakeGitHub``); none asserts a decision helper in isolation.

Decision map (recon ``dws-recon.md`` section a.4 / a.5):

- no open PR, pre-lock escalations: declared-blocked, zero-artifact loop,
  zero-artifact throttle exemption, cross-repo scope gate;
- no open PR, reclaim family: #417 relabel, label-write failure retry, local
  park, terminal-label-only flag, dead-dispatched backstop and its throttle
  re-arm, redispatch cap and its throttle exemption;
- no open PR, pushed branch: PR opened / PR create failed (stranded) / salvage
  push refreshing the head / live-PID stale-outcome handoff;
- open PR and spine cases live in
  ``test_dead_worker_sweep_characterization_with_pr.py`` (800-line cap).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from _dead_worker_sweep_characterization_fixtures import (
    NoPrGitHub,
    events_of,
    iso,
    issue_entry,
    make_repo_with_pushed_branch,
    no_pr_bed,
    run_sweep,
    seed_issue,
    sessions_dir_for,
    sweep_config,
    write_aged_outcome,
    write_blocked_outcome,
)
from _local_park_fixtures import (
    _add_worktree_commit,
    _init_repo,
    _local_config,
    _seed_dead_dispatched,
    _write_issue,
)
from _local_park_fixtures import _shallow_wts  # noqa: F401  (registers ``shallow_wts``)
from charlie_work.config import LabelConfig
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import runtime_paths
from charlie_work.state import PASSIVE_OPEN_STATUS

# ---------------------------------------------------------------------------
# No open PR: pre-lock escalations
# ---------------------------------------------------------------------------


def test_no_pr_worker_declared_blocked_escalates_to_operator_queue(tmp_path: Path) -> None:
    branch = "agent/issue-1453-test"
    config = sweep_config(max_auto_redispatch=3)
    config, paths, gh = no_pr_bed(
        tmp_path, 1453, config=config, branch_name=branch, dispatched_at=iso()
    )
    write_blocked_outcome(tmp_path, config, branch, "cross_repo_scope", "targets job-cannon")

    run_sweep(
        tmp_path,
        paths,
        config,
        gh,
        patches=(
            patch("charlie_work.workflow.remote_branch_head_sha", return_value=None),
            patch("charlie_work.workflow.remote_branch_ahead_count", return_value=(0, None)),
        ),
    )

    entry = issue_entry(paths, 1453)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worker_declared_blocked"
    assert (1453, config.labels.in_progress) in gh.labels_removed
    # Pre-lock fallback label, then the post-lock reap transition swaps it for
    # the operator queue edge.
    assert (1453, config.labels.human_needed) in gh.labels_added
    assert (1453, config.labels.operator_queue) in gh.labels_added
    assert (1453, config.labels.ready) not in gh.labels_added
    (blocked,) = events_of(paths, "worker_declared_blocked")
    assert blocked["payload"]["reason_kind"] == "cross_repo_scope"
    assert events_of(paths, "orphan_sweep_redispatch_escalated") == []


def _write_zero_artifact_post_mortem(tmp_path: Path, number: int) -> None:
    import json

    (sessions_dir_for(tmp_path) / f"issue-{number}.post-mortem.json").write_text(
        json.dumps(
            {
                "issue_number": number,
                "generated_at": iso(),
                "db_path": "",
                "matched": False,
                "attempts": [
                    {"ref": f"refs/charlie/attempts/{n}", "ahead_of_main": 0, "recorded_at": iso()}
                    for n in (1, 2)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_no_pr_zero_artifact_dispatch_loop_escalates(tmp_path: Path) -> None:
    config, paths, gh = no_pr_bed(tmp_path, 1983, dispatched_at=iso())
    _write_zero_artifact_post_mortem(tmp_path, 1983)

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 1983)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "zero_artifact_dispatch_loop"
    assert (1983, config.labels.human_needed) in gh.labels_added
    assert (1983, config.labels.ready) not in gh.labels_added


def test_no_pr_zero_artifact_throttle_death_is_relabeled_not_escalated(tmp_path: Path) -> None:
    config, paths, gh = no_pr_bed(
        tmp_path, 1983, dispatched_at=iso(), dead_worker_failure_kind="rate_limited"
    )
    _write_zero_artifact_post_mortem(tmp_path, 1983)

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 1983)
    assert entry.get("status") != "escalated"
    assert (1983, config.labels.human_needed) not in gh.labels_added
    assert (1983, config.labels.in_progress) in gh.labels_removed
    assert (1983, config.labels.ready) in gh.labels_added
    assert len(events_of(paths, "session_failed_relabeled")) == 1


def test_no_pr_cross_repo_scoped_issue_escalates_cross_repo_hop(tmp_path: Path) -> None:
    import json

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "fleet.json").write_text(
        json.dumps(
            {
                "version": 1,
                "repos": {
                    "Senkichi/charlie-work": {"repo_root": "/tmp/cw"},
                    "Senkichi/job-cannon": {"repo_root": "/tmp/jc"},
                },
            }
        ),
        encoding="utf-8",
    )
    config, paths, gh = no_pr_bed(
        tmp_path, 709, title="job-cannon: docs are stale", dispatched_at=iso()
    )
    gh.name_with_owner = lambda: "Senkichi/charlie-work"  # type: ignore[method-assign]

    run_sweep(tmp_path, paths, config, gh, fleet_dir=fleet_dir)

    entry = issue_entry(paths, 709)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "cross_repo_hop"
    assert (709, config.labels.in_progress) in gh.labels_removed
    assert (709, config.labels.human_needed) in gh.labels_added
    assert (709, config.labels.ready) not in gh.labels_added
    (event,) = events_of(paths, "session_failed_escalated")
    assert event["payload"]["reason"] == "cross_repo_hop"


# ---------------------------------------------------------------------------
# No open PR: reclaim family
# ---------------------------------------------------------------------------


def test_no_pr_active_label_is_reclaimed_to_ready(tmp_path: Path) -> None:
    config, paths, gh = no_pr_bed(tmp_path, 1176, dispatched_at="2026-07-14T17:24:55Z")

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 1176)
    # The #282 liveness fingerprint survives; status stays dispatched.
    assert entry["status"] == "dispatched"
    assert entry["worker_pid"] == 99999
    assert entry["orphan_flagged_at"] is not None
    assert (1176, config.labels.in_progress) in gh.labels_removed
    assert (1176, config.labels.ready) in gh.labels_added
    (event,) = events_of(paths, "session_failed_relabeled")
    assert event["payload"]["label_write_ok"] is True
    assert event["payload"]["reason"] == "dead_worker_no_open_pr_orphan_sweep"
    assert events_of(paths, "orphaned_worker_drift") == []


class _FlakyLabelGitHub(NoPrGitHub):
    fail_remove = True

    def remove_issue_label(self, number: int, label: str) -> bool:
        self.labels_removed.append((number, label))
        return not self.fail_remove


def test_no_pr_reclaim_label_failure_is_recorded_then_completed_on_recovery(
    tmp_path: Path,
) -> None:
    config, paths, gh = no_pr_bed(tmp_path, 1176, dispatched_at="2026-07-14T17:24:55Z")
    flaky = _FlakyLabelGitHub(repo_root=tmp_path)
    flaky.issues = gh.issues

    run_sweep(tmp_path, paths, config, flaky)

    entry = issue_entry(paths, 1176)
    assert entry["status"] == "dispatched"
    assert entry["worker_pid"] == 99999
    (first,) = events_of(paths, "session_failed_relabeled")
    assert first["payload"]["label_write_ok"] is False
    assert (1176, config.labels.in_progress) in flaky.labels_removed

    flaky.fail_remove = False
    flaky.labels_removed = []
    run_sweep(tmp_path, paths, config, flaky)

    assert (1176, config.labels.in_progress) in flaky.labels_removed
    assert (1176, config.labels.ready) in flaky.labels_added
    events = events_of(paths, "session_failed_relabeled")
    assert [e["payload"]["label_write_ok"] for e in events] == [False, True]
    assert issue_entry(paths, 1176)["status"] == "dispatched"


def test_no_pr_terminal_label_only_is_flagged_and_reaped_after_grace(tmp_path: Path) -> None:
    config = sweep_config(dead_dispatched_reap_minutes=60)
    config, paths, gh = no_pr_bed(
        tmp_path,
        1421,
        config=config,
        labels=(config.labels.human_needed,),
        dispatched_at="2026-08-09T07:33:24Z",
    )

    run_sweep(tmp_path, paths, config, gh)
    entry = issue_entry(paths, 1421)
    # Nothing to reclaim: drift branch stamps both markers, status unchanged.
    assert entry["status"] == "dispatched"
    assert entry["orphan_flagged_at"] is not None
    assert entry["orphan_drift_at"] is not None
    assert gh.labels_added == []

    seed_issue(paths, 1421, orphan_drift_at=iso(minutes_ago=120))
    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 1421)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "dead_dispatched_worker_reap"
    assert entry["reason_class"] == "mechanical"
    (reaped,) = events_of(paths, "dead_dispatched_worker_reaped")
    assert reaped["payload"]["reap_minutes"] == 60


def test_no_pr_throttle_death_backstop_rearms_instead_of_escalating(tmp_path: Path) -> None:
    config = sweep_config(dead_dispatched_reap_minutes=60)
    config, paths, gh = no_pr_bed(
        tmp_path,
        1993,
        config=config,
        labels=(config.labels.human_needed,),
        dead_worker_failure_kind="rate_limited",
        orphan_flagged_at=iso(minutes_ago=180),
        orphan_drift_at=iso(minutes_ago=180),
        orphan_drift_fingerprint='{"dead": true}',
    )

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 1993)
    assert entry["status"] == "dispatched"
    assert entry["throttle_reap_rearm_count"] == 1
    assert "escalation_reason" not in entry
    (event,) = events_of(paths, "dead_dispatched_throttle_rearmed")
    assert event["payload"]["issue_number"] == 1993
    assert events_of(paths, "dead_dispatched_worker_reaped") == []


def _cap_bed(tmp_path: Path, **fields: Any) -> tuple[Any, Any, NoPrGitHub]:
    """Dead dispatch #4 of a no-progress loop: three counted attempts already
    inside the window, none of them this dispatch."""
    config = sweep_config(max_auto_redispatch=3)
    return no_pr_bed(
        tmp_path,
        1243,
        config=config,
        dispatched_at="2026-08-14T00:04:00Z",
        orphan_redispatch_head_sha="none:none",
        orphan_redispatch_counted_dispatch="2026-08-14T00:03:00Z:99999",
        orphan_redispatch_at=[iso(minutes_ago=m) for m in (3, 2, 1)],
        **fields,
    )


def test_no_pr_redispatch_cap_exceeded_without_progress_escalates(tmp_path: Path) -> None:
    config, paths, gh = _cap_bed(tmp_path)

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 1243)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "orphan_sweep_redispatch_cap_exceeded"
    assert entry["reason_class"] == "mechanical"
    (event,) = events_of(paths, "orphan_sweep_redispatch_escalated")
    assert event["payload"]["redispatch_count"] == 4
    assert events_of(paths, "session_failed_relabeled") == []
    # Current wire shape: the escalation swaps the active label for the
    # operator queue (the ready label rides along on the same transition).
    assert (1243, config.labels.in_progress) in gh.labels_removed
    assert (1243, config.labels.operator_queue) in gh.labels_added


def test_no_pr_redispatch_cap_ignores_throttle_death(tmp_path: Path) -> None:
    config, paths, gh = _cap_bed(tmp_path, dead_worker_failure_kind="rate_limited")

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 1243)
    assert entry["status"] == "dispatched"
    assert len(entry["orphan_redispatch_at"]) == 3
    assert events_of(paths, "orphan_sweep_redispatch_escalated") == []
    assert len(events_of(paths, "session_failed_relabeled")) == 1


def test_no_pr_redispatch_cap_resets_on_moving_head(tmp_path: Path) -> None:
    config, paths, gh = _cap_bed(tmp_path)

    run_sweep(
        tmp_path,
        paths,
        config,
        gh,
        patches=(
            patch("charlie_work.workflow.remote_branch_head_sha", return_value="movedhead"),
            patch("charlie_work.workflow.remote_branch_ahead_count", return_value=(0, None)),
        ),
    )

    entry = issue_entry(paths, 1243)
    assert entry["status"] == "dispatched"
    assert len(entry["orphan_redispatch_at"]) == 1
    assert entry["orphan_redispatch_head_sha"] != "none:none"
    assert events_of(paths, "orphan_sweep_redispatch_escalated") == []


def test_no_pr_local_backend_parks_committed_work_for_review(
    tmp_path: Path, shallow_wts: Path
) -> None:
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)
    branch = "agent/issue-1923-x"
    _add_worktree_commit(repo_root, config, branch)
    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(issues_dir, 1923, labels=(labels_cfg.ready, labels_cfg.in_progress))
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, 1923, branch, armed_drift_minutes_ago=120)

    run_sweep(tmp_path, paths, config, gh, fleet_dir=tmp_path / "fleet")

    names = {entry["name"] for entry in gh.issue_view(1923)["labels"]}
    assert labels_cfg.review_ready in names
    assert labels_cfg.in_progress not in names
    entry = issue_entry(paths, 1923)
    # Parked in one write: off dispatched, drift markers cleared, so the
    # armed 120-minute backstop cannot escalate already-parked work.
    assert entry["status"] == PASSIVE_OPEN_STATUS
    assert "orphan_drift_at" not in entry
    (ready,) = events_of(paths, "local_work_ready")
    assert ready["payload"]["branch"] == branch
    assert events_of(paths, "session_failed_relabeled") == []
    assert events_of(paths, "dead_dispatched_worker_reaped") == []


# ---------------------------------------------------------------------------
# No open PR: pushed branch
# ---------------------------------------------------------------------------

_PUSHED_BRANCH = "agent/issue-935-workers-push-a-finished-branch-but-cannot-open-t"


def _pushed_branch_bed(tmp_path: Path, *, pr_create_return: int | None):
    repo_root = make_repo_with_pushed_branch(tmp_path, _PUSHED_BRANCH)
    config, paths, gh = no_pr_bed(tmp_path, 935, repo_root=repo_root, branch_name=_PUSHED_BRANCH)
    gh.dry_run = False
    gh.pr_create_return = pr_create_return
    return config, paths, gh


def test_no_pr_pushed_branch_opens_pr_and_advances_to_pr_open(tmp_path: Path) -> None:
    config, paths, gh = _pushed_branch_bed(tmp_path, pr_create_return=9001)

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 935)
    assert entry["status"] == PASSIVE_OPEN_STATUS
    assert entry["pr_number"] == 9001
    (event,) = events_of(paths, "orphaned_worker_opened_pr")
    assert event["payload"]["reason"] == "dead_worker_branch_pushed_no_pr"
    assert event["payload"]["worker_reported"] is False
    assert events_of(paths, "worker_handoff_pr_opened") == []
    assert gh.prs_created[0]["head"] == _PUSHED_BRANCH
    assert (935, config.labels.in_progress) in gh.labels_removed
    assert (935, config.labels.pr_open) in gh.labels_added


def test_no_pr_pushed_branch_pr_create_failure_is_stranded_not_redispatched(
    tmp_path: Path,
) -> None:
    config, paths, gh = _pushed_branch_bed(tmp_path, pr_create_return=None)
    # A worker-reported push whose PR create failed (terminal record carries it).
    from charlie_work.process_utils import (
        worker_terminal_status_path,
        write_worker_terminal_status,
    )

    write_worker_terminal_status(
        worker_terminal_status_path(sessions_dir_for(tmp_path), 935, "claude"),
        pid=1234,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:05:00Z",
        duration_seconds=300.0,
        worker_outcome={
            "push_succeeded": True,
            "pr_created": False,
            "error": "gh unauthenticated",
        },
    )

    run_sweep(tmp_path, paths, config, gh)

    entry = issue_entry(paths, 935)
    assert entry["status"] == "dispatched"
    assert "pr_number" not in entry
    (event,) = events_of(paths, "pr_create_failed_branch_stranded")
    assert event["payload"]["reason"] == "dead_worker_branch_pushed_pr_create_failed"
    assert event["payload"]["worker_reported"] is True
    assert event["payload"]["pr_create_error"] is not None
    drift = [
        e
        for e in events_of(paths, "orphaned_worker_drift")
        if e["payload"].get("reason") == "dead_worker_no_open_pr"
    ]
    assert drift == []


def test_live_pid_stale_handoff_outcome_opens_pr_without_waiting_for_exit(tmp_path: Path) -> None:
    config, paths, gh = _pushed_branch_bed(tmp_path, pr_create_return=9001)
    write_aged_outcome(
        tmp_path / "repo",
        config,
        _PUSHED_BRANCH,
        {
            "push_succeeded": True,
            "pr_created": False,
            "pr_title": "fix: hand off while the PID lingers",
            "pr_body": "Closes #935\n\nDrafted body.",
        },
        age_seconds=3600,
    )

    run_sweep(tmp_path, paths, config, gh, pid_alive=True)

    entry = issue_entry(paths, 935)
    assert entry["status"] == PASSIVE_OPEN_STATUS
    assert entry["pr_number"] == 9001
    assert gh.prs_created[0]["title"] == "fix: hand off while the PID lingers"
    (event,) = events_of(paths, "worker_handoff_pr_opened")
    assert event["payload"]["worker_pid_still_running"] is True
    assert events_of(paths, "orphaned_worker_opened_pr") == []
    # A completed handoff is not a death.
    assert "worker_death_at" not in entry and "orphan_drift_at" not in entry
