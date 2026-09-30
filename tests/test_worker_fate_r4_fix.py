"""wf-r4 fixes: one #2010 permission-denial gate for every blocked escalation,
and rule 1's stale-evidence event for a dropped stale terminal record."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from _fakes_github import FakeGitHub
from _orphan_sweep_fixtures import _dead_worker_rework_bed, _run_orphan_sweep, _write_outcome

from charlie_work import worker_fate
from charlie_work.config import (
    WORKER_OUTCOME_FILENAME,
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.dead_worker_sweep.live_handoff import collect_stale_live_handoff_pids
from charlie_work.paths import resolved_layout, runtime_paths
from charlie_work.state import PASSIVE_OPEN_STATUS, save_state
from charlie_work.worktree import worktree_path_for_branch
from charlie_work.write_gate import WriteGate
from charlie_work.instrumentation import query_events
from charlie_work.process_utils import write_worker_terminal_status
from charlie_work.state import load_state, parse_iso_timestamp

DENIAL = "Bash was denied. If you approve command execution, I can finish these steps."


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _blocked_payload(detail: str) -> dict[str, Any]:
    return {
        "outcome": "blocked",
        "reason_kind": "other",
        "detail": detail,
        "push_succeeded": True,
        "pr_created": False,
        "head_sha": "abc123",
    }


def _assert_escalation(
    tmp_path: Path, bed: tuple[Any, ...], *, detail: str, escalated: bool
) -> None:
    config, paths, fake_gh, _dispatched_at = bed
    _write_outcome(paths, tmp_path, _blocked_payload(detail), mtime=datetime.now(UTC))

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    declared = [e for e in state.get("events", []) if e.get("kind") == "worker_declared_blocked"]
    if escalated:
        assert entry["status"] == "escalated"
        assert entry["escalation_reason"] == "worker_declared_blocked"
        assert declared
    else:
        # The ordinary dead-worker reset path, positively: the bed's post-state
        # on origin/main, not merely "not escalated".
        assert entry["status"] == "rework_requested"
        assert entry["dispatched_at"] is None
        assert entry.get("escalation_reason") != "worker_declared_blocked"
        assert not declared
        assert (207, config.labels.operator_queue) not in fake_gh.labels_added


@pytest.mark.parametrize(
    ("decision", "pr_state_status"),
    [("request_changes", None), ("approved", "rework_requested")],
)
def test_with_pr_permission_denial_blocked_outcome_is_not_operator_escalated(
    tmp_path: Path, decision: str, pr_state_status: str | None
) -> None:
    """Issue #2010, with-PR lane (both sweep sites): the headless
    permission-denial signature is a worker-config defect, not a blocked task,
    so it takes the ordinary reset path exactly as on origin/main."""
    bed = _dead_worker_rework_bed(tmp_path, decision=decision, pr_state_status=pr_state_status)
    _assert_escalation(tmp_path, bed, detail=DENIAL, escalated=False)


@pytest.mark.parametrize(
    ("decision", "pr_state_status"),
    [("request_changes", None), ("approved", "rework_requested")],
)
def test_with_pr_genuine_blocked_outcome_still_escalates(
    tmp_path: Path, decision: str, pr_state_status: str | None
) -> None:
    """Positive control for the test above: same bed, non-denial detail."""
    bed = _dead_worker_rework_bed(tmp_path, decision=decision, pr_state_status=pr_state_status)
    _assert_escalation(tmp_path, bed, detail="needs a human to disambiguate", escalated=True)


def _write_stale_terminal(tmp_path: Path) -> Path:
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    write_worker_terminal_status(
        sessions_dir / "issue-207.claude-code.terminal.json",
        pid=4242,
        exit_code=0,
        started_at=_iso(now - timedelta(hours=3, minutes=5)),
        ended_at=_iso(now - timedelta(hours=3)),
        duration_seconds=300.0,
    )
    return sessions_dir


def test_sweep_emits_stale_evidence_event_for_dropped_stale_terminal_record(
    tmp_path: Path,
) -> None:
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="request_changes"
    )
    _write_stale_terminal(tmp_path)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    events = query_events(paths.state_file, kind="worker_evidence_stale")
    stale = [e for e in events if e["payload"]["source"] == "terminal"]
    assert len(stale) == 1
    assert stale[0]["payload"]["reason"] == "older_than_dispatch"
    assert load_state(paths.state_file)["issues"]["207"]["stale_evidence_reported"]


def test_dead_dispatched_reap_reports_dropped_stale_terminal_record(tmp_path: Path) -> None:
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="request_changes"
    )
    now = datetime.now(UTC)
    _write_stale_terminal(tmp_path)
    # Drift was first surfaced two hours ago: past the 60-minute #654 grace.
    state = load_state(paths.state_file)
    state["issues"]["207"]["orphan_drift_at"] = _iso(now - timedelta(hours=2))
    save_state(paths.state_file, state)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    assert state["issues"]["207"]["escalation_reason"] == "dead_dispatched_worker_reap"
    events = query_events(paths.state_file, kind="worker_evidence_stale")
    assert [e["payload"]["source"] for e in events] == ["terminal"]


def test_fresh_terminal_record_reports_only_what_it_drops() -> None:
    dispatched = parse_iso_timestamp("2026-09-29T10:00:00Z")
    stale = {"ended_at": "2026-09-29T09:00:00Z", "exit_code": 0}
    fresh = {"ended_at": "2026-09-29T10:05:00Z", "exit_code": 0}
    fates: list[worker_fate.WorkerFate] = []

    assert (
        worker_fate.fresh_terminal_record(fresh, dispatched, issue_number=7, on_fate=fates.append)
        is fresh
    )
    assert fates == []
    assert (
        worker_fate.fresh_terminal_record(stale, dispatched, issue_number=7, on_fate=fates.append)
        is None
    )
    (fate,) = fates
    assert fate.basis.issue_number == 7
    (evidence,) = fate.basis.stale
    assert evidence.source is worker_fate.EvidenceSource.TERMINAL
    assert evidence.written_at == parse_iso_timestamp("2026-09-29T09:00:00Z")


# -- B1: the #2010 exemption decides the fate once, not only the escalation --


def _no_pr_pushed_bed(tmp_path: Path, *, detail: str) -> tuple[Any, Any, Path, Any]:
    """A dead-PID, no-PR issue 2010 whose fresh worktree outcome is a pushed
    ``blocked`` declaration with ``detail`` (remote ahead count is patched by
    the caller)."""
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    branch = "agent/issue-2010-test"
    state = load_state(paths.state_file)
    state["issues"]["2010"] = {
        "status": "dispatched",
        "dispatched_at": _iso(datetime.now(UTC) - timedelta(hours=1)),
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "branch_name": branch,
    }
    save_state(paths.state_file, state)
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = worktree_path_for_branch(
        tmp_path, branch, resolved_layout(config, tmp_path).worktrees
    )
    worktree_path.mkdir(parents=True, exist_ok=True)
    (worktree_path / WORKER_OUTCOME_FILENAME).write_text(
        json.dumps(_blocked_payload(detail)), encoding="utf-8"
    )

    class _NoPrGitHub(FakeGitHub):
        def pr_list(self) -> list[dict[str, Any]]:
            return []

    fake_gh = _NoPrGitHub(repo_root=tmp_path)
    fake_gh.issues = [
        {
            "number": 2010,
            "title": "test issue",
            "url": "https://example.test/issues/2010",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []
    return config, paths, sessions_dir, fake_gh


def _sweep_no_pr_pushed(detail: str, tmp_path: Path) -> tuple[dict[str, Any], list[Any]]:
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config, paths, sessions_dir, fake_gh = _no_pr_pushed_bed(tmp_path, detail=detail)
    opened: list[Any] = []

    def _fake_open_pr(**kwargs: Any) -> tuple[int, None, None]:
        opened.append(kwargs["branch"])
        return 555, None, None

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch("charlie_work.workflow.remote_branch_head_sha", return_value="abc123"),
        patch("charlie_work.workflow.remote_branch_ahead_count", return_value=(3, None)),
        patch("charlie_work.workflow._open_pr_for_orphaned_branch", side_effect=_fake_open_pr),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            fake_gh,
            write_gate=WriteGate(dry_run=False, state_path=paths.state_file, repo="charlie-work"),
        )
    return load_state(paths.state_file)["issues"]["2010"], opened


def test_no_pr_permission_denial_blocked_with_pushed_commits_still_opens_the_pr(
    tmp_path: Path,
) -> None:
    """Issue #2010: the exemption must not also suppress the pushed-branch
    PR-open path. On origin/main a permission-denial worker whose branch is
    ahead got its PR opened; the exempt Blocked fate used to shadow
    PushedWithoutPr and send the issue to reclaim/redispatch instead."""
    entry, opened = _sweep_no_pr_pushed(DENIAL, tmp_path)

    assert opened == ["agent/issue-2010-test"]
    assert entry["status"] == PASSIVE_OPEN_STATUS
    assert entry["pr_number"] == 555


def test_no_pr_genuine_blocked_with_pushed_commits_escalates_without_a_pr(
    tmp_path: Path,
) -> None:
    """Positive control: same bed, non-denial detail -> escalated, no PR."""
    entry, opened = _sweep_no_pr_pushed("needs a human to disambiguate", tmp_path)

    assert opened == []
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worker_declared_blocked"


def _live_handoff_candidates(tmp_path: Path, detail: str) -> dict[int, dict[str, Any]]:
    now = datetime.now(UTC)
    branch = "agent/issue-9200"
    worktrees_dir = tmp_path / "worktrees"
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / WORKER_OUTCOME_FILENAME
    outcome_path.write_text(json.dumps(_blocked_payload(detail)), encoding="utf-8")
    fresh_ts = (now - timedelta(minutes=20)).timestamp()
    os.utime(outcome_path, (fresh_ts, fresh_ts))
    return collect_stale_live_handoff_pids(
        {
            9200: {
                "branch_name": branch,
                "worker_pid": 4242,
                "dispatched_at": _iso(now - timedelta(hours=1)),
            }
        },
        worker_outcome_finalize_minutes=15,
        repo_root=tmp_path,
        worktrees_dir=worktrees_dir,
        now=now,
        sessions_dir=tmp_path / "sessions",
    )


def test_live_handoff_routes_a_permission_denial_blocked_push_as_on_main(tmp_path: Path) -> None:
    assert set(_live_handoff_candidates(tmp_path, DENIAL)) == {9200}


def test_live_handoff_still_skips_a_genuine_blocked_declaration(tmp_path: Path) -> None:
    """Positive control for the test above."""
    assert _live_handoff_candidates(tmp_path, "needs a human to disambiguate") == {}
