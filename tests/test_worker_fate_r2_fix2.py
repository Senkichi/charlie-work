"""Regressions for the wf-r2 review-2 nits: N1 (live handoff never finalizes a
blocked declaration), N3 (stranded-salvage failure events are deduplicated),
N4 (the with-PR blocked check reads the terminal record's embedded outcome once
the worktree is reaped) and N5 (the writer-marker liveness check goes through the
single ``worker_fate.is_alive`` seam).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from _orphan_sweep_fixtures import _dead_worker_rework_bed, _run_orphan_sweep
from _unescalate_fixtures import _events, _stranded_worktree_bed
from charlie_work.config import WORKER_OUTCOME_FILENAME
from charlie_work.foreign_worktree import write_worktree_marker
from charlie_work.live_handoff_finalize import collect_stale_live_handoff_pids
from charlie_work.process_utils import write_worker_terminal_status
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.worktree import worktree_path_for_branch


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def test_n1_live_handoff_does_not_finalize_a_blocked_declaration(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    branch = "agent/issue-9107"
    worktrees_dir = tmp_path / "worktrees"
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / WORKER_OUTCOME_FILENAME
    outcome_path.write_text(
        json.dumps(
            {
                "outcome": "blocked",
                "reason_kind": "ambiguous_scope",
                "push_succeeded": True,
                "pr_created": False,
                "head_sha": "b01dface",
            }
        ),
        encoding="utf-8",
    )
    fresh_ts = (now - timedelta(minutes=2)).timestamp()
    os.utime(outcome_path, (fresh_ts, fresh_ts))

    candidates = collect_stale_live_handoff_pids(
        {9107: {"branch_name": branch, "worker_pid": 4242}},
        worker_outcome_finalize_minutes=15,
        repo_root=tmp_path,
        worktrees_dir=worktrees_dir,
        now=now,
        sessions_dir=tmp_path / "sessions",
    )

    assert candidates == {}


def test_n4_with_pr_blocked_declaration_survives_a_reaped_worktree(tmp_path: Path) -> None:
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="request_changes"
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    # No worktree outcome file on disk: only the watcher's embedded copy.
    write_worker_terminal_status(
        sessions_dir / "issue-207.claude.terminal.json",
        pid=99999,
        exit_code=0,
        started_at=_iso(now - timedelta(minutes=10)),
        ended_at=_iso(now - timedelta(minutes=1)),
        duration_seconds=540.0,
        worker_outcome={
            "outcome": "blocked",
            "reason_kind": "ambiguous_scope",
            "detail": "needs a human to disambiguate",
            "head_sha": "abc123",
        },
        worker_outcome_written_at=_iso(now - timedelta(minutes=2)),
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worker_declared_blocked"
    assert (207, config.labels.operator_queue) in fake_gh.labels_added


def _seed_stranded_issue(app, branch: str) -> None:
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "worktree_unsafe_local_commits",
            "reason_class": "judgment",
            "branch_name": branch,
        }
        save_state(app.paths.state_file, state)


def test_n5_writer_marker_liveness_goes_through_worker_fate_is_alive(tmp_path: Path) -> None:
    branch = "agent/issue-123-marker-seam"
    app, _repo, wt_path, _remote = _stranded_worktree_bed(tmp_path, branch, origin=False)
    _seed_stranded_issue(app, branch)
    # A pid that is not alive for real; only the patched seam says it is.
    write_worktree_marker(wt_path, 2_999_999, "worker-session-1", kind="worker")

    with patch("charlie_work.worker_fate.is_alive", return_value=True):
        result = app.unescalate(None, 123, dry_run=False)

    assert result.data.get("worktree_still_unsafe") is True
    failed = _events(load_state(app.paths.state_file), "worktree_unsafe_stranded_salvage_failed")
    assert [e["payload"]["skip_reason"] for e in failed] == ["live_writer_marker"]


def test_n3_repeated_identical_salvage_failure_is_recorded_once(tmp_path: Path) -> None:
    branch = "agent/issue-123-dedup"
    app, _repo, wt_path, _remote = _stranded_worktree_bed(tmp_path, branch, origin=False)
    _seed_stranded_issue(app, branch)
    write_worktree_marker(wt_path, 2_999_999, "worker-session-1", kind="worker")

    with patch("charlie_work.worker_fate.is_alive", return_value=True):
        for _ in range(3):
            assert app._salvage_stranded_before_clear(123, branch, wt_path) is None

    state = load_state(app.paths.state_file)
    assert len(_events(state, "worktree_unsafe_stranded_salvage_failed")) == 1
