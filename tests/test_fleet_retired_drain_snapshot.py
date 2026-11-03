"""Drain rule for retired repos in the supervisor snapshot (issue #2754).

``fleet stop --drain`` exits when ``fleet_live_worker_count`` reaches zero, and
that sums ``_take_fleet_snapshot``. A retired repo that is still winding down
must therefore stay in the snapshot until it is quiescent -- otherwise the
supervisor can exit while the repo still has a worker running -- while a
quiescent retired repo drops out entirely (no deltas, no per-pass noise).
The session/review caps are a separate decision and still exclude every
retired repo (``test_non_quiescent_retired_repo_is_excluded_from_fleet_caps``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from charlie_work.fleet_dispatch import _take_fleet_snapshot
from charlie_work.fleet_stop import fleet_live_worker_count

_KEY = "local/localrepo"


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fleet(tmp_path: Path, *, retired: bool, worker: bool) -> Path:
    """A one-repo fleet; ``worker`` seeds one worker sidecar in its sessions dir."""
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        f"fleet:\n  retired: {str(retired).lower()}\nlocal_issues:\n  enabled: true\n",
        encoding="utf-8",
    )
    state_dir = repo / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    if worker:
        sessions = state_dir / "dispatches" / "sessions"
        sessions.mkdir(parents=True)
        (sessions / "issue-1.json").write_text(
            json.dumps(
                {
                    "issue_number": 1,
                    "branch": "agent/issue-1",
                    "worktree_path": str(repo / "worktrees" / "issue-1"),
                    "prompt_path": str(repo / "prompt.md"),
                    "command": ["devin"],
                    "pid": 1234,
                    "started_at": _now_iso(),
                    "log_path": str(repo / "log.txt"),
                }
            ),
            encoding="utf-8",
        )
    stamp = _now_iso()
    entry = {
        "repo_root": str(repo),
        "name_with_owner": _KEY,
        "config_path": str(config),
        "state_dir": str(state_dir),
        "first_seen": stamp,
        "last_seen": stamp,
    }
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": {_KEY: entry}}), encoding="utf-8"
    )
    return fleet_dir


@pytest.mark.parametrize(
    ("retired", "worker", "alive", "in_snapshot", "live_workers"),
    [
        # Positive control: the count machinery does see a live worker, so the
        # retired rows below are the retirement rule, not an inert fixture.
        (False, True, True, True, 1),
        # The drain rule: a winding-down retired repo's live worker still counts.
        (True, True, True, True, 1),
        # A dead sidecar is unfinished reap work: still winding down, so the
        # repo stays in the snapshot (its delta triggers the reaping pass).
        (True, True, False, True, 0),
        # Quiescent retired repo: omitted entirely.
        (True, False, False, False, 0),
    ],
    ids=["active-live", "retired-live", "retired-pending-reap", "retired-quiescent"],
)
def test_retired_repo_leaves_the_snapshot_only_once_quiescent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retired: bool,
    worker: bool,
    alive: bool,
    in_snapshot: bool,
    live_workers: int,
) -> None:
    fleet_dir = _fleet(tmp_path, retired=retired, worker=worker)
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_a, **_k: alive)

    snapshot = _take_fleet_snapshot(fleet_dir_override=str(fleet_dir))

    assert {key for key, _snap in snapshot.repo_snapshots} == ({_KEY} if in_snapshot else set())
    assert fleet_live_worker_count(snapshot) == live_workers


def test_unreadable_live_work_keeps_a_retired_repo_in_the_drain_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error reading live work must fail toward keeping the drain alive."""
    fleet_dir = _fleet(tmp_path, retired=True, worker=True)
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_a, **_k: True)

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("sessions dir unreadable")

    monkeypatch.setattr("charlie_work.repo_retirement.repo_live_work", _boom)

    snapshot = _take_fleet_snapshot(fleet_dir_override=str(fleet_dir))

    assert {key for key, _snap in snapshot.repo_snapshots} == {_KEY}
    assert fleet_live_worker_count(snapshot) == 1
