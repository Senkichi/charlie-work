"""Ghost-corroboration coverage for the fleet-wide worker session count.

Split from ``test_fleet_registry.py`` for issue #2230: the characterization
test (plus its post-flip assertion) pushed that file over the 800-line
per-module cap the file-size ratchet enforces (issue #1442), so the ghost
coverage lives in its own module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def test_count_fleet_live_sessions_corroborates_ghost_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Issue #2230 behaviour flip: the fleet-wide worker count now applies the
    same state.json ghost corroboration the review lane has had since #2084.

    A repo whose state.json records a ``dispatched`` issue with a live
    ``worker_pid`` but no session sidecar -- a "ghost", in the sense of issue
    #343 -- counts against the fleet cap. Before the flip the walk counted
    live sidecars only, so the ghost was invisible to
    ``count_fleet_live_sessions`` (and therefore to the fleet
    ``global_max_concurrent_sessions`` cap and the self-deploy sync
    deferral, its two consumers).

    MUTATION GATE: reverting ``count_fleet_live_sessions`` to the pre-flip
    ``iter_workers``-only body makes this test fail -- the count reverts to 0
    and the ghost worker looks like free capacity again.
    """
    from charlie_work import layout
    from charlie_work.fleet_registry import count_fleet_live_sessions

    fleet_dir = tmp_path / ".fleet"
    fleet_dir.mkdir(parents=True)

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    state_dir = repo / ".var" / "charlie-work"
    sessions_dir = layout.sessions_dir_default(state_dir)
    sessions_dir.mkdir(parents=True)

    # Ghost: state.json says issue 7 is dispatched with a live worker_pid
    # (this test process -- a None start_time fails open on the identity
    # check), but no sidecar exists under sessions_dir.
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {
                    "7": {
                        "status": "dispatched",
                        "worker_pid": os.getpid(),
                        "worker_process_start_time": None,
                    }
                },
                "prs": {},
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    (fleet_dir / "fleet.json").write_text(
        json.dumps(
            {
                "version": 1,
                "repos": {
                    "owner/repo": {
                        "repo_root": str(repo),
                        "name_with_owner": "owner/repo",
                        "config_path": str(repo / "orchestrator.config.yaml"),
                        "state_dir": str(state_dir),
                        "first_seen": "2024-01-01T00:00:00Z",
                        "last_seen": "2024-01-01T00:00:00Z",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    live_count, skipped_repos = count_fleet_live_sessions(None)

    assert live_count == 1  # ghost worker counts, same as the review lane
    assert skipped_repos == []
