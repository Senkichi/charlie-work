"""Tripwire tests for the shared sessions.db fixture helper.

These tests keep the helper's schema pinned to the real queries in
``src/charlie_work/post_mortem.py``. If the production queries change, the
helper must change too — and the failure is localized here instead of a
scattered drift incident across the test suite.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from _sessions_db_fixtures import make_sessions_db
from charlie_work.config import OrchestratorConfig, PostMortemConfig
from charlie_work.post_mortem import classify_and_record, real_activity_for_worker
from charlie_work.worker import WorkerView


def test_make_sessions_db_schema_satisfies_post_mortem_queries(tmp_path: Path) -> None:
    """The helper-created schema must let every production query run without error.

    Exercises ``_find_matching_session`` and ``_extract_last_n_nodes`` (via
    ``classify_and_record``) and the ``message_nodes`` lookup in
    ``real_activity_for_worker``. A future schema drift in production that is
    not mirrored into the helper will fail here first.
    """
    db_path = tmp_path / "sessions.db"
    worktree_path = "C:/repo/.var/worktrees/issue-42"
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    started_at = "2026-07-11T11:55:00+00:00"
    created_at = "2026-07-11T11:56:00"

    make_sessions_db(
        db_path,
        session_id="sess-1",
        working_directory=worktree_path,
        created_at=created_at,
        rows=[
            {
                "role": "tool",
                "content": (
                    'Tool blocked: {"decision": "block", "reason": "push-gate hook rejected"}'
                ),
                "created_at": created_at,
            }
        ],
    )

    config = OrchestratorConfig(post_mortem=PostMortemConfig(db_path=str(db_path)))
    worker = WorkerView(
        adapter_kind="devin",
        issue_number=42,
        repo_key="",
        pid=None,
        started_at=started_at,
        process_start_time=None,
        log_path=str(Path(worktree_path) / "session.log"),
        worktree_path=worktree_path,
        error=None,
        failure_kind=None,
        reclaimed=None,
    )
    sessions_dir = tmp_path / "sessions"

    # classify_and_record must find the session and parse the message node.
    result = classify_and_record(sessions_dir, config, worker, now=now)
    assert result == "worker_blocked"

    # real_activity_for_worker must read the latest message node created_at.
    probe = real_activity_for_worker(
        config.post_mortem,
        worktree_path,
        started_at,
        pid=None,
        now=now,
    )
    assert probe.latest_source == "sessions.db"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 56, 0, tzinfo=UTC)
