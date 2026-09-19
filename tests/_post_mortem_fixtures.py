"""Shared fixtures for the ``tests/test_post_mortem_*.py`` siblings (issue #1567).

Moved verbatim out of ``tests/test_post_mortem.py``: the ``WorkerView``
builder, the sessions.db fixture wrappers around ``make_sessions_db`` (the
real-schema helper in ``_sessions_db_fixtures``), the ``OrchestratorConfig``
builder, and the shared ``_NOW`` instant.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _sessions_db_fixtures import make_sessions_db

from charlie_work.config import OrchestratorConfig, PostMortemConfig
from charlie_work.worker import WorkerView


def _make_worker(
    *,
    issue_number: int = 42,
    worktree_path: str = "C:/repo/.var/worktrees/issue-42",
    started_at: str = "2026-07-11T11:55:00+00:00",
    adapter_kind: str = "devin",
    pid: int | None = None,
) -> WorkerView:
    return WorkerView(
        adapter_kind=adapter_kind,
        issue_number=issue_number,
        repo_key="",
        pid=pid,
        started_at=started_at,
        process_start_time=None,
        log_path=str(Path(worktree_path) / "session.log"),
        worktree_path=worktree_path,
        error=None,
        failure_kind=None,
        reclaimed=None,
    )


def _node_to_row(spec: tuple | list) -> dict[str, Any]:
    """Convert a ``(role, content, created_at)`` or ``(role, content, created_at, extra)``
    tuple into the row-dict format used by ``make_sessions_db``.
    """
    role, content, created_at = spec[0], spec[1], spec[2]
    row: dict[str, Any] = {"role": role, "content": content, "created_at": created_at}
    if len(spec) > 3:
        extra = spec[3]
        if extra:
            row["extra"] = extra
    return row


def _build_sessions_db(
    db_path: Path,
    *,
    session_id: str = "sess-1",
    working_directory: str = "C:/repo/.var/worktrees/issue-42",
    created_at: str | int = "2026-07-11T11:56:00",
    nodes: tuple[tuple, ...] | list[tuple] = (),
) -> None:
    """Build a fixture sessions.db using the shared real-schema helper.

    ``nodes`` entries are ``(role, content, created_at)`` or
    ``(role, content, created_at, extra)`` where ``extra`` is a dict merged
    into the ``chat_message`` JSON blob (e.g. ``tool_calls`` on an
    assistant node, ``tool_call_id`` on a tool-result node).
    """
    make_sessions_db(
        db_path,
        session_id=session_id,
        working_directory=working_directory,
        created_at=created_at,
        rows=[_node_to_row(spec) for spec in nodes],
    )


def _insert_session_row(
    db_path: Path,
    *,
    session_id: str,
    working_directory: str,
    created_at: str,
    nodes: list[tuple[str, str, str]] = (),
) -> None:
    """Insert an additional session row and its message_nodes into an existing
    fixture sessions.db created by ``_build_sessions_db``.
    """
    make_sessions_db(
        db_path,
        session_id=session_id,
        working_directory=working_directory,
        created_at=created_at,
        rows=[_node_to_row(spec) for spec in nodes],
    )


def _config_with_db(db_path: Path, **overrides: object) -> OrchestratorConfig:
    pm_kwargs: dict[str, Any] = {"db_path": str(db_path)}
    pm_kwargs.update(overrides)
    return OrchestratorConfig(post_mortem=PostMortemConfig(**pm_kwargs))


_NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
