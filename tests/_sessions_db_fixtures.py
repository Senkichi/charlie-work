"""Shared sessions.db fixture helpers.

The real ``message_nodes`` schema was verified against a live Devin CLI
sessions.db (2026-07-12, ~268k rows): role and content live inside the
``chat_message`` JSON blob, per-session ordering is by ``node_id``, and
``created_at`` is an epoch integer. These helpers create the real tables and
populate them from intent-oriented row dicts so tests never hand-roll DDL.

Keep this module in sync with ``src/charlie_work/post_mortem.py``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


# Real production DDL, copied verbatim from a live sessions.db (2026-07-12).
# See this module's docstring.
SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  working_directory TEXT NOT NULL,
  backend_type TEXT NOT NULL,
  model TEXT NOT NULL,
  agent_mode TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  last_activity_at INTEGER NOT NULL,
  title TEXT,
  main_chain_id INTEGER,
  shell_last_seen_index INTEGER DEFAULT 0,
  cogs_json TEXT,
  workspace_dirs TEXT,
  hidden INTEGER NOT NULL DEFAULT 0,
  metadata TEXT
)
"""

MESSAGE_NODES_DDL = """
CREATE TABLE IF NOT EXISTS message_nodes (
  row_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  node_id INTEGER NOT NULL,
  parent_node_id INTEGER,
  chat_message TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  metadata TEXT,
  FOREIGN KEY (session_id) REFERENCES sessions(id),
  UNIQUE(session_id, node_id)
)
"""


def make_sessions_db(
    db_path: Path,
    *,
    session_id: str = "sess-1",
    working_directory: str = "C:/repo/.var/worktrees/issue-42",
    created_at: str | int = "2026-07-11T11:56:00",
    last_activity_at: str | int | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> None:
    """Create ``sessions.db`` tables and insert one session with its message nodes.

    ``rows`` are dicts with ``role``, ``content``, and ``created_at``. Optional
    ``node_id``/``parent_node_id`` override the defaults (sequential ``node_id``
    from 1, ``parent_node_id`` = ``node_id - 1`` for ``node_id > 1``). Any
    ``extra`` dict is merged into the ``chat_message`` JSON blob, so callers can
    add ``tool_calls`` or ``tool_call_id`` without hand-rolling the JSON.

    ``created_at`` values are inserted as given: the real columns are declared
    INTEGER, but SQLite affinity stores a non-numeric ISO string as TEXT
    exactly the same way the production schema does.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if last_activity_at is None:
        last_activity_at = created_at

    rows = rows or []
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(SESSIONS_DDL)
        conn.execute(MESSAGE_NODES_DDL)
        conn.execute(
            "INSERT INTO sessions (id, working_directory, backend_type, model, "
            "agent_mode, created_at, last_activity_at) VALUES (?, ?, '', '', '', ?, ?)",
            (session_id, working_directory, created_at, last_activity_at),
        )
        _insert_message_nodes(conn, session_id, rows)
        conn.commit()
    finally:
        conn.close()


def _insert_message_nodes(
    conn: sqlite3.Connection,
    session_id: str,
    rows: list[dict[str, Any]],
) -> None:
    for index, row in enumerate(rows, start=1):
        node_id = row.get("node_id", index)
        parent_node_id = row.get("parent_node_id")
        if parent_node_id is None and node_id > 1:
            parent_node_id = node_id - 1

        chat_message = {
            "message_id": f"msg-{node_id}",
            "role": row["role"],
            "content": row["content"],
            "metadata": None,
        }
        extra = row.get("extra")
        if isinstance(extra, dict):
            chat_message.update(extra)

        conn.execute(
            "INSERT INTO message_nodes (session_id, node_id, parent_node_id, "
            "chat_message, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                node_id,
                parent_node_id,
                json.dumps(chat_message, separators=(",", ":")),
                row["created_at"],
            ),
        )
