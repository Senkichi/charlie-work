"""Tripwire tests for the shared sessions.db fixture helper.

These tests keep the helper's schema pinned to the real queries in
``src/charlie_work/post_mortem.py``. If the production queries change, the
helper must change too — and the failure is localized here instead of a
scattered drift incident across the test suite.
"""

from __future__ import annotations

import ast
import re
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


# Match the SQL used to create a message_nodes table, including the
# IF NOT EXISTS variant. `\s+` already spans newlines, so this catches DDL
# wrapped across multiple lines inside a single string literal (a case a
# per-line scan misses entirely). The trailing `\b` avoids false positives
# on names like `message_nodes_archive`.
_MESSAGE_NODES_DDL_PATTERN = re.compile(
    r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+message_nodes\b",
    re.IGNORECASE,
)


class _StringLiteralCollector(ast.NodeVisitor):
    """Collects every string constant in a module.

    A plain tree walk over ``ast.Constant`` nodes already reaches f-string
    literal segments (the non-interpolated parts of an ``ast.JoinedStr``)
    and docstrings, since both are represented as ``Constant`` nodes in the
    tree — no special-casing required beyond visiting ``Constant``.
    """

    def __init__(self) -> None:
        self.string_nodes: list[ast.Constant] = []

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.string_nodes.append(node)
        self.generic_visit(node)


def test_no_hand_rolled_message_nodes_ddl_in_tests() -> None:
    """Every DDL that creates the ``message_nodes`` table in tests must live
    in ``_sessions_db_fixtures.py``.

    Hand-rolled SQL that creates a ``message_nodes`` table in test fixtures is
    the drift class that broke main in incident #316. Use the shared
    ``make_sessions_db`` helper (or add behavior there) instead of reintroducing
    a one-off table definition.

    This walks the AST of every test file and matches the DDL pattern against
    each string literal's *full* value (per the AST-based precedent in
    ``test_load_state_locked.py``), rather than scanning line-by-line — a
    per-line scan cannot see DDL that a literal wraps across multiple lines,
    e.g. ``"CREATE TABLE IF NOT EXISTS\\nmessage_nodes ("``.
    """
    tests_dir = Path(__file__).resolve().parent

    offenders: list[str] = []
    for source_file in sorted(tests_dir.rglob("*.py")):
        if source_file.name == "_sessions_db_fixtures.py":
            continue

        try:
            source_text = source_file.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            offenders.append(
                f"{source_file.relative_to(tests_dir.parent)}: "
                f"could not decode as UTF-8 to scan for hand-rolled DDL ({exc})"
            )
            continue

        tree = ast.parse(source_text, filename=str(source_file))
        collector = _StringLiteralCollector()
        collector.visit(tree)

        for node in collector.string_nodes:
            if _MESSAGE_NODES_DDL_PATTERN.search(node.value):
                rel_path = source_file.relative_to(tests_dir.parent)
                snippet = " ".join(node.value.split())
                if len(snippet) > 120:
                    snippet = snippet[:117] + "..."
                offenders.append(f"{rel_path}:{node.lineno}: {snippet!r}")

    assert not offenders, (
        "Hand-rolled message_nodes DDL found in tests. "
        "Use _sessions_db_fixtures.make_sessions_db instead:\n" + "\n".join(offenders)
    )
