"""``classify_and_record`` classification and extraction tests.

Split out of ``tests/test_post_mortem.py`` (issue #1567, Track 1): the
worker_blocked detection path (with mutation gate), chat_message extraction
(tool_call_id join, non-JSON degradation), graceful degradation for missing /
schema-drifted / unmatched sessions.db, the disabled no-op, and
config-extensible ``signature_rules``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from _post_mortem_fixtures import (
    _NOW,
    _build_sessions_db,
    _config_with_db,
    _make_worker,
)

from charlie_work.config import (
    SignatureRule,
)
from charlie_work.post_mortem import (
    classify_and_record,
    read_post_mortem,
)


# ---------------------------------------------------------------------------
# worker_blocked detection (with mutation gate)
# ---------------------------------------------------------------------------


def test_classify_and_record_detects_worker_blocked(tmp_path: Path) -> None:
    """A role=tool node whose content is prefixed 'Tool blocked:' (the
    documented push-gate hook block payload shape) must be classified as
    failure_kind='worker_blocked', with the reason stripped of the prefix
    and the terminal tool name recovered.

    MUTATION GATE: this test's ability to fail is verified by mutating
    post_mortem._BLOCK_CONTENT_PREFIX in _classify_nodes (see the module-
    level docstring below for the verbatim transcript recorded during
    development). Reverting the source restores a pass.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[
            ("assistant", "planning next step", "2026-07-11T11:57:00"),
            (
                "tool",
                json.dumps({"tool": "bash", "command": "git push"}),
                "2026-07-11T11:58:00",
            ),
            (
                "tool",
                'Tool blocked: {"decision": "block", "reason": "push-gate hook rejected"}',
                "2026-07-11T11:59:00",
            ),
        ],
    )
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"

    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.failure_kind == "worker_blocked"
    assert record.terminal_reason is not None
    assert record.terminal_reason.startswith('{"decision"')
    assert not record.terminal_reason.startswith("Tool blocked:")


def test_classify_and_record_no_block_signature_returns_none(tmp_path: Path) -> None:
    """A session whose tool nodes contain no block signature must return
    None (fall through to existing log-tail classification) and record a
    post-mortem with failure_kind=None."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[
            ("tool", json.dumps({"tool": "bash", "command": "pytest"}), "2026-07-11T11:57:00"),
        ],
    )
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.failure_kind is None
    assert record.terminal_tool == "bash"


def test_classify_and_record_writes_failure_kind_to_devin_sidecar(tmp_path: Path) -> None:
    """When worker_blocked is detected, the devin-shell sidecar must be
    updated with failure_kind='worker_blocked' directly (the mechanism that
    makes update_session_record_with_failure_classification's existing
    "skip if already classified" short-circuit take over, per the module
    docstring's integration contract)."""
    from charlie_work.devin_shell import SessionRecord, _sidecar_path, _write_json

    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[("tool", "Tool blocked: push rejected", "2026-07-11T11:58:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(adapter_kind="devin")
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)

    sidecar = SessionRecord(
        issue_number=worker.issue_number,
        branch="agent/issue-42",
        worktree_path=worker.worktree_path,
        prompt_path="",
        command=(),
        pid=None,
        started_at=worker.started_at,
        log_path=worker.log_path,
    )
    _write_json(_sidecar_path(sessions_dir, worker.issue_number), sidecar.to_dict())

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)
    assert result == "worker_blocked"

    with _sidecar_path(sessions_dir, worker.issue_number).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["failure_kind"] == "worker_blocked"


def test_extract_resolves_tool_name_via_tool_call_id_join(tmp_path: Path) -> None:
    """The REAL chat_message shape puts the tool name in the assistant
    node's ``tool_calls[].name``, not in the tool-result node's content —
    the extractor must resolve ``MessageNode.tool_name`` by joining the
    tool node's ``tool_call_id`` back to the assistant's tool_calls, and
    classification must surface it as ``terminal_tool``."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[
            (
                "assistant",
                "",
                int(datetime(2026, 7, 11, 11, 57, 0, tzinfo=UTC).timestamp()),
                {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "bash",
                            "arguments": {"command": "git push"},
                            "index": 0,
                            "kind": "function",
                        }
                    ]
                },
            ),
            (
                "tool",
                'Tool blocked: {"decision": "block", "reason": "push-gate hook rejected"}',
                int(datetime(2026, 7, 11, 11, 58, 0, tzinfo=UTC).timestamp()),
                {"tool_call_id": "call_1"},
            ),
        ],
    )
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.terminal_tool == "bash"
    roles = [n.role for n in record.message_nodes]
    assert roles == ["assistant", "tool"]
    tool_node = record.message_nodes[-1]
    assert tool_node.tool_name == "bash"
    assert tool_node.content.startswith("Tool blocked:")
    # Epoch created_at is normalized to ISO-8601 in the sidecar.
    assert tool_node.created_at == "2026-07-11T11:58:00+00:00"


def test_extract_non_json_chat_message_degrades_to_raw_content(tmp_path: Path) -> None:
    """A chat_message blob that is not JSON (or not a dict) must degrade to
    role="" with the raw text as content — never raise, and never abort the
    rest of the extraction (the surrounding parseable nodes still classify)."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[("tool", "Tool blocked: push rejected", "2026-07-11T11:58:00")],
    )
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO message_nodes (session_id, node_id, parent_node_id, "
            "chat_message, created_at) VALUES ('sess-1', 2, 1, ?, ?)",
            ("this is not json at all", "2026-07-11T11:59:00"),
        )
        conn.commit()
    finally:
        conn.close()
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    raw_node = record.message_nodes[-1]
    assert raw_node.role == ""
    assert raw_node.content == "this is not json at all"


# ---------------------------------------------------------------------------
# Degradation: locked / absent / schema-drifted DB never raises
# ---------------------------------------------------------------------------


def test_classify_and_record_missing_db_degrades_gracefully(tmp_path: Path) -> None:
    """A sessions.db path that does not exist must record extraction_error
    and return None — never raise."""
    db_path = tmp_path / "does-not-exist" / "sessions.db"
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False
    assert record.extraction_error is not None


def test_classify_and_record_schema_drift_degrades_gracefully(tmp_path: Path) -> None:
    """A sessions.db whose schema doesn't match what this module expects
    (e.g. missing message_nodes table) must record extraction_error and
    return None — never raise."""
    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(db_path)
    try:
        # Only a "sessions" table with a totally different schema - no
        # working_directory/created_at columns at all.
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, unrelated_column TEXT)")
        conn.execute("INSERT INTO sessions (id, unrelated_column) VALUES ('sess-1', 'x')")
        conn.commit()
    finally:
        conn.close()

    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False
    assert record.extraction_error is not None


def test_classify_and_record_disabled_is_a_noop(tmp_path: Path) -> None:
    """post_mortem.enabled=False must skip extraction entirely (no sidecar
    written), per the config's opt-out contract."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[("tool", "Tool blocked: x", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path, enabled=False)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    assert read_post_mortem(sessions_dir, worker.issue_number) is None


def test_classify_and_record_no_matching_session_degrades_gracefully(tmp_path: Path) -> None:
    """A DB that opens fine but has no session matching this worker's
    working_directory/time window must record matched=False with a reason,
    not raise."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-999-different",
        nodes=[("tool", "Tool blocked: x", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False
    assert record.extraction_error is not None


# ---------------------------------------------------------------------------
# config-extensible signature_rules
# ---------------------------------------------------------------------------


def test_classify_and_record_uses_custom_signature_rules(tmp_path: Path) -> None:
    """signature_rules is config-extensible: a custom pattern/kind not in
    the default two rules must still classify correctly."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[("tool", "CUSTOM_QUOTA_SIGNATURE detected", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(
        db_path,
        signature_rules=(SignatureRule(pattern="CUSTOM_QUOTA_SIGNATURE", kind="worker_blocked"),),
    )
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
