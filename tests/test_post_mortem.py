"""Tests for post_mortem.py (issue #261): worker post-mortem extraction from
the Devin CLI's local session store, and the worker_blocked classification
that suppresses hot redispatch into a push-gate hook.

The fixture sessions.db built here uses the REAL production schema, copied
verbatim from a live ``%APPDATA%\\devin\\cli\\sessions.db`` (2026-07-12,
~268k message_nodes rows): message_nodes has no ``id``/``role``/``content``
columns — role and content live inside the ``chat_message`` JSON blob,
ordering is by per-session ``node_id``, and ``created_at`` is an epoch
integer. An earlier revision of these fixtures used the schema post_mortem
merely assumed, which let every test pass while the real queries failed on
``no such column: role`` in production. Schema-mismatch degradation to a
recorded ``extraction_error`` is verified separately below.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


from charlie_work.attempt_refs import AttemptSnapshot
from charlie_work.config import OrchestratorConfig, PostMortemConfig, SignatureRule, WatchdogConfig
from charlie_work.post_mortem import (
    ActivitySource,
    MessageNode,
    RealActivityProbe,
    classify_and_record,
    merge_attempt_snapshot,
    read_post_mortem,
    real_activity_for_worker,
)
from charlie_work.worker import WorkerView

from _sessions_db_fixtures import make_sessions_db


_NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)


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


def test_classify_and_record_locked_db_falls_back_to_temp_copy(tmp_path: Path) -> None:
    """A sessions.db held open with an exclusive lock by another connection
    must not prevent extraction — the copy-to-temp fallback must still
    succeed (or, if it can't, degrade gracefully; either way, never raise).
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[("tool", json.dumps({"tool": "bash"}), "2026-07-11T11:57:00")],
    )

    # Hold an exclusive lock on the DB via a separate connection + BEGIN EXCLUSIVE,
    # simulating the live Devin CLI process holding the file open.
    locker = sqlite3.connect(db_path, timeout=1)
    locker.execute("BEGIN EXCLUSIVE")

    try:
        config = _config_with_db(db_path)
        worker = _make_worker()
        sessions_dir = tmp_path / "sessions"

        # Must not raise regardless of whether the fallback succeeds.
        classify_and_record(sessions_dir, config, worker, now=_NOW)

        record = read_post_mortem(sessions_dir, worker.issue_number)
        assert record is not None
        # Either the temp-copy fallback succeeded (matched=True) or it also
        # failed and extraction_error was recorded — both are acceptable
        # degradation outcomes; a raised exception is the only failure.
        assert record.matched is True or record.extraction_error is not None
    finally:
        locker.rollback()
        locker.close()


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
# Log-tail fallback for worker_blocked (issue #260, corrected premise): when
# sessions.db extraction misses entirely (unavailable / no matching session),
# the worker's own log tail is the only remaining signal. "A tool was
# rejected by the user" is the Devin CLI's own log/stdout surfacing of a
# PreToolUse hook block -- distinct from the "Tool blocked:" prefix that
# appears in sessions.db message-node content, which the tests above cover.
# ---------------------------------------------------------------------------


def test_classify_and_record_log_tail_fallback_detects_worker_blocked_when_db_unavailable(
    tmp_path: Path,
) -> None:
    """(a) A tool-rejection tail with sessions.db unavailable (no session
    matches this worker's working_directory/time window) must still
    classify as worker_blocked via the log-tail fallback -- no
    throttled_until, no retry semantics, and the sidecar failure_kind is
    written the same way the DB-based path does (so the escalate/suppress-
    redispatch integration in workflow.py / reconcile.py picks it up
    unchanged; see test_charlie_work.py's
    test_classify_dead_sessions_worker_blocked_log_tail_fallback_escalates_and_suppresses_redispatch
    for the full escalation-level mirror of
    test_classify_dead_sessions_worker_blocked_escalates_and_suppresses_redispatch).

    MUTATION GATE: commenting out the
    `_classify_worker_blocked_from_log_tail(...)` branch in
    post_mortem.classify_and_record (so only the DB-based path can ever
    return "worker_blocked") makes this test fail with
    `assert None == "worker_blocked"`. Restoring the branch passes again.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        # A session exists but for an unrelated working_directory, so
        # _find_matching_session never matches this worker -- DB-based
        # extraction degrades to matched=False, exactly as if the DB were
        # entirely empty for this worker.
        working_directory="C:/repo/.var/worktrees/issue-999-different",
        nodes=[("tool", json.dumps({"tool": "bash"}), "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)

    log_path = tmp_path / "issue-42.log"
    log_path.write_text("Error: A tool was rejected by the user.\n", encoding="utf-8")
    worker = WorkerView(
        adapter_kind="devin",
        issue_number=42,
        repo_key="",
        pid=None,
        started_at="2026-07-11T11:55:00+00:00",
        process_start_time=None,
        log_path=str(log_path),
        worktree_path="C:/repo/.var/worktrees/issue-42",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"

    # The DB-based path still recorded its own (unmatched) diagnostic --
    # the log-tail fallback is additive, it does not overwrite that record.
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False


def test_classify_and_record_log_tail_fallback_writes_failure_kind_to_devin_sidecar(
    tmp_path: Path,
) -> None:
    """Same as test_classify_and_record_writes_failure_kind_to_devin_sidecar
    but via the log-tail fallback (DB unavailable) rather than the DB path --
    proves the sidecar-write integration is identical regardless of which
    source detected worker_blocked."""
    from charlie_work.devin_shell import SessionRecord, _sidecar_path, _write_json

    # No sessions.db at all -- _open_readonly fails outright.
    config = _config_with_db(tmp_path / "does-not-exist" / "sessions.db")

    log_path = tmp_path / "issue-42.log"
    log_path.write_text("Error: A tool was rejected by the user.\n", encoding="utf-8")
    worker = WorkerView(
        adapter_kind="devin",
        issue_number=42,
        repo_key="",
        pid=None,
        started_at="2026-07-11T11:55:00+00:00",
        process_start_time=None,
        log_path=str(log_path),
        worktree_path="C:/repo/.var/worktrees/issue-42",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )
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
    # No throttle retry semantics leak into the sidecar from this path.
    assert "throttled_until" not in payload


def test_classify_and_record_log_tail_fallback_ignores_genuine_rate_limit_tail(
    tmp_path: Path,
) -> None:
    """(b) A genuine provider rate-limit tail (no worker_blocked signature
    among default signature_rules) must NOT be classified worker_blocked by
    the log-tail fallback -- rate-limit retry/cooldown is a separate,
    adapter-owned concern (devin_shell._classify_session_failure /
    get_rate_limit_defer_until via throttle_signatures.match_throttle_tail),
    not this module's."""
    config = _config_with_db(tmp_path / "does-not-exist" / "sessions.db")

    log_path = tmp_path / "issue-42.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )
    worker = WorkerView(
        adapter_kind="devin",
        issue_number=42,
        repo_key="",
        pid=None,
        started_at="2026-07-11T11:55:00+00:00",
        process_start_time=None,
        log_path=str(log_path),
        worktree_path="C:/repo/.var/worktrees/issue-42",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None


def test_classify_and_record_log_tail_fallback_unknown_tail_returns_none(
    tmp_path: Path,
) -> None:
    """(c) An unrelated log tail (neither a worker_blocked signature nor a
    throttle signature) must fall through to None, leaving the caller's
    fallback_kind ("stalled") to apply -- proven at the workflow level by
    the existing test_update_session_record_unknown_tail_falls_back_to_stalled."""
    config = _config_with_db(tmp_path / "does-not-exist" / "sessions.db")

    log_path = tmp_path / "issue-42.log"
    log_path.write_text("Error: something completely unrelated went wrong\n", encoding="utf-8")
    worker = WorkerView(
        adapter_kind="devin",
        issue_number=42,
        repo_key="",
        pid=None,
        started_at="2026-07-11T11:55:00+00:00",
        process_start_time=None,
        log_path=str(log_path),
        worktree_path="C:/repo/.var/worktrees/issue-42",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None


def test_classify_and_record_log_tail_fallback_skipped_when_disabled(tmp_path: Path) -> None:
    """post_mortem.enabled=False must also suppress the log-tail fallback --
    it shares the same opt-out contract as the DB-based path (both derive
    from PostMortemConfig.signature_rules), so a fully disabled section
    means fully disabled, not "DB off but log-tail fallback still on."""
    config = _config_with_db(tmp_path / "does-not-exist" / "sessions.db", enabled=False)

    log_path = tmp_path / "issue-42.log"
    log_path.write_text("Error: A tool was rejected by the user.\n", encoding="utf-8")
    worker = WorkerView(
        adapter_kind="devin",
        issue_number=42,
        repo_key="",
        pid=None,
        started_at="2026-07-11T11:55:00+00:00",
        process_start_time=None,
        log_path=str(log_path),
        worktree_path="C:/repo/.var/worktrees/issue-42",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    assert read_post_mortem(sessions_dir, worker.issue_number) is None


# ---------------------------------------------------------------------------
# merge_attempt_snapshot
# ---------------------------------------------------------------------------


def test_merge_attempt_snapshot_folds_ref_into_existing_sidecar(tmp_path: Path) -> None:
    """merge_attempt_snapshot must attach the ref name/ahead-count to an
    existing post-mortem sidecar without corrupting message_nodes (a prior
    draft used asdict()+reconstruct, which silently turned MessageNode
    instances into plain dicts - see attempt_refs merge bug fix)."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[("tool", json.dumps({"tool": "bash"}), "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    classify_and_record(sessions_dir, config, worker, now=_NOW)
    before = read_post_mortem(sessions_dir, worker.issue_number)
    assert before is not None
    assert before.attempts == ()

    snapshot = AttemptSnapshot(
        ref_name="refs/charlie/attempts/issue-42/attempt-1",
        old_tip="deadbeef" * 5,
        ahead_of_main_count=3,
    )
    merge_attempt_snapshot(sessions_dir, worker.issue_number, snapshot, now=_NOW)

    after = read_post_mortem(sessions_dir, worker.issue_number)
    assert after is not None
    assert len(after.attempts) == 1
    assert after.attempts[0].ref == "refs/charlie/attempts/issue-42/attempt-1"
    assert after.attempts[0].ahead_of_main == 3
    # message_nodes must survive untouched, still typed as MessageNode.
    assert after.message_nodes == before.message_nodes
    assert all(isinstance(n, MessageNode) for n in after.message_nodes)


def test_merge_attempt_snapshot_appends_without_overwriting_prior_attempts(
    tmp_path: Path,
) -> None:
    """A sidecar can outlive more than one redispatch attempt before it is
    next read/rotated — a second merge_attempt_snapshot call must append a
    second AttemptAttachment, not overwrite the first one (issue #261 F4:
    the prior implementation used dataclasses.replace on singular
    attempt_ref/attempt_ahead_of_main fields, silently losing the first
    attempt's ref)."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(db_path, nodes=[])
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"
    classify_and_record(sessions_dir, config, worker, now=_NOW)

    first = AttemptSnapshot(
        ref_name="refs/charlie/attempts/issue-42/attempt-1",
        old_tip="a" * 40,
        ahead_of_main_count=1,
    )
    second = AttemptSnapshot(
        ref_name="refs/charlie/attempts/issue-42/attempt-2",
        old_tip="b" * 40,
        ahead_of_main_count=2,
    )
    merge_attempt_snapshot(sessions_dir, worker.issue_number, first, now=_NOW)
    merge_attempt_snapshot(sessions_dir, worker.issue_number, second, now=_NOW)

    after = read_post_mortem(sessions_dir, worker.issue_number)
    assert after is not None
    assert [a.ref for a in after.attempts] == [
        "refs/charlie/attempts/issue-42/attempt-1",
        "refs/charlie/attempts/issue-42/attempt-2",
    ]
    assert [a.ahead_of_main for a in after.attempts] == [1, 2]


def test_merge_attempt_snapshot_noop_when_no_sidecar_exists(tmp_path: Path) -> None:
    """No existing post-mortem sidecar -> nothing to attach to -> no-op,
    never creates a sidecar out of thin air."""
    sessions_dir = tmp_path / "sessions"
    snapshot = AttemptSnapshot(
        ref_name="refs/charlie/attempts/issue-1/attempt-1",
        old_tip="abc123",
        ahead_of_main_count=1,
    )

    merge_attempt_snapshot(sessions_dir, 1, snapshot)

    assert read_post_mortem(sessions_dir, 1) is None


def test_merge_attempt_snapshot_noop_when_ref_name_is_none(tmp_path: Path) -> None:
    """A snapshot with no ref (nothing was preserved) must not touch an
    existing sidecar."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(db_path, nodes=[])
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"
    classify_and_record(sessions_dir, config, worker, now=_NOW)
    before = read_post_mortem(sessions_dir, worker.issue_number)
    assert before is not None

    snapshot = AttemptSnapshot(ref_name=None, old_tip=None, ahead_of_main_count=None)
    merge_attempt_snapshot(sessions_dir, worker.issue_number, snapshot)

    after = read_post_mortem(sessions_dir, worker.issue_number)
    assert after == before


# ---------------------------------------------------------------------------
# read_post_mortem: corrupt sidecar never raises
# ---------------------------------------------------------------------------


def test_read_post_mortem_corrupt_json_returns_none(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "issue-5.post-mortem.json").write_text("{not json", encoding="utf-8")

    assert read_post_mortem(sessions_dir, 5) is None


def test_read_post_mortem_absent_returns_none(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    assert read_post_mortem(sessions_dir, 5) is None


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


# ---------------------------------------------------------------------------
# Session-window matching precision (issue #261 F1+F2): defensive created_at
# parsing (epoch int, naive string) and a widened fallback window when
# worker.started_at itself is unparseable.
# ---------------------------------------------------------------------------


def test_find_matching_session_accepts_epoch_int_created_at(tmp_path: Path) -> None:
    """An epoch-integer created_at (e.g. 1783760135) is the REAL production
    shape — sessions.created_at is declared INTEGER in the live schema.
    Must match."""
    db_path = tmp_path / "sessions.db"
    created_at_dt = datetime(2026, 7, 11, 11, 56, 0, tzinfo=UTC)
    _build_sessions_db(db_path, created_at=int(created_at_dt.timestamp()))
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None  # no message_nodes -> nothing to classify as blocked
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.session_id == "sess-1"


def test_find_matching_session_accepts_naive_iso_string_created_at(tmp_path: Path) -> None:
    """A naive ISO-8601 created_at (no tz offset) — the drift shape, since
    the live schema declares the column INTEGER but SQLite affinity stores
    a writer-inserted string as TEXT — must be treated as UTC and matched
    correctly against the tz-aware match window from worker.started_at."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        created_at="2026-07-11T11:56:00",  # naive, no tz suffix
        nodes=[],
    )
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.window_start_fallback is None  # started_at parsed fine


def test_find_matching_session_ignores_out_of_window_row_even_if_lexicographically_greater(
    tmp_path: Path,
) -> None:
    """Regression guard for the SQL-side lexicographic-string-comparison bug:
    a naive created_at string that would have sorted "greater than" the
    isoformat()'d window bound as plain text, but is chronologically outside
    the window once actually parsed, must NOT match."""
    db_path = tmp_path / "sessions.db"
    # worker started_at=11:55:00Z, margin=120s -> window [11:53:00, 12:02:00]Z.
    # A calendar day later sorts "greater" lexicographically but is far
    # outside the real time window once parsed.
    _build_sessions_db(db_path, created_at="2026-07-12T11:56:00", nodes=[])
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False


def test_classify_and_record_unparseable_started_at_widens_fallback_window(
    tmp_path: Path,
) -> None:
    """When worker.started_at itself fails to parse, the match window must
    widen to the config lookback from "now" (not a narrow now-minus-margin
    window) — a session that started well before "now" (e.g. 2h earlier,
    outside the old ~240s window but inside the 6h default lookback) must
    still be found, and window_start_fallback must record that this
    fallback fired."""
    db_path = tmp_path / "sessions.db"
    session_created_at = _NOW.replace(hour=10)  # 2 hours before _NOW (12:00)
    _build_sessions_db(
        db_path,
        created_at=session_created_at.strftime("%Y-%m-%dT%H:%M:%S"),
        nodes=[],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(started_at="not-a-timestamp")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.window_start_fallback == "unparseable_started_at"


def test_classify_and_record_unparseable_started_at_no_match_still_records_fallback(
    tmp_path: Path,
) -> None:
    """Even when nothing matches within the widened fallback window, the
    fact that the fallback fired must still be recorded — otherwise a false
    non-match caused by an unparseable started_at is indistinguishable from
    "genuinely no session ran," which is exactly the diagnosability gap F1/F2
    close."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-999-different",
        nodes=[],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(started_at="not-a-timestamp")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False
    assert record.window_start_fallback == "unparseable_started_at"


def test_real_activity_for_worker_sessions_db_source(tmp_path: Path) -> None:
    """A matched sessions.db with message_nodes is the freshest real activity
    source. Node created_at values are epoch integers here — the real
    production shape (message_nodes.created_at is declared INTEGER)."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[
            ("user", "hello", int(datetime(2026, 7, 11, 11, 55, 0, tzinfo=UTC).timestamp())),
            (
                "assistant",
                "working",
                int(datetime(2026, 7, 11, 11, 59, 0, tzinfo=UTC).timestamp()),
            ),
        ],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(pid=12345)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        worker.worktree_path,
        worker.started_at,
        worker.pid,
        now,
    )

    assert probe.latest_source == "sessions.db"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 59, 0, tzinfo=UTC)
    # Per-PID log directory does not exist, so it should be recorded as missing.
    per_pid = next(s for s in probe.sources if s.name == "devin_per_pid_log")
    assert per_pid.error is not None
    assert per_pid.timestamp is None


def test_real_activity_for_worker_per_pid_log_source(tmp_path: Path) -> None:
    """Per-PID Devin log mtime is picked when sessions.db is missing."""
    db_path = tmp_path / "sessions.db"
    # sessions.db is explicitly non-existent so the probe must fall back cleanly.
    config = _config_with_db(db_path)
    worker = _make_worker(pid=99999)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    log_path = logs_dir / f"devin_20260711_114500_{worker.pid}.log"
    log_path.write_text("some devin log\n")
    mtime = datetime(2026, 7, 11, 11, 58, 0, tzinfo=UTC).timestamp()
    log_path.touch()
    # Set explicit mtime after touch
    import os

    os.utime(log_path, (mtime, mtime))

    probe = real_activity_for_worker(
        config.post_mortem,
        worker.worktree_path,
        worker.started_at,
        worker.pid,
        now,
    )

    assert probe.latest_source == "devin_per_pid_log"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 58, 0, tzinfo=UTC)
    sessions_db = next(s for s in probe.sources if s.name == "sessions.db")
    assert sessions_db.error is not None
    assert sessions_db.timestamp is None


def test_real_activity_for_worker_prefers_latest_source(tmp_path: Path) -> None:
    """latest_timestamp and latest_source come from the freshest of both sources."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[
            ("assistant", "plan", "2026-07-11T11:57:00"),
        ],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(pid=11111)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    log_path = logs_dir / f"devin_20260711_115000_{worker.pid}.log"
    log_path.write_text("log\n")
    mtime = datetime(2026, 7, 11, 11, 59, 30, tzinfo=UTC).timestamp()
    import os

    os.utime(log_path, (mtime, mtime))

    probe = real_activity_for_worker(
        config.post_mortem,
        worker.worktree_path,
        worker.started_at,
        worker.pid,
        now,
    )

    assert probe.latest_source == "devin_per_pid_log"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 59, 30, tzinfo=UTC)


def test_real_activity_for_worker_worktree_files_mtime_source(tmp_path: Path) -> None:
    """Issue #353: worktree file mtimes are a fourth real-activity source."""
    db_path = tmp_path / "sessions.db"
    config = _config_with_db(db_path)
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    source_file = worktree_path / "src" / "foo.py"
    source_file.parent.mkdir()
    source_file.write_text("# hello", encoding="utf-8")

    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    mtime = (now - timedelta(minutes=15)).timestamp()
    os.utime(source_file, (mtime, mtime))

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
        worktree_mtime_max_depth=4,
    )
    probe = real_activity_for_worker(
        config.post_mortem,
        str(worktree_path),
        "2026-07-11T11:30:00+00:00",
        None,
        now,
        watchdog_config=watchdog,
    )

    source = next(s for s in probe.sources if s.name == "worktree_files_mtime")
    assert source.error is None
    assert source.timestamp == datetime(2026, 7, 11, 11, 45, 0, tzinfo=UTC)
    assert source.staleness_seconds == 15 * 60
    assert source.threshold_minutes == 45
    assert probe.latest_source == "worktree_files_mtime"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 45, 0, tzinfo=UTC)
    # The per-source threshold lets a 15-minute-old worktree write veto a 20-minute stall window.
    assert probe.is_fresh(20) is True


def test_real_activity_for_worker_worktree_files_mtime_stale(tmp_path: Path) -> None:
    """Issue #353: worktree mtime older than its threshold is not fresh."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / "file.txt").write_text("x", encoding="utf-8")

    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    mtime = (now - timedelta(minutes=60)).timestamp()
    os.utime(worktree_path / "file.txt", (mtime, mtime))

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
    )
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        "2026-07-11T10:00:00+00:00",
        None,
        now,
        watchdog_config=watchdog,
    )

    source = next(s for s in probe.sources if s.name == "worktree_files_mtime")
    assert source.timestamp == datetime(2026, 7, 11, 11, 0, 0, tzinfo=UTC)
    assert source.threshold_minutes == 45
    assert probe.is_fresh(20) is False


def test_real_activity_for_worker_worktree_files_mtime_checkout_noise_ignored(
    tmp_path: Path,
) -> None:
    """Issue #353: checkout-time mtimes are not treated as post-start activity.

    A freshly-checked-out worktree whose files all date to session start and
    have not been written to since must not veto a stall verdict.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    source_file = worktree_path / "foo.py"
    source_file.write_text("# hello", encoding="utf-8")

    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    started_at = datetime(2026, 7, 11, 11, 30, 0, tzinfo=UTC)
    checkout_mtime = started_at.timestamp()
    os.utime(source_file, (checkout_mtime, checkout_mtime))

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
    )
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        started_at.isoformat(),
        None,
        now,
        watchdog_config=watchdog,
    )

    source = next(s for s in probe.sources if s.name == "worktree_files_mtime")
    assert source.error is None
    assert source.timestamp == started_at
    assert source.threshold_minutes == 0
    assert source.staleness_seconds == (now - started_at).total_seconds()
    assert probe.is_fresh(20) is False
    assert probe.is_fresh(5) is False


def test_real_activity_for_worker_worktree_files_mtime_depth_and_exclude(
    tmp_path: Path,
) -> None:
    """Issue #353: worktree mtime scan respects max_depth and excluded directories."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / "a.txt").write_text("a", encoding="utf-8")
    deep_dir = worktree_path / "deep" / "nested"
    deep_dir.mkdir(parents=True)
    (deep_dir / "b.txt").write_text("b", encoding="utf-8")
    git_dir = worktree_path / ".git"
    git_dir.mkdir()
    (git_dir / "c.txt").write_text("c", encoding="utf-8")

    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    os.utime(worktree_path / "a.txt", ((now - timedelta(minutes=10)).timestamp(),) * 2)
    os.utime(deep_dir / "b.txt", ((now - timedelta(minutes=5)).timestamp(),) * 2)
    os.utime(git_dir / "c.txt", ((now - timedelta(minutes=1)).timestamp(),) * 2)

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
        worktree_mtime_max_depth=1,
        worktree_mtime_exclude_dirs=(".git", ".venv"),
    )
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        "2026-07-11T11:30:00+00:00",
        None,
        now,
        watchdog_config=watchdog,
    )

    source = next(s for s in probe.sources if s.name == "worktree_files_mtime")
    # max_depth=1 includes the root and one level below it; the file at deep/nested is too deep.
    # .git is excluded, so its 1-minute-old file must not dominate the result.
    assert source.timestamp == datetime(2026, 7, 11, 11, 50, 0, tzinfo=UTC)


def test_real_activity_for_worker_worktree_files_mtime_disabled(tmp_path: Path) -> None:
    """Issue #353: worktree mtime source can be disabled."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / "file.txt").write_text("x", encoding="utf-8")

    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    watchdog = WatchdogConfig(worktree_mtime_enabled=False)
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        "",
        None,
        now,
        watchdog_config=watchdog,
    )

    assert not any(s.name == "worktree_files_mtime" for s in probe.sources)


def test_real_activity_for_worker_worktree_files_mtime_missing_path(tmp_path: Path) -> None:
    """Issue #353: a missing worktree is recorded as an errored source, not a crash."""
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    watchdog = WatchdogConfig(worktree_mtime_enabled=True)
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(tmp_path / "does-not-exist"),
        "",
        None,
        now,
        watchdog_config=watchdog,
    )

    source = next(s for s in probe.sources if s.name == "worktree_files_mtime")
    assert source.error is not None
    assert source.timestamp is None


def test_real_activity_probe_is_fresh_uses_source_threshold() -> None:
    """Issue #353: RealActivityProbe.is_fresh honors per-source thresholds."""
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    stale_for_short_window = now - timedelta(minutes=30)
    worktree_source = ActivitySource(
        name="worktree_files_mtime",
        timestamp=stale_for_short_window,
        staleness_seconds=30 * 60,
        error=None,
        threshold_minutes=45,
    )
    probe = RealActivityProbe(sources=(worktree_source,))
    assert probe.is_fresh(20) is True
    assert probe.is_fresh(60) is True

    generic_source = ActivitySource(
        name="sessions.db",
        timestamp=stale_for_short_window,
        staleness_seconds=30 * 60,
        error=None,
    )
    probe_generic = RealActivityProbe(sources=(generic_source,))
    assert probe_generic.is_fresh(20) is False
    assert probe_generic.is_fresh(60) is True


def test_real_activity_probe_to_payload() -> None:
    """to_payload serializes datetimes into JSON-safe strings and lists sources."""
    ts = datetime(2026, 7, 11, 11, 59, 0, tzinfo=UTC)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(name="sessions.db", timestamp=ts, staleness_seconds=60.0, error=None),
            ActivitySource(
                name="devin_per_pid_log", timestamp=None, staleness_seconds=None, error="no pid"
            ),
        )
    )
    payload = probe.to_payload()
    assert payload["latest_timestamp"] == "2026-07-11T11:59:00+00:00"
    assert payload["latest_source"] == "sessions.db"
    assert payload["sources"][0]["timestamp"] == "2026-07-11T11:59:00+00:00"
    assert payload["sources"][0]["staleness_seconds"] == 60.0
    assert payload["sources"][1]["error"] == "no pid"


# ---------------------------------------------------------------------------
# working_directory normalization (issue #281)
# ---------------------------------------------------------------------------


def test_classify_and_record_matches_worktree_with_forward_slash_separator(
    tmp_path: Path,
) -> None:
    """A sessions.db row using forward slashes must match a worker whose
    worktree_path uses native Windows backslashes."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[("tool", "Tool blocked: push rejected", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.session_id == "sess-1"


def test_classify_and_record_matches_worktree_with_different_case(tmp_path: Path) -> None:
    """A sessions.db row whose working_directory differs only by case must
    match the worker's worktree_path."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="c:/repo/.var/worktrees/issue-42",
        nodes=[("tool", "Tool blocked: push rejected", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.session_id == "sess-1"


def test_classify_and_record_matches_worktree_with_trailing_separator(
    tmp_path: Path,
) -> None:
    """A sessions.db row whose working_directory has a trailing separator must
    still match the worker's worktree_path."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42/",
        nodes=[("tool", "Tool blocked: push rejected", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.session_id == "sess-1"


def test_classify_and_record_matches_worktree_with_msys_style_leading_slash(
    tmp_path: Path,
) -> None:
    r"""A sessions.db row using a POSIX/MSYS-style /c/... path must match a
    worker whose worktree_path is a native Windows C:\... path."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="/c/repo/.var/worktrees/issue-42",
        nodes=[("tool", "Tool blocked: push rejected", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.session_id == "sess-1"


def test_classify_and_record_normalized_match_outside_window_still_within_window_error(
    tmp_path: Path,
) -> None:
    """If a normalized working_directory matches a row but its created_at is
    outside the match window, the existing "within the time window" error path
    still fires exactly as today."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="c:/repo/.var/worktrees/issue-42",
        created_at="2026-07-12T11:56:00",
        nodes=[],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False
    assert record.extraction_error is not None
    assert "within the time window" in record.extraction_error


def test_classify_and_record_no_normalized_match_surfaces_distinct_directories(
    tmp_path: Path,
) -> None:
    """When exact and normalized working_directory matching both fail, the
    extraction_error must include a sample of distinct working_directory values
    actually present in sessions.db for diagnostics."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-999-different",
        nodes=[],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False
    assert record.extraction_error is not None
    assert "sample distinct working_directory values" in record.extraction_error
    assert "C:/repo/.var/worktrees/issue-999-different" in record.extraction_error


def test_classify_and_record_multiple_normalized_sessions_returns_latest(
    tmp_path: Path,
) -> None:
    """A reused worktree with multiple prior dead sessions in the same
    normalized working directory must return the most recent in-window session's
    transcript, just as it does once the row is found."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="c:/repo/.var/worktrees/issue-42/",
        created_at="2026-07-11T11:54:00",
        nodes=[("tool", json.dumps({"tool": "bash"}), "2026-07-11T11:54:00")],
    )
    _insert_session_row(
        db_path,
        session_id="sess-2",
        working_directory="C:/repo/.var/worktrees/issue-42",
        created_at="2026-07-11T11:56:00",
        nodes=[("tool", "Tool blocked: push rejected", "2026-07-11T11:56:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.session_id == "sess-2"
    assert record.failure_kind == "worker_blocked"


def test_real_activity_for_worker_matches_normalized_worktree_path(
    tmp_path: Path,
) -> None:
    """real_activity_for_worker must also match a sessions.db row whose
    working_directory is a differently-formatted path for the same logical
    worktree."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="/c/repo/.var/worktrees/issue-42",
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(pid=12345)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        r"C:\repo\.var\worktrees\issue-42",
        worker.started_at,
        worker.pid,
        now,
    )

    assert probe.latest_source == "sessions.db"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 57, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Issue #343: working_directory suffix-match fallback for real fleet shapes
# ---------------------------------------------------------------------------


def test_real_activity_for_worker_matches_real_fleet_working_directory_shape(
    tmp_path: Path,
) -> None:
    """sessions.db corroboration must match a fleet worker session even when
    the Devin CLI recorded a ``working_directory`` sharing no absolute-prefix
    relationship at all with the worktree path this process computed.

    Fixture shapes are taken verbatim from a live production post-mortem
    (issue #343, 2026-07-13, ``issue-203.post-mortem.json``): every sampled
    distinct ``working_directory`` in the real sessions.db was rooted under
    an unrelated ``AppData\\Local\\Temp\\...`` tree, never under the fleet
    worktree's own ``repos\\charlie-work\\.var\\charlie-work\\worktrees\\...``
    root -- even though the trailing worktrees-dir/issue-slug segments that
    identify the dispatch are identical. Before the suffix-match fallback,
    neither the exact nor the normalized full-path comparison in
    ``_find_matching_session`` could ever match this shape, so sessions.db
    corroboration was permanently inconclusive for real fleet dispatches.

    MUTATION GATE: reverting ``_find_matching_session``'s suffix-match tier
    (or ``_working_directory_suffix``) to fall straight from the normalized
    comparison to "no session found" makes this test fail -- the fixture row's
    working_directory has a different drive letter AND a completely different
    directory tree above the shared ``worktrees/<issue-slug>`` tail, so
    neither the exact-match nor the ``_normalize_working_directory`` tier can
    find it.
    """
    db_path = tmp_path / "sessions.db"
    real_fleet_worktree_path = (
        r"C:\Users\senki\repos\charlie-work\.var\charlie-work\worktrees"
        r"\agent-issue-203-redundant-re-dispatch"
    )
    # Recorded working_directory shares zero prefix with the worktree path
    # above -- only the trailing (worktrees-dir, issue-slug) segment pair
    # matches, exactly like the production sample values.
    recorded_working_directory = (
        r"C:\Users\senki\AppData\Local\Temp\claude\some-other-session-root"
        r"\worktrees\agent-issue-203-redundant-re-dispatch"
    )
    _build_sessions_db(
        db_path,
        working_directory=recorded_working_directory,
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        real_fleet_worktree_path,
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
    )

    assert probe.latest_source == "sessions.db"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 57, 0, tzinfo=UTC)


def test_find_matching_session_suffix_fallback_rejects_different_issue_slug(
    tmp_path: Path,
) -> None:
    """The suffix-match fallback must not collapse two different issues'
    worktrees just because they share the same worktrees-dir parent segment.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory=(
            r"C:\Users\senki\AppData\Local\Temp\other-root"
            r"\worktrees\agent-issue-999-unrelated"
        ),
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        r"C:\Users\senki\repos\charlie-work\.var\charlie-work\worktrees\agent-issue-203-fix",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
    )

    assert probe.latest_timestamp is None
    db_source = next(s for s in probe.sources if s.name == "sessions.db")
    assert db_source.error is not None


# ---------------------------------------------------------------------------
# Issue #639: worker_kind skips Devin sources for non-Devin workers
# ---------------------------------------------------------------------------


def test_real_activity_for_worker_skips_devin_sources_for_non_devin_kind(
    tmp_path: Path,
) -> None:
    """Issue #639: when ``worker_kind`` is a non-Devin adapter (claude-code,
    api, manual, command), the Devin-specific sources (sessions.db and
    per-PID Devin log) are skipped entirely — a non-Devin worker has no
    Devin subject to look up. The probe must not contain those sources at
    all, so no permanent "no session found" / "no pid" errors are produced.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        "C:/repo/.var/worktrees/issue-42",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
        worker_kind="claude-code",
    )

    # No Devin sources at all — they were skipped.
    source_names = {s.name for s in probe.sources}
    assert "sessions.db" not in source_names
    assert "devin_per_pid_log" not in source_names
    # No errored sources (the Devin sources that would have errored are absent).
    assert all(s.error is None for s in probe.sources)


def test_real_activity_for_worker_skips_devin_sources_for_api_kind(
    tmp_path: Path,
) -> None:
    """Issue #639: ``api``-routed workers (which delegate to claude-code) also
    have no Devin subject. The Devin sources must be skipped for
    ``worker_kind="api"`` just as for ``"claude-code"``.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        "C:/repo/.var/worktrees/issue-42",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
        worker_kind="api",
    )

    source_names = {s.name for s in probe.sources}
    assert "sessions.db" not in source_names
    assert "devin_per_pid_log" not in source_names


def test_real_activity_for_worker_consults_devin_sources_for_devin_shell(
    tmp_path: Path,
) -> None:
    """Issue #639 regression guard: ``worker_kind="devin-shell"`` must still
    consult the Devin sources. The skip only applies to non-Devin kinds.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        "C:/repo/.var/worktrees/issue-42",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
        worker_kind="devin-shell",
    )

    source_names = {s.name for s in probe.sources}
    assert "sessions.db" in source_names
    assert "devin_per_pid_log" in source_names
    # sessions.db matched and has a fresh timestamp.
    assert probe.latest_source == "sessions.db"


def test_real_activity_for_worker_consults_devin_sources_for_devin_view_kind(
    tmp_path: Path,
) -> None:
    """Issue #639: the ``WorkerView.adapter_kind`` convention uses
    ``"devin"`` (not ``"devin-shell"``). Both must be recognized as Devin
    kinds so the watchdog's ``real_activity_probe_for`` wrapper — which passes
    ``view.adapter_kind`` — does not accidentally skip Devin sources for
    Devin-shell workers.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        "C:/repo/.var/worktrees/issue-42",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
        worker_kind="devin",
    )

    source_names = {s.name for s in probe.sources}
    assert "sessions.db" in source_names
    assert "devin_per_pid_log" in source_names


def test_real_activity_for_worker_consults_all_sources_when_kind_unknown(
    tmp_path: Path,
) -> None:
    """Issue #639: ``worker_kind=None`` (unknown) preserves the pre-#639
    behavior — all sources are consulted. This is the backward-compatibility
    path for callers that have not been updated to pass ``worker_kind``.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        "C:/repo/.var/worktrees/issue-42",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
    )

    source_names = {s.name for s in probe.sources}
    assert "sessions.db" in source_names
    assert "devin_per_pid_log" in source_names


def test_find_matching_session_suffix_fallback_requires_parent_segment_match(
    tmp_path: Path,
) -> None:
    """The suffix-match fallback must compare the segment above the
    issue-slug leaf too, not just the leaf segment alone (issue #343
    Finding 3).

    ``test_find_matching_session_suffix_fallback_rejects_different_issue_slug``
    above is satisfied by the two working_directory values having different
    trailing slug segments, so it passes at
    ``_WORKING_DIRECTORY_SUFFIX_SEGMENTS`` == 1 or == 2 alike -- it does not
    by itself pin the segment count. This test constructs a DB row whose
    working_directory shares the exact same trailing slug segment as the
    target worktree but sits under an unrelated parent directory (not the
    fleet worktrees-dir), so a 1-segment suffix would wrongly match it while
    the real 2-segment suffix correctly rejects it.

    MUTATION GATE: shrinking ``_WORKING_DIRECTORY_SUFFIX_SEGMENTS`` from 2 to
    1 makes this test fail -- the unrelated row would wrongly match on the
    shared leaf segment alone.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory=(
            r"C:\Users\senki\AppData\Local\Temp\some-unrelated-tool-cache"
            r"\agent-issue-203-fix"
        ),
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        r"C:\Users\senki\repos\charlie-work\.var\charlie-work\worktrees\agent-issue-203-fix",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
    )

    assert probe.latest_timestamp is None
    db_source = next(s for s in probe.sources if s.name == "sessions.db")
    assert db_source.error is not None
