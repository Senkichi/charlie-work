"""Tests for post_mortem.py (issue #261): worker post-mortem extraction from
the Devin CLI's local session store, and the worker_blocked classification
that suppresses hot redispatch into a push-gate hook.

The ``message_nodes`` schema this module reads is not officially documented
(see post_mortem.py's module docstring and extraction-dossier.md item 23) —
these tests build a fixture sessions.db against the schema this module
assumes, and separately verify that any schema mismatch degrades to a
recorded ``extraction_error`` rather than raising.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


from charlie_work.attempt_refs import AttemptSnapshot
from charlie_work.config import OrchestratorConfig, PostMortemConfig, SignatureRule
from charlie_work.post_mortem import (
    MessageNode,
    classify_and_record,
    merge_attempt_snapshot,
    read_post_mortem,
)
from charlie_work.worker import WorkerView


def _build_sessions_db_epoch_created_at(
    db_path: Path,
    *,
    session_id: str = "sess-1",
    working_directory: str = "C:/repo/.var/worktrees/issue-42",
    created_at_epoch: int,
) -> None:
    """Build a fixture sessions.db whose ``created_at`` column is declared
    INTEGER (not TEXT) and holds a raw epoch value — reproduces a real
    sessions.db observed storing created_at as an epoch int (e.g.
    ``1783760135``) for some rows. SQLite's TEXT column affinity would
    coerce an inserted int to text, so this needs its own schema rather
    than reusing ``_build_sessions_db``.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, working_directory TEXT, created_at INTEGER)"
        )
        conn.execute(
            "CREATE TABLE message_nodes ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, "
            "content TEXT, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO sessions (id, working_directory, created_at) VALUES (?, ?, ?)",
            (session_id, working_directory, created_at_epoch),
        )
        conn.commit()
    finally:
        conn.close()


_NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)


def _make_worker(
    *,
    issue_number: int = 42,
    worktree_path: str = "C:/repo/.var/worktrees/issue-42",
    started_at: str = "2026-07-11T11:55:00+00:00",
    adapter_kind: str = "devin",
) -> WorkerView:
    return WorkerView(
        adapter_kind=adapter_kind,
        issue_number=issue_number,
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


def _build_sessions_db(
    db_path: Path,
    *,
    session_id: str = "sess-1",
    working_directory: str = "C:/repo/.var/worktrees/issue-42",
    created_at: str = "2026-07-11T11:56:00",
    nodes: list[tuple[str, str, str]] = (),
) -> None:
    """Build a fixture sessions.db against the schema post_mortem.py assumes:
    sessions(id, working_directory, created_at) and
    message_nodes(id, session_id, role, content, created_at).
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, working_directory TEXT, created_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE message_nodes ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, "
            "content TEXT, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO sessions (id, working_directory, created_at) VALUES (?, ?, ?)",
            (session_id, working_directory, created_at),
        )
        for role, content, node_created_at in nodes:
            conn.execute(
                "INSERT INTO message_nodes (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, node_created_at),
            )
        conn.commit()
    finally:
        conn.close()


def _config_with_db(db_path: Path, **overrides) -> OrchestratorConfig:
    pm_kwargs = {"db_path": str(db_path)}
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
    """A real sessions.db has been observed storing created_at as a raw
    epoch integer (e.g. 1783760135) rather than an ISO string — the column
    is declared TEXT but SQLite's type affinity does not coerce a
    writer-inserted INTEGER value. Must still match."""
    db_path = tmp_path / "sessions.db"
    created_at_dt = datetime(2026, 7, 11, 11, 56, 0, tzinfo=UTC)
    _build_sessions_db_epoch_created_at(db_path, created_at_epoch=int(created_at_dt.timestamp()))
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
    """A naive ISO-8601 created_at (no tz offset, as sessions.db actually
    stores it) must be treated as UTC and matched correctly against the
    tz-aware match window computed from worker.started_at."""
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
