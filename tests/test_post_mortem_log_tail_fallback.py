"""``classify_and_record`` log-tail fallback tests (issue #260).

Split out of ``tests/test_post_mortem.py`` (issue #1567, Track 1): when
sessions.db extraction misses entirely (unavailable / no matching session),
the worker's own log tail is the only remaining signal. "A tool was
rejected by the user" is the Devin CLI's own log/stdout surfacing of a
PreToolUse hook block -- distinct from the "Tool blocked:" prefix that
appears in sessions.db message-node content.
"""

from __future__ import annotations

import json
from pathlib import Path

from _post_mortem_fixtures import (
    _NOW,
    _build_sessions_db,
    _config_with_db,
)

from charlie_work.post_mortem import (
    classify_and_record,
    read_post_mortem,
)
from charlie_work.worker import WorkerView


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
