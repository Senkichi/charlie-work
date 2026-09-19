"""On-disk session-state tests for scripts/worker_stop_gate.py.

Split verbatim out of ``tests/test_worker_stop_gate.py`` (issue #1573,
Track 1 shoulder). Shared helpers and the ``gate``/``repo`` fixtures live
in ``tests/_worker_stop_gate_fixtures.py`` -- the ``tests/_*.py``
hoisted-fixture convention is the sanctioned import target for shared
test helpers (see ``tests/test_zero_cross_test_import_guard.py``).

Covers the state dir, bounded-retry/exhaustion counters, session-id
sanitization, atomic counter persistence, and stale-file pruning.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from _worker_stop_gate_fixtures import _load_module, gate as gate


def test_state_dir_creates_and_returns_directory_under_temp(tmp_path, monkeypatch):
    # Deliberately does not use the ``gate`` fixture: that fixture patches
    # ``_state_dir`` itself, which would defeat the point of exercising its
    # real body (tempfile.gettempdir()-based) here.
    module = _load_module()
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))

    result = module._state_dir()

    assert result == tmp_path / "worker_stop_gate"
    assert result.is_dir()


# ---------------------------------------------------------------------------
# Bounded-retry / exhaustion contract.
# ---------------------------------------------------------------------------


def test_decide_and_report_blocks_up_to_cap_then_exhausts(gate, capsys):
    # Anti-vacuity guard (review round, #1259 follow-up 1): both the loop
    # count and the expected codes below are hardcoded literals, not derived
    # from MAX_BLOCKS_PER_SESSION -- with both bounds coming from the same
    # constant under test, this assertion previously passed even with
    # MAX_BLOCKS_PER_SESSION == 0 (a gate that never actually blocks). The
    # explicit assert is a tripwire on that one specific value; the literal
    # codes list is what actually makes the rest of the test fail if the cap
    # changes out from under it.
    assert gate.MAX_BLOCKS_PER_SESSION == 3
    session_id = "sess-cap"
    failing = gate.GateResult(block=True, reason="boom")

    codes = [gate._decide_and_report(session_id, failing) for _ in range(4)]

    assert codes == [2, 2, 2, 0]
    assert gate.EXHAUSTED_MARKER in capsys.readouterr().err
    # Exhaustion ends the streak -- it must not spend a lifetime session
    # budget, or one early stumble silently disarms the gate for good.
    assert gate._read_block_count(gate._counter_path(session_id)) == 0


def test_decide_and_report_exhaustion_starts_a_fresh_streak(gate, capsys):
    session_id = "sess-cap-fresh"
    failing = gate.GateResult(block=True, reason="boom")

    for _ in range(gate.MAX_BLOCKS_PER_SESSION + 1):
        gate._decide_and_report(session_id, failing)
    capsys.readouterr()  # discard exhaustion output from the setup loop

    rc = gate._decide_and_report(session_id, failing)

    assert rc == gate.BLOCK_EXIT_CODE
    assert gate.EXHAUSTED_MARKER not in capsys.readouterr().err
    assert gate._read_block_count(gate._counter_path(session_id)) == 1


def test_decide_and_report_passing_result_never_blocks_or_increments(gate):
    session_id = "sess-pass"
    passing = gate.GateResult(block=False)

    rc = gate._decide_and_report(session_id, passing)

    assert rc == 0
    assert gate._read_block_count(gate._counter_path(session_id)) == 0


def test_decide_and_report_pass_resets_an_in_progress_streak(gate):
    session_id = "sess-reset-midstreak"
    failing = gate.GateResult(block=True, reason="boom")
    passing = gate.GateResult(block=False)

    gate._decide_and_report(session_id, failing)
    gate._decide_and_report(session_id, failing)
    assert gate._read_block_count(gate._counter_path(session_id)) == 2

    rc = gate._decide_and_report(session_id, passing)

    assert rc == 0
    assert gate._read_block_count(gate._counter_path(session_id)) == 0

    # And the next failure starts a fresh streak at count=1, not count=3.
    rc = gate._decide_and_report(session_id, failing)
    assert rc == gate.BLOCK_EXIT_CODE
    assert gate._read_block_count(gate._counter_path(session_id)) == 1


# ---------------------------------------------------------------------------
# Session-id sanitization.
# ---------------------------------------------------------------------------


def test_safe_session_id_falls_back_on_missing_or_non_string(gate):
    assert gate._safe_session_id(None) == gate._FALLBACK_SESSION_ID
    assert gate._safe_session_id("") == gate._FALLBACK_SESSION_ID
    assert gate._safe_session_id(123) == gate._FALLBACK_SESSION_ID


def test_safe_session_id_strips_path_unsafe_characters(gate):
    result = gate._safe_session_id("../../etc/passwd")

    assert "/" not in result
    assert re.fullmatch(r"[A-Za-z0-9_-]+", result)


# ---------------------------------------------------------------------------
# Counter persistence (atomic tmp+replace, per the repo's JSON-write invariant).
# ---------------------------------------------------------------------------


def test_write_block_count_is_atomic_tmp_replace(gate, tmp_path):
    target = tmp_path / "sess.count"

    gate._write_block_count(target, 2)

    assert target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"count": 2}


def test_read_block_count_defaults_to_zero_on_missing_or_corrupt(gate, tmp_path):
    missing = tmp_path / "missing.count"
    assert gate._read_block_count(missing) == 0

    corrupt = tmp_path / "corrupt.count"
    corrupt.write_text("not json", encoding="utf-8")
    assert gate._read_block_count(corrupt) == 0

    negative = tmp_path / "negative.count"
    negative.write_text(json.dumps({"count": -1}), encoding="utf-8")
    assert gate._read_block_count(negative) == 0


def test_append_invocation_log_is_best_effort_on_io_error(gate, monkeypatch, tmp_path):
    bad_dir = tmp_path / "is-a-dir"
    bad_dir.mkdir()
    monkeypatch.setattr(gate, "_log_path", lambda _session_id: bad_dir)

    gate._append_invocation_log("sess", "message")  # must not raise


# ---------------------------------------------------------------------------
# State-dir pruning (review round, #1259 follow-up 3).
# ---------------------------------------------------------------------------


def test_prune_stale_state_files_deletes_old_keeps_fresh_and_other_suffixes(gate, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    old_count = state_dir / "old.count"
    old_log = state_dir / "old.log"
    fresh_count = state_dir / "fresh.count"
    other_suffix = state_dir / "old.other"
    for path in (old_count, old_log, fresh_count, other_suffix):
        path.write_text("x", encoding="utf-8")

    stale_time = time.time() - gate.STATE_FILE_MAX_AGE_SECONDS - 3600
    os.utime(old_count, (stale_time, stale_time))
    os.utime(old_log, (stale_time, stale_time))
    os.utime(other_suffix, (stale_time, stale_time))

    gate._prune_stale_state_files(state_dir)

    assert not old_count.exists()
    assert not old_log.exists()
    assert fresh_count.exists()
    assert other_suffix.exists()  # not a .count/.log suffix -- never touched


def test_prune_stale_state_files_swallows_missing_directory(gate, tmp_path):
    missing = tmp_path / "does-not-exist"

    gate._prune_stale_state_files(missing)  # must not raise


def test_prune_stale_state_files_swallows_per_file_unlink_error(gate, tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    stubborn = state_dir / "stubborn.count"
    stubborn.write_text("x", encoding="utf-8")
    stale_time = time.time() - gate.STATE_FILE_MAX_AGE_SECONDS - 3600
    os.utime(stubborn, (stale_time, stale_time))

    def _boom(self):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", _boom)

    gate._prune_stale_state_files(state_dir)  # must not raise; failure is swallowed
