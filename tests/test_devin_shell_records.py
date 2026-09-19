"""Session-record sidecar tests for the devin-shell adapter.

Split out of ``tests/test_devin_shell.py`` (issue #1542, Track-1 pilot):
``read_session_records`` round-trips and foreign-sidecar skipping,
``_sidecar_path``/``_write_json`` storage details, and the ``SessionRecord``
log-stat field round-trip/persist surface (issue #160).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from _devin_shell_fixtures import (
    _FAKE_DEVIN_SLEEP,
    _install_fake_create_worktree,
    _write_fake_devin,
)

from charlie_work.devin_shell import (
    SessionRecord,
    launch_devin_session,
    read_session_records,
    _sidecar_path,
    _write_json,
)


def test_read_session_records_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_SLEEP)
    prompt_a = tmp_path / "a.md"
    prompt_b = tmp_path / "b.md"
    prompt_a.write_text("a", encoding="utf-8")
    prompt_b.write_text("b", encoding="utf-8")

    _install_fake_create_worktree(monkeypatch, tmp_path)

    launch_devin_session(
        1,
        "agent/issue-1",
        prompt_a,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script), "{prompt_path}"),
    )
    launch_devin_session(
        2,
        "agent/issue-2",
        prompt_b,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script), "{prompt_path}"),
    )

    records = read_session_records(sessions_dir)

    assert len(records) == 2
    by_issue = {record.issue_number: record for record in records}
    assert set(by_issue) == {1, 2}
    assert by_issue[1].branch == "agent/issue-1"
    assert by_issue[2].branch == "agent/issue-2"
    assert all(isinstance(record, SessionRecord) for record in records)


def test_read_session_records_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert read_session_records(tmp_path / "does-not-exist") == []


def test_read_session_records_skips_malformed_sidecar(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "issue-99.json").write_text("{not valid json", encoding="utf-8")

    assert read_session_records(sessions_dir) == []


def test_read_session_records_skips_claude_code_sidecars(tmp_path: Path) -> None:
    # Both adapters share one sessions_dir. The devin glob `issue-*.json` also
    # matches the claude-code adapter's `issue-N.claude.json` sidecars, so
    # read_session_records must skip them (otherwise doctor double-counts every
    # Claude worker and tries to parse a foreign schema).
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    devin_payload = {
        "issue_number": 5,
        "branch": "agent/issue-5",
        "worktree_path": "/tmp/wt/issue-5",
        "prompt_path": "p.md",
        "command": ["devin", "--print"],
        "pid": 1234,
        "started_at": "2026-01-01T00:00:00Z",
        "log_path": "issue-5.log",
        "error": None,
    }
    (sessions_dir / "issue-5.json").write_text(json.dumps(devin_payload), encoding="utf-8")
    # A claude-code sidecar with a deliberately foreign schema: if the exclusion
    # regressed, from_dict would choke on the missing devin-shaped keys.
    (sessions_dir / "issue-6.claude.json").write_text(
        json.dumps({"issue_number": 6, "worktree": "wt", "pid": 5678}), encoding="utf-8"
    )

    records = read_session_records(sessions_dir)

    assert [record.issue_number for record in records] == [5]


def test_read_session_records_skips_api_sidecars(tmp_path: Path) -> None:
    """The new ``issue-N.api.json`` sidecar suffix is also ignored by the devin
    reader by construction of the stem regex, but guard it explicitly."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    devin_payload = {
        "issue_number": 7,
        "branch": "agent/issue-7",
        "worktree_path": "/tmp/wt/issue-7",
        "prompt_path": "p.md",
        "command": ["devin", "--print"],
        "pid": 7777,
        "started_at": "2026-01-01T00:00:00Z",
        "log_path": "issue-7.log",
        "error": None,
    }
    (sessions_dir / "issue-7.json").write_text(json.dumps(devin_payload), encoding="utf-8")
    # A deliberately foreign api sidecar: if the exclusion regressed, the stem
    # (``issue-7.api``) would not match ``^issue-\d+$`` and it would be skipped.
    (sessions_dir / "issue-8.api.json").write_text(
        json.dumps({"issue_number": 8, "worktree": "wt", "pid": 8888}), encoding="utf-8"
    )

    records = read_session_records(sessions_dir)

    assert [record.issue_number for record in records] == [7]


def test_read_session_records_skips_post_mortem_sidecars(tmp_path: Path) -> None:
    """Issue #343 Finding 1: the devin glob `issue-*.json` also matches
    post_mortem's `issue-N.post-mortem.json` sidecars (both live in the same
    sessions_dir). SessionRecord.from_dict only strictly requires
    `issue_number` (present in a PostMortemRecord payload too), so an
    unfiltered post-mortem file parses into a bogus SessionRecord(pid=None,
    log_path=""). That phantom bypasses `if w.pid is not None:`
    corroboration downstream in the dead-session reaper and reaches
    `reap_sidecar`, which resolves to the SAME path as the real
    `issue-N.json` sidecar and deletes it -- silently reaping a worker whose
    liveness was never actually re-verified.

    read_session_records must skip `issue-N.post-mortem.json` (and, for
    completeness, the pre-existing `issue-N.claude.json` exclusion) and
    return exactly the one real devin sidecar.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    devin_payload = {
        "issue_number": 343,
        "branch": "agent/issue-343-x",
        "worktree_path": "/tmp/wt/issue-343",
        "prompt_path": "p.md",
        "command": ["devin", "--print"],
        "pid": 4242,
        "started_at": "2026-01-01T00:00:00Z",
        "log_path": "issue-343.log",
        "error": None,
    }
    (sessions_dir / "issue-343.json").write_text(json.dumps(devin_payload), encoding="utf-8")
    (sessions_dir / "issue-343.log").write_text("Working...\n", encoding="utf-8")
    # A post-mortem sidecar with a foreign schema (issue_number is the only
    # key it shares with SessionRecord) -- if the exclusion regressed, this
    # would parse into a bogus pid=None SessionRecord.
    post_mortem_payload = {
        "issue_number": 343,
        "generated_at": "2026-01-01T00:05:00Z",
        "db_path": "C:/fake/sessions.db",
        "matched": False,
        "extraction_error": "no session found matching working_directory",
    }
    (sessions_dir / "issue-343.post-mortem.json").write_text(
        json.dumps(post_mortem_payload), encoding="utf-8"
    )
    # The claude-code adapter's sidecar, included for completeness alongside
    # the pre-existing exclusion this test also guards.
    (sessions_dir / "issue-343.claude.json").write_text(
        json.dumps({"issue_number": 343, "worktree": "wt", "pid": 9999}), encoding="utf-8"
    )

    records = read_session_records(sessions_dir)

    assert len(records) == 1
    assert records[0].issue_number == 343
    assert records[0].pid == 4242


def test_sidecar_path_returns_correct_path(tmp_path: Path) -> None:
    """_sidecar_path returns the expected path for a given issue number."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    path = _sidecar_path(sessions_dir, 123)
    assert path == sessions_dir / "issue-123.json"


def test_session_record_log_stat_fields_roundtrip(tmp_path: Path) -> None:
    """SessionRecord with last_activity_at and log_bytes fields round-trips through to_dict/from_dict."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    original = SessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/worktree-1",
        prompt_path="/tmp/prompt-1.md",
        command=("devin", "prompt.md"),
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        log_path="/tmp/issue-1.log",
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        last_activity_at="2026-07-06T01:30:45Z",
        log_bytes=2048,
    )

    payload = original.to_dict()
    assert "last_activity_at" in payload
    assert "log_bytes" in payload
    assert payload["last_activity_at"] == "2026-07-06T01:30:45Z"
    assert payload["log_bytes"] == 2048

    reconstructed = SessionRecord.from_dict(payload)
    assert reconstructed.last_activity_at == "2026-07-06T01:30:45Z"
    assert reconstructed.log_bytes == 2048


def test_session_record_from_dict_missing_log_stat_fields(tmp_path: Path) -> None:
    """A payload missing last_activity_at and log_bytes still constructs with None defaults."""
    payload = {
        "issue_number": 1,
        "branch": "agent/issue-1",
        "worktree_path": "/tmp/worktree-1",
        "prompt_path": "/tmp/prompt-1.md",
        "command": ["devin", "prompt.md"],
        "pid": 12345,
        "started_at": "2026-07-06T00:00:00Z",
        "log_path": "/tmp/issue-1.log",
        # error, failure_kind, process_start_time, reclaimed, last_activity_at, log_bytes omitted
    }

    record = SessionRecord.from_dict(payload)
    assert record.last_activity_at is None
    assert record.log_bytes is None


def test_session_record_log_stat_fields_persist_to_sidecar(tmp_path: Path) -> None:
    """SessionRecord with log stat fields can be written to and read from a sidecar file."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42",
        worktree_path="/tmp/worktree-42",
        prompt_path="/tmp/prompt-42.md",
        command=("devin", "prompt.md"),
        pid=54321,
        started_at="2026-07-06T00:00:00Z",
        log_path="/tmp/issue-42.log",
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        last_activity_at="2026-07-06T02:15:30Z",
        log_bytes=4096,
    )

    sidecar_path = _sidecar_path(sessions_dir, 42)
    _write_json(sidecar_path, record.to_dict())

    # Read back through read_session_records
    records = read_session_records(sessions_dir)
    assert len(records) == 1
    restored = records[0]
    assert restored.last_activity_at == "2026-07-06T02:15:30Z"
    assert restored.log_bytes == 4096
