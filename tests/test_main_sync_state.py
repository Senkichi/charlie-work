"""main_sync_state: per-repo counters survive restarts and never crash on bad files."""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.main_sync_state import RepoSyncState, load_state, save_state, state_path


def test_round_trip(tmp_path: Path) -> None:
    state = RepoSyncState(
        last_skip_key="blocked:a:b",
        failure_streak=2,
        defer_since=1000.5,
        defer_count=3,
        next_attempt_at=2000.0,
        history_alarmed=True,
        budget_skipped=True,
    )
    save_state(tmp_path, "owner/repo", state)
    assert state_path(tmp_path, "owner/repo") == tmp_path / "main-sync" / "owner__repo.json"
    assert load_state(tmp_path, "owner/repo") == state


def test_missing_file_is_fresh(tmp_path: Path) -> None:
    assert load_state(tmp_path, "o/r") == RepoSyncState()


@pytest.mark.parametrize(
    "text",
    [
        "{not json",
        "[]",
        '{"failure_streak": "two"}',
        '{"defer_since": true}',
        '{"failure_alarmed": 1}',
    ],
)
def test_corrupt_or_mistyped_file_is_fresh(tmp_path: Path, text: str) -> None:
    path = state_path(tmp_path, "o/r")
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    assert load_state(tmp_path, "o/r") == RepoSyncState()


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    path = state_path(tmp_path, "o/r")
    path.parent.mkdir(parents=True)
    path.write_text('{"failure_streak": 1, "from_the_future": 9}', encoding="utf-8")
    assert load_state(tmp_path, "o/r") == RepoSyncState(failure_streak=1)


def test_save_never_raises(tmp_path: Path) -> None:
    blocker = tmp_path / "main-sync"
    blocker.write_text("a file where the directory should be", encoding="utf-8")
    save_state(tmp_path, "o/r", RepoSyncState(failure_streak=1))  # logs, does not raise


def test_alarmed_property() -> None:
    assert not RepoSyncState().alarmed
    for flag in ("failure_alarmed", "defer_alarmed", "history_alarmed", "rollback_alarmed"):
        assert RepoSyncState(**{flag: True}).alarmed
