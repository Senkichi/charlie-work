"""``state.json`` round-trips and cache-tolerant reads (issue #3117)."""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.foreman import paths
from charlie_work.foreman.state import (
    AuthoredIssue,
    ForemanState,
    load_state,
    save_state,
    with_updates,
)


def _foreman(tmp_path: Path) -> Path:
    fdir = paths.foreman_dir(tmp_path, "g1")
    fdir.mkdir(parents=True)
    return fdir


def test_missing_state_reads_as_paused_defaults(tmp_path: Path) -> None:
    state = load_state(_foreman(tmp_path))
    assert state == ForemanState()
    assert state.state == "paused"


def test_corrupt_state_reads_as_defaults(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    (fdir / "state.json").write_text("{not json", encoding="utf-8")
    assert load_state(fdir) == ForemanState()


def test_state_round_trip(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    original = ForemanState(
        state="active",
        event_cursor=42,
        lifecycle_fingerprint="abc123",
        turns_total=9,
        turns_today=2,
        turns_today_date="2026-11-25",
        consecutive_failures=1,
        last_turn_at="2026-11-25T10:00:00Z",
        last_scheduled_turn_at="2026-11-25T09:00:00Z",
        issues={3090: AuthoredIssue(number=3090, body_hash="h1")},
        version=1,
    )
    saved = save_state(fdir, original)
    assert saved.ok, saved.error
    assert load_state(fdir) == original


def test_unknown_state_value_falls_back_to_paused(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    (fdir / "state.json").write_text(
        json.dumps({"state": "transcended", "turns_total": 3}), encoding="utf-8"
    )
    state = load_state(fdir)
    assert state.state == "paused"
    assert state.turns_total == 3  # salvageable fields still load


def test_save_rejects_unknown_state(tmp_path: Path) -> None:
    result = save_state(_foreman(tmp_path), ForemanState(state="transcended"))
    assert not result.ok
    assert "transcended" in result.error


def test_with_updates_returns_new_value(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    original = load_state(fdir)
    updated = with_updates(original, state="active", turns_total=1)
    assert original.state == "paused"  # never mutated in place
    assert updated.state == "active"
    save_state(fdir, updated)
    assert load_state(fdir).state == "active"


def test_issues_round_trip_and_salvage(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    (fdir / "state.json").write_text(
        json.dumps(
            {
                "state": "active",
                "issues": {
                    "3090": {"body_hash": "h1"},
                    "notanumber": {"body_hash": "h2"},
                    "3091": 42,
                },
            }
        ),
        encoding="utf-8",
    )
    state = load_state(fdir)
    assert state.issues == {3090: AuthoredIssue(number=3090, body_hash="h1")}
