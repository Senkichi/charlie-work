"""``state.json`` round-trips and cache-tolerant reads (issue #3117)."""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.foreman import paths
from charlie_work.foreman.state import (
    AuthoredIssue,
    ForemanState,
    read_state_snapshot,
    save_state,
    with_updates,
)


def _foreman(tmp_path: Path) -> Path:
    fdir = paths.foreman_dir(tmp_path, "g1")
    fdir.mkdir(parents=True)
    return fdir


def test_missing_state_reads_as_paused_defaults(tmp_path: Path) -> None:
    state = read_state_snapshot(_foreman(tmp_path))
    assert state == ForemanState()
    assert state.state == "paused"


def test_corrupt_state_reads_as_defaults(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    state_file = fdir / "state.json"
    state_file.write_text("{not json", encoding="utf-8")
    assert read_state_snapshot(fdir) == ForemanState()
    # The read must never mutate: a corrupt cache is left byte-identical on
    # disk — no quarantine rename, no rewrite — so a racing atomic replace can
    # never be turned into the issue #310 hazard. This is the property that
    # makes read_state_snapshot the sanctioned unlocked reader (issue #3141).
    assert state_file.read_text(encoding="utf-8") == "{not json"
    assert list(fdir.iterdir()) == [state_file]


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
    assert read_state_snapshot(fdir) == original


def test_unknown_state_value_falls_back_to_paused(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    (fdir / "state.json").write_text(
        json.dumps({"state": "transcended", "turns_total": 3}), encoding="utf-8"
    )
    state = read_state_snapshot(fdir)
    assert state.state == "paused"
    assert state.turns_total == 3  # salvageable fields still load


def test_save_rejects_unknown_state(tmp_path: Path) -> None:
    result = save_state(_foreman(tmp_path), ForemanState(state="transcended"))
    assert not result.ok
    assert "transcended" in result.error


def test_save_to_missing_foreman_dir_is_an_error(tmp_path: Path) -> None:
    """``write_json_atomic`` mkdirs the parent — saving to a removed or
    never-created foreman must refuse first, not silently re-materialize a
    charterless ``foremen/<name>/`` that ``gang list`` would then show."""
    fdir = paths.foreman_dir(tmp_path, "ghost")
    result = save_state(fdir, ForemanState(state="active"))
    assert not result.ok
    assert "parent directory" in result.error
    assert not fdir.exists()
    assert not (fdir / "state.json").exists()


def test_with_updates_returns_new_value(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    original = read_state_snapshot(fdir)
    updated = with_updates(original, state="active", turns_total=1)
    assert original.state == "paused"  # never mutated in place
    assert updated.state == "active"
    save_state(fdir, updated)
    assert read_state_snapshot(fdir).state == "active"


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
    state = read_state_snapshot(fdir)
    assert state.issues == {3090: AuthoredIssue(number=3090, body_hash="h1")}
