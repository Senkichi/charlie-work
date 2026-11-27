"""The foreman pass Turn: bookkeeping, crash resume, failure accounting (issue #3122).

Split out of the original ``test_foreman_pass.py`` under the file-size
ratchet (issue #1442): this module covers everything after the predicate
holds — the Turn journal a real Turn leaves behind, idempotent crash-resume
at Apply, ``consecutive_failures -> faulted`` accounting, turn-dir
retention, and the end-to-end file-backed issue filing. Wake-predicate and
deferral coverage lives in ``test_foreman_pass.py``; shared builders in
``tests/_foreman_pass_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _foreman_collect_helpers import FOREMAN_NAME, REPO_KEY, _World
from _foreman_pass_fixtures import (
    _events,
    _patch_state,
    _quiesce,
    _run,
    _succeeding_decide,
    _turn_dirs,
    _world,
)
from _worktree_fixtures import (
    _wt_scratch as _register_wt_scratch,  # noqa: F401 -- registers the wt_scratch fixture
)

from charlie_work.foreman import items as f_items
from charlie_work.foreman import journal as f_journal
from charlie_work.foreman import memory as f_memory
from charlie_work.foreman import paths as f_paths
from charlie_work.foreman import pass_run
from charlie_work.foreman.collect import lifecycle_fingerprint
from charlie_work.foreman.state import read_state_snapshot
from charlie_work.host.fakes import FakeDecideLauncher
from charlie_work.local_issues import github_client_for
from charlie_work.github import GitHub
from charlie_work.subprocess_runner import RunResult

# ---------------------------------------------------------------------------
# Turn bookkeeping
# ---------------------------------------------------------------------------


def test_successful_turn_persists_bookkeeping(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _succeeding_decide(fake_host)

    passed = _run(world)

    outcome = passed.outcomes[0]
    assert outcome.outcome == "turn"
    turn_id = outcome.turn_id
    assert turn_id is not None and turn_id.startswith(f_paths.TURN_DIR_PREFIX)

    turn_dir = f_paths.turn_dir(world.foreman_root, turn_id)
    assert f_paths.briefing_path(world.foreman_root, turn_id).is_file()
    assert f_paths.turn_meta_path(world.foreman_root, turn_id).is_file()
    assert f_paths.actions_path(world.foreman_root, turn_id).is_file()
    # A zero-action Apply writes no applied.jsonl records.
    assert f_paths.complete_path(world.foreman_root, turn_id).is_file()
    assert (turn_dir / "decide.log").is_file()

    state = read_state_snapshot(world.foreman_root)
    assert state.turns_total == 4
    assert state.turns_today == 3
    assert state.turns_today_date == "2026-11-25"
    assert state.consecutive_failures == 0
    assert state.last_turn_at == "2026-11-25T12:00:00Z"
    assert state.last_turn_id == turn_id
    assert state.lifecycle_fingerprint == lifecycle_fingerprint(world.foreman())
    # Cursors advance past the seeded entries.
    assert state.inbox_cursor >= 1
    assert state.signal_cursor >= 1
    assert state.event_cursor > 5
    # The issue register from the briefing lands on state — owned issues 11
    # and 13 plus prior-authored 12 (absent from the snapshot, still tracked).
    assert set(state.issues) == {11, 12, 13}

    journal = f_journal.read_journal(world.foreman_root)
    assert any(entry.turn_id == turn_id for entry in journal)


def test_turn_state_roll_after_failures(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """A successful Turn resets consecutive_failures."""
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _patch_state(world, consecutive_failures=2)
    _succeeding_decide(fake_host)

    _run(world)

    assert read_state_snapshot(world.foreman_root).consecutive_failures == 0


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def _crash_survivor(world: _World, actions: list[dict[str, Any]] | None = None) -> str:
    """A Turn that died after Decide: actions.json + turn.json, no complete."""
    turn_id = "turn-20261125T115900Z"
    turn_dir = f_paths.turn_dir(world.foreman_root, turn_id)
    turn_dir.mkdir(parents=True)
    (f_paths.actions_path(world.foreman_root, turn_id)).write_text(
        json.dumps(
            {
                "turn_summary": "resumed turn",
                "actions": actions or [],
            }
        ),
        encoding="utf-8",
    )
    f_paths.turn_meta_path(world.foreman_root, turn_id).write_text(
        json.dumps(
            {
                "turn_id": turn_id,
                "wake": pass_run.WAKE_LIFECYCLE,
                "event_cursor": 9,
                "inbox_cursor": 7,
                "signal_cursor": 4,
                "lifecycle_fingerprint": "fp-resumed",
                "issues": {"11": {"body_hash": "hashAAA"}},
            }
        ),
        encoding="utf-8",
    )
    return turn_id


def test_pass_resumes_crashed_turn_at_apply(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _quiesce(world)  # no predicate — the survivor alone wakes the foreman
    turn_id = _crash_survivor(
        world,
        actions=[{"type": "write_memory", "path": "resumed-note", "content": "kept\n"}],
    )
    launcher = _succeeding_decide(fake_host)

    passed = _run(world)

    outcome = passed.outcomes[0]
    assert outcome.outcome == "turn" and outcome.turn_id == turn_id
    assert launcher.calls == []  # Apply replays; the model is never re-asked
    assert f_paths.complete_path(world.foreman_root, turn_id).is_file()
    memory = f_memory.read_memory(world.foreman_root, "resumed-note")
    assert memory.ok and "kept" in memory.value
    state = read_state_snapshot(world.foreman_root)
    assert state.last_turn_id == turn_id
    # Bookkeeping replays from turn.json, not from a fresh Collect.
    assert state.inbox_cursor == 7
    assert state.signal_cursor == 4
    assert state.lifecycle_fingerprint == "fp-resumed"
    assert state.turns_total == 4
    completed = _events(world, pass_run.EVENT_TURN_COMPLETED)
    assert completed[0]["wake"] == pass_run.WAKE_LIFECYCLE


def test_resumed_turn_skips_bookkeeping_replay(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """Bookkeeping already landed: the resume only repairs complete.json."""
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _quiesce(world)
    turn_id = _crash_survivor(world)
    _patch_state(
        world,
        last_turn_id=turn_id,
        turns_total=9,
        last_turn_at="2026-11-25T10:00:00Z",
    )
    _succeeding_decide(fake_host)

    passed = _run(world)

    assert passed.outcomes[0].outcome == "turn"
    state = read_state_snapshot(world.foreman_root)
    # No double-count, no journal re-append.
    assert state.turns_total == 9
    assert state.last_turn_at == "2026-11-25T10:00:00Z"
    assert sum(1 for e in f_journal.read_journal(world.foreman_root) if e.turn_id == turn_id) == 0
    assert f_paths.complete_path(world.foreman_root, turn_id).is_file()


# ---------------------------------------------------------------------------
# Failure accounting
# ---------------------------------------------------------------------------


def _failing_decide(fake_host: Any) -> FakeDecideLauncher:
    launcher = FakeDecideLauncher(
        [RunResult(returncode=1, stdout="", stderr="unexpected model death")]
    )
    fake_host(decide_launch=launcher)
    return launcher


def test_failed_turn_increments_and_events(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _failing_decide(fake_host)

    passed = _run(world)

    outcome = passed.outcomes[0]
    assert outcome.outcome == "failed"
    assert "unexpected model death" in (outcome.detail or "")
    state = read_state_snapshot(world.foreman_root)
    assert state.consecutive_failures == 1
    assert state.state == "active"
    failed = _events(world, pass_run.EVENT_TURN_FAILED)
    assert failed[0]["stage"] == "decide"
    assert failed[0]["consecutive_failures"] == 1


def test_three_failures_fault_the_foreman(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _failing_decide(fake_host)

    for _ in range(pass_run.FAULT_LIMIT):
        _run(world)

    state = read_state_snapshot(world.foreman_root)
    assert state.consecutive_failures == pass_run.FAULT_LIMIT
    assert state.state == "faulted"
    reports = [
        item for item in f_items.load_items(world.foreman_root) if item.kind == f_items.ITEM_REPORT
    ]
    assert len(reports) == 1
    assert "faulted" in (reports[0].text or "")
    assert reports[0].command == f"charlie gang resume {FOREMAN_NAME}"

    # A faulted foreman is skipped, not retried, on the next pass.
    passed = _run(world)
    assert passed.outcomes[0].outcome == "skipped"
    assert passed.outcomes[0].reason == "state:faulted"


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def test_turn_dir_retention_keeps_newest(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _succeeding_decide(fake_host)
    turns = world.foreman_root / "turns"
    for i in range(pass_run.TURN_RETENTION + 4):
        (turns / f"turn-20200101T{i:06d}").mkdir()

    passed = _run(world)

    assert passed.outcomes[0].outcome == "turn"
    kept = _turn_dirs(world)
    ours = [name for name in kept if name.startswith(f_paths.TURN_DIR_PREFIX)]
    assert len(ours) == pass_run.TURN_RETENTION
    assert "turn-20200101T000000" not in kept  # oldest pruned
    assert "t-0011" in kept  # foreign names are never touched


# ---------------------------------------------------------------------------
# End-to-end: the local-issues lane
# ---------------------------------------------------------------------------


def test_end_to_end_files_a_local_issue(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _succeeding_decide(
        fake_host,
        actions=[
            {
                "type": "file_issue",
                "key": "slice-two",
                "repo": REPO_KEY,
                "title": "Slice two",
                "body": "The foreman's follow-up slice.",
            }
        ],
    )

    passed = _run(world)

    assert passed.outcomes[0].outcome == "turn"
    gh = github_client_for(world.repo_root, world.foreman().config, github=GitHub, dry_run=False)
    issues = gh.issue_list(labels=["foreman:g1"], state="open")
    assert len(issues) == 1
    assert issues[0]["title"] == "Slice two"
    applied = _events(world, "foreman_action_applied")
    assert applied and applied[0]["key"] == "slice-two"
