"""The foreman pass: wake predicate, deferral bounds, Turn, resume, faulted (issue #3122).

Every scenario builds on the seeded world in
``tests/_foreman_collect_helpers.py``; Decide runs through the host
``decide_launch`` port (``FakeDecideLauncher``) and Apply through the
repo's configured issue client — ``local_issues.enabled`` keeps even the
end-to-end cases file-backed. Nothing here starts a model or touches ``gh``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _foreman_collect_helpers import (
    FOREMAN_NAME,
    NOW,
    REPO_KEY,
    _build_world,
    _World,
)
from _worktree_fixtures import (
    _wt_scratch as _register_wt_scratch,  # noqa: F401 -- registers the wt_scratch fixture
)

from charlie_work import instrumentation
from charlie_work.file_lock import try_acquire_byte_range_lock
from charlie_work.fleet_pause import write_fleet_pause
from charlie_work.foreman import items as f_items
from charlie_work.foreman import journal as f_journal
from charlie_work.foreman import memory as f_memory
from charlie_work.foreman import paths as f_paths
from charlie_work.foreman import pass_run
from charlie_work.foreman.collect import lifecycle_fingerprint
from charlie_work.foreman.state import (
    ForemanState,
    read_state_snapshot,
    save_state,
    with_updates,
)
from charlie_work.foreman.store import scaffold_foreman
from charlie_work.host.fakes import FakeDecideLauncher
from charlie_work.local_issues import github_client_for
from charlie_work.github import GitHub
from charlie_work.subprocess_runner import RunResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _envelope(actions: list[dict[str, Any]] | None = None) -> str:
    """A successful harness result envelope carrying ``structured_output``."""
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "duration_ms": 1234,
            "num_turns": 3,
            "result": "",
            "total_cost_usd": 0.0123,
            "usage": {"input_tokens": 11, "output_tokens": 22},
            "structured_output": {
                "turn_summary": "turn finished",
                "actions": actions or [],
            },
        }
    )


def _world(tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> _World:
    world = _build_world(tmp_path, wt_scratch, monkeypatch, **kw)
    # The pass's Apply resolves the repo's issue client from config; the
    # file-backed lane keeps every Turn hermetic.
    (world.repo_root / "orchestrator.config.yaml").write_text(
        "local_issues:\n  enabled: true\n", encoding="utf-8"
    )
    (world.repo_root / "docs" / "issues").mkdir(parents=True, exist_ok=True)
    return world


def _stamp(now: datetime) -> str:
    return now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _patch_state(world: _World, **changes: Any) -> ForemanState:
    state = read_state_snapshot(world.foreman_root)
    saved = save_state(world.foreman_root, with_updates(state, **changes))
    assert saved.ok, saved.error
    return saved.value


def _quiesce(world: _World, **changes: Any) -> ForemanState:
    """Drain every wake source so only the tested condition can fire."""
    foreman = world.foreman()
    inbox = f_journal.read_inbox(world.foreman_root)
    signals = f_journal.read_signals(world.foreman_root)
    base: dict[str, Any] = {
        "lifecycle_fingerprint": lifecycle_fingerprint(foreman),
        "inbox_cursor": max((m.id for m in inbox), default=0),
        "signal_cursor": max((s.id for s in signals), default=0),
        # Past the seeded 09:00 item answer, inside the min-turn-interval.
        "last_turn_at": "2026-11-25T11:00:00Z",
    }
    base.update(changes)
    return _patch_state(world, **base)


def _succeeding_decide(
    fake_host: Any, actions: list[dict[str, Any]] | None = None
) -> FakeDecideLauncher:
    launcher = FakeDecideLauncher([RunResult(returncode=0, stdout=_envelope(actions), stderr="")])
    fake_host(decide_launch=launcher)
    return launcher


def _events(world: _World, kind: str) -> list[dict[str, Any]]:
    """Event payloads of ``kind`` recorded in the foreman's events.db."""
    return [
        row["payload"]
        for row in instrumentation.query_events(f_paths.state_path(world.foreman_root), kind=kind)
    ]


def _turn_dirs(world: _World) -> list[str]:
    root = world.foreman_root / "turns"
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir())


def _run(world: _World, **kw: Any) -> pass_run.PassResult:
    result = pass_run.run_pass(world.fleet_dir, now=NOW, **kw)
    assert result.ok, result.error
    return result.value


# ---------------------------------------------------------------------------
# Wake predicate
# ---------------------------------------------------------------------------


def test_wake_fires_on_fingerprint_change(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    launcher = _succeeding_decide(fake_host)

    passed = _run(world)

    assert [(o.name, o.outcome) for o in passed.outcomes] == [(FOREMAN_NAME, "turn")]
    completed = _events(world, pass_run.EVENT_TURN_COMPLETED)
    assert len(completed) == 1
    assert completed[0]["wake"] == pass_run.WAKE_LIFECYCLE
    assert completed[0]["input_tokens"] == 11
    assert completed[0]["output_tokens"] == 22
    assert launcher.calls  # Decide ran


def test_wake_fires_on_unread_inbox(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _quiesce(world, inbox_cursor=0)  # the seeded message becomes unread
    _succeeding_decide(fake_host)

    passed = _run(world)

    assert passed.outcomes[0].outcome == "turn"
    assert _events(world, pass_run.EVENT_TURN_COMPLETED)[0]["wake"] == pass_run.WAKE_INBOX


def test_wake_fires_on_new_item_answer(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    # The seeded item answered at 09:00; put last_turn_at just before it.
    _quiesce(world, last_turn_at="2026-11-25T08:30:00Z")
    _succeeding_decide(fake_host)

    passed = _run(world)

    assert passed.outcomes[0].outcome == "turn"
    assert _events(world, pass_run.EVENT_TURN_COMPLETED)[0]["wake"] == pass_run.WAKE_ITEM_ANSWER


def test_wake_fires_on_unread_signal(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _quiesce(world, signal_cursor=0)
    _succeeding_decide(fake_host)

    passed = _run(world)

    assert passed.outcomes[0].outcome == "turn"
    assert _events(world, pass_run.EVENT_TURN_COMPLETED)[0]["wake"] == pass_run.WAKE_SIGNAL


def test_wake_fires_on_report_cadence(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    charter = (world.foreman_root / "charter.yaml").read_text(encoding="utf-8")
    (world.foreman_root / "charter.yaml").write_text(
        charter + 'cadence:\n  report_at: "00:00"\n', encoding="utf-8"
    )
    _quiesce(world, last_scheduled_turn_at=None)
    _succeeding_decide(fake_host)

    passed = _run(world)

    assert passed.outcomes[0].outcome == "turn"
    assert _events(world, pass_run.EVENT_TURN_COMPLETED)[0]["wake"] == pass_run.WAKE_REPORT_DUE
    # A scheduled Turn stamps last_scheduled_turn_at so it does not refire.
    state = read_state_snapshot(world.foreman_root)
    assert state.last_scheduled_turn_at == "2026-11-25T12:00:00Z"


def test_wake_fires_on_max_quiet(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    # 25h > max_quiet_hours (24) — and the seeded item answer predates it.
    items_file = f_paths.items_path(world.foreman_root)
    raw = json.loads(items_file.read_text(encoding="utf-8"))
    for record in raw["items"]:
        if record.get("answered_at"):
            record["answered_at"] = "2026-11-24T10:00:00Z"
    items_file.write_text(json.dumps(raw), encoding="utf-8")
    _quiesce(world, last_turn_at="2026-11-24T11:00:00Z")
    _succeeding_decide(fake_host)

    passed = _run(world)

    assert passed.outcomes[0].outcome == "turn"
    assert _events(world, pass_run.EVENT_TURN_COMPLETED)[0]["wake"] == pass_run.WAKE_QUIET


def test_idle_when_nothing_wakes(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _quiesce(world)
    launcher = _succeeding_decide(fake_host)

    passed = _run(world)

    assert [(o.name, o.outcome) for o in passed.outcomes] == [(FOREMAN_NAME, "idle")]
    assert launcher.calls == []
    assert _turn_dirs(world) == ["t-0011"]  # only the seeded prior Turn
    assert _events(world, pass_run.EVENT_TURN_COMPLETED) == []


# ---------------------------------------------------------------------------
# Skipped states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["paused", "done", "faulted"])
def test_pass_skips_non_active_states(
    tmp_path: Path,
    wt_scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_host,
    state: str,
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _patch_state(world, state=state)
    launcher = _succeeding_decide(fake_host)

    passed = _run(world)

    assert passed.outcomes[0].outcome == "skipped"
    assert passed.outcomes[0].reason == f"state:{state}"
    assert launcher.calls == []


def test_degraded_foreman_still_turns(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _patch_state(world, state="degraded")
    _succeeding_decide(fake_host)

    passed = _run(world)

    assert passed.outcomes[0].outcome == "turn"


# ---------------------------------------------------------------------------
# Deferral bounds
# ---------------------------------------------------------------------------


def test_deferred_by_min_turn_interval(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    # 15 minutes since the last Turn; the bound is 30.
    _patch_state(world, last_turn_at="2026-11-25T11:45:00Z")
    launcher = _succeeding_decide(fake_host)

    passed = _run(world)

    outcome = passed.outcomes[0]
    assert outcome.outcome == "deferred"
    assert outcome.reason == pass_run.DEFER_MIN_INTERVAL
    deferred = _events(world, pass_run.EVENT_WAKE_DEFERRED)
    assert deferred[0]["reason"] == pass_run.DEFER_MIN_INTERVAL
    assert deferred[0]["wake"] == pass_run.WAKE_LIFECYCLE
    assert launcher.calls == []
    assert read_state_snapshot(world.foreman_root).consecutive_failures == 0


def test_deferred_by_max_turns_per_day(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _patch_state(world, turns_today=12, turns_today_date="2026-11-25", last_turn_at=None)
    launcher = _succeeding_decide(fake_host)

    passed = _run(world)

    assert passed.outcomes[0].reason == pass_run.DEFER_MAX_TURNS
    assert _events(world, pass_run.EVENT_WAKE_DEFERRED)[0]["reason"] == (pass_run.DEFER_MAX_TURNS)
    assert launcher.calls == []


def test_deferred_by_fleet_pause(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    write_fleet_pause(str(world.fleet_dir), reason="operator")
    launcher = _succeeding_decide(fake_host)

    passed = _run(world)

    assert passed.outcomes[0].reason == pass_run.DEFER_FLEET_PAUSED
    assert _events(world, pass_run.EVENT_WAKE_DEFERRED)[0]["reason"] == (
        pass_run.DEFER_FLEET_PAUSED
    )
    assert launcher.calls == []


def test_deferred_by_stale_snapshot(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    # The pass's Collect refuses a snapshot older than 2x the full-pass
    # interval (default 900s -> the seeded 30s one is fine; 4h is not).
    world.write_snapshot([], written_at="2026-11-25T08:00:00Z")
    launcher = _succeeding_decide(fake_host)

    passed = _run(world)

    outcome = passed.outcomes[0]
    assert outcome.outcome == "deferred"
    assert outcome.reason == pass_run.DEFER_STALE_SNAPSHOT
    assert _events(world, pass_run.EVENT_WAKE_DEFERRED)[0]["reason"] == (
        pass_run.DEFER_STALE_SNAPSHOT
    )
    assert launcher.calls == []
    # A stale-snapshot deferral is not a Turn failure.
    assert read_state_snapshot(world.foreman_root).consecutive_failures == 0


def test_deferred_decide_does_not_count_as_failure(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    launcher = FakeDecideLauncher(
        [
            RunResult(
                returncode=1,
                stdout="",
                stderr="API Error: weekly usage quota has been exhausted",
            )
        ]
    )
    fake_host(decide_launch=launcher)

    passed = _run(world)

    outcome = passed.outcomes[0]
    assert outcome.outcome == "deferred"
    assert outcome.reason == pass_run.DEFER_DECIDE
    assert _events(world, pass_run.EVENT_WAKE_DEFERRED)[0]["reason"] == (pass_run.DEFER_DECIDE)
    state = read_state_snapshot(world.foreman_root)
    assert state.consecutive_failures == 0
    # The half-finished Turn holds no actions.json — it is never a resume.
    assert pass_run._resumable_turn(world.foreman_root) is None


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
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_prints_plan_and_changes_nothing(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _succeeding_decide(
        fake_host,
        actions=[
            {
                "type": "file_issue",
                "key": "k1",
                "repo": REPO_KEY,
                "title": "Would file",
                "body": "body",
            }
        ],
    )

    passed = _run(world, dry_run=True)

    outcome = passed.outcomes[0]
    assert outcome.outcome == "dry_run"
    plan = json.loads(outcome.detail or "{}")
    assert plan["plan"][0]["type"] == "file_issue"
    # The dry Turn uses the dry- prefix so it is never a resume candidate.
    assert outcome.turn_id is not None
    assert outcome.turn_id.startswith(f_paths.DRY_TURN_DIR_PREFIX)
    # State is untouched and no issue was filed.
    state = read_state_snapshot(world.foreman_root)
    assert state.turns_total == 3
    assert state.last_turn_id is None
    gh = github_client_for(world.repo_root, world.foreman().config, github=GitHub, dry_run=False)
    assert gh.issue_list(state="open") == []


def test_dry_run_no_decide_stops_after_briefing(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    launcher = _succeeding_decide(fake_host)

    passed = _run(world, dry_run=True, no_decide=True)

    outcome = passed.outcomes[0]
    assert outcome.outcome == "dry_run" and outcome.reason == "no_decide"
    assert launcher.calls == []
    assert f_paths.briefing_path(world.foreman_root, outcome.turn_id).is_file()
    assert read_state_snapshot(world.foreman_root).turns_total == 3


def test_no_decide_requires_dry_run(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    result = pass_run.run_pass(world.fleet_dir, no_decide=True)
    assert not result.ok
    assert "--no-decide" in (result.error or "")


# ---------------------------------------------------------------------------
# Locking and pass-level gates
# ---------------------------------------------------------------------------


def test_pass_lock_contention_short_circuits(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _succeeding_decide(fake_host)
    held = try_acquire_byte_range_lock(f_paths.pass_lock_path(world.fleet_dir))
    assert held is not None
    try:
        passed = _run(world)
        assert passed.locked is True
        assert passed.outcomes == ()
    finally:
        held.release()


def test_turn_lock_contention_skips_the_foreman(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    launcher = _succeeding_decide(fake_host)
    held = try_acquire_byte_range_lock(f_paths.turn_lock_path(world.foreman_root))
    assert held is not None
    try:
        passed = _run(world)
        outcome = passed.outcomes[0]
        assert outcome.outcome == "skipped"
        assert outcome.reason == "turn lock held"
        assert launcher.calls == []
    finally:
        held.release()


def test_pass_disabled_by_config(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    launcher = _succeeding_decide(fake_host)
    (world.fleet_dir / "config.yaml").write_text("foreman:\n  enabled: false\n", encoding="utf-8")

    passed = _run(world)

    assert passed.disabled is True
    assert passed.outcomes == ()
    assert launcher.calls == []


def test_no_foremen_is_an_empty_pass(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True)
    result = pass_run.run_pass(fleet_dir, now=NOW)
    assert result.ok
    assert result.value.outcomes == ()
    assert not result.value.locked and not result.value.disabled


def test_only_restricts_the_pass(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _succeeding_decide(fake_host)
    other = scaffold_foreman(world.fleet_dir, "g2")
    assert other.ok, other.error
    g2_root = world.fleet_dir / "foremen" / "g2"
    (g2_root / "charter.yaml").write_text(
        (world.foreman_root / "charter.yaml")
        .read_text(encoding="utf-8")
        .replace("name: g1", "name: g2"),
        encoding="utf-8",
    )
    # scaffold_foreman seeds state.json paused — g2 is a skip, not a Turn.

    passed = _run(world)
    assert {outcome.name for outcome in passed.outcomes} == {"g1", "g2"}

    passed = _run(world, only="g2")
    assert [(o.name, o.outcome) for o in passed.outcomes] == [("g2", "skipped")]


def test_only_unknown_foreman_is_an_error(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    result = pass_run.run_pass(world.fleet_dir, only="nope", now=NOW)
    assert not result.ok
    assert "nope" in (result.error or "")


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


# ---------------------------------------------------------------------------
# The CLI seam
# ---------------------------------------------------------------------------


def test_scheme_command_via_cli(
    tmp_path: Path,
    wt_scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_host,
    capsys: pytest.CaptureFixture,
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    launcher = _succeeding_decide(fake_host)
    # cli.main runs the pass at real wall time — refresh the seeded snapshot
    # (stamped at the suite's fixed NOW) past the staleness bound.
    world.write_snapshot([], written_at=_stamp(datetime.now(UTC)))

    from charlie_work import cli

    rc = cli.main(
        [
            "--fleet-dir",
            str(world.fleet_dir),
            "scheme",
            "--dry-run",
            "--no-decide",
        ]
    )

    assert rc == 0
    assert launcher.calls == []
    assert "dry-run" in capsys.readouterr().out


def test_scheme_command_reports_locked(
    tmp_path: Path,
    wt_scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    held = try_acquire_byte_range_lock(f_paths.pass_lock_path(world.fleet_dir))
    assert held is not None
    try:
        from charlie_work import cli

        rc = cli.main(["--fleet-dir", str(world.fleet_dir), "scheme"])
    finally:
        held.release()

    assert rc == 0
    assert "pass.lock" in capsys.readouterr().out
