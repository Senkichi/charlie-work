"""The foreman pass: wake predicate, skip states, deferral bounds (issue #3122).

Split out of the original 869-line module under the file-size ratchet
(issue #1442): this module keeps the *predicate* half — every wake reason,
the non-turn states, and every deferral bound. Turn bookkeeping, resume,
and failure accounting live in ``test_foreman_pass_turn.py``; dry-run
modes, locks, pass-level gates, and the CLI seam live in
``test_foreman_pass_cli.py``. Shared world/decide/event builders live in
``tests/_foreman_pass_fixtures.py``.

Every scenario builds on the seeded world in
``tests/_foreman_collect_helpers.py``; Decide runs through the host
``decide_launch`` port (``FakeDecideLauncher``) and Apply through the
repo's configured issue client — ``local_issues.enabled`` keeps even the
end-to-end cases file-backed. Nothing here starts a model or touches ``gh``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _foreman_collect_helpers import FOREMAN_NAME
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

from charlie_work.fleet_pause import write_fleet_pause
from charlie_work.foreman import paths as f_paths
from charlie_work.foreman import pass_run
from charlie_work.foreman.state import read_state_snapshot
from charlie_work.host.fakes import FakeDecideLauncher
from charlie_work.subprocess_runner import RunResult

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
