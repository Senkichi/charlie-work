"""The limits inside ``charlie scheme`` passes — pass-level wiring (issue #3123).

Split out of ``test_foreman_limits.py`` under the file-size ratchet
(issue #1442): this module covers ``evaluate_limits``' interaction with
``pass_run.run_pass`` — the limits evaluation at pass start, the
``foreman_pass_completed`` fleet event, notify after a failed Turn, and the
``--dry-run`` gate that keeps a dry pass from persisting the limits' side
effects on a live foreman. The limit mechanics themselves (breaker,
unattended, verdict, items, alarms, doctor) live in
``test_foreman_limits.py``; shared builders in
``tests/_foreman_pass_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _foreman_pass_fixtures import (
    _authored,
    _events,
    _finish,
    _patch_state,
    _run,
    _succeeding_decide,
    _world,
)
from _worktree_fixtures import (
    _wt_scratch as _register_wt_scratch,  # noqa: F401 -- registers the wt_scratch fixture
)

from charlie_work import instrumentation, layout
from charlie_work.foreman import items as f_items
from charlie_work.foreman import limits as flimits
from charlie_work.foreman import pass_run
from charlie_work.foreman.state import read_state_snapshot


def test_run_pass_writes_fleet_pass_event(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)

    _run(world)

    fleet_events = [
        row["payload"]
        for row in instrumentation.query_events(
            layout.state_file_path(world.fleet_dir), kind=pass_run.EVENT_PASS_COMPLETED
        )
    ]
    assert len(fleet_events) == 1
    assert fleet_events[0]["foremen"] == 1


def test_run_pass_limits_eval_before_turn(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """The limits evaluation runs at pass start even for a quiet foreman."""
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _succeeding_decide(fake_host)
    _patch_state(world, issues=_authored(20, 21, 22))
    _finish(
        world,
        (20, "human_needed"),
        (21, "operator_queue"),
        (22, "human_needed"),
        (30, "done"),
        (31, "done"),
        (32, "done"),
        (33, "done"),
        (34, "human_needed"),
    )

    passed = _run(world)

    # The seeded world has a wake reason (the lifecycle fingerprint moved),
    # so the Turn lands — and the default breaker (sample=10) stays quiet on
    # three finished issues: limits ran, no trip, the Turn proceeded.
    assert passed.outcomes[0].outcome == "turn"
    state = read_state_snapshot(world.foreman_root)
    assert state.degraded_reasons == ()
    assert state.turns_total == 4


def test_run_pass_breaker_trip_gates_armed_turn(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """A trip during the pass leaves the foreman degraded for that same pass."""
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _succeeding_decide(fake_host)
    # Ten finished escalated foreman issues + a done baseline: the default
    # sample of 10 is full and the rate (1.0 > 1.5 * 0) trips in-pass.
    authored = _authored(*range(20, 30))
    _patch_state(world, issues=authored)
    _finish(
        world,
        *[(n, "human_needed") for n in range(20, 30)],
        (40, "done"),
        (41, "done"),
    )

    passed = _run(world)

    state = read_state_snapshot(world.foreman_root)
    assert state.state == "degraded"
    assert "breaker" in state.degraded_reasons
    # Degraded gates arming, not planning — the Turn still ran.
    assert passed.outcomes[0].outcome == "turn"
    assert state.turns_total == 4


def test_failed_turn_still_notifies_items(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    world = _world(tmp_path, wt_scratch, monkeypatch)
    from charlie_work.host.fakes import FakeDecideLauncher
    from charlie_work.subprocess_runner import RunResult

    fake_host(
        decide_launch=FakeDecideLauncher([RunResult(returncode=1, stdout="", stderr="boom")])
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        pass_run, "notify_unnotified_items", lambda root, *a, **k: calls.append(root)
    )

    _run(world)

    assert calls == [world.foreman_root]


def test_dry_run_pass_skips_limits_side_effects(
    tmp_path: Path, wt_scratch: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """--dry-run must not degrade the foreman, raise items, or force arming.

    Every limit is armed here — a tripping breaker, a stale decision item,
    and a due-and-failing verdict — so an unguarded evaluate_limits would
    touch state.json, items.json, and events.db in one pass.
    """
    world = _world(tmp_path, wt_scratch, monkeypatch)
    _succeeding_decide(fake_host)
    _patch_state(world, issues=_authored(*range(20, 40)))
    _finish(
        world,
        *[(n, "human_needed") for n in range(20, 40)],
        (50, "done"),
        (51, "done"),
    )
    stale = f_items.create_item(
        world.foreman_root,
        kind="decision",
        question="Stale question?",
        options=("a", "b"),
        recommended=0,
        created_at="2026-11-10T00:00:00Z",
    )
    assert stale.ok, stale.error
    before_state = read_state_snapshot(world.foreman_root)
    before_items = f_items.load_items(world.foreman_root)

    passed = _run(world, dry_run=True)

    assert passed.outcomes[0].outcome == "dry_run"
    assert read_state_snapshot(world.foreman_root) == before_state
    assert f_items.load_items(world.foreman_root) == before_items
    assert _events(world, flimits.EVENT_BREAKER_TRIPPED) == []
    assert _events(world, flimits.EVENT_UNATTENDED_LIMIT) == []
    assert _events(world, flimits.EVENT_VERDICT) == []
