"""The limits inside ``charlie scheme`` passes — the dry-run gate (issue #3123).

Split out of ``test_foreman_limits.py`` under the file-size ratchet
(issue #1442): this module covers ``evaluate_limits``' interaction with
``pass_run.run_pass`` modes — a ``--dry-run`` pass must not persist the
limits' side effects on a live foreman. The limit mechanics themselves
(breaker, unattended, verdict, items, alarms, doctor) live in
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

from charlie_work.foreman import items as f_items
from charlie_work.foreman import limits as flimits
from charlie_work.foreman.state import read_state_snapshot


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
