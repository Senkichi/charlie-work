"""Shared helpers for the ``charlie scheme`` pass test modules (issue #3122).

Hoisted when ``tests/test_foreman_pass.py`` crossed the 800-line module cap
(file-size ratchet, issue #1442) and split into seam-named siblings —
``test_foreman_pass.py`` (wake predicate + deferral bounds),
``test_foreman_pass_turn.py`` (Turn bookkeeping, resume, failure
accounting), and ``test_foreman_pass_cli.py`` (dry-run modes, locks,
pass-level gates, the CLI seam). The cross-test-module import guard
(``tests/test_zero_cross_test_import_guard.py``, issue #1284) forbids a
``test_*.py -> test_*.py`` import, so the builders more than one sibling
shares live here: the seeded-world wrapper, the Decide-launcher installer,
the state patchers, and the pass/events readers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _foreman_collect_helpers import NOW, _build_world, _World

from charlie_work import instrumentation
from charlie_work.foreman import journal as f_journal
from charlie_work.foreman import paths as f_paths
from charlie_work.foreman import pass_run
from charlie_work.foreman.collect import lifecycle_fingerprint
from charlie_work.foreman.state import (
    ForemanState,
    read_state_snapshot,
    save_state,
    with_updates,
)
from charlie_work.host.fakes import FakeDecideLauncher
from charlie_work.subprocess_runner import RunResult


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
