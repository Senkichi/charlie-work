"""Tests for the sustained-window capacity-starvation escalation (issue #763).

Extracted from ``tests/test_fleet_dispatch.py`` and ``tests/test_config.py``
so new test code does not land in an over-cap monolith (file-size ratchet,
issue #1442). Covers:

* Config validation (``parse_runner_capacity_escalation`` / ``load_config``)
* Detector unit tests (edge trigger, sustained window, recovery, dry-run,
  disabled, no-slack-budget)
* Operator-digest surface (``_build_fleet_attention_digest``)
* Prologue integration (``_run_fleet_allocation_prologue`` end-to-end)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from charlie_work import layout
from charlie_work.capacity_starvation_escalation import (
    RunnerCapacityEscalationConfig,
    _load_capacity_starvation_state,
    _starved_repos_from_plan,
    detect_capacity_starvation_escalation,
)
from charlie_work.config import (
    ConfigError,
    OrchestratorConfig,
    RunnerAllocationConfig,
    RunnerScalingConfig,
    load_config,
)
from charlie_work.global_config import load_layered_config
from charlie_work.fleet_dispatch import (
    _build_fleet_attention_digest,
    _run_fleet_allocation_prologue,
)
from charlie_work.instrumentation import query_events
from ci_fleet.runner_allocation import AllocationPlan, RepoTarget
from ci_fleet.runner_allocation_pass import AllocationPassResult


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_fleet_dispatch.py / test_config.py to keep
# this module self-contained — no cross-test-module import precedent exists
# in this repo, and these are small enough that duplication is cheaper than
# a shared test-helpers module).
# ---------------------------------------------------------------------------


def _write_config(config_file: Path, content: str) -> None:
    config_file.write_text(content, encoding="utf-8")


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


_BASE_YAML = """\
labels:
  ready: automated-ready
  queued: agent:queued
  in_progress: agent:in-progress
runtime:
  state_dir: .var/charlie-work
"""


def _make_repo(tmp_path: Path, name: str, *, api_worker: str | None) -> Path:
    """Create a repo dir with a config file. api_worker is the YAML snippet or None."""
    repo = tmp_path / name
    repo.mkdir(parents=True)
    config = repo / "orchestrator.config.yaml"
    content = _BASE_YAML
    if api_worker is not None:
        content += "\n" + api_worker
    config.write_text(content, encoding="utf-8")
    (repo / ".var" / "charlie-work").mkdir(parents=True)
    return repo


def _make_fleet_json(tmp_path: Path, fleet_dir: Path, repos: dict[str, dict[str, Any]]) -> None:
    fleet_json = fleet_dir / "fleet.json"
    fleet_json.parent.mkdir(parents=True, exist_ok=True)
    registry = {"version": 1, "repos": repos}
    fleet_json.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _starved_plan(
    *,
    budget: int = 8,
    repo: str = "Senkichi/charlie-work",
    demand: int = 5,
    capacity: int = 2,
    running: int = 2,
    other_running: int = 4,
) -> AllocationPlan:
    """A plan where ``repo`` is starved (demand > capacity) with budget slack.

    ``other_running`` is a second repo's running count; combined with ``running``
    it sets the host-wide total below ``budget`` so ``spare_budget > 0`` -- the
    condition that makes starvation worth escalating (idle budget a bigger
    registration could fill).
    """
    targets = (
        RepoTarget(repo=repo, target=running, running=running, demand=demand, capacity=capacity),
        RepoTarget(
            repo="Senkichi/job-cannon",
            target=other_running,
            running=other_running,
            demand=0,
            capacity=5,
        ),
    )
    return AllocationPlan(budget=budget, budget_reason="configured", targets=targets, changes=())


def _escalation_config(**overrides: Any) -> RunnerCapacityEscalationConfig:
    return RunnerCapacityEscalationConfig(**overrides)


# ---------------------------------------------------------------------------
# Config validation tests (moved from test_config.py)
# ---------------------------------------------------------------------------


def test_runner_capacity_escalation_accepts_valid_config(tmp_path: Path) -> None:
    """Issue #763: a well-formed runner_capacity_escalation section parses through
    to the dataclass with the configured (non-default) values, not silently
    falling back to defaults."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
runner_capacity_escalation:
  enabled: false
  starvation_escalation_minutes: 30
""",
    )
    config = load_config(config_file)
    assert config.runner_capacity_escalation.enabled is False
    assert config.runner_capacity_escalation.starvation_escalation_minutes == 30


def test_runner_capacity_escalation_enabled_rejects_non_bool(tmp_path: Path) -> None:
    """Issue #763: runner_capacity_escalation.enabled must be a bool (config.py
    build_config_from_data, the `rce_enabled` type check)."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
runner_capacity_escalation:
  enabled: "true"
""",
    )
    with pytest.raises(ConfigError, match="runner_capacity_escalation.*enabled.*must be a bool"):
        load_config(config_file)


def test_runner_capacity_escalation_starvation_escalation_minutes_rejects_non_int(
    tmp_path: Path,
) -> None:
    """Issue #763: starvation_escalation_minutes must be an int, and a bool
    (which is a subtype of int in Python) is explicitly excluded by the
    `isinstance(rce_minutes, bool)` guard in config.py."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
runner_capacity_escalation:
  starvation_escalation_minutes: true
""",
    )
    with pytest.raises(
        ConfigError,
        match="runner_capacity_escalation.*starvation_escalation_minutes.*must be an int",
    ):
        load_config(config_file)


def test_runner_capacity_escalation_starvation_escalation_minutes_rejects_non_positive(
    tmp_path: Path,
) -> None:
    """Issue #763: starvation_escalation_minutes must be > 0 -- a zero or
    negative sustained window would let a single starved pass escalate
    immediately, defeating the sustained-window design that guards against a
    single-pass spike raising a false alarm."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
runner_capacity_escalation:
  starvation_escalation_minutes: 0
""",
    )
    with pytest.raises(
        ConfigError,
        match="runner_capacity_escalation.*starvation_escalation_minutes.*must be > 0",
    ):
        load_config(config_file)


# --- Layered-config host-wide-only rejection (global_config.load_layered_config) ---
# Mirrors the runner_allocation precedent in tests/test_config.py
# (test_load_layered_config_rejects_per_repo_runner_allocation and its
# 'accepts at global layer' counterpart). runner_capacity_escalation is the
# same shape of host-wide concern (issue #763), so the per-repo rejection
# branch in global_config.py:200-205 needs the same regression coverage.


def test_load_layered_config_rejects_per_repo_runner_capacity_escalation(
    tmp_path: Path,
) -> None:
    """Issue #763: a per-repo ``runner_capacity_escalation`` section must be
    rejected by ``load_layered_config``.

    The merge is section-by-section with the per-repo file winning per key, so
    without the explicit rejection in ``global_config.load_layered_config`` a
    per-repo ``orchestrator.config.yaml`` could silently override a host-wide
    capacity signal -- three repos holding three opinions about one machine's
    starvation window. The section is documented host-wide-only (see
    ``RunnerCapacityEscalationConfig``); make the invalid state
    unrepresentable rather than merely unused, exactly as ``runner_allocation``
    does (issue #600).
    """
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text(
        "runner_capacity_escalation:\n  enabled: true\n", encoding="utf-8"
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_config(
        repo_root / "orchestrator.config.yaml",
        "runner_capacity_escalation:\n  enabled: true\n  starvation_escalation_minutes: 2\n",
    )

    with pytest.raises(ConfigError, match="host-wide only"):
        load_layered_config(repo_root, fleet_dir_override=str(fleet))


def test_load_layered_config_accepts_global_runner_capacity_escalation(
    tmp_path: Path,
) -> None:
    """The rejection is scoped to the per-repo layer; the global fleet layer
    keeps ``runner_capacity_escalation`` and parses it through to the dataclass
    with the configured (non-default) values, not silently falling back to
    defaults."""
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text(
        "runner_capacity_escalation:\n  enabled: false\n  starvation_escalation_minutes: 30\n",
        encoding="utf-8",
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    # No per-repo config at all -- the global layer stands alone.
    config = load_layered_config(repo_root, fleet_dir_override=str(fleet))
    assert config.runner_capacity_escalation.enabled is False
    assert config.runner_capacity_escalation.starvation_escalation_minutes == 30


# ---------------------------------------------------------------------------
# Detector unit tests (moved from test_fleet_dispatch.py)
# ---------------------------------------------------------------------------


def test_capacity_escalation_first_starved_pass_records_but_does_not_escalate(
    tmp_path: Path,
) -> None:
    """A single starved pass must arm the window, not raise a false alarm.

    The sustained window exists precisely so a transient spike that clears on
    the next pass does not escalate. The first starved pass records the episode
    start in the sidecar and emits nothing.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_state_path = fleet_dir / "state.json"
    now = datetime.now(UTC)

    events = detect_capacity_starvation_escalation(
        _starved_plan(),
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=_escalation_config(),
        dry_run=False,
        now=now,
    )

    assert events == []
    # The sidecar records the episode start, escalated=False.
    state = _load_capacity_starvation_state(
        layout.capacity_starvation_state_path(override=str(fleet_dir))
    )
    assert "Senkichi/charlie-work" in state
    assert state["Senkichi/charlie-work"]["escalated"] is False
    # No escalation event in events.db.
    assert query_events(fleet_state_path, kind="runner_capacity_starvation_escalation") == []


def test_capacity_escalation_fires_after_sustained_window(tmp_path: Path) -> None:
    """Once starvation persists past the configured window, escalate once.

    The escalation must reach both the operator digest (the returned event
    dict) and the durable fleet-level events.db (the queryable record), and
    name the starved repo so an operator knows *which* repo is starving.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_state_path = fleet_dir / "state.json"
    start = datetime.now(UTC)

    # Pass 1: arm the window.
    detect_capacity_starvation_escalation(
        _starved_plan(),
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=_escalation_config(starvation_escalation_minutes=15),
        dry_run=False,
        now=start,
    )

    # Pass 2: still starved, but only 5 min in -- below the 15-min window.
    events_below = detect_capacity_starvation_escalation(
        _starved_plan(),
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=_escalation_config(starvation_escalation_minutes=15),
        dry_run=False,
        now=start + timedelta(minutes=5),
    )
    assert events_below == []

    # Pass 3: 16 min in -- the window is crossed.
    events = detect_capacity_starvation_escalation(
        _starved_plan(),
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=_escalation_config(starvation_escalation_minutes=15),
        dry_run=False,
        now=start + timedelta(minutes=16),
    )

    assert len(events) == 1
    assert events[0]["type"] == "runner_capacity_starvation_escalation"
    assert events[0]["repo"] == "Senkichi/charlie-work"
    assert events[0]["demand"] == 5
    assert events[0]["capacity"] == 2
    assert events[0]["spare_budget"] == 2  # budget 8 - running 2 - running 4
    assert "provision" in events[0]["reason"]

    # Durable record in the fleet-level events.db at error level.
    rows = query_events(fleet_state_path, kind="runner_capacity_starvation_escalation")
    assert len(rows) == 1
    assert rows[0]["repo"] == "Senkichi/charlie-work"
    assert rows[0]["payload"]["demand"] == 5
    assert rows[0]["level"] == "error"

    # The sidecar marks the episode as escalated.
    state = _load_capacity_starvation_state(
        layout.capacity_starvation_state_path(override=str(fleet_dir))
    )
    assert state["Senkichi/charlie-work"]["escalated"] is True


def test_capacity_escalation_is_edge_triggered_does_not_refire(tmp_path: Path) -> None:
    """After firing once, subsequent starved passes must stay silent.

    A level-triggered escalation would turn one real signal into an unbounded
    stream of identical rows, one per pass, for as long as the condition
    holds -- the same failure shape #799's edge trigger exists to prevent. The
    sidecar's ``escalated`` flag suppresses re-firing until recovery.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_state_path = fleet_dir / "state.json"
    start = datetime.now(UTC)
    cfg = _escalation_config(starvation_escalation_minutes=10)

    detect_capacity_starvation_escalation(
        _starved_plan(),
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=cfg,
        dry_run=False,
        now=start,
    )
    fired = detect_capacity_starvation_escalation(
        _starved_plan(),
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=cfg,
        dry_run=False,
        now=start + timedelta(minutes=11),
    )
    assert len(fired) == 1

    # Subsequent passes while still starved: silent.
    for minutes in (12, 20, 30):
        silent = detect_capacity_starvation_escalation(
            _starved_plan(),
            fleet_dir_override=str(fleet_dir),
            fleet_state_path=fleet_state_path,
            escalation_config=cfg,
            dry_run=False,
            now=start + timedelta(minutes=minutes),
        )
        assert silent == []

    # Still exactly one durable row.
    rows = query_events(fleet_state_path, kind="runner_capacity_starvation_escalation")
    assert len(rows) == 1


def test_capacity_escalation_recovery_resets_window(tmp_path: Path) -> None:
    """A repo that recovers drops out of the sidecar; the next episode is fresh.

    Without this, a repo that starves, recovers, and starves again would
    escalate immediately on the second episode's first pass -- reusing the
    first episode's elapsed window. The sidecar must clear on recovery so each
    episode measures its own sustained duration.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_state_path = fleet_dir / "state.json"
    start = datetime.now(UTC)
    cfg = _escalation_config(starvation_escalation_minutes=10)

    # Episode 1: arm, then escalate.
    detect_capacity_starvation_escalation(
        _starved_plan(),
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=cfg,
        dry_run=False,
        now=start,
    )
    detect_capacity_starvation_escalation(
        _starved_plan(),
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=cfg,
        dry_run=False,
        now=start + timedelta(minutes=11),
    )
    state = _load_capacity_starvation_state(
        layout.capacity_starvation_state_path(override=str(fleet_dir))
    )
    assert "Senkichi/charlie-work" in state

    # Recovery: demand drops to capacity (no longer starved).
    recovered_plan = AllocationPlan(
        budget=8,
        budget_reason="configured",
        targets=(
            RepoTarget(
                repo="Senkichi/charlie-work",
                target=2,
                running=2,
                demand=2,
                capacity=2,
            ),
            RepoTarget(
                repo="Senkichi/job-cannon",
                target=4,
                running=4,
                demand=0,
                capacity=5,
            ),
        ),
        changes=(),
    )
    detect_capacity_starvation_escalation(
        recovered_plan,
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=cfg,
        dry_run=False,
        now=start + timedelta(minutes=12),
    )
    state = _load_capacity_starvation_state(
        layout.capacity_starvation_state_path(override=str(fleet_dir))
    )
    assert "Senkichi/charlie-work" not in state

    # Episode 2: first starved pass must NOT escalate (fresh window).
    events = detect_capacity_starvation_escalation(
        _starved_plan(),
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=cfg,
        dry_run=False,
        now=start + timedelta(minutes=13),
    )
    assert events == []
    state = _load_capacity_starvation_state(
        layout.capacity_starvation_state_path(override=str(fleet_dir))
    )
    assert state["Senkichi/charlie-work"]["escalated"] is False


def test_capacity_escalation_inert_in_dry_run(tmp_path: Path) -> None:
    """A dry-run preview must not write the sidecar or events.db.

    A dry-run that advanced the episode start or fired the escalation would
    consume the rising edge the next real pass needs to see, and would
    escalate a fleet nobody had actually run -- the same side-effect rule the
    allocation pass's hysteresis persist follows.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_state_path = fleet_dir / "state.json"
    now = datetime.now(UTC)

    events = detect_capacity_starvation_escalation(
        _starved_plan(),
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=_escalation_config(starvation_escalation_minutes=1),
        dry_run=True,
        now=now,
    )

    assert events == []
    assert not layout.capacity_starvation_state_path(override=str(fleet_dir)).exists()
    assert query_events(fleet_state_path, kind="runner_capacity_starvation_escalation") == []


def test_capacity_escalation_disabled_emits_nothing(tmp_path: Path) -> None:
    """A disabled section must not write the sidecar or events.db."""
    fleet_dir = tmp_path / "fleet"
    fleet_state_path = fleet_dir / "state.json"
    now = datetime.now(UTC)

    events = detect_capacity_starvation_escalation(
        _starved_plan(),
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=_escalation_config(enabled=False),
        dry_run=False,
        now=now,
    )

    assert events == []
    assert not layout.capacity_starvation_state_path(override=str(fleet_dir)).exists()


def test_capacity_escalation_no_slack_budget_is_not_starvation(tmp_path: Path) -> None:
    """demand > capacity with a fully-subscribed budget is contention, not starvation.

    The spare-budget clause is what makes the signal worth raising: idle
    headroom a bigger registration could fill. A host running at its full
    budget has nothing to re-register into, so escalating would mislead an
    operator into provisioning a runner that would just trade one shortage
    for another.
    """
    # budget=6, running 2 + 4 = 6 -> spare_budget=0.
    plan = _starved_plan(budget=6)
    fleet_dir = tmp_path / "fleet"
    fleet_state_path = fleet_dir / "state.json"

    starved = _starved_repos_from_plan(plan)
    assert starved == []

    events = detect_capacity_starvation_escalation(
        plan,
        fleet_dir_override=str(fleet_dir),
        fleet_state_path=fleet_state_path,
        escalation_config=_escalation_config(starvation_escalation_minutes=1),
        dry_run=False,
        now=datetime.now(UTC),
    )
    assert events == []


# ---------------------------------------------------------------------------
# Operator-digest surface + prologue integration tests
# ---------------------------------------------------------------------------


def test_capacity_escalation_surfaces_in_operator_digest() -> None:
    """The escalation event maps to an ERROR AttentionEntry naming the repo.

    The digest is the surface operators actually read; an escalation that
    lands only in events.db is invisible capability loss -- the exact failure
    shape this issue exists to close. The entry is occurrence-style (the
    detector is edge-triggered), so it must pass through the digest's
    cross-pass dedup unchanged rather than be collapsed.
    """
    events = [
        {
            "repo_key": "fleet",
            "type": "runner_capacity_starvation_escalation",
            "repo": "Senkichi/charlie-work",
            "demand": 5,
            "capacity": 2,
            "running": 2,
            "spare_budget": 2,
            "reason": "Senkichi/charlie-work: CI demand 5 exceeds 2 runners",
        }
    ]
    digest = _build_fleet_attention_digest(events)
    assert len(digest.transitions) == 1
    entry = digest.transitions[0]
    assert entry.health == "ERROR"
    assert entry.adapter_kind == "Senkichi/charlie-work"
    assert "exceeds 2 runners" in (entry.last_log_line or "")


def test_capacity_escalation_wired_into_allocation_prologue(tmp_path: Path) -> None:
    """The prologue must drive the detector end-to-end on a real pass.

    Unit tests cover the detector in isolation; this confirms the prologue
    actually invokes it after a successful allocation pass (not after an error
    or skip), threads the fleet-level state path, and surfaces the escalation
    in the returned event list that feeds the fleet digest.
    """
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(repo / ".var")}},
    )
    plan = _starved_plan(
        repo="owner/anchor", budget=8, demand=5, capacity=2, running=2, other_running=4
    )
    result = AllocationPassResult(ok=True, plan=plan)

    config = OrchestratorConfig(
        runner_allocation=RunnerAllocationConfig(enabled=True, managed_root="C:/actions-runners"),
        runner_scaling=RunnerScalingConfig(managed_root="C:/fallback-root"),
        runner_capacity_escalation=_escalation_config(starvation_escalation_minutes=10),
    )

    # Pass 1: arms the window, no escalation yet. No runner_allocation event
    # either -- the plan moved nothing and carried no notes, so a balanced
    # host stays quiet (the escalation arming is a sidecar write, not an event).
    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=result),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(str(fleet_dir), config, dry_run=False)
    assert events == []

    # Pre-seed the sidecar so pass 2 is past the window (simulating the
    # episode having started 11 minutes ago), then run the prologue again.
    state_path = layout.capacity_starvation_state_path(override=str(fleet_dir))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "generated_at": _iso(datetime.now(UTC) - timedelta(minutes=11)),
                "repos": {
                    "owner/anchor": {
                        "starved_since": _iso(datetime.now(UTC) - timedelta(minutes=11)),
                        "escalated": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=result),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(str(fleet_dir), config, dry_run=False)

    types = [e["type"] for e in events]
    assert "runner_capacity_starvation_escalation" in types
    esc = next(e for e in events if e["type"] == "runner_capacity_starvation_escalation")
    assert esc["repo"] == "owner/anchor"

    # Durable record in the fleet-level events.db.
    rows = query_events(fleet_dir / "state.json", kind="runner_capacity_starvation_escalation")
    assert len(rows) == 1
    assert rows[0]["repo"] == "owner/anchor"


def test_capacity_escalation_prologue_inert_when_section_disabled(tmp_path: Path) -> None:
    """A disabled escalation section must not write the sidecar via the prologue."""
    fleet_dir = tmp_path / "fleet"
    repo = _make_repo(tmp_path, "anchor", api_worker=None)
    _make_fleet_json(
        tmp_path,
        fleet_dir,
        {"owner/anchor": {"repo_root": str(repo), "state_dir": str(repo / ".var")}},
    )
    plan = _starved_plan(repo="owner/anchor")
    result = AllocationPassResult(ok=True, plan=plan)
    config = OrchestratorConfig(
        runner_allocation=RunnerAllocationConfig(enabled=True, managed_root="C:/actions-runners"),
        runner_scaling=RunnerScalingConfig(managed_root="C:/fallback-root"),
        runner_capacity_escalation=_escalation_config(enabled=False),
    )

    with (
        patch("charlie_work.fleet_dispatch.run_allocation_pass", return_value=result),
        patch("charlie_work.fleet_dispatch.GitHub"),
    ):
        events = _run_fleet_allocation_prologue(str(fleet_dir), config, dry_run=False)

    assert all(e["type"] != "runner_capacity_starvation_escalation" for e in events)
    assert not layout.capacity_starvation_state_path(override=str(fleet_dir)).exists()
