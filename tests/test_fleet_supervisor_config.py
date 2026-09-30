"""Tests for the fleet-scoped supervisor config section (issue #1978).

Issue #1964's step 1 moves the knobs only the cross-repo fleet supervisor
daemon reads out of ``SupervisorConfig`` into ``FleetSupervisorConfig``
under the ``fleet_supervisor:`` section. During the migration window each
moved key is also honored from its legacy ``supervisor.<key>`` location --
registered in ``DEPRECATED_CONFIG_KEYS`` so reads emit
``config_key_deprecated_read`` and the retirement sweep arms removal issue
#1979 -- with ``fleet_supervisor.<key>`` winning and a disagreement between
the two locations failing the load instead of silently picking one.

Shared helpers live in ``tests/_config_deprecations_fixtures.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from charlie_work import layout
from charlie_work.config import (
    ConfigError,
    FleetSupervisorConfig,
    SupervisorConfig,
    build_config_from_data,
    known_config_sections,
    load_config,
)
from charlie_work.config_deprecations import DEPRECATED_CONFIG_KEYS
from charlie_work.fleet_supervisor_config import FLEET_SUPERVISOR_SECTION
from charlie_work.global_config import load_layered_config
from charlie_work.instrumentation import query_events
from _config_deprecations_fixtures import _repo_state_path, _write_repo_config

_MOVED_KEYS = {f.name for f in dataclasses.fields(FleetSupervisorConfig)}


# ---------------------------------------------------------------------------
# Dataclass shape
# ---------------------------------------------------------------------------


def test_fleet_supervisor_config_is_frozen() -> None:
    """FleetSupervisorConfig is a frozen dataclass like every config object."""
    cfg = FleetSupervisorConfig()
    assert dataclasses.is_dataclass(cfg)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        cfg.zero_pass_alarm = 9  # type: ignore[misc]


def test_fleet_supervisor_config_defaults() -> None:
    """Defaults are unchanged from the pre-move SupervisorConfig fields."""
    cfg = FleetSupervisorConfig()
    assert cfg.max_pass_runtime_seconds == 1800
    assert cfg.self_deploy_failure_alarm == 3
    assert cfg.self_deploy_pull_ci_fleet is False
    assert cfg.zero_pass_alarm == 3
    assert cfg.wedge_kill_loop_alarm == 3
    assert cfg.dependency_sync_starvation_seconds == 14400
    assert cfg.fleet_lane_concurrency == 8
    assert cfg.reap_sweep_interval_seconds == 300


def test_supervisor_config_no_longer_carries_fleet_knobs() -> None:
    """The moved fields are gone from SupervisorConfig; the shared ones stay."""
    names = {f.name for f in dataclasses.fields(SupervisorConfig)}
    assert _MOVED_KEYS.isdisjoint(names)
    assert names == {
        "poll_interval_seconds",
        "full_pass_interval_seconds",
        "active_cooldown_seconds",
        "max_runtime_minutes",
    }


def test_fleet_supervisor_is_a_known_top_level_section() -> None:
    """``fleet_supervisor:`` is accepted at the top level of a config file."""
    assert FLEET_SUPERVISOR_SECTION in known_config_sections()


def test_orchestrator_config_exposes_fleet_supervisor() -> None:
    config = load_config()
    assert isinstance(config.fleet_supervisor, FleetSupervisorConfig)


# ---------------------------------------------------------------------------
# Deprecation registry
# ---------------------------------------------------------------------------


def test_every_moved_key_is_registered_for_removal_issue_1979() -> None:
    """Each relocated key is a ``supervisor.<key>`` -> ``fleet_supervisor.<key>``
    deprecation entry; deriving the expectation from the dataclass keeps the
    assertion in lockstep with any future moved key."""
    moved = {
        (entry.section, entry.key): entry
        for entry in DEPRECATED_CONFIG_KEYS
        if entry.section == "supervisor"
    }
    assert set(moved) == {("supervisor", key) for key in _MOVED_KEYS}
    for key in _MOVED_KEYS:
        entry = moved[("supervisor", key)]
        assert entry.replacement == f"{FLEET_SUPERVISOR_SECTION}.{key}"
        assert entry.removal_issue == 1979


# ---------------------------------------------------------------------------
# Parsing: new location, defaults, unknown keys, types
# ---------------------------------------------------------------------------


def test_fleet_supervisor_parses_custom_values(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
fleet_supervisor:
  max_pass_runtime_seconds: 900
  self_deploy_failure_alarm: 5
  self_deploy_pull_ci_fleet: true
  zero_pass_alarm: 7
  wedge_kill_loop_alarm: 2
  dependency_sync_starvation_seconds: 600
  fleet_lane_concurrency: 4
  reap_sweep_interval_seconds: 90
""",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.fleet_supervisor.max_pass_runtime_seconds == 900
    assert config.fleet_supervisor.self_deploy_failure_alarm == 5
    assert config.fleet_supervisor.self_deploy_pull_ci_fleet is True
    assert config.fleet_supervisor.zero_pass_alarm == 7
    assert config.fleet_supervisor.wedge_kill_loop_alarm == 2
    assert config.fleet_supervisor.dependency_sync_starvation_seconds == 600
    assert config.fleet_supervisor.fleet_lane_concurrency == 4
    assert config.fleet_supervisor.reap_sweep_interval_seconds == 90


def test_fleet_supervisor_unknown_key_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "fleet_supervisor:\n  zero_pass_alarm: 7\n  unknown_key: 99\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="fleet_supervisor.*unknown_key"):
        load_config(config_file)


def test_fleet_supervisor_wrong_type_raises(tmp_path: Path) -> None:
    """New-location int fields reject non-ints (same check the supervisor
    section ran before the move)."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        'fleet_supervisor:\n  fleet_lane_concurrency: "not-an-int"\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ConfigError,
        match=r"^fleet_supervisor\.fleet_lane_concurrency: expected int, got 'not-an-int' \(str\)$",
    ):
        load_config(config_file)


def test_fleet_supervisor_bool_key_rejects_non_bool(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        'fleet_supervisor:\n  self_deploy_pull_ci_fleet: "not-a-bool"\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ConfigError,
        match=r"^fleet_supervisor\.self_deploy_pull_ci_fleet: expected bool, got 'not-a-bool' \(str\)$",
    ):
        load_config(config_file)


@pytest.mark.parametrize("key", sorted(_MOVED_KEYS))
def test_legacy_supervisor_location_still_parses(tmp_path: Path, key: str) -> None:
    """Every moved key is still honored from ``supervisor.<key>``."""
    value = "true" if key == "self_deploy_pull_ci_fleet" else "7"
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        f"supervisor:\n  {key}: {value}\n",
        encoding="utf-8",
    )
    config = load_config(config_file)
    expected: object = True if key == "self_deploy_pull_ci_fleet" else 7
    assert getattr(config.fleet_supervisor, key) == expected


def test_legacy_wrong_type_still_raises(tmp_path: Path) -> None:
    """A moved key under ``supervisor:`` is type-checked at its own location."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        'supervisor:\n  dependency_sync_starvation_seconds: "not-an-int"\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ConfigError,
        match=r"^supervisor\.dependency_sync_starvation_seconds: expected int, got 'not-an-int' \(str\)$",
    ):
        load_config(config_file)


def test_surviving_supervisor_keys_parse_alongside_moved_ones(tmp_path: Path) -> None:
    """The per-repo cadence knobs still live under ``supervisor:`` and a
    moved key in the same section does not break their parsing."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
supervisor:
  poll_interval_seconds: 10
  max_pass_runtime_seconds: 900
""",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.supervisor.poll_interval_seconds == 10
    assert config.fleet_supervisor.max_pass_runtime_seconds == 900


# ---------------------------------------------------------------------------
# Precedence and conflict
# ---------------------------------------------------------------------------


def test_new_location_wins_over_legacy(tmp_path: Path) -> None:
    """When both locations agree the value parses once, from the new one."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
supervisor:
  zero_pass_alarm: 5
fleet_supervisor:
  zero_pass_alarm: 5
""",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.fleet_supervisor.zero_pass_alarm == 5


def test_conflicting_locations_raise_config_error(tmp_path: Path) -> None:
    """Different values in the two locations fail the load, naming both."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
supervisor:
  zero_pass_alarm: 5
fleet_supervisor:
  zero_pass_alarm: 9
""",
        encoding="utf-8",
    )
    with pytest.raises(
        ConfigError,
        match="zero_pass_alarm.*fleet_supervisor.*supervisor",
    ):
        load_config(config_file)


def test_build_config_from_data_does_not_mutate_caller_dict() -> None:
    """Legacy-key hoisting happens on the function's private deepcopy."""
    data = {"supervisor": {"zero_pass_alarm": 5, "poll_interval_seconds": 10}}
    config = build_config_from_data(data)
    assert config.fleet_supervisor.zero_pass_alarm == 5
    assert data == {"supervisor": {"zero_pass_alarm": 5, "poll_interval_seconds": 10}}


# ---------------------------------------------------------------------------
# Deprecated-read telemetry and layering
# ---------------------------------------------------------------------------


def test_legacy_read_emits_deprecated_key_event(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(repo, "supervisor:\n  zero_pass_alarm: 5\n")
    load_config(repo / "orchestrator.config.yaml")

    rows = query_events(_repo_state_path(repo), kind="config_key_deprecated_read")
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["section"] == "supervisor"
    assert payload["key"] == "zero_pass_alarm"
    assert payload["replacement"] == "fleet_supervisor.zero_pass_alarm"
    assert payload["issue_number"] == 1979


def test_new_location_read_emits_no_event(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(repo, "fleet_supervisor:\n  zero_pass_alarm: 5\n")
    load_config(repo / "orchestrator.config.yaml")

    assert query_events(_repo_state_path(repo), kind="config_key_deprecated_read") == []


def test_layered_config_honors_legacy_key_from_global_layer(tmp_path: Path) -> None:
    """A legacy key in the fleet layer merges into the fleet-scoped value and
    is reported against the layer file it was read from."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / layout.GLOBAL_CONFIG_FILENAME).write_text(
        "supervisor:\n  fleet_lane_concurrency: 4\n", encoding="utf-8"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(repo, "supervisor:\n  poll_interval_seconds: 10\n")

    config = load_layered_config(repo, fleet_dir_override=str(fleet_dir))

    assert config.fleet_supervisor.fleet_lane_concurrency == 4
    assert config.supervisor.poll_interval_seconds == 10
    fleet_rows = query_events(layout.state_file_path(fleet_dir), kind="config_key_deprecated_read")
    assert len(fleet_rows) == 1
    assert fleet_rows[0]["payload"]["key"] == "fleet_lane_concurrency"
    assert fleet_rows[0]["payload"]["source"] == str(fleet_dir / layout.GLOBAL_CONFIG_FILENAME)


def test_layered_config_rejects_per_repo_fleet_supervisor(tmp_path: Path) -> None:
    """``fleet_supervisor`` is host-wide only, same rule as
    ``runner_allocation``/``runner_capacity_escalation``: every knob on it
    belongs to the one fleet supervisor daemon, so a per-repo config
    declaring the section is rejected outright rather than allowed to
    shadow the operator's global values."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / layout.GLOBAL_CONFIG_FILENAME).write_text(
        "fleet_supervisor:\n  zero_pass_alarm: 9\n", encoding="utf-8"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(repo, "fleet_supervisor:\n  zero_pass_alarm: 5\n")

    with pytest.raises(ConfigError, match="host-wide only"):
        load_layered_config(repo, fleet_dir_override=str(fleet_dir))


def test_layered_config_cross_layer_disagreement_resolves_per_layer(
    tmp_path: Path,
) -> None:
    """A ``fleet_supervisor.<key>`` in the global layer disagreeing with a
    legacy ``supervisor.<key>`` in the repo layer is NOT a merged-view
    conflict: each layer resolves new>legacy on its own before the merge,
    so the repo layer wins the disputed key by the ordinary repo-over-global
    rule -- and the rest of the global layer is NOT discarded over it (the
    pre-fix outcome this replaces: the merged-view conflict tripped the #665
    rescue and dropped every global section)."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    global_path = fleet_dir / layout.GLOBAL_CONFIG_FILENAME
    global_path.write_text(
        "fleet_supervisor:\n  zero_pass_alarm: 9\n"
        "notify:\n  enabled: true\n"
        "runner_scaling:\n  min_runners: 4\n",
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    repo_config = _write_repo_config(repo, "supervisor:\n  zero_pass_alarm: 5\n")

    config = load_layered_config(repo, fleet_dir_override=str(fleet_dir))

    # The disputed key resolves deterministically: the repo layer wins, like
    # every other repo-layer key.
    assert config.fleet_supervisor.zero_pass_alarm == 5
    # The global layer's other sections survive the disagreement.
    assert config.notify.enabled is True
    assert config.runner_scaling.min_runners == 4
    # Both layers were read -- no rescue discard.
    assert config.sources == (str(global_path), str(repo_config))


def test_layered_config_same_file_conflict_in_global_layer_raises(
    tmp_path: Path,
) -> None:
    """Both spellings disagreeing inside the *global* file is still the
    same-file ConfigError -- per-layer resolution happens before the merge,
    so a conflicting global file cannot be laundered into a cross-layer
    repo override."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / layout.GLOBAL_CONFIG_FILENAME).write_text(
        "supervisor:\n  zero_pass_alarm: 5\nfleet_supervisor:\n  zero_pass_alarm: 9\n",
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(repo, "supervisor:\n  poll_interval_seconds: 10\n")

    with pytest.raises(ConfigError, match="zero_pass_alarm.*fleet_supervisor.*supervisor"):
        load_layered_config(repo, fleet_dir_override=str(fleet_dir))


def test_layered_config_repo_legacy_key_folds_and_still_emits(
    tmp_path: Path,
) -> None:
    """A repo layer's legacy ``supervisor.<key>`` folds into its effective
    ``fleet_supervisor`` mapping AND still emits the deprecation read -- the
    fold works on a copy so ``emit_deprecated_key_reads`` sees the raw file's
    legacy spelling. The ``notify`` assertion pins that the value came from
    the merged load: without the per-layer fold, the same-file-looking
    conflict trips the #665 rescue, which still lands this key at 5 (the
    repo-only reload adopts it) but silently drops the global layer's other
    sections."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / layout.GLOBAL_CONFIG_FILENAME).write_text(
        "fleet_supervisor:\n  wedge_kill_loop_alarm: 9\nnotify:\n  enabled: true\n",
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    repo_config = _write_repo_config(repo, "supervisor:\n  wedge_kill_loop_alarm: 5\n")

    config = load_layered_config(repo, fleet_dir_override=str(fleet_dir))

    assert config.fleet_supervisor.wedge_kill_loop_alarm == 5
    assert config.notify.enabled is True
    rows = query_events(_repo_state_path(repo), kind="config_key_deprecated_read")
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["section"] == "supervisor"
    assert payload["key"] == "wedge_kill_loop_alarm"
    assert payload["source"] == str(repo_config)
