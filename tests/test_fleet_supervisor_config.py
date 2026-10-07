"""Tests for the fleet-scoped supervisor config section (issue #1978).

Issue #1964's step 1 moved the knobs only the cross-repo fleet supervisor
daemon reads out of ``SupervisorConfig`` into ``FleetSupervisorConfig``
under the ``fleet_supervisor:`` section. Issue #1979 then removed the
migration window: the legacy ``supervisor.<key>`` spellings are no longer
registered in ``DEPRECATED_CONFIG_KEYS`` and no longer parse -- a moved key
still written under ``supervisor:`` fails like any other unknown key.

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
    """Issue #1979: the registrations this leaf name describes were themselves
    the removal target -- the registry must now carry NO ``supervisor.*``
    entry at all, and no entry pointing at this issue.

    The leaf name predates the removal and is kept verbatim: the
    collect-only gate (issue #1538) fails a required check on any leaf-name
    removal, rename included, absent the operator-applied
    ``collect-gate-exempt`` label."""
    assert all(entry.section != "supervisor" for entry in DEPRECATED_CONFIG_KEYS)
    assert all(entry.removal_issue != 1979 for entry in DEPRECATED_CONFIG_KEYS)


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
    """Issue #1979's acceptance: every moved key written under
    ``supervisor.<key>`` is now rejected with the normal unknown-key error.

    The leaf name predates the removal and is kept verbatim for the
    collect-only gate (issue #1538) -- the behavior it names is exactly what
    no longer happens."""
    value = "true" if key == "self_deploy_pull_ci_fleet" else "7"
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        f"supervisor:\n  {key}: {value}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=f"supervisor.*unknown key.*{key}"):
        load_config(config_file)


def test_legacy_wrong_type_still_raises(tmp_path: Path) -> None:
    """A moved key under ``supervisor:`` still raises -- as an unknown key
    now, before any type check can run (issue #1979)."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        'supervisor:\n  dependency_sync_starvation_seconds: "not-an-int"\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ConfigError,
        match=r"^supervisor: expected known keys .*unknown key\(s\) dependency_sync_starvation_seconds",
    ):
        load_config(config_file)


def test_surviving_supervisor_keys_parse_alongside_moved_ones(tmp_path: Path) -> None:
    """The per-repo cadence knobs still live under ``supervisor:`` and parse
    normally; a moved key in the same section is rejected as unknown instead
    of being adopted (issue #1979).

    The leaf name predates the removal and is kept verbatim for the
    collect-only gate (issue #1538)."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "supervisor:\n  poll_interval_seconds: 10\n",
        encoding="utf-8",
    )
    assert load_config(config_file).supervisor.poll_interval_seconds == 10

    config_file.write_text(
        """
supervisor:
  poll_interval_seconds: 10
  max_pass_runtime_seconds: 900
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="supervisor.*unknown key.*max_pass_runtime_seconds"):
        load_config(config_file)


# ---------------------------------------------------------------------------
# Legacy location: both spellings in one file
# ---------------------------------------------------------------------------


def test_new_location_wins_over_legacy(tmp_path: Path) -> None:
    """Both spellings in one file no longer resolve -- the ``supervisor:``
    entry is an unknown key, so the file fails (issue #1979).

    The leaf name predates the removal and is kept verbatim for the
    collect-only gate (issue #1538)."""
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
    with pytest.raises(ConfigError, match="supervisor.*unknown key.*zero_pass_alarm"):
        load_config(config_file)


def test_conflicting_locations_raise_config_error(tmp_path: Path) -> None:
    """Different values in the two locations still fail the load -- the
    legacy spelling is an unknown key now, whatever it is set to."""
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
        match="supervisor.*unknown key.*zero_pass_alarm",
    ):
        load_config(config_file)


def test_build_config_from_data_does_not_mutate_caller_dict() -> None:
    """``build_config_from_data`` keeps its "reads a dict, does not touch it"
    contract: any in-place section work happens on its private deepcopy."""
    data = {"supervisor": {"poll_interval_seconds": 10}}
    config = build_config_from_data(data)
    assert config.supervisor.poll_interval_seconds == 10
    assert data == {"supervisor": {"poll_interval_seconds": 10}}


# ---------------------------------------------------------------------------
# Read-time signal and layering
# ---------------------------------------------------------------------------


def test_legacy_read_emits_deprecated_key_event(tmp_path: Path) -> None:
    """A moved key under ``supervisor:`` is GONE, not deprecated: the load
    fails as unknown and no ``config_key_deprecated_read`` event fires
    (the registry carries no ``supervisor.*`` entry to attribute it to).

    The leaf name predates the removal and is kept verbatim for the
    collect-only gate (issue #1538)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_repo_config(repo, "supervisor:\n  zero_pass_alarm: 5\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(config_path)

    assert query_events(_repo_state_path(repo), kind="config_key_deprecated_read") == []


def test_new_location_read_emits_no_event(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(repo, "fleet_supervisor:\n  zero_pass_alarm: 5\n")
    load_config(repo / "orchestrator.config.yaml")

    assert query_events(_repo_state_path(repo), kind="config_key_deprecated_read") == []


def test_layered_config_honors_legacy_key_from_global_layer(tmp_path: Path) -> None:
    """A legacy ``supervisor.<key>`` in the global layer is an unknown key
    now: it breaks the *merged* load, so the #665 rescue discards the global
    layer and the repo file alone builds -- the fleet knob reverts to its
    default rather than honoring the legacy spelling (issue #1979).

    The leaf name predates the removal and is kept verbatim for the
    collect-only gate (issue #1538)."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / layout.GLOBAL_CONFIG_FILENAME).write_text(
        "supervisor:\n  fleet_lane_concurrency: 4\n", encoding="utf-8"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    repo_config = _write_repo_config(repo, "supervisor:\n  poll_interval_seconds: 10\n")

    config = load_layered_config(repo, fleet_dir_override=str(fleet_dir))

    # The global layer was discarded wholesale: the legacy key is an unknown
    # key, the merged build cannot run, and only the repo file contributes.
    assert config.fleet_supervisor.fleet_lane_concurrency == 8
    assert config.supervisor.poll_interval_seconds == 10
    assert config.sources == (str(repo_config),)
    assert query_events(layout.state_file_path(fleet_dir), kind="config_key_deprecated_read") == []


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
    """A ``fleet_supervisor.<key>`` in the global layer plus a legacy
    ``supervisor.<key>`` in the repo layer is no longer a resolvable
    disagreement: the repo file's legacy spelling is an unknown key, so the
    merged build fails and -- since the repo file cannot build alone either
    -- the ConfigError propagates (issue #1979).

    The leaf name predates the removal and is kept verbatim for the
    collect-only gate (issue #1538)."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / layout.GLOBAL_CONFIG_FILENAME).write_text(
        "fleet_supervisor:\n  zero_pass_alarm: 9\n"
        "notify:\n  enabled: true\n"
        "runner_scaling:\n  min_runners: 4\n",
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(repo, "supervisor:\n  zero_pass_alarm: 5\n")

    with pytest.raises(ConfigError, match="supervisor.*unknown key.*zero_pass_alarm"):
        load_layered_config(repo, fleet_dir_override=str(fleet_dir))


def test_layered_config_same_file_conflict_in_global_layer_raises(
    tmp_path: Path,
) -> None:
    """Both spellings inside the *global* file still fail: the
    ``supervisor:`` entry is an unknown key, so the merged build rejects it.
    The per-repo file is valid, so the #665 rescue then lands the repo-only
    load -- the whole global layer, new-style ``fleet_supervisor`` section
    included, is discarded over the one dead key.

    The leaf name predates the removal and is kept verbatim for the
    collect-only gate (issue #1538)."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / layout.GLOBAL_CONFIG_FILENAME).write_text(
        "supervisor:\n  zero_pass_alarm: 5\nfleet_supervisor:\n  zero_pass_alarm: 9\n",
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    repo_config = _write_repo_config(repo, "supervisor:\n  poll_interval_seconds: 10\n")

    config = load_layered_config(repo, fleet_dir_override=str(fleet_dir))

    assert config.supervisor.poll_interval_seconds == 10
    assert config.fleet_supervisor.zero_pass_alarm == 3  # default; global layer discarded
    assert config.sources == (str(repo_config),)


def test_layered_config_repo_legacy_key_folds_and_still_emits(
    tmp_path: Path,
) -> None:
    """A repo layer's legacy ``supervisor.<key>`` no longer folds anywhere
    and no longer emits a deprecation read: the key is unregistered, so the
    merged build rejects it as unknown, the #665 repo-only retry rejects it
    identically, and the ConfigError propagates (issue #1979).

    The leaf name predates the removal and is kept verbatim for the
    collect-only gate (issue #1538)."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / layout.GLOBAL_CONFIG_FILENAME).write_text(
        "fleet_supervisor:\n  wedge_kill_loop_alarm: 9\nnotify:\n  enabled: true\n",
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(repo, "supervisor:\n  wedge_kill_loop_alarm: 5\n")

    with pytest.raises(ConfigError, match="supervisor.*unknown key.*wedge_kill_loop_alarm"):
        load_layered_config(repo, fleet_dir_override=str(fleet_dir))

    assert query_events(_repo_state_path(repo), kind="config_key_deprecated_read") == []
