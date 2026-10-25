"""``runner_allocation.deferred_demand`` (spec B1, interface I-1) via ci-fleet 0.8.0.

``deferred_demand: []`` is the kill switch (spec A5): no deferred sets, today's
demand accounting.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from charlie_work.config import ConfigError, OrchestratorConfig, build_config_from_data
from ci_fleet.config import DeferredDemand

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _deferred(entries: object) -> object:
    data = {"runner_allocation": {"deferred_demand": entries}}
    return build_config_from_data(data).runner_allocation.deferred_demand


def test_default_is_empty() -> None:
    assert OrchestratorConfig().runner_allocation.deferred_demand == ()


def test_entries_parse_with_lowercased_labels() -> None:
    parsed = _deferred([{"labels": ["Self-Hosted", "cap-py"], "after_seconds": 180}])
    assert parsed == (
        DeferredDemand(labels=frozenset({"self-hosted", "cap-py"}), after_seconds=180),
    )


def test_empty_list_is_the_kill_switch() -> None:
    assert _deferred([]) == ()


@pytest.mark.parametrize("bad", [0, -5, "180", 1.5])
def test_after_seconds_must_be_a_positive_int(bad: object) -> None:
    with pytest.raises(
        ConfigError, match=r"runner_allocation\.deferred_demand\[0\]\.after_seconds"
    ):
        _deferred([{"labels": ["cap-py"], "after_seconds": bad}])


def test_unknown_entry_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"deferred_demand"):
        _deferred([{"labels": ["cap-py"], "after_seconds": 180, "age": 1}])


def test_ci_fleet_floor_is_0_8_0() -> None:
    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert "ci-fleet>=0.8.0" in project["dependencies"]


@pytest.mark.parametrize("kind", ["deferred_demand_spilled", "platform_feature_skipped"])
def test_ci_fleet_0_8_0_event_kinds_have_levels(kind: str) -> None:
    level = Path(__file__).resolve().parent.parent / "src/charlie_work/event_levels"
    assert (level / f"{kind}.level").read_text(encoding="utf-8").splitlines()[0] == "info"
