"""``runner_allocation.deferred_demand`` (spec B1, interface I-1) via ci-fleet 0.8.0.

``deferred_demand: []`` is the kill switch (spec A5): no deferred sets, today's
demand accounting.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import pytest
import yaml
from charlie_work.config import ConfigError, OrchestratorConfig, build_config_from_data
from charlie_work.config_validation import ConstructionError
from charlie_work.global_config import load_layered_config
from charlie_work.instrumentation import _LEVEL_BY_KIND
from ci_fleet.config import DeferredDemand

_VALID_ENTRY = {"labels": ["cap-py"], "after_seconds": 180}
_BAD_AFTER_SECONDS = [0, -5, "180", 1.5]
_REPO_YAML = "dispatch:\n  default_limit: 2\n"


def _deferred(entries: object) -> object:
    data = {"runner_allocation": {"deferred_demand": entries}}
    return build_config_from_data(data).runner_allocation.deferred_demand


def _load_layered(tmp_path: Path, entry: dict[str, object]) -> OrchestratorConfig:
    """Load a fleet-global layer holding one ``deferred_demand`` entry over a valid repo layer.

    The repo layer is non-empty on purpose: the #665 rescue only applies when there is a
    per-repo config to rescue onto, so an empty one would make every error propagate.
    """
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    global_yaml = yaml.safe_dump({"runner_allocation": {"deferred_demand": [entry]}})
    (fleet / "config.yaml").write_text(global_yaml, encoding="utf-8")
    (repo / "orchestrator.config.yaml").write_text(_REPO_YAML, encoding="utf-8")
    return load_layered_config(repo, None, fleet_dir_override=str(fleet))


def test_default_is_empty() -> None:
    assert OrchestratorConfig().runner_allocation.deferred_demand == ()


def test_entries_parse_with_lowercased_labels() -> None:
    parsed = _deferred([{"labels": ["Self-Hosted", "cap-py"], "after_seconds": 180}])
    assert parsed == (
        DeferredDemand(labels=frozenset({"self-hosted", "cap-py"}), after_seconds=180),
    )


def test_empty_list_is_the_kill_switch() -> None:
    assert _deferred([]) == ()


@pytest.mark.parametrize("bad", _BAD_AFTER_SECONDS)
def test_after_seconds_must_be_a_positive_int(bad: object) -> None:
    with pytest.raises(
        ConfigError, match=r"runner_allocation\.deferred_demand\[0\]\.after_seconds"
    ):
        _deferred([{"labels": ["cap-py"], "after_seconds": bad}])


def test_unknown_entry_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"deferred_demand"):
        _deferred([{"labels": ["cap-py"], "after_seconds": 180, "age": 1}])


# The layered-load outcome of a bad entry in the host-wide global layer is split on purpose
# (see the comment on ``OrchestratorConfig.runner_allocation``): ``after_seconds`` has a
# use-site ``Entries`` rule, so it is a rescuable ``FieldError``; ``labels`` has none, so
# ci-fleet's own constructor rejects it as a loud ``ConstructionError``.
def test_valid_deferred_demand_in_the_global_layer_survives_the_layered_load(
    tmp_path: Path,
) -> None:
    # Positive control for the two tests below: same fixture, only the entry differs, so
    # their rescued / loud outcomes come from the bad value and not from the fixture.
    config = _load_layered(tmp_path, _VALID_ENTRY)
    assert [Path(s).name for s in config.sources] == ["config.yaml", "orchestrator.config.yaml"]
    assert config.runner_allocation.deferred_demand == (
        DeferredDemand(labels=frozenset({"cap-py"}), after_seconds=180),
    )


@pytest.mark.parametrize("bad", _BAD_AFTER_SECONDS)
def test_bad_after_seconds_in_the_global_layer_is_rescued_onto_the_repo_layer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, bad: object
) -> None:
    with caplog.at_level(logging.WARNING, logger="charlie_work.global_config"):
        config = _load_layered(tmp_path, {**_VALID_ENTRY, "after_seconds": bad})

    # The global layer is discarded: only the repo layer contributes, so the host-wide
    # knob falls back to its default (the kill-switch value).
    assert [Path(s).name for s in config.sources] == ["orchestrator.config.yaml"]
    assert config.dispatch.default_limit == 2
    assert config.runner_allocation.deferred_demand == ()
    # ...but not silently: one warning names the discarded layer and the failing key.
    warnings = [
        r.getMessage() for r in caplog.records if "global layer was discarded" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "runner_allocation.deferred_demand[0].after_seconds" in warnings[0]


@pytest.mark.parametrize(
    "bad",
    [[], "cap-py", [1]],
    ids=["empty", "bare-string", "non-string-label"],
)
def test_bad_labels_in_the_global_layer_stays_loud(tmp_path: Path, bad: object) -> None:
    with pytest.raises(ConstructionError, match=r"runner_allocation\.deferred_demand\[0\]"):
        _load_layered(tmp_path, {**_VALID_ENTRY, "labels": bad})


def test_installed_ci_fleet_ships_the_deferred_demand_spill_module() -> None:
    # Capability, not a version string: ``ci_fleet.spill`` is the spill-runner surface
    # ``deferred_demand`` configures, absent from ci-fleet before 0.8.0 and present in every
    # later release, so this fails on a too-old install and survives future bumps.
    # ``tests/test_ci_fleet_floor_guard.py`` keeps pyproject's floor, uv.lock and the
    # installed ci-fleet equal, so the installed module is the floor's capability.
    importlib.import_module("ci_fleet.spill")


@pytest.mark.parametrize("kind", ["deferred_demand_spilled", "platform_feature_skipped"])
def test_ci_fleet_event_kinds_are_registered_as_info(kind: str) -> None:
    # ``.get``, not ``_classify_level``: the classifier defaults an unregistered kind to
    # "info", so asserting "info" through it could never fail.
    assert _LEVEL_BY_KIND.get(kind) == "info"
