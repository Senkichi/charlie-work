"""Configured provider-key names reach ci_fleet's listener env strip (issue #2902).

These tests go through ``ci_fleet.runners._sanitize_env``, the function that
builds every listener's environment, rather than asserting on the list of names.
A name the list holds but the launch keeps would otherwise pass.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from ci_fleet import env_strip, runners

from charlie_work import layout
from charlie_work.ci_fleet_env_strip import fleet_provider_key_names


def _provider_yaml(key_name: str) -> str:
    return (
        "api_worker:\n"
        "  providers:\n"
        "    test-provider:\n"
        "      base_url: https://example.invalid/anthropic\n"
        f"      api_key_env: {key_name}\n"
        "      model: test-model\n"
        "      input_usd_per_mtok: 3.0\n"
        "      output_usd_per_mtok: 15.0\n"
    )


def _repo(tmp_path: Path, name: str, config_text: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "orchestrator.config.yaml").write_text(config_text, encoding="utf-8")
    return root


def _fleet(tmp_path: Path, *roots: Path, global_config: str | None = None) -> str:
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    registry = {"repos": {f"owner/{root.name}": {"repo_root": str(root)} for root in roots}}
    (fleet / layout.FLEET_REGISTRY_FILENAME).write_text(json.dumps(registry), encoding="utf-8")
    if global_config is not None:
        (fleet / layout.GLOBAL_CONFIG_FILENAME).write_text(global_config, encoding="utf-8")
    return str(fleet)


@pytest.fixture
def restore_source() -> Iterator[None]:
    """The installed source is process-wide, and the seam test reads it."""
    previous = env_strip.get_env_strip_source()
    yield
    assert previous is not None
    env_strip.set_env_strip_source(previous)


def test_every_repos_key_name_leaves_the_listener_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_source: None
) -> None:
    """Names from different repos both go, and case does not protect a name.

    Windows upper-cases ``os.environ`` keys, so a config spelled in mixed case
    has to match the upper-cased variable.
    """
    fleet = _fleet(
        tmp_path,
        _repo(tmp_path, "alpha", _provider_yaml("ALPHA_PROVIDER_KEY")),
        _repo(tmp_path, "beta", _provider_yaml("Beta_Mixed_Key")),
    )
    monkeypatch.setenv("ALPHA_PROVIDER_KEY", "alpha-value")
    monkeypatch.setenv("BETA_MIXED_KEY", "beta-value")
    monkeypatch.setenv("CHARLIE_WORK_TEST_CONTROL", "kept")

    env_strip.set_env_strip_source(lambda: fleet_provider_key_names(fleet))
    env = {name.casefold(): value for name, value in runners._sanitize_env().items()}

    assert "alpha_provider_key" not in env
    assert "beta_mixed_key" not in env
    assert env["charlie_work_test_control"] == "kept"


def test_a_config_that_fails_to_load_keeps_the_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One bad repo costs its own names only, and says so in the log."""
    broken = _repo(tmp_path, "broken", "api_worker: [unterminated\n")
    fleet = _fleet(tmp_path, broken, _repo(tmp_path, "good", _provider_yaml("GOOD_PROVIDER_KEY")))

    with caplog.at_level(logging.WARNING, logger="charlie_work.ci_fleet_env_strip"):
        names = fleet_provider_key_names(fleet)

    assert names == {"GOOD_PROVIDER_KEY"}
    assert str(broken) in caplog.text


def test_the_global_layer_counts_when_no_repo_can_be_read(tmp_path: Path) -> None:
    """A fleet-wide key is still named with an empty or unreadable registry."""
    fleet = _fleet(tmp_path, global_config=_provider_yaml("FLEET_PROVIDER_KEY"))

    assert fleet_provider_key_names(fleet) == {"FLEET_PROVIDER_KEY"}


def test_a_missing_fleet_dir_names_nothing_and_does_not_raise(tmp_path: Path) -> None:
    """A plain single-repo checkout has no fleet dir at all."""
    assert fleet_provider_key_names(str(tmp_path / "no-such-fleet")) == frozenset()
