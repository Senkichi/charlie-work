"""Tests for the host-wide ``dashboard:`` config section (ADR-0008, ADR-0007)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from charlie_work import layout
from charlie_work.config import (
    ConfigError,
    DashboardConfig,
    OrchestratorConfig,
    build_config_from_data,
    load_config,
)
from charlie_work.global_config import load_layered_config


def test_defaults_on_a_bare_config() -> None:
    cfg = build_config_from_data({}).dashboard
    assert cfg == DashboardConfig(
        enabled=True,
        host="127.0.0.1",
        port=8765,
        poll_interval_seconds=20,
        collector_interval_seconds=30,
        rollup_interval_seconds=120,
    )


def test_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DashboardConfig().port = 1  # type: ignore[misc]


def test_overrides_parse_through() -> None:
    cfg = build_config_from_data(
        {
            "dashboard": {
                "enabled": False,
                "host": "0.0.0.0",
                "port": 9000,
                "poll_interval_seconds": 5,
            }
        }
    ).dashboard
    assert (cfg.enabled, cfg.host, cfg.port, cfg.poll_interval_seconds) == (
        False,
        "0.0.0.0",
        9000,
        5,
    )
    assert cfg.collector_interval_seconds == 30


@pytest.mark.parametrize(
    "key, value",
    [
        ("port", 0),
        ("port", 65536),
        ("port", -1),
        ("port", "8765"),
        ("port", True),
        ("host", ""),
        ("host", "  "),
        ("host", 5),
        ("enabled", "yes"),
        ("enabled", 1),
        ("poll_interval_seconds", 0),
        ("collector_interval_seconds", 0),
        ("rollup_interval_seconds", -5),
        ("rollup_interval_seconds", 1.5),
    ],
)
def test_invalid_values_are_rejected(key: str, value: object) -> None:
    with pytest.raises(ConfigError, match=rf"^dashboard\.{key}: "):
        build_config_from_data({"dashboard": {key: value}})


def test_port_bounds_are_inclusive() -> None:
    assert build_config_from_data({"dashboard": {"port": 1}}).dashboard.port == 1
    assert build_config_from_data({"dashboard": {"port": 65535}}).dashboard.port == 65535


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"^dashboard: "):
        build_config_from_data({"dashboard": {"prot": 1}})


def test_per_repo_layer_rejects_the_host_wide_section(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "orchestrator.config.yaml").write_text(
        "dashboard:\n  port: 9000\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="host-wide only"):
        load_layered_config(repo_root, fleet_dir_override=str(fleet))


def test_global_layer_accepts_the_section(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text("dashboard:\n  port: 9000\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "orchestrator.config.yaml").write_text("{}\n", encoding="utf-8")
    cfg = load_layered_config(repo_root, fleet_dir_override=str(fleet))
    assert isinstance(cfg, OrchestratorConfig)
    assert cfg.dashboard.port == 9000
    assert cfg.dashboard.host == "127.0.0.1"


def test_single_file_load_config_parses_the_section(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("dashboard:\n  enabled: false\n", encoding="utf-8")
    assert load_config(path).dashboard.enabled is False


def test_dashboard_db_path_honours_fleet_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "f"))
    assert layout.dashboard_db_path() == tmp_path / "f" / "dashboard.db"
    assert (
        layout.dashboard_db_path(override=str(tmp_path / "o")) == tmp_path / "o" / "dashboard.db"
    )
