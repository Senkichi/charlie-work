"""``main_sync:`` config section (docs/specs/main-sync.md, "post_sync contract")."""

from __future__ import annotations

import pytest

from charlie_work.config import ConfigError, OrchestratorConfig, build_config_from_data
from charlie_work.main_sync_config import MainSyncConfig


def test_defaults_ship_on_with_no_post_sync() -> None:
    cfg = OrchestratorConfig()
    assert cfg.main_sync == MainSyncConfig()
    assert cfg.main_sync.enabled is True
    assert cfg.main_sync.post_sync_command == ()
    assert cfg.main_sync.post_sync_when_paths == ()
    assert cfg.main_sync.post_sync_timeout_seconds == 300


def test_yaml_lists_become_tuples() -> None:
    cfg = build_config_from_data(
        {
            "main_sync": {
                "enabled": False,
                "post_sync_command": ["uv", "sync", "--locked"],
                "post_sync_when_paths": ["uv.lock", "pyproject.toml"],
                "post_sync_timeout_seconds": 120,
            }
        }
    )
    assert cfg.main_sync == MainSyncConfig(
        enabled=False,
        post_sync_command=("uv", "sync", "--locked"),
        post_sync_when_paths=("uv.lock", "pyproject.toml"),
        post_sync_timeout_seconds=120,
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("post_sync_command", "uv sync --locked"),  # a shell string, not argv
        ("post_sync_command", ["uv", 3]),
        ("post_sync_command", None),
        ("post_sync_when_paths", "uv.lock"),
        ("post_sync_timeout_seconds", 0),
        ("post_sync_timeout_seconds", "300"),
        ("enabled", "yes"),
        ("zzz_unknown", 1),
    ],
)
def test_bad_values_are_rejected(key: str, value: object) -> None:
    with pytest.raises(ConfigError):
        build_config_from_data({"main_sync": {key: value}})
