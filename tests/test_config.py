"""Tests for config.py validation, especially new config keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import ConfigError, DispatchConfig, load_config


def _write_config(config_file: Path, content: str) -> None:
    config_file.write_text(content, encoding="utf-8")


def test_load_config_worktree_mtime_enabled_rejects_non_bool(tmp_path: Path) -> None:
    """Issue #353: watchdog.worktree_mtime_enabled must be a bool."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
watchdog:
  worktree_mtime_enabled: "true"
""",
    )
    with pytest.raises(ConfigError, match="worktree_mtime_enabled.*must be a bool"):
        load_config(config_file)


def test_load_config_worktree_mtime_threshold_minutes_rejects_non_int(tmp_path: Path) -> None:
    """Issue #353: watchdog.worktree_mtime_threshold_minutes must be an int."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
watchdog:
  worktree_mtime_threshold_minutes: "45"
""",
    )
    with pytest.raises(ConfigError, match="worktree_mtime_threshold_minutes.*must be an int"):
        load_config(config_file)


def test_load_config_worktree_mtime_max_depth_rejects_non_int(tmp_path: Path) -> None:
    """Issue #353: watchdog.worktree_mtime_max_depth must be an int."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
watchdog:
  worktree_mtime_max_depth: "4"
""",
    )
    with pytest.raises(ConfigError, match="worktree_mtime_max_depth.*must be an int"):
        load_config(config_file)


def test_load_config_worktree_mtime_exclude_dirs_rejects_non_string_element(
    tmp_path: Path,
) -> None:
    """Issue #353: watchdog.worktree_mtime_exclude_dirs must be a list of strings."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
watchdog:
  worktree_mtime_exclude_dirs:
    - .git
    - 42
""",
    )
    with pytest.raises(
        ConfigError, match="worktree_mtime_exclude_dirs.*must be a list of strings"
    ):
        load_config(config_file)


def test_load_config_worktree_mtime_exclude_dirs_coerces_list_to_tuple(
    tmp_path: Path,
) -> None:
    """Issue #353: watchdog.worktree_mtime_exclude_dirs is a list in YAML and a tuple in config."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
watchdog:
  worktree_mtime_enabled: true
  worktree_mtime_threshold_minutes: 45
  worktree_mtime_max_depth: 4
  worktree_mtime_exclude_dirs:
    - .git
    - .venv
""",
    )
    config = load_config(config_file)
    assert config.watchdog.worktree_mtime_enabled is True
    assert config.watchdog.worktree_mtime_threshold_minutes == 45
    assert config.watchdog.worktree_mtime_max_depth == 4
    assert config.watchdog.worktree_mtime_exclude_dirs == (".git", ".venv")


def test_dispatch_config_injected_paths_derived_from_templates() -> None:
    """Issue #381: default injected_paths are derived from worker/rework templates."""
    config = DispatchConfig()
    assert config.injected_paths == (
        ".devin/prompts/worker.md",
        ".devin/prompts/rework.md",
    )

    custom = DispatchConfig(worker_template="worker_claude_code.md", rework_template="rework.md")
    assert custom.injected_paths == (
        ".devin/prompts/worker_claude_code.md",
        ".devin/prompts/rework.md",
    )


def test_load_config_injected_paths_coerces_list_to_tuple(tmp_path: Path) -> None:
    """Issue #381: injected_paths list in YAML becomes a tuple in DispatchConfig."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
dispatch:
  injected_paths:
    - .orchestrator-prompt.md
    - .devin/prompts/custom.md
""",
    )
    config = load_config(config_file)
    assert config.dispatch.injected_paths == (
        ".orchestrator-prompt.md",
        ".devin/prompts/custom.md",
    )
