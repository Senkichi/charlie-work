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


def test_dispatch_config_injected_paths_default_excludes_injected_files() -> None:
    """Issue #381/#400: default injected_paths exclude the in-worktree Claude Code prompt file and writer marker."""
    from charlie_work.config import CLAUDE_CODE_PROMPT_FILENAME, WRITER_MARKER_FILENAME

    config = DispatchConfig()
    assert config.injected_paths == (CLAUDE_CODE_PROMPT_FILENAME, WRITER_MARKER_FILENAME)


def test_claude_code_prompt_filename_in_default_injected_paths() -> None:
    """Issue #381: the Claude Code prompt file is excluded from dirty checks by default."""
    from charlie_work.claude_code import PROMPT_FILENAME
    from charlie_work.config import CLAUDE_CODE_PROMPT_FILENAME

    assert PROMPT_FILENAME is CLAUDE_CODE_PROMPT_FILENAME
    assert CLAUDE_CODE_PROMPT_FILENAME in DispatchConfig().injected_paths


def test_load_config_injected_paths_coerces_list_to_tuple(tmp_path: Path) -> None:
    """Issue #381/#400: injected_paths list in YAML becomes a tuple and the writer marker is always appended."""
    from charlie_work.config import WRITER_MARKER_FILENAME

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
        WRITER_MARKER_FILENAME,
    )


def test_dispatch_config_injected_paths_normalizes_backslashes() -> None:
    """Issue #381/#400: backslash separators in an override are normalized to '/' and the writer marker is appended."""
    from charlie_work.config import WRITER_MARKER_FILENAME

    config = DispatchConfig(injected_paths=[r".devin\prompts\worker.md"])
    assert config.injected_paths == (".devin/prompts/worker.md", WRITER_MARKER_FILENAME)


def test_load_config_injected_paths_normalizes_backslashes(tmp_path: Path) -> None:
    """Issue #381/#400: YAML override with Windows-style separators is normalized and the writer marker is appended."""
    from charlie_work.config import WRITER_MARKER_FILENAME

    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        r"""dispatch:
  injected_paths:
    - '.devin\prompts\worker.md'
""",
    )
    config = load_config(config_file)
    assert config.dispatch.injected_paths == (".devin/prompts/worker.md", WRITER_MARKER_FILENAME)


def test_load_config_review_dispatch_defaults() -> None:
    """ReviewDispatchConfig defaults are safe (disabled, separate dir, cap/backoff)."""
    config_file = Path("nonexistent.yaml")
    config = load_config(config_file)
    assert config.review_dispatch.enabled is False
    assert config.review_dispatch.reviews_dir == ".var/charlie-work/dispatches/reviews"
    assert config.review_dispatch.max_local_review_processes == 0
    assert config.review_dispatch.max_retries == 3
    assert config.review_dispatch.retry_backoff_minutes == 30


def test_load_config_review_dispatch_override(tmp_path: Path) -> None:
    """review_dispatch keys are read from YAML."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review_dispatch:
  enabled: true
  reviews_dir: .var/reviews
  max_local_review_processes: 4
  max_retries: 5
  retry_backoff_minutes: 10
""",
    )
    config = load_config(config_file)
    assert config.review_dispatch.enabled is True
    assert config.review_dispatch.reviews_dir == ".var/reviews"
    assert config.review_dispatch.max_local_review_processes == 4
    assert config.review_dispatch.max_retries == 5
    assert config.review_dispatch.retry_backoff_minutes == 10


def test_load_config_review_dispatch_rejects_invalid_retry_values(tmp_path: Path) -> None:
    """Issue #495: review_dispatch cap/backoff values must be non-negative ints."""
    config_file = tmp_path / "orchestrator.config.yaml"
    for key, value in (
        ("max_retries", "three"),
        ("max_retries", -1),
        ("retry_backoff_minutes", "fast"),
        ("retry_backoff_minutes", -5),
    ):
        _write_config(
            config_file,
            f"""review_dispatch:
  enabled: true
  {key}: {value}
""",
        )
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert key in str(exc_info.value)
