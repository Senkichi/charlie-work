"""``load_config`` validation of merge-queue fields: ``mergequeue_label``,
``mergequeue_wedge_hours``, and the queue-bot login.

Split out of ``tests/test_charlie_work.py`` (issue #1552, Track-1
wave 6/8).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import (
    ConfigError,
    load_config,
)


def test_load_config_rejects_non_string_mergequeue_label(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  mergequeue_label: 123
"""
    )
    with pytest.raises(ConfigError, match="mergequeue_label.*must be a string"):
        load_config(config_file)


def test_load_config_rejects_empty_mergequeue_label(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  mergequeue_label: "   "
"""
    )
    with pytest.raises(ConfigError, match="mergequeue_label.*must not be empty"):
        load_config(config_file)


def test_load_config_accepts_valid_mergequeue_label(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  mergequeue_label: mergequeue
"""
    )
    config = load_config(config_file)
    assert config.auto_merge.mergequeue_label == "mergequeue"


def test_load_config_strips_mergequeue_label_whitespace(tmp_path: Path) -> None:
    """Adversarial review finding #3: mergequeue_label.strip() is used only to
    validate truthiness, but the unstripped value must not thread verbatim
    into `gh pr edit --add-label` — surrounding whitespace is not a valid (or
    intended) part of a GitHub label name."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  mergequeue_label: "  mergequeue  "
"""
    )
    config = load_config(config_file)
    assert config.auto_merge.mergequeue_label == "mergequeue"


def test_load_config_parses_mergequeue_wedge_hours(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  mergequeue_wedge_hours: 6
"""
    )
    config = load_config(config_file)
    assert config.auto_merge.mergequeue_wedge_hours == 6.0


def test_load_config_rejects_negative_mergequeue_wedge_hours(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  mergequeue_wedge_hours: -1
"""
    )
    with pytest.raises(ConfigError, match="must not be negative"):
        load_config(config_file)


def test_load_config_rejects_non_number_mergequeue_wedge_hours(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  mergequeue_wedge_hours: "soon"
"""
    )
    with pytest.raises(ConfigError, match="must be a number"):
        load_config(config_file)


def test_load_config_rejects_non_string_queue_bot_login(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  queue_bot_login: 123
"""
    )
    with pytest.raises(ConfigError, match="queue_bot_login.*must be a string"):
        load_config(config_file)


def test_load_config_rejects_empty_queue_bot_login(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  queue_bot_login: "   "
"""
    )
    with pytest.raises(ConfigError, match="queue_bot_login.*must not be empty"):
        load_config(config_file)


def test_load_config_accepts_and_strips_queue_bot_login(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  queue_bot_login: "  aviator-app[bot]  "
"""
    )
    config = load_config(config_file)
    assert config.auto_merge.queue_bot_login == "aviator-app[bot]"
