"""``load_config`` validation of the ``watchdog`` section: terminal error
markers and the additive new-field defaults.

Split out of ``tests/test_charlie_work.py`` (issue #1552, Track-1
wave 6/8).
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.config import load_config


def test_config_accepts_watchdog_terminal_error_markers(tmp_path: Path) -> None:
    """A YAML block with terminal_error_markers loads correctly."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  enabled: true
  stall_minutes: 20
  terminal_error_markers:
    - "Error: A tool was rejected"
    - "Error: Agent error:"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.watchdog.enabled is True
    assert config.watchdog.stall_minutes == 20
    assert config.watchdog.terminal_error_markers == (
        "Error: A tool was rejected",
        "Error: Agent error:",
    )


def test_config_defaults_watchdog_terminal_error_markers(tmp_path: Path) -> None:
    """A YAML block without terminal_error_markers uses the default."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  enabled: true
  stall_minutes: 20
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.watchdog.enabled is True
    assert config.watchdog.stall_minutes == 20
    assert config.watchdog.terminal_error_markers == (
        "Error: A tool was rejected",
        "Error: Agent error:",
    )


def test_config_rejects_invalid_watchdog_terminal_error_markers_type(tmp_path: Path) -> None:
    """terminal_error_markers must be a list of strings."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  terminal_error_markers: "not a list"
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for invalid terminal_error_markers type")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid terminal_error_markers type")

    assert "section 'watchdog'" in message
    assert "terminal_error_markers" in message
    assert "must be a list" in message


def test_config_rejects_invalid_watchdog_terminal_error_markers_element_type(
    tmp_path: Path,
) -> None:
    """terminal_error_markers elements must be strings."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  terminal_error_markers:
    - "valid string"
    - 123
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError(
            "expected ConfigError for invalid terminal_error_markers element type"
        )
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError(
            "expected ConfigError for invalid terminal_error_markers element type"
        )

    assert "section 'watchdog'" in message
    assert "terminal_error_markers" in message
    assert "must be a list of strings" in message


def test_config_rejects_unknown_watchdog_key(tmp_path: Path) -> None:
    """Unknown keys under watchdog raise ConfigError listing valid keys."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  enabled: true
  bad_key: value
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for unknown watchdog key")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown watchdog key")

    assert "section 'watchdog'" in message
    assert "bad_key" in message
    # Should list valid keys
    assert "enabled" in message
    assert "stall_minutes" in message
    assert "terminal_error_markers" in message
    assert "cost_budget_usd" in message
    assert "token_budget" in message
    assert "cost_budget_action" in message
    assert "wall_clock_minutes" in message
    assert "wall_clock_kill" in message
    assert "loop_stall_multiplier" in message
    assert "loop_kill" in message


def test_config_watchdog_new_fields_have_defaults(tmp_path: Path) -> None:
    """New watchdog fields (wall_clock_minutes, wall_clock_kill, loop_stall_multiplier, loop_kill) have defaults."""
    config_path = tmp_path / "orchestrator.config.yaml"
    # Config without the new fields (pre-#162 style)
    config_path.write_text(
        """watchdog:
  enabled: true
  stall_minutes: 20
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.watchdog.enabled is True
    assert config.watchdog.stall_minutes == 20
    assert config.watchdog.cost_budget_usd is None
    assert config.watchdog.token_budget is None
    assert config.watchdog.cost_budget_action == "warn"
    assert config.watchdog.wall_clock_minutes == 240
    assert config.watchdog.wall_clock_kill is False
    assert config.watchdog.loop_stall_multiplier == 2
    assert config.watchdog.loop_kill is False


def test_config_watchdog_accepts_new_fields(tmp_path: Path) -> None:
    """New watchdog fields can be set explicitly in YAML."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  enabled: true
  stall_minutes: 20
  wall_clock_minutes: 300
  wall_clock_kill: true
  loop_stall_multiplier: 3
  loop_kill: true
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.watchdog.wall_clock_minutes == 300
    assert config.watchdog.wall_clock_kill is True
    assert config.watchdog.loop_stall_multiplier == 3
    assert config.watchdog.loop_kill is True


def test_watchdog_config_additive_redispatch_fields(tmp_path: Path) -> None:
    """Test that WatchdogConfig loads with defaults when new fields are missing (issue #165)."""
    # Create a config file without the new fields
    config_path = tmp_path / "orchestrator.yaml"
    config_content = """
watchdog:
  enabled: true
  stall_minutes: 20
"""
    config_path.write_text(config_content, encoding="utf-8")

    # Load the config - should not raise ConfigError
    config = load_config(config_path)

    # Verify defaults are applied
    assert config.watchdog.redispatch_window_minutes == 240
    assert config.watchdog.max_auto_redispatch == 3
