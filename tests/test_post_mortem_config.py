"""Tests for PostMortemConfig, carved out of test_charlie_work.py (#1284)."""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import PostMortemConfig, SignatureRule, load_config


# ---------------------------------------------------------------------------
# PostMortemConfig tests (issue #261)
# ---------------------------------------------------------------------------


def test_post_mortem_config_defaults() -> None:
    """PostMortemConfig defaults are stable and load_config picks them up.

    Issue #260 (corrected premise) added a third default rule: "A tool was
    rejected by the user" is the Devin CLI's own log/stdout surfacing of a
    PreToolUse hook block — distinct from the "Tool blocked:" prefix that
    appears in sessions.db message-node content — and drives
    post_mortem.classify_and_record's log-tail fallback when the DB is
    unavailable (see test_post_mortem.py's log-tail fallback tests).
    """
    config = load_config()
    assert isinstance(config.post_mortem, PostMortemConfig)
    assert config.post_mortem.enabled is True
    assert config.post_mortem.db_path == ""
    assert config.post_mortem.message_node_limit == 10
    assert config.post_mortem.match_window_margin_seconds == 120
    assert config.post_mortem.signature_rules == (
        SignatureRule(pattern=r"Tool blocked:", kind="worker_blocked"),
        SignatureRule(pattern=r"decision\s*:\s*block", kind="worker_blocked"),
        SignatureRule(pattern=r"A tool was rejected by the user", kind="worker_blocked"),
    )


def test_post_mortem_config_parses_custom_values(tmp_path: Path) -> None:
    """Custom post_mortem section values, including signature_rules, are parsed correctly."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
post_mortem:
  enabled: false
  db_path: "C:/custom/sessions.db"
  message_node_limit: 5
  match_window_margin_seconds: 30
  signature_rules:
    - pattern: "custom-block-signature"
      kind: "worker_blocked"
"""
    )
    config = load_config(config_file)
    assert config.post_mortem.enabled is False
    assert config.post_mortem.db_path == "C:/custom/sessions.db"
    assert config.post_mortem.message_node_limit == 5
    assert config.post_mortem.match_window_margin_seconds == 30
    assert config.post_mortem.signature_rules == (
        SignatureRule(pattern="custom-block-signature", kind="worker_blocked"),
    )


def test_post_mortem_config_unknown_key_raises(tmp_path: Path) -> None:
    """Unknown keys in post_mortem section raise ConfigError."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
post_mortem:
  enabled: true
  unknown_key: 99
"""
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(config_file)


def test_post_mortem_config_wrong_type_raises(tmp_path: Path) -> None:
    """Wrong types in post_mortem section raise ConfigError."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
post_mortem:
  message_node_limit: "not-an-int"
"""
    )
    with pytest.raises(ConfigError, match="must be an int"):
        load_config(config_file)


def test_post_mortem_config_signature_rules_wrong_shape_raises(tmp_path: Path) -> None:
    """signature_rules must be a list of {pattern, kind} mappings."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
post_mortem:
  signature_rules: "not-a-list"
"""
    )
    with pytest.raises(ConfigError, match="must be a list"):
        load_config(config_file)


def test_post_mortem_config_signature_rule_bad_regex_raises(tmp_path: Path) -> None:
    """An invalid regex pattern in a signature rule raises ConfigError."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
post_mortem:
  signature_rules:
    - pattern: "["
      kind: "worker_blocked"
"""
    )
    with pytest.raises(ConfigError, match="not a valid regex"):
        load_config(config_file)


def test_post_mortem_config_is_frozen() -> None:
    """PostMortemConfig is a frozen dataclass."""
    import dataclasses

    cfg = PostMortemConfig()
    assert dataclasses.is_dataclass(cfg)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        cfg.enabled = False  # type: ignore[misc]
