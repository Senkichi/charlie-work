"""Tests for the infra-blocked config section (issue #1383).

Extracted from ``tests/test_config.py`` as part of the attachment-contracts
ratchet remedy (issue #1616): ``test_config.py`` exceeded its baselined
ceiling, and the infra-blocked validation test family -- added by #1421 but
not captured by the stale baseline -- is a coherent group that belongs in its
own module alongside the config-validation test pattern it follows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import ConfigError, load_config


def _write_config(config_file: Path, content: str) -> None:
    config_file.write_text(content, encoding="utf-8")


_IB_HEADER = "auto_merge:\n  required_checks: [Tests passed]\n  infra_blocked:\n"


def test_load_config_infra_blocked_defaults_when_absent(tmp_path: Path) -> None:
    """An absent infra_blocked section yields InfraBlockedConfig's documented
    defaults -- no silent zero/None, matching the sibling
    stale_checks default-fallback shape."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        "auto_merge:\n  required_checks: [Tests passed]\n",
    )
    config = load_config(config_file)
    cfg = config.auto_merge.infra_blocked
    assert cfg.enabled is True
    assert cfg.instant_fail_seconds == 10
    assert cfg.persistence_passes == 3
    assert cfg.escalation_window_minutes == 60
    assert cfg.annotation_patterns == (
        "the job was not started",
        "actions budget is preventing further use",
        "no runner matching",
        "usage limit",
    )


def test_load_config_infra_blocked_rejects_non_mapping(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        "auto_merge:\n  required_checks: [Tests passed]\n  infra_blocked: not-a-map\n",
    )
    with pytest.raises(ConfigError, match="infra_blocked.*must be a mapping"):
        load_config(config_file)


def test_load_config_infra_blocked_rejects_unknown_key(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    bogus_key: 1\n",
    )
    with pytest.raises(ConfigError, match="infra_blocked.*has unknown key"):
        load_config(config_file)


def test_load_config_infra_blocked_annotation_patterns_rejects_non_list(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    annotation_patterns: billing-exhausted\n",
    )
    with pytest.raises(ConfigError, match="annotation_patterns.*must be a list of strings"):
        load_config(config_file)


def test_load_config_infra_blocked_instant_fail_seconds_rejects_non_int(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    instant_fail_seconds: ten\n",
    )
    with pytest.raises(ConfigError, match="instant_fail_seconds.*must be an int"):
        load_config(config_file)


def test_load_config_infra_blocked_instant_fail_seconds_rejects_bool(
    tmp_path: Path,
) -> None:
    """YAML boolean true is not a valid integer count (matches the
    stale_checks_max_retriggers bool-rejection convention)."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    instant_fail_seconds: true\n",
    )
    with pytest.raises(ConfigError, match="instant_fail_seconds.*must be an int"):
        load_config(config_file)


def test_load_config_infra_blocked_instant_fail_seconds_rejects_negative(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    instant_fail_seconds: -1\n",
    )
    with pytest.raises(ConfigError, match="instant_fail_seconds.*must be >= 0"):
        load_config(config_file)


def test_load_config_infra_blocked_persistence_passes_rejects_non_int(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    persistence_passes: three\n",
    )
    with pytest.raises(ConfigError, match="persistence_passes.*must be an int"):
        load_config(config_file)


def test_load_config_infra_blocked_persistence_passes_rejects_bool(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    persistence_passes: false\n",
    )
    with pytest.raises(ConfigError, match="persistence_passes.*must be an int"):
        load_config(config_file)


def test_load_config_infra_blocked_persistence_passes_rejects_negative(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    persistence_passes: -2\n",
    )
    with pytest.raises(ConfigError, match="persistence_passes.*must be >= 0"):
        load_config(config_file)


def test_load_config_infra_blocked_escalation_window_minutes_rejects_non_int(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    escalation_window_minutes: 1h\n",
    )
    with pytest.raises(ConfigError, match="escalation_window_minutes.*must be an int"):
        load_config(config_file)


def test_load_config_infra_blocked_escalation_window_minutes_rejects_bool(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    escalation_window_minutes: true\n",
    )
    with pytest.raises(ConfigError, match="escalation_window_minutes.*must be an int"):
        load_config(config_file)


def test_load_config_infra_blocked_escalation_window_minutes_rejects_negative(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    escalation_window_minutes: -5\n",
    )
    with pytest.raises(ConfigError, match="escalation_window_minutes.*must be >= 0"):
        load_config(config_file)


def test_load_config_infra_blocked_enabled_rejects_non_bool(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    enabled: yes-please\n",
    )
    with pytest.raises(ConfigError, match="infra_blocked.enabled.*must be a bool"):
        load_config(config_file)
