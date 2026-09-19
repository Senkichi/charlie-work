"""``load_config`` parsing and validation: scalar coercion, unknown-key
and unknown-section rejection, dispatch ordering, launch stagger, and
runtime throttle/session-limit marker sections.

Split out of ``tests/test_charlie_work.py`` (issue #1552, Track-1
wave 6/8): the ``test_load_config_*`` seam minus merge-queue and
placeholder-validation members, which live in
``tests/test_charlie_work_load_config_mergequeue.py`` and
``tests/test_charlie_work_load_config_placeholders.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from charlie_work.config import load_config


def test_load_config_rejects_non_int_failed_attempt_alarm(tmp_path: Path) -> None:
    """Issue #254: auto_merge.failed_attempt_alarm must be an int."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  failed_attempt_alarm: "three"
"""
    )
    with pytest.raises(ConfigError, match="failed_attempt_alarm.*must be an int"):
        load_config(config_file)


def test_load_config_update_open_prs_boolean_aliases(tmp_path: Path) -> None:
    """update_open_prs boolean aliases are normalized for backward compatibility."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  update_open_prs: true
"""
    )
    config = load_config(config_file)
    assert config.auto_merge.update_open_prs == "all"

    config_file.write_text(
        """
auto_merge:
  update_open_prs: false
  require_current_base: false
"""
    )
    config = load_config(config_file)
    assert config.auto_merge.update_open_prs == "off"


def test_load_config_update_open_prs_string_values(tmp_path: Path) -> None:
    """update_open_prs accepts all/next/off string values."""
    config_file = tmp_path / "orchestrator.config.yaml"
    for value in ("all", "next", "off", "ALL", "Next", "OFF"):
        config_file.write_text(
            f"""
auto_merge:
  update_open_prs: {value}
  require_current_base: false
"""
        )
        config = load_config(config_file)
        assert config.auto_merge.update_open_prs == value.lower()


def test_load_config_update_open_prs_rejects_invalid_value(tmp_path: Path) -> None:
    """update_open_prs rejects unknown string values."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  update_open_prs: sometimes
"""
    )
    with pytest.raises(ConfigError, match="update_open_prs.*'all', 'next', 'off'"):
        load_config(config_file)


def test_load_config_update_branch_strategy_values(tmp_path: Path) -> None:
    """Issue #404: update_branch_strategy accepts front_of_train/broadcast/off."""
    from charlie_work.config import ConfigError, load_config

    config_file = tmp_path / "orchestrator.config.yaml"
    for value in ("front_of_train", "broadcast", "off"):
        config_file.write_text(
            f"""
auto_merge:
  update_branch_strategy: {value}
  require_current_base: false
"""
        )
        config = load_config(config_file)
        assert config.auto_merge.update_branch_strategy == value

    config_file.write_text(
        """
auto_merge:
  update_branch_strategy: sometimes
"""
    )
    with pytest.raises(
        ConfigError, match="update_branch_strategy.*'front_of_train', 'broadcast', or 'off'"
    ):
        load_config(config_file)


def test_load_config_runtime_throttle_error_markers(tmp_path: Path) -> None:
    """RuntimeConfig.throttle_error_markers is configurable from YAML."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
runtime:
  throttle_error_markers:
    - "Reached overall message rate limit"
    - "A tool was rejected by the user"
    - "custom provider failure"
"""
    )
    config = load_config(config_file)
    assert config.runtime.throttle_error_markers == (
        "Reached overall message rate limit",
        "A tool was rejected by the user",
        "custom provider failure",
    )


def test_load_config_runtime_session_limit_markers(tmp_path: Path) -> None:
    """RuntimeConfig.session_limit_markers is configurable from YAML (issue #651/#652)."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
runtime:
  session_limit_markers:
    - "hit your session limit"
    - "custom session-limit phrasing"
"""
    )
    config = load_config(config_file)
    assert config.runtime.session_limit_markers == (
        "hit your session limit",
        "custom session-limit phrasing",
    )


def test_load_config_names_unknown_keys_and_section(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "review:\n  max_rework_cycles: 2\n  max_rework_cylces: 3\n", encoding="utf-8"
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")

    assert "section 'review'" in message
    assert "max_rework_cylces" in message
    assert "max_rework_cycles" in message  # valid keys listed for the operator


def test_load_config_rejects_unknown_top_level_sections(tmp_path: Path) -> None:
    """Issue #12: typo'd top-level config section is rejected, not silently ignored."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text("auto-merge:\n  enabled: false\n", encoding="utf-8")

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown top-level section")

    assert "unknown config section(s)" in message
    assert "auto-merge" in message
    assert "auto_merge" in message  # valid section name listed


def test_load_config_rejects_broken_yaml(tmp_path: Path) -> None:
    """Issue #12: malformed YAML yields YAMLError, not raw traceback."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text("labels:\n  ready: automated-ready\n  bad: [unclosed", encoding="utf-8")

    try:
        load_config(config_path)
    except yaml.YAMLError:
        # Expected: YAML parsing error
        pass
    else:  # pragma: no cover
        raise AssertionError("expected YAMLError for malformed YAML")


def test_load_config_rejects_invalid_dispatch_order(tmp_path: Path) -> None:
    """Issue #151: invalid dispatch.order config value is rejected at load."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "dispatch:\n  order: invalid\n",
        encoding="utf-8",
    )

    from charlie_work.config import ConfigError, load_config

    try:
        load_config(config_file)
        raise AssertionError("expected ConfigError for invalid dispatch.order")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid dispatch.order")

    assert "dispatch" in message
    assert "order" in message
    assert "oldest" in message or "newest" in message


def test_load_config_parses_dispatch_launch_stagger_seconds_override(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "dispatch:\n  launch_stagger_seconds: 10\n",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.dispatch.launch_stagger_seconds == 10


def test_load_config_rejects_negative_launch_stagger_seconds(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "dispatch:\n  launch_stagger_seconds: -1\n",
        encoding="utf-8",
    )

    try:
        load_config(config_file)
        raise AssertionError("expected ConfigError for negative launch_stagger_seconds")
    except ConfigError as exc:
        message = str(exc)

    assert "dispatch" in message
    assert "launch_stagger_seconds" in message


def test_load_config_rejects_wrong_type_launch_stagger_seconds(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "dispatch:\n  launch_stagger_seconds: not-a-number\n",
        encoding="utf-8",
    )

    try:
        load_config(config_file)
        raise AssertionError("expected ConfigError for non-int launch_stagger_seconds")
    except ConfigError as exc:
        message = str(exc)

    assert "dispatch" in message
    assert "launch_stagger_seconds" in message


def test_load_config_rejects_unknown_test_adequacy_key(tmp_path: Path) -> None:
    """Unknown keys under test_adequacy raise ConfigError listing valid keys."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "test_adequacy:\n  enabled: true\n  bad_key: value\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for unknown test_adequacy key")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown test_adequacy key")

    assert "section 'test_adequacy'" in message
    assert "bad_key" in message
    # Should list valid keys
    assert "enabled" in message


def test_load_config_rejects_unknown_coverage_probe_key(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        "coverage_probe:\n  enabled: true\n  bad_key: value\n",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for unknown coverage_probe key")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown coverage_probe key")

    assert "section 'coverage_probe'" in message
    assert "bad_key" in message
    assert "enabled" in message
