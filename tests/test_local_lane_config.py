"""Tests for the local-file issue source's review/merge lane config defaults
(issue #1844): ``load_config`` re-defaults ``review_dispatch.enabled`` and
``dispatch.rework_template`` when ``local_issues.enabled`` is true, so a
no-remote repo gets the automated worker -> review -> suite -> merge lane by
default instead of parking every finished ticket at an operator gate. Explicit
operator values are kill switches and always win.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import (
    LOCAL_REWORK_TEMPLATE,
    ConfigError,
    LocalIssuesConfig,
    OrchestratorConfig,
    load_config,
)


def test_load_config_local_issues_enables_review_dispatch(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("local_issues:\n  enabled: true\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.review_dispatch.enabled is True


def test_load_config_explicit_review_dispatch_disabled_wins(tmp_path: Path) -> None:
    """Kill switch: an explicit ``review_dispatch.enabled: false`` must hold."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "local_issues:\n  enabled: true\nreview_dispatch:\n  enabled: false\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.review_dispatch.enabled is False


def test_load_config_explicit_review_dispatch_enabled_unchanged(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "local_issues:\n  enabled: true\nreview_dispatch:\n  enabled: true\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.review_dispatch.enabled is True


def test_load_config_local_issues_disabled_keeps_review_dispatch_off(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("local_issues:\n  enabled: false\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.review_dispatch.enabled is False


def test_load_config_no_local_issues_section_keeps_review_dispatch_off() -> None:
    config = load_config(None)

    assert config.local_issues.enabled is False
    assert config.review_dispatch.enabled is False


def test_load_config_local_issues_redefaults_rework_template(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("local_issues:\n  enabled: true\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.dispatch.rework_template == LOCAL_REWORK_TEMPLATE


def test_load_config_explicit_rework_template_wins_over_local_redefault(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "local_issues:\n  enabled: true\ndispatch:\n  rework_template: rework.md\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.dispatch.rework_template == "rework.md"


def test_load_config_local_issues_disabled_keeps_default_rework_template(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("local_issues:\n  enabled: false\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.dispatch.rework_template == "rework.md"


def test_load_config_rework_redefault_is_load_config_specific() -> None:
    """MUTATION CHECK (contrast): building ``local_issues.enabled=True`` via the
    dataclass constructors bypasses ``load_config``'s re-default branch, so the
    rework template must stay at the remote default. If the branch were deleted,
    ``load_config`` would collapse to this result."""
    direct = OrchestratorConfig(local_issues=LocalIssuesConfig(enabled=True))

    assert direct.dispatch.rework_template == "rework.md"
    assert direct.review_dispatch.enabled is False


def test_load_config_local_issues_auto_merge_stays_enabled(tmp_path: Path) -> None:
    """``auto_merge.enabled`` already defaults True for every config; the local
    lane relies on that default, and explicit ``false`` remains the kill
    switch."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("local_issues:\n  enabled: true\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.auto_merge.enabled is True


def test_load_config_explicit_auto_merge_disabled_wins(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "local_issues:\n  enabled: true\nauto_merge:\n  enabled: false\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.auto_merge.enabled is False


def test_load_config_local_lane_stall_hours_defaults_to_12(tmp_path: Path) -> None:
    """Issue #1968: an absent ``local_lane`` section keeps the default."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("local_issues:\n  enabled: true\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.local_lane.kill_switch_stall_hours == 12.0


def test_load_config_local_lane_explicit_stall_hours(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        "local_issues:\n  enabled: true\nlocal_lane:\n  kill_switch_stall_hours: 4\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.local_lane.kill_switch_stall_hours == 4


@pytest.mark.parametrize("raw", ["soon", "true", "-1"], ids=["string", "bool", "negative"])
def test_load_config_local_lane_rejects_bad_stall_hours(tmp_path: Path, raw: str) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        f"local_lane:\n  kill_switch_stall_hours: {raw}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="kill_switch_stall_hours"):
        load_config(config_file)


def test_load_config_local_lane_rejects_unknown_key(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("local_lane:\n  bogus_knob: 1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown key"):
        load_config(config_file)
