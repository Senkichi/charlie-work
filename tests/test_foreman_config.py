"""The ``foreman:`` config section and ``labels.foreman_prefix`` (issue #3117)."""

from __future__ import annotations

import pytest

from charlie_work.config import LabelConfig, OrchestratorConfig, build_config_from_data
from charlie_work.config_validation import ConfigError, host_wide_sections
from charlie_work.foreman_config import FOREMAN_DEFAULT_TOOLS, ForemanConfig
from charlie_work.role_chain import RoleEntry


def test_defaults_build() -> None:
    config = OrchestratorConfig().foreman
    assert isinstance(config, ForemanConfig)
    assert config.enabled is True
    assert config.harness == "claude-code"
    assert config.model == "claude-sonnet-5-5"  # sonnet-tier primary
    assert config.tools == FOREMAN_DEFAULT_TOOLS == "Read,Grep,Glob"
    assert config.fallbacks == ()
    assert config.max_model_turns > 0
    assert config.pass_interval_minutes > 0


def test_section_parses_from_config_data() -> None:
    config = build_config_from_data(
        {
            "foreman": {
                "enabled": False,
                "harness": "claude-code",
                "model": "claude-opus-4-7",
                "tools": "Read",
                "max_model_turns": 20,
                "pass_interval_minutes": 15,
                "fallbacks": [{"harness": "claude-code", "model": "claude-haiku-4-5"}],
            }
        }
    )
    assert config.foreman.enabled is False
    assert config.foreman.model == "claude-opus-4-7"
    assert config.foreman.pass_interval_minutes == 15
    assert config.foreman.fallbacks == (
        RoleEntry(harness="claude-code", model="claude-haiku-4-5", effort=""),
    )


def test_role_chain_resolves() -> None:
    """The chain is the same shape the reviewer role produces (primary + fallbacks)."""
    config = build_config_from_data(
        {"foreman": {"fallbacks": [{"harness": "claude-code", "model": "claude-haiku-4-5"}]}}
    )
    chain = config.foreman.chain
    assert len(chain) == 2
    assert chain[0] == RoleEntry(harness="claude-code", model="claude-sonnet-5-5", effort="")
    assert chain[1].model == "claude-haiku-4-5"


def test_bad_harness_rejected() -> None:
    """Foreman uses the reviewer (read-only) harness set — a worker-only harness fails."""
    with pytest.raises(ConfigError, match="foreman.harness"):
        build_config_from_data({"foreman": {"harness": "command"}})


def test_fallback_effort_rejected() -> None:
    """Foreman fallbacks pin no effort knob — the reviewer-style entries forbid it."""
    with pytest.raises(ConfigError):
        build_config_from_data(
            {
                "foreman": {
                    "fallbacks": [{"harness": "claude-code", "model": "m", "effort": "high"}]
                }
            }
        )


def test_section_is_host_wide() -> None:
    """Foreman settings live only in the fleet-global layer (ADR-0010)."""
    assert "foreman" in host_wide_sections()


def test_foreman_prefix_label() -> None:
    assert LabelConfig().foreman_prefix == "foreman:"
    # A prefix, not a managed label — the author creates it, so it must not be
    # bootstrap-created or stripped by the label state machine.
    labels = LabelConfig()
    assert labels.foreman_prefix not in labels.all
    assert labels.foreman_prefix not in labels.workflow_labels
