"""Round-trip coverage for role-config Phase 1 (issue TBD): confirms both
pre-existing example configs (old-style keys only) and a pure new-style
config all load successfully through the real load_config() entry point,
with the worker/reviewer role config resolving to the values those old keys
always meant.
"""

from pathlib import Path

import pytest

from charlie_work.config import ConfigError, build_config_from_data, load_config

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_roundtrip_claude_code_example_config() -> None:
    config = load_config(_EXAMPLES_DIR / "orchestrator.config.claude-code.yaml")

    assert config.devin.adapter == "claude-code"
    assert config.worker.harness == "claude-code"
    # This example config does not set claude_code.model -- worker.model
    # must resolve to the shared CLI-pin default, not an empty string (the
    # exact regression Task 3 guards against at the unit level).
    assert config.worker.model == config.claude_code.model
    assert config.worker.model != ""
    assert config.reviewer.harness == "claude-code"
    assert config.reviewer.model == config.worker.model
    assert any("devin.adapter is deprecated" in msg for msg in config.deprecations)


def test_roundtrip_devin_example_config() -> None:
    config = load_config(_EXAMPLES_DIR / "orchestrator.config.devin.yaml")

    assert config.devin.adapter == "devin-shell"
    assert config.worker.harness == "devin-shell"
    assert config.worker.model == config.devin.worker_model
    # This example sets cross_family.enabled: true with an explicit command
    # -- confirms the presence-based deprecation fires regardless of the
    # cross_family.command customization.
    assert any("cross_family is deprecated" in msg for msg in config.deprecations)
    assert any("devin.adapter is deprecated" in msg for msg in config.deprecations)


def test_roundtrip_pure_new_style_config_has_no_deprecations() -> None:
    config = build_config_from_data(
        {
            "worker": {"harness": "devin-shell", "model": "glm-5-2"},
            "reviewer": {"harness": "claude-code", "model": "claude-opus-4-1", "effort": "high"},
        }
    )

    assert config.worker.harness == "devin-shell"
    assert config.worker.model == "glm-5-2"
    assert config.reviewer.model == "claude-opus-4-1"
    assert config.reviewer.effort == "high"
    # Legacy fields are still populated by the dual-write for any untouched
    # call site, but no legacy KEY was actually present in the input, so
    # there is nothing to warn about.
    assert config.devin.adapter == "devin-shell"
    assert config.devin.worker_model == "glm-5-2"
    assert config.deprecations == ()


def test_roundtrip_new_style_conflicting_with_old_style_raises() -> None:
    with pytest.raises(ConfigError, match="devin.adapter.*claude-shell.*worker.harness"):
        build_config_from_data(
            {"devin": {"adapter": "claude-shell"}, "worker": {"harness": "devin-shell"}}
        )
