"""Round-trip coverage for role-config Phase 1 (issue TBD): confirms both
shipped example configs and a pure new-style config all load successfully
through the real load_config() entry point, with the worker/reviewer role
config resolving to the expected values.

Phase 2 Track D (2026-08-30 role-config-phase2-deletions plan, Task 1)
rewrote both shipped example configs to the new worker:/reviewer: schema
directly, so loading them no longer exercises any legacy
devin.adapter/claude_code.model mapping.

Phase 2 Track E (this file, same session) then deleted the dual-accept
resolver in config.py itself -- `devin.adapter`/`claude_code.model` are not
read, mapped, or reconciled with worker.harness/reviewer.harness at all any
more; they are simply unrecognized keys, same as any other typo. There is no
longer a "legacy key conflicts with new key" path anywhere in config.py, so
this file's old conflict test has been replaced with a plain
unknown-key-raises test below (`test_roundtrip_legacy_devin_adapter_key_raises`).
"""

from pathlib import Path

import pytest

from charlie_work.config import (
    _DEFAULT_CLAUDE_MODEL,
    ConfigError,
    build_config_from_data,
    load_config,
)

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_roundtrip_claude_code_example_config() -> None:
    config = load_config(_EXAMPLES_DIR / "orchestrator.config.claude-code.yaml")

    assert config.worker.harness == "claude-code"
    # This example does not set worker.model -- it stays unset (empty
    # string) at config-load time. The fallback to the shared CLI-pin
    # default (_DEFAULT_CLAUDE_MODEL) happens in claude_code.py's
    # _apply_model_pin at worker-launch time, not in the config layer.
    assert config.worker.model == ""
    assert config.reviewer.harness == "claude-code"
    # This example does not set reviewer.model either -- it resolves to
    # ReviewerRoleConfig's own dataclass default, _DEFAULT_CLAUDE_MODEL.
    assert config.reviewer.model == _DEFAULT_CLAUDE_MODEL


def test_roundtrip_devin_example_config() -> None:
    config = load_config(_EXAMPLES_DIR / "orchestrator.config.devin.yaml")

    assert config.worker.harness == "devin-shell"
    # Explicitly set to "" in the example (CLI default model).
    assert config.worker.model == ""
    # review_dispatch is off in this profile, but reviewer: is still
    # declared for when it is flipped on -- harness defaults to
    # claude-code and no model override is set.
    assert config.reviewer.harness == "claude-code"
    assert config.reviewer.model == _DEFAULT_CLAUDE_MODEL


def test_roundtrip_pure_new_style_config_loads_cleanly() -> None:
    """A config that only ever uses the new worker:/reviewer: keys loads
    with no ConfigError and resolves both roles' harness/model/effort
    exactly as given -- there is no dual-write to any legacy field to
    assert on any more (devin.adapter/claude_code.model were deleted)."""
    config = build_config_from_data(
        {
            "worker": {"harness": "devin-shell", "model": "glm-5-2"},
            "reviewer": {"harness": "claude-code", "model": "claude-opus-4-1", "effort": "high"},
        }
    )

    assert config.worker.harness == "devin-shell"
    assert config.worker.model == "glm-5-2"
    assert config.reviewer.harness == "claude-code"
    assert config.reviewer.model == "claude-opus-4-1"
    assert config.reviewer.effort == "high"


def test_roundtrip_legacy_devin_adapter_key_raises() -> None:
    """`devin.adapter` is no longer read, mapped, or reconciled with
    worker.harness -- it is simply an unrecognized key under the `devin:`
    section, same as any other typo, regardless of what else is set
    alongside it."""
    with pytest.raises(ConfigError, match=r"unknown key\(s\) in config section 'devin': adapter"):
        build_config_from_data(
            {"devin": {"adapter": "claude-shell"}, "worker": {"harness": "devin-shell"}}
        )
