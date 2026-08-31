"""Round-trip coverage for role-config Phase 1 (issue TBD): confirms both
shipped example configs and a pure new-style config all load successfully
through the real load_config() entry point, with the worker/reviewer role
config resolving to the expected values.

Phase 2 Track D (2026-08-30 role-config-phase2-deletions plan, Task 1)
rewrote both shipped example configs to the new worker:/reviewer: schema
directly, so loading them no longer exercises the legacy
devin.adapter/cross_family dual-accept path this file originally asserted
on -- that conflict/mapping path is still covered by
test_roundtrip_new_style_conflicting_with_old_style_raises below and by the
config layer's own dual-accept unit tests.
"""

from pathlib import Path

import pytest

from charlie_work.config import ConfigError, build_config_from_data, load_config

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_roundtrip_claude_code_example_config() -> None:
    config = load_config(_EXAMPLES_DIR / "orchestrator.config.claude-code.yaml")

    # devin.adapter is back-derived from the new-style worker.harness for
    # any remaining reader of the legacy field -- the example itself no
    # longer sets it directly.
    assert config.devin.adapter == "claude-code"
    assert config.worker.harness == "claude-code"
    # This example config does not set worker.model -- it must resolve to
    # the shared CLI-pin default, not an empty string (the exact regression
    # Task 3 guards against at the unit level).
    assert config.worker.model == config.claude_code.model
    assert config.worker.model != ""
    assert config.reviewer.harness == "claude-code"
    assert config.reviewer.model == config.worker.model
    # New-style worker:/reviewer: keys only -- no legacy devin.adapter key
    # is present in the YAML, so no deprecation warning fires.
    assert config.deprecations == ()


def test_roundtrip_devin_example_config() -> None:
    config = load_config(_EXAMPLES_DIR / "orchestrator.config.devin.yaml")

    # devin.adapter is back-derived from the new-style worker.harness for
    # any remaining reader of the legacy field -- the example itself no
    # longer sets it directly.
    assert config.devin.adapter == "devin-shell"
    assert config.worker.harness == "devin-shell"
    assert config.worker.model == config.devin.worker_model
    assert config.reviewer.harness == "claude-code"
    # New-style worker:/reviewer: keys only -- the cross_family section and
    # devin.adapter key were deleted from this example in Phase 2 Track D,
    # so no deprecation warning fires.
    assert config.deprecations == ()


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
