"""Tests for the worker/reviewer role-config registry and dual-accept
migration (issues #1512, #1527, #1535).

Extracted from ``tests/test_config.py`` as part of the attachment-contracts
ratchet remedy (issue #1616): ``test_config.py`` exceeded its baselined
ceiling, and the over-ceiling growth is the role-config test family added by
the phase-1/phase-2 harness-registry migration -- worker/reviewer role config
defaults, harness validation against the registry, reviewer effort experiment
knobs, rescue worker/reviewer role wiring, and the dual-accept bridge deletion
rejection tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.config import (
    ConfigError,
    DispatchConfig,
    RescueConfig,
    ReviewerRoleConfig,
    WorkerRoleConfig,
    build_config_from_data,
    known_config_sections,
    load_config,
)


def _write_config(config_file: Path, content: str) -> None:
    config_file.write_text(content, encoding="utf-8")


def test_dispatch_config_has_no_worker_model_tier_field() -> None:
    assert not hasattr(DispatchConfig(), "worker_model_tier")


def test_load_config_reviewer_effort_experiment_defaults() -> None:
    """reviewer.effort_experiment_fraction/salt default to disabled (0.0/'').

    Relocated from review_dispatch.review_effort_experiment_* (role-config
    Phase 2 Track E deleted those ReviewDispatchConfig fields; the
    equivalent knobs now live on ReviewerRoleConfig)."""
    config = load_config(Path("nonexistent.yaml"))
    assert config.reviewer.effort_experiment_fraction == 0.0
    assert config.reviewer.effort_experiment_salt == ""


def test_load_config_reviewer_effort_experiment_override(tmp_path: Path) -> None:
    """reviewer.effort/effort_experiment_fraction/salt are read from YAML."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """reviewer:
  effort: medium
  effort_experiment_fraction: 0.25
  effort_experiment_salt: epoch-2
""",
    )
    config = load_config(config_file)
    assert config.reviewer.effort == "medium"
    assert config.reviewer.effort_experiment_fraction == 0.25
    assert config.reviewer.effort_experiment_salt == "epoch-2"


def test_load_config_reviewer_effort_experiment_fraction_rejects_bool(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """reviewer:
  effort_experiment_fraction: true
""",
    )
    with pytest.raises(ConfigError, match="effort_experiment_fraction.*must be a number"):
        load_config(config_file)


def test_load_config_reviewer_effort_experiment_fraction_rejects_non_number(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """reviewer:
  effort_experiment_fraction: "0.5"
""",
    )
    with pytest.raises(ConfigError, match="effort_experiment_fraction.*must be a number"):
        load_config(config_file)


@pytest.mark.parametrize("value", [-0.01, 1.01, 2, -1])
def test_load_config_reviewer_effort_experiment_fraction_rejects_out_of_range(
    tmp_path: Path, value: float
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        f"""reviewer:
  effort_experiment_fraction: {value}
""",
    )
    with pytest.raises(ConfigError, match=r"effort_experiment_fraction.*must be in \[0.0, 1.0\]"):
        load_config(config_file)


def test_load_config_reviewer_effort_experiment_fraction_without_effort_rejected(
    tmp_path: Path,
) -> None:
    """fraction > 0.0 with effort unset (default '') must fail loud at
    load time -- treatment would otherwise silently mean 'no --effort pin'."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """reviewer:
  effort_experiment_fraction: 0.25
""",
    )
    with pytest.raises(
        ConfigError,
        match="effort_experiment_fraction.*is 0.25 but 'effort' is unset",
    ):
        load_config(config_file)


def test_load_config_reviewer_effort_experiment_fraction_with_effort_accepted(
    tmp_path: Path,
) -> None:
    """fraction > 0.0 WITH effort set is a valid, accepted config."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """reviewer:
  effort: high
  effort_experiment_fraction: 0.25
""",
    )
    config = load_config(config_file)
    assert config.reviewer.effort == "high"
    assert config.reviewer.effort_experiment_fraction == 0.25


def test_load_config_reviewer_effort_experiment_fraction_zero_without_effort_accepted() -> None:
    """The default config (fraction=0.0, effort unset) must keep loading."""
    config = load_config(Path("nonexistent.yaml"))
    assert config.reviewer.effort_experiment_fraction == 0.0
    assert config.reviewer.effort == ""


def test_load_config_reviewer_effort_experiment_salt_rejects_non_str(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """reviewer:
  effort_experiment_salt: 123
""",
    )
    with pytest.raises(ConfigError, match="effort_experiment_salt.*must be a string"):
        load_config(config_file)


def test_worker_role_config_defaults_and_frozen() -> None:
    cfg = WorkerRoleConfig()
    assert cfg.harness == "manual"
    assert cfg.model == ""
    with pytest.raises(AttributeError):
        cfg.harness = "claude-code"  # type: ignore[misc]


def test_reviewer_role_config_defaults_and_frozen() -> None:
    cfg = ReviewerRoleConfig()
    assert cfg.harness == "claude-code"
    assert cfg.model == "claude-sonnet-5"
    assert cfg.effort == ""
    assert cfg.effort_experiment_fraction == 0.0
    assert cfg.effort_experiment_salt == ""
    with pytest.raises(AttributeError):
        cfg.model = "x"  # type: ignore[misc]


def test_rescue_config_worker_and_reviewer_role_defaults() -> None:
    cfg = RescueConfig()
    assert cfg.worker == WorkerRoleConfig(harness="claude-code", model="claude-opus-4-1")
    assert cfg.reviewer == WorkerRoleConfig(harness="devin", model="codex")
    # Legacy fields are untouched by this task -- still their own defaults.
    assert cfg.worker_adapter == "claude-code"
    assert cfg.worker_model == "claude-opus-4-1"
    assert cfg.reviewer_adapter == "devin"
    assert cfg.reviewer_model == "codex"


def test_rescue_config_reviewer_command_default() -> None:
    """Regression pin (role-config Phase 2 Track E, item 2d)."""
    assert RescueConfig().reviewer_command == (
        "devin",
        "--model",
        "{model}",
        "-p",
        "--prompt-file",
        "{prompt_path}",
    )


def test_worker_and_reviewer_are_known_config_sections() -> None:
    sections = known_config_sections()
    assert "worker" in sections
    assert "reviewer" in sections
    # Provenance fields must never be forgeable as a YAML section.
    assert "sources" not in sections


def test_build_config_from_data_rescue_worker_reviewer_are_workerroleconfig_instances() -> None:
    cfg = build_config_from_data(
        {"rescue": {"worker": {"harness": "devin-shell", "model": "glm-5-2"}}}
    )
    assert isinstance(cfg.rescue.worker, WorkerRoleConfig)
    assert cfg.rescue.worker == WorkerRoleConfig(harness="devin-shell", model="glm-5-2")
    assert isinstance(cfg.rescue.reviewer, WorkerRoleConfig)
    assert cfg.rescue.reviewer == WorkerRoleConfig(harness="devin", model="codex")


def test_build_config_from_data_wires_worker_and_reviewer_end_to_end() -> None:
    """The dual-accept bridge that used to translate ``devin.adapter`` /
    ``devin.worker_model`` into ``worker.harness`` / ``worker.model`` is gone
    (role-config Phase 2 Track E) -- config must be written directly in the
    new ``worker:``/``reviewer:`` shape for it to wire through."""
    cfg = build_config_from_data(
        {
            "worker": {"harness": "devin-shell", "model": "glm-5-2"},
            "reviewer": {"model": "claude-opus-4-1"},
        }
    )
    assert cfg.worker.harness == "devin-shell"
    assert cfg.worker.model == "glm-5-2"
    assert cfg.reviewer.model == "claude-opus-4-1"


def test_build_config_from_data_devin_adapter_is_rejected_as_unknown_key() -> None:
    """``devin.adapter`` used to soft-deprecate into ``worker.harness`` via
    the now-deleted dual-accept bridge (role-config Phase 2 Track E); with
    that bridge gone and ``DevinConfig.adapter`` itself deleted, it falls
    straight through to ``_build_section``'s generic unknown-key rejection."""
    with pytest.raises(ConfigError, match=r"unknown key\(s\) in config section 'devin'.*adapter"):
        build_config_from_data({"devin": {"adapter": "devin-shell"}})


def test_build_config_from_data_worker_model_tier_key_is_rejected() -> None:
    """``dispatch.worker_model_tier`` was a pure deletion (Phase 2 Task 2, no
    migration target), not a dual-accept rename -- so its presence is no
    longer a soft deprecation warning. It falls straight through to
    ``_build_section``'s generic unknown-key rejection, same as any other
    key a section's dataclass doesn't declare."""
    with pytest.raises(ConfigError, match=r"unknown key\(s\) in config section 'dispatch'"):
        build_config_from_data({"dispatch": {"worker_model_tier": "capable"}})


def test_build_config_from_data_fallback_adapter_is_rejected_as_unknown_key() -> None:
    """``api_worker.fallback_adapter`` was deleted (role-config Phase 2, Track B);
    the dual-accept bridge that used to soft-warn on it is gone too (Track E),
    so it now falls straight through to the generic unknown-key ConfigError."""
    with pytest.raises(ConfigError, match="fallback_adapter"):
        build_config_from_data({"api_worker": {"fallback_adapter": "devin-shell"}})


def test_build_config_from_data_cross_family_section_is_rejected_as_unknown_section() -> None:
    with pytest.raises(ConfigError, match="cross_family"):
        build_config_from_data({"cross_family": {"command": ["devin", "--model", "{model}"]}})


def test_build_config_from_data_empty_bodied_cross_family_section_is_rejected() -> None:
    """An empty-bodied cross_family: {} section used to be specially tolerated
    by _DEPRECATED_SECTIONS_WITHOUT_A_FIELD; that carve-out is gone (role-config
    Phase 2 Track E), so an empty body is rejected exactly like a populated one."""
    with pytest.raises(ConfigError, match="cross_family"):
        build_config_from_data({"cross_family": {}})


def test_build_config_from_data_unknown_worker_key_still_raises_configerror() -> None:
    with pytest.raises(ConfigError, match="unknown key.*worker.*bogus"):
        build_config_from_data({"worker": {"bogus": "x"}})


def test_build_config_from_data_invalid_worker_harness_is_rejected() -> None:
    """Harness membership validation moved from the deleted dual-accept
    resolver to the top-level ``worker = _build_section(...)`` call site
    (role-config Phase 2 Track E); this pins the relocated check so a future
    refactor cannot silently drop it."""
    with pytest.raises(ConfigError, match=r"section 'worker' key 'harness' must be one of"):
        build_config_from_data({"worker": {"harness": "bogus-harness"}})


def test_build_config_from_data_reviewer_harness_accepts_any_registered_harness() -> None:
    """Issue #1513: reviewer.harness is no longer pinned to claude-code only --
    it accepts any harness the ``harnesses.HARNESS_REGISTRY`` marks
    review-capable (currently claude-code, devin-shell, api) and rejects only
    a harness that isn't registered at all. This replaces the old pinned test
    that asserted the pre-#1513 claude-code-only asymmetry: that asymmetry is
    the bug #1513 fixes, not an invariant to protect. ``"devin"`` (the
    WorkerView.adapter_kind value, not the harness name ``"devin-shell"``) is
    deliberately used for the rejection case so this test cannot pass merely
    because a stale/adapter_kind-shaped name was typo'd in place of the real
    harness name."""
    config = build_config_from_data({"reviewer": {"harness": "devin-shell"}})
    assert config.reviewer.harness == "devin-shell"

    with pytest.raises(ConfigError, match=r"section 'reviewer' key 'harness' must be one of"):
        build_config_from_data({"reviewer": {"harness": "devin"}})


def test_build_config_from_data_worker_harness_matches_registry() -> None:
    """Issue #1513: worker-harness validation is derived from
    ``harnesses.WORKER_HARNESSES`` -- not a second hardcoded list in
    config.py -- so every currently-registered harness is accepted."""
    from charlie_work.harnesses import WORKER_HARNESSES

    for harness in WORKER_HARNESSES:
        config = build_config_from_data({"worker": {"harness": harness}})
        assert config.worker.harness == harness

    with pytest.raises(ConfigError, match=r"section 'worker' key 'harness' must be one of"):
        build_config_from_data({"worker": {"harness": "not-a-real-harness"}})


def test_build_config_from_data_reviewer_harness_matches_registry() -> None:
    """Issue #1513: reviewer-harness validation is derived from
    ``harnesses.REVIEWER_HARNESSES`` -- the same registry worker validation
    reads and ``adapters.py``'s dispatch table is drift-checked against --
    not a separate hardcoded set."""
    from charlie_work.harnesses import REVIEWER_HARNESSES

    for harness in REVIEWER_HARNESSES:
        config = build_config_from_data({"reviewer": {"harness": harness}})
        assert config.reviewer.harness == harness

    with pytest.raises(ConfigError, match=r"section 'reviewer' key 'harness' must be one of"):
        build_config_from_data({"reviewer": {"harness": "not-a-real-harness"}})
