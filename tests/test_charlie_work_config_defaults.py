"""Config defaults and layered sources: ``test_default_config_*``
(built-in defaults), ``test_global_config_*`` (global-vs-per-repo
layering), and ``test_supervisor_config_*`` (supervisor section).

Split out of ``tests/test_charlie_work.py`` (issue #1552, Track-1
wave 6/8).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from charlie_work.config import (
    SupervisorConfig,
    TestAdequacyConfig,
    load_config,
)


def test_default_config_enables_auto_merge() -> None:
    config = load_config()

    assert config.auto_merge.enabled is True
    # A shared package cannot know a consumer's CI check names; unconfigured
    # means empty, and `doctor` flags it.
    assert config.auto_merge.required_checks == ()
    assert config.labels.ready == "automated-ready"


def test_default_config_failed_attempt_alarm() -> None:
    """Issue #254: default merge attempt alarm threshold is 3."""
    config = load_config()
    assert config.auto_merge.failed_attempt_alarm == 3


def test_default_config_update_open_prs_is_next() -> None:
    """Default update_open_prs is merge-train mode."""
    config = load_config()
    assert config.auto_merge.update_open_prs == "next"


def test_default_config_update_branch_strategy_is_front_of_train() -> None:
    """Issue #404: default update_branch_strategy is front-of-train."""
    config = load_config()
    assert config.auto_merge.update_branch_strategy == "front_of_train"


def test_default_config_require_current_base() -> None:
    """Default require_current_base is True."""
    config = load_config()
    assert config.auto_merge.require_current_base is True


def test_default_config_tee_stream_json_disabled() -> None:
    """ClaudeCodeConfig.tee_stream_json defaults to False (issue #160)."""
    config = load_config()
    assert config.claude_code.tee_stream_json is False


def test_default_config_runner_scaling_disabled() -> None:
    """RunnerScalingConfig.enabled defaults to False (issue #232)."""
    config = load_config()
    assert config.runner_scaling.enabled is False


def test_default_config_throttle_error_markers() -> None:
    """RuntimeConfig.throttle_error_markers defaults to genuine provider
    throttle signatures only.

    Issue #260, corrected premise: "A tool was rejected by the user" was
    originally a default here, but it is the Devin CLI's own surfacing of a
    PreToolUse hook block, not a provider throttle condition — it must never
    be a default throttle marker (retry/cooldown semantics are wrong for a
    hard hook block). See PostMortemConfig.signature_rules for the
    worker_blocked rule that owns that signature instead.
    """
    config = load_config()
    assert "Reached overall message rate limit" in config.runtime.throttle_error_markers
    assert "rate limit" in config.runtime.throttle_error_markers
    assert "too many requests" in config.runtime.throttle_error_markers
    assert "A tool was rejected by the user" not in config.runtime.throttle_error_markers


def test_default_config_session_limit_markers() -> None:
    """RuntimeConfig.session_limit_markers defaults to the narrow CLI
    session-limit death-message phrasing only (issue #651/#652).

    Unlike throttle_error_markers (which includes generic substrings like
    "rate limit" / "usage limit" appropriate for worker log tails), this list
    contains only specific CLI death-message phrasing safe to match against
    reviewer analysis prose. Generic markers must NOT appear here: reviewer
    launches force tee_stream_json=True, making log_path and events_path
    byte-identical, so a marker matched against the log tail is also matched
    against the parsed assistant text -- generic markers would false-positive
    on legitimate rate-limit/quota review commentary.
    """
    config = load_config()
    assert "hit your session limit" in config.runtime.session_limit_markers
    # Generic markers that appear in review commentary must NOT be here.
    assert "rate limit" not in config.runtime.session_limit_markers
    assert "usage limit" not in config.runtime.session_limit_markers
    assert "too many requests" not in config.runtime.session_limit_markers


def test_default_config_disables_test_adequacy() -> None:
    """TestAdequacyConfig defaults to disabled with all default values."""
    config = load_config()

    assert config.test_adequacy == TestAdequacyConfig()
    assert config.test_adequacy.enabled is False


def test_default_config_disables_coverage_probe() -> None:
    """CoverageProbeConfig defaults to disabled with all default values
    (issues #1260/#1261) -- an absent config block is a no-op."""
    from charlie_work.config import CoverageProbeConfig

    config = load_config()

    assert config.coverage_probe == CoverageProbeConfig()
    assert config.coverage_probe.enabled is False


def test_global_config_no_global_file(tmp_path: Path) -> None:
    """Test that load_layered_config behaves like load_config when no global file exists."""
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    # No global config, no repo config
    config = load_layered_config(repo_root, None, fleet_dir_override=str(tmp_path / "fleet"))

    # Should match default config
    default_config = load_config(None)
    assert config.labels.ready == default_config.labels.ready
    assert config.dispatch.default_limit == default_config.dispatch.default_limit


def test_global_config_global_only(tmp_path: Path) -> None:
    """Test that global config values apply when no per-repo override exists."""
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # Create global config with a custom value
    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text("dispatch:\n  max_concurrent_sessions: 5\n", encoding="utf-8")

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))

    assert config.dispatch.max_concurrent_sessions == 5


def test_global_config_per_repo_wins(tmp_path: Path) -> None:
    """Test that per-repo config overrides global config."""
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # Create global config
    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text("dispatch:\n  max_concurrent_sessions: 5\n", encoding="utf-8")

    # Create per-repo config with different value
    repo_config_path = repo_root / "orchestrator.config.yaml"
    repo_config_path.write_text("dispatch:\n  max_concurrent_sessions: 10\n", encoding="utf-8")

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))

    # Per-repo value should win
    assert config.dispatch.max_concurrent_sessions == 10


def test_global_config_unknown_key_raises(tmp_path: Path) -> None:
    """Test that unknown keys in global config raise ConfigError."""
    from charlie_work.config import ConfigError
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # Create global config with unknown top-level section
    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text("unknown_section:\n  foo: bar\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown config section"):
        load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))


def test_global_config_invalid_global_keeps_per_repo_config(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A present-but-invalid global layer must not discard a valid per-repo config.

    Regression for issue #665: when the global config file exists but fails
    validation (e.g. an unknown key), the merged load raises ConfigError.
    Callers (fleet_dispatch) catch ConfigError and skip the repo, silently
    discarding a *valid* per-repo config -- the #623 failure shape (host-wide
    knobs silently disabled) via a different trigger. The per-repo config
    must survive, and the discard must be loud (a warning), not silent.
    """
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # Global layer is present but invalid (unknown top-level section).
    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text("unknown_section:\n  foo: bar\n", encoding="utf-8")

    # Per-repo config is valid and carries a distinctive value.
    repo_config_path = repo_root / "orchestrator.config.yaml"
    repo_config_path.write_text("dispatch:\n  max_concurrent_sessions: 7\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="charlie_work.global_config"):
        config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))

    # The per-repo value survives -- the broken global layer did not discard it.
    assert config.dispatch.max_concurrent_sessions == 7

    # The discard is loud, not silent: a warning names the discarded global path.
    warning = "\n".join(
        r.getMessage() for r in caplog.records if "global layer was discarded" in r.getMessage()
    )
    assert warning, "discarding an invalid global layer must be logged as a warning"
    assert str(global_config_path) in warning, "the warning must name the global path"


def test_global_config_invalid_global_no_per_repo_still_raises(tmp_path: Path) -> None:
    """With no per-repo config to rescue, an invalid global layer must still raise.

    Silently defaulting here would itself reproduce the #623 shape (host-wide
    knobs silently disabled): the operator's only config is broken, and a
    pristine-defaults return hides that. The error must propagate.
    """
    from charlie_work.config import ConfigError
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # Global layer is present but invalid; no per-repo config exists.
    (fleet_dir_path / "config.yaml").write_text("unknown_section:\n  foo: bar\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown config section"):
        load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))


def test_global_config_invalid_per_repo_still_raises(tmp_path: Path) -> None:
    """An invalid *per-repo* config must still raise even with a valid global layer.

    The fallback rescues a valid per-repo config from a broken global layer, not
    the reverse: when the per-repo config is itself broken, the error propagates
    so the operator learns their per-repo file is invalid.
    """
    from charlie_work.config import ConfigError
    from charlie_work.global_config import load_layered_config

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    # Global layer is valid.
    (fleet_dir_path / "config.yaml").write_text(
        "dispatch:\n  max_concurrent_sessions: 5\n", encoding="utf-8"
    )
    # Per-repo config is invalid (unknown section).
    (repo_root / "orchestrator.config.yaml").write_text(
        "unknown_section:\n  foo: bar\n", encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="unknown config section"):
        load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))


def test_supervisor_config_defaults() -> None:
    """SupervisorConfig defaults are stable and load_config picks them up."""
    config = load_config()
    assert isinstance(config.supervisor, SupervisorConfig)
    assert config.supervisor.poll_interval_seconds == 20
    assert config.supervisor.full_pass_interval_seconds == 300
    assert config.supervisor.active_cooldown_seconds == 30
    assert config.supervisor.max_runtime_minutes == 0
    assert config.supervisor.max_pass_runtime_seconds == 1800
    assert config.supervisor.self_deploy_failure_alarm == 3
    assert config.supervisor.zero_pass_alarm == 3


def test_supervisor_config_parses_custom_values(tmp_path: Path) -> None:
    """Custom supervisor section values are parsed correctly."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
supervisor:
  poll_interval_seconds: 10
  full_pass_interval_seconds: 120
  active_cooldown_seconds: 15
  max_runtime_minutes: 60
  max_pass_runtime_seconds: 900
  self_deploy_failure_alarm: 5
  zero_pass_alarm: 7
"""
    )
    config = load_config(config_file)
    assert config.supervisor.poll_interval_seconds == 10
    assert config.supervisor.full_pass_interval_seconds == 120
    assert config.supervisor.active_cooldown_seconds == 15
    assert config.supervisor.max_runtime_minutes == 60
    assert config.supervisor.max_pass_runtime_seconds == 900
    assert config.supervisor.self_deploy_failure_alarm == 5
    assert config.supervisor.zero_pass_alarm == 7


def test_supervisor_config_unknown_key_raises(tmp_path: Path) -> None:
    """Unknown keys in supervisor section raise ConfigError."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
supervisor:
  poll_interval_seconds: 10
  unknown_key: 99
"""
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(config_file)


def test_supervisor_config_wrong_type_raises(tmp_path: Path) -> None:
    """Wrong types in supervisor section raise ConfigError."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
supervisor:
  poll_interval_seconds: "not-an-int"
"""
    )
    with pytest.raises(ConfigError, match="must be an int"):
        load_config(config_file)


def test_supervisor_config_self_deploy_failure_alarm_wrong_type_raises(tmp_path: Path) -> None:
    """Wrong type for self_deploy_failure_alarm raises ConfigError.

    Issue #817 item 5 added this field alongside the existing supervisor int
    fields; the supervisor section has its own manual int-type-validation
    tuple in config.py (separate from the generic _build_section machinery),
    which needed the new key added explicitly. Locks that in so a future
    refactor of the tuple can't silently drop validation for this field.
    """
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
supervisor:
  self_deploy_failure_alarm: "not-an-int"
"""
    )
    with pytest.raises(ConfigError, match="must be an int"):
        load_config(config_file)


def test_supervisor_config_zero_pass_alarm_wrong_type_raises(tmp_path: Path) -> None:
    """Wrong type for zero_pass_alarm raises ConfigError.

    Issue #855 added this field alongside the existing supervisor int
    fields; the supervisor section has its own manual int-type-validation
    tuple in config.py (separate from the generic _build_section machinery),
    which needed the new key added explicitly -- mirrors
    test_supervisor_config_self_deploy_failure_alarm_wrong_type_raises.
    Locks that in so a future refactor of the tuple can't silently drop
    validation for this field.
    """
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
supervisor:
  zero_pass_alarm: "not-an-int"
"""
    )
    with pytest.raises(ConfigError, match="must be an int"):
        load_config(config_file)


def test_supervisor_config_is_frozen() -> None:
    """SupervisorConfig is a frozen dataclass."""
    import dataclasses

    cfg = SupervisorConfig()
    assert dataclasses.is_dataclass(cfg)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        cfg.poll_interval_seconds = 99  # type: ignore[misc]
