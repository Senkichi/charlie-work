"""Config dataclass parsing/defaults/validation and config-path discovery.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import errno
from pathlib import Path
import pytest
from _fakes_github import FakeGitHub
from _helpers import EXAMPLES_DIR
from charlie_work.config import (
    AutoMergeConfig,
    OrchestratorConfig,
    SignatureRule,
    find_config_path,
    load_config,
)
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
import charlie_work.state as state_module


def test_runner_scaling_config_parses_with_enabled_flag(tmp_path: Path) -> None:
    """RunnerScalingConfig parses with enabled=true and custom values."""
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
runner_scaling:
  enabled: true
  managed_root: "C:\\\\actions-runners"
  runner_dir_prefix: "jc-"
  runner_name_template: "jc-selfhost-{n}"
  package_zip: "C:\\\\packages\\\\runner.zip"
  min_runners: 2
  max_runners: 20
  ram_per_job_gb: 4.0
  min_free_ram_gb: 8.0
  max_host_cpu_pct: 90.0
  idle_scale_down_minutes: 30
  cooldown_minutes: 10
"""
    )
    config = load_config(config_file)
    assert config.runner_scaling.enabled is True
    assert config.runner_scaling.managed_root == "C:\\actions-runners"
    assert config.runner_scaling.runner_dir_prefix == "jc-"
    assert config.runner_scaling.runner_name_template == "jc-selfhost-{n}"
    assert config.runner_scaling.package_zip == "C:\\packages\\runner.zip"
    assert config.runner_scaling.min_runners == 2
    assert config.runner_scaling.max_runners == 20
    assert config.runner_scaling.ram_per_job_gb == 4.0
    assert config.runner_scaling.min_free_ram_gb == 8.0
    assert config.runner_scaling.max_host_cpu_pct == 90.0
    assert config.runner_scaling.idle_scale_down_minutes == 30
    assert config.runner_scaling.cooldown_minutes == 10


def test_runner_scaling_config_rejects_invalid_numeric_types(tmp_path: Path) -> None:
    """RunnerScalingConfig rejects non-numeric values for numeric fields."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
runner_scaling:
  enabled: true
  min_runners: "not-a-number"
"""
    )
    with pytest.raises(ConfigError, match="must be an int"):
        load_config(config_file)


def test_runner_scaling_config_rejects_invalid_string_types(tmp_path: Path) -> None:
    """RunnerScalingConfig rejects non-string values for string fields."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
runner_scaling:
  enabled: true
  managed_root: 123
"""
    )
    with pytest.raises(ConfigError, match="must be a string"):
        load_config(config_file)


def test_runner_scaling_config_rejects_invalid_boolean_type(tmp_path: Path) -> None:
    """RunnerScalingConfig rejects non-boolean values for enabled field."""
    from charlie_work.config import ConfigError

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
runner_scaling:
  enabled: "true"
"""
    )
    with pytest.raises(ConfigError, match="must be a bool"):
        load_config(config_file)


def test_auto_merge_config_rejects_stale_base_deadlock(tmp_path: Path) -> None:
    """Issue #368: require_current_base=True + update_open_prs='off' is a silent
    permanent merge deadlock, so it is rejected at config construction.
    """
    from charlie_work.config import AutoMergeConfig, ConfigError, OrchestratorConfig, load_config

    with pytest.raises(ConfigError, match="permanent merge deadlock"):
        AutoMergeConfig(require_current_base=True, update_open_prs="off")

    with pytest.raises(ConfigError, match="permanent merge deadlock"):
        AutoMergeConfig(require_current_base=True, update_open_prs=False)

    with pytest.raises(ConfigError, match="permanent merge deadlock"):
        OrchestratorConfig(
            auto_merge=AutoMergeConfig(require_current_base=True, update_open_prs="off")
        )

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        """
auto_merge:
  update_open_prs: off
"""
    )
    with pytest.raises(ConfigError, match="permanent merge deadlock"):
        load_config(config_file)

    # Coherent combinations load without error.
    assert AutoMergeConfig(require_current_base=False, update_open_prs="off")
    assert AutoMergeConfig(require_current_base=True, update_open_prs="next")
    assert AutoMergeConfig(require_current_base=True, update_open_prs="all")


def test_auto_merge_config_mergequeue_label_defaults_to_none(tmp_path: Path) -> None:
    """Aviator MergeQueue handoff (task #10) is off by default: the default
    AutoMergeConfig() must preserve today's self-merge behavior byte-for-byte."""
    from charlie_work.config import AutoMergeConfig

    assert AutoMergeConfig().mergequeue_label is None


def test_coverage_probe_config_is_frozen() -> None:
    from charlie_work.config import CoverageProbeConfig
    from dataclasses import FrozenInstanceError

    config = CoverageProbeConfig()
    try:
        config.enabled = True  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except FrozenInstanceError:
        pass


def test_orchestrator_config_ctor_wires_coverage_probe_field() -> None:
    """OrchestratorConfig() carries a coverage_probe field with defaults."""
    from charlie_work.config import CoverageProbeConfig

    config = OrchestratorConfig()

    assert config.coverage_probe == CoverageProbeConfig()


def test_find_config_path_prefers_explicit_then_repo_root(tmp_path: Path) -> None:
    explicit = tmp_path / "elsewhere.yaml"
    assert find_config_path(tmp_path, explicit) == explicit

    assert find_config_path(tmp_path) is None

    repo_config = tmp_path / "orchestrator.config.yaml"
    repo_config.write_text("labels:\n  ready: automated-ready\n", encoding="utf-8")
    assert find_config_path(tmp_path) == repo_config


def test_auto_merge_config_queue_bot_login_defaults_to_none(tmp_path: Path) -> None:
    """Issue #1194: the default AutoMergeConfig() must disable queue sync-merge
    recognition entirely, preserving today's #502 tripwire behavior."""
    assert AutoMergeConfig().queue_bot_login is None


def test_auto_merge_config_mergequeue_wedge_hours_defaults_to_24() -> None:
    """Issue #1401: default AutoMergeConfig() enables the time-in-mergequeue
    watchdog at 24h -- the live #1751 case ran 28h+ undetected, so the default
    must be on, not opt-in."""
    assert AutoMergeConfig().mergequeue_wedge_hours == 24.0


def test_claude_code_example_config_selects_claude_worker() -> None:
    config = load_config(EXAMPLES_DIR / "orchestrator.config.claude-code.yaml")

    assert config.dispatch.worker_template == "worker_claude_code.md"


def test_claude_code_example_config_sets_bounded_xdist_worker_env() -> None:
    config = load_config(EXAMPLES_DIR / "orchestrator.config.claude-code.yaml")

    # The shipped example bounds local test parallelism at the launch boundary
    # (the RUNBOOK "Local host saturation ceiling" section references this).
    assert config.claude_code.worker_env == {"PYTEST_XDIST_AUTO_NUM_WORKERS": "2"}


def test_describe_config_file_separates_absent_from_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config that could not be reached must not read as one that is absent.

    ``Path.exists()`` returns a bare False for every error in
    ``pathlib._ignore_error`` -- ENOENT, ENOTDIR, EBADF, ELOOP and the Windows
    unready-device / unresolvable-path winerrors. All of them take
    load_layered_config's silent-``{}`` branch and yield pristine dataclass
    defaults with no error raised, which is #590's symptom exactly. The cause
    must survive into the log instead of collapsing into "absent".

    ENOTDIR is used rather than EACCES deliberately: permission errors are *not*
    in the ignored set, so they raise instead of silently defaulting, which is
    why they cannot be #590's mechanism.
    """
    from charlie_work.global_config import describe_config_file

    missing = tmp_path / "nope.yaml"
    assert describe_config_file(missing) == "absent"

    present = tmp_path / "config.yaml"
    present.write_text("dispatch: {}\n", encoding="utf-8")
    assert describe_config_file(present) == f"present bytes={present.stat().st_size}"

    real_stat = Path.stat

    def not_a_dir(self: Path, *args: object, **kwargs: object) -> object:
        if self == present:
            raise NotADirectoryError(errno.ENOTDIR, "Not a directory")
        return real_stat(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", not_a_dir)

    # The precondition that makes this helper necessary: exists() hides this.
    assert present.exists() is False, "precondition: exists() collapses ENOTDIR to False"

    described = describe_config_file(present)
    assert described != "absent", "an unreachable config must not read as absent"
    assert described.startswith("UNREADABLE"), f"cause was lost: {described!r}"
    assert "NotADirectoryError" in described, "the failure cause must reach the log"


def test_signature_rule_is_frozen() -> None:
    """SignatureRule is a frozen dataclass."""
    import dataclasses

    rule = SignatureRule(pattern="x", kind="worker_blocked")
    assert dataclasses.is_dataclass(rule)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        rule.kind = "other"  # type: ignore[misc]


def test_orchestrator_app_init_wires_event_ring_size_from_config(tmp_path: Path) -> None:
    """Issue #525: OrchestratorApp.__init__ sets state.EVENT_RING_SIZE from
    RuntimeConfig.event_ring_size so the default append_event cap is
    config-driven. A regression here silently leaves the ring at the hardcoded
    default regardless of operator config."""
    from charlie_work.config import RuntimeConfig

    custom_size = 7777
    config = OrchestratorConfig(runtime=RuntimeConfig(event_ring_size=custom_size))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    # Snapshot the module global before construction and restore it after so
    # the test does not leak the override into other tests in the same process.
    saved = state_module.EVENT_RING_SIZE
    try:
        OrchestratorApp(tmp_path, paths, config, FakeGitHub())
        assert state_module.EVENT_RING_SIZE == custom_size
    finally:
        state_module.EVENT_RING_SIZE = saved
