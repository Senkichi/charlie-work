"""Tests for config.py validation, especially new config keys."""

from __future__ import annotations

from pathlib import Path

import pytest
from types import MappingProxyType

from charlie_work.config import (
    ApiBudgetConfig,
    ApiProviderConfig,
    ApiWorkerConfig,
    ConfigError,
    DispatchConfig,
    OrchestratorConfig,
    RuntimeConfig,
    load_config,
)
from charlie_work.global_config import load_layered_config


def _write_config(config_file: Path, content: str) -> None:
    config_file.write_text(content, encoding="utf-8")


def test_load_config_worktree_mtime_enabled_rejects_non_bool(tmp_path: Path) -> None:
    """Issue #353: watchdog.worktree_mtime_enabled must be a bool."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
watchdog:
  worktree_mtime_enabled: "true"
""",
    )
    with pytest.raises(ConfigError, match="worktree_mtime_enabled.*must be a bool"):
        load_config(config_file)


def test_load_config_worktree_mtime_threshold_minutes_rejects_non_int(tmp_path: Path) -> None:
    """Issue #353: watchdog.worktree_mtime_threshold_minutes must be an int."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
watchdog:
  worktree_mtime_threshold_minutes: "45"
""",
    )
    with pytest.raises(ConfigError, match="worktree_mtime_threshold_minutes.*must be an int"):
        load_config(config_file)


def test_load_config_worktree_mtime_max_depth_rejects_non_int(tmp_path: Path) -> None:
    """Issue #353: watchdog.worktree_mtime_max_depth must be an int."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
watchdog:
  worktree_mtime_max_depth: "4"
""",
    )
    with pytest.raises(ConfigError, match="worktree_mtime_max_depth.*must be an int"):
        load_config(config_file)


def test_load_config_worktree_mtime_exclude_dirs_rejects_non_string_element(
    tmp_path: Path,
) -> None:
    """Issue #353: watchdog.worktree_mtime_exclude_dirs must be a list of strings."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
watchdog:
  worktree_mtime_exclude_dirs:
    - .git
    - 42
""",
    )
    with pytest.raises(
        ConfigError, match="worktree_mtime_exclude_dirs.*must be a list of strings"
    ):
        load_config(config_file)


def test_load_config_worktree_mtime_exclude_dirs_coerces_list_to_tuple(
    tmp_path: Path,
) -> None:
    """Issue #353: watchdog.worktree_mtime_exclude_dirs is a list in YAML and a tuple in config."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
watchdog:
  worktree_mtime_enabled: true
  worktree_mtime_threshold_minutes: 45
  worktree_mtime_max_depth: 4
  worktree_mtime_exclude_dirs:
    - .git
    - .venv
""",
    )
    config = load_config(config_file)
    assert config.watchdog.worktree_mtime_enabled is True
    assert config.watchdog.worktree_mtime_threshold_minutes == 45
    assert config.watchdog.worktree_mtime_max_depth == 4
    assert config.watchdog.worktree_mtime_exclude_dirs == (".git", ".venv")


def test_dispatch_config_injected_paths_default_excludes_injected_files() -> None:
    """Issue #381/#400: default injected_paths exclude the in-worktree Claude Code prompt file and writer marker."""
    from charlie_work.config import CLAUDE_CODE_PROMPT_FILENAME, WRITER_MARKER_FILENAME

    config = DispatchConfig()
    assert config.injected_paths == (CLAUDE_CODE_PROMPT_FILENAME, WRITER_MARKER_FILENAME)


def test_claude_code_prompt_filename_in_default_injected_paths() -> None:
    """Issue #381: the Claude Code prompt file is excluded from dirty checks by default."""
    from charlie_work.claude_code import PROMPT_FILENAME
    from charlie_work.config import CLAUDE_CODE_PROMPT_FILENAME

    assert PROMPT_FILENAME is CLAUDE_CODE_PROMPT_FILENAME
    assert CLAUDE_CODE_PROMPT_FILENAME in DispatchConfig().injected_paths


def test_load_config_injected_paths_coerces_list_to_tuple(tmp_path: Path) -> None:
    """Issue #381/#400: injected_paths list in YAML becomes a tuple and the writer marker is always appended."""
    from charlie_work.config import WRITER_MARKER_FILENAME

    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
dispatch:
  injected_paths:
    - .orchestrator-prompt.md
    - .devin/prompts/custom.md
""",
    )
    config = load_config(config_file)
    assert config.dispatch.injected_paths == (
        ".orchestrator-prompt.md",
        ".devin/prompts/custom.md",
        WRITER_MARKER_FILENAME,
    )


def test_dispatch_config_injected_paths_normalizes_backslashes() -> None:
    """Issue #381/#400: backslash separators in an override are normalized to '/' and the writer marker is appended."""
    from charlie_work.config import WRITER_MARKER_FILENAME

    config = DispatchConfig(injected_paths=[r".devin\prompts\worker.md"])
    assert config.injected_paths == (".devin/prompts/worker.md", WRITER_MARKER_FILENAME)


def test_load_config_injected_paths_normalizes_backslashes(tmp_path: Path) -> None:
    """Issue #381/#400: YAML override with Windows-style separators is normalized and the writer marker is appended."""
    from charlie_work.config import WRITER_MARKER_FILENAME

    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        r"""dispatch:
  injected_paths:
    - '.devin\prompts\worker.md'
""",
    )
    config = load_config(config_file)
    assert config.dispatch.injected_paths == (".devin/prompts/worker.md", WRITER_MARKER_FILENAME)


def test_load_config_review_dispatch_defaults() -> None:
    """ReviewDispatchConfig defaults are safe (disabled, separate dir, bounded)."""
    config_file = Path("nonexistent.yaml")
    config = load_config(config_file)
    assert config.review_dispatch.enabled is False
    assert config.review_dispatch.reviews_dir == ".var/charlie-work/dispatches/reviews"
    assert config.review_dispatch.max_local_review_processes == 2


def test_load_config_review_dispatch_override(tmp_path: Path) -> None:
    """review_dispatch keys are read from YAML."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review_dispatch:
  enabled: true
  reviews_dir: .var/reviews
  max_local_review_processes: 4
""",
    )
    config = load_config(config_file)
    assert config.review_dispatch.enabled is True
    assert config.review_dispatch.reviews_dir == ".var/reviews"
    assert config.review_dispatch.max_local_review_processes == 4


def test_load_config_review_effort_experiment_defaults() -> None:
    """review_effort_experiment_fraction/salt default to disabled (0.0/'')."""
    config = load_config(Path("nonexistent.yaml"))
    assert config.review_dispatch.review_effort_experiment_fraction == 0.0
    assert config.review_dispatch.review_effort_experiment_salt == ""


def test_load_config_review_effort_experiment_override(tmp_path: Path) -> None:
    """review_effort_experiment_fraction/salt are read from YAML."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review_dispatch:
  review_effort: medium
  review_effort_experiment_fraction: 0.25
  review_effort_experiment_salt: epoch-2
""",
    )
    config = load_config(config_file)
    assert config.review_dispatch.review_effort == "medium"
    assert config.review_dispatch.review_effort_experiment_fraction == 0.25
    assert config.review_dispatch.review_effort_experiment_salt == "epoch-2"


def test_load_config_review_effort_experiment_fraction_rejects_bool(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review_dispatch:
  review_effort_experiment_fraction: true
""",
    )
    with pytest.raises(ConfigError, match="review_effort_experiment_fraction.*must be a number"):
        load_config(config_file)


def test_load_config_review_effort_experiment_fraction_rejects_non_number(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review_dispatch:
  review_effort_experiment_fraction: "0.5"
""",
    )
    with pytest.raises(ConfigError, match="review_effort_experiment_fraction.*must be a number"):
        load_config(config_file)


@pytest.mark.parametrize("value", [-0.01, 1.01, 2, -1])
def test_load_config_review_effort_experiment_fraction_rejects_out_of_range(
    tmp_path: Path, value: float
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        f"""review_dispatch:
  review_effort_experiment_fraction: {value}
""",
    )
    with pytest.raises(
        ConfigError, match=r"review_effort_experiment_fraction.*must be in \[0.0, 1.0\]"
    ):
        load_config(config_file)


def test_load_config_review_effort_experiment_fraction_without_effort_rejected(
    tmp_path: Path,
) -> None:
    """fraction > 0.0 with review_effort unset (default '') must fail loud at
    load time -- treatment would otherwise silently mean 'no --effort pin'."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review_dispatch:
  review_effort_experiment_fraction: 0.25
""",
    )
    with pytest.raises(
        ConfigError,
        match="review_effort_experiment_fraction.*is 0.25 but 'review_effort' is unset",
    ):
        load_config(config_file)


def test_load_config_review_effort_experiment_fraction_with_effort_accepted(
    tmp_path: Path,
) -> None:
    """fraction > 0.0 WITH review_effort set is a valid, accepted config."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review_dispatch:
  review_effort: high
  review_effort_experiment_fraction: 0.25
""",
    )
    config = load_config(config_file)
    assert config.review_dispatch.review_effort == "high"
    assert config.review_dispatch.review_effort_experiment_fraction == 0.25


def test_load_config_review_effort_experiment_fraction_zero_without_effort_accepted() -> None:
    """The default config (fraction=0.0, review_effort unset) must keep loading."""
    config = load_config(Path("nonexistent.yaml"))
    assert config.review_dispatch.review_effort_experiment_fraction == 0.0
    assert config.review_dispatch.review_effort == ""


def test_load_config_review_effort_experiment_salt_rejects_non_str(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review_dispatch:
  review_effort_experiment_salt: 123
""",
    )
    with pytest.raises(ConfigError, match="review_effort_experiment_salt.*must be a string"):
        load_config(config_file)


def test_runtime_config_event_ring_size_default() -> None:
    """Issue #525: RuntimeConfig.event_ring_size defaults to 2000."""
    assert RuntimeConfig().event_ring_size == 2000


def test_load_config_event_ring_size_override(tmp_path: Path) -> None:
    """Issue #525: runtime.event_ring_size is configurable from YAML."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  event_ring_size: 5000
""",
    )
    config = load_config(config_file)
    assert config.runtime.event_ring_size == 5000


def test_load_config_event_ring_size_rejects_invalid(tmp_path: Path) -> None:
    """Issue #525: runtime.event_ring_size must be an int."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  event_ring_size: "lots"
""",
    )
    with pytest.raises(ConfigError, match="event_ring_size.*must be an int"):
        load_config(config_file)


def test_load_config_event_ring_size_rejects_zero(tmp_path: Path) -> None:
    """Issue #525: event_ring_size=0 is rejected — it would disable truncation.

    append_event truncates via events[-max_size:]; because -0 == 0 in Python,
    max_size=0 yields events[0:] (the FULL list), causing unbounded growth —
    the exact failure the cap exists to prevent. There is no sensible "disable"
    semantic for a bounded ring, so 0 is invalid.
    """
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  event_ring_size: 0
""",
    )
    with pytest.raises(ConfigError, match="event_ring_size.*must be >= 1"):
        load_config(config_file)


def test_load_config_event_ring_size_rejects_negative(tmp_path: Path) -> None:
    """Issue #525: negative event_ring_size is rejected."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  event_ring_size: -5
""",
    )
    with pytest.raises(ConfigError, match="event_ring_size.*must be >= 1"):
        load_config(config_file)


def test_load_config_runtime_throttle_resume_margin_default() -> None:
    """RuntimeConfig.throttle_resume_margin_s defaults to 90 seconds."""
    from charlie_work.config import RuntimeConfig

    config = RuntimeConfig()
    assert config.throttle_resume_margin_s == 90


def test_load_config_runtime_throttle_resume_margin_override(tmp_path: Path) -> None:
    """runtime.throttle_resume_margin_s is read from YAML."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  throttle_resume_margin_s: 120
""",
    )
    config = load_config(config_file)
    assert config.runtime.throttle_resume_margin_s == 120


def test_load_config_runtime_throttle_resume_margin_rejects_non_int(
    tmp_path: Path,
) -> None:
    """runtime.throttle_resume_margin_s must be an int."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  throttle_resume_margin_s: "90"
""",
    )
    with pytest.raises(ConfigError, match="throttle_resume_margin_s.*must be an int"):
        load_config(config_file)


def test_load_config_runtime_throttle_resume_margin_rejects_negative(
    tmp_path: Path,
) -> None:
    """runtime.throttle_resume_margin_s must be >= 0."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  throttle_resume_margin_s: -1
""",
    )
    with pytest.raises(ConfigError, match="throttle_resume_margin_s.*must be >= 0"):
        load_config(config_file)


def test_load_config_readiness_no_ci_minutes_rejects_bool_true(tmp_path: Path) -> None:
    """Issue #474: YAML boolean true is not a valid integer timeout."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """auto_merge:
  readiness_no_ci_minutes: true
""",
    )
    with pytest.raises(ConfigError, match="readiness_no_ci_minutes.*must be an int"):
        load_config(config_file)


def test_load_config_readiness_no_ci_minutes_rejects_bool_false(tmp_path: Path) -> None:
    """Issue #474: YAML boolean false silently disables the gate if treated as 0."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """auto_merge:
  readiness_no_ci_minutes: false
""",
    )
    with pytest.raises(ConfigError, match="readiness_no_ci_minutes.*must be an int"):
        load_config(config_file)


def test_load_config_readiness_no_ci_minutes_rejects_negative(tmp_path: Path) -> None:
    """Issue #474: negative timeout is semantically meaningless."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """auto_merge:
  readiness_no_ci_minutes: -1
""",
    )
    with pytest.raises(ConfigError, match="readiness_no_ci_minutes.*must not be negative"):
        load_config(config_file)


def test_load_config_readiness_no_ci_minutes_accepts_valid_int(tmp_path: Path) -> None:
    """Issue #474: zero disables the gate; positive integers enable it."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """auto_merge:
  readiness_no_ci_minutes: 0
""",
    )
    config = load_config(config_file)
    assert config.auto_merge.readiness_no_ci_minutes == 0

    _write_config(
        config_file,
        """auto_merge:
  readiness_no_ci_minutes: 30
""",
    )
    config = load_config(config_file)
    assert config.auto_merge.readiness_no_ci_minutes == 30


def test_api_worker_config_defaults() -> None:
    """An absent api_worker section yields safe defaults and no behavior change."""
    config = load_config()
    assert config.api_worker.enabled is False
    assert config.api_worker.provider == ""
    assert config.api_worker.max_concurrent_sessions == 1
    assert config.api_worker.fallback_adapter == "devin-shell"
    assert config.api_worker.worker_template == "worker_claude_code.md"
    assert config.api_worker.rework_template == "rework.md"
    assert isinstance(config.api_worker.providers, MappingProxyType)
    assert len(config.api_worker.providers) == 0
    assert isinstance(config.api_worker.budget, ApiBudgetConfig)
    assert config.api_worker.budget.max_usd_per_session == 0.0
    assert config.api_worker.budget.preflight_reserve_usd == 1.0
    assert config.api_worker.budget.max_usd_per_day == 5.0
    assert config.api_worker.budget.lifetime_usd == 15.0


API_WORKER_SAMPLE = """api_worker:
  enabled: false
  provider: kimi-k3
  max_concurrent_sessions: 1
  providers:
    kimi-k3:
      base_url: https://api.moonshot.ai/anthropic
      api_key_env: MOONSHOT_API_KEY
      model: kimi-k3
      input_usd_per_mtok: 3.0
      output_usd_per_mtok: 15.0
      cached_input_usd_per_mtok: 0.30
  budget:
    max_usd_per_session: 0
    preflight_reserve_usd: 1.00
    max_usd_per_day: 5.00
    lifetime_usd: 15.00
  fallback_adapter: devin-shell
  worker_template: worker_claude_code.md
  rework_template: rework.md
"""


def test_load_config_api_worker_parses_and_round_trips(tmp_path: Path) -> None:
    """A complete api_worker block parses into the expected frozen dataclasses."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, API_WORKER_SAMPLE)
    config = load_config(config_file)

    assert config.api_worker.enabled is False
    assert config.api_worker.provider == "kimi-k3"
    assert isinstance(config.api_worker.providers, MappingProxyType)
    assert len(config.api_worker.providers) == 1
    assert "kimi-k3" in config.api_worker.providers
    assert config.api_worker.providers["kimi-k3"] == ApiProviderConfig(
        base_url="https://api.moonshot.ai/anthropic",
        api_key_env="MOONSHOT_API_KEY",
        model="kimi-k3",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cached_input_usd_per_mtok=0.30,
    )
    assert config.api_worker.budget == ApiBudgetConfig(
        max_usd_per_session=0.0,
        preflight_reserve_usd=1.0,
        max_usd_per_day=5.0,
        lifetime_usd=15.0,
    )


def test_api_worker_providers_mapping_is_immutable(tmp_path: Path) -> None:
    """The providers registry is a MappingProxyType and cannot be mutated after load."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, API_WORKER_SAMPLE)
    config = load_config(config_file)

    with pytest.raises(TypeError):
        config.api_worker.providers["new"] = ApiProviderConfig(
            base_url="https://example.com",
            api_key_env="KEY",
            model="m",
            input_usd_per_mtok=1.0,
            output_usd_per_mtok=2.0,
            cached_input_usd_per_mtok=0.1,
        )


def test_load_config_api_worker_enabled_validates_active_provider(tmp_path: Path) -> None:
    """When enabled, the named provider must exist and have positive pricing."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """api_worker:
  enabled: true
  provider: kimi-k3
  providers:
    kimi-k3:
      base_url: https://api.moonshot.ai/anthropic
      api_key_env: MOONSHOT_API_KEY
      model: kimi-k3
      input_usd_per_mtok: 3.0
      output_usd_per_mtok: 15.0
      cached_input_usd_per_mtok: 0.30
""",
    )
    config = load_config(config_file)
    assert config.api_worker.enabled is True
    assert config.api_worker.provider == "kimi-k3"


def test_load_config_api_worker_enabled_rejects_unknown_provider(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """api_worker:
  enabled: true
  provider: missing
  providers:
    kimi-k3:
      base_url: https://api.moonshot.ai/anthropic
      api_key_env: MOONSHOT_API_KEY
      model: kimi-k3
      input_usd_per_mtok: 3.0
      output_usd_per_mtok: 15.0
      cached_input_usd_per_mtok: 0.30
""",
    )
    with pytest.raises(ConfigError, match="not a key in api_worker.providers"):
        load_config(config_file)


def test_load_config_api_worker_enabled_rejects_empty_api_key_env(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """api_worker:
  enabled: true
  provider: kimi-k3
  providers:
    kimi-k3:
      base_url: https://api.moonshot.ai/anthropic
      api_key_env: ""
      model: kimi-k3
      input_usd_per_mtok: 3.0
      output_usd_per_mtok: 15.0
      cached_input_usd_per_mtok: 0.30
""",
    )
    with pytest.raises(ConfigError, match="api_key_env.*must be a non-empty string"):
        load_config(config_file)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_usd_per_mtok", 0),
        ("input_usd_per_mtok", -1.0),
        ("output_usd_per_mtok", 0.0),
        ("output_usd_per_mtok", -1.0),
    ],
)
def test_load_config_api_worker_enabled_rejects_non_positive_pricing(
    tmp_path: Path, field: str, value: float
) -> None:
    """Active provider input/output pricing must be strictly positive."""
    config_file = tmp_path / "orchestrator.config.yaml"
    prices = {
        "input_usd_per_mtok": 3.0,
        "output_usd_per_mtok": 15.0,
        "cached_input_usd_per_mtok": 0.30,
    }
    prices[field] = value
    price_lines = "\n".join(f"      {k}: {v}" for k, v in prices.items())
    provider_yaml = (
        "      base_url: https://api.moonshot.ai/anthropic\n"
        "      api_key_env: MOONSHOT_API_KEY\n"
        f"      model: kimi-k3\n{price_lines}\n"
    )
    _write_config(
        config_file,
        f"""api_worker:
  enabled: true
  provider: kimi-k3
  providers:
    kimi-k3:
{provider_yaml}""",
    )
    with pytest.raises(ConfigError, match=f"{field}.*must be > 0"):
        load_config(config_file)


def test_load_config_api_worker_enabled_accepts_zero_cached_input_pricing(
    tmp_path: Path,
) -> None:
    """A provider with no cached-input discount may set cached_input_usd_per_mtok to 0."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """api_worker:
  enabled: true
  provider: kimi-k3
  providers:
    kimi-k3:
      base_url: https://api.moonshot.ai/anthropic
      api_key_env: MOONSHOT_API_KEY
      model: kimi-k3
      input_usd_per_mtok: 3.0
      output_usd_per_mtok: 15.0
      cached_input_usd_per_mtok: 0.0
""",
    )
    config = load_config(config_file)
    assert config.api_worker.enabled is True
    assert config.api_worker.providers["kimi-k3"].cached_input_usd_per_mtok == 0.0


def test_load_config_api_worker_enabled_rejects_negative_cached_input_pricing(
    tmp_path: Path,
) -> None:
    """Cached-input pricing must be non-negative."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """api_worker:
  enabled: true
  provider: kimi-k3
  providers:
    kimi-k3:
      base_url: https://api.moonshot.ai/anthropic
      api_key_env: MOONSHOT_API_KEY
      model: kimi-k3
      input_usd_per_mtok: 3.0
      output_usd_per_mtok: 15.0
      cached_input_usd_per_mtok: -0.1
""",
    )
    with pytest.raises(ConfigError, match="cached_input_usd_per_mtok.*must be >= 0"):
        load_config(config_file)


def test_load_config_api_worker_rejects_missing_required_provider_key(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """api_worker:
  enabled: false
  provider: kimi-k3
  providers:
    kimi-k3:
      api_key_env: MOONSHOT_API_KEY
      model: kimi-k3
      input_usd_per_mtok: 3.0
      output_usd_per_mtok: 15.0
""",
    )
    with pytest.raises(ConfigError, match="missing required key 'base_url'"):
        load_config(config_file)


def test_load_config_api_worker_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """api_worker:
  unknown_key: value
""",
    )
    with pytest.raises(ConfigError, match=r"unknown key\(s\) in config section 'api_worker'"):
        load_config(config_file)


def test_load_config_api_worker_rejects_unknown_provider_key(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """api_worker:
  enabled: false
  providers:
    kimi-k3:
      base_url: https://api.moonshot.ai/anthropic
      api_key_env: MOONSHOT_API_KEY
      model: kimi-k3
      input_usd_per_mtok: 3.0
      output_usd_per_mtok: 15.0
      cached_input_usd_per_mtok: 0.30
      extra: value
""",
    )
    with pytest.raises(ConfigError, match=r"providers\.kimi-k3.*has unknown key\(s\)"):
        load_config(config_file)


@pytest.mark.parametrize("value", [True, False])
def test_load_config_api_worker_rejects_bool_max_concurrent_sessions(
    tmp_path: Path, value: bool
) -> None:
    """Boolean values are not valid ints for max_concurrent_sessions."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        f"""api_worker:
  max_concurrent_sessions: {str(value).lower()}
""",
    )
    with pytest.raises(ConfigError, match="max_concurrent_sessions.*must be an int"):
        load_config(config_file)


def test_load_config_api_worker_rejects_invalid_types(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """api_worker:
  enabled: "true"
  max_concurrent_sessions: "one"
  budget: not-a-mapping
  providers:
    - not-a-mapping
""",
    )
    with pytest.raises(ConfigError, match="enabled.*must be a bool"):
        load_config(config_file)


def test_load_config_api_worker_rejects_budget_negative(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """api_worker:
  budget:
    max_usd_per_session: -1.0
""",
    )
    with pytest.raises(ConfigError, match="budget.max_usd_per_session.*must be >= 0"):
        load_config(config_file)


def test_global_config_api_worker_layered_merge(tmp_path: Path) -> None:
    """Fleet-level config sets api_worker; per-repo keys override it section-wise."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text(API_WORKER_SAMPLE, encoding="utf-8")

    repo_config_path = repo_root / "orchestrator.config.yaml"
    repo_config_path.write_text(
        """api_worker:
  enabled: true
""",
        encoding="utf-8",
    )

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))

    assert config.api_worker.enabled is True
    assert config.api_worker.provider == "kimi-k3"
    assert "kimi-k3" in config.api_worker.providers
    assert config.api_worker.providers["kimi-k3"].api_key_env == "MOONSHOT_API_KEY"
    assert config.api_worker.budget.lifetime_usd == 15.0


def test_global_config_api_worker_per_repo_overrides_scalar_and_keeps_global_providers(
    tmp_path: Path,
) -> None:
    """Per-repo can override top-level api_worker keys without redeclaring providers."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text(API_WORKER_SAMPLE, encoding="utf-8")

    repo_config_path = repo_root / "orchestrator.config.yaml"
    repo_config_path.write_text(
        """api_worker:
  max_concurrent_sessions: 4
""",
        encoding="utf-8",
    )

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))

    assert config.api_worker.max_concurrent_sessions == 4
    assert config.api_worker.provider == "kimi-k3"
    assert "kimi-k3" in config.api_worker.providers


def test_global_config_api_worker_per_repo_partial_budget_keeps_other_caps(
    tmp_path: Path,
) -> None:
    """A repo-level partial api_worker.budget override must not reset other caps."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text(API_WORKER_SAMPLE, encoding="utf-8")

    repo_config_path = repo_root / "orchestrator.config.yaml"
    repo_config_path.write_text(
        """api_worker:
  budget:
    max_usd_per_session: 10.0
""",
        encoding="utf-8",
    )

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))

    assert config.api_worker.budget.max_usd_per_session == 10.0
    assert config.api_worker.budget.preflight_reserve_usd == 1.0
    assert config.api_worker.budget.max_usd_per_day == 5.0
    assert config.api_worker.budget.lifetime_usd == 15.0


def test_global_config_api_worker_per_repo_partial_providers_keeps_global_providers(
    tmp_path: Path,
) -> None:
    """A repo-level partial api_worker.providers override must not replace the whole registry."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text(API_WORKER_SAMPLE, encoding="utf-8")

    repo_config_path = repo_root / "orchestrator.config.yaml"
    repo_config_path.write_text(
        """api_worker:
  enabled: true
  provider: local-k3
  providers:
    local-k3:
      base_url: http://localhost:11434/v1
      api_key_env: OLLAMA_API_KEY
      model: local-k3
      input_usd_per_mtok: 0.5
      output_usd_per_mtok: 0.5
      cached_input_usd_per_mtok: 0.0
""",
        encoding="utf-8",
    )

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))

    assert "kimi-k3" in config.api_worker.providers
    assert "local-k3" in config.api_worker.providers
    assert config.api_worker.provider == "local-k3"
    assert config.api_worker.providers["local-k3"].cached_input_usd_per_mtok == 0.0
    assert config.api_worker.budget.preflight_reserve_usd == 1.0


def test_global_config_non_api_worker_dict_section_is_replaced_not_merged(
    tmp_path: Path,
) -> None:
    """Layered config only deep-merges api_worker; other sections keep old semantics."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)

    global_config_path = fleet_dir_path / "config.yaml"
    global_config_path.write_text(
        """claude_code:
  worker_env:
    GLOBAL: global
""",
        encoding="utf-8",
    )

    repo_config_path = repo_root / "orchestrator.config.yaml"
    repo_config_path.write_text(
        """claude_code:
  worker_env:
    REPO: repo
""",
        encoding="utf-8",
    )

    config = load_layered_config(repo_root, None, fleet_dir_override=str(fleet_dir_path))

    assert config.claude_code.worker_env == {"REPO": "repo"}


def test_orchestrator_config_defaults_include_api_worker() -> None:
    config = OrchestratorConfig()
    assert isinstance(config.api_worker, ApiWorkerConfig)
    assert config.api_worker.enabled is False


# ---------------------------------------------------------------------------
# LabelConfig — complexity:high routing hint (issue #481)
# ---------------------------------------------------------------------------


def test_label_config_complexity_high_default() -> None:
    from charlie_work.config import LabelConfig

    assert LabelConfig().complexity_high == "complexity:high"


def test_label_config_complexity_high_in_all_for_bootstrap() -> None:
    """The hint is in ``all`` so bootstrap_labels creates it on GitHub."""
    from charlie_work.config import LabelConfig

    labels = LabelConfig()
    assert labels.complexity_high in labels.all


def test_label_config_complexity_high_not_in_active_set() -> None:
    """The hint must never affect issue selection/exclusion (not in active)."""
    from charlie_work.config import LabelConfig

    labels = LabelConfig()
    assert labels.complexity_high not in labels.active


def test_label_config_complexity_high_not_in_terminal_set() -> None:
    from charlie_work.config import LabelConfig

    labels = LabelConfig()
    assert labels.complexity_high not in labels.terminal


def test_label_config_complexity_high_not_a_workflow_label() -> None:
    """The hint is a routing hint, not a workflow state — not in workflow_labels."""
    from charlie_work.config import LabelConfig

    labels = LabelConfig()
    assert labels.complexity_high not in labels.workflow_labels


def test_label_config_complexity_high_is_overridable(tmp_path: Path) -> None:
    """The hint string is configurable via the labels: section like every other label."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """labels:
  complexity_high: difficulty:hard
""",
    )
    config = load_config(config_file)
    assert config.labels.complexity_high == "difficulty:hard"


def test_label_config_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    from charlie_work.config import LabelConfig

    labels = LabelConfig()
    with pytest.raises(FrozenInstanceError):
        labels.complexity_high = "x"  # type: ignore[misc]
