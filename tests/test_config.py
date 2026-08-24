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
    build_config_from_data,
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
    """Issue #381/#400/#989: default injected_paths exclude the in-worktree Claude Code prompt file and both protocol markers."""
    from charlie_work.config import (
        CLAUDE_CODE_PROMPT_FILENAME,
        WORKER_OUTCOME_FILENAME,
        WRITER_MARKER_FILENAME,
    )

    config = DispatchConfig()
    assert config.injected_paths == (
        CLAUDE_CODE_PROMPT_FILENAME,
        WRITER_MARKER_FILENAME,
        WORKER_OUTCOME_FILENAME,
    )


def test_protocol_markers_survive_an_operator_injected_paths_override() -> None:
    """Issue #400/#989: an explicit override replaces the prompt-file default but must
    not drop either protocol marker.

    ``__post_init__`` builds its base list from the override when one is given, so the
    prompt filename is deliberately dropped -- the operator has taken control of that.
    The two protocol markers are appended unconditionally instead, because neither can
    ever be worker-authored work: dropping one silently pins the worktree against
    reclamation.
    """
    from charlie_work.config import (
        CLAUDE_CODE_PROMPT_FILENAME,
        WORKER_OUTCOME_FILENAME,
        WRITER_MARKER_FILENAME,
    )

    config = DispatchConfig(injected_paths=(".devin/prompts/worker.md",))

    assert ".devin/prompts/worker.md" in config.injected_paths
    assert WRITER_MARKER_FILENAME in config.injected_paths
    assert WORKER_OUTCOME_FILENAME in config.injected_paths
    # Negative control: the override really did displace the default, so the
    # assertions above are testing the unconditional append and not merely the
    # default list surviving.
    assert CLAUDE_CODE_PROMPT_FILENAME not in config.injected_paths


def test_claude_code_prompt_filename_in_default_injected_paths() -> None:
    """Issue #381: the Claude Code prompt file is excluded from dirty checks by default."""
    from charlie_work.claude_code import PROMPT_FILENAME
    from charlie_work.config import CLAUDE_CODE_PROMPT_FILENAME

    assert PROMPT_FILENAME is CLAUDE_CODE_PROMPT_FILENAME
    assert CLAUDE_CODE_PROMPT_FILENAME in DispatchConfig().injected_paths


def test_load_config_injected_paths_coerces_list_to_tuple(tmp_path: Path) -> None:
    """Issue #381/#400/#989: injected_paths list in YAML becomes a tuple and both protocol markers are always appended."""
    from charlie_work.config import WORKER_OUTCOME_FILENAME, WRITER_MARKER_FILENAME

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
        WORKER_OUTCOME_FILENAME,
    )


def test_dispatch_config_injected_paths_normalizes_backslashes() -> None:
    """Issue #381/#400/#989: backslash separators in an override are normalized to '/' and both protocol markers are appended."""
    from charlie_work.config import WORKER_OUTCOME_FILENAME, WRITER_MARKER_FILENAME

    config = DispatchConfig(injected_paths=[r".devin\prompts\worker.md"])
    assert config.injected_paths == (
        ".devin/prompts/worker.md",
        WRITER_MARKER_FILENAME,
        WORKER_OUTCOME_FILENAME,
    )


def test_load_config_injected_paths_normalizes_backslashes(tmp_path: Path) -> None:
    """Issue #381/#400/#989: YAML override with Windows-style separators is normalized and both protocol markers are appended."""
    from charlie_work.config import WORKER_OUTCOME_FILENAME, WRITER_MARKER_FILENAME

    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        r"""dispatch:
  injected_paths:
    - '.devin\prompts\worker.md'
""",
    )
    config = load_config(config_file)
    assert config.dispatch.injected_paths == (
        ".devin/prompts/worker.md",
        WRITER_MARKER_FILENAME,
        WORKER_OUTCOME_FILENAME,
    )


def test_load_config_review_dispatch_defaults() -> None:
    """ReviewDispatchConfig defaults are safe (disabled, sentinel dir, bounded).

    ``reviews_dir`` defaults to ``""`` -- the sentinel meaning "derive from
    ``runtime.state_dir``" via ``paths.resolved_layout`` -- rather than a
    hardcoded literal. See
    test_paths.py::test_resolved_layout_default_config_matches_historical_literals
    for proof the resolved value still matches the historical path.
    """
    config_file = Path("nonexistent.yaml")
    config = load_config(config_file)
    assert config.review_dispatch.enabled is False
    assert config.review_dispatch.reviews_dir == ""
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


def test_load_config_quota_probe_defaults() -> None:
    """QuotaProbeConfig defaults: enabled, flat 15-minute interval, Haiku."""
    config = load_config(Path("nonexistent.yaml"))
    assert config.quota_probe.enabled is True
    assert config.quota_probe.interval_minutes == 15
    assert config.quota_probe.model == "claude-haiku-4-5"
    assert config.quota_probe.timeout_seconds == 60
    assert config.quota_probe.prompt


def test_load_config_quota_probe_override(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """quota_probe:
  enabled: false
  interval_minutes: 30
  model: claude-haiku-custom
  timeout_seconds: 20
  prompt: "ping"
""",
    )
    config = load_config(config_file)
    assert config.quota_probe.enabled is False
    assert config.quota_probe.interval_minutes == 30
    assert config.quota_probe.model == "claude-haiku-custom"
    assert config.quota_probe.timeout_seconds == 20
    assert config.quota_probe.prompt == "ping"


def test_load_config_quota_probe_enabled_rejects_non_bool(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, "quota_probe:\n  enabled: not-a-bool\n")
    with pytest.raises(ConfigError, match="quota_probe.*enabled.*must be a bool"):
        load_config(config_file)


def test_load_config_supervisor_self_deploy_pull_ci_fleet_default() -> None:
    """SupervisorConfig.self_deploy_pull_ci_fleet defaults to False (issue #552)."""
    config = load_config(Path("nonexistent.yaml"))
    assert config.supervisor.self_deploy_pull_ci_fleet is False


def test_load_config_supervisor_self_deploy_pull_ci_fleet_accepts_true(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """supervisor:
  self_deploy_pull_ci_fleet: true
""",
    )
    config = load_config(config_file)
    assert config.supervisor.self_deploy_pull_ci_fleet is True


def test_load_config_supervisor_self_deploy_pull_ci_fleet_rejects_non_bool(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, "supervisor:\n  self_deploy_pull_ci_fleet: not-a-bool\n")
    with pytest.raises(ConfigError, match="supervisor.*self_deploy_pull_ci_fleet.*must be a bool"):
        load_config(config_file)


@pytest.mark.parametrize("value", [0, -1])
def test_load_config_quota_probe_interval_minutes_rejects_non_positive(
    tmp_path: Path, value: int
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, f"quota_probe:\n  interval_minutes: {value}\n")
    with pytest.raises(ConfigError, match="quota_probe.*interval_minutes.*must be >= 1"):
        load_config(config_file)


def test_load_config_quota_probe_interval_minutes_rejects_non_int(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, "quota_probe:\n  interval_minutes: '15'\n")
    with pytest.raises(ConfigError, match="quota_probe.*interval_minutes.*must be an int"):
        load_config(config_file)


def test_load_config_reconcile_pass_terminal_state_alert_days_default() -> None:
    """Issue #947: defaults to 2 days so a stuck human_needed issue is
    caught quickly without paging on same-day escalate/unescalate cycles."""
    config = load_config(Path("nonexistent.yaml"))
    assert config.reconcile_pass.terminal_state_alert_days == 2


def test_load_config_reconcile_pass_terminal_state_alert_days_override(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, "reconcile_pass:\n  terminal_state_alert_days: 5\n")
    config = load_config(config_file)
    assert config.reconcile_pass.terminal_state_alert_days == 5


@pytest.mark.parametrize("value", [0, -1])
def test_load_config_reconcile_pass_terminal_state_alert_days_rejects_non_positive(
    tmp_path: Path, value: int
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, f"reconcile_pass:\n  terminal_state_alert_days: {value}\n")
    with pytest.raises(
        ConfigError, match="reconcile_pass.*terminal_state_alert_days.*must be >= 1"
    ):
        load_config(config_file)


def test_load_config_reconcile_pass_terminal_state_alert_days_rejects_non_int(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, "reconcile_pass:\n  terminal_state_alert_days: '2'\n")
    with pytest.raises(
        ConfigError, match="reconcile_pass.*terminal_state_alert_days.*must be an int"
    ):
        load_config(config_file)


def test_load_config_quota_probe_model_rejects_empty(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, "quota_probe:\n  model: '   '\n")
    with pytest.raises(ConfigError, match="quota_probe.*model.*must not be empty"):
        load_config(config_file)


def test_load_config_quota_probe_timeout_seconds_rejects_non_positive(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, "quota_probe:\n  timeout_seconds: 0\n")
    with pytest.raises(ConfigError, match="quota_probe.*timeout_seconds.*must be >= 1"):
        load_config(config_file)


def test_load_config_quota_probe_prompt_rejects_empty(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, "quota_probe:\n  prompt: ''\n")
    with pytest.raises(ConfigError, match="quota_probe.*prompt.*must not be empty"):
        load_config(config_file)


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


def test_load_config_event_ring_size_rejects_bool(tmp_path: Path) -> None:
    """``event_ring_size: true`` must be rejected, not silently used as 1.

    ``bool`` is an ``int`` subclass, so the bare ``isinstance(x, int)`` this
    validator used accepted YAML's ``true`` and let it through as ``1``. That
    is worse than a hard failure: ``append_event`` truncates via
    ``events[-1:]``, giving a one-entry ring that looks configured and drops
    every event but the last, with nothing reporting it. The zero and negative
    cases below already fail closed; this one failed *open*.

    Regression-proved: reverting the guard to a bare ``isinstance(x, int)``
    leaves the rest of this file green, so nothing else covers it.
    """
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  event_ring_size: true
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


def test_runtime_config_escalated_label_repair_max_per_pass_default() -> None:
    """Issue #1088: RuntimeConfig.escalated_label_repair_max_per_pass defaults to 10."""
    assert RuntimeConfig().escalated_label_repair_max_per_pass == 10


def test_load_config_escalated_label_repair_max_per_pass_override(tmp_path: Path) -> None:
    """Issue #1088: runtime.escalated_label_repair_max_per_pass is configurable
    from YAML."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  escalated_label_repair_max_per_pass: 25
""",
    )
    config = load_config(config_file)
    assert config.runtime.escalated_label_repair_max_per_pass == 25


def test_load_config_escalated_label_repair_max_per_pass_rejects_non_int(
    tmp_path: Path,
) -> None:
    """Issue #1088: escalated_label_repair_max_per_pass must be an int."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  escalated_label_repair_max_per_pass: "lots"
""",
    )
    with pytest.raises(ConfigError, match="escalated_label_repair_max_per_pass.*must be an int"):
        load_config(config_file)


def test_load_config_escalated_label_repair_max_per_pass_rejects_bool(
    tmp_path: Path,
) -> None:
    """Issue #1088: ``bool`` is a subclass of ``int`` in Python, so a bare
    ``isinstance(x, int)`` check would silently accept ``true``/``false`` as
    1/0. The parser explicitly rejects bool -- this pins that it keeps doing
    so (event_ring_size's validator right above does NOT have this guard, so
    this is a genuine behavioral difference between the two knobs, not
    incidental).
    """
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  escalated_label_repair_max_per_pass: true
""",
    )
    with pytest.raises(ConfigError, match="escalated_label_repair_max_per_pass.*must be an int"):
        load_config(config_file)


def test_load_config_escalated_label_repair_max_per_pass_rejects_negative(
    tmp_path: Path,
) -> None:
    """Issue #1088: negative escalated_label_repair_max_per_pass is rejected."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  escalated_label_repair_max_per_pass: -1
""",
    )
    with pytest.raises(ConfigError, match="escalated_label_repair_max_per_pass.*must be >= 0"):
        load_config(config_file)


def test_load_config_escalated_label_repair_max_per_pass_accepts_zero(
    tmp_path: Path,
) -> None:
    """Issue #1088: 0 means unlimited (matching graphql_rate_limit_threshold's
    "0 disables the guard" convention), so it must be accepted, not rejected --
    unlike event_ring_size, which rejects 0 for the opposite reason (see
    test_load_config_event_ring_size_rejects_zero above).
    """
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  escalated_label_repair_max_per_pass: 0
""",
    )
    config = load_config(config_file)
    assert config.runtime.escalated_label_repair_max_per_pass == 0


def test_runtime_config_gh_timeout_seconds_default() -> None:
    """RuntimeConfig.gh_timeout_seconds defaults to 120.0 seconds."""
    assert RuntimeConfig().gh_timeout_seconds == 120.0


def test_load_config_gh_timeout_seconds_override(tmp_path: Path) -> None:
    """runtime.gh_timeout_seconds is configurable from YAML and reaches
    RuntimeConfig."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  gh_timeout_seconds: 45.5
""",
    )
    config = load_config(config_file)
    assert config.runtime.gh_timeout_seconds == 45.5


def test_load_config_gh_timeout_seconds_rejects_zero(tmp_path: Path) -> None:
    """gh_timeout_seconds=0 would time out instantly, failing every gh call."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  gh_timeout_seconds: 0
""",
    )
    with pytest.raises(ConfigError, match="gh_timeout_seconds.*must be > 0"):
        load_config(config_file)


def test_load_config_gh_timeout_seconds_rejects_negative(tmp_path: Path) -> None:
    """Negative gh_timeout_seconds is rejected."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  gh_timeout_seconds: -1
""",
    )
    with pytest.raises(ConfigError, match="gh_timeout_seconds.*must be > 0"):
        load_config(config_file)


def test_load_config_gh_timeout_seconds_rejects_bool(tmp_path: Path) -> None:
    """A bool must not silently pass the `isinstance(x, (int, float))` check
    (bool is an int subclass in Python)."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  gh_timeout_seconds: true
""",
    )
    with pytest.raises(ConfigError, match="gh_timeout_seconds.*must be a number"):
        load_config(config_file)


def test_load_config_gh_timeout_seconds_rejects_non_number(tmp_path: Path) -> None:
    """gh_timeout_seconds must be a number, not a string."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runtime:
  gh_timeout_seconds: "120"
""",
    )
    with pytest.raises(ConfigError, match="gh_timeout_seconds.*must be a number"):
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


def test_load_config_stale_checks_grace_minutes_rejects_bool_true(tmp_path: Path) -> None:
    """Issue #1274 (W17): YAML boolean true is not a valid integer minutes value."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  stale_checks_grace_minutes: true
""",
    )
    with pytest.raises(ConfigError, match="stale_checks_grace_minutes.*must be an int"):
        load_config(config_file)


def test_load_config_stale_checks_grace_minutes_rejects_bool_false(tmp_path: Path) -> None:
    """Issue #1274 (W17): YAML boolean false silently means 0 if treated as int."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  stale_checks_grace_minutes: false
""",
    )
    with pytest.raises(ConfigError, match="stale_checks_grace_minutes.*must be an int"):
        load_config(config_file)


def test_load_config_stale_checks_grace_minutes_rejects_negative(tmp_path: Path) -> None:
    """Issue #1274 (W17): negative minutes is semantically meaningless."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  stale_checks_grace_minutes: -1
""",
    )
    with pytest.raises(ConfigError, match="stale_checks_grace_minutes.*must not be negative"):
        load_config(config_file)


def test_load_config_stale_checks_grace_minutes_accepts_valid_int(tmp_path: Path) -> None:
    """Issue #1274 (W17): zero and positive integers are both accepted."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  stale_checks_grace_minutes: 0
""",
    )
    config = load_config(config_file)
    assert config.review.stale_checks_grace_minutes == 0

    _write_config(
        config_file,
        """review:
  stale_checks_grace_minutes: 20
""",
    )
    config = load_config(config_file)
    assert config.review.stale_checks_grace_minutes == 20


def test_load_config_stale_checks_max_retriggers_rejects_bool_true(tmp_path: Path) -> None:
    """Issue #1274 (W17): YAML boolean true is not a valid integer count."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  stale_checks_max_retriggers: true
""",
    )
    with pytest.raises(ConfigError, match="stale_checks_max_retriggers.*must be an int"):
        load_config(config_file)


def test_load_config_stale_checks_max_retriggers_rejects_bool_false(tmp_path: Path) -> None:
    """Issue #1274 (W17): YAML boolean false silently means 0 if treated as int."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  stale_checks_max_retriggers: false
""",
    )
    with pytest.raises(ConfigError, match="stale_checks_max_retriggers.*must be an int"):
        load_config(config_file)


def test_load_config_stale_checks_max_retriggers_rejects_negative(tmp_path: Path) -> None:
    """Issue #1274 (W17): a negative retrigger cap is semantically meaningless."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  stale_checks_max_retriggers: -1
""",
    )
    with pytest.raises(ConfigError, match="stale_checks_max_retriggers.*must not be negative"):
        load_config(config_file)


def test_load_config_stale_checks_max_retriggers_accepts_valid_int(tmp_path: Path) -> None:
    """Issue #1274 (W17): zero (no retries) and positive counts are both accepted."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  stale_checks_max_retriggers: 0
""",
    )
    config = load_config(config_file)
    assert config.review.stale_checks_max_retriggers == 0

    _write_config(
        config_file,
        """review:
  stale_checks_max_retriggers: 5
""",
    )
    config = load_config(config_file)
    assert config.review.stale_checks_max_retriggers == 5


def test_load_config_stale_checks_defaults_when_absent(tmp_path: Path) -> None:
    """Issue #1274 (W17): an absent `review` section (or absent keys within an
    otherwise-present one) falls back to ReviewConfig's documented defaults --
    no silent zero/None, matching the sibling
    auto_merge.ci_run_never_created_grace_minutes default-fallback shape.
    """
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(config_file, "review:\n  max_rework_cycles: 2\n")
    config = load_config(config_file)
    assert config.review.stale_checks_grace_minutes == 15
    assert config.review.stale_checks_max_retriggers == 3


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


# ---------------------------------------------------------------------------
# LabelConfig — operator_queue (issue #1266: mechanical-escalation routing)
# ---------------------------------------------------------------------------


def test_label_config_operator_queue_default() -> None:
    from charlie_work.config import LabelConfig

    assert LabelConfig().operator_queue == "agent:operator-queue"


def test_label_config_operator_queue_in_terminal_set() -> None:
    """A mechanically-escalated issue must never re-enter dispatch.

    ``operator_queue`` has to be a member of ``terminal`` -- that set is what
    ``OrchestratorApp._is_dispatchable`` (and the standalone dispatch-backlog
    reachability check) intersect against to exclude an issue from selection.
    Without this membership, an operator-queued issue would still carry
    ``automated-ready`` and get redispatched out from under the operator.
    """
    from charlie_work.config import LabelConfig

    labels = LabelConfig()
    assert labels.operator_queue in labels.terminal


def test_label_config_operator_queue_in_workflow_labels() -> None:
    """Unlike ``prose_only_deps``, operator_queue is actively added/removed by
    automated ``labels.py`` transitions (the operator_queued/
    redispatch_operator_queued edges, and the de-escalation cap-exhaustion
    path), so it must be a ``workflow_labels`` member -- otherwise
    ``_compute_remove`` would never strip it on a transition away from it.
    """
    from charlie_work.config import LabelConfig

    labels = LabelConfig()
    assert labels.operator_queue in labels.workflow_labels


def test_label_config_operator_queue_in_all_for_bootstrap() -> None:
    """The label is in ``all`` so bootstrap_labels creates it on GitHub."""
    from charlie_work.config import LabelConfig

    labels = LabelConfig()
    assert labels.operator_queue in labels.all


def test_label_config_operator_queue_not_in_active_set() -> None:
    """A terminal escalation state is not an "actively being worked" state."""
    from charlie_work.config import LabelConfig

    labels = LabelConfig()
    assert labels.operator_queue not in labels.active


def test_label_config_operator_queue_is_overridable(tmp_path: Path) -> None:
    """The label string is configurable via the labels: section like every
    other label -- issue #1266 forbids hardcoding it anywhere but here."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """labels:
  operator_queue: agent:needs-operator
""",
    )
    config = load_config(config_file)
    assert config.labels.operator_queue == "agent:needs-operator"


# --- Issue #600: runner_allocation is host-wide only; cross-validate floors ---


def test_load_layered_config_rejects_per_repo_runner_allocation(tmp_path: Path) -> None:
    """Issue #600: a per-repo ``runner_allocation`` section must be rejected.

    The merge in ``load_layered_config`` is section-by-section with the per-repo
    file winning per key, so without an explicit rejection a per-repo
    ``orchestrator.config.yaml`` could silently override a host-wide knob. The
    section is documented host-wide-only (see ``RunnerAllocationConfig``); make
    the invalid state unrepresentable rather than merely unused.
    """
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text("runner_allocation:\n  enabled: true\n", encoding="utf-8")

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_config(
        repo_root / "orchestrator.config.yaml",
        "runner_allocation:\n  enabled: true\n  min_running_per_repo: 2\n",
    )

    with pytest.raises(ConfigError, match="host-wide only"):
        load_layered_config(repo_root, fleet_dir_override=str(fleet))


def test_load_layered_config_accepts_global_runner_allocation(tmp_path: Path) -> None:
    """The rejection is scoped to the per-repo layer; the global layer keeps it."""
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / "config.yaml").write_text(
        "runner_allocation:\n  enabled: true\n  min_running_per_repo: 2\n",
        encoding="utf-8",
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    # No per-repo config at all -- the global layer stands alone.
    config = load_layered_config(repo_root, fleet_dir_override=str(fleet))
    assert config.runner_allocation.enabled is True
    assert config.runner_allocation.min_running_per_repo == 2


def test_load_config_rejects_allocation_floor_above_scaling_floor(tmp_path: Path) -> None:
    """Issue #600: when both features are enabled, the allocation floor must not
    exceed the scaling floor -- allocation caps each repo's target at its
    registered runner count, so a higher ``min_running_per_repo`` silently
    degrades with nothing reconciling the two."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runner_scaling:
  enabled: true
  min_runners: 1
runner_allocation:
  enabled: true
  min_running_per_repo: 3
""",
    )
    with pytest.raises(ConfigError, match="floors disagree"):
        load_config(config_file)


def test_load_config_accepts_equal_floors_both_enabled(tmp_path: Path) -> None:
    """Equal floors are the unambiguous single-source-of-truth case."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runner_scaling:
  enabled: true
  min_runners: 2
runner_allocation:
  enabled: true
  min_running_per_repo: 2
""",
    )
    config = load_config(config_file)
    assert config.runner_scaling.enabled is True
    assert config.runner_allocation.enabled is True
    assert config.runner_scaling.min_runners == config.runner_allocation.min_running_per_repo


def test_load_config_accepts_scaling_floor_above_allocation_floor(tmp_path: Path) -> None:
    """``min_runners > min_running_per_repo`` is a legitimate buffer: registered
    but parked runners that allocation promotes on demand. Only the unsatisfiable
    direction (allocation > scaling) is rejected."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runner_scaling:
  enabled: true
  min_runners: 5
runner_allocation:
  enabled: true
  min_running_per_repo: 1
""",
    )
    config = load_config(config_file)
    assert config.runner_scaling.min_runners == 5
    assert config.runner_allocation.min_running_per_repo == 1


def test_load_config_skips_floor_check_when_only_one_enabled(tmp_path: Path) -> None:
    """The cross-section check fires only when both features are enabled; a
    mismatched floor with one disabled is not a conflict (the disabled feature
    imposes no floor)."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """runner_scaling:
  enabled: false
  min_runners: 1
runner_allocation:
  enabled: true
  min_running_per_repo: 9
""",
    )
    config = load_config(config_file)
    assert config.runner_scaling.enabled is False
    assert config.runner_allocation.enabled is True
    assert config.runner_allocation.min_running_per_repo == 9


def test_main_ci_reclaim_defaults_enabled_with_ci_yml(tmp_path: Path) -> None:
    """Issue #863/#815: an absent ``main_ci_reclaim`` block must still enable
    the pass (rollback knob, not opt-in -- see MainCiReclaimConfig's
    docstring), defaulting to this repo's actual workflow filename."""
    config = load_config(None)
    assert config.main_ci_reclaim.enabled is True
    assert config.main_ci_reclaim.workflow_filename == "ci.yml"


def test_main_ci_reclaim_enabled_rejects_non_bool(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
main_ci_reclaim:
  enabled: "true"
""",
    )
    with pytest.raises(ConfigError, match="main_ci_reclaim.*enabled.*must be a bool"):
        load_config(config_file)


def test_main_ci_reclaim_workflow_filename_rejects_non_string(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
main_ci_reclaim:
  workflow_filename: 5
""",
    )
    with pytest.raises(ConfigError, match="main_ci_reclaim.*workflow_filename.*must be a string"):
        load_config(config_file)


def test_main_ci_reclaim_rejects_unknown_key(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
main_ci_reclaim:
  interval_minutes: 15
""",
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(config_file)


def test_main_ci_reclaim_can_be_disabled_and_repointed(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """
main_ci_reclaim:
  enabled: false
  workflow_filename: "tests.yml"
""",
    )
    config = load_config(config_file)
    assert config.main_ci_reclaim.enabled is False
    assert config.main_ci_reclaim.workflow_filename == "tests.yml"


def test_build_config_from_data_require_worker_github_token_rejects_non_bool() -> None:
    """Issue #1001: dispatch.require_worker_github_token must be a bool."""
    with pytest.raises(ConfigError, match="require_worker_github_token.*must be a bool"):
        build_config_from_data({"dispatch": {"require_worker_github_token": "true"}})


def test_provider_suspended_is_deterministic_escalation_failure_kind() -> None:
    """Issue #1342: ``provider_suspended`` must sit in
    DETERMINISTIC_ESCALATION_FAILURE_KINDS so a suspended provider account
    escalates to an operator on the first occurrence instead of burning the
    auto-redispatch cap on a deterministic external billing failure."""
    from charlie_work.config import DETERMINISTIC_ESCALATION_FAILURE_KINDS

    assert "provider_suspended" in DETERMINISTIC_ESCALATION_FAILURE_KINDS
