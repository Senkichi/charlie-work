"""Tests for config.py validation, especially new config keys."""

from __future__ import annotations

from pathlib import Path

import pytest
from types import MappingProxyType

from charlie_work import config as config_module
from charlie_work.config import (
    ApiBudgetConfig,
    ApiProviderConfig,
    ApiWorkerConfig,
    ClaudeCodeConfig,
    ConfigError,
    DispatchConfig,
    OrchestratorConfig,
    RescueConfig,
    ReviewerRoleConfig,
    RuntimeConfig,
    WorkerRoleConfig,
    build_config_from_data,
    known_config_sections,
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


def test_load_config_review_dispatch_file_size_cap_lines_override(tmp_path: Path) -> None:
    """Issue #1445: ``review_dispatch.file_size_cap_lines`` is read from YAML
    via the ``_RD_INT_KEYS`` wiring in ``build_config_from_data`` -- the
    ``review_dispatch:`` section (``ReviewDispatchConfig``), not the unrelated
    ``review:`` section (``ReviewConfig``)."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review_dispatch:
  file_size_cap_lines: 1234
""",
    )
    config = load_config(config_file)
    assert config.review_dispatch.file_size_cap_lines == 1234


def test_load_config_review_dispatch_file_size_cap_lines_rejects_non_int(
    tmp_path: Path,
) -> None:
    """Issue #1445: ``_RD_INT_KEYS`` validation rejects a non-int
    ``file_size_cap_lines`` in the ``review_dispatch:`` section."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review_dispatch:
  file_size_cap_lines: "not-an-int"
""",
    )
    with pytest.raises(ConfigError, match="review_dispatch.*file_size_cap_lines.*must be an int"):
        load_config(config_file)


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
    with pytest.raises(ConfigError, match="effort_experiment_fraction.*must be a number"):
        load_config(config_file)


def test_load_config_review_effort_experiment_fraction_rejects_non_number(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review_dispatch:
  review_effort_experiment_fraction: "0.5"
""",
    )
    with pytest.raises(ConfigError, match="effort_experiment_fraction.*must be a number"):
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
    with pytest.raises(ConfigError, match="effort_experiment_salt.*must be a string"):
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


# ---------------------------------------------------------------------------
# Issue #1132: foreign_issue_ref_confirm_passes / foreign_issue_ref_reprobe_hours
# validation-branch unit tests (per-knob quartet, mirroring the
# stale_checks_max_retriggers precedent above).
# ---------------------------------------------------------------------------


def test_load_config_foreign_issue_ref_confirm_passes_rejects_bool_true(
    tmp_path: Path,
) -> None:
    """YAML boolean true is not a valid integer confirmation count."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  foreign_issue_ref_confirm_passes: true
""",
    )
    with pytest.raises(ConfigError, match="foreign_issue_ref_confirm_passes.*must be an int"):
        load_config(config_file)


def test_load_config_foreign_issue_ref_confirm_passes_rejects_bool_false(
    tmp_path: Path,
) -> None:
    """YAML boolean false silently means 0 if treated as int; reject it."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  foreign_issue_ref_confirm_passes: false
""",
    )
    with pytest.raises(ConfigError, match="foreign_issue_ref_confirm_passes.*must be an int"):
        load_config(config_file)


def test_load_config_foreign_issue_ref_confirm_passes_rejects_below_one(
    tmp_path: Path,
) -> None:
    """A confirmation count below 1 disables the confirmation gate (parking on
    the first not-found), which defeats the transient-failure bound the knob
    exists to provide."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  foreign_issue_ref_confirm_passes: 0
""",
    )
    with pytest.raises(ConfigError, match="foreign_issue_ref_confirm_passes.*must be >= 1"):
        load_config(config_file)


def test_load_config_foreign_issue_ref_confirm_passes_accepts_valid_int(
    tmp_path: Path,
) -> None:
    """1 (one-pass park, original behavior) and >=2 (confirmation gate) are
    both accepted."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  foreign_issue_ref_confirm_passes: 1
""",
    )
    config = load_config(config_file)
    assert config.review.foreign_issue_ref_confirm_passes == 1

    _write_config(
        config_file,
        """review:
  foreign_issue_ref_confirm_passes: 3
""",
    )
    config = load_config(config_file)
    assert config.review.foreign_issue_ref_confirm_passes == 3


def test_load_config_foreign_issue_ref_reprobe_hours_rejects_bool_true(
    tmp_path: Path,
) -> None:
    """YAML boolean true is not a valid integer re-probe cadence."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  foreign_issue_ref_reprobe_hours: true
""",
    )
    with pytest.raises(ConfigError, match="foreign_issue_ref_reprobe_hours.*must be an int"):
        load_config(config_file)


def test_load_config_foreign_issue_ref_reprobe_hours_rejects_bool_false(
    tmp_path: Path,
) -> None:
    """YAML boolean false silently means 0 if treated as int; reject it."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  foreign_issue_ref_reprobe_hours: false
""",
    )
    with pytest.raises(ConfigError, match="foreign_issue_ref_reprobe_hours.*must be an int"):
        load_config(config_file)


def test_load_config_foreign_issue_ref_reprobe_hours_rejects_negative(
    tmp_path: Path,
) -> None:
    """A negative re-probe cadence is semantically meaningless."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  foreign_issue_ref_reprobe_hours: -1
""",
    )
    with pytest.raises(ConfigError, match="foreign_issue_ref_reprobe_hours.*must not be negative"):
        load_config(config_file)


def test_load_config_foreign_issue_ref_reprobe_hours_accepts_valid_int(
    tmp_path: Path,
) -> None:
    """0 (self-heal disabled, operator-only remedy) and positive cadences are
    both accepted."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        """review:
  foreign_issue_ref_reprobe_hours: 0
""",
    )
    config = load_config(config_file)
    assert config.review.foreign_issue_ref_reprobe_hours == 0

    _write_config(
        config_file,
        """review:
  foreign_issue_ref_reprobe_hours: 48
""",
    )
    config = load_config(config_file)
    assert config.review.foreign_issue_ref_reprobe_hours == 48


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


# ---------------------------------------------------------------------------
# Issue #1383: auto_merge.infra_blocked validation error paths.
#
# Mirrors the per-field rejection convention established for every other
# auto_merge / review sub-section (e.g. stale_checks_*): unknown key,
# wrong type per field, bool-for-int rejection, and negative-value
# rejection. The happy path (YAML parse + round-trip) is covered in
# test_charlie_work.py::test_infra_blocked_config_parses_from_yaml.
# ---------------------------------------------------------------------------


_IB_HEADER = "auto_merge:\n  required_checks: [Tests passed]\n  infra_blocked:\n"


def test_load_config_infra_blocked_defaults_when_absent(tmp_path: Path) -> None:
    """An absent infra_blocked section yields InfraBlockedConfig's documented
    defaults -- no silent zero/None, matching the sibling
    stale_checks default-fallback shape."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        "auto_merge:\n  required_checks: [Tests passed]\n",
    )
    config = load_config(config_file)
    cfg = config.auto_merge.infra_blocked
    assert cfg.enabled is True
    assert cfg.instant_fail_seconds == 10
    assert cfg.persistence_passes == 3
    assert cfg.escalation_window_minutes == 60
    assert cfg.annotation_patterns == (
        "the job was not started",
        "actions budget is preventing further use",
        "no runner matching",
        "usage limit",
    )


def test_load_config_infra_blocked_rejects_non_mapping(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        "auto_merge:\n  required_checks: [Tests passed]\n  infra_blocked: not-a-map\n",
    )
    with pytest.raises(ConfigError, match="infra_blocked.*must be a mapping"):
        load_config(config_file)


def test_load_config_infra_blocked_rejects_unknown_key(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    bogus_key: 1\n",
    )
    with pytest.raises(ConfigError, match="infra_blocked.*has unknown key"):
        load_config(config_file)


def test_load_config_infra_blocked_annotation_patterns_rejects_non_list(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    annotation_patterns: billing-exhausted\n",
    )
    with pytest.raises(ConfigError, match="annotation_patterns.*must be a list of strings"):
        load_config(config_file)


def test_load_config_infra_blocked_instant_fail_seconds_rejects_non_int(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    instant_fail_seconds: ten\n",
    )
    with pytest.raises(ConfigError, match="instant_fail_seconds.*must be an int"):
        load_config(config_file)


def test_load_config_infra_blocked_instant_fail_seconds_rejects_bool(
    tmp_path: Path,
) -> None:
    """YAML boolean true is not a valid integer count (matches the
    stale_checks_max_retriggers bool-rejection convention)."""
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    instant_fail_seconds: true\n",
    )
    with pytest.raises(ConfigError, match="instant_fail_seconds.*must be an int"):
        load_config(config_file)


def test_load_config_infra_blocked_instant_fail_seconds_rejects_negative(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    instant_fail_seconds: -1\n",
    )
    with pytest.raises(ConfigError, match="instant_fail_seconds.*must be >= 0"):
        load_config(config_file)


def test_load_config_infra_blocked_persistence_passes_rejects_non_int(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    persistence_passes: three\n",
    )
    with pytest.raises(ConfigError, match="persistence_passes.*must be an int"):
        load_config(config_file)


def test_load_config_infra_blocked_persistence_passes_rejects_bool(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    persistence_passes: false\n",
    )
    with pytest.raises(ConfigError, match="persistence_passes.*must be an int"):
        load_config(config_file)


def test_load_config_infra_blocked_persistence_passes_rejects_negative(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    persistence_passes: -2\n",
    )
    with pytest.raises(ConfigError, match="persistence_passes.*must be >= 0"):
        load_config(config_file)


def test_load_config_infra_blocked_escalation_window_minutes_rejects_non_int(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    escalation_window_minutes: 1h\n",
    )
    with pytest.raises(ConfigError, match="escalation_window_minutes.*must be an int"):
        load_config(config_file)


def test_load_config_infra_blocked_escalation_window_minutes_rejects_bool(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    escalation_window_minutes: true\n",
    )
    with pytest.raises(ConfigError, match="escalation_window_minutes.*must be an int"):
        load_config(config_file)


def test_load_config_infra_blocked_escalation_window_minutes_rejects_negative(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    escalation_window_minutes: -5\n",
    )
    with pytest.raises(ConfigError, match="escalation_window_minutes.*must be >= 0"):
        load_config(config_file)


def test_load_config_infra_blocked_enabled_rejects_non_bool(tmp_path: Path) -> None:
    config_file = tmp_path / "orchestrator.config.yaml"
    _write_config(
        config_file,
        _IB_HEADER + "    enabled: yes-please\n",
    )
    with pytest.raises(ConfigError, match="infra_blocked.enabled.*must be a bool"):
        load_config(config_file)


def test_provider_suspended_is_deterministic_escalation_failure_kind() -> None:
    """Issue #1342: ``provider_suspended`` must sit in
    DETERMINISTIC_ESCALATION_FAILURE_KINDS so a suspended provider account
    escalates to an operator on the first occurrence instead of burning the
    auto-redispatch cap on a deterministic external billing failure."""
    from charlie_work.config import DETERMINISTIC_ESCALATION_FAILURE_KINDS

    assert "provider_suspended" in DETERMINISTIC_ESCALATION_FAILURE_KINDS


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


def test_orchestrator_config_worker_reviewer_deprecations_defaults() -> None:
    cfg = OrchestratorConfig()
    assert cfg.worker == WorkerRoleConfig()
    assert cfg.reviewer == ReviewerRoleConfig()
    assert cfg.deprecations == ()


def test_worker_and_reviewer_are_known_config_sections() -> None:
    sections = known_config_sections()
    assert "worker" in sections
    assert "reviewer" in sections
    # Provenance fields must never be forgeable as a YAML section.
    assert "deprecations" not in sections
    assert "sources" not in sections


def test_claude_code_config_model_default_uses_shared_constant() -> None:
    from charlie_work.config import _DEFAULT_CLAUDE_MODEL

    assert ClaudeCodeConfig().model == _DEFAULT_CLAUDE_MODEL
    assert ReviewerRoleConfig().model == _DEFAULT_CLAUDE_MODEL


def test_resolve_dual_accept_new_only() -> None:
    value, deprecated = config_module._resolve_dual_accept(
        old_present=False,
        old_value=None,
        old_label="old.x",
        new_present=True,
        new_value="Y",
        new_label="new.x",
        default="D",
    )
    assert value == "Y"
    assert deprecated is False


def test_resolve_dual_accept_old_only() -> None:
    value, deprecated = config_module._resolve_dual_accept(
        old_present=True,
        old_value="X",
        old_label="old.x",
        new_present=False,
        new_value=None,
        new_label="new.x",
        default="D",
    )
    assert value == "X"
    assert deprecated is True


def test_resolve_dual_accept_neither_uses_default() -> None:
    value, deprecated = config_module._resolve_dual_accept(
        old_present=False,
        old_value=None,
        old_label="old.x",
        new_present=False,
        new_value=None,
        new_label="new.x",
        default="D",
    )
    assert value == "D"
    assert deprecated is False


def test_resolve_dual_accept_agreeing_values_are_deprecated_not_conflicting() -> None:
    value, deprecated = config_module._resolve_dual_accept(
        old_present=True,
        old_value="X",
        old_label="old.x",
        new_present=True,
        new_value="X",
        new_label="new.x",
        default="D",
    )
    assert value == "X"
    assert deprecated is True


def test_resolve_dual_accept_disagreeing_values_raise() -> None:
    with pytest.raises(ConfigError, match="old.x.*X.*new.x.*Y"):
        config_module._resolve_dual_accept(
            old_present=True,
            old_value="X",
            old_label="old.x",
            new_present=True,
            new_value="Y",
            new_label="new.x",
            default="D",
        )


def test_role_section_absent_key_inserts_empty_dict_back_into_data() -> None:
    data: dict = {}
    section = config_module._role_section(data, "worker")
    assert section == {}
    assert data["worker"] is section  # must be the SAME object, not a detached copy


def test_role_section_non_dict_value_is_coerced_and_inserted() -> None:
    data = {"worker": "not-a-dict"}
    section = config_module._role_section(data, "worker")
    assert section == {}
    assert data["worker"] is section


def test_role_section_existing_dict_is_returned_as_is() -> None:
    data = {"worker": {"harness": "api"}}
    section = config_module._role_section(data, "worker")
    assert section is data["worker"]
    assert section == {"harness": "api"}


def test_resolve_role_dual_accept_scaffold_returns_empty_list() -> None:
    assert config_module._resolve_role_dual_accept({}) == []
    assert config_module._resolve_role_dual_accept({"unrelated": 1}) == []


def test_resolve_role_dual_accept_worker_harness_old_only() -> None:
    data = {"devin": {"adapter": "devin-shell"}}
    config_module._resolve_role_dual_accept(data)
    assert data["devin"]["adapter"] == "devin-shell"
    assert data["worker"]["harness"] == "devin-shell"


def test_resolve_role_dual_accept_worker_harness_new_only() -> None:
    data = {"worker": {"harness": "api"}}
    config_module._resolve_role_dual_accept(data)
    assert data["devin"]["adapter"] == "api"
    assert data["worker"]["harness"] == "api"


def test_resolve_role_dual_accept_worker_harness_neither_defaults_to_manual() -> None:
    data: dict = {}
    config_module._resolve_role_dual_accept(data)
    assert data["devin"]["adapter"] == "manual"
    assert data["worker"]["harness"] == "manual"


def test_resolve_role_dual_accept_worker_harness_conflict_raises() -> None:
    data = {"devin": {"adapter": "devin-shell"}, "worker": {"harness": "api"}}
    with pytest.raises(ConfigError, match="devin.adapter.*devin-shell.*worker.harness.*api"):
        config_module._resolve_role_dual_accept(data)


def test_resolve_role_dual_accept_worker_harness_rejects_invalid_value() -> None:
    data = {"worker": {"harness": "bogus"}}
    with pytest.raises(ConfigError, match="worker.*harness.*bogus"):
        config_module._resolve_role_dual_accept(data)


def test_resolve_role_dual_accept_worker_model_devin_shell_old_only() -> None:
    data = {"devin": {"adapter": "devin-shell", "worker_model": "glm-5-2"}}
    config_module._resolve_role_dual_accept(data)
    assert data["devin"]["worker_model"] == "glm-5-2"
    assert data["worker"]["model"] == "glm-5-2"
    # claude_code.model is unclaimed by a devin-shell WORKER, but (as of Task 4)
    # the reviewer resolution repurposes it as the reviewer's legacy model key
    # in this branch -- it still gets populated, with the independently
    # resolved reviewer model (here the shared default, since neither
    # claude_code.model nor reviewer.model was set).
    assert data["claude_code"]["model"] == config_module._DEFAULT_CLAUDE_MODEL


def test_resolve_role_dual_accept_worker_model_devin_shell_new_only() -> None:
    data = {"worker": {"harness": "devin-shell", "model": "glm-5-2"}}
    config_module._resolve_role_dual_accept(data)
    assert data["devin"]["worker_model"] == "glm-5-2"
    assert data["worker"]["model"] == "glm-5-2"


def test_resolve_role_dual_accept_worker_model_devin_shell_conflict_raises() -> None:
    data = {
        "devin": {"adapter": "devin-shell", "worker_model": "glm-5-2"},
        "worker": {"harness": "devin-shell", "model": "sonnet-5"},
    }
    with pytest.raises(ConfigError, match="devin.worker_model.*glm-5-2.*worker.model.*sonnet-5"):
        config_module._resolve_role_dual_accept(data)


def test_resolve_role_dual_accept_worker_model_devin_shell_no_model_defaults_empty() -> None:
    data = {"devin": {"adapter": "devin-shell"}}
    config_module._resolve_role_dual_accept(data)
    assert data["worker"]["model"] == ""
    assert data["devin"]["worker_model"] == ""


def test_resolve_role_dual_accept_worker_model_claude_code_old_only() -> None:
    data = {"devin": {"adapter": "claude-code"}, "claude_code": {"model": "claude-opus-4-1"}}
    config_module._resolve_role_dual_accept(data)
    assert data["worker"]["model"] == "claude-opus-4-1"
    assert data["claude_code"]["model"] == "claude-opus-4-1"


def test_resolve_role_dual_accept_worker_model_claude_code_new_only() -> None:
    data = {"worker": {"harness": "claude-code", "model": "claude-opus-4-1"}}
    config_module._resolve_role_dual_accept(data)
    assert data["claude_code"]["model"] == "claude-opus-4-1"
    assert data["worker"]["model"] == "claude-opus-4-1"


def test_resolve_role_dual_accept_worker_model_claude_code_conflict_raises() -> None:
    data = {
        "devin": {"adapter": "claude-code"},
        "claude_code": {"model": "claude-opus-4-1"},
        "worker": {"model": "sonnet-5"},
    }
    with pytest.raises(
        ConfigError,
        match=r"claude_code\.model \(as worker\).*claude-opus-4-1.*worker\.model.*sonnet-5",
    ):
        config_module._resolve_role_dual_accept(data)


def test_resolve_role_dual_accept_worker_model_claude_code_no_model_defaults_to_shared_constant() -> (
    None
):
    # This is the regression this task must not reintroduce: an unconfigured
    # claude-code worker must still get the CLI-pin default, never "".
    data = {"devin": {"adapter": "claude-code"}}
    config_module._resolve_role_dual_accept(data)
    assert data["worker"]["model"] == config_module._DEFAULT_CLAUDE_MODEL
    assert data["claude_code"]["model"] == config_module._DEFAULT_CLAUDE_MODEL


def test_resolve_role_dual_accept_reviewer_harness_defaults_claude_code() -> None:
    data: dict = {}
    config_module._resolve_role_dual_accept(data)
    assert data["reviewer"]["harness"] == "claude-code"


def test_resolve_role_dual_accept_reviewer_harness_rejects_non_claude_code() -> None:
    data = {"reviewer": {"harness": "devin"}}
    with pytest.raises(ConfigError, match="reviewer.*harness.*claude-code.*devin"):
        config_module._resolve_role_dual_accept(data)


def test_resolve_role_dual_accept_reviewer_model_inherits_claude_code_model_when_worker_also_claude_code() -> (
    None
):
    # Today's actual behavior: a review launch with no model_override falls
    # back to claude_code.model. With harness=claude-code and no
    # reviewer.model set, the reviewer must inherit the SAME resolved value
    # as the worker.
    data = {"devin": {"adapter": "claude-code"}, "claude_code": {"model": "claude-opus-4-1"}}
    config_module._resolve_role_dual_accept(data)
    assert data["worker"]["model"] == "claude-opus-4-1"
    assert data["reviewer"]["model"] == "claude-opus-4-1"


def test_resolve_role_dual_accept_reviewer_model_splits_from_worker_model_no_conflict() -> None:
    # The primary incremental-migration path: keep claude_code.model driving
    # the (claude-code) worker, add reviewer.model to decouple the reviewer.
    # This must NOT raise -- it is not a conflict, it is the intended split.
    data = {
        "devin": {"adapter": "claude-code"},
        "claude_code": {"model": "claude-opus-4-1"},
        "reviewer": {"model": "claude-sonnet-5"},
    }
    config_module._resolve_role_dual_accept(data)
    assert data["worker"]["model"] == "claude-opus-4-1"
    assert data["reviewer"]["model"] == "claude-sonnet-5"
    # claude_code.model stays claimed by the worker -- unaffected by the split.
    assert data["claude_code"]["model"] == "claude-opus-4-1"


def test_resolve_role_dual_accept_reviewer_model_non_claude_code_worker_old_only() -> None:
    data = {"devin": {"adapter": "devin-shell"}, "claude_code": {"model": "claude-opus-4-1"}}
    config_module._resolve_role_dual_accept(data)
    assert data["reviewer"]["model"] == "claude-opus-4-1"
    assert data["claude_code"]["model"] == "claude-opus-4-1"


def test_resolve_role_dual_accept_reviewer_model_non_claude_code_worker_conflict_raises() -> None:
    data = {
        "devin": {"adapter": "devin-shell"},
        "claude_code": {"model": "claude-opus-4-1"},
        "reviewer": {"model": "claude-sonnet-5"},
    }
    with pytest.raises(
        ConfigError,
        match=r"claude_code\.model \(as reviewer\).*claude-opus-4-1.*reviewer\.model.*claude-sonnet-5",
    ):
        config_module._resolve_role_dual_accept(data)


def test_resolve_role_dual_accept_reviewer_model_no_config_anywhere_defaults_to_shared_constant() -> (
    None
):
    data: dict = {}
    config_module._resolve_role_dual_accept(data)
    assert data["reviewer"]["model"] == config_module._DEFAULT_CLAUDE_MODEL


def test_resolve_role_dual_accept_reviewer_effort_old_only() -> None:
    data = {"review_dispatch": {"review_effort": "high"}}
    config_module._resolve_role_dual_accept(data)
    assert data["review_dispatch"]["review_effort"] == "high"
    assert data["reviewer"]["effort"] == "high"


def test_resolve_role_dual_accept_reviewer_effort_new_only() -> None:
    data = {"reviewer": {"effort": "high"}}
    config_module._resolve_role_dual_accept(data)
    assert data["review_dispatch"]["review_effort"] == "high"
    assert data["reviewer"]["effort"] == "high"


def test_resolve_role_dual_accept_reviewer_effort_conflict_raises() -> None:
    data = {"review_dispatch": {"review_effort": "high"}, "reviewer": {"effort": "low"}}
    with pytest.raises(
        ConfigError, match="review_dispatch.review_effort.*high.*reviewer.effort.*low"
    ):
        config_module._resolve_role_dual_accept(data)


def test_resolve_role_dual_accept_reviewer_experiment_fraction_and_salt_dual_accept() -> None:
    data = {
        "reviewer": {
            "effort": "high",
            "effort_experiment_fraction": 0.5,
            "effort_experiment_salt": "abc",
        }
    }
    config_module._resolve_role_dual_accept(data)
    assert data["review_dispatch"]["review_effort_experiment_fraction"] == 0.5
    assert data["review_dispatch"]["review_effort_experiment_salt"] == "abc"


def test_resolve_role_dual_accept_reviewer_experiment_fraction_conflict_raises() -> None:
    data = {
        "review_dispatch": {"review_effort_experiment_fraction": 0.5},
        "reviewer": {"effort_experiment_fraction": 0.75},
    }
    with pytest.raises(
        ConfigError,
        match="review_dispatch.review_effort_experiment_fraction.*0.5.*"
        "reviewer.effort_experiment_fraction.*0.75",
    ):
        config_module._resolve_role_dual_accept(data)


def test_resolve_role_dual_accept_rescue_worker_old_only() -> None:
    data = {"rescue": {"worker_adapter": "devin-shell", "worker_model": "glm-5-2"}}
    config_module._resolve_role_dual_accept(data)
    assert data["rescue"]["worker_adapter"] == "devin-shell"
    assert data["rescue"]["worker_model"] == "glm-5-2"
    assert data["rescue"]["worker"] == {"harness": "devin-shell", "model": "glm-5-2"}


def test_resolve_role_dual_accept_rescue_worker_new_only() -> None:
    data = {"rescue": {"worker": {"harness": "devin-shell", "model": "glm-5-2"}}}
    config_module._resolve_role_dual_accept(data)
    assert data["rescue"]["worker_adapter"] == "devin-shell"
    assert data["rescue"]["worker_model"] == "glm-5-2"


def test_resolve_role_dual_accept_rescue_worker_conflict_raises() -> None:
    data = {
        "rescue": {
            "worker_adapter": "claude-code",
            "worker": {"harness": "devin-shell"},
        }
    }
    with pytest.raises(
        ConfigError, match="rescue.worker_adapter.*claude-code.*rescue.worker.harness.*devin-shell"
    ):
        config_module._resolve_role_dual_accept(data)


def test_resolve_role_dual_accept_rescue_reviewer_old_only() -> None:
    data = {"rescue": {"reviewer_adapter": "devin", "reviewer_model": "codex"}}
    config_module._resolve_role_dual_accept(data)
    assert data["rescue"]["reviewer"] == {"harness": "devin", "model": "codex"}


def test_resolve_role_dual_accept_rescue_defaults_preserve_current_rescueconfig_defaults() -> None:
    data: dict = {}
    config_module._resolve_role_dual_accept(data)
    assert data["rescue"]["worker_adapter"] == "claude-code"
    assert data["rescue"]["worker_model"] == "claude-opus-4-1"
    assert data["rescue"]["reviewer_adapter"] == "devin"
    assert data["rescue"]["reviewer_model"] == "codex"
    assert data["rescue"]["worker"] == {"harness": "claude-code", "model": "claude-opus-4-1"}
    assert data["rescue"]["reviewer"] == {"harness": "devin", "model": "codex"}


def test_build_config_from_data_rescue_worker_reviewer_are_workerroleconfig_instances() -> None:
    cfg = build_config_from_data(
        {"rescue": {"worker": {"harness": "devin-shell", "model": "glm-5-2"}}}
    )
    assert isinstance(cfg.rescue.worker, WorkerRoleConfig)
    assert cfg.rescue.worker == WorkerRoleConfig(harness="devin-shell", model="glm-5-2")
    assert isinstance(cfg.rescue.reviewer, WorkerRoleConfig)
    assert cfg.rescue.reviewer == WorkerRoleConfig(harness="devin", model="codex")


def test_build_config_from_data_wires_worker_and_reviewer_end_to_end() -> None:
    cfg = build_config_from_data({"devin": {"adapter": "devin-shell", "worker_model": "glm-5-2"}})
    assert cfg.worker.harness == "devin-shell"
    assert cfg.worker.model == "glm-5-2"
    assert cfg.devin.adapter == "devin-shell"
    assert cfg.devin.worker_model == "glm-5-2"


def test_build_config_from_data_records_deprecations_for_old_keys() -> None:
    cfg = build_config_from_data({"devin": {"adapter": "devin-shell"}})
    assert any("devin.adapter is deprecated" in msg for msg in cfg.deprecations)


def test_build_config_from_data_no_deprecations_for_pure_new_style_config() -> None:
    cfg = build_config_from_data(
        {"worker": {"harness": "devin-shell"}, "reviewer": {"model": "x"}}
    )
    assert cfg.deprecations == ()


def test_build_config_from_data_worker_model_tier_presence_is_deprecated() -> None:
    cfg = build_config_from_data({"dispatch": {"worker_model_tier": "capable"}})
    assert any("dispatch.worker_model_tier is deprecated" in msg for msg in cfg.deprecations)


def test_build_config_from_data_fallback_adapter_presence_is_deprecated() -> None:
    cfg = build_config_from_data({"api_worker": {"fallback_adapter": "devin-shell"}})
    assert any("api_worker.fallback_adapter is deprecated" in msg for msg in cfg.deprecations)


def test_build_config_from_data_cross_family_section_presence_is_deprecated_even_without_enabled() -> (
    None
):
    cfg = build_config_from_data({"cross_family": {"command": ["devin", "--model", "{model}"]}})
    assert any("cross_family is deprecated" in msg for msg in cfg.deprecations)


def test_build_config_from_data_cross_family_deprecation_names_emergent_status() -> None:
    cfg = build_config_from_data(
        {
            "cross_family": {"enabled": True},
            "devin": {"adapter": "devin-shell", "worker_model": "glm-5-2"},
            "reviewer": {"model": "claude-sonnet-5"},
        }
    )
    msg = next(m for m in cfg.deprecations if m.startswith("cross_family"))
    assert "worker='glm-5-2'" in msg
    assert "reviewer='claude-sonnet-5'" in msg
    assert "cross-family: yes" in msg


def test_build_config_from_data_unknown_worker_key_still_raises_configerror() -> None:
    with pytest.raises(ConfigError, match="unknown key.*worker.*bogus"):
        build_config_from_data({"worker": {"bogus": "x"}})
