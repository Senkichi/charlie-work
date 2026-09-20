"""``load_config`` validation of misc sections: cost/token budgets,
worker env mappings, materialize_dirs, merge_flags, and review_effort.

Split out of ``tests/test_charlie_work.py`` (issue #1552, Track-1
wave 6/8).
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.config import load_config


def test_config_accepts_cost_token_budgets(tmp_path: Path) -> None:
    """A YAML block with cost/token budget fields loads correctly."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  enabled: true
  stall_minutes: 20
  cost_budget_usd: 10.0
  token_budget: 100000
  cost_budget_action: warn
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.watchdog.enabled is True
    assert config.watchdog.stall_minutes == 20
    assert config.watchdog.cost_budget_usd == 10.0
    assert config.watchdog.token_budget == 100000
    assert config.watchdog.cost_budget_action == "warn"


def test_config_defaults_cost_token_budgets(tmp_path: Path) -> None:
    """A YAML block without cost/token budget fields uses the defaults (None)."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  enabled: true
  stall_minutes: 20
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.watchdog.enabled is True
    assert config.watchdog.stall_minutes == 20
    assert config.watchdog.cost_budget_usd is None
    assert config.watchdog.token_budget is None
    assert config.watchdog.cost_budget_action == "warn"


def test_config_rejects_invalid_cost_budget_usd_type(tmp_path: Path) -> None:
    """cost_budget_usd must be a number."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  cost_budget_usd: "not a number"
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for invalid cost_budget_usd type")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid cost_budget_usd type")

    assert "section 'watchdog'" in message
    assert "cost_budget_usd" in message
    assert "must be a number" in message


def test_config_rejects_invalid_token_budget_type(tmp_path: Path) -> None:
    """token_budget must be an int."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  token_budget: "not an int"
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for invalid token_budget type")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid token_budget type")

    assert "section 'watchdog'" in message
    assert "token_budget" in message
    assert "must be an int" in message


def test_config_rejects_invalid_cost_budget_action_type(tmp_path: Path) -> None:
    """cost_budget_action must be a string."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  cost_budget_action: 123
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for invalid cost_budget_action type")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid cost_budget_action type")

    assert "section 'watchdog'" in message
    assert "cost_budget_action" in message
    assert "must be a string" in message


def test_config_rejects_invalid_cost_budget_action_value(tmp_path: Path) -> None:
    """cost_budget_action must be 'warn' or 'kill'."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """watchdog:
  cost_budget_action: "invalid"
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
        raise AssertionError("expected ConfigError for invalid cost_budget_action value")
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for invalid cost_budget_action value")

    assert "section 'watchdog'" in message
    assert "cost_budget_action" in message
    assert "must be 'warn' or 'kill'" in message


def test_config_worker_env_coerces_values_to_str(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "claude_code:\n  worker_env:\n    PYTEST_XDIST_AUTO_NUM_WORKERS: 2\n",
        encoding="utf-8",
    )

    config = load_config(path)

    # YAML parses the bare 2 as an int; env values must be strings for Popen.
    assert config.claude_code.worker_env == {"PYTEST_XDIST_AUTO_NUM_WORKERS": "2"}


def test_config_rejects_non_mapping_worker_env(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    path = tmp_path / "c.yaml"
    # A plausible operator typo: a scalar instead of a name->value mapping.
    # Must fail at load, not as an AttributeError when a worker launches.
    path.write_text('claude_code:\n  worker_env: "2"\n', encoding="utf-8")

    try:
        load_config(path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")

    assert "worker_env" in message
    assert "claude_code" in message


def test_config_rejects_non_mapping_devin_worker_env(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    path = tmp_path / "c.yaml"
    # Same validation for devin.worker_env
    path.write_text('devin:\n  worker_env: "2"\n', encoding="utf-8")

    try:
        load_config(path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")

    assert "worker_env" in message
    assert "devin" in message


def test_config_rejects_non_list_materialize_dirs(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    path = tmp_path / "c.yaml"
    # A plausible operator typo: a scalar instead of a list.
    path.write_text('dispatch:\n  materialize_dirs: ".devin"\n', encoding="utf-8")

    try:
        load_config(path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")

    assert "materialize_dirs" in message
    assert "dispatch" in message
    assert "list" in message


def test_config_rejects_merge_flags_not_starting_with_double_dash(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    path = tmp_path / "c.yaml"
    # A plausible operator typo: a flag without the -- prefix.
    # Must fail at load with a clear error message.
    path.write_text('auto_merge:\n  merge_flags: ["admin"]\n', encoding="utf-8")

    try:
        load_config(path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")

    assert "merge_flags" in message
    assert "auto_merge" in message
    assert "must start with '--'" in message


def test_config_accepts_valid_merge_flags(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    # Valid merge_flags with -- prefix (non-managed flags only).
    path.write_text('auto_merge:\n  merge_flags: ["--auto", "--subject"]\n', encoding="utf-8")

    config = load_config(path)

    assert config.auto_merge.merge_flags == ("--auto", "--subject")


def test_config_rejects_orchestrator_managed_merge_flags(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    # Test each orchestrator-managed flag
    for flag in ["--merge", "--rebase", "--squash", "--delete-branch", "--admin"]:
        path = tmp_path / "c.yaml"
        path.write_text(f'auto_merge:\n  merge_flags: ["{flag}"]\n', encoding="utf-8")

        try:
            load_config(path)
            raise AssertionError(f"expected ConfigError for {flag}")
        except ConfigError as exc:
            message = str(exc)
            assert "merge_flags" in message
            assert "auto_merge" in message
            assert "managed by the orchestrator" in message
            assert flag in message


def test_config_rejects_orchestrator_managed_merge_flags_equals_form(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    # Test that --flag=value forms are also rejected (normalization splits on '=')
    # --delete-branch=true is the critical case: it bypasses exact match but is valid gh syntax
    for flag in ["--delete-branch=true", "--squash=true"]:
        path = tmp_path / "c.yaml"
        path.write_text(f'auto_merge:\n  merge_flags: ["{flag}"]\n', encoding="utf-8")

        try:
            load_config(path)
            raise AssertionError(f"expected ConfigError for {flag}")
        except ConfigError as exc:
            message = str(exc)
            assert "merge_flags" in message
            assert "auto_merge" in message
            assert "managed by the orchestrator" in message
            assert flag in message


def test_config_rejects_merge_flags_scalar(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    path = tmp_path / "c.yaml"
    # YAML scalar instead of list - this would iterate per-character
    path.write_text('auto_merge:\n  merge_flags: "--admin"\n', encoding="utf-8")

    try:
        load_config(path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")

    assert "merge_flags" in message
    assert "auto_merge" in message
    assert "must be a list" in message


def test_config_rejects_non_string_review_effort(tmp_path: Path) -> None:
    from charlie_work.config import ConfigError

    path = tmp_path / "c.yaml"
    path.write_text("reviewer:\n  effort: 3\n", encoding="utf-8")

    try:
        load_config(path)
        raise AssertionError("expected ConfigError")
    except ConfigError as exc:
        message = str(exc)

    assert "effort" in message
    assert "reviewer" in message
    assert "must be a string" in message


def test_config_accepts_string_review_effort(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("reviewer:\n  effort: high\n", encoding="utf-8")

    config = load_config(path)

    assert config.reviewer.effort == "high"
