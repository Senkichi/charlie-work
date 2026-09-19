"""``load_config`` placeholder validation for shell-command templates:
unknown/empty/positional placeholders, bare/unclosed/stray braces, and
the claude-code and dispatch command templates.

Split out of ``tests/test_charlie_work.py`` (issue #1552, Track-1
wave 6/8).
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.config import load_config


def test_load_config_rejects_unknown_shell_command_placeholder(tmp_path: Path) -> None:
    """Issue #4: unknown placeholder in shell_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  shell_command:\n    - devin\n    - "{unknown_placeholder}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown placeholder")

    assert "devin.shell_command" in message
    assert "unknown_placeholder" in message


def test_load_config_rejects_empty_placeholder_in_shell_command(tmp_path: Path) -> None:
    """Issue #4: empty placeholder {} in shell_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  shell_command:\n    - devin\n    - "{}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for empty placeholder")

    assert "devin.shell_command" in message
    assert "empty placeholder" in message


def test_load_config_rejects_unknown_claude_code_command_placeholder(tmp_path: Path) -> None:
    """Issue #4: unknown placeholder in claude_code.command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'claude_code:\n  command:\n    - claude\n    - "{bad_token}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown placeholder")

    assert "claude_code.command" in message
    assert "bad_token" in message


def test_load_config_accepts_valid_placeholders(tmp_path: Path) -> None:
    """Issue #4: valid placeholders in command templates are accepted."""
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        """devin:
  shell_command:
    - devin
    - "{prompt_path}"
    - "{issue_number}"
    - "{branch}"
claude_code:
  command:
    - claude
    - "{prompt_path}"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.devin.shell_command == ("devin", "{prompt_path}", "{issue_number}", "{branch}")
    assert config.claude_code.command == ("claude", "{prompt_path}")


def test_load_config_rejects_bare_brace_in_shell_command(tmp_path: Path) -> None:
    """Issue #4: bare { in shell_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  shell_command:\n    - devin\n    - "test{"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for bare brace")

    assert "devin.shell_command" in message
    assert "malformed placeholder" in message


def test_load_config_rejects_unclosed_brace_in_shell_command(tmp_path: Path) -> None:
    """Issue #4: unclosed {prompt_path in shell_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  shell_command:\n    - devin\n    - "{prompt_path"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unclosed brace")

    assert "devin.shell_command" in message
    assert "malformed placeholder" in message


def test_load_config_rejects_stray_closing_brace_in_shell_command(tmp_path: Path) -> None:
    """Issue #4: stray } in shell_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  shell_command:\n    - devin\n    - "test}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for stray closing brace")

    assert "devin.shell_command" in message
    assert "malformed placeholder" in message


def test_load_config_rejects_positional_placeholder_in_shell_command(tmp_path: Path) -> None:
    """Issue #4: positional {0} in shell_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  shell_command:\n    - devin\n    - "{0}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for positional placeholder")

    assert "devin.shell_command" in message
    # Positional placeholders are caught as unknown (not in allowed set) or malformed
    assert "unknown placeholder" in message or "malformed placeholder" in message


def test_load_config_rejects_unknown_placeholder_in_dispatch_command(tmp_path: Path) -> None:
    """Issue #4: unknown placeholder in dispatch_command is rejected at load."""
    from charlie_work.config import ConfigError

    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text(
        'devin:\n  dispatch_command:\n    - echo\n    - "{bad_token}"',
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ConfigError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError for unknown placeholder")

    assert "devin.dispatch_command" in message
    assert "bad_token" in message
