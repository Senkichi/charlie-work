"""Command-template tests for the devin-shell adapter.

Split out of ``tests/test_devin_shell.py`` (issue #1542, Track-1 pilot):
these tests pin ``DEFAULT_COMMAND_TEMPLATE``'s shape (``--permission-mode
dangerous``, ``--respect-workspace-trust false``, flag adjacency) and the
issue/branch placeholder + worker-model injection rendering of the launch
command.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from _devin_shell_fixtures import (
    _FAKE_DEVIN_SLEEP,
    _install_fake_create_worktree,
    _write_fake_devin,
)

from charlie_work.devin_shell import DEFAULT_COMMAND_TEMPLATE, launch_devin_session


def test_default_command_template_contains_permission_mode_dangerous() -> None:
    """Devin CLI defaults to --permission-mode auto (read-only); headless
    workers stall the moment they need git/uv/gh. The default template must
    explicitly pass --permission-mode dangerous."""
    template = DEFAULT_COMMAND_TEMPLATE
    template_str = " ".join(template)
    assert "--permission-mode" in template_str, (
        "DEFAULT_COMMAND_TEMPLATE must contain '--permission-mode'"
    )
    assert "dangerous" in template_str, (
        "DEFAULT_COMMAND_TEMPLATE must set --permission-mode dangerous"
    )
    assert "{model_args}" in template_str, (
        "DEFAULT_COMMAND_TEMPLATE must contain '{model_args}' placeholder for config-driven model selection"
    )


def test_default_command_template_disables_workspace_trust() -> None:
    """The 2026-08 Devin CLI enforces workspace trust in --print mode and
    fails hard in untrusted directories (worker worktrees always are, being
    created fresh per issue). The template must pass the CLI's documented
    escape hatch for non-interactive runs, as consecutive argv tokens."""
    template = DEFAULT_COMMAND_TEMPLATE
    assert "--respect-workspace-trust" in template, (
        "DEFAULT_COMMAND_TEMPLATE must contain '--respect-workspace-trust'"
    )
    idx = template.index("--respect-workspace-trust")
    assert template[idx + 1] == "false", (
        f"Expected 'false' after '--respect-workspace-trust', got {template[idx + 1]!r}"
    )


def test_default_command_template_permission_mode_flag_is_adjacent() -> None:
    """--permission-mode and dangerous must be consecutive argv tokens."""
    # After rendering with an empty worker_model, {model_args} becomes an empty
    # string and is filtered out, so --permission-mode and dangerous are adjacent.
    from charlie_work.devin_shell import _render_command

    rendered = _render_command(
        DEFAULT_COMMAND_TEMPLATE,
        issue_number=1,
        branch="x",
        prompt_path=Path("p.md"),
        worker_model="",
    )
    idx = rendered.index("--permission-mode")
    assert rendered[idx + 1] == "dangerous", (
        f"Expected 'dangerous' after '--permission-mode', got {rendered[idx + 1]!r}"
    )


def test_command_template_renders_issue_and_branch_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_SLEEP)

    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_devin_session(
        42,
        "agent/issue-42-widgets",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(
            sys.executable,
            str(script),
            "--issue",
            "{issue_number}",
            "--branch",
            "{branch}",
            "--prompt-file",
            "{prompt_path}",
        ),
    )

    assert record.command == (
        sys.executable,
        str(script),
        "--issue",
        "42",
        "--branch",
        "agent/issue-42-widgets",
        "--prompt-file",
        str(prompt_path),
    )
    # Regression guard: empty strings should not appear in the rendered command
    assert "" not in record.command


def test_command_template_injects_model_when_worker_model_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When worker_model is set, the rendered command must include --model <value>."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_SLEEP)

    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_devin_session(
        42,
        "agent/issue-42-widgets",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script), "{model_args}", "{prompt_path}"),
        worker_model="claude-sonnet-4-5",
    )

    # The rendered command must include --model claude-sonnet-4-5 as separate tokens
    assert "--model" in record.command
    model_idx = record.command.index("--model")
    assert record.command[model_idx + 1] == "claude-sonnet-4-5"
    # Verify the custom template structure
    assert record.command[0] == sys.executable
    assert str(script) in record.command
    # Regression guard: empty strings should not appear in the rendered command
    assert "" not in record.command


def test_command_template_omits_model_when_worker_model_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When worker_model is empty, the rendered command must omit --model."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_SLEEP)

    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_devin_session(
        42,
        "agent/issue-42-widgets",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script), "{model_args}", "{prompt_path}"),
        worker_model="",
    )

    # The rendered command must NOT include --model
    assert "--model" not in record.command
    # Verify the custom template structure
    assert record.command[0] == sys.executable
    assert str(script) in record.command
    # Regression guard: empty strings should not appear in the rendered command
    assert "" not in record.command
