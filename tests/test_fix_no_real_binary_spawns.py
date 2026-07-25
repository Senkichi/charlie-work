"""Regression tests for the suite-wide real-CLI spawn guard (issue #569).

Launch-path tests that omit ``command_template`` used to resolve the default
template's bare ``"claude"`` through ``resolve_cli_binary`` and genuinely
spawn the operator's real installed CLI — burning API quota (Sonnet *and*
Opus configs), raising "your prompt came through empty" OS toasts, and
flashing console windows from the orphan sessions' hook children — on every
local suite run and every push through the self-hosted CI runner. The
autouse ``_no_real_cli_binaries`` conftest fixture closes that hole; these
tests pin its contract.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from charlie_work.claude_code import launch_claude_worker
from test_claude_code_adapter import _install_fake_create_worktree


def test_default_template_launch_never_resolves_a_real_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A launch with no command_template must never spawn a real installed
    claude: on machines where the CLI exists it is replaced by the harmless
    interpreter fake; where it doesn't, the launch fails as before."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=tmp_path / "sessions",
    )

    if shutil.which("claude"):
        assert record.command[0] == sys.executable
        assert "claude" not in Path(record.command[0]).name.lower()
    else:
        assert record.error is not None


def test_explicit_absolute_path_still_resolves_normally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests that pass explicit absolute paths (sys.executable fakes) keep
    resolving unchanged through the guard."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=tmp_path / "sessions",
        command_template=(sys.executable, "-c", "import sys; sys.stdin.read()"),
    )

    assert record.command[0] == sys.executable
    assert record.error is None
