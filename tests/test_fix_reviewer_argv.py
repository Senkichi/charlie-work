"""Regression tests for review-pipeline audit Finding 1.

The installed Claude Code CLI hard-rejects `--print` + `--output-format
stream-json` unless `--verbose` is also present ("Error: When using --print,
--output-format=stream-json requires --verbose"). `dispatch_reviews()`
unconditionally sets `tee_stream_json=True` for reviewer sessions
(workflow.py), so every reviewer launch crashed in <1s until
`launch_claude_worker`'s tee_stream_json branch was fixed to pair the two
flags. None of the state-machine tests added by PRs #547/#549/#550 exercised
the literal argv handed to the CLI, so this crash went uncaught.

These tests drive the real `launch_claude_worker` command-construction path
(not a private helper) and assert on `record.command`, the fully-rendered
argv actually passed to `popen_worker`.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from charlie_work import claude_code
from charlie_work.claude_code import ClaudeWorkerRecord, launch_claude_worker
from charlie_work.worktree import WorktreeInfo


def _fake_worktree(tmp_path: Path, branch: str) -> WorktreeInfo:
    worktree_path = tmp_path / "worktrees" / branch.replace("/", "-")
    worktree_path.mkdir(parents=True, exist_ok=True)
    return WorktreeInfo(path=worktree_path, branch=branch, venv_junction=None)


def _install_fake_create_worktree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_create_worktree(
        repo_root,
        branch,
        *,
        base_ref="HEAD",
        worktrees_dir=None,
        venv_source=None,
        materialize_dirs=(),
        rework=False,
        recovery=None,
        issue_number=None,
        config=None,
        sessions_dir=None,
    ):
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)


def _fake_claude_script(tmp_path: Path, name: str = "fake_claude.py") -> tuple[str, ...]:
    """Stand-in for the `claude` binary: reads stdin and exits 0 immediately.

    Ignores any extra argv (e.g. --output-format/--verbose) since this test
    only cares about what argv `launch_claude_worker` constructs, not CLI
    argument parsing.
    """
    script_path = tmp_path / name
    script_path.write_text(
        textwrap.dedent(
            """
            import sys

            sys.stdin.read()
            print("ok")
            """
        ),
        encoding="utf-8",
    )
    return (sys.executable, str(script_path))


def _launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    command_template: tuple[str, ...],
    tee_stream_json: bool,
) -> ClaudeWorkerRecord:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    return launch_claude_worker(
        99,
        "agent/issue-99-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=command_template,
        tee_stream_json=tee_stream_json,
    )


def test_tee_stream_json_enabled_pairs_verbose_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """tee_stream_json=True (forced for every reviewer, workflow.py
    dispatch_reviews) must inject --verbose alongside --output-format
    stream-json — the CLI rejects the pair without it."""
    record = _launch(
        monkeypatch,
        tmp_path,
        command_template=_fake_claude_script(tmp_path),
        tee_stream_json=True,
    )

    assert record.ok, record.error
    assert "--output-format" in record.command
    assert "stream-json" in record.command
    assert "--verbose" in record.command


def test_tee_stream_json_disabled_injects_neither_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ordinary worker dispatch defaults tee_stream_json to False
    (config.py ClaudeCodeConfig.tee_stream_json) and must not pick up either
    flag as a side effect of the reviewer fix."""
    record = _launch(
        monkeypatch,
        tmp_path,
        command_template=_fake_claude_script(tmp_path),
        tee_stream_json=False,
    )

    assert record.ok, record.error
    assert "--output-format" not in record.command
    assert "stream-json" not in record.command
    assert "--verbose" not in record.command


def test_tee_stream_json_does_not_duplicate_existing_verbose_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A command_template that already carries --verbose (e.g. a future
    caller-supplied override) must not get a second, duplicate flag
    appended."""
    base = _fake_claude_script(tmp_path)
    template_with_verbose = (*base, "--verbose")

    record = _launch(
        monkeypatch,
        tmp_path,
        command_template=template_with_verbose,
        tee_stream_json=True,
    )

    assert record.ok, record.error
    assert record.command.count("--verbose") == 1
    assert "--output-format" in record.command
    assert "stream-json" in record.command
