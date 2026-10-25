"""Issue #2836 launch-argv tool-scope and MCP-isolation coverage.

Extracted here rather than appended to
``tests/test_claude_code_adapter_launch_review.py`` because that module was
already near the file-size cap (tests/test_file_size_ratchet.py): the full
``launch_claude_worker`` argv assertions for both roles, the
caller-supplied-template strip, both kill switches, the ``cli_pins`` argv
invariance, and the quota probe's argv. The pure ``_apply_tool_scope``
rewriter units live in ``tests/test_claude_code_adapter_command_template.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from _claude_adapter_fixtures import (
    _fake_claude_script,
    _install_fake_create_worktree,
)

from charlie_work import claude_code
from charlie_work.claude_code import launch_claude_worker
from charlie_work.config import (
    ClaudeCodeConfig,
    OrchestratorConfig,
    QuotaProbeConfig,
)
from charlie_work.worktree import WorktreeInfo

_WORKER_TOOLS = "Bash,Read,Edit,Write,Grep,Glob,Monitor,TaskStop"
_REVIEWER_TOOLS = "Bash,Read,Grep,Glob,Write"
_ISOLATED_MCP_FLAGS = ("--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}')
_HEAD_SHA = "a" * 40


def _assert_isolated(argv: tuple[str, ...]) -> None:
    for flag in _ISOLATED_MCP_FLAGS:
        assert flag in argv


@pytest.fixture(autouse=True)
def _no_real_spawn(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Keep every test in this module off real subprocess spawns (issue #2689).

    These tests assert only on the rendered argv. A review launch with the
    default template (or a template whose argv[0] is the literal ``claude``)
    resolves to the real binary on a host that has the CLI installed and
    spawns a live agent session -- and fails with WinError 2 on a CI runner
    that does not. Faking the two seams -- ``popen_worker`` and
    ``create_review_checkout`` -- makes the result independent of the host.
    Returns the argv list of every (faked) spawn so a test can prove the
    launch reached the spawn seam.
    """
    spawned: list[tuple[str, ...]] = []

    def fake_popen_worker(command, *args, **kwargs):
        spawned.append(tuple(command))
        return SimpleNamespace(pid=4242)

    def fake_create_review_checkout(
        repo_root: Path, pr_number: int, head_sha: str, *, reviews_dir: Path
    ) -> WorktreeInfo:
        checkout_path = reviews_dir / f"pr-{pr_number}"
        checkout_path.mkdir(parents=True, exist_ok=True)
        return WorktreeInfo(path=checkout_path, branch=head_sha, venv_junction=None)

    monkeypatch.setattr(claude_code, "popen_worker", fake_popen_worker)
    monkeypatch.setattr(claude_code, "create_review_checkout", fake_create_review_checkout)
    return spawned


def test_launch_claude_worker_argv_pins_worker_tool_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        2836,
        "agent/issue-2836-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert record.ok, record.error
    assert record.command.count("--tools") == 1
    idx = record.command.index("--tools")
    assert record.command[idx + 1] == _WORKER_TOOLS
    _assert_isolated(record.command)


def test_launch_claude_worker_review_argv_pins_reviewer_tool_scope(
    tmp_path: Path, _no_real_spawn: list[tuple[str, ...]]
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "reviews"

    record = launch_claude_worker(
        2837,
        "agent/issue-2836-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=_HEAD_SHA,
    )

    assert record.ok, record.error
    assert _no_real_spawn == [record.command]
    assert record.command.count("--tools") == 1
    idx = record.command.index("--tools")
    assert record.command[idx + 1] == _REVIEWER_TOOLS
    _assert_isolated(record.command)


def test_launch_claude_worker_command_template_tool_override_stripped_for_both_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller-supplied command_template carrying its own wider --tools /
    --mcp-config comes out with only the configured scope, both roles."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "reviews"
    sneaky_template = (
        "claude",
        "-p",
        "--tools",
        "Artifact,Workflow",
        "--tools=Agent",
        "--mcp-config",
        "other.json",
        "--mcp-config=other2.json",
    )

    _install_fake_create_worktree(monkeypatch, tmp_path)
    worker_record = launch_claude_worker(
        2838,
        "agent/issue-2836-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=sneaky_template,
    )

    assert worker_record.ok, worker_record.error
    assert worker_record.command.count("--tools") == 1
    idx = worker_record.command.index("--tools")
    assert worker_record.command[idx + 1] == _WORKER_TOOLS
    assert "Artifact" not in worker_record.command
    assert "other.json" not in worker_record.command
    assert "other2.json" not in worker_record.command
    _assert_isolated(worker_record.command)

    review_record = launch_claude_worker(
        2839,
        "agent/issue-2836-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=sneaky_template,
        review=True,
        head_sha=_HEAD_SHA,
    )

    assert review_record.ok, review_record.error
    assert review_record.command.count("--tools") == 1
    idx = review_record.command.index("--tools")
    assert review_record.command[idx + 1] == _REVIEWER_TOOLS
    assert "Artifact" not in review_record.command
    assert "other.json" not in review_record.command
    _assert_isolated(review_record.command)


def test_launch_claude_worker_tool_kill_switches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """worker_tools: "" adds no --tools flag; isolate_mcp: false adds no MCP
    flags — both restore today's (pre-#2836) argv shape."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    config = OrchestratorConfig(claude_code=ClaudeCodeConfig(worker_tools="", isolate_mcp=False))

    record = launch_claude_worker(
        2840,
        "agent/issue-2836-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        config=config,
    )

    assert record.ok, record.error
    assert not any(tok.startswith("--tools") for tok in record.command)
    assert "--strict-mcp-config" not in record.command
    assert "--mcp-config" not in record.command


def test_launch_claude_worker_cli_pins_path_argv_has_no_tool_scope_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cli_pins (non-Claude CLI) path's argv is unchanged: no --tools and
    no MCP-isolation flags may leak into a foreign CLI's argv."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    base_template = ("somecli", "-p")

    def cli_pins(template: tuple[str, ...]) -> tuple[str, ...]:
        return template + ("--model", "provider/model")

    record = launch_claude_worker(
        2841,
        "agent/issue-2836-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=base_template,
        cli_pins=cli_pins,
    )

    assert record.ok, record.error
    assert record.command[: len(base_template)] == base_template
    assert not any(tok.startswith("--tools") for tok in record.command)
    assert "--strict-mcp-config" not in record.command
    assert "--mcp-config" not in record.command


def test_quota_probe_argv_carries_reviewer_tool_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_quota_probe builds its argv from _REVIEW_COMMAND_TEMPLATE, so it
    gets the reviewer tool scope + MCP isolation pins too (issue #2836)."""
    captured: list[tuple] = []

    def fake_run_captured(argv, **kwargs):
        captured.append(tuple(argv))
        return claude_code.RunResult(stdout="", stderr="", returncode=0, timed_out=False)

    monkeypatch.setattr(claude_code, "run_captured", fake_run_captured)
    monkeypatch.setattr(claude_code, "resolve_cli_binary", lambda binary: binary)

    ok = claude_code.run_quota_probe(
        repo_root=tmp_path,
        config=OrchestratorConfig(
            quota_probe=QuotaProbeConfig(model="claude-haiku-5"),
        ),
    )

    assert ok is True
    argv = captured[0]
    assert argv.count("--tools") == 1
    idx = argv.index("--tools")
    assert argv[idx + 1] == _REVIEWER_TOOLS
    _assert_isolated(argv)
