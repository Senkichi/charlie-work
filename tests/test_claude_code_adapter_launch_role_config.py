"""Launch-time worker/reviewer role-config resolution: model
selection fallbacks, review-effort isolation, permission mode.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from _claude_adapter_fixtures import (
    _init_real_repo,
    _install_fake_create_worktree,
)

from charlie_work import claude_code
from charlie_work.config import (
    ClaudeCodeConfig,
    OrchestratorConfig,
    ReviewerRoleConfig,
    WorkerRoleConfig,
)
from charlie_work.claude_code import launch_claude_worker


def test_launch_claude_worker_pins_configured_model_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #530: a worker launch with no explicit config must still pin
    the shared default model (``claude_code._DEFAULT_CLAUDE_MODEL``) — never
    fall back to ambient global CLI state (the 2026-07-22 outage: every
    reviewer launch silently inherited an interactive session's premium
    `/model` choice and hit a credits wall)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
    )

    assert "--model" in record.command
    idx = record.command.index("--model")
    assert record.command[idx + 1] == claude_code._DEFAULT_CLAUDE_MODEL


def test_launch_claude_worker_honors_configured_model_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    # Role-config Phase 1.5: the model-pin fallback (no model_override passed)
    # reads worker.model, not claude_code.model -- see claude_code.py's single
    # enforcement point in launch_claude_worker.
    config = OrchestratorConfig(worker=WorkerRoleConfig(model="claude-opus-4-8"))

    record = launch_claude_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        config=config,
    )

    assert record.command.count("--model") == 1
    idx = record.command.index("--model")
    assert record.command[idx + 1] == "claude-opus-4-8"


def test_launch_claude_worker_model_override_wins_over_worker_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1245: an explicit ``model_override`` is pinned as the ``--model``
    value instead of ``worker.model`` (role-config Phase 1.5; formerly
    ``claude_code.model``). This is the seam the api adapter uses to pin the
    provider's model, and the reviewer launch site uses to pin
    ``reviewer.model``. The two values deliberately differ so a regression
    that ignores the override is caught."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    config = OrchestratorConfig(worker=WorkerRoleConfig(model="claude-sonnet-5"))

    record = launch_claude_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        config=config,
        model_override="kimi-k3",
    )

    assert record.command.count("--model") == 1
    idx = record.command.index("--model")
    assert record.command[idx + 1] == "kimi-k3"
    assert record.command[idx + 1] != "claude-sonnet-5"


def test_launch_claude_worker_empty_worker_model_falls_back_to_claude_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Model-pin safety (the 2026-07-22 ambient-model outage class): an empty
    ``worker.model`` must still resolve to ``_DEFAULT_CLAUDE_MODEL`` at the
    single command-construction point in ``launch_claude_worker`` -- never an
    empty ``--model`` value, which would fall through to ambient CLI/global
    state instead of a config-controlled default.

    Constructs ``OrchestratorConfig`` directly (not via ``load_config``) so
    an empty ``worker.model`` reaches ``launch_claude_worker`` unmodified --
    this isolates the launch-boundary safety net from any defaulting
    ``load_config`` itself might apply upstream, so the assertion holds
    regardless of how (or whether) ``load_config`` fills in a default before
    the config ever reaches this function.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    config = OrchestratorConfig(
        worker=WorkerRoleConfig(harness="claude-code", model=""),
    )

    record = launch_claude_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        config=config,
    )

    assert record.command.count("--model") == 1
    idx = record.command.index("--model")
    assert record.command[idx + 1] == claude_code._DEFAULT_CLAUDE_MODEL


def test_launch_claude_worker_empty_reviewer_model_override_falls_back_to_claude_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same model-pin safety net as
    ``test_launch_claude_worker_empty_worker_model_falls_back_to_claude_default``,
    but for the reviewer path: an explicitly empty ``model_override`` (as
    would result from a raw ``config.reviewer.model`` of ``""`` passed
    through directly, without the ``or None`` guard the real
    ``dispatch_reviews`` call site applies) must still resolve to
    ``_DEFAULT_CLAUDE_MODEL`` rather than pinning ``--model`` to an empty
    string.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    config = OrchestratorConfig(
        worker=WorkerRoleConfig(model="claude-should-not-be-read-either"),
        reviewer=ReviewerRoleConfig(model=""),
    )

    record = launch_claude_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        config=config,
        model_override=config.reviewer.model,
    )

    assert record.command.count("--model") == 1
    idx = record.command.index("--model")
    assert record.command[idx + 1] == claude_code._DEFAULT_CLAUDE_MODEL
    assert record.command[idx + 1] != "claude-should-not-be-read-either"


def test_launch_claude_worker_worker_never_uses_review_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A worker (non-review) launch must never pick up reviewer.effort, even
    when it's set and differs from claude_code.effort."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(effort="low"),
        reviewer=ReviewerRoleConfig(effort="high"),
    )

    record = launch_claude_worker(
        43,
        "agent/issue-43-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        config=config,
    )

    assert record.command.count("--effort") == 1
    idx = record.command.index("--effort")
    assert record.command[idx + 1] == "low"


def test_launch_claude_worker_worker_defaults_to_accept_edits_permission_mode(
    tmp_path: Path,
) -> None:
    """Non-review (worker) launches keep the pre-existing acceptEdits default."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "sessions"

    record = launch_claude_worker(
        502,
        "agent/issue-502-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
    )

    assert "--permission-mode" in record.command
    mode_index = record.command.index("--permission-mode")
    assert record.command[mode_index + 1] == "acceptEdits"
