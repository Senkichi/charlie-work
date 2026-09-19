"""Environment sanitization and ``worker_env`` injection: VIRTUAL_ENV
drop/set, override precedence, and the pytest-xdist cap passthrough.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
import pytest

from _claude_adapter_fixtures import (
    _fake_claude_script,
    _install_fake_create_worktree,
)
from _worker_marker_wait import read_worker_marker

from charlie_work import claude_code
from charlie_work.claude_code import launch_claude_worker
from charlie_work.env_sanitize import sanitize_env
from charlie_work.worktree import WorktreeInfo


def test_launch_claude_worker_injects_worker_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    # An inherited var must survive the merge; the injected var must appear.
    monkeypatch.setenv("CHARLIE_INHERITED", "inherited-value")

    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-probe.txt").write_text(
                os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS", "<unset>")
                + "|"
                + os.environ.get("CHARLIE_INHERITED", "<unset>"),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        99,
        "agent/issue-99-env",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
        env={"PYTEST_XDIST_AUTO_NUM_WORKERS": "2"},
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-probe.txt"
    # Injected var present AND orchestrator env inherited (merge, not replace).
    read_worker_marker(probe_path, expected="2|inherited-value")


def test_launch_claude_worker_worker_env_overrides_sanitize_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """worker_env overrides sanitize_env: operator-provided VIRTUAL_ENV wins.

    This is a mutation gate for the merge order in launch_claude_worker:
    the current order is {**sanitize_env(...), **worker_env}, so worker_env
    clobbers sanitized keys. If the order is inverted (worker_env first,
    sanitize_env clobbering it), this test fails.

    The fixture uses with_venv=True so sanitize_env actively SETS VIRTUAL_ENV
    (instead of POP-ing it), making the merge order sensitive.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path, with_venv=True)

    # Set a VIRTUAL_ENV in the orchestrator's environment (which sanitize_env
    # would normally strip). Then provide an explicit override via worker_env.
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/venv")

    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-probe.txt").write_text(
                os.environ.get("VIRTUAL_ENV", "<unset>"),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        140,
        "agent/issue-140-env-override",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
        env={"VIRTUAL_ENV": "/custom/override/venv"},
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-probe.txt"
    # worker_env VIRTUAL_ENV override wins over sanitize_env's stripping
    read_worker_marker(probe_path, expected="/custom/override/venv")


def test_launch_claude_worker_worker_env_pytest_cap_override_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #646: an explicit worker_env cap must win over the ambient env and default."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path, with_venv=True)
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "6")

    record = launch_claude_worker(
        142,
        "agent/issue-142-cap-override",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        env={"PYTEST_XDIST_AUTO_NUM_WORKERS": "1", "UV_NO_SYNC": "0"},
    )

    assert record.ok
    assert record.xdist_cap == "1"
    assert record.uv_no_sync == "0"


# ---------------------------------------------------------------------------
# Regression: VIRTUAL_ENV sanitization
# ---------------------------------------------------------------------------


def test_sanitize_env_drops_virtual_env_when_no_worktree_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When worktree has no .venv, VIRTUAL_ENV and UV_PROJECT_ENVIRONMENT must be dropped."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    # Set parent env variables (simulating orchestrator leak)
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    env = sanitize_env(worktree_path)

    assert "VIRTUAL_ENV" not in env, "VIRTUAL_ENV must be dropped when worktree has no .venv"
    assert "UV_PROJECT_ENVIRONMENT" not in env, (
        "UV_PROJECT_ENVIRONMENT must be dropped when worktree has no .venv"
    )


def test_sanitize_env_sets_worktree_venv_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When worktree has a real .venv, VIRTUAL_ENV must be set and UV_PROJECT_ENVIRONMENT dropped."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    worktree_venv = worktree_path / ".venv"
    worktree_venv.mkdir()

    # Set parent env variables (simulating orchestrator leak)
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    env = sanitize_env(worktree_path)

    assert env.get("VIRTUAL_ENV") == str(worktree_venv), (
        "VIRTUAL_ENV must be set to worktree .venv"
    )
    assert "UV_PROJECT_ENVIRONMENT" not in env, (
        "UV_PROJECT_ENVIRONMENT must be dropped; uv's default is the same .venv path (issue #649)"
    )


def test_sanitize_env_preserves_other_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Other environment variables must be preserved."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/home/user")
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")

    env = sanitize_env(worktree_path)

    assert env.get("PATH") == "/usr/bin:/bin", "PATH must be preserved"
    assert env.get("HOME") == "/home/user", "HOME must be preserved"
    assert "VIRTUAL_ENV" not in env, "VIRTUAL_ENV must be dropped"


def test_launch_sanitizes_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """launch_claude_worker must sanitize the environment before spawning the worker (stdin-fed path)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Seed parent env with leak variables
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    # Script that writes the actual env it received to a file
    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-received.txt").write_text(
                str(os.environ.get("VIRTUAL_ENV", "<unset>"))
                + "|"
                + str(os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>")),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        117,
        "agent/issue-117-sanitize",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-received.txt"
    read_worker_marker(probe_path)


def test_launch_sanitizes_with_worktree_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When worktree has .venv, VIRTUAL_ENV must be set to that path."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    # Create a fake worktree with .venv
    worktree_path = tmp_path / "worktrees" / "agent-issue-117-venv"
    worktree_path.mkdir(parents=True)
    worktree_venv = worktree_path / ".venv"
    worktree_venv.mkdir()

    def fake_create_worktree(*args, **kwargs):
        return WorktreeInfo(path=worktree_path, branch="agent/issue-117-venv", venv_junction=None)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)

    # Seed parent env with leak variables
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    # Script that writes the actual env it received to a file
    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-received.txt").write_text(
                str(os.environ.get("VIRTUAL_ENV", "<unset>"))
                + "|"
                + str(os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>")),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        117,
        "agent/issue-117-venv",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-received.txt"
    read_worker_marker(
        probe_path,
        expected=f"{str(worktree_venv)}|<unset>",
        reason=(
            "VIRTUAL_ENV must be pinned to the worktree .venv; "
            "UV_PROJECT_ENVIRONMENT must be dropped (issue #649)"
        ),
    )


def test_launch_preserves_worker_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit worker_env VIRTUAL_ENV override must survive sanitization (no worktree .venv)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Seed parent env with leak variables
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    # Script that writes the actual env it received to a file
    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-received.txt").write_text(
                str(os.environ.get("VIRTUAL_ENV", "<unset>"))
                + "|"
                + str(os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>"))
                + "|"
                + str(os.environ.get("CUSTOM_VAR", "<unset>")),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        117,
        "agent/issue-117-override",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
        env={"VIRTUAL_ENV": "/custom/.venv", "CUSTOM_VAR": "custom-value"},
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-received.txt"
    read_worker_marker(
        probe_path,
        expected="/custom/.venv|<unset>|custom-value",
        reason="user-provided VIRTUAL_ENV must win, UV_PROJECT_ENVIRONMENT must be dropped",
    )


def test_launch_override_precedence_with_worktree_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User-provided VIRTUAL_ENV override must win over worktree .venv (merge order test)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    # Create a fake worktree with .venv
    worktree_path = tmp_path / "worktrees" / "agent-issue-117-override-venv"
    worktree_path.mkdir(parents=True)
    worktree_venv = worktree_path / ".venv"
    worktree_venv.mkdir()

    def fake_create_worktree(*args, **kwargs):
        return WorktreeInfo(
            path=worktree_path, branch="agent/issue-117-override-venv", venv_junction=None
        )

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)

    # Seed parent env with leak variables
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    # Script that writes the actual env it received to a file
    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-received.txt").write_text(
                str(os.environ.get("VIRTUAL_ENV", "<unset>"))
                + "|"
                + str(os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>"))
                + "|"
                + str(os.environ.get("CUSTOM_VAR", "<unset>")),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        117,
        "agent/issue-117-override-venv",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
        env={"VIRTUAL_ENV": "/custom/.venv", "CUSTOM_VAR": "custom-value"},
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-received.txt"
    read_worker_marker(
        probe_path,
        expected="/custom/.venv|<unset>|custom-value",
        reason=(
            "user-provided VIRTUAL_ENV must win over worktree .venv "
            "(merge order: sanitizer first, then user overrides); "
            "UV_PROJECT_ENVIRONMENT stays dropped since the override did not touch it (issue #649)"
        ),
    )


def test_launch_sanitizes_environment_with_prompt_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """launch_claude_worker must sanitize the environment before spawning the worker (argv path with {prompt_path})."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Seed parent env with leak variables
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    # Script that reads prompt from argv and writes the actual env it received to a file
    script_path = tmp_path / "env_probe_argv.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import sys
            import os
            from pathlib import Path

            # Read prompt from argv (the {prompt_path} placeholder)
            prompt_path = Path(sys.argv[1])
            prompt_content = prompt_path.read_text(encoding="utf-8")

            # Write the env we received
            Path("env-received.txt").write_text(
                str(os.environ.get("VIRTUAL_ENV", "<unset>"))
                + "|"
                + str(os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>")),
                encoding="utf-8",
            )
            print("ok")
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        117,
        "agent/issue-117-sanitize-argv",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path), "{prompt_path}"),
    )

    assert record.ok
    assert record.prompt_path in record.command

    worktree_path = Path(record.worktree_path)
    probe_path = worktree_path / "env-received.txt"
    read_worker_marker(
        probe_path,
        expected="<unset>|<unset>",
        reason="VIRTUAL_ENV and UV_PROJECT_ENVIRONMENT must be dropped (no worktree .venv)",
    )
