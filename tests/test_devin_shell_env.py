"""Environment-sanitization tests for the devin-shell launch path.

Split out of ``tests/test_devin_shell.py`` (issue #1542, Track-1 pilot):
``sanitize_env``'s VIRTUAL_ENV / GitHub-token / GH_CONFIG_DIR handling
and the launch-side proof that no orchestrator credential channel
reaches the worker (operator-scoped GH token pass-through included).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from _devin_shell_fixtures import _install_fake_create_worktree

from charlie_work.devin_shell import launch_devin_session
from charlie_work.env_sanitize import sanitize_env


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


def test_sanitize_env_drops_github_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """sanitize_env must drop GH_TOKEN and GITHUB_TOKEN so workers do not inherit the orchestrator's admin token (issue #502)."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    monkeypatch.setenv("GH_TOKEN", "admin-orchestrator-token")
    monkeypatch.setenv("GITHUB_TOKEN", "admin-orchestrator-token")

    env = sanitize_env(worktree_path)

    assert "GH_TOKEN" not in env, "GH_TOKEN must be dropped"
    assert "GITHUB_TOKEN" not in env, "GITHUB_TOKEN must be dropped"


def test_sanitize_env_drops_enterprise_github_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sanitize_env must drop GH_ENTERPRISE_TOKEN/GITHUB_ENTERPRISE_TOKEN so a worker cannot fall back on a GHES credential to merge (issue #502)."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    monkeypatch.setenv("GH_ENTERPRISE_TOKEN", "ghes-admin-token")
    monkeypatch.setenv("GITHUB_ENTERPRISE_TOKEN", "ghes-admin-token")

    env = sanitize_env(worktree_path)

    assert "GH_ENTERPRISE_TOKEN" not in env, "GH_ENTERPRISE_TOKEN must be dropped"
    assert "GITHUB_ENTERPRISE_TOKEN" not in env, "GITHUB_ENTERPRISE_TOKEN must be dropped"


def test_sanitize_env_isolates_gh_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sanitize_env must point GH_CONFIG_DIR at a worktree-local empty directory so gh cannot use the orchestrator's stored credentials (issue #502)."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    # Simulate the orchestrator having a live gh login.
    orchestrator_config_dir = tmp_path / "orchestrator-gh-config"
    orchestrator_config_dir.mkdir()
    monkeypatch.setenv("GH_CONFIG_DIR", str(orchestrator_config_dir))

    env = sanitize_env(worktree_path)

    expected_gh_config_dir = worktree_path / ".var" / "gh-config"
    assert env.get("GH_CONFIG_DIR") == str(expected_gh_config_dir), (
        f"GH_CONFIG_DIR must be isolated to the worktree, got {env.get('GH_CONFIG_DIR')!r}"
    )
    assert expected_gh_config_dir.is_dir(), "isolated gh config directory must be created"
    # The orchestrator's stored credential directory must not leak.
    assert env.get("GH_CONFIG_DIR") != str(orchestrator_config_dir)


def test_sanitize_env_leaves_no_channel_for_orchestrator_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credential isolation must not be a no-op: no orchestrator credential
    material may reach the worker through ANY environment channel (issue #502).

    The sibling tests above enumerate the four token variable names and the
    config-dir override. That is a checklist, and a checklist cannot show the
    control has teeth — a fifth channel, or a token smuggled in an unrelated
    variable, passes every one of them. This test asserts the invariant
    instead: given an orchestrator whose ``gh`` credential store is genuinely
    populated, the secret string must appear in no value of the sanitized
    environment, and the directory ``gh`` is pointed at must contain no
    ``hosts.yml`` for it to read. Those are the two mechanisms by which ``gh``
    resolves an identity, so together they are what makes ``gh pr merge``
    impossible for a worker rather than merely inconvenient.

    Deliberately hermetic: it does not shell out to ``gh``. A live
    ``gh auth status`` probe would depend on the runner having ``gh`` installed
    and on CI not injecting its own ``GH_TOKEN`` — it would pass here and be
    meaningless or flaky there.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    secret = "ghp_orchestrator_admin_secret_do_not_leak"

    # A real, populated gh credential store for the orchestrator.
    orchestrator_config_dir = tmp_path / "orchestrator-gh-config"
    orchestrator_config_dir.mkdir()
    (orchestrator_config_dir / "hosts.yml").write_text(
        f"github.com:\n    oauth_token: {secret}\n    user: orchestrator\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GH_CONFIG_DIR", str(orchestrator_config_dir))
    monkeypatch.setenv("GH_TOKEN", secret)
    monkeypatch.setenv("GITHUB_TOKEN", secret)
    monkeypatch.setenv("GH_ENTERPRISE_TOKEN", secret)
    monkeypatch.setenv("GITHUB_ENTERPRISE_TOKEN", secret)

    env = sanitize_env(worktree_path)

    # Channel 1: environment. The secret must survive in no value, whatever
    # the variable happens to be called.
    leaked = sorted(name for name, value in env.items() if secret in str(value))
    assert leaked == [], f"orchestrator credential leaked via env var(s): {leaked}"

    # Channel 2: gh's stored-credential file. The worker's config dir must
    # exist (so gh does not fall back to the platform default) and must hold
    # no hosts.yml for gh to authenticate from.
    worker_config_dir = Path(env["GH_CONFIG_DIR"])
    assert worker_config_dir.is_dir()
    assert not (worker_config_dir / "hosts.yml").exists(), (
        "worker gh config dir must not contain a credential store"
    )
    assert list(worker_config_dir.iterdir()) == [], (
        f"worker gh config dir must be empty, found {list(worker_config_dir.iterdir())}"
    )
    # And it must not simply be the orchestrator's directory under a new name.
    assert worker_config_dir.resolve() != orchestrator_config_dir.resolve()


def test_launch_sanitizes_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """launch_devin_session must sanitize the environment before spawning the worker."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")

    # Script that writes VIRTUAL_ENV to stdout so we can verify it's sanitized
    env_script = tmp_path / "echo_env.py"
    env_script.write_text(
        "import os, sys\nsys.stdout.write(os.environ.get('VIRTUAL_ENV', 'UNSET') + '\\n')\nsys.stdout.flush()\n",
        encoding="utf-8",
    )

    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Set parent VIRTUAL_ENV (simulating orchestrator leak)
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")

    record = launch_devin_session(
        55,
        "agent/issue-55-test",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(env_script)),
    )

    assert record.error is None
    assert record.pid is not None

    # Give the subprocess a moment, then verify it didn't inherit VIRTUAL_ENV
    deadline = time.time() + 5
    while time.time() < deadline:
        log_text = Path(record.log_path).read_text(encoding="utf-8")
        if log_text.strip():
            break
        time.sleep(0.05)
    log_text = Path(record.log_path).read_text(encoding="utf-8").strip()

    assert log_text == "UNSET", (
        f"Worker inherited VIRTUAL_ENV={log_text!r}, expected UNSET (sanitized)"
    )


def test_launch_passes_operator_scoped_gh_token_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator-supplied scoped GH_TOKEN must reach the worker (issue #502).

    Stripping the orchestrator's token is only half the contract. The documented
    escape hatch is ``devin.worker_env``/``claude_code.worker_env``, which is
    merged AFTER sanitize_env precisely so a scoped PAT wins. If that merge order
    ever inverted, sanitization would silently eat the operator's token and every
    worker would lose ``gh`` entirely — an availability failure that the
    token-stripping tests above would still report as a pass. This asserts the
    happy path end to end: the spawned process really does see the scoped value.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")

    env_script = tmp_path / "echo_gh_token.py"
    env_script.write_text(
        "import os, sys\n"
        "sys.stdout.write(os.environ.get('GH_TOKEN', 'UNSET') + '\\n')\n"
        "sys.stdout.flush()\n",
        encoding="utf-8",
    )

    _install_fake_create_worktree(monkeypatch, tmp_path)

    # The orchestrator's own admin token is present and must NOT be what wins.
    monkeypatch.setenv("GH_TOKEN", "admin-orchestrator-token")

    record = launch_devin_session(
        56,
        "agent/issue-56-test",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(env_script)),
        worker_env={"GH_TOKEN": "scoped-worker-pat"},
    )

    assert record.error is None
    assert record.pid is not None

    deadline = time.time() + 5
    while time.time() < deadline:
        log_text = Path(record.log_path).read_text(encoding="utf-8")
        if log_text.strip():
            break
        time.sleep(0.05)
    log_text = Path(record.log_path).read_text(encoding="utf-8").strip()

    assert log_text == "scoped-worker-pat", (
        f"worker_env GH_TOKEN must survive sanitization, worker saw {log_text!r}"
    )
