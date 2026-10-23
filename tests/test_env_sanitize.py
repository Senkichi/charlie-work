"""Tests for env_sanitize's issue #646 dispatch-parallelism cap and issue #649
shared-venv confinement.

Covers the box-wide-saturation fix in isolation: sanitize_env()'s safe
defaults for PYTEST_XDIST_AUTO_NUM_WORKERS/UV_NO_SYNC, the
resolve_pytest_cap()/resolve_uv_no_sync() precedence helpers, and the handling
of .venv reparse points (junctions/symlinks) so a worker cannot rewrite a
shared venv. Adapter-level integration coverage (does
launch_claude_worker/launch_devin_session actually persist the resolved values
onto the sidecar) lives in test_claude_code_adapter.py / test_devin_shell.py
alongside their existing sanitize_env merge-order tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work import claude_code
from charlie_work.config import (
    ApiProviderConfig,
    ApiWorkerConfig,
    OrchestratorConfig,
)
from charlie_work.env_sanitize import (
    DEFAULT_PYTEST_XDIST_AUTO_NUM_WORKERS,
    PYTEST_XDIST_AUTO_NUM_WORKERS_VAR,
    UV_NO_SYNC_VAR,
    resolve_pytest_cap,
    resolve_uv_no_sync,
    sanitize_env,
)
from charlie_work.worktree import WorktreeInfo, _create_junction_or_symlink, is_junction

# ---------------------------------------------------------------------------
# sanitize_env: safe-default injection
# ---------------------------------------------------------------------------


def test_sanitize_env_injects_default_pytest_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no ambient cap set, sanitize_env must inject the safe default."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv(PYTEST_XDIST_AUTO_NUM_WORKERS_VAR, raising=False)

    env = sanitize_env(worktree_path)

    assert env[PYTEST_XDIST_AUTO_NUM_WORKERS_VAR] == DEFAULT_PYTEST_XDIST_AUTO_NUM_WORKERS


def test_sanitize_env_respects_ambient_pytest_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator-exported ambient cap must survive — setdefault, not overwrite."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.setenv(PYTEST_XDIST_AUTO_NUM_WORKERS_VAR, "4")

    env = sanitize_env(worktree_path)

    assert env[PYTEST_XDIST_AUTO_NUM_WORKERS_VAR] == "4"


def test_sanitize_env_sets_uv_no_sync_when_venv_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree with its own real .venv must get UV_NO_SYNC=1 by default.

    UV_NO_SYNC is only ever paired with a confirmed, real local .venv — never
    with a reparse point that points into a shared venv.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".venv").mkdir()
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    env = sanitize_env(worktree_path)

    assert env[UV_NO_SYNC_VAR] == "1"


def test_sanitize_env_omits_uv_no_sync_when_no_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree without its own .venv must NOT get UV_NO_SYNC forced on it.

    UV_NO_SYNC is only ever paired with a confirmed .venv (see sanitize_env's
    docstring) — a worktree that legitimately manages its own venv must still
    be able to run `uv sync` normally.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    env = sanitize_env(worktree_path)

    assert UV_NO_SYNC_VAR not in env


def test_sanitize_env_respects_ambient_uv_no_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator-exported ambient UV_NO_SYNC must survive even with a .venv present."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".venv").mkdir()
    monkeypatch.setenv(UV_NO_SYNC_VAR, "0")

    env = sanitize_env(worktree_path)

    assert env[UV_NO_SYNC_VAR] == "0"


# ---------------------------------------------------------------------------
# resolve_pytest_cap: 3-way precedence (config > env > default)
# ---------------------------------------------------------------------------


def test_sanitize_env_drops_inherited_pwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An inherited PWD names the orchestrator's directory; opencode trusts it
    over the real cwd, so a stale value ran a worker's git in the main checkout.
    The sanitized env must carry neither PWD nor OLDPWD."""
    monkeypatch.setenv("PWD", str(tmp_path / "orchestrator-checkout"))
    monkeypatch.setenv("OLDPWD", str(tmp_path / "elsewhere"))

    env = sanitize_env(tmp_path)

    assert "PWD" not in env
    assert "OLDPWD" not in env


def test_resolve_pytest_cap_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv(PYTEST_XDIST_AUTO_NUM_WORKERS_VAR, raising=False)

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_pytest_cap(sanitized, None)

    assert (value, source) == (DEFAULT_PYTEST_XDIST_AUTO_NUM_WORKERS, "default")


def test_resolve_pytest_cap_ambient_env_wins_over_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.setenv(PYTEST_XDIST_AUTO_NUM_WORKERS_VAR, "6")

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_pytest_cap(sanitized, None)

    assert (value, source) == ("6", "env")


def test_resolve_pytest_cap_config_wins_over_ambient_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config-level worker_env override must beat both the ambient env and the default.

    This is the merge-order invariant documented in config.py: worker_env is
    merged AFTER sanitize_env's output, so an explicit operator override wins.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.setenv(PYTEST_XDIST_AUTO_NUM_WORKERS_VAR, "6")

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_pytest_cap(sanitized, {PYTEST_XDIST_AUTO_NUM_WORKERS_VAR: "1"})

    assert (value, source) == ("1", "config")


# ---------------------------------------------------------------------------
# resolve_uv_no_sync: 3-way precedence + the no-venv case
# ---------------------------------------------------------------------------


def test_resolve_uv_no_sync_no_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, None)

    assert (value, source) == (None, "no-venv")


def test_resolve_uv_no_sync_default_with_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".venv").mkdir()
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, None)

    assert (value, source) == ("1", "default")


def test_resolve_uv_no_sync_ambient_env_wins_over_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".venv").mkdir()
    monkeypatch.setenv(UV_NO_SYNC_VAR, "0")

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, None)

    assert (value, source) == ("0", "env")


def test_resolve_uv_no_sync_config_wins_over_ambient_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".venv").mkdir()
    monkeypatch.setenv(UV_NO_SYNC_VAR, "0")

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, {UV_NO_SYNC_VAR: "1"})

    assert (value, source) == ("1", "config")


def test_resolve_uv_no_sync_ambient_env_survives_without_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: an ambient UV_NO_SYNC must be reported even with no .venv.

    sanitize_env() never *pops* an ambient UV_NO_SYNC -- it only skips the
    setdefault when there's no .venv -- so the value still reaches the child
    process regardless of .venv presence. A prior version of this function
    checked the no-venv case before the ambient-env case and incorrectly
    reported (None, "no-venv") here, understating what the subprocess
    actually inherited.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.setenv(UV_NO_SYNC_VAR, "1")

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, None)

    assert (value, source) == ("1", "env")


def test_resolve_uv_no_sync_config_can_force_it_without_a_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit worker_env override must win even when no .venv is present."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, {UV_NO_SYNC_VAR: "1"})

    assert (value, source) == ("1", "config")


def test_resolve_uv_no_sync_reparse_point_reports_no_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .venv that is a reparse point must not be reported as an owned venv."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".git").write_text("gitdir: /fake/path", encoding="utf-8")

    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    _create_junction_or_symlink(worktree_path / ".venv", shared_venv)
    assert is_junction(worktree_path / ".venv")

    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)
    sanitized = sanitize_env(worktree_path)
    value, source = resolve_uv_no_sync(worktree_path, sanitized, None)

    assert (value, source) == (None, "no-venv")


# ---------------------------------------------------------------------------
# Issue #649: a .venv reparse point must not masquerade as an owned local venv
# ---------------------------------------------------------------------------


def test_sanitize_env_sets_virtual_env_and_uv_no_sync_for_owned_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree with a real local .venv must get VIRTUAL_ENV and UV_NO_SYNC=1."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    worktree_venv = worktree_path / ".venv"
    worktree_venv.mkdir()
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    env = sanitize_env(worktree_path)

    assert env.get("VIRTUAL_ENV") == str(worktree_venv)
    assert "UV_PROJECT_ENVIRONMENT" not in env, (
        "UV_PROJECT_ENVIRONMENT is uv's default .venv path; setting it is a no-op"
    )
    assert env[UV_NO_SYNC_VAR] == "1"


def test_sanitize_env_omits_uv_project_environment_when_no_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree without its own .venv must NOT get UV_PROJECT_ENVIRONMENT set.

    With no target venv there is nothing to pin, and any ambient value was the
    orchestrator's leak (popped unconditionally on entry). The key must be
    absent so uv is free to manage its own venv.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)

    env = sanitize_env(worktree_path)

    assert "UV_PROJECT_ENVIRONMENT" not in env


def test_sanitize_env_orchestrator_uv_project_environment_never_survives_unchanged_with_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator's ambient UV_PROJECT_ENVIRONMENT must never reach the
    worker unchanged — pins issue #117's leak guarantee, which this change must
    not erode.

    When the worktree has a real .venv, the orchestrator's value is popped and
    not re-set. uv's default project environment is .venv in the project root,
    which is the worktree's own venv, so the leaked value is replaced by
    absence rather than passed through. When it has no .venv, the value is
    simply absent.
    """
    orchestrator_venv = "/orchestrator/.venv"
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", orchestrator_venv)

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    worktree_venv = worktree_path / ".venv"
    worktree_venv.mkdir()

    env = sanitize_env(worktree_path)

    assert "UV_PROJECT_ENVIRONMENT" not in env, (
        "UV_PROJECT_ENVIRONMENT must be dropped, not pinned to the worktree venv"
    )
    assert env.get("VIRTUAL_ENV") == str(worktree_venv)
    assert env.get("VIRTUAL_ENV") != orchestrator_venv


def test_sanitize_env_orchestrator_uv_project_environment_never_survives_unchanged_without_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator's ambient UV_PROJECT_ENVIRONMENT must be dropped when
    the worktree has no .venv (issue #117 leak guard, undisturbed by #649).
    """
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    env = sanitize_env(worktree_path)

    assert "UV_PROJECT_ENVIRONMENT" not in env


def test_sanitize_env_unlinks_junctioned_shared_venv_in_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .venv reparse point in a worktree must be unlinked, not pinned.

    Setting UV_PROJECT_ENVIRONMENT to the .venv path is a no-op over uv's
    default project-environment lookup, and a junctioned .venv still resolves
    to the shared target. sanitize_env must detect the reparse point and remove
    it so uv is forced to create a real local .venv instead.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / ".git").write_text("gitdir: /fake/path", encoding="utf-8")

    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    marker = shared_venv / "site-packages-marker.txt"
    marker.write_text("shared\n", encoding="utf-8")

    _create_junction_or_symlink(worktree_path / ".venv", shared_venv)
    assert is_junction(worktree_path / ".venv")

    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")
    monkeypatch.delenv(UV_NO_SYNC_VAR, raising=False)

    env = sanitize_env(worktree_path)

    # The reparse point is gone; the shared venv is untouched.
    assert not is_junction(worktree_path / ".venv")
    assert not (worktree_path / ".venv").exists()
    assert marker.read_text(encoding="utf-8") == "shared\n"

    # No venv hints are passed to the worker.
    assert "VIRTUAL_ENV" not in env
    assert "UV_PROJECT_ENVIRONMENT" not in env
    assert UV_NO_SYNC_VAR not in env


def test_sanitize_env_does_not_unlink_repo_root_venv_junction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sanitize_env must not unlink a .venv reparse point at the repo root.

    Repo roots have a .git directory, not a .git file. The sanitizer should
    leave the venv alone there and simply not set any venv variables.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    marker = shared_venv / "site-packages-marker.txt"
    marker.write_text("shared\n", encoding="utf-8")

    _create_junction_or_symlink(repo_root / ".venv", shared_venv)
    assert is_junction(repo_root / ".venv")

    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    env = sanitize_env(repo_root)

    # Junction at repo root is not touched.
    assert is_junction(repo_root / ".venv")
    assert marker.read_text(encoding="utf-8") == "shared\n"

    assert "VIRTUAL_ENV" not in env
    assert "UV_PROJECT_ENVIRONMENT" not in env


def _api_orchestrator_config(
    *,
    provider_name: str = "kimi-k3",
    api_key_env: str = "MOONSHOT_API_KEY",
) -> OrchestratorConfig:
    """Build an OrchestratorConfig directly (the pattern
    test_charlie_work_dispatch_adapters.py uses) with one enabled api provider
    whose key env var is the modern default test name."""
    provider = ApiProviderConfig(
        base_url="https://api.moonshot.ai/anthropic",
        api_key_env=api_key_env,
        model="kimi-k3",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cached_input_usd_per_mtok=0.30,
    )
    api_cfg = ApiWorkerConfig(
        enabled=True,
        provider=provider_name,
        max_concurrent_sessions=1,
        providers={provider_name: provider},
    )
    return OrchestratorConfig(api_worker=api_cfg)


def test_sanitize_env_pops_configured_provider_api_key_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #2795: the provider-key variable names configured under
    ``api_worker.providers`` must be popped from the sanitized worker env.

    The supervisor carries the provider key in its own env (the api adapter
    resolves it there), and every claude_code/devin_shell/devin_review_resume/
    rescue_review launch passes through sanitize_env — so a key left in the
    sanitized base reaches those worker sessions ambiently. With
    ``config=None`` nothing extra is stripped, preserving every caller that
    has no config in scope.
    """
    config = _api_orchestrator_config()
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-configured-provider-key")

    env_without_config = sanitize_env(tmp_path)
    assert env_without_config["MOONSHOT_API_KEY"] == "sk-configured-provider-key"

    env = sanitize_env(tmp_path, config=config)
    assert "MOONSHOT_API_KEY" not in env, (
        "a configured apiKeyName under api_worker.providers must be popped from "
        "the sanitized worker env (issue #2795)"
    )


def test_sanitize_env_strips_every_providers_env_name_not_only_the_active_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All configured providers' key env names are stripped, not just the
    active provider's: the supervisor's env carries whatever the operator
    stored, and a worker must not inherit any of them."""
    provider = ApiProviderConfig(
        base_url="https://api.example.com/anthropic",
        api_key_env="EXAMPLE_API_KEY",
        model="example-m",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
    )
    api_cfg = ApiWorkerConfig(
        enabled=True,
        provider="shared",
        providers={
            "shared": provider,
            "alt": ApiProviderConfig(
                base_url="https://api.example.com/anthropic",
                api_key_env="ALT_EXAMPLE_API_KEY",
                model="example-m",
                input_usd_per_mtok=3.0,
                output_usd_per_mtok=15.0,
            ),
        },
    )
    config = OrchestratorConfig(api_worker=api_cfg)
    monkeypatch.setenv("EXAMPLE_API_KEY", "sk-active")
    monkeypatch.setenv("ALT_EXAMPLE_API_KEY", "sk-alt")

    env = sanitize_env(tmp_path, config=config)

    assert "EXAMPLE_API_KEY" not in env
    assert "ALT_EXAMPLE_API_KEY" not in env


def test_api_worker_launch_strips_provider_key_but_routes_auth_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unset: the api adapter keeps working while the ambient key is stripped.

    A sanitized launch env (sanitize_env(..., config=cfg) pops MOONSHOT_API_KEY)
    must not lose the api path's routing: launch_api_worker merges provider_env
    (ANTHROPIC_AUTH_TOKEN etc.) AFTER sanitization, so the child still receives
    ANTHROPIC_AUTH_TOKEN even though the provider-key variable name itself is
    gone from the worker session's env.
    """
    import sys
    import textwrap

    from _worker_marker_wait import read_worker_marker

    from charlie_work.api_worker import launch_api_worker

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktrees_dir = tmp_path / "worktrees"
    worktree_path = worktrees_dir / "worktree"
    worktree_path.mkdir(parents=True)
    sessions_dir = tmp_path / "sessions"

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
        return WorktreeInfo(path=worktree_path, branch=branch, venv_junction=None)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-probe.txt").write_text(
                "|".join(
                    [
                        os.environ.get("ANTHROPIC_AUTH_TOKEN", "<unset>"),
                        os.environ.get("MOONSHOT_API_KEY", "<unset>"),
                    ]
                ),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    config = _api_orchestrator_config()
    record = launch_api_worker(
        100,
        "agent/issue-100-sanitize",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=config.api_worker,
        command_template=(sys.executable, str(script_path)),
        config=config,
    )

    assert record.ok, record.error
    probe_path = Path(record.worktree_path) / "env-probe.txt"
    auth_token, provider_key = read_worker_marker(
        probe_path,
        expected="sk-test-key-value-1234|<unset>",
        reason=(
            "issue #2795: the api child receives ANTHROPIC_AUTH_TOKEN (provider "
            "env merged after sanitization) but NOT the ambient provider key "
            "variable itself"
        ),
    ).split("|")
    assert auth_token == "sk-test-key-value-1234"
    assert provider_key == "<unset>"
