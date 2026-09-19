"""``bootstrap_command`` / ``CommandContext`` config-repo misroute guard.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _cli_fixtures import _FakeGitHub
from charlie_work import cli
from charlie_work.config import (
    ConfigError,
    OrchestratorConfig,
)
from charlie_work.paths import runtime_paths


def _fake_repo(root: Path) -> Path:
    """A directory git's fallback resolution will treat as a work-tree root.

    `git rev-parse` fails inside it (no HEAD), which drives `find_repo_root`
    into its documented `.git`-walking fallback — deterministic and offline.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").mkdir(exist_ok=True)
    return root


def test_config_from_another_repo_is_refused(tmp_path: Path) -> None:
    """Issue #895: --config selects the config, never the state.

    The real incident: `charlie --config <job-cannon> tripwire ack 1392` run from
    a charlie-work cwd wrote job-cannon's ack into charlie-work's state, exit 0,
    while job-cannon's finding kept pinning ok=False.
    """
    repo_a = _fake_repo(tmp_path / "charlie-work")
    repo_b = _fake_repo(tmp_path / "job-cannon")
    foreign_config = repo_b / "orchestrator.config.yaml"
    foreign_config.write_text("runtime:\n  state_dir: .var/charlie-work\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        cli._assert_config_repo_matches(foreign_config, repo_a)

    message = str(excinfo.value)
    assert str(repo_b) in message
    assert str(repo_a) in message
    # The error must carry the corrective invocation, not just the diagnosis.
    assert "--repo" in message


def test_config_from_the_same_repo_is_allowed(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path / "charlie-work")
    own_config = repo / "orchestrator.config.yaml"
    own_config.write_text("runtime:\n  state_dir: .var/charlie-work\n", encoding="utf-8")

    cli._assert_config_repo_matches(own_config, repo)  # must not raise


def test_config_outside_any_git_repo_is_allowed(tmp_path: Path) -> None:
    """A shared/layered config legitimately lives outside the repo it configures.

    The gate fires only when the config provably belongs to a *different* work
    tree — otherwise this would break exactly the deployment shape it must not
    touch.
    """
    repo = _fake_repo(tmp_path / "charlie-work")
    shared = tmp_path / "shared-config"
    shared.mkdir()
    loose_config = shared / "orchestrator.config.yaml"
    loose_config.write_text("runtime:\n  state_dir: .var/charlie-work\n", encoding="utf-8")

    cli._assert_config_repo_matches(loose_config, repo)  # must not raise


def test_no_config_flag_is_allowed(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path / "charlie-work")
    cli._assert_config_repo_matches(None, repo)  # must not raise


# --------------------------------------------------------------------------
# bootstrap_command / CommandContext (issue #705)
# --------------------------------------------------------------------------


def test_bootstrap_command_returns_frozen_context_with_all_four_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """bootstrap_command bundles repo_root, config, paths, gh in one call (#705).

    The four-call sequence (find_repo_root -> load_layered_config ->
    runtime_paths -> GitHub) was previously duplicated across every command
    handler.  This test pins that the single shared helper returns a frozen
    dataclass whose fields are mutually consistent: gh.repo_root matches
    ctx.repo_root, paths is derived from the same config's state_dir, and the
    context is immutable.
    """
    repo = _fake_repo(tmp_path / "charlie-work")
    config = OrchestratorConfig()

    monkeypatch.setattr(cli, "find_repo_root", lambda repo_arg, explicit=False, **kw: repo)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: config)
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)

    args = cli.build_parser().parse_args(["--repo", str(repo), "roll-call"])
    ctx = cli.bootstrap_command(args)

    assert isinstance(ctx, cli.CommandContext)
    assert ctx.repo_root == repo
    assert ctx.config is config
    # paths must be derived from the same config's state_dir against repo_root
    expected_paths = runtime_paths(repo, config.runtime.state_dir)
    assert ctx.paths == expected_paths
    # gh must be constructed with the same repo_root and config.runtime
    assert isinstance(ctx.gh, _FakeGitHub)


def test_command_context_is_frozen(tmp_path: Path) -> None:
    """CommandContext must be a frozen dataclass (CLAUDE.md invariant)."""
    import dataclasses

    repo = _fake_repo(tmp_path / "charlie-work")
    ctx = cli.CommandContext(
        repo_root=repo,
        config=OrchestratorConfig(),
        paths=runtime_paths(repo, OrchestratorConfig().runtime.state_dir),
        gh=_FakeGitHub(),
    )
    assert dataclasses.is_dataclass(ctx)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.repo_root = tmp_path  # type: ignore[misc]


def test_bootstrap_command_includes_config_repo_misroute_guard(
    tmp_path: Path,
) -> None:
    """bootstrap_command inherits _assert_config_repo_matches (issue #895).

    Previously only build_app checked that --config does not point into a
    different repo's work tree.  Centralizing the bootstrap means every command
    handler now gets this guard, not just the ones that happened to call
    build_app.  This test verifies the guard fires through bootstrap_command.
    """
    repo_a = _fake_repo(tmp_path / "charlie-work")
    repo_b = _fake_repo(tmp_path / "job-cannon")
    foreign_config = repo_b / "orchestrator.config.yaml"
    foreign_config.write_text("runtime:\n  state_dir: .var/charlie-work\n", encoding="utf-8")

    args = cli.build_parser().parse_args(
        ["--repo", str(repo_a), "--config", str(foreign_config), "roll-call"]
    )
    with pytest.raises(ConfigError) as excinfo:
        cli.bootstrap_command(args)
    assert str(repo_b) in str(excinfo.value)
    assert "--repo" in str(excinfo.value)
