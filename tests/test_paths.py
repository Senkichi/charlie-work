"""Gate tests for A2 (centralize-charlie-work-config-templates.md): every
sentinel-style state-child config field (``devin.sessions_dir``,
``devin.session_manifest``, ``devin.session_results``,
``review_dispatch.reviews_dir``, ``notify.file_path``,
``claude_code.worktrees_dir``) must resolve through
:func:`charlie_work.paths.resolved_layout`, the single unification point that
replaced eight independent, error-prone re-implementations of the "empty
means derive from runtime.state_dir" check.

Covers the plan's three required gates:
  (i)   default config derives the same paths as the pre-A2 hardcoded literals
  (ii)  overriding runtime.state_dir moves every child together
  (ii-b) an explicit child override still wins over the state_dir-derived
        default -- the other half of (ii)'s assertion, per the plan's
        13a-6 gap: (ii) only proved "no override -> children follow
        state_dir"; it never proved an explicit override survives a
        simultaneous state_dir override instead of being silently
        re-derived under it.
  (iii) dispatch and worktree-clean resolve to the identical worktrees root,
        exercised against the two real production call sites (not two calls
        to resolved_layout(), which would only prove the function is
        deterministic, not that both call sites stay wired to it)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from charlie_work import cli
from charlie_work.config import (
    ClaudeCodeConfig,
    DevinConfig,
    NotifyConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
    RuntimeConfig,
)
from charlie_work.paths import resolved_layout, runtime_paths
from charlie_work.worktree import WorktreeCleanResult
from charlie_work.workflow import OrchestratorApp


class _NullGitHub:
    """Placeholder ``gh`` for OrchestratorApp construction.

    ``OrchestratorApp.__init__`` only stores ``gh`` on ``self``, and
    ``_adapter_settings()`` never touches it -- no methods needed here.
    """


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    return tmp_path


def test_resolved_layout_default_config_matches_historical_literals(tmp_path: Path) -> None:
    """(i) With no overrides, every resolved path must equal the exact
    literal strings config.py hardcoded before A2 (pulled from the pre-A2
    source, not re-derived via layout.*_default() -- asserting derived values
    against themselves would prove nothing about the derivation being right).
    """
    config = OrchestratorConfig()

    layout_view = resolved_layout(config, tmp_path)

    assert layout_view.sessions_dir == tmp_path / ".var/charlie-work/dispatches/sessions"
    assert (
        layout_view.session_manifest
        == tmp_path / ".var/charlie-work/dispatches/session-manifest.json"
    )
    assert (
        layout_view.session_results
        == tmp_path / ".var/charlie-work/dispatches/session-results.json"
    )
    assert layout_view.reviews_dir == tmp_path / ".var/charlie-work/dispatches/reviews"
    assert layout_view.worktrees == tmp_path / ".var/charlie-work/worktrees"
    assert layout_view.cross_family == tmp_path / ".var/charlie-work/cross-family"
    assert layout_view.notify.file_path == str(tmp_path / ".var/charlie-work/notify/digest.jsonl")


def test_resolved_layout_overridden_state_dir_moves_every_child(tmp_path: Path) -> None:
    """(ii) Overriding runtime.state_dir alone (no per-field overrides) must
    move every child path under the new root together -- the exact property
    whose absence produced #712 (each call site independently deriving its
    own root, some honouring the override and some not).
    """
    config = OrchestratorConfig(runtime=RuntimeConfig(state_dir="custom-state"))
    new_root = tmp_path / "custom-state"

    layout_view = resolved_layout(config, tmp_path)

    assert layout_view.sessions_dir == new_root / "dispatches" / "sessions"
    assert layout_view.session_manifest == new_root / "dispatches" / "session-manifest.json"
    assert layout_view.session_results == new_root / "dispatches" / "session-results.json"
    assert layout_view.reviews_dir == new_root / "dispatches" / "reviews"
    assert layout_view.worktrees == new_root / "worktrees"
    assert layout_view.cross_family == new_root / "cross-family"
    assert layout_view.notify.file_path == str(new_root / "notify" / "digest.jsonl")


def test_resolved_layout_explicit_child_override_wins_over_state_dir(tmp_path: Path) -> None:
    """(ii-b) An explicit per-field override must still win when runtime.state_dir
    is *also* overridden -- the other half of (ii)'s claim. (ii) only proves
    "no override -> children follow state_dir"; it never proves an explicit
    override survives rather than being silently re-derived under the new
    state_dir root. Per the plan's 13a-6 gap, this half had no named test.

    Every sentinel-style state-child field is set explicitly here, alongside
    an overridden state_dir, and each must resolve to its own explicit value
    -- NOT nested under the overridden state_dir. ``cross_family`` has no
    override sentinel of its own (config.py has no ``cross_family_dir``
    field), so it is asserted as a contrast: it must still derive from the
    overridden state_dir root, proving the state_dir override itself did
    take effect and the other fields' independence from it is not an
    artifact of a no-op override.
    """
    config = OrchestratorConfig(
        runtime=RuntimeConfig(state_dir="custom-state"),
        devin=DevinConfig(
            sessions_dir="explicit-sessions",
            session_manifest="explicit-manifest.json",
            session_results="explicit-results.json",
        ),
        review_dispatch=ReviewDispatchConfig(reviews_dir="explicit-reviews"),
        claude_code=ClaudeCodeConfig(worktrees_dir="explicit-worktrees"),
        notify=NotifyConfig(file_path="explicit-notify/digest.jsonl"),
    )

    layout_view = resolved_layout(config, tmp_path)

    assert layout_view.sessions_dir == tmp_path / "explicit-sessions"
    assert layout_view.session_manifest == tmp_path / "explicit-manifest.json"
    assert layout_view.session_results == tmp_path / "explicit-results.json"
    assert layout_view.reviews_dir == tmp_path / "explicit-reviews"
    assert layout_view.worktrees == tmp_path / "explicit-worktrees"
    assert layout_view.notify.file_path == str(tmp_path / "explicit-notify" / "digest.jsonl")
    # Contrast: no override sentinel exists for cross_family, so it must still
    # derive from the overridden state_dir root -- proving the state_dir
    # override is live, not that everything above happened to no-op.
    assert layout_view.cross_family == tmp_path / "custom-state" / "cross-family"


def _dispatch_worktrees_root(repo: Path, config: OrchestratorConfig) -> Path:
    """Resolve the worktrees root exactly as dispatch does in production:
    ``OrchestratorApp._adapter_settings().worktrees_dir`` (workflow.py)."""
    paths = runtime_paths(repo, config.runtime.state_dir)
    app = OrchestratorApp(repo, paths, config, _NullGitHub())
    return app._adapter_settings().worktrees_dir


def _clean_worktrees_root(
    repo: Path, config: OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Resolve the worktrees root exactly as ``charlie worktree-clean`` does
    in production: capture the second positional argument
    ``run_worktree_clean_command`` (cli.py) passes into ``clean_worktrees``."""
    captured: dict[str, Path] = {}

    def _fake_clean_worktrees(
        repo_root: Path,
        worktrees_dir: Path,
        state: Any,
        cfg: OrchestratorConfig,
        gh: Any,
        *,
        dry_run: bool = False,
    ) -> WorktreeCleanResult:
        captured["worktrees_dir"] = worktrees_dir
        return WorktreeCleanResult(ok=True, message="ok", data={})

    monkeypatch.setattr(cli, "GitHub", lambda *a, **k: _NullGitHub())
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: config)
    monkeypatch.setattr(cli, "clean_worktrees", _fake_clean_worktrees)

    args = argparse.Namespace(repo=repo, config=None, fleet_dir=None, dry_run=False)
    result = cli.run_worktree_clean_command(args)
    assert result.ok is True
    return captured["worktrees_dir"]


def test_dispatch_and_clean_worktrees_root_agree_state_dir_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(iii) axis 1 -- ``runtime.state_dir`` overridden, no explicit
    ``claude_code.worktrees_dir``. This is the exact #712 shape: pre-A2,
    dispatch fell back to the unconditional default tree while
    ``worktree-clean`` swept the configured tree."""
    repo = _make_repo(tmp_path)
    config = OrchestratorConfig(runtime=RuntimeConfig(state_dir="custom-state"))

    dispatch_root = _dispatch_worktrees_root(repo, config)
    clean_root = _clean_worktrees_root(repo, config, monkeypatch)

    assert dispatch_root == clean_root
    assert dispatch_root == repo / "custom-state" / "worktrees"


def test_dispatch_and_clean_worktrees_root_agree_explicit_worktrees_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(iii) axis 2 -- ``claude_code.worktrees_dir`` set, default state_dir.
    This is the reversed-trigger sibling of #712: dispatch honoured this
    override before A2, but ``worktree-clean`` (cli.py) passed
    ``layout.worktrees_dir(paths.root)`` unconditionally and never read it.
    Latent on this host (no live config sets the field), but the same root
    cause as axis 1 -- two call sites independently deciding where worktrees
    live -- so it gets the same regression gate."""
    repo = _make_repo(tmp_path)
    config = OrchestratorConfig(claude_code=ClaudeCodeConfig(worktrees_dir="alt-worktrees-root"))

    dispatch_root = _dispatch_worktrees_root(repo, config)
    clean_root = _clean_worktrees_root(repo, config, monkeypatch)

    assert dispatch_root == clean_root
    assert dispatch_root == repo / "alt-worktrees-root"
