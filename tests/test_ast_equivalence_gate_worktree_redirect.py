"""Tests for the AST-equivalence gate's linked-worktree redirect opt-out
(issue #1600).

``ast-equivalence-check`` is a read-only diagnostic.  ``bootstrap_command``
defaults to redirecting a linked-worktree cwd to the shared main worktree
root (issue #648 state-safety), but that redirect makes the gate silently
inspect the main worktree's diff instead of the worktree it was invoked
from.  The command must therefore call ``bootstrap_command`` with
``redirect_to_main_worktree=False``.

This test lives in its own module rather than ``tests/test_ast_equivalence_gate.py``
because that file sits just under the 800-line file-size cap (issue #1442
ratchet) and the regression test for #1600 would push it over -- the ratchet
is fail-closed for a brand-new over-cap file with no baseline entry.

The ``test_bootstrap_command_forwards_redirect_to_main_worktree_false`` test
was relocated here from ``tests/test_cli.py`` as part of the attachment-contracts
ratchet remedy (issue #1616): ``test_cli.py`` exceeded its baselined ceiling by
one member, and the over-ceiling test is the #1603 round-2 review test that
exercises the same ``redirect_to_main_worktree`` forwarding path this module
already documents.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from _cli_fixtures import _FakeGitHub

from charlie_work import cli
from charlie_work.ast_equivalence_gate_command import (
    run_ast_equivalence_check_command,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.subprocess_runner import RunResult


def _make_run_result(stdout: str = "", ok: bool = True) -> RunResult:
    return RunResult(
        returncode=0 if ok else 1,
        stdout=stdout,
        stderr="",
        error=None if ok else "error",
    )


def _fake_repo(root: Path) -> Path:
    """A directory git's fallback resolution will treat as a work-tree root.

    ``git rev-parse`` fails inside it (no HEAD), which drives ``find_repo_root``
    into its documented ``.git``-walking fallback -- deterministic and offline.
    Duplicated from ``tests/test_cli.py`` so this module stays self-contained.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").mkdir(exist_ok=True)
    return root


def test_cli_ast_equivalence_check_passes_no_redirect_to_bootstrap(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #1600: ``ast-equivalence-check`` is a read-only diagnostic and must
    call ``bootstrap_command`` with ``redirect_to_main_worktree=False`` so it
    inspects the worktree it was invoked from, not the shared main worktree.
    Without this, running the gate from a linked worktree silently reports the
    main worktree's diff."""
    from charlie_work import cli as cli_module

    captured: dict[str, object] = {}

    def mock_bootstrap(args, **kwargs):
        captured["kwargs"] = kwargs
        from charlie_work.config import OrchestratorConfig
        from charlie_work.github import GitHub
        from charlie_work.paths import RuntimePaths

        return cli_module.CommandContext(
            repo_root=tmp_path,
            config=OrchestratorConfig(),
            paths=RuntimePaths.__new__(RuntimePaths),
            gh=GitHub(repo_root=tmp_path, runtime=None, dry_run=True),
        )

    def mock_run_captured(cmd, cwd, timeout_seconds=60, **kw):
        return _make_run_result(stdout="")

    monkeypatch.setattr(cli_module, "bootstrap_command", mock_bootstrap)
    monkeypatch.setattr(cli_module, "run_captured", mock_run_captured)

    args = argparse.Namespace(
        command="ast-equivalence-check",
        base="base",
        shim_file=None,
        output=None,
        generate_shims=None,
        repo=None,
        config=None,
        fleet_dir=None,
        dry_run=True,
    )
    result = run_ast_equivalence_check_command(args)
    assert result.ok is True
    assert captured["kwargs"] == {"redirect_to_main_worktree": False}


def test_bootstrap_command_forwards_redirect_to_main_worktree_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Issue #1600: ``bootstrap_command`` must forward
    ``redirect_to_main_worktree`` to ``find_repo_root``.

    The existing ``test_bootstrap_command_returns_frozen_context_with_all_four_fields``
    stubs ``find_repo_root`` with a ``**kw``-swallowing lambda, so nothing
    would catch a regression in the 3-line forwarding hunk (cli.py) — the exact
    line implementing this issue's fix.  This test exercises the *real*
    ``bootstrap_command`` with a *captured* ``find_repo_root`` and asserts the
    kwarg is actually forwarded, both for the opt-out (False) and the default
    (True) so a dropped or renamed kwarg fails loudly.
    """
    repo = _fake_repo(tmp_path / "charlie-work")
    config = OrchestratorConfig()

    captured: dict[str, object] = {}

    def capturing_find_repo_root(cwd, *, explicit=False, redirect_to_main_worktree=True):
        captured["cwd"] = cwd
        captured["explicit"] = explicit
        captured["redirect_to_main_worktree"] = redirect_to_main_worktree
        return repo

    monkeypatch.setattr(cli, "find_repo_root", capturing_find_repo_root)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: config)
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)

    args = cli.build_parser().parse_args(["--repo", str(repo), "roll-call"])

    # Opt-out path (#1600): the kwarg must reach find_repo_root as False.
    cli.bootstrap_command(args, redirect_to_main_worktree=False)
    assert captured["redirect_to_main_worktree"] is False

    # Default path: the kwarg must reach find_repo_root as True.
    cli.bootstrap_command(args)
    assert captured["redirect_to_main_worktree"] is True
