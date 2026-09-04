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
"""

from __future__ import annotations

import argparse
from pathlib import Path

from charlie_work.ast_equivalence_gate_command import (
    run_ast_equivalence_check_command,
)
from charlie_work.subprocess_runner import RunResult


def _make_run_result(stdout: str = "", ok: bool = True) -> RunResult:
    return RunResult(
        returncode=0 if ok else 1,
        stdout=stdout,
        stderr="",
        error=None if ok else "error",
    )


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
