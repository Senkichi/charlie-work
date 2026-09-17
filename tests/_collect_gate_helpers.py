"""Shared helpers for the collect-only-gate test modules (issue #1538 / #1686).

The zero-cross-test-import guard requires every ``test_*.py`` file to be
self-contained; fixtures and helpers shared between
``test_collect_only_gate.py`` and ``test_collect_only_gate_exemption.py``
live here (same ``tests/_*.py`` seam ``_fakes_github.py`` uses).
"""

from __future__ import annotations

import argparse
from pathlib import Path

_CI_YML = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _make_cli_args(
    tmp_path: Path,
    *,
    base_collect: str = "base_collect.txt",
    head_collect: str = "head_collect.txt",
    output: str | None = None,
    pr: int | None = None,
) -> argparse.Namespace:
    """Build the argparse namespace for ``collect-only-check``."""
    return argparse.Namespace(
        command="collect-only-check",
        base_collect=base_collect,
        head_collect=head_collect,
        output=output,
        pr=pr,
        repo=None,
        config=None,
        fleet_dir=None,
        dry_run=True,
    )


def _apply_cli_mocks(monkeypatch, tmp_path: Path, *, gh=None, config=None) -> None:
    """Mock ``cli.bootstrap_command`` to return a context rooted at *tmp_path*.

    ``gh``/``config`` are injectable so exemption tests can control the live
    labels query (``ctx.gh.pr_view``) and the configured label name.
    """
    from charlie_work import cli as cli_module

    def mock_bootstrap(args):
        from charlie_work.config import OrchestratorConfig
        from charlie_work.github import GitHub
        from charlie_work.paths import RuntimePaths

        return cli_module.CommandContext(
            repo_root=tmp_path,
            config=config if config is not None else OrchestratorConfig(),
            paths=RuntimePaths.__new__(RuntimePaths),
            gh=gh if gh is not None else GitHub(repo_root=tmp_path, runtime=None, dry_run=True),
        )

    monkeypatch.setattr(cli_module, "bootstrap_command", mock_bootstrap)
