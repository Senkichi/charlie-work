"""Shared environment sanitization for worker subprocesses.

Provides a single implementation of environment sanitization to prevent
VIRTUAL_ENV and UV_PROJECT_ENVIRONMENT leaks from the orchestrator into
worker sessions. Used by claude_code, devin_shell, and cross_family adapters.

Security (issue #502): workers run with ``--permission-mode dangerous`` and
must not inherit the orchestrator's admin-scoped ``GH_TOKEN``/``GITHUB_TOKEN``.
These variables are stripped from the sanitized base environment. Operators who
want workers to use ``gh`` must supply a scoped token via ``worker_env`` in the
adapter config (``devin.worker_env`` or ``claude_code.worker_env``); the adapter
merges ``worker_env`` AFTER sanitization, so an explicit scoped ``GH_TOKEN``
wins and the orchestrator's admin token never reaches the worker process.
"""

from __future__ import annotations

import os
from pathlib import Path


def sanitize_env(target_path: Path) -> dict[str, str]:
    """Return a sanitized environment for worker subprocesses.

    Drops VIRTUAL_ENV and UV_PROJECT_ENVIRONMENT from the parent environment
    to prevent the orchestrator's venv from leaking into worker sessions. If the
    target path contains a .venv directory, VIRTUAL_ENV is set to that path
    instead of being dropped.

    This is a defense-in-depth measure: workers should resolve their own
    environment via uv run --active or similar, not inherit the orchestrator's.

    Args:
        target_path: The worktree or repo path to check for a .venv directory.

    Returns:
        A sanitized environment dictionary.
    """
    env = dict(os.environ)
    target_venv = target_path / ".venv"

    # Always pop UV_PROJECT_ENVIRONMENT first to prevent leaks
    env.pop("UV_PROJECT_ENVIRONMENT", None)

    if target_venv.is_dir():
        # Target has its own venv — use it
        env["VIRTUAL_ENV"] = str(target_venv)
    else:
        # No target venv — drop VIRTUAL_ENV to prevent leaks
        env.pop("VIRTUAL_ENV", None)

    # Issue #502: never inherit the orchestrator's GitHub auth tokens. Workers
    # must use an operator-supplied scoped token via worker_env.GH_TOKEN.
    for token_var in ("GH_TOKEN", "GITHUB_TOKEN"):
        env.pop(token_var, None)

    return env


__all__ = ["sanitize_env"]
