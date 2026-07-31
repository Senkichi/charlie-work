"""Shared environment sanitization for worker subprocesses.

Provides a single implementation of environment sanitization to prevent
VIRTUAL_ENV and UV_PROJECT_ENVIRONMENT leaks from the orchestrator into
worker sessions, and to isolate GitHub CLI authentication so workers cannot
use the orchestrator's stored credentials. Used by claude_code, devin_shell,
and cross_family adapters.

Security (issue #502): workers must not inherit the orchestrator's
admin-scoped ``GH_TOKEN``/``GITHUB_TOKEN`` (or the GHES equivalents
``GH_ENTERPRISE_TOKEN``/``GITHUB_ENTERPRISE_TOKEN``) or ``gh`` stored
credentials. These variables are stripped from the sanitized base environment,
and ``GH_CONFIG_DIR`` is forced to a worktree-local empty directory so ``gh``
cannot fall back to the orchestrator's ``gh auth login`` state. Operators who
want workers to use ``gh`` must supply a scoped token via ``worker_env`` in the
adapter config (``devin.worker_env`` or ``claude_code.worker_env``); the adapter
merges ``worker_env`` AFTER sanitization, so an explicit scoped ``GH_TOKEN``
wins and the orchestrator's config/token never reaches the worker process.

Dispatch parallelism cap (issue #646): the same "sanitize once, merge
worker_env after" chokepoint also injects a safe-default
``PYTEST_XDIST_AUTO_NUM_WORKERS`` and, when the target has a real local
``.venv`` (not a reparse point), ``UV_NO_SYNC=1`` — both via ``dict.setdefault``,
so an ambient value already present in the orchestrator's own environment, or
an operator override supplied via ``worker_env``, always wins over the default.
See ``resolve_pytest_cap``/``resolve_uv_no_sync`` for the precedence helper used
by callers that need to log which layer supplied the final value.

Shared-venv confinement (issue #649): ``sanitize_env`` does not rely on a
``UV_PROJECT_ENVIRONMENT`` pin to protect a junctioned shared venv. Once the
orchestrator's value is popped, pinning the variable to ``.venv`` is a no-op
over uv's default project-environment lookup, and a junctioned ``.venv`` still
resolves to the shared target. Instead, ``sanitize_env`` treats a ``.venv`` that
is a Windows junction or POSIX symlink (a reparse point) as a view into a
shared venv, not an owned local environment. Inside a git worktree, that
reparse point is unlinked (the target is never touched) so a worker's
``uv sync`` cannot rewrite the shared venv through it. A real local ``.venv``
directory keeps ``VIRTUAL_ENV`` and the ``UV_NO_SYNC`` convenience guard;
``UV_PROJECT_ENVIRONMENT`` is intentionally not set because uv's default
project environment is already ``.venv`` in the project root.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import layout
from .worktree import _unlink_reparse_point, is_junction

# Issue #646: box-wide dispatch parallelism cap. A consuming repo's own
# pyproject.toml commonly ships `addopts = "... -n auto --dist loadscope"`
# (e.g. job-cannon), so a bare `pytest` invoked inside a worker claims every
# physical core on the shared box. With several worker tracks dispatched
# concurrently, each spawning its own uncapped `-n auto` suite, a handful of
# sessions can spawn dozens of xdist workers on an 8-core box (measured
# incident 2026-07-26: 5 stacked local suites, ~36 xdist workers on 8 physical
# cores, CI runtime 6.4min -> ~145min). This mirrors the manual operator
# recipe already used by job-cannon's scripts/orchestrator/launch_devin_worker.sh,
# which exports this same variable ("3 concurrent tracks x 2 xdist workers ~= 8
# physical cores") — this constant is that same value, applied automatically
# instead of requiring every dispatch path to remember to set it. Cited by
# path + variable name rather than line number: the previous form pointed at
# .var/devin-orchestrator/…:49, and both the directory and the line moved.
PYTEST_XDIST_AUTO_NUM_WORKERS_VAR = "PYTEST_XDIST_AUTO_NUM_WORKERS"
DEFAULT_PYTEST_XDIST_AUTO_NUM_WORKERS = "2"

# Issue #646 / #649: pairs with the cap above. UV_NO_SYNC gates only `uv run`'s
# *implicit* sync (the `--no-sync` flag, per uv docs) — it does NOT gate an
# explicit `uv sync`, which still resolves and rewrites the project environment.
# So UV_NO_SYNC alone cannot protect a shared/junctioned .venv from a worker's
# explicit `uv sync` rewriting it out from under sibling workers. The real
# protection comes from `sanitize_env` treating only a real, non-reparse-point
# .venv as an owned local venv, and unlinking a .venv reparse point inside a
# git worktree so uv is forced to create and manage a real local .venv instead.
# UV_NO_SYNC is still worth keeping alongside a confirmed real .venv: it
# prevents `uv run`'s implicit sync from pruning extras a worker needs, but it
# is a convenience guard, not the shared-venv safety boundary. UV_NO_SYNC must
# only ever be set alongside a confirmed real .venv — never unconditionally,
# since a worktree that legitimately manages its own venv still wants uv to
# manage its own.
UV_NO_SYNC_VAR = "UV_NO_SYNC"
_UV_NO_SYNC_DEFAULT = "1"


def _is_owned_venv(path: Path) -> bool:
    """Return True if ``path`` is a real directory, not a reparse point."""
    return path.is_dir() and not is_junction(path)


def _is_git_worktree(path: Path) -> bool:
    """Return True if ``path`` looks like a git worktree (has a .git file)."""
    git_meta = path / ".git"
    return git_meta.exists() and not git_meta.is_dir()


def sanitize_env(target_path: Path) -> dict[str, str]:
    """Return a sanitized environment for worker subprocesses.

    Drops VIRTUAL_ENV and UV_PROJECT_ENVIRONMENT from the parent environment
    to prevent the orchestrator's venv from leaking into worker sessions. If the
    target path contains a real local ``.venv`` directory, VIRTUAL_ENV is set to
    that path and ``UV_NO_SYNC`` is defaulted to ``1`` so ``uv run`` does not
    implicitly re-sync an already-good environment.

    A ``.venv`` that is a Windows junction or POSIX symlink is *not* an owned
    local venv; it is a window into a shared venv. Inside a git worktree (a
    ``.git`` file, not a directory), that reparse point is unlinked so a
    worker's ``uv sync`` cannot rewrite the shared venv through it. The target
    of the reparse point is never touched.

    ``UV_PROJECT_ENVIRONMENT`` is intentionally not set by this function. After
    the orchestrator's value is popped, uv's default project environment is
    already ``.venv`` in the project root, so setting the variable would be a
    no-op. The previous attempt to use it as a safety pin for a junctioned
    shared venv did not work (issue #649).

    Args:
        target_path: The worktree or repo path to check for a .venv directory.

    Returns:
        A sanitized environment dictionary.
    """
    env = dict(os.environ)
    target_venv = target_path / ".venv"

    # Always pop these first to prevent leaks from the orchestrator (issue #117):
    # any ambient UV_PROJECT_ENVIRONMENT at this point is by definition the
    # orchestrator's.
    env.pop("UV_PROJECT_ENVIRONMENT", None)

    # A .venv reparse point (junction/symlink) is a view into a shared venv,
    # not an owned local venv. Unlink it inside a worktree so uv cannot
    # resolve/rewrite the shared venv through it.
    if is_junction(target_venv) and _is_git_worktree(target_path):
        try:
            _unlink_reparse_point(target_venv)
        except OSError:
            # Best-effort: if we cannot unlink, fall through and at least do
            # not pin uv to the shared venv.
            pass

    if _is_owned_venv(target_venv):
        # Target has its own real local venv — let direct `python` invocations
        # find it. uv ignores VIRTUAL_ENV and resolves .venv by default, which
        # is the same path, so we do not need to set UV_PROJECT_ENVIRONMENT.
        env["VIRTUAL_ENV"] = str(target_venv)
        # Convenience guard: don't let `uv run` implicitly prune extras the
        # worker may have added; explicit `uv sync` is still possible.
        env.setdefault(UV_NO_SYNC_VAR, _UV_NO_SYNC_DEFAULT)
    else:
        # No owned local venv — drop VIRTUAL_ENV to prevent leaks.
        env.pop("VIRTUAL_ENV", None)

    # Issue #502: never inherit the orchestrator's GitHub auth tokens. Workers
    # must use an operator-supplied scoped token via worker_env.GH_TOKEN. Strip
    # both the dotcom tokens and the GHES enterprise tokens so a worker cannot
    # fall back on an enterprise-scoped credential to run ``gh pr merge``.
    for token_var in (
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GH_ENTERPRISE_TOKEN",
        "GITHUB_ENTERPRISE_TOKEN",
    ):
        env.pop(token_var, None)

    # Issue #502: isolate gh's config directory so the worker cannot fall back
    # to the orchestrator's stored ``gh auth login`` credentials. A
    # worktree-local empty directory is created on demand. If an operator later
    # merges a scoped ``GH_TOKEN`` via ``worker_env``, it takes precedence over
    # any (empty) stored credential.
    gh_config_dir = layout.gh_config_dir(target_path)
    gh_config_dir.mkdir(parents=True, exist_ok=True)
    env["GH_CONFIG_DIR"] = str(gh_config_dir)

    # Issue #646: cap xdist worker fan-out and guard a real local .venv from a
    # concurrent uv sync. setdefault() so an already-exported ambient value
    # always wins over the built-in safety-net default — and a caller merging
    # a config-level worker_env override AFTER this function returns (the
    # established pattern in claude_code.py/devin_shell.py) can still override
    # either value. See resolve_pytest_cap()/resolve_uv_no_sync() for the full
    # 3-way precedence (config > env > default) used at the launch sites for
    # logging.
    env.setdefault(PYTEST_XDIST_AUTO_NUM_WORKERS_VAR, DEFAULT_PYTEST_XDIST_AUTO_NUM_WORKERS)

    return env


def resolve_pytest_cap(
    sanitized_env: dict[str, str], worker_env_override: dict[str, str] | None
) -> tuple[str, str]:
    """Return ``(value, source)`` for ``PYTEST_XDIST_AUTO_NUM_WORKERS`` after
    the full merge a launcher performs: ``sanitize_env()`` output, then
    ``worker_env_override`` (the adapter config knob) layered on top — the
    same order ``launch_claude_worker``/``launch_devin_session`` use to build
    the actual subprocess environment.

    ``source`` is one of:
      - ``"config"``  -- an operator-supplied ``worker_env`` value won
      - ``"env"``     -- inherited from the orchestrator's own ambient
        environment (already exported before this process started)
      - ``"default"`` -- neither was set; the built-in safety-net default
        (``DEFAULT_PYTEST_XDIST_AUTO_NUM_WORKERS``) applied

    Used purely for launch-time diagnostics (issue #646) — does not itself
    mutate any environment.
    """
    override = (worker_env_override or {}).get(PYTEST_XDIST_AUTO_NUM_WORKERS_VAR)
    if override is not None:
        return str(override), "config"
    if PYTEST_XDIST_AUTO_NUM_WORKERS_VAR in os.environ:
        return sanitized_env[PYTEST_XDIST_AUTO_NUM_WORKERS_VAR], "env"
    return (
        sanitized_env.get(
            PYTEST_XDIST_AUTO_NUM_WORKERS_VAR, DEFAULT_PYTEST_XDIST_AUTO_NUM_WORKERS
        ),
        "default",
    )


def resolve_uv_no_sync(
    target_path: Path,
    sanitized_env: dict[str, str],
    worker_env_override: dict[str, str] | None,
) -> tuple[str | None, str]:
    """Return ``(value, source)`` for ``UV_NO_SYNC`` after the same 3-way
    merge described in ``resolve_pytest_cap``.

    ``value`` is ``None`` only when ``target_path`` has no real local ``.venv``,
    no override forced one, AND no ambient ``UV_NO_SYNC`` was already exported —
    the safety-net default is only ever injected alongside a real ``.venv``
    (see ``sanitize_env``'s docstring for why an unconditional default would
    be wrong for a worktree that legitimately manages its own venv). But
    ``sanitize_env`` never *pops* an ambient ``UV_NO_SYNC`` — it only skips
    the ``setdefault`` when there's no owned ``.venv`` — so an operator-exported
    value still reaches the child process regardless of ``.venv``
    presence. The ambient-env check must therefore run BEFORE the no-venv
    short-circuit, or this function reports ``None`` while the subprocess
    actually inherits a real value.

    ``source`` adds ``"no-venv"`` to the three values ``resolve_pytest_cap``
    uses, for the case where no value applies at all.
    """
    override = (worker_env_override or {}).get(UV_NO_SYNC_VAR)
    if override is not None:
        return str(override), "config"
    if UV_NO_SYNC_VAR in os.environ:
        return sanitized_env[UV_NO_SYNC_VAR], "env"
    if not _is_owned_venv(target_path / ".venv"):
        return None, "no-venv"
    return sanitized_env.get(UV_NO_SYNC_VAR, _UV_NO_SYNC_DEFAULT), "default"


__all__ = [
    "sanitize_env",
    "resolve_pytest_cap",
    "resolve_uv_no_sync",
    "PYTEST_XDIST_AUTO_NUM_WORKERS_VAR",
    "DEFAULT_PYTEST_XDIST_AUTO_NUM_WORKERS",
    "UV_NO_SYNC_VAR",
]
