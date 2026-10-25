"""Shared environment sanitization for worker subprocesses.

Provides a single implementation of environment sanitization to prevent
VIRTUAL_ENV and UV_PROJECT_ENVIRONMENT leaks from the orchestrator into
worker sessions, and to isolate GitHub CLI authentication so workers cannot
use the orchestrator's stored credentials. Used by claude_code, devin_shell,
and rescue_review's cross-family adapter. It also strips an ambient
``ACP_BACKEND`` (issue #2784): set by the Devin Desktop integrated terminal,
it makes devin.exe's credential loader enforce the ACP server policy and fail
"Error: Not logged in" instead of using local CLI credentials.

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
worker_env after" pass (the chokepoint every WORKER/REVIEWER SESSION goes
through; since #2795, ``strip_provider_api_keys`` also covers the local
merge-gate suite launch -- but this is still not the only child-process env
path on this host: raw
``run_captured`` children and ci_fleet runner listeners/job steps inherit
``os.environ`` untouched; issue #2795 documents the accepted residual
exposure) also injects a safe-default
``PYTEST_XDIST_AUTO_NUM_WORKERS`` and, when the target has a real local
``.venv`` (not a reparse point), ``UV_NO_SYNC=1`` — both via ``dict.setdefault``,
so an ambient value already present in the orchestrator's own environment, or
an operator override supplied via ``worker_env``, always wins over the default.
See ``resolve_pytest_cap``/``resolve_uv_no_sync`` for the precedence helper used
by callers that need to log which layer supplied the final value.

Session-scoped temp isolation (issue #1767): ``sanitize_env`` points
TMP/TEMP/TMPDIR at a worktree-local directory (``layout.worker_tmp_dir`` by
default, or the caller-supplied ``tmp_dir`` for a caller with no worktree of
its own -- see ``rescue_review.run_cross_family_review``) so two concurrent
worker sessions on this host cannot collide on a predictable shared temp
path -- the observed incident was a worker fetching a PR body to a literal
``/tmp/pr-body.md`` and reading back a different session's body. This covers
every consumer that resolves a temp directory through the environment
(``tempfile.gettempdir()``, Node's ``os.tmpdir()``, .NET's
``Path.GetTempPath()``, ``cmd.exe``/PowerShell's ``%TEMP%``, ``mktemp``, ...).

The directory is cleared (best-effort) immediately before each call recreates
it, so a worktree that a rework dispatch reuses does not silently hand the
new round the previous round's leftover scratch files at the same predictable
path -- the same collision class #1767 was opened to eliminate, just cross-
round instead of cross-session. It is also seeded with a ``.gitignore``
containing ``*`` so its populated contents can never register as worker-
authored dirt in a target repo whose own ``.gitignore`` does not happen to
cover ``.var/`` -- a single point of enforcement independent of any target
repo's config (verified: a nested ``.gitignore`` of ``*`` suppresses the
directory, including the ``.gitignore`` file itself, from
``git status --porcelain``).

It does NOT retarget a literal ``/tmp/...`` path written from a process
spawned by Git Bash on this host: MSYS caches its ``/tmp`` ("usertemp") mount
in a shared, install-wide table seeded once while any ``bash.exe`` from that
Git install is alive, and does not recompute it from a new process's own
TMP/TEMP -- verified by launching a standalone ``bash.exe`` with TMP/TEMP
overridden and observing ``cygpath -w /tmp`` still resolve to the pre-existing
shared temp dir, unchanged. See issue #1767 for the full probe. Issue #1780
closed the residual at the layers environment sanitizing cannot reach: the
``session_scratch_dir`` prompt section instructs workers to use ``$TMPDIR``
and never a literal ``/tmp/...`` path, ``assert_session_scratch_dir`` keeps
that instruction present in every rendered worker/rework prompt even under a
repo-local flat override, and the dead-session lane in ``dead_worker_reap``
emits a ``worker_literal_tmp_path`` warning when a reaped session's
stream-json transcript shows a literal ``/tmp`` shell command anyway.

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
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from . import layout
from .ledger_context import ledger_env
from .test_slots import ROLE_GATE, arm_env
from .worktree import _unlink_reparse_point, is_junction

# Issue #646: box-wide dispatch parallelism cap. A consuming repo's own
# pyproject.toml commonly ships `addopts = "... -n auto --dist loadscope"`
# (e.g. a sibling repo), so a bare `pytest` invoked inside a worker claims every
# physical core on the shared box. With several worker tracks dispatched
# concurrently, each spawning its own uncapped `-n auto` suite, a handful of
# sessions can spawn dozens of xdist workers on an 8-core box (measured
# incident 2026-07-26: 5 stacked local suites, ~36 xdist workers on 8 physical
# cores, CI runtime 6.4min -> ~145min). This mirrors the manual operator
# recipe already used by a sibling repo's scripts/orchestrator/launch_devin_worker.sh,
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

# Issue #2836: with this set to "false" by sanitize_env, a `claude -p` session
# loads no claude.ai connectors (verified 2026-10-24: every mcp__claude_ai_*
# tool, including Gmail create_draft, disappears from the tool list). It pairs
# with the argv-side --strict-mcp-config --mcp-config {"mcpServers":{}} pin in
# tool_scope._apply_tool_scope: the argv flag drops project/user-scope MCP
# servers, this environment gate drops claude.ai connectors.
ENABLE_CLAUDEAI_MCP_SERVERS_VAR = "ENABLE_CLAUDEAI_MCP_SERVERS"

if TYPE_CHECKING:
    from .config import OrchestratorConfig
    from .test_slots import SlotPoolConfig


def provider_api_key_env_vars(config: "OrchestratorConfig | None") -> tuple[str, ...]:
    """Return the provider-key *variable names* configured under
    ``api_worker.providers``, derived from config (issue #2795) — never a
    hardcoded list, so a repo that re-keys its providers stays in step.

    The api adapter resolves the provider key from the ORCHESTRATOR process's
    own env (``api_worker.launch_api_worker``: ``os.environ.get(provider.api_key_env)``),
    which means the key value sits in the supervisor's ambient environment.
    Every other worker launch goes through ``sanitize_env``, so the names
    returned here must be popped there — otherwise every claude_code /
    devin_shell / devin_review_resume / rescue_review session inherits the
    orchestrator's provider key ambiently. The api adapter keeps working
    because its routing env (``ANTHROPIC_AUTH_TOKEN`` et al., built by
    ``api_worker._provider_env``) is merged AFTER sanitization.

    Only the variable NAMES are handled here; the key VALUE never passes
    through this module's return value (it is popped, not copied). Names are
    de-duplicated and sorted so several providers sharing one env var name
    strip identically.

    Scope note (issue #2795): sanitize_env covers worker/reviewer SESSION
    launches, and :func:`build_local_gate_env` additionally covers the local
    merge-gate suite launch (``orchestration.local_merge_gate`` ->
    ``local_suite_runner.launch_suite_gate``), which runs the PR branch's
    worker-authored test suite. Other children of the orchestrator or
    supervisor process inherit ``os.environ`` raw —
    ``subprocess_runner.run_captured`` children
    (``env = {**os.environ, ...}``) and, via ci_fleet's own
    ``runners._sanitize_env`` (which strips only ``UV_/VIRTUAL_ENV/PYTHON/
    PIP_/CLAUDE`` prefixes), every runner listener the allocator starts and
    every self-hosted CI job step. That residual ambient exposure is
    documented as accepted in docs/RUNBOOK.md's API-worker section; stripping
    those surfaces is a ci_fleet-side or future charlie-work change keyed on
    the same config-derived names this function returns.
    """
    if config is None:
        return ()
    api_worker = getattr(config, "api_worker", None)
    if api_worker is None:
        return ()
    providers = getattr(api_worker, "providers", None) or {}
    return tuple(
        sorted(
            {
                provider.api_key_env
                for provider in providers.values()
                if provider is not None and getattr(provider, "api_key_env", "")
            }
        )
    )


# Issue #502/#1001: the GitHub token variable names sanitize_env() strips from
# every worker subprocess's environment. The single source of truth for both
# sanitize_env's strip loop and the worker-github-token predicate shared by
# doctor.py's preflight check and workflow.py's dispatch gate (issue #1001).
# Mirrored nowhere — doctor.py imports this constant instead of maintaining a
# private copy (the pre-#1001 _STRIPPED_GH_TOKEN_VARS was deleted when this
# became the shared predicate).
STRIPPED_GH_TOKEN_VARS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
)


def strip_provider_api_keys(
    env: dict[str, str],
    config: "OrchestratorConfig | None",
) -> dict[str, str]:
    """Pop the configured provider-key variable *names* from ``env`` in place,
    and return ``env`` for chaining.

    The one implementation every env-construction site that copies the
    supervisor's ambient environment into a child process must run through
    (issue #2795): :func:`sanitize_env` for worker/reviewer session launches
    and :func:`build_local_gate_env` for the local merge-gate suite launch
    (-> ``local_suite_runner.launch_suite_gate``) built from
    ``{**os.environ, ...}`` — that child runs the PR branch's worker-authored
    test suite, so it must not inherit the supervisor's provider key either.

    See :func:`provider_api_key_env_vars` for where the names come from and
    which surfaces still carry the residual exposure.
    """
    key_vars = provider_api_key_env_vars(config)
    if key_vars and os.name == "nt":
        # Windows casefolds environment-variable KEYS: dict(os.environ) hands
        # back upper-cased keys, so an exact env.pop("Mixed_Case") silently
        # fails open — while the api adapter's own os.environ.get() lookup
        # would still FIND the value (os.environ.get is case-insensitive on
        # Windows). Match case-insensitively so the strip keeps up with the
        # api adapter's read; the pop still deletes whatever key casing the
        # dict actually carries.
        wanted = {name.upper() for name in key_vars}
        for env_key in [k for k in env if k.upper() in wanted]:
            del env[env_key]
    else:
        # POSIX keeps case-sensitive env keys, so the exact-name pop is exact.
        for env_key in key_vars:
            env.pop(env_key, None)
    return env


def build_local_gate_env(
    test_slots: "SlotPoolConfig",
    issue_number: int,
    config: "OrchestratorConfig | None",
) -> dict[str, str]:
    """Build the local merge-gate suite launch env (issue #2795).

    The gate suite runs the PR branch's worker-authored tests, so its env is
    the supervisor's ambient environment with the gate slot plugin armed and
    the test ledger tagged, then the config-derived provider-key strip applied
    AFTER the ``os.environ`` copy and BEFORE the spawn. Extracted from
    ``orchestration.local_merge_gate`` to keep that file within its size cap;
    ``tests/test_local_merge_gate_env.py`` drives the real gate launch path
    through this helper.
    """
    gate_env = {
        **os.environ,
        **arm_env(test_slots, role=ROLE_GATE),
        **ledger_env("gate", issue_number),
    }
    return strip_provider_api_keys(gate_env, config)


# Env every claude-code worker launches with unless `worker_env` overrides it
# (issue #2096). Verified against Claude Code CLI 2.1.286: with this set, the
# Bash tool has no `run_in_background` parameter. In print mode, ending the
# turn ends the session, so a background Bash task would be killed with the
# work uncommitted.
WORKER_ENV_DEFAULTS: dict[str, str] = {"CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1"}


def build_worker_env(
    sanitized_env: Mapping[str, str], env: Mapping[str, object] | None
) -> dict[str, str]:
    """Merge sanitized base env, harness defaults, then operator ``worker_env``.

    Later layers win, so an operator ``worker_env`` value can still override a
    harness default.
    """
    return {
        **sanitized_env,
        **WORKER_ENV_DEFAULTS,
        **{str(k): str(v) for k, v in (env or {}).items()},
    }


def _is_owned_venv(path: Path) -> bool:
    """Return True if ``path`` is a real directory, not a reparse point."""
    return path.is_dir() and not is_junction(path)


def _is_git_worktree(path: Path) -> bool:
    """Return True if ``path`` looks like a git worktree (has a .git file)."""
    git_meta = path / ".git"
    return git_meta.exists() and not git_meta.is_dir()


def sanitize_env(
    target_path: Path,
    *,
    tmp_dir: Path | None = None,
    config: "OrchestratorConfig | None" = None,
) -> dict[str, str]:
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
        tmp_dir: Override for the session-scoped TMP/TEMP/TMPDIR directory
            (issue #1767). Defaults to ``layout.worker_tmp_dir(target_path)``,
            which is keyed on ``target_path`` and correct for every caller
            that owns a worktree's full create/teardown lifecycle
            (``claude_code``, ``devin_shell``). A caller with no such
            lifecycle of its own -- ``rescue_review.run_cross_family_review``,
            whose ``target_path`` is the shared main checkout, not a
            per-session worktree -- must pass an explicit, per-call directory
            it creates and reclaims itself; otherwise every call against the
            same ``target_path`` would derive the same shared, never-cleaned
            directory. See :func:`layout.worker_tmp_dir` for the default
            derivation.

        config: The orchestrator config, so the provider-key variable names
            configured under ``api_worker.providers`` are popped too
            (issue #2795) — see :func:`provider_api_key_env_vars`. The default
            of ``None`` fails open (strips nothing extra) for out-of-tree
            callers; every in-tree launch site must pass it explicitly — an
            AST guard test (``tests/test_env_sanitize.py``) fails when a
            ``sanitize_env(`` call in ``src/`` omits ``config=``.

    Returns:
        A sanitized environment dictionary.
    """
    env = dict(os.environ)
    target_venv = target_path / ".venv"

    # Always pop these first to prevent leaks from the orchestrator (issue #117):
    # any ambient UV_PROJECT_ENVIRONMENT at this point is by definition the
    # orchestrator's.
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    # An inherited PWD names the ORCHESTRATOR's directory, not the worker's
    # cwd. opencode (Bun) trusts PWD over the real process cwd for its tools:
    # with a stale PWD, a worker launched with cwd=<worktree> ran git in the
    # main checkout and committed there (observed 2026-10-06). Dropping it lets
    # every shell/runtime derive PWD from the actual cwd.
    env.pop("PWD", None)
    env.pop("OLDPWD", None)
    # Issue #2784: strip an IDE ACP session marker from the invoking terminal
    # (e.g. a Devin Desktop integrated terminal). With ACP_BACKEND present,
    # devin.exe --print enforces the ACP server credential policy and exits 1
    # with "Error: Not logged in" even when local CLI credentials are valid.
    env.pop("ACP_BACKEND", None)

    # Issue #2836: gate claude.ai connectors out of every Claude child when MCP
    # isolation is on (the config default). This is the one place it is set so
    # every launch through sanitize_env gets it. config=None resolves to the
    # defaults, which isolate (matching OrchestratorConfig's own default).
    _claude_code_cfg = getattr(config, "claude_code", None)
    isolate_mcp = True if _claude_code_cfg is None else _claude_code_cfg.isolate_mcp
    if isolate_mcp:
        env[ENABLE_CLAUDEAI_MCP_SERVERS_VAR] = "false"
    else:
        env.pop(ENABLE_CLAUDEAI_MCP_SERVERS_VAR, None)

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
    for token_var in STRIPPED_GH_TOKEN_VARS:
        env.pop(token_var, None)

    # Issue #2795: never let the orchestrator's api-provider keys reach a
    # worker session ambiently. The api adapter resolves the key from the
    # supervisor's own env, which is why the supervisor carries it here —
    # see strip_provider_api_keys() for the full invariant. config=None
    # fails open (strips nothing); every in-tree launch site passes config
    # under the AST guard test, so None only ever reaches out-of-tree callers.
    strip_provider_api_keys(env, config)

    # Issue #502: isolate gh's config directory so the worker cannot fall back
    # to the orchestrator's stored ``gh auth login`` credentials. A
    # worktree-local empty directory is created on demand. If an operator later
    # merges a scoped ``GH_TOKEN`` via ``worker_env``, it takes precedence over
    # any (empty) stored credential.
    gh_config_dir = layout.gh_config_dir(target_path)
    gh_config_dir.mkdir(parents=True, exist_ok=True)
    env["GH_CONFIG_DIR"] = str(gh_config_dir)

    # Issue #1767: give every worker subprocess a session-scoped TMP/TEMP/
    # TMPDIR so two concurrent sessions on this host cannot collide on a
    # predictable shared temp path. Keyed on target_path like GH_CONFIG_DIR
    # above (unless the caller passes an explicit override -- see the
    # ``tmp_dir`` parameter docstring) -- each worktree is already unique per
    # active session.
    resolved_tmp_dir = tmp_dir if tmp_dir is not None else layout.worker_tmp_dir(target_path)
    # Clear any leftover content before recreating: a rework dispatch reuses
    # the same worktree (and therefore the same tmp_dir) round over round, so
    # without this a new round's worker would start with the previous
    # round's scratch files still in place at the same predictable path --
    # the exact cross-session collision class #1767 exists to eliminate,
    # just cross-round instead of cross-process. Best-effort, mirroring the
    # reparse-point unlink above: a leftover locked file from a just-killed
    # previous worker must not block this launch, and leaving one stray file
    # behind is still strictly better isolation than the pre-fix shared host
    # temp dir.
    shutil.rmtree(resolved_tmp_dir, ignore_errors=True)
    # Directory creation can fail (permissions, disk full, path length); this
    # raises OSError rather than swallowing it, exactly like GH_CONFIG_DIR's
    # mkdir above, so callers report a real launch failure instead of
    # silently handing a worker an unusable/shared temp path. See the module
    # docstring for the verified Git-Bash/MSYS caveat this does NOT cover.
    resolved_tmp_dir.mkdir(parents=True, exist_ok=True)
    # Self-ignoring regardless of the target repo's own .gitignore: a
    # populated worker-tmp dir inside a worktree would otherwise register as
    # worker-authored dirt on any repo whose .gitignore does not happen to
    # cover .var/ (verified: a nested .gitignore of "*" suppresses the whole
    # directory, including this file itself, from `git status --porcelain`).
    (resolved_tmp_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    env["TMP"] = str(resolved_tmp_dir)
    env["TEMP"] = str(resolved_tmp_dir)
    env["TMPDIR"] = str(resolved_tmp_dir)

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
    "provider_api_key_env_vars",
    "strip_provider_api_keys",
    "resolve_pytest_cap",
    "resolve_uv_no_sync",
    "PYTEST_XDIST_AUTO_NUM_WORKERS_VAR",
    "DEFAULT_PYTEST_XDIST_AUTO_NUM_WORKERS",
    "UV_NO_SYNC_VAR",
    "ENABLE_CLAUDEAI_MCP_SERVERS_VAR",
    "STRIPPED_GH_TOKEN_VARS",
]
