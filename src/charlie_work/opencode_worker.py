"""opencode adapter -- the opencode CLI (``opencode run``) as worker and reviewer.

A headless ``opencode run`` session is launched through the same
worktree/prompt/env-sanitisation/Popen/terminal-watcher/sidecar stack as a
claude-code worker: this module delegates to ``claude_code.launch_claude_worker``
with ``adapter_kind="opencode"`` (sidecars land as ``issue-<n>.opencode.json``,
the shared ``ClaudeWorkerRecord`` shape) and supplies its own CLI flag pins via
``cli_pins`` -- the Claude ``--model``/``--effort`` pins are not valid opencode
argv. It owns only what is opencode-specific: the default command template, the
``provider/model`` resolution, and the permission posture env.

Review-capable since the harness registry flipped ``review=True``: a
``review=True`` launch goes through the same ``launch_claude_worker`` review
path as a claude-code reviewer (detached-HEAD checkout, ``-review``
log/sidecar suffixes), but the claude plan-mode flag pin is replaced by two
opencode equivalents -- the ``OPENCODE_CONFIG_CONTENT`` deny policy
(``opencode_review_policy._REVIEWER_CONFIG``) and
:func:`sanitize_review_command_template`, which strips every
agent/session/server mover and PINS ``--standalone``. The posture lives in
env, so the argv half exists to guarantee that env actually applies.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import launch_events
from .atomic_write import write_text_atomic
from .claude_code import (
    ClaudeWorkerRecord,
    _error_record,
    _log_path,
    _write_record,
    launch_claude_worker,
)
from .config import OrchestratorConfig
from . import opencode_review_policy
from .opencode_log import opencode_data_dir
from .opencode_review_policy import (  # noqa: F401  (deliberate re-export)
    _AUTH_CONTENT_ENV,
    _REVIEWER_CONFIG,
    _REVIEW_ENV_STRIP,
    _REVIEW_ENV_STRIP_PREFIXES,
)

# ``opencode run`` with the prompt fed on stdin (no positional message) and raw
# JSON events on stdout -- one object per line, ``type`` at top level, which
# ``opencode_log.provider_error_digest`` parses so the quota/throttle
# classifiers read only opencode's own error records, never tool output. ``--auto``
# approves permission requests that config does not explicitly deny; without it
# ``run`` auto-REJECTS them (external_directory et al.) rather than prompting.
# ``--standalone``: without it ``run`` connects to the operator's SHARED
# background service (verified live 2026-11-18: a service process is already
# running) -- the service owns its own env/config/db, so the per-issue
# XDG_DATA_HOME db, the OPENCODE_CONFIG_CONTENT posture and the seeded
# auth.json would all be bypassed. A private server per launch makes every
# env pin actually apply.
# ``--print-logs --log-level warn``: on a provider 429 (OpenCode Go usage limit)
# opencode retries internally, honouring retry-after -- hours, for a 5h limit --
# and without these flags the log stays EMPTY the whole time, so the quota
# classifier never sees it. With them, each failed attempt writes an
# ``level=ERROR`` line to stderr (merged into the log) on the first attempt --
# 1.x ``error.error="AI_APICallError: <provider message>"``, 2.x
# ``cause="AI.Error: <provider message>"`` plus a {"type":"retry"} event on
# stdout -- both decoded by opencode_log. The stall watchdog then reaps the
# silent retry sleep and classifies that tail.
#
# Version note (verified against the installed binary, 2026-11-18):
# ``--log-level`` choices are LOWERCASE on 2.3.0 (``warn``/``warning``/...);
# the 1.x-era uppercase ``WARN`` is rejected and killed every launch
# instantaneously. The 2.x value is used unconditionally -- 2.x is the
# supported binary shape (doctor._probe_opencode validates every template
# flag against the installed binary's own ``run --help`` so a drift fails the
# doctor probe, not the launch).
DEFAULT_COMMAND_TEMPLATE: tuple[str, ...] = (
    "opencode",
    "run",
    "--auto",
    "--standalone",
    "--format",
    "json",
    "--print-logs",
    "--log-level",
    "warn",
)

# Injected as OPENCODE_CONFIG_CONTENT (highest-precedence inline config). The
# worker runs unattended in an isolated worktree, so every tool is allowed --
# parity with the claude-code worker's ``--permission-mode bypassPermissions``
# and the Devin workers. ``autoupdate`` off: a self-update mid-fleet would swap
# the binary under concurrently running sessions. ``share`` off: sessions must
# never be published to opencode.ai. ``snapshot`` off: opencode's undo snapshots
# keep a shadow git repo of the worktree, which contends on index.lock across
# concurrent workers and duplicates the tree for no benefit (git is the record).
_WORKER_CONFIG: dict[str, Any] = {
    "$schema": "https://opencode.ai/config.json",
    "permission": {"*": "allow"},
    "autoupdate": False,
    "share": "disabled",
    "snapshot": False,
}


_WORKER_ENV: dict[str, str] = {
    "OPENCODE_CONFIG_CONTENT": json.dumps(_WORKER_CONFIG, separators=(",", ":")),
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
    # Worker hermeticity: opencode injects the operator's personal
    # ~/.claude/CLAUDE.md by default (observed on 1.x). Its "pause before
    # pushing" style rules are what made swe-1.7 workers decline to push.
    # The repo's own CLAUDE.md / AGENTS.md still load.
    # Two names because the enforcement surface moved with the major version
    # (verified in the installed 2.3.0 bundle's env table,
    # 2026-11-18: ``config.claude = !(S(OPENCODE_DISABLE_CLAUDE_CODE) ||
    # S(OPENCODE_DISABLE_CLAUDE_CODE_SKILLS))``):
    # - OPENCODE_DISABLE_CLAUDE_CODE_PROMPT is the 1.x name (0 occurrences
    #   in the 2.3.0 binary -- a no-op there, kept so a 1.x binary keeps the
    #   posture).
    # - OPENCODE_DISABLE_CLAUDE_CODE is the 2.x name for the same switch.
    "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "1",
    "OPENCODE_DISABLE_CLAUDE_CODE": "1",
    # Default bash-tool timeout is 2 minutes; a full local test run exceeds it.
    # 1.x-only name (0 occurrences in the 2.3.0 binary; kept for 1.x
    # binaries -- inert but harmless on 2.x).
    "OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS": "1800000",
}


def _host_auth_path(environ: Mapping[str, str]) -> Path:
    """Where the operator's ``opencode auth login`` stored credentials."""
    data_home = environ.get("XDG_DATA_HOME")
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return base / "opencode" / "auth.json"


def _scoped_auth_content(auth_path: Path, provider: str) -> str | None:
    """The host auth.json reduced to ``provider``'s entry, or None.

    Only the selected provider's credential reaches the worker: the host file
    also holds every other provider the operator logged into (Anthropic OAuth,
    Copilot, ...), and the worker's tools and children inherit its env -- one
    ``env`` dump in tool output would copy them into the log.
    """
    try:
        entries = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entries, dict) or provider not in entries:
        return None
    return json.dumps({provider: entries[provider]}, separators=(",", ":"))


def _scope_auth_json(text: str, provider: str) -> str | None:
    """An operator-supplied ``OPENCODE_AUTH_CONTENT`` reduced to ``provider``.

    The env var is the documented operator injection point (it used to be
    opencode's own credential channel on 1.x). Parsing and scoping -- not
    dropping -- is what keeps a reviewer authenticated when the operator
    authenticates through it, while still containing the leak surface: only
    ``provider``'s entry survives, so a multi-provider value (or a
    ``wellknown`` entry pointing at remote config) cannot ride into the
    child env. A value that is not a JSON object returns None so the caller
    falls through to the host auth.json; an object that lacks ``provider``
    scopes to an empty map -- an operator-supplied value is authoritative.
    """
    try:
        entries = json.loads(text)
    except ValueError:
        return None
    if not isinstance(entries, dict):
        return None
    return json.dumps(
        {provider: entries[provider]} if provider in entries else {},
        separators=(",", ":"),
    )


def resolve_scoped_auth(
    worker_env: Mapping[str, str] | None,
    environ: Mapping[str, str],
    provider: str,
) -> str | None:
    """The selected provider's credential, in priority order, else None.

    1. An operator-supplied ``OPENCODE_AUTH_CONTENT`` (``worker_env`` first,
       then the ambient environ -- both are declared operator channels),
       scoped to ``provider`` by :func:`_scope_auth_json`.
    2. The host ``auth.json`` written by ``opencode auth login``, scoped by
       :func:`_scoped_auth_content`.
    """
    for source in (*([worker_env or {}]), environ):
        for key, value in source.items():
            if key.upper() == _AUTH_CONTENT_ENV:
                scoped = _scope_auth_json(value, provider)
                if scoped is not None:
                    return scoped
    return _scoped_auth_content(_host_auth_path(environ), provider)


def seed_auth_file(data_dir: Path, scoped_auth: str | None) -> Path | None:
    """Write the scoped credential where opencode 2.x reads auth from.

    opencode 2.x dropped the ``OPENCODE_AUTH_CONTENT`` env channel (0
    occurrences in the installed 2.3.0 binary -- verified): credentials live
    in ``<XDG_DATA_HOME>/opencode/auth.json``, imported into the data dir's
    credential DB once at first-server start (the binary's
    ``import_legacy_credentials`` migration -- a later edit to auth.json is
    ignored). The launcher wipes the per-issue data dir before every launch,
    which previously wiped the credential with it (every launch then died on
    ``provider.no-route``), so the scoped credential is written back
    atomically AFTER the wipe -- the fresh dir re-runs the import against it.
    The file lives inside the per-issue data dir, which the reap removes
    with the sidecar -- it never outlives the session it authenticates.
    ``None`` (no credential anywhere) writes nothing: the child may still
    authenticate from provider env keys.

    Credential lifetime (deliberate, documented because it is a change from
    the env-only posture ``OPENCODE_AUTH_CONTENT`` originally had): this file
    is the ONLY place the orchestrator writes a provider credential to disk.
    It lives inside the per-issue data dir, which ``WorkerView.reap_sidecar``
    removes with the session (``shutil.rmtree`` of ``opencode_data_dir``), and
    the orphan sweep covers sidecars for sessions that died before the reap.
    It must therefore never be relocated outside that directory, and any new
    reader of the data dir must treat ``opencode/auth.json`` as a secret.
    """
    if not scoped_auth:
        return None
    auth_file = data_dir / "opencode" / "auth.json"
    try:
        auth_file.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(auth_file, scoped_auth, encoding="utf-8")
    except OSError:
        # The launcher's own mkdir/reporting path surfaces a real storage
        # failure; a missing credential file degrades to the pre-fix
        # behaviour (unauthenticated launch), never a lost launch.
        return None
    return auth_file


def worker_env_for(
    issue_number: int,
    sessions_dir: Path,
    worker_env: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    provider: str = "opencode-go",
    review: bool = False,
) -> dict[str, str]:
    """The opencode-specific env layered under the operator's ``worker_env``.

    Each worker gets its own opencode data dir (``XDG_DATA_HOME``): concurrent
    ``opencode run`` processes sharing one ``opencode.db`` die at startup with
    ``database is locked`` (observed 1/6 at 6-way; upstream #33320/#44892).
    The selected ``provider``'s credential (only that entry -- see
    :func:`resolve_scoped_auth`) is resolved here and forwarded in-memory via
    ``OPENCODE_AUTH_CONTENT`` (opencode 1.x's credential channel; child env
    only, never written by the orchestrator to a sidecar, log or argv).
    opencode 2.x ignores that env var -- :func:`seed_auth_file` writes the
    same scoped credential to ``<XDG_DATA_HOME>/opencode/auth.json`` after
    the launcher's data-dir wipe, which is the only channel 2.x reads.
    uv ignores ``XDG_DATA_HOME`` on Windows, so the worker's toolchain dirs
    are unaffected.
    """
    environ = os.environ if environ is None else environ
    env = {
        **_WORKER_ENV,
        "XDG_DATA_HOME": str(opencode_data_dir(sessions_dir, issue_number)),
    }
    # Operator-declared OPENCODE_AUTH_CONTENT (worker_env / ambient) is parsed
    # and SCOPED to provider rather than passed through or dropped: a reviewer
    # whose operator authenticates through the documented env channel stays
    # authenticated, and the other providers' entries in a multi-provider
    # value cannot ride into the child env.
    scoped_auth = resolve_scoped_auth(worker_env, environ, provider)
    if scoped_auth is not None:  # else opencode may still auth from provider env keys
        env[_AUTH_CONTENT_ENV] = scoped_auth
    # Operator worker_env wins over the posture defaults (same merge order
    # launch_claude_worker applies to worker_env over the sanitised base env).
    env = {**env, **(worker_env or {})}
    if review:
        opencode_review_policy.apply_review_env_pins(
            env,
            base_pins=_WORKER_ENV,
            data_dir=opencode_data_dir(sessions_dir, issue_number),
            scoped_auth=scoped_auth,
        )
    return env


def sanitize_review_command_template(command: tuple[str, ...]) -> tuple[str, ...]:
    """Strip session/agent/server-moving flags and PIN ``--standalone``.

    ``--agent`` selects a permissive agent profile over the config's deny
    policy; 2.x's ``--continue``/``--session``/``--worktree``/``--fork`` move
    the session (and ``--worktree`` the CHECKOUT -- it would defeat
    ``create_review_checkout``'s detached-HEAD review checkout, the invariant
    the detached review is built on); ``--server`` connects the session to a
    server process whose env/config/permissions are NOT this launch's, which
    would bypass the whole env posture the same way the shared background
    service does. No operator template may move any of those boundaries.

    ``--standalone`` is therefore ADDED when absent, not left to the default
    template: the read-only posture is env-only (``OPENCODE_CONFIG_CONTENT``,
    the pinned data dir, the seeded credential), and ``run`` applies that env
    only to a private server it starts itself. Without ``--standalone`` the
    session runs on the operator's shared service, which owns its own env --
    so an operator ``opencode.command`` override would otherwise silently
    downgrade every review to the shared service's posture. Pinning the flag
    here makes that invariant independent of which template won.

    ``--session`` is value-taking like ``--agent``; ``--continue``/``--fork``
    stand alone; ``--worktree``/``-w`` take an OPTIONAL name. ``--server`` is
    value-taking (a URL). No agent is pinned in place of the stripped
    ``--agent``: the deny policy already supplies the read-only invariant, and
    pinning ``plan`` would only matter if agent policies replaced (rather than
    refined) config permissions, in which case it could LOOSEN the bash
    allowlist -- strip-and-pin-standalone is the deterministic choice either
    way.
    """
    command = _strip_flag(command, "--agent", "--session", "-s", "--server")
    out: list[str] = []
    standalone = ("--continue", "--fork", "--worktree", "-c", "-w")
    skip_next_if_flagless = False
    for token in command:
        if skip_next_if_flagless:
            # --worktree takes an OPTIONAL worktree name: a following
            # flagless token IS the name, and leaving it behind would turn
            # it into the positional message (overriding the stdin prompt).
            skip_next_if_flagless = False
            if not token.startswith("-"):
                continue
        if token in standalone or any(token.startswith(f"{flag}=") for flag in standalone):
            if token in ("--worktree", "-w"):
                skip_next_if_flagless = True
            continue
        out.append(token)
    # The pin, not a default: see the docstring. Appended rather than spliced
    # so a template that already carries it is left byte-identical.
    if "--standalone" not in out:
        out.append("--standalone")
    return tuple(out)


def resolve_model(model: str, provider: str) -> str:
    """``provider/model`` for ``--model``; a model already naming a provider passes through."""
    model = model.strip()
    if not model or "/" in model or not provider:
        return model
    return f"{provider}/{model}"


def _strip_flag(command: tuple[str, ...], *flags: str) -> tuple[str, ...]:
    """Drop every occurrence of ``flags`` (bare flag + value, or ``--flag=value``)."""
    out: list[str] = []
    skip_next = False
    for token in command:
        if skip_next:
            skip_next = False
            continue
        if token in flags:
            skip_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in flags):
            continue
        out.append(token)
    return tuple(out)


def pin_flags(command: tuple[str, ...], model: str, variant: str) -> tuple[str, ...]:
    """Last-flag-wins pins: one authoritative ``--model``.

    opencode 2.x has NO ``--variant`` flag (verified against the installed
    2.3.0 binary: ``opencode run --variant x`` prints ``run --help`` and exits
    1 -- the previous pin killed every launch before any request). The
    reasoning-effort variant is expressed INSIDE the model string instead,
    ``provider/model#variant`` (runtime-verified: ``--model fake/m#high``
    launches). A model that already carries ``#<variant>`` (operator passthrough)
    keeps its own; the operator ``--variant`` tokens are stripped either way
    -- on 2.x they are invalid argv, and the resolved variant is owned here.
    """
    pinned = _strip_flag(command, "--model", "-m", "--variant") + (
        ("--model", f"{model}#{variant}") if variant and "#" not in model else ("--model", model)
    )
    return pinned


def launch_opencode_worker(
    issue_number: int,
    branch: str,
    prompt_text: str,
    *,
    repo_root: Path,
    sessions_dir: Path,
    worktrees_dir: Path | None = None,
    venv_source: Path | None = None,
    command_template: tuple[str, ...] | None = None,
    worker_env: dict[str, str] | None = None,
    materialize_dirs: tuple[str, ...] = (),
    rework: bool = False,
    review: bool = False,
    head_sha: str | None = None,
    review_checkouts: Path | None = None,
    legacy_review_checkouts: Path | None = None,
    recovery: dict[str, Any] | None = None,
    base_ref: str = "",
    model_override: str | None = None,
    resolved_review_effort: str | None = None,
    config: OrchestratorConfig | None = None,
) -> ClaudeWorkerRecord:
    """Launch a headless opencode worker or reviewer for ``issue_number``. Never raises.

    The model is ``model_override`` when given (a review launch's selected
    chain entry -- the reviewer role's model lives on ``config.reviewer``, not
    ``config.worker``), else ``config.worker.model`` -- for a worker role-chain
    fallback launch, ``role_selection.worker_config_for`` has already
    substituted the selected entry's model there -- resolved against
    ``config.opencode.provider``. An empty model is refused as an error record
    rather than launched: opencode would otherwise fall back to ambient CLI
    state (its last-used model), the ambient-model outage class
    ``claude_code._apply_model_pin`` guards against.

    On a ``review=True`` launch the command template passes through
    ``sanitize_review_command_template`` (the ``--agent``/``--continue``/
    ``--session``/``--worktree``/``--server`` strip plus the ``--standalone``
    PIN -- opencode's argv-level session, server and permission-boundary
    movers) and the env through ``worker_env_for(review=True)`` (the
    ``_REVIEWER_CONFIG`` deny policy), pairing with ``create_review_checkout``
    exactly as the claude-code plan-mode pin does. ``resolved_review_effort``
    is folded into the ``--model`` pin as ``provider/model#variant`` (opencode
    2.x's only reasoning-effort surface -- there is no ``--variant`` flag;
    empty falls back to ``opencode.variant``); ``max_turns`` has no opencode
    equivalent and is never pinned.

    The data dir is wiped BEFORE the launch (a relaunch must not inherit the
    dead session's db), which also wipes the scoped credential 2.x reads
    from ``<XDG_DATA_HOME>/opencode/auth.json`` -- :func:`seed_auth_file`
    writes it back after the wipe.
    """
    resolved_config = config or OrchestratorConfig()
    opencode = resolved_config.opencode
    model = resolve_model(
        model_override if model_override is not None else resolved_config.worker.model,
        opencode.provider,
    )
    launch_role = "reviewer" if review else "worker"
    if not model:
        record = _error_record(
            issue_number=issue_number,
            branch=branch,
            worktree_path="",
            prompt_path="",
            command=(),
            log_path=str(_log_path(sessions_dir, issue_number, review=review)),
            error=f"opencode {launch_role} launch refused: {launch_role}.model is empty "
            "(no ambient-model fallback)",
            adapter_kind="opencode",
            state_path=launch_events.state_path_for(repo_root, resolved_config),
            role=launch_role,
            error_class=launch_events.LAUNCH_ERR_CONFIG,
        )
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return _write_record(sessions_dir, record)
    provider = model.split("/", 1)[0]
    data_dir = opencode_data_dir(sessions_dir, issue_number)
    # A relaunch (rework, recovery) starts from a fresh data dir; the reap
    # removes it when the worker dies (WorkerView.reap_sidecar).
    shutil.rmtree(data_dir, ignore_errors=True)
    # Seed the 2.x credential channel AFTER the wipe (1.x reads the same
    # scoped credential from OPENCODE_AUTH_CONTENT in the env below).
    environ = os.environ
    seed_auth_file(data_dir, resolve_scoped_auth(worker_env, environ, provider))
    env = worker_env_for(issue_number, sessions_dir, worker_env, provider=provider, review=review)
    template = command_template or opencode.command or DEFAULT_COMMAND_TEMPLATE
    if review:
        template = sanitize_review_command_template(template)
    variant = (resolved_review_effort or "") if review else ""
    variant = variant or opencode.variant
    return launch_claude_worker(
        issue_number,
        branch,
        prompt_text,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        worktrees_dir=worktrees_dir,
        venv_source=venv_source,
        command_template=template,
        env=env,
        materialize_dirs=materialize_dirs,
        rework=rework,
        review=review,
        head_sha=head_sha,
        review_checkouts=review_checkouts,
        legacy_review_checkouts=legacy_review_checkouts,
        recovery=recovery,
        base_ref=base_ref,
        config=resolved_config,
        adapter_kind="opencode",
        provider=provider,
        cli_pins=lambda command: pin_flags(command, model, variant),
    )
