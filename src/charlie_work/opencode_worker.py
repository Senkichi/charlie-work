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
(``_REVIEWER_CONFIG``) and the ``--agent`` strip in
``sanitize_review_command_template`` -- because a reviewer's read-only
posture on this harness lives in env, not argv.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import launch_events
from .claude_code import (
    ClaudeWorkerRecord,
    _error_record,
    _log_path,
    _write_record,
    launch_claude_worker,
)
from .config import OrchestratorConfig
from .opencode_log import opencode_data_dir

# ``opencode run`` with the prompt fed on stdin (no positional message) and raw
# JSON events on stdout -- one object per line, ``type`` at top level, which
# ``opencode_log.provider_error_digest`` parses so the quota/throttle
# classifiers read only opencode's own error records, never tool output. ``--auto``
# approves permission requests that config does not explicitly deny; without it
# ``run`` auto-REJECTS them (external_directory et al.) rather than prompting.
# ``--print-logs --log-level WARN``: on a provider 429 (OpenCode Go usage limit)
# opencode retries internally, honouring retry-after -- hours, for a 5h limit --
# and without these flags the log stays EMPTY the whole time, so the quota
# classifier never sees it. With them, each failed attempt writes a
# ``level=ERROR ... stream error ... AI_APICallError: <provider message>`` line
# to stderr (merged into the log) on the first attempt (verified against a fake
# 429 provider, 2026-10-06). The stall watchdog then reaps the silent retry
# sleep and classifies that tail.
DEFAULT_COMMAND_TEMPLATE: tuple[str, ...] = (
    "opencode",
    "run",
    "--auto",
    "--format",
    "json",
    "--print-logs",
    "--log-level",
    "WARN",
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

# Reviewer posture (V1 ``permission`` object -- installed opencode is 1.x, so
# the V2 ``permissions`` list does not apply). This is the read-only
# counterpart of claude-code's ``--permission-mode plan``: ``edit`` covers
# the edit/write/patch tools, ``bash`` is an exact-prefix allowlist of the
# read-only git/gh verbs the review packet names plus the targeted single
# pytest run the prompt permits, and every other action defaults to deny.
# ``external_directory`` stays allow -- the packet artifacts ($diff_path,
# $pr_json_path) live under .var outside the review checkout. A ``deny``
# surfaces to the model as a tool error, not a session-ending rejection, so
# straying outside the allowlist costs the reviewer a retry, not the review.
_REVIEWER_CONFIG: dict[str, Any] = {
    "$schema": "https://opencode.ai/config.json",
    "permission": {
        "*": "deny",
        "read": "allow",
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "todowrite": "allow",
        "webfetch": "allow",
        "websearch": "allow",
        "external_directory": "allow",
        # Redundant with "*": "deny" but named explicitly: this key covers the
        # edit/write/patch tools and is the whole point of the reviewer policy.
        "edit": "deny",
        "bash": {
            "*": "deny",
            "git status": "allow",
            "git status *": "allow",
            "git diff": "allow",
            "git diff *": "allow",
            "git log": "allow",
            "git log *": "allow",
            "git show": "allow",
            "git show *": "allow",
            "git blame *": "allow",
            "git grep *": "allow",
            "git ls-files *": "allow",
            "git rev-parse *": "allow",
            "git branch --show-current": "allow",
            "git branch --list *": "allow",
            "git remote -v": "allow",
            "git remote show *": "allow",
            "git describe *": "allow",
            "gh pr view": "allow",
            "gh pr view *": "allow",
            "gh pr diff *": "allow",
            "gh pr checks": "allow",
            "gh pr checks *": "allow",
            "gh issue view": "allow",
            "gh issue view *": "allow",
            "gh repo view *": "allow",
            "gh run list": "allow",
            "gh run list *": "allow",
            "gh run view *": "allow",
            "gh search *": "allow",
            "pytest *": "allow",
            "python -m pytest *": "allow",
            "uv run pytest *": "allow",
            "uv run --extra dev pytest *": "allow",
        },
    },
    "autoupdate": False,
    "share": "disabled",
    "snapshot": False,
}

_WORKER_ENV: dict[str, str] = {
    "OPENCODE_CONFIG_CONTENT": json.dumps(_WORKER_CONFIG, separators=(",", ":")),
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
    # Worker hermeticity: opencode injects the operator's personal
    # ~/.claude/CLAUDE.md by default (observed). Its "pause before pushing"
    # style rules are what made swe-1.7 workers decline to push. The repo's own
    # CLAUDE.md / AGENTS.md and .claude/skills still load.
    "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "1",
    # Default bash-tool timeout is 2 minutes; a full local test run exceeds it.
    "OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS": "1800000",
}

_AUTH_CONTENT_ENV = "OPENCODE_AUTH_CONTENT"


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
    ``database is locked`` (observed 1/6 at 6-way; upstream #33320/#44892). A
    fresh data dir has no ``auth.json``, so the selected ``provider``'s host
    credential (only that entry -- see ``_scoped_auth_content``) is forwarded
    in-memory via ``OPENCODE_AUTH_CONTENT``: child env only, never written by
    the orchestrator to a sidecar, log or argv. uv ignores
    ``XDG_DATA_HOME`` on Windows, so the worker's toolchain dirs are unaffected.
    """
    environ = os.environ if environ is None else environ
    env = {
        **_WORKER_ENV,
        "XDG_DATA_HOME": str(opencode_data_dir(sessions_dir, issue_number)),
    }
    if _AUTH_CONTENT_ENV not in environ:
        auth = _scoped_auth_content(_host_auth_path(environ), provider)
        if auth is not None:  # else opencode may still auth from provider env keys
            env[_AUTH_CONTENT_ENV] = auth
    # Operator worker_env wins over the posture defaults (same merge order
    # launch_claude_worker applies to worker_env over the sanitised base env).
    env = {**env, **(worker_env or {})}
    if review:
        # Read-only invariant, the opencode counterpart of the
        # --permission-mode plan pin _sanitize_review_command_template applies
        # for claude-code reviewers: no operator worker_env may grant a
        # reviewer write access, so the deny policy is applied AFTER the
        # merge, overriding any OPENCODE_CONFIG_CONTENT the caller supplied.
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(_REVIEWER_CONFIG, separators=(",", ":"))
    return env


def sanitize_review_command_template(command: tuple[str, ...]) -> tuple[str, ...]:
    """Strip ``--agent`` flags from a reviewer's command template.

    ``--agent`` is the one opencode argv token that can move the permission
    boundary (a ``--agent build``-style pin would evaluate the session against
    that agent's own policy instead of the config's). The rest of the posture
    lives in ``OPENCODE_CONFIG_CONTENT`` (``_REVIEWER_CONFIG``), which no argv
    token overrides -- opencode has no ``--permission-mode``-style flag. No
    agent is pinned in its place: the deny policy already supplies the
    read-only invariant, and pinning ``plan`` would only matter if agent
    policies replaced (rather than refined) config permissions, in which case
    it could LOOSEN the bash allowlist -- strip-only is the deterministic
    choice either way.
    """
    return _strip_flag(command, "--agent")


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
    """Last-flag-wins pins: one authoritative ``--model`` (and ``--variant`` when set)."""
    pinned = _strip_flag(command, "--model", "-m") + ("--model", model)
    if variant:
        pinned = _strip_flag(pinned, "--variant") + ("--variant", variant)
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
    ``sanitize_review_command_template`` (the ``--agent`` strip -- opencode's
    only argv-level permission vector) and the env through
    ``worker_env_for(review=True)`` (the ``_REVIEWER_CONFIG`` deny policy),
    pairing with ``create_review_checkout`` exactly as the claude-code
    plan-mode pin does. ``resolved_review_effort`` is pinned as ``--variant``
    (opencode's reasoning-effort flag; empty falls back to
    ``opencode.variant``); ``max_turns`` has no opencode equivalent and is
    never pinned.
    """
    resolved_config = config or OrchestratorConfig()
    opencode = resolved_config.opencode
    model = resolve_model(
        model_override if model_override is not None else resolved_config.worker.model,
        opencode.provider,
    )
    if not model:
        record = _error_record(
            issue_number=issue_number,
            branch=branch,
            worktree_path="",
            prompt_path="",
            command=(),
            log_path=str(_log_path(sessions_dir, issue_number, review=review)),
            error="opencode worker launch refused: worker.model is empty "
            "(no ambient-model fallback)",
            adapter_kind="opencode",
            state_path=launch_events.state_path_for(repo_root, resolved_config),
            role="reviewer" if review else "worker",
            error_class=launch_events.LAUNCH_ERR_CONFIG,
        )
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return _write_record(sessions_dir, record)
    provider = model.split("/", 1)[0]
    # A relaunch (rework, recovery) starts from a fresh data dir; the reap
    # removes it when the worker dies (WorkerView.reap_sidecar).
    shutil.rmtree(opencode_data_dir(sessions_dir, issue_number), ignore_errors=True)
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
        recovery=recovery,
        base_ref=base_ref,
        config=resolved_config,
        adapter_kind="opencode",
        provider=provider,
        cli_pins=lambda command: pin_flags(command, model, variant),
    )
