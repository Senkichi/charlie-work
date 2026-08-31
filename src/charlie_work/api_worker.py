"""API worker adapter — Claude Code CLI powered by an Anthropic-compatible endpoint.

Third worker adapter kind (``api``): the Claude Code CLI launched with provider
environment injected so any Anthropic-compatible endpoint (Kimi K3 via Moonshot,
etc.) powers the session. The launch/supervision stack is delegated to
``claude_code.launch_claude_worker`` (parameterized with ``adapter_kind="api"``
so sidecars land as ``issue-<n>.api.json``); this module owns only the
provider-registry resolution, env construction, and the never-raise / no-key-
material invariants that are specific to paid-API routing.

Design: ``docs/design/api-worker-adapter.md``. The provider registry and budget
caps live on ``config.ApiWorkerConfig`` (added in #475); the
``launch_claude_worker`` ``adapter_kind``/``provider`` parameters were added in
#476. This issue (#478) delivers the adapter module and worker-view wiring.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .claude_code import ClaudeWorkerRecord, launch_claude_worker
from .config import ApiWorkerConfig, OrchestratorConfig
from .paths import runtime_paths

# Provider env injected into the child process. The auth token travels ONLY in
# the child process env — it is never written to a sidecar, log, prompt, or
# command argv (enforced by test_api_worker_no_key_material_in_sidecar_or_record).
#
# ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN are the standard pair for routing
# Claude Code at an Anthropic-compatible endpoint (vs ANTHROPIC_API_KEY, which
# targets the official Anthropic API). ANTHROPIC_MODEL pins the main session
# model so the CLI does not fall back to ambient /model state.
#
# The small/fast-model slot is pinned to the SAME configured model so no
# auxiliary call (built-in Explore/claude-code-guide agents, planning/summarizing
# steps) silently routes to Anthropic's Haiku against the configured endpoint.
# Both the current canonical name (ANTHROPIC_DEFAULT_HAIKU_MODEL, v2.1.2+) and
# the deprecated-but-still-functional name (ANTHROPIC_SMALL_FAST_MODEL) are set
# so the override holds regardless of the deployed CLI version.
# Sources: https://code.claude.com/docs/en/model-config (env var table);
# https://github.com/anthropics/claude-code/issues/17844 (deprecation note).
_SMALL_FAST_MODEL_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
)


def _provider_env(base_url: str, auth_token: str, model: str) -> dict[str, str]:
    """Build the provider env dict injected into the child process.

    The auth token value is placed ONLY in this dict, which is merged into the
    child process env by ``launch_claude_worker``; it never reaches a sidecar,
    log, prompt, or argv.
    """
    env: dict[str, str] = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": auth_token,
        "ANTHROPIC_MODEL": model,
    }
    # Pin the small/fast-model slot to the same configured model so no
    # auxiliary call escapes to Anthropic's Haiku against the custom endpoint.
    for var in _SMALL_FAST_MODEL_ENV_VARS:
        env[var] = model
    return env


def _budget_exhausted_error(api_worker_config: ApiWorkerConfig, *, state_dir: Path) -> str | None:
    """Return an error message when the api-worker budget is exhausted, else None.

    The refusal gate for the daily/lifetime caps (issue #1514): mirrors the
    headroom check ``doctor.py`` / ``fleet_dispatch.py`` surface, but as a
    launch refusal rather than a report. Uses ``budget_status`` +
    ``load_ledger`` / ``ledger_path`` exactly as those reporting-only
    consumers do — no re-implementation of the headroom math.

    A missing or corrupt ledger is treated as empty spend (``load_ledger``
    quarantines a corrupt file and returns an empty ``Ledger``), so a fresh
    or unreadable ledger never refuses a launch — only real recorded spend
    against the configured caps does.
    """
    from datetime import UTC, datetime

    from .api_budget import budget_status, ledger_path, load_ledger

    ledger = load_ledger(ledger_path(state_dir))
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    status = budget_status(ledger, api_worker_config.budget, today)
    if status.daily_headroom and status.lifetime_headroom:
        return None
    budget = api_worker_config.budget
    parts: list[str] = []
    if not status.daily_headroom:
        parts.append(
            f"daily ${status.spent_today_usd:.2f}/${budget.max_usd_per_day:.2f} exhausted"
        )
    if not status.lifetime_headroom:
        parts.append(
            f"lifetime ${status.lifetime_spent_usd:.2f}/${budget.lifetime_usd:.2f} exhausted"
        )
    return "api worker budget exhausted: " + "; ".join(parts)


def launch_api_worker(
    issue_number: int,
    branch: str,
    prompt_text: str,
    *,
    repo_root: Path,
    sessions_dir: Path,
    api_worker_config: ApiWorkerConfig,
    worktrees_dir: Path | None = None,
    venv_source: Path | None = None,
    command_template: tuple[str, ...] | None = None,
    worker_env: dict[str, str] | None = None,
    materialize_dirs: tuple[str, ...] = (),
    rework: bool = False,
    recovery: dict[str, Any] | None = None,
    base_ref: str = "",
    config: OrchestratorConfig | None = None,
) -> ClaudeWorkerRecord:
    """Resolve the active API provider, build its env, and delegate to
    ``claude_code.launch_claude_worker`` with ``adapter_kind="api"``.

    Never raises: a missing key env var, an unknown/disabled provider, or any
    launch failure comes back as a ``ClaudeWorkerRecord`` with ``.error`` set
    (errors as values; CLAUDE.md invariant). The API key value travels only in
    the child process env — it is never written to a sidecar, log, prompt, or
    command argv.

    ``tee_stream_json`` is force-enabled for api sessions (the budget ledger
    depends on events.jsonl), regardless of any caller-supplied preference.

    Args:
        api_worker_config: The resolved ``ApiWorkerConfig`` registry section.
            ``enabled`` must be True and ``provider`` must name a key in
            ``providers``; both are validated at config load, but this function
            re-checks defensively so a caller that constructs the config
            directly (e.g. a test) still gets an error record rather than a
            ``KeyError``.
    """
    # Defensive re-check (config load already validates enabled+provider, but a
    # directly-constructed config or a stale registry must not raise here).
    if not api_worker_config.enabled:
        return _error_record(
            issue_number,
            branch,
            sessions_dir,
            error="api_worker.enabled is false; cannot launch api worker",
        )
    provider_name = api_worker_config.provider
    provider = api_worker_config.providers.get(provider_name)
    if provider is None:
        return _error_record(
            issue_number,
            branch,
            sessions_dir,
            error=(
                f"api_worker.provider {provider_name!r} is not in "
                f"api_worker.providers; cannot launch api worker"
            ),
            # The configured provider name is known even though resolution
            # failed; carry it so the sidecar/record identify which provider
            # was attempted (useful for triage without leaking key material).
            provider=provider_name,
        )

    # Resolve the auth token from the named env var. The key VALUE never enters
    # config, a sidecar, or argv — only the env var NAME does. A missing var is
    # an error value, not a raise.
    auth_token = os.environ.get(provider.api_key_env)
    if not auth_token:
        return _error_record(
            issue_number,
            branch,
            sessions_dir,
            error=(
                f"environment variable {provider.api_key_env!r} (api_key_env for "
                f"provider {provider_name!r}) is not set or empty; cannot launch "
                f"api worker"
            ),
            provider=provider_name,
        )

    # Budget preflight (issue #1514): refuse the launch when the daily or
    # lifetime cap is exhausted. This is the refusal gate that lived in
    # routing._api_preflight before routing.py was deleted in Phase 2 Track B
    # (PR #1517); without it the caps in api_budget.budget_status were computed
    # and displayed (doctor.py, fleet_dispatch.py) but nothing refused a launch
    # when they ran out. Uses budget_status + load_ledger/ledger_path exactly as
    # those reporting-only consumers do. The ledger lives under the runtime
    # state_dir, resolved from ``config`` (always passed in the production
    # dispatch path). When ``config`` is absent the preflight is skipped — a
    # directly-constructed config without a state_dir cannot be located, the
    # same defensive posture as the enabled/provider/auth-token checks above.
    if config is not None:
        state_root = runtime_paths(repo_root, config.runtime.state_dir).root
        budget_error = _budget_exhausted_error(api_worker_config, state_dir=state_root)
        if budget_error is not None:
            # Emit a best-effort event so the operator can see launches are
            # being held by the budget (log_event is best-effort: any I/O error
            # is caught and logged, never breaking the launch path).
            from . import layout as _layout
            from .instrumentation import log_event

            log_event(
                _layout.state_file_path(state_root),
                "api_budget_refused",
                {
                    "issue_number": issue_number,
                    "branch": branch,
                    "provider": provider_name,
                    "error": budget_error,
                },
                level="warning",
            )
            return _error_record(
                issue_number,
                branch,
                sessions_dir,
                error=budget_error,
                provider=provider_name,
            )

    provider_env = _provider_env(provider.base_url, auth_token, provider.model)
    # Merge provider env OVER any configured worker env so the provider routing
    # vars (base url / token / model) can never be silently overridden by an
    # operator's worker_env entry. launch_claude_worker then merges this combined
    # env over sanitize_env(worktree.path), so operator worker_env values that
    # do not collide with provider routing still apply.
    merged_env: dict[str, str] = {}
    if worker_env:
        merged_env.update({str(k): str(v) for k, v in worker_env.items()})
    merged_env.update(provider_env)

    try:
        return launch_claude_worker(
            issue_number,
            branch,
            prompt_text,
            repo_root=repo_root,
            sessions_dir=sessions_dir,
            worktrees_dir=worktrees_dir,
            venv_source=venv_source,
            command_template=command_template,
            env=merged_env,
            materialize_dirs=materialize_dirs,
            rework=rework,
            recovery=recovery,
            base_ref=base_ref,
            # Force-enabled: the budget ledger depends on events.jsonl.
            tee_stream_json=True,
            config=config,
            adapter_kind="api",
            provider=provider_name,
            # Issue #1245: pin the provider's model as the ``--model`` flag.
            # The Claude Code CLI gives ``--model`` precedence over the
            # ``ANTHROPIC_MODEL`` env var _provider_env also injects, so
            # without this the claude_code section's model would win and the
            # provider's model selection would be dead argv-side. The env
            # injection stays as belt-and-suspenders (it also covers
            # auxiliary small-fast-model calls).
            model_override=provider.model,
        )
    except Exception as exc:
        # launch_claude_worker itself never raises, but a future regression or a
        # failure in this module's own plumbing must still come back as a value.
        return _error_record(
            issue_number,
            branch,
            sessions_dir,
            error=f"api worker launch failed: {exc}",
            provider=provider_name,
        )


def _error_record(
    issue_number: int,
    branch: str,
    sessions_dir: Path,
    *,
    error: str,
    provider: str = "",
) -> ClaudeWorkerRecord:
    """Write an error sidecar (issue-<n>.api.json) and return the record.

    Mirrors claude_code._error_record's never-raise + atomic-write contract.
    The sidecar carries no key material: only the error string, which is
    constructed from the env var NAME (never the value) and the provider name.
    ``provider`` is the configured provider name when known (carried through so
    the sidecar/record identify which provider was attempted); empty for the
    disabled-config path where no provider was resolved.
    """
    from .claude_code import _error_record as _claude_error_record

    record = _claude_error_record(
        issue_number=issue_number,
        branch=branch,
        worktree_path="",
        prompt_path="",
        command=(),
        log_path=str(sessions_dir / f"issue-{issue_number}.claude.log"),
        error=error,
        adapter_kind="api",
        provider=provider,
    )
    # Write the sidecar atomically via the shared claude_code helper so the
    # error is durable even when the failure happened before launch_claude_worker
    # got to write its own sidecar.
    from .claude_code import _write_record

    sessions_dir.mkdir(parents=True, exist_ok=True)
    return _write_record(sessions_dir, record)


__all__ = ["launch_api_worker"]
