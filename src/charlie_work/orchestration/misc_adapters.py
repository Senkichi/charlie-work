"""Adapter-settings builders moved out of ``OrchestratorApp`` (Track 2 Phase B, L03).

Bodies relocated verbatim from ``charlie_work.workflow`` per the delegation plan
(design doc ``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). ``workflow_delegation._install_delegates`` re-attaches each
top-level ``def`` here onto ``OrchestratorApp`` unwrapped, so ``self`` binds via
the descriptor protocol exactly as the lexical methods did.
"""

from __future__ import annotations

from dataclasses import replace as dataclasses_replace

from charlie_work.adapters import AdapterSettings


def _adapter_settings(self, *, adapter: str | None = None) -> AdapterSettings:
    claude = self.config.claude_code
    devin = self.config.devin
    worker = self.config.worker
    api_worker = self.config.api_worker
    resolved_adapter = adapter if adapter is not None else worker.harness
    # Use adapter-specific venv_source and worker_env
    if resolved_adapter == "devin-shell":
        venv_source = self._resolve(devin.venv_source) if devin.venv_source else None
        worker_env = devin.worker_env
    elif resolved_adapter == "claude-code":
        venv_source = self._resolve(claude.venv_source) if claude.venv_source else None
        worker_env = claude.worker_env
    elif resolved_adapter == "api":
        # api workers are Claude Code CLI processes with provider env
        # injected, so they reuse the claude-code venv/env resolution
        # (shared venv junction, worker_env overrides). The provider
        # routing vars (ANTHROPIC_BASE_URL/AUTH_TOKEN/MODEL) are merged
        # inside launch_api_worker, over any worker_env values, so an
        # operator's worker_env cannot accidentally override the provider.
        venv_source = self._resolve(claude.venv_source) if claude.venv_source else None
        worker_env = claude.worker_env
    else:
        venv_source = None
        worker_env = {}
    return AdapterSettings(
        adapter=resolved_adapter,
        dispatch_command=devin.dispatch_command,
        command_timeout_seconds=devin.command_timeout_seconds,
        sessions_dir=self._layout.sessions_dir,
        shell_command=devin.shell_command,
        claude_command=claude.command,
        worktrees_dir=self._layout.worktrees,
        venv_source=venv_source,
        worker_env=worker_env,
        # worker.model is only meaningful for the harness it was resolved
        # against (worker.harness). When `adapter` overrides the
        # configured harness -- a fallback to a different adapter --
        # fall back to devin.worker_model instead: role-config Phase 2 (Track E)
        # deleted the dual-accept bridge that used to mirror worker.model
        # onto devin.worker_model, so the two are independent config
        # values again. devin.worker_model is now the dedicated,
        # separately-configured model for a devin-shell fallback launch
        # (set it explicitly if a routed fallback should pin a model);
        # it is never overwritten by worker.model regardless of what the
        # primary configured harness is.
        worker_model=(worker.model if resolved_adapter == worker.harness else devin.worker_model),
        materialize_dirs=self.config.dispatch.materialize_dirs,
        dry_run=self.dry_run,
        base_ref=self.config.dispatch.base_ref,
        tee_stream_json=claude.tee_stream_json,
        launch_stagger_seconds=self.config.dispatch.launch_stagger_seconds,
        api_worker_config=api_worker if resolved_adapter == "api" else None,
        config=self.config,
    )


def _rescue_adapter_settings(self) -> AdapterSettings:
    """AdapterSettings for a rescue-tier rework dispatch (issue #555).

    Mirrors the "claude-code" branch of ``_adapter_settings()`` exactly
    (same venv/worker_env/command resolution), but forces
    ``adapter="claude-code"`` regardless of the primary configured
    ``worker.harness`` — the rescue tier always uses the claude-code
    adapter — and overrides ``worker.model`` to ``rescue.worker_model``
    via a one-off config copy. This is the "adapter/model overridden
    from RescueConfig" the rescue rework reuses the existing
    rework-dispatch path with, never a parallel launch path.

    Role-config Phase 1.5: ``launch_claude_worker``'s single model-pin
    enforcement point now reads ``resolved_config.worker.model`` (not
    ``resolved_config.claude_code.model``) whenever no ``model_override``
    is passed -- and this call site passes none, same as the primary
    claude-code dispatch path. The override must therefore land on
    ``worker.model``, not ``claude_code.model``, or the rescue tier would
    silently launch with the ordinary worker model instead of the
    stronger rescue model.
    """
    claude = self.config.claude_code
    rescue_config = dataclasses_replace(
        self.config,
        worker=dataclasses_replace(self.config.worker, model=self.config.rescue.worker_model),
    )
    return AdapterSettings(
        adapter="claude-code",
        sessions_dir=self._layout.sessions_dir,
        claude_command=claude.command,
        worktrees_dir=self._layout.worktrees,
        venv_source=self._resolve(claude.venv_source) if claude.venv_source else None,
        worker_env=claude.worker_env,
        materialize_dirs=self.config.dispatch.materialize_dirs,
        dry_run=self.dry_run,
        base_ref=self.config.dispatch.base_ref,
        tee_stream_json=claude.tee_stream_json,
        launch_stagger_seconds=self.config.dispatch.launch_stagger_seconds,
        config=rescue_config,
    )
