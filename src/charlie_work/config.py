from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_FILENAME = "orchestrator.config.yaml"


class ConfigError(ValueError):
    """A config file was structurally invalid (unknown keys, wrong shapes)."""


def _validate_command_placeholders(
    command: str | tuple[str, ...],
    allowed_placeholders: set[str],
    config_key: str,
) -> None:
    """Validate that a command template uses only allowed placeholders.

    Raises ConfigError if an unknown or malformed placeholder is found.
    """
    # Pattern to match {placeholder} tokens
    placeholder_pattern = re.compile(r"\{([^{}]*)\}")

    parts = command if isinstance(command, tuple) else (command,)
    for part in parts:
        matches = placeholder_pattern.findall(part)
        for match in matches:
            if match == "":
                raise ConfigError(
                    f"config section '{config_key}': empty placeholder {{}} is not allowed"
                )
            if match not in allowed_placeholders:
                raise ConfigError(
                    f"config section '{config_key}': unknown placeholder {{{match}}} "
                    f"(allowed: {', '.join(sorted(allowed_placeholders))})"
                )


@dataclass(frozen=True)
class LabelConfig:
    ready: str = "automated-ready"
    queued: str = "agent:queued"
    in_progress: str = "agent:in-progress"
    pr_open: str = "agent:pr-open"
    reviewing: str = "agent:reviewing"
    needs_rework: str = "agent:needs-rework"
    blocked: str = "agent:blocked"
    done: str = "agent:done"
    human_needed: str = "agent:human-needed"

    @property
    def terminal(self) -> set[str]:
        return {self.blocked, self.done, self.human_needed}

    @property
    def active(self) -> set[str]:
        return {self.queued, self.in_progress, self.pr_open, self.reviewing, self.needs_rework}

    @property
    def all(self) -> list[str]:
        return [
            self.ready,
            self.queued,
            self.in_progress,
            self.pr_open,
            self.reviewing,
            self.needs_rework,
            self.blocked,
            self.done,
            self.human_needed,
        ]


@dataclass(frozen=True)
class DispatchConfig:
    default_limit: int = 3
    branch_prefix: str = "agent/issue"
    worker_model_tier: str = "capable"
    # Package template rendered for worker prompts. "worker.md" targets Devin
    # sessions (skills-based loop); "worker_claude_code.md" targets Claude Code
    # workers (direct shell loop). A repo-local prompts dir overrides by filename.
    worker_template: str = "worker.md"
    # Global concurrency governor: cap total live worker sessions across fresh,
    # rework, and recovery dispatch. Unset/0 preserves current unlimited behavior.
    max_concurrent_sessions: int = 0


@dataclass(frozen=True)
class ReviewConfig:
    # Enforced in record_review: past this many request_changes cycles the PR
    # escalates to a human instead of another rework dispatch. 2 per operator
    # decision (2026-07-01) — iteration past ~2 rounds thrashes.
    max_rework_cycles: int = 2
    require_tests_or_rationale: bool = True
    require_issue_link: bool = True


@dataclass(frozen=True)
class AutoMergeConfig:
    enabled: bool = True
    strategy: str = "squash"
    # Post-merge branch deletion is best-effort and can never abort the
    # merge/label sequence (the empericus local-worktree failure mode).
    delete_branch: bool = True
    require_approved_review: bool = True
    required_checks: tuple[str, ...] = ()
    # After a successful ship-it merge, update remaining open agent PRs
    # (same-repo + configured branch prefix) to rebase them against the
    # new base. Default false to preserve existing behavior; per-PR failures
    # (e.g., conflicts) are reported as values and never abort the merge pass.
    update_open_prs: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    state_dir: str = ".var/charlie-work"
    # Repo-local template dir searched before the package defaults. Relative
    # paths resolve against the consumer repo root.
    prompts_dir: str | None = None


@dataclass(frozen=True)
class DevinConfig:
    # "manual" writes a session manifest for the operator; "command" runs a
    # blocking dispatch_command per issue; "devin-shell" launches headless
    # `devin` CLI sessions non-blocking with sidecar tracking (devin_shell.py);
    # "claude-code" launches Claude Code workers in isolated git worktrees
    # (claude_code.py, configured under the claude_code section).
    adapter: str = "manual"
    session_manifest: str = ".var/charlie-work/dispatches/session-manifest.json"
    session_results: str = ".var/charlie-work/dispatches/session-results.json"
    dispatch_command: str | tuple[str, ...] = ""
    command_timeout_seconds: int = 300
    # devin-shell adapter: sidecar JSON + per-session logs live here.
    sessions_dir: str = ".var/charlie-work/dispatches/sessions"
    # devin-shell launch command; empty means devin_shell.DEFAULT_COMMAND_TEMPLATE.
    # Placeholders: {prompt_path} {issue_number} {branch} {model_args}.
    shell_command: tuple[str, ...] = ()
    # devin-shell worker model; empty string means CLI default. When set,
    # injects "--model <value>" into the rendered command via {model_args}.
    worker_model: str = ""


@dataclass(frozen=True)
class ClaudeCodeConfig:
    """Settings for the claude-code worker adapter (devin.adapter: claude-code)."""

    # Empty means claude_code.DEFAULT_COMMAND_TEMPLATE; the rendered worker
    # prompt is fed via stdin unless the template names {prompt_path}.
    command: tuple[str, ...] = ()
    # None -> worktree.py default (<repo_root>/.var/charlie-work/worktrees).
    worktrees_dir: str | None = None
    # Relative to the consumer repo root; junctioned into each worktree so
    # workers share one venv (operator decision 2026-07-01). None disables.
    venv_source: str | None = ".venv"
    # Extra environment variables merged over the orchestrator's env in every
    # worker's launch process. Primary use: bound local test parallelism on a
    # shared host — set PYTEST_XDIST_AUTO_NUM_WORKERS to cap `pytest -n auto` at
    # the launch boundary (so K concurrent workers stay near one xdist worker
    # per core instead of oversubscribing into swap) WITHOUT editing the suite's
    # pyproject addopts — CI never sees this var, so it keeps full parallelism.
    # See docs/RUNBOOK.md "Local host saturation ceiling (claude-code adapter)".
    # A non-mapping value is rejected with ConfigError at load; values -> str.
    worker_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossFamilyConfig:
    """Auto cross-family (non-Claude) adversarial pass over specs and PRs.

    ``enabled`` defaults False so an absent config block is a no-op. Trivially
    removable: flip ``enabled`` to false (or drop the block) and the
    orchestrator behaves exactly as before.
    """

    enabled: bool = False
    model: str = "codex"
    command: str | tuple[str, ...] = (
        "devin",
        "--model",
        "{model}",
        "-p",
        "--prompt-file",
        "{prompt_path}",
    )
    timeout_seconds: int = 300


@dataclass(frozen=True)
class OrchestratorConfig:
    labels: LabelConfig = field(default_factory=LabelConfig)
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    auto_merge: AutoMergeConfig = field(default_factory=AutoMergeConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    devin: DevinConfig = field(default_factory=DevinConfig)
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)
    cross_family: CrossFamilyConfig = field(default_factory=CrossFamilyConfig)


def find_config_path(repo_root: Path, explicit: Path | None = None) -> Path | None:
    """Resolve the config file: an explicit path wins; otherwise the consumer
    repo's root-level ``orchestrator.config.yaml`` if present; otherwise None
    (pure dataclass defaults)."""
    if explicit is not None:
        return explicit
    candidate = repo_root / DEFAULT_CONFIG_FILENAME
    return candidate if candidate.exists() else None


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def _build_section(cls: type, name: str, data: dict[str, Any]) -> Any:
    """Construct a config dataclass, turning unknown YAML keys into a readable
    error (a bare ``TypeError`` from ``cls(**data)`` names neither the section
    nor the valid keys — hostile to consumers mid-migration)."""
    valid = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in config section '{name}': {', '.join(unknown)} "
            f"(valid: {', '.join(sorted(valid))})"
        )
    return cls(**data)


def load_config(path: Path | None = None) -> OrchestratorConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path and path.exists() else {}
    data = raw if isinstance(raw, dict) else {}
    # Validate top-level keys before processing sections
    known_sections = {f.name for f in fields(OrchestratorConfig)}
    unknown = sorted(set(data) - known_sections)
    if unknown:
        raise ConfigError(
            f"unknown config section(s): {', '.join(unknown)} "
            f"(valid: {', '.join(sorted(known_sections))})"
        )
    labels = _build_section(LabelConfig, "labels", _section(data, "labels"))
    dispatch = _build_section(DispatchConfig, "dispatch", _section(data, "dispatch"))
    review = _build_section(ReviewConfig, "review", _section(data, "review"))
    auto_merge_data = _section(data, "auto_merge")
    required_checks = auto_merge_data.get("required_checks")
    if isinstance(required_checks, list):
        auto_merge_data["required_checks"] = tuple(str(item) for item in required_checks)
    auto_merge = _build_section(AutoMergeConfig, "auto_merge", auto_merge_data)
    runtime = _build_section(RuntimeConfig, "runtime", _section(data, "runtime"))
    devin_data = _section(data, "devin")
    for command_key in ("dispatch_command", "shell_command"):
        command_value = devin_data.get(command_key)
        if isinstance(command_value, list):
            devin_data[command_key] = tuple(str(item) for item in command_value)
    # Validate shell_command placeholders (after list->tuple conversion)
    shell_command = devin_data.get("shell_command")
    if shell_command:
        _validate_command_placeholders(
            shell_command,
            {"prompt_path", "issue_number", "branch", "model_args"},
            "devin.shell_command",
        )
    devin = _build_section(DevinConfig, "devin", devin_data)
    claude_code_data = _section(data, "claude_code")
    claude_command = claude_code_data.get("command")
    if isinstance(claude_command, list):
        claude_code_data["command"] = tuple(str(item) for item in claude_command)
    # Validate claude_code.command placeholders
    claude_command = claude_code_data.get("command")
    if claude_command:
        _validate_command_placeholders(
            claude_command,
            {"prompt_path", "issue_number", "branch"},
            "claude_code.command",
        )
    worker_env = claude_code_data.get("worker_env")
    if worker_env is not None:
        if not isinstance(worker_env, dict):
            raise ConfigError(
                "config section 'claude_code' key 'worker_env' must be a mapping of "
                f"env-var names to values, got {type(worker_env).__name__}"
            )
        claude_code_data["worker_env"] = {str(k): str(v) for k, v in worker_env.items()}
    claude_code = _build_section(ClaudeCodeConfig, "claude_code", claude_code_data)
    cross_family_data = _section(data, "cross_family")
    cf_command = cross_family_data.get("command")
    if isinstance(cf_command, list):
        cross_family_data["command"] = tuple(str(item) for item in cf_command)
    # Validate cross_family.command placeholders
    cf_command = cross_family_data.get("command")
    if cf_command:
        _validate_command_placeholders(
            cf_command,
            {"prompt_path", "issue_number", "branch", "model"},
            "cross_family.command",
        )
    cross_family = _build_section(CrossFamilyConfig, "cross_family", cross_family_data)
    return OrchestratorConfig(
        labels=labels,
        dispatch=dispatch,
        review=review,
        auto_merge=auto_merge,
        runtime=runtime,
        devin=devin,
        claude_code=claude_code,
        cross_family=cross_family,
    )
