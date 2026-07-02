from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_FILENAME = "orchestrator.config.yaml"


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
    orchestrator_model_tier: str = "top-reasoning"
    # Package template rendered for worker prompts. "worker.md" targets Devin
    # sessions (skills-based loop); "worker_claude_code.md" targets Claude Code
    # workers (direct shell loop). A repo-local prompts dir overrides by filename.
    worker_template: str = "worker.md"


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
    allow_auto_merge_when_pending: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    state_dir: str = ".var/devin-orchestrator"
    # Repo-local template dir searched before the package defaults. Relative
    # paths resolve against the consumer repo root.
    prompts_dir: str | None = None


@dataclass(frozen=True)
class DevinConfig:
    adapter: str = "manual"
    session_manifest: str = ".var/devin-orchestrator/dispatches/session-manifest.json"
    session_results: str = ".var/devin-orchestrator/dispatches/session-results.json"
    dispatch_command: str | tuple[str, ...] = ""
    command_timeout_seconds: int = 300


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


def load_config(path: Path | None = None) -> OrchestratorConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path and path.exists() else {}
    data = raw if isinstance(raw, dict) else {}
    labels = LabelConfig(**_section(data, "labels"))
    dispatch = DispatchConfig(**_section(data, "dispatch"))
    review = ReviewConfig(**_section(data, "review"))
    auto_merge_data = _section(data, "auto_merge")
    required_checks = auto_merge_data.get("required_checks")
    if isinstance(required_checks, list):
        auto_merge_data["required_checks"] = tuple(str(item) for item in required_checks)
    auto_merge = AutoMergeConfig(**auto_merge_data)
    runtime = RuntimeConfig(**_section(data, "runtime"))
    devin_data = _section(data, "devin")
    dispatch_command = devin_data.get("dispatch_command")
    if isinstance(dispatch_command, list):
        devin_data["dispatch_command"] = tuple(str(item) for item in dispatch_command)
    devin = DevinConfig(**devin_data)
    cross_family_data = _section(data, "cross_family")
    cf_command = cross_family_data.get("command")
    if isinstance(cf_command, list):
        cross_family_data["command"] = tuple(str(item) for item in cf_command)
    cross_family = CrossFamilyConfig(**cross_family_data)
    return OrchestratorConfig(
        labels=labels,
        dispatch=dispatch,
        review=review,
        auto_merge=auto_merge,
        runtime=runtime,
        devin=devin,
        cross_family=cross_family,
    )
