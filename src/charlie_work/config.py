from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from charlie_work.github import ORCHESTRATOR_MANAGED_MERGE_FLAGS

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
        # Simulate render to catch malformed placeholders that the regex misses
        # (bare {, unclosed {prompt_path, stray }, positional {0})
        try:
            part.format(**{p: "" for p in allowed_placeholders})
        except (ValueError, KeyError, IndexError) as e:
            raise ConfigError(
                f"config section '{config_key}': malformed placeholder in '{part}': {e}"
            ) from e


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
    # Repo-root-relative paths copied into each worktree after creation
    # (e.g. [".devin"]). Copy-not-link (workers may write marker files);
    # skip-if-tracked (tracked paths are already present). Errors surface as
    # values in SessionRecord.error.
    materialize_dirs: tuple[str, ...] = ()
    # Base ref for fresh worktree creation. Empty string (default) means auto-resolve
    # to origin/<default-branch>. If set, must be a valid git ref (e.g., "origin/main",
    # "HEAD", or a commit SHA). When the resolved base ref is a remote-tracking ref
    # (origin/<branch>), git fetch is run before worktree creation to ensure the
    # worktree bases off the latest remote tip instead of a stale local HEAD.
    base_ref: str = ""
    # Dispatch order: "oldest" (default) selects issues by creation date ascending,
    # "newest" selects by creation date descending (previous behavior).
    order: str = "oldest"


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
    # Merge with `gh pr merge --admin` — required when the base branch is
    # protected (required reviews/checks) and the operator's gh auth has admin
    # on the repo. Without it, protected-main merges bounce to the operator.
    admin: bool = False
    # Extra flags appended to the `gh pr merge` invocation (e.g., ["--admin"]
    # for single-operator repos, ["--auto"] for merge-queue/auto-merge flows).
    # Placeholder-free passthrough, validated to start with "--". Default empty
    # preserves current behavior. Takes precedence over the legacy `admin` field.
    merge_flags: tuple[str, ...] = ()
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
    # Relative to the consumer repo root; junctioned into each worktree so
    # workers share one venv (operator decision 2026-07-01). None disables.
    venv_source: str | None = ".venv"
    # Extra environment variables merged over the orchestrator's env in every
    # devin-shell worker's launch process. Primary use: bound local test
    # parallelism on a shared host — set PYTEST_XDIST_AUTO_NUM_WORKERS to cap
    # `pytest -n auto` at the launch boundary (so K concurrent workers stay
    # near one xdist worker per core instead of oversubscribing into swap)
    # WITHOUT editing the suite's pyproject addopts — CI never sees this var,
    # so it keeps full parallelism. A non-mapping value is rejected with
    # ConfigError at load; values -> str.
    #
    # Merge order: worker_env is merged AFTER sanitize_env, so operator-provided
    # values override sanitized keys (e.g., worker_env={"VIRTUAL_ENV": "/path"}
    # reintroduces VIRTUAL_ENV even though sanitize_env strips it). This is
    # intentional: explicit operator overrides win over sanitization.
    worker_env: dict[str, str] = field(default_factory=dict)


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
    #
    # Merge order: worker_env is merged AFTER sanitize_env, so operator-provided
    # values override sanitized keys (e.g., worker_env={"VIRTUAL_ENV": "/path"}
    # reintroduces VIRTUAL_ENV even though sanitize_env strips it). This is
    # intentional: explicit operator overrides win over sanitization.
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
class WatchdogConfig:
    """Stall watchdog for detecting live-PID-but-dead-agent zombies.

    Detects sessions where the PID is alive but the agent loop has died
    (e.g., tool denial in --print mode leaving child processes holding
    the stdout pipe open). The watchdog checks log file mtime and terminal
    error markers to identify stalled sessions.
    """

    enabled: bool = True
    stall_minutes: int = 20


@dataclass(frozen=True)
class TestAdequacyConfig:
    """Config for the opt-in test-adequacy gate (janitor.check_test_adequacy).

    ``enabled`` defaults False so an absent config block is a no-op — mirrors
    CrossFamilyConfig (config.py:236). Nothing reads this config yet; the
    structural check and review-routing land in separate issues.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    enabled: bool = False
    min_product_lines: int = 10
    test_path_globs: tuple[str, ...] = ("tests/**", "test_*.py", "*_test.py", "conftest.py")
    exempt_path_globs: tuple[str, ...] = ("*.md", "docs/**", "*.lock", "*.toml", "*.cfg", "*.ini")
    assertion_markers: tuple[str, ...] = (
        "assert ",
        "pytest.raises",
        "raises(",
        "assert_called",
        "self.assert",
    )
    comment_prefixes: tuple[str, ...] = ("#",)
    require_assertions: bool = False
    exempt_marker: str = "Test-exempt:"
    # Tier 3 (reserved, deferred): diff-coverage extension, not read by any
    # code path yet.
    coverage_enabled: bool = False
    coverage_command: tuple[str, ...] = ()
    min_diff_coverage: float = 0.0


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
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    test_adequacy: TestAdequacyConfig = field(default_factory=TestAdequacyConfig)


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
    dispatch_data = _section(data, "dispatch")
    materialize_dirs = dispatch_data.get("materialize_dirs")
    if materialize_dirs is not None:
        if not isinstance(materialize_dirs, list):
            raise ConfigError(
                "config section 'dispatch' key 'materialize_dirs' must be a list of "
                f"directory paths, got {type(materialize_dirs).__name__}"
            )
        dispatch_data["materialize_dirs"] = tuple(str(item) for item in materialize_dirs)
    base_ref = dispatch_data.get("base_ref")
    if base_ref is not None and not isinstance(base_ref, str):
        raise ConfigError(
            "config section 'dispatch' key 'base_ref' must be a string, "
            f"got {type(base_ref).__name__}"
        )
    order = dispatch_data.get("order")
    if order is not None and not isinstance(order, str):
        raise ConfigError(
            f"config section 'dispatch' key 'order' must be a string, got {type(order).__name__}"
        )
    if order is not None and order not in ("oldest", "newest"):
        raise ConfigError(
            f"config section 'dispatch' key 'order' must be 'oldest' or 'newest', got '{order}'"
        )
    dispatch = _build_section(DispatchConfig, "dispatch", dispatch_data)
    review = _build_section(ReviewConfig, "review", _section(data, "review"))
    auto_merge_data = _section(data, "auto_merge")
    required_checks = auto_merge_data.get("required_checks")
    if isinstance(required_checks, list):
        auto_merge_data["required_checks"] = tuple(str(item) for item in required_checks)
    merge_flags = auto_merge_data.get("merge_flags")
    if merge_flags is not None and not isinstance(merge_flags, list):
        raise ConfigError(
            "config section 'auto_merge' key 'merge_flags' must be a list of flags, "
            f"got {type(merge_flags).__name__}"
        )
    if isinstance(merge_flags, list):
        auto_merge_data["merge_flags"] = tuple(str(item) for item in merge_flags)
    # Validate merge_flags: each flag must start with "--"
    merge_flags = auto_merge_data.get("merge_flags")
    if merge_flags:
        for flag in merge_flags:
            if not str(flag).startswith("--"):
                raise ConfigError(
                    f"config section 'auto_merge' key 'merge_flags': flag '{flag}' "
                    f"must start with '--'"
                )
        # Reject flags that conflict with orchestrator-managed behavior
        # These are either appended by merge_pr itself (strategy flags) or
        # deliberately excluded (branch deletion is handled separately)
        # Normalize by splitting on '=' to catch --flag=value forms
        for flag in merge_flags:
            flag_name = str(flag).split("=", 1)[0]
            if flag_name in ORCHESTRATOR_MANAGED_MERGE_FLAGS:
                raise ConfigError(
                    f"config section 'auto_merge' key 'merge_flags': flag '{flag}' "
                    f"is managed by the orchestrator and cannot be specified in merge_flags"
                )
    auto_merge = _build_section(AutoMergeConfig, "auto_merge", auto_merge_data)
    runtime = _build_section(RuntimeConfig, "runtime", _section(data, "runtime"))
    devin_data = _section(data, "devin")
    for command_key in ("dispatch_command", "shell_command"):
        command_value = devin_data.get(command_key)
        if isinstance(command_value, list):
            devin_data[command_key] = tuple(str(item) for item in command_value)
    # Validate dispatch_command placeholders (after list->tuple conversion)
    dispatch_command = devin_data.get("dispatch_command")
    if dispatch_command:
        _validate_command_placeholders(
            dispatch_command,
            {"prompt_path", "issue_number", "branch"},
            "devin.dispatch_command",
        )
    # Validate shell_command placeholders (after list->tuple conversion)
    shell_command = devin_data.get("shell_command")
    if shell_command:
        _validate_command_placeholders(
            shell_command,
            {"prompt_path", "issue_number", "branch", "model_args"},
            "devin.shell_command",
        )
    worker_env = devin_data.get("worker_env")
    if worker_env is not None:
        if not isinstance(worker_env, dict):
            raise ConfigError(
                "config section 'devin' key 'worker_env' must be a mapping of "
                f"env-var names to values, got {type(worker_env).__name__}"
            )
        devin_data["worker_env"] = {str(k): str(v) for k, v in worker_env.items()}
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
    watchdog = _build_section(WatchdogConfig, "watchdog", _section(data, "watchdog"))
    test_adequacy_data = _section(data, "test_adequacy")

    # Five tuple-of-str fields: reject non-list, coerce elements to str.
    _TEST_ADEQUACY_TUPLE_FIELDS = (
        "test_path_globs",
        "exempt_path_globs",
        "assertion_markers",
        "comment_prefixes",
        "coverage_command",
    )
    for key in _TEST_ADEQUACY_TUPLE_FIELDS:
        value = test_adequacy_data.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ConfigError(
                f"config section 'test_adequacy' key '{key}' must be a list of "
                f"strings, got {type(value).__name__}"
            )
        for item in value:
            if not isinstance(item, str):
                raise ConfigError(
                    f"config section 'test_adequacy' key '{key}' must be a list of "
                    f"strings, got element of type {type(item).__name__}"
                )
        test_adequacy_data[key] = tuple(value)

    # Scalar fields: isinstance rejection, mirroring base_ref (config.py:326-331).
    min_product_lines = test_adequacy_data.get("min_product_lines")
    if min_product_lines is not None and not isinstance(min_product_lines, int):
        raise ConfigError(
            "config section 'test_adequacy' key 'min_product_lines' must be an "
            f"int, got {type(min_product_lines).__name__}"
        )
    min_diff_coverage = test_adequacy_data.get("min_diff_coverage")
    if min_diff_coverage is not None and not isinstance(min_diff_coverage, (int, float)):
        raise ConfigError(
            "config section 'test_adequacy' key 'min_diff_coverage' must be a "
            f"float, got {type(min_diff_coverage).__name__}"
        )
    exempt_marker = test_adequacy_data.get("exempt_marker")
    if exempt_marker is not None:
        if not isinstance(exempt_marker, str) or not exempt_marker:
            raise ConfigError(
                "config section 'test_adequacy' key 'exempt_marker' must be a non-empty string"
            )
    for bool_key in ("enabled", "coverage_enabled", "require_assertions"):
        bool_value = test_adequacy_data.get(bool_key)
        if bool_value is not None and not isinstance(bool_value, bool):
            raise ConfigError(
                f"config section 'test_adequacy' key '{bool_key}' must be a bool, "
                f"got {type(bool_value).__name__}"
            )

    test_adequacy = _build_section(TestAdequacyConfig, "test_adequacy", test_adequacy_data)
    return OrchestratorConfig(
        labels=labels,
        dispatch=dispatch,
        review=review,
        auto_merge=auto_merge,
        runtime=runtime,
        devin=devin,
        claude_code=claude_code,
        cross_family=cross_family,
        watchdog=watchdog,
        test_adequacy=test_adequacy,
    )
