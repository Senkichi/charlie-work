from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from charlie_work.github import ORCHESTRATOR_MANAGED_MERGE_FLAGS

DEFAULT_CONFIG_FILENAME = "orchestrator.config.yaml"

# Root-relative path the Claude Code adapter writes to in each worktree.
CLAUDE_CODE_PROMPT_FILENAME = ".orchestrator-prompt.md"

# Worktree writer marker used to enforce single-writer-per-branch (issue #400).
WRITER_MARKER_FILENAME = ".charlie-writer.json"


def _normalize_injected_paths(paths: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return path strings with Windows backslash separators normalized to '/'.

    Git reports worktree paths with forward slashes even on Windows hosts, while
    YAML/config overrides may contain backslashes if the operator writes them
    unquoted. Normalizing once at the config boundary makes matching in
    ``_worker_authored_dirty`` independent of how the separator was encoded.
    """
    return tuple(str(p).replace("\\", "/") for p in paths)


DETERMINISTIC_ESCALATION_FAILURE_KINDS: frozenset[str] = frozenset(
    {"worker_blocked", "worktree_unsafe", "rework_branch_conflict"}
)
# Deliberately excluded: "worktree_probe_failed" (see worktree.WorktreeProbeFailedError).
# A failed safety probe (e.g. git status --porcelain hitting an index lock) is
# transient contention, not a confirmed-dirty worktree — it must take the
# ordinary redispatch-cap path instead of escalating on first occurrence
# (issue #288 follow-up, PR #314).


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
    prose_only_deps: str = "agent:prose-only-deps"

    @property
    def terminal(self) -> set[str]:
        return {self.blocked, self.done, self.human_needed, self.prose_only_deps}

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
            self.prose_only_deps,
        ]

    @property
    def workflow_labels(self) -> set[str]:
        """All workflow labels (agent:* states) excluding the ready marker."""
        return {
            self.queued,
            self.in_progress,
            self.pr_open,
            self.reviewing,
            self.needs_rework,
            self.blocked,
            self.done,
            self.human_needed,
        }


@dataclass(frozen=True)
class DispatchConfig:
    default_limit: int = 3
    branch_prefix: str = "agent/issue"
    worker_model_tier: str = "capable"
    # Package template rendered for worker prompts. "worker.md" targets Devin
    # sessions (skills-based loop); "worker_claude_code.md" targets Claude Code
    # workers (direct shell loop). A repo-local prompts dir overrides by filename.
    worker_template: str = "worker.md"
    # Package template rendered for rework prompts. Mirrors worker_template so the
    # orchestrator has a single source of truth for the rework prompt filename.
    rework_template: str = "rework.md"
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
    # Seconds to sleep between consecutive worker-session launches within a
    # single dispatch pass (fresh or rework lane). Bursting several launches
    # back-to-back can trip a provider's message rate limit (observed:
    # Devin's "overall message rate limit" firing when 3 sessions launched
    # within 6 seconds, killing all three instantly). 0 disables the stagger.
    launch_stagger_seconds: int = 45
    # Per-pass cap for merge-finalization of merged-PR-referenced ready issues
    # (label transition + close). A large backlog of closed issues carrying a
    # stale ready marker cannot monopolize a pass. 0 disables finalization.
    finalize_limit: int = 25
    # Worktree-relative paths owned by the orchestrator and excluded from
    # "is the worktree dirty?" checks. By default the Claude Code adapter's
    # in-worktree prompt file and the per-worktree writer marker are excluded.
    # The Devin shell adapter writes rendered prompts outside the worktree, and no
    # orchestrator code copies them into ``.devin/prompts/...`` by default;
    # operators whose config materializes such a directory or whose worker
    # writes prompt files back into the worktree must set ``injected_paths``
    # explicitly. Paths are normalized to forward slashes so Windows-style
    # backslash separators in config still match git-reported paths.
    injected_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Normalize to a tuple of forward-slash strings. The writer marker is
        # always excluded so it can never be mistaken for worker-authored work.
        if self.injected_paths:
            base = list(self.injected_paths)
        else:
            base = [CLAUDE_CODE_PROMPT_FILENAME]
        if WRITER_MARKER_FILENAME not in base:
            base.append(WRITER_MARKER_FILENAME)
        object.__setattr__(self, "injected_paths", _normalize_injected_paths(base))


@dataclass(frozen=True)
class ReviewConfig:
    # Enforced in record_review: past this many request_changes cycles the PR
    # escalates to a human instead of another rework dispatch. 2 per operator
    # decision (2026-07-01) — iteration past ~2 rounds thrashes.
    max_rework_cycles: int = 2
    require_tests_or_rationale: bool = True
    require_issue_link: bool = True


@dataclass(frozen=True)
class ReviewDispatchConfig:
    # Issue #370: concurrent reviewer launcher for queued PRs. This is a
    # deterministic loop stage, not a provider governor; reviewers use
    # launch_claude_worker with no concurrency clamp for rate-limit reasons.
    enabled: bool = False
    # Per-PR review sidecar + log directory. MUST be distinct from
    # devin.sessions_dir so worker concurrency accounting is not poisoned by
    # reviewer processes.
    reviews_dir: str = ".var/charlie-work/dispatches/reviews"
    # Local-only process bound. 0 means unlimited; raise this only if local
    # CPU/disk from concurrent reviewer worktrees becomes a visible bottleneck.
    # Default is 2 so a host that enables review_dispatch without overriding
    # this key does not run an unbounded number of local Claude Code reviewers.
    max_local_review_processes: int = 2
    # Provider-token budget slots. Limits how many reviewers can be in flight
    # simultaneously against the Claude usage budget. When a slot frees (a
    # reviewer finishes), the next poll dispatches another. 0 means unlimited.
    max_concurrent_reviews: int = 3
    # Fixed interval between quota-probe attempts after a reviewer launch hits
    # the usage wall. A probe is a single reviewer launch; this many minutes
    # must elapse before the next probe. No escalation backoff.
    quota_probe_interval_minutes: int = 15
    # Approximate provider usage-limit reset window in hours. When a reviewer
    # launch hits the wall, the global reviewer quota is held exhausted for at
    # least this long while probes run every ``quota_probe_interval_minutes``.
    quota_reset_hours: int = 5
    # Issue #524: review-lane sibling of watchdog.stall_minutes. 0 disables
    # stall detection; a live reviewer whose sidecar log is idle for longer
    # than this is killed and its claim released so the review slot is freed.
    stall_minutes: int = 0
    # Issue #524: consecutive stall kills before the PR is escalated to
    # agent:human-needed instead of kill-relaunch looping. Only consulted when
    # ``stall_minutes`` is > 0.
    max_stall_attempts: int = 3


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
    # After this many consecutive approved-but-unmergeable passes, emit a
    # merge_failed_attempt_alarm event and warning. 0 disables the alarm.
    failed_attempt_alarm: int = 3
    # Strategy controlling which open agent PRs are rebased after a
    # successful ship-it merge.
    #
    # - "front_of_train" (default): merge-train mode — only update the head
    #   of the approved queue, so a single merge step causes at most one CI
    #   reset on a single-runner train.
    # - "broadcast": update every eligible open tracked PR branch. Intended for
    #   multi-runner setups where parallel CI runs are available. PRs whose
    #   current review decision is request_changes, escalated, or blocked are
    #   still skipped.
    # - "off": never update open PR branches.
    update_branch_strategy: str = "front_of_train"
    # Legacy alias for update_branch_strategy. Kept for backward compatibility.
    # When set, it is normalized and mapped to update_branch_strategy.
    #   true / "all"  -> "broadcast"
    #   "next"        -> "front_of_train"
    #   false / "off" -> "off"
    update_open_prs: str | bool | None = None
    # When True, merge_ready verifies that a PR's merge-base is the current
    # tip of its base branch before merging. If the base has moved ahead of the
    # PR (e.g. a prior merge in the same train), the merge is deferred and a
    # merge_deferred_stale_base event is recorded. Operators may set False to
    # restore the legacy behavior that trusts mergeStateStatus only.
    require_current_base: bool = True
    # Aviator MergeQueue handoff (task #10). When set, an approved+green PR is
    # labeled with this string INSTEAD of being self-merged by the
    # orchestrator; Aviator's queue picks up the label and merges
    # asynchronously. State records status="mergequeue" (never "merged") so
    # the merge_ready idempotency short-circuit does not fire; the existing
    # reconcile.py merged_outside_orchestrator drift path already reconciles
    # status to "merged" (+ label transition) once GitHub reports the PR
    # merged, so no new post-merge bookkeeping is added here. Default None
    # preserves today's self-merge behavior byte-for-byte.
    mergequeue_label: str | None = None

    def __post_init__(self) -> None:
        legacy_to_strategy = {
            "next": "front_of_train",
            "all": "broadcast",
            "off": "off",
        }
        strategy_to_legacy = {
            "front_of_train": "next",
            "broadcast": "all",
            "off": "off",
        }

        # If the legacy alias is set, normalize it and derive the canonical
        # strategy from it. This preserves existing config files and tests.
        raw_legacy = self.update_open_prs
        if raw_legacy is not None:
            if isinstance(raw_legacy, bool):
                legacy_value = "all" if raw_legacy else "off"
            elif isinstance(raw_legacy, str):
                legacy_value = raw_legacy.lower()
                if legacy_value not in legacy_to_strategy:
                    raise ConfigError(
                        "config section 'auto_merge' key 'update_open_prs' must be "
                        "'all', 'next', 'off', or a boolean, "
                        f"got {raw_legacy!r}"
                    )
            else:
                raise ConfigError(
                    "config section 'auto_merge' key 'update_open_prs' must be "
                    "a string or boolean, "
                    f"got {type(raw_legacy).__name__}"
                )
            object.__setattr__(self, "update_open_prs", legacy_value)
            object.__setattr__(self, "update_branch_strategy", legacy_to_strategy[legacy_value])
        else:
            raw_strategy = self.update_branch_strategy
            if isinstance(raw_strategy, bool):
                strategy = "broadcast" if raw_strategy else "off"
            elif isinstance(raw_strategy, str):
                strategy = raw_strategy.lower()
            else:
                raise ConfigError(
                    "config section 'auto_merge' key 'update_branch_strategy' must be "
                    f"a string or boolean, got {type(raw_strategy).__name__}"
                )
            if strategy not in strategy_to_legacy:
                raise ConfigError(
                    "config section 'auto_merge' key 'update_branch_strategy' must be "
                    f"'front_of_train', 'broadcast', or 'off', got {self.update_branch_strategy!r}"
                )
            object.__setattr__(self, "update_branch_strategy", strategy)
            object.__setattr__(self, "update_open_prs", strategy_to_legacy[strategy])

        if self.require_current_base and self.update_branch_strategy == "off":
            raise ConfigError(
                "config section 'auto_merge': require_current_base=True with "
                "update_branch_strategy='off' creates a permanent merge deadlock: the base "
                "must be current but the branch is never synced. Set "
                "require_current_base: false, or set update_branch_strategy to 'front_of_train' or 'broadcast'."
            )


@dataclass(frozen=True)
class RuntimeConfig:
    state_dir: str = ".var/charlie-work"
    # Repo-local template dir searched before the package defaults. Relative
    # paths resolve against the consumer repo root.
    prompts_dir: str | None = None
    # Literal substrings matched against the last 2KB of worker logs to detect
    # genuine provider rate-limit/throttle conditions (retryable after a
    # cooldown). Extend via config instead of editing code.
    #
    # Issue #260 (corrected premise): "A tool was rejected by the user" was
    # originally included here, but it is the Devin CLI's own surfacing of a
    # PreToolUse hook block (a hard, non-transient failure), not a provider
    # throttle condition — see PostMortemConfig.signature_rules'
    # worker_blocked rule, which owns that signature instead. Do not re-add
    # it here: retry semantics (throttled_until, hot redispatch after
    # cooldown) are wrong for a hook block.
    throttle_error_markers: tuple[str, ...] = (
        "Reached overall message rate limit",
        "rate limit",
        "too many requests",
        "usage limit",
    )
    # Bounded retry for transient GitHub API failures (TLS blips, connection
    # resets, gateway 5xx, secondary rate limits, etc.) in GitHub.run().
    # These knobs apply fleet-wide; keep them in RuntimeConfig so GitHub stays
    # a frozen value object with no mutable state.
    gh_max_retries: int = 3
    gh_retry_base_seconds: float = 1.0
    # Pre-emptive GraphQL rate-limit guard. Before starting quota-heavy phases
    # (mop-up sweeps, merged-PR listings), GitHub.check_graphql_rate_limit()
    # verifies ``resources.graphql.remaining`` from ``gh api rate_limit`` is at
    # least this value. Set to 0 to disable the guard.
    graphql_rate_limit_threshold: int = 1500


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
    # workers share one venv. Disabled by default (issue #112): a devin-shell
    # worker has a full shell and can run uv sync, which would rewrite the
    # shared venv's editable-install .pth to point at the worktree, causing
    # other worktrees to silently import that worktree's code. Set a relative
    # path to opt back into the shared-venv junction; None keeps each worktree
    # isolated.
    venv_source: str | None = None
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
    # Disabled by default: a claude-code worker has a full agentic shell and
    # can run uv sync, which would rewrite the shared venv's editable install
    # metadata to point at the worktree (issue #274). Setting a relative path
    # re-enables the legacy shared-venv junction; None disables it.
    venv_source: str | None = None
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
    # Opt-in: tee Claude Code's --output-format stream-json to a separate events.jsonl file.
    # When enabled, the worker launch command is extended with --output-format stream-json
    # and the structured JSONL output is written to issue-<n>.events.jsonl alongside the
    # plaintext log. This enables downstream parsing of tool_call_count, turn_count, tokens,
    # and cost_usd for tripwires and progress reporting. Default False until #162/#163 land.
    tee_stream_json: bool = False


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
    redispatch_window_minutes: int = 240
    max_auto_redispatch: int = 3
    terminal_error_markers: tuple[str, ...] = (
        "Error: A tool was rejected",
        "Error: Agent error:",
    )
    # Wall-clock deadline (absolute age cap) - applies to both adapters
    wall_clock_minutes: int = 240
    wall_clock_kill: bool = False
    # Loop/no-progress detection (Claude Code only) - window = stall_minutes * multiplier
    loop_stall_multiplier: int = 2
    loop_kill: bool = False
    # Cost/token budget tripwire (issue #163). None/0 = disabled.
    # When enabled, checks cumulative usage from Claude Code's tee'd events.jsonl.
    cost_budget_usd: float | None = None
    token_budget: int | None = None
    # Action when budget is exceeded: "warn" (default, no kill) or "kill"
    cost_budget_action: str = "warn"
    # Launch stall detection (issue #221): grace period for shim materialization.
    # Sessions whose log has not grown past the shim marker within this window
    # are classified as launch_stalled and reaped. Default 5 minutes.
    launch_stall_grace_minutes: int = 5
    # Rate-limit deferral (issue #247): when a stalled-looking worker's log tail
    # matches a provider rate-limit signature, defer the stall kill until the
    # parsed reset time plus this slack has elapsed. Default 2 minutes.
    rate_limit_defer_enabled: bool = True
    rate_limit_defer_slack_minutes: int = 2
    # Inconclusive real-activity probe deferral cap (issue #338). A dead worker
    # whose probe is inconclusive (all sources errored or no match yet) is
    # deferred this many passes before Signal-1 reaps it. Prevents a permanently
    # broken probe from pinning a slot indefinitely while still allowing the
    # downstream liveness checks to avoid false reaps.
    max_inconclusive_probe_deferrals: int = 3
    # Issue #439: a PR whose updatedAt is older than this many minutes and has
    # an empty statusCheckRollup is treated as stuck before review (the worker
    # died before CI could start). It is routed to the rework pipeline with a
    # rebase-onto-main brief. 0 disables the stale-empty-check predicate.
    pre_review_rework_stale_minutes: int = 30
    # Worktree file mtime corroboration (issue #353): a fourth real-activity
    # source that detects progress by scanning files written in the worker's
    # checkout. Workers read/plan for a while before writing, so this source uses
    # its own generous threshold rather than stall_minutes.
    worktree_mtime_enabled: bool = True
    worktree_mtime_threshold_minutes: int = 45
    worktree_mtime_max_depth: int = 4
    worktree_mtime_exclude_dirs: tuple[str, ...] = (".git", ".venv")


@dataclass(frozen=True)
class TestAdequacyConfig:
    """Config for the opt-in test-adequacy gate (janitor.check_test_adequacy).

    ``enabled`` defaults False so an absent config block is a no-op — mirrors
    CrossFamilyConfig (config.py:236). When enabled, ``OrchestratorApp.review()``
    runs the structural check (``janitor.check_test_adequacy``) before packet
    generation: a Tier-1 "pure skip" failure auto-records a ``request_changes``
    decision, and a passing PR gets a test-quality rubric folded into the review
    packet (Tier-2). Tier-3 diff-coverage fields below remain reserved/unread.
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
    stub_test_seam_keywords: tuple[str, ...] = (
        "route",
        "e2e",
        "byte",
        "call_model",
        "dispatch",
        "lock",
        "concurrent",
    )
    exempt_marker: str = "Test-exempt:"
    # Tier 3 (reserved, deferred): diff-coverage extension, not read by any
    # code path yet.
    coverage_enabled: bool = False
    coverage_command: tuple[str, ...] = ()
    min_diff_coverage: float = 0.0


@dataclass(frozen=True)
class FleetConfig:
    """Fleet-wide configuration for multi-repo coordination.

    ``global_max_concurrent_sessions`` caps total live worker sessions across
    all registered repos. Default 0 (unlimited) preserves current per-repo-only
    behavior. This addresses worker-count oversubscription only; CPU/RAM
    oversubscription via xdist requires operator discipline (see RUNBOOK.md).
    """

    global_max_concurrent_sessions: int = 0


@dataclass(frozen=True)
class NotifyConfig:
    """Pluggable needs-attention notification sink.

    Detect (supervisor) and escalate (label policy) are separate concerns — this
    section only decides where a digest goes once a needs-attention transition
    has already fired. ``enabled`` defaults False so an absent config block is
    a no-op — mirrors CrossFamilyConfig (config.py:236).
    """

    enabled: bool = False
    sink: str = "file"  # "webhook" | "desktop" | "shell" | "file"
    webhook_url: str = ""
    shell_command: tuple[str, ...] = ()
    file_path: str = ".var/charlie-work/notify/digest.jsonl"


@dataclass(frozen=True)
class RunnersConfig:
    """GitHub Actions runner management.

    ``cancel_superseded_main_runs`` defaults False so an absent config block is
    a no-op — mirrors CrossFamilyConfig (config.py:236). When enabled, cancels
    queued runs on the default branch for the configured workflow, keeping only
    the newest (its tree contains every earlier merge).

    ``fleet_autoscale_prologue`` defaults False so an absent config block is
    a no-op. When enabled, runs the autoscale decision before fleet bash-rats.
    """

    enabled: bool = False
    cancel_superseded_main_runs: bool = False
    default_branch: str = "main"
    workflow_name: str = ""
    fleet_autoscale_prologue: bool = False


@dataclass(frozen=True)
class RunnerScalingConfig:
    """Self-hosted GitHub Actions runner pool scaling configuration.

    ``enabled`` defaults False so an absent config block is a no-op — mirrors
    CrossFamilyConfig (config.py:236). This is the foundation for read-only
    observability; scaling actions are deferred to future issues.
    """

    enabled: bool = False
    # Root directory where runner instances are managed (e.g., "C:\\actions-runners")
    managed_root: str = ""
    # Directory name prefix for runner instances (e.g., "jc-" for "jc-1", "jc-2")
    runner_dir_prefix: str = "jc-"
    # Template for GitHub runner names (e.g., "jc-9800x3d-{n}" where {n} is the instance number)
    runner_name_template: str = "jc-{n}"
    # Path to the runner package zip file for installation
    package_zip: str = ""
    # Minimum number of runners to maintain in the pool
    min_runners: int = 1
    # Maximum number of runners allowed in the pool
    max_runners: int = 10
    # Estimated RAM required per job in GB (empirical: ~2)
    ram_per_job_gb: float = 2.0
    # Minimum free RAM required in GB before scaling up
    min_free_ram_gb: float = 4.0
    # Maximum host CPU percentage before scaling up
    max_host_cpu_pct: float = 80.0
    # Minutes of idle time before scaling down runners
    idle_scale_down_minutes: int = 15
    # Cooldown period between scaling actions in minutes
    cooldown_minutes: int = 5


@dataclass(frozen=True)
class SupervisorConfig:
    """Configuration for the supervised infill loop (``charlie bash-rats`` default mode).

    ``poll_interval_seconds``: how often to check for local-signal deltas when
    no pass is warranted (default 20 s).
    ``full_pass_interval_seconds``: fallback — run a pass even without a local
    delta to catch GitHub-side changes (default 300 s / 5 min).
    ``active_cooldown_seconds``: sleep after a pass that dispatched or merged
    something (default 30 s — stagger starts, respect rate limits).
    ``max_runtime_minutes``: hard wall-clock cap; 0 = unlimited (default).
    """

    poll_interval_seconds: int = 20
    full_pass_interval_seconds: int = 300
    active_cooldown_seconds: int = 30
    max_runtime_minutes: int = 0


@dataclass(frozen=True)
class SignatureRule:
    """One config-extensible pattern -> failure_kind mapping.

    ``pattern`` is a regex (re.IGNORECASE, re.search semantics) matched
    against extracted post-mortem tool-call content. Operators can add
    project-specific block-hook signatures without a code change — see
    PostMortemConfig.signature_rules.
    """

    pattern: str
    kind: str


@dataclass(frozen=True)
class PostMortemConfig:
    """Devin sessions.db post-mortem extraction (issue #261, extends #260).

    After a worker is reaped as dead, this reads the last N ``message_nodes``
    from the Devin CLI's local session store (sessions.db) to recover the
    terminal tool call — most importantly a ``decision:block`` (a worker
    killed by a push-gate hook mid-work, not a stall/crash/quota exhaustion).
    That distinction matters operationally: a blocked worker's local commits
    are real and worth preserving (see attempt_refs.py), and hot-redispatching
    it into the same hook immediately is pointless — it escalates instead.

    ``enabled`` defaults True (unlike the other opt-in sections) because
    post-mortem extraction is read-only, best-effort, and degrades to a
    silent no-op on any DB problem (missing file, lock, schema drift) —
    there is no behavior to opt out of that isn't already a no-op on failure.

    ``signature_rules`` classifies extracted message-node content into a
    ``failure_kind``; only the ``worker_blocked`` kind currently changes
    reaper behavior (suppresses hot redispatch, escalates instead), but the
    list is config-extensible so new terminal-tool signatures can be added
    without a code change.

    These same rules also drive ``post_mortem.classify_and_record``'s
    log-tail fallback (issue #260, corrected premise): when sessions.db
    extraction is disabled, unavailable (locked/missing/schema-drifted), or
    matched but found no ``worker_blocked`` signature among the message
    nodes, the worker's own log tail is the only remaining signal. Rather
    than add a parallel config surface for that fallback, it reuses this
    same ``signature_rules`` list (filtered to ``kind == "worker_blocked"``)
    so a new block-hook signature is one config edit, not two. The
    ``"A tool was rejected by the user"`` rule below is that fallback's
    primary target — it is the exact phrasing the Devin CLI prints to its
    own log/stdout when a PreToolUse hook blocks a tool call, distinct from
    the ``"Tool blocked:"`` prefix that appears in sessions.db message-node
    content.
    """

    enabled: bool = True
    # "" resolves to %APPDATA%\devin\cli\sessions.db at read time (env-expanded,
    # never hardcoded — see post_mortem._default_db_path).
    db_path: str = ""
    # How many of the most recent message_nodes to pull per matched session.
    message_node_limit: int = 10
    # Slack applied to both ends of the [started_at, reaped_at] window when
    # matching a session by working_directory (clock skew / write-lag tolerance).
    match_window_margin_seconds: int = 120
    # When worker.started_at itself fails to parse, the match window can no
    # longer be anchored to it — falling back to a narrow now-minus-margin
    # window (the old behavior) missed real sessions that started well
    # before "now" (the reap can run long after the worker actually died).
    # Widen to this lookback from "now" instead; recorded on the resulting
    # PostMortemRecord as window_start_fallback so a false non-match is
    # diagnosable. Default 6h comfortably covers any single dispatch.
    unparseable_started_at_lookback_seconds: int = 21600
    signature_rules: tuple[SignatureRule, ...] = (
        SignatureRule(pattern=r"Tool blocked:", kind="worker_blocked"),
        SignatureRule(pattern=r"decision\s*:\s*block", kind="worker_blocked"),
        SignatureRule(pattern=r"A tool was rejected by the user", kind="worker_blocked"),
    )


@dataclass(frozen=True)
class OrchestratorConfig:
    labels: LabelConfig = field(default_factory=LabelConfig)
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    review_dispatch: ReviewDispatchConfig = field(default_factory=ReviewDispatchConfig)
    auto_merge: AutoMergeConfig = field(default_factory=AutoMergeConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    devin: DevinConfig = field(default_factory=DevinConfig)
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)
    cross_family: CrossFamilyConfig = field(default_factory=CrossFamilyConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    test_adequacy: TestAdequacyConfig = field(default_factory=TestAdequacyConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    runners: RunnersConfig = field(default_factory=RunnersConfig)
    runner_scaling: RunnerScalingConfig = field(default_factory=RunnerScalingConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    post_mortem: PostMortemConfig = field(default_factory=PostMortemConfig)


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
    launch_stagger_seconds = dispatch_data.get("launch_stagger_seconds")
    if launch_stagger_seconds is not None and not isinstance(launch_stagger_seconds, int):
        raise ConfigError(
            "config section 'dispatch' key 'launch_stagger_seconds' must be an int, "
            f"got {type(launch_stagger_seconds).__name__}"
        )
    if launch_stagger_seconds is not None and launch_stagger_seconds < 0:
        raise ConfigError(
            "config section 'dispatch' key 'launch_stagger_seconds' must be >= 0, "
            f"got {launch_stagger_seconds}"
        )
    finalize_limit = dispatch_data.get("finalize_limit")
    if finalize_limit is not None and not isinstance(finalize_limit, int):
        raise ConfigError(
            "config section 'dispatch' key 'finalize_limit' must be an int, "
            f"got {type(finalize_limit).__name__}"
        )
    if finalize_limit is not None and finalize_limit < 0:
        raise ConfigError(
            f"config section 'dispatch' key 'finalize_limit' must be >= 0, got {finalize_limit}"
        )
    injected_paths = dispatch_data.get("injected_paths")
    if injected_paths is not None:
        if not isinstance(injected_paths, list):
            raise ConfigError(
                "config section 'dispatch' key 'injected_paths' must be a list of "
                f"strings, got {type(injected_paths).__name__}"
            )
        for item in injected_paths:
            if not isinstance(item, str):
                raise ConfigError(
                    "config section 'dispatch' key 'injected_paths' must be a list of "
                    f"strings, got element of type {type(item).__name__}"
                )
        dispatch_data["injected_paths"] = _normalize_injected_paths(
            tuple(str(item) for item in injected_paths)
        )
    dispatch = _build_section(DispatchConfig, "dispatch", dispatch_data)
    review = _build_section(ReviewConfig, "review", _section(data, "review"))
    review_dispatch_data = _section(data, "review_dispatch")
    for rd_bool_key in ("enabled",):
        rd_bool_value = review_dispatch_data.get(rd_bool_key)
        if rd_bool_value is not None and not isinstance(rd_bool_value, bool):
            raise ConfigError(
                f"config section 'review_dispatch' key '{rd_bool_key}' must be a bool, "
                f"got {type(rd_bool_value).__name__}"
            )
    rd_reviews_dir = review_dispatch_data.get("reviews_dir")
    if rd_reviews_dir is not None and not isinstance(rd_reviews_dir, str):
        raise ConfigError(
            "config section 'review_dispatch' key 'reviews_dir' must be a string, "
            f"got {type(rd_reviews_dir).__name__}"
        )
    rd_max_local = review_dispatch_data.get("max_local_review_processes")
    if rd_max_local is not None and not isinstance(rd_max_local, int):
        raise ConfigError(
            "config section 'review_dispatch' key 'max_local_review_processes' must be an int, "
            f"got {type(rd_max_local).__name__}"
        )
    if rd_max_local is not None and rd_max_local < 0:
        raise ConfigError(
            "config section 'review_dispatch' key 'max_local_review_processes' must be >= 0, "
            f"got {rd_max_local}"
        )
    for rd_int_key in ("stall_minutes", "max_stall_attempts"):
        rd_int_value = review_dispatch_data.get(rd_int_key)
        if rd_int_value is not None and not isinstance(rd_int_value, int):
            raise ConfigError(
                f"config section 'review_dispatch' key '{rd_int_key}' must be an int, "
                f"got {type(rd_int_value).__name__}"
            )
        if rd_int_value is not None and rd_int_value < 0:
            raise ConfigError(
                f"config section 'review_dispatch' key '{rd_int_key}' must be >= 0, "
                f"got {rd_int_value}"
            )
    review_dispatch = _build_section(ReviewDispatchConfig, "review_dispatch", review_dispatch_data)
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
    failed_attempt_alarm = auto_merge_data.get("failed_attempt_alarm")
    if failed_attempt_alarm is not None and not isinstance(failed_attempt_alarm, int):
        raise ConfigError(
            "config section 'auto_merge' key 'failed_attempt_alarm' must be an int, "
            f"got {type(failed_attempt_alarm).__name__}"
        )
    mergequeue_label = auto_merge_data.get("mergequeue_label")
    if mergequeue_label is not None:
        if not isinstance(mergequeue_label, str):
            raise ConfigError(
                "config section 'auto_merge' key 'mergequeue_label' must be a string, "
                f"got {type(mergequeue_label).__name__}"
            )
        stripped_mergequeue_label = mergequeue_label.strip()
        if not stripped_mergequeue_label:
            raise ConfigError(
                "config section 'auto_merge' key 'mergequeue_label' must not be empty"
            )
        # Store the stripped value: this threads verbatim into
        # `gh pr edit --add-label <label>`, so surrounding whitespace must
        # not survive into the actual GitHub label name.
        auto_merge_data["mergequeue_label"] = stripped_mergequeue_label
    auto_merge = _build_section(AutoMergeConfig, "auto_merge", auto_merge_data)
    runtime_data = _section(data, "runtime")
    throttle_error_markers = runtime_data.get("throttle_error_markers")
    if throttle_error_markers is not None:
        if not isinstance(throttle_error_markers, list):
            raise ConfigError(
                "config section 'runtime' key 'throttle_error_markers' must be a list of "
                f"strings, got {type(throttle_error_markers).__name__}"
            )
        for item in throttle_error_markers:
            if not isinstance(item, str):
                raise ConfigError(
                    "config section 'runtime' key 'throttle_error_markers' must be a list of "
                    f"strings, got element of type {type(item).__name__}"
                )
        runtime_data["throttle_error_markers"] = tuple(throttle_error_markers)
    gh_max_retries = runtime_data.get("gh_max_retries")
    if gh_max_retries is not None and not isinstance(gh_max_retries, int):
        raise ConfigError(
            "config section 'runtime' key 'gh_max_retries' must be an int, "
            f"got {type(gh_max_retries).__name__}"
        )
    gh_retry_base_seconds = runtime_data.get("gh_retry_base_seconds")
    if gh_retry_base_seconds is not None and not isinstance(gh_retry_base_seconds, (int, float)):
        raise ConfigError(
            "config section 'runtime' key 'gh_retry_base_seconds' must be a number, "
            f"got {type(gh_retry_base_seconds).__name__}"
        )
    graphql_rate_limit_threshold = runtime_data.get("graphql_rate_limit_threshold")
    if graphql_rate_limit_threshold is not None:
        if not isinstance(graphql_rate_limit_threshold, int):
            raise ConfigError(
                "config section 'runtime' key 'graphql_rate_limit_threshold' must be an int, "
                f"got {type(graphql_rate_limit_threshold).__name__}"
            )
        if graphql_rate_limit_threshold < 0:
            raise ConfigError(
                "config section 'runtime' key 'graphql_rate_limit_threshold' must be >= 0, "
                f"got {graphql_rate_limit_threshold}"
            )
    runtime = _build_section(RuntimeConfig, "runtime", runtime_data)
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
    watchdog_data = _section(data, "watchdog")
    terminal_error_markers = watchdog_data.get("terminal_error_markers")
    if terminal_error_markers is not None:
        if not isinstance(terminal_error_markers, list):
            raise ConfigError(
                "config section 'watchdog' key 'terminal_error_markers' must be a list of "
                f"strings, got {type(terminal_error_markers).__name__}"
            )
        for item in terminal_error_markers:
            if not isinstance(item, str):
                raise ConfigError(
                    "config section 'watchdog' key 'terminal_error_markers' must be a list of "
                    f"strings, got element of type {type(item).__name__}"
                )
        watchdog_data["terminal_error_markers"] = tuple(terminal_error_markers)
    # Validate cost_budget_usd
    cost_budget_usd = watchdog_data.get("cost_budget_usd")
    if cost_budget_usd is not None and not isinstance(cost_budget_usd, (int, float)):
        raise ConfigError(
            "config section 'watchdog' key 'cost_budget_usd' must be a number, "
            f"got {type(cost_budget_usd).__name__}"
        )
    # Validate token_budget
    token_budget = watchdog_data.get("token_budget")
    if token_budget is not None and not isinstance(token_budget, int):
        raise ConfigError(
            "config section 'watchdog' key 'token_budget' must be an int, "
            f"got {type(token_budget).__name__}"
        )
    # Validate cost_budget_action
    cost_budget_action = watchdog_data.get("cost_budget_action")
    if cost_budget_action is not None:
        if not isinstance(cost_budget_action, str):
            raise ConfigError(
                "config section 'watchdog' key 'cost_budget_action' must be a string, "
                f"got {type(cost_budget_action).__name__}"
            )
        if cost_budget_action not in ("warn", "kill"):
            raise ConfigError(
                f"config section 'watchdog' key 'cost_budget_action' must be 'warn' or 'kill', "
                f"got '{cost_budget_action}'"
            )
    # Validate launch_stall_grace_minutes
    launch_stall_grace_minutes = watchdog_data.get("launch_stall_grace_minutes")
    if launch_stall_grace_minutes is not None and not isinstance(launch_stall_grace_minutes, int):
        raise ConfigError(
            "config section 'watchdog' key 'launch_stall_grace_minutes' must be an int, "
            f"got {type(launch_stall_grace_minutes).__name__}"
        )
    # Validate rate-limit deferral config (issue #247)
    rate_limit_defer_enabled = watchdog_data.get("rate_limit_defer_enabled")
    if rate_limit_defer_enabled is not None and not isinstance(rate_limit_defer_enabled, bool):
        raise ConfigError(
            "config section 'watchdog' key 'rate_limit_defer_enabled' must be a bool, "
            f"got {type(rate_limit_defer_enabled).__name__}"
        )
    rate_limit_defer_slack_minutes = watchdog_data.get("rate_limit_defer_slack_minutes")
    if rate_limit_defer_slack_minutes is not None and not isinstance(
        rate_limit_defer_slack_minutes, int
    ):
        raise ConfigError(
            "config section 'watchdog' key 'rate_limit_defer_slack_minutes' must be an int, "
            f"got {type(rate_limit_defer_slack_minutes).__name__}"
        )
    max_inconclusive_probe_deferrals = watchdog_data.get("max_inconclusive_probe_deferrals")
    if max_inconclusive_probe_deferrals is not None and not isinstance(
        max_inconclusive_probe_deferrals, int
    ):
        raise ConfigError(
            "config section 'watchdog' key 'max_inconclusive_probe_deferrals' must be an int, "
            f"got {type(max_inconclusive_probe_deferrals).__name__}"
        )
    pre_review_rework_stale_minutes = watchdog_data.get("pre_review_rework_stale_minutes")
    if pre_review_rework_stale_minutes is not None and not isinstance(
        pre_review_rework_stale_minutes, int
    ):
        raise ConfigError(
            "config section 'watchdog' key 'pre_review_rework_stale_minutes' must be an int, "
            f"got {type(pre_review_rework_stale_minutes).__name__}"
        )
    # Validate worktree mtime corroboration config (issue #353)
    worktree_mtime_enabled = watchdog_data.get("worktree_mtime_enabled")
    if worktree_mtime_enabled is not None and not isinstance(worktree_mtime_enabled, bool):
        raise ConfigError(
            "config section 'watchdog' key 'worktree_mtime_enabled' must be a bool, "
            f"got {type(worktree_mtime_enabled).__name__}"
        )
    for worktree_int_key in ("worktree_mtime_threshold_minutes", "worktree_mtime_max_depth"):
        worktree_int_value = watchdog_data.get(worktree_int_key)
        if worktree_int_value is not None and not isinstance(worktree_int_value, int):
            raise ConfigError(
                f"config section 'watchdog' key '{worktree_int_key}' must be an int, "
                f"got {type(worktree_int_value).__name__}"
            )
    worktree_mtime_exclude_dirs = watchdog_data.get("worktree_mtime_exclude_dirs")
    if worktree_mtime_exclude_dirs is not None:
        if not isinstance(worktree_mtime_exclude_dirs, list):
            raise ConfigError(
                "config section 'watchdog' key 'worktree_mtime_exclude_dirs' must be a list of "
                f"strings, got {type(worktree_mtime_exclude_dirs).__name__}"
            )
        for item in worktree_mtime_exclude_dirs:
            if not isinstance(item, str):
                raise ConfigError(
                    "config section 'watchdog' key 'worktree_mtime_exclude_dirs' must be a list of "
                    f"strings, got element of type {type(item).__name__}"
                )
        watchdog_data["worktree_mtime_exclude_dirs"] = tuple(
            str(item) for item in worktree_mtime_exclude_dirs
        )
    watchdog = _build_section(WatchdogConfig, "watchdog", watchdog_data)
    test_adequacy_data = _section(data, "test_adequacy")

    # Six tuple-of-str fields: reject non-list, coerce elements to str.
    _TEST_ADEQUACY_TUPLE_FIELDS = (
        "test_path_globs",
        "exempt_path_globs",
        "assertion_markers",
        "comment_prefixes",
        "stub_test_seam_keywords",
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
    fleet_data = _section(data, "fleet")
    global_max = fleet_data.get("global_max_concurrent_sessions")
    if global_max is not None and not isinstance(global_max, int):
        raise ConfigError(
            "config section 'fleet' key 'global_max_concurrent_sessions' must be an "
            f"int, got {type(global_max).__name__}"
        )
    fleet = _build_section(FleetConfig, "fleet", fleet_data)
    notify_data = _section(data, "notify")
    shell_command = notify_data.get("shell_command")
    if isinstance(shell_command, list):
        notify_data["shell_command"] = tuple(str(item) for item in shell_command)
    notify = _build_section(NotifyConfig, "notify", notify_data)
    runners_data = _section(data, "runners")
    # Validate runners config fields
    for bool_key in ("enabled", "cancel_superseded_main_runs"):
        bool_value = runners_data.get(bool_key)
        if bool_value is not None and not isinstance(bool_value, bool):
            raise ConfigError(
                f"config section 'runners' key '{bool_key}' must be a bool, "
                f"got {type(bool_value).__name__}"
            )
    for str_key in ("default_branch", "workflow_name"):
        str_value = runners_data.get(str_key)
        if str_value is not None and not isinstance(str_value, str):
            raise ConfigError(
                f"config section 'runners' key '{str_key}' must be a string, "
                f"got {type(str_value).__name__}"
            )
    runners = _build_section(RunnersConfig, "runners", runners_data)
    runner_scaling_data = _section(data, "runner_scaling")
    # Validate numeric fields
    for numeric_key in (
        "min_runners",
        "max_runners",
        "idle_scale_down_minutes",
        "cooldown_minutes",
    ):
        value = runner_scaling_data.get(numeric_key)
        if value is not None and not isinstance(value, int):
            raise ConfigError(
                f"config section 'runner_scaling' key '{numeric_key}' must be an int, "
                f"got {type(value).__name__}"
            )
    for float_key in ("ram_per_job_gb", "min_free_ram_gb", "max_host_cpu_pct"):
        value = runner_scaling_data.get(float_key)
        if value is not None and not isinstance(value, (int, float)):
            raise ConfigError(
                f"config section 'runner_scaling' key '{float_key}' must be a number, "
                f"got {type(value).__name__}"
            )
    # Validate string fields
    for str_key in ("managed_root", "runner_dir_prefix", "runner_name_template", "package_zip"):
        value = runner_scaling_data.get(str_key)
        if value is not None and not isinstance(value, str):
            raise ConfigError(
                f"config section 'runner_scaling' key '{str_key}' must be a string, "
                f"got {type(value).__name__}"
            )
    # Validate boolean field
    enabled = runner_scaling_data.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ConfigError(
            f"config section 'runner_scaling' key 'enabled' must be a bool, "
            f"got {type(enabled).__name__}"
        )
    runner_scaling = _build_section(RunnerScalingConfig, "runner_scaling", runner_scaling_data)
    supervisor_data = _section(data, "supervisor")
    for int_key in (
        "poll_interval_seconds",
        "full_pass_interval_seconds",
        "active_cooldown_seconds",
        "max_runtime_minutes",
    ):
        value = supervisor_data.get(int_key)
        if value is not None and not isinstance(value, int):
            raise ConfigError(
                f"config section 'supervisor' key '{int_key}' must be an int, "
                f"got {type(value).__name__}"
            )
    supervisor = _build_section(SupervisorConfig, "supervisor", supervisor_data)
    post_mortem_data = _section(data, "post_mortem")
    pm_enabled = post_mortem_data.get("enabled")
    if pm_enabled is not None and not isinstance(pm_enabled, bool):
        raise ConfigError(
            f"config section 'post_mortem' key 'enabled' must be a bool, "
            f"got {type(pm_enabled).__name__}"
        )
    db_path = post_mortem_data.get("db_path")
    if db_path is not None and not isinstance(db_path, str):
        raise ConfigError(
            f"config section 'post_mortem' key 'db_path' must be a string, "
            f"got {type(db_path).__name__}"
        )
    for int_key in (
        "message_node_limit",
        "match_window_margin_seconds",
        "unparseable_started_at_lookback_seconds",
    ):
        value = post_mortem_data.get(int_key)
        if value is not None and not isinstance(value, int):
            raise ConfigError(
                f"config section 'post_mortem' key '{int_key}' must be an int, "
                f"got {type(value).__name__}"
            )
    signature_rules = post_mortem_data.get("signature_rules")
    if signature_rules is not None:
        if not isinstance(signature_rules, list):
            raise ConfigError(
                "config section 'post_mortem' key 'signature_rules' must be a list of "
                f"{{pattern, kind}} mappings, got {type(signature_rules).__name__}"
            )
        built_rules: list[SignatureRule] = []
        for i, item in enumerate(signature_rules):
            if not isinstance(item, dict):
                raise ConfigError(
                    f"config section 'post_mortem' key 'signature_rules[{i}]' must be a "
                    f"mapping with 'pattern' and 'kind' keys, got {type(item).__name__}"
                )
            unknown_rule_keys = sorted(set(item) - {"pattern", "kind"})
            if unknown_rule_keys:
                raise ConfigError(
                    f"config section 'post_mortem' key 'signature_rules[{i}]' has unknown "
                    f"key(s): {', '.join(unknown_rule_keys)} (valid: pattern, kind)"
                )
            pattern = item.get("pattern")
            kind = item.get("kind")
            if not isinstance(pattern, str) or not pattern:
                raise ConfigError(
                    f"config section 'post_mortem' key 'signature_rules[{i}].pattern' must "
                    "be a non-empty string"
                )
            if not isinstance(kind, str) or not kind:
                raise ConfigError(
                    f"config section 'post_mortem' key 'signature_rules[{i}].kind' must "
                    "be a non-empty string"
                )
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ConfigError(
                    f"config section 'post_mortem' key 'signature_rules[{i}].pattern' is not "
                    f"a valid regex: {exc}"
                ) from exc
            built_rules.append(SignatureRule(pattern=pattern, kind=kind))
        post_mortem_data["signature_rules"] = tuple(built_rules)
    post_mortem = _build_section(PostMortemConfig, "post_mortem", post_mortem_data)
    return OrchestratorConfig(
        labels=labels,
        dispatch=dispatch,
        review=review,
        review_dispatch=review_dispatch,
        auto_merge=auto_merge,
        runtime=runtime,
        devin=devin,
        claude_code=claude_code,
        cross_family=cross_family,
        watchdog=watchdog,
        test_adequacy=test_adequacy,
        fleet=fleet,
        notify=notify,
        runners=runners,
        runner_scaling=runner_scaling,
        supervisor=supervisor,
        post_mortem=post_mortem,
    )
