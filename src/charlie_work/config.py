from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from charlie_work.github import ORCHESTRATOR_MANAGED_MERGE_FLAGS

# LOAD-BEARING RE-EXPORT — NOT AN UNUSED IMPORT. Do not delete; the `noqa`
# below marks a deliberate re-export, not a lint concession.
#
# Re-exported in place, not re-declared. These two dataclasses are compared and
# isinstance-checked across the seam, and two structurally identical frozen
# dataclasses are never equal to each other — so a local re-declaration breaks
# equality and isinstance at runtime without breaking any import.
#
# `tests/test_ci_fleet_seams.py` guards this seam; deletion fails the suite.
# See the matching note in github.py for the same pattern applied to
# GitHubError, where the consequence is uncaught exceptions.
from ci_fleet.config import (  # noqa: F401  (deliberate re-export)
    RunnerAllocationConfig,
    RunnerScalingConfig,
)

from . import layout
from .issue_comments import DEFAULT_INCLUDED_ASSOCIATIONS as DEFAULT_COMMENT_ASSOCIATIONS

DEFAULT_CONFIG_FILENAME = "orchestrator.config.yaml"

# Root-relative path the Claude Code adapter writes to in each worktree.
CLAUDE_CODE_PROMPT_FILENAME = ".orchestrator-prompt.md"

# Worktree writer marker used to enforce single-writer-per-branch (issue #400).
WRITER_MARKER_FILENAME = ".charlie-writer.json"

# Structured outcome a worker writes when it pushed a branch but could not open a
# PR (per the ``$section_push_pr_outcome`` prompt contract); read back by
# ``worktree.read_worker_outcome``. Lives here rather than in ``worktree`` so that
# ``DispatchConfig`` can exclude it without importing ``worktree`` (which imports
# this module) -- the same arrangement as ``WRITER_MARKER_FILENAME`` above.
WORKER_OUTCOME_FILENAME = ".worker-outcome.json"


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
    merge_hold: str = "agent:merge-hold"
    # Routing hint, NOT a workflow state (issue #481). Never a member of
    # ``active``/``terminal``/``workflow_labels`` — it must not affect issue
    # selection or exclusion. Included in ``all`` so ``bootstrap_labels``
    # creates it on GitHub with a sensible description. Human-applied at filing
    # time; read by routing.select_adapter to send a complex first-pass issue to
    # the api worker instead of the weaker default worker.
    complexity_high: str = "complexity:high"

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
            self.merge_hold,
            self.complexity_high,
        ]

    @property
    def workflow_labels(self) -> set[str]:
        """All workflow labels (agent:* states) excluding the ready marker.

        ``merge_hold`` is intentionally excluded: it is a transient operator
        signal, not a workflow state. Including it here would make every
        non-terminal transition (``review_started``, ``rework_requested``, …)
        strip the hold from the issue, violating the issue #496 persistence
        requirement. Terminal transitions strip it explicitly via ``extra_remove``
        (see ``labels._edges``).
        """
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
    # Issue comments rendered into the worker prompt (issue #872). A worker that
    # only sees ``issue.body`` cannot see the comment that corrected it.
    #
    # The include filter is an allow-list on GitHub's ``authorAssociation``, not
    # a deny-list of bot logins: a new bot then needs no code change to be
    # excluded, and the comment payload carries no ``is_bot`` field to key off.
    # ``viewerDidAuthor`` looks like the natural self-filter and is the wrong
    # one -- the orchestrator authenticates as the operator's own account, so it
    # is true for exactly the human corrections this feature exists to deliver.
    worker_prompt_comment_associations: tuple[str, ...] = DEFAULT_COMMENT_ASSOCIATIONS
    # Logins whose comments are dropped regardless of association. Empty by
    # default; the escape hatch for a bot that comments as a COLLABORATOR, or
    # for the orchestrator's own future issue-side chatter.
    worker_prompt_excluded_comment_authors: tuple[str, ...] = ()
    # Prompt budget. Newest comments win when either bound binds, and the
    # rendered block says how many were dropped rather than truncating silently.
    # 0 disables the respective bound.
    worker_prompt_max_comments: int = 20
    worker_prompt_max_comment_chars: int = 12000
    # Issue #946: maximum age (in minutes) of the most recent non-empty
    # ``dispatch`` event (payload ``issue_numbers != []``) before a warning
    # fires while the unfiltered backlog is observed to be non-empty. 0
    # disables the check.
    dispatch_staleness_minutes: int = 240

    def __post_init__(self) -> None:
        # Normalize to a tuple of forward-slash strings. The writer marker is
        # always excluded so it can never be mistaken for worker-authored work.
        if self.injected_paths:
            base = list(self.injected_paths)
        else:
            base = [CLAUDE_CODE_PROMPT_FILENAME]
        for protocol_file in (WRITER_MARKER_FILENAME, WORKER_OUTCOME_FILENAME):
            if protocol_file not in base:
                base.append(protocol_file)
        object.__setattr__(self, "injected_paths", _normalize_injected_paths(base))
        # Coerce the comment-filter sequences here rather than only in
        # ``load_config``: this is the one path every construction goes through,
        # so a direct ``DispatchConfig(...)`` in a test or a caller cannot smuggle
        # in a list and silently make a frozen instance unhashable. A bare string
        # is wrapped rather than iterated -- ``tuple("OWNER")`` would otherwise
        # yield five single-character "associations" that match nothing.
        for field_name in (
            "worker_prompt_comment_associations",
            "worker_prompt_excluded_comment_authors",
        ):
            value = getattr(self, field_name)
            normalized = (str(value),) if isinstance(value, str) else tuple(str(v) for v in value)
            object.__setattr__(self, field_name, normalized)


@dataclass(frozen=True)
class ReviewConfig:
    # Enforced in record_review: past this many request_changes cycles the PR
    # escalates to a human instead of another rework dispatch. 2 per operator
    # decision (2026-07-01) — iteration past ~2 rounds thrashes.
    max_rework_cycles: int = 2
    require_tests_or_rationale: bool = True
    require_issue_link: bool = True
    # Enforced in review()'s janitor gate: past this many rework routes for a
    # genuine merge conflict (mergeable=CONFLICTING or mergeStateStatus=DIRTY),
    # the PR escalates to a human instead of looping forever. Without this cap
    # a conflicting PR that a rework worker never actually rebases re-logs the
    # identical janitor_gate failure every pass indefinitely (cost-spirals.md
    # Finding 1).
    max_conflict_rework_attempts: int = 2
    # Same as max_conflict_rework_attempts, for the janitor's no-op-rework
    # signal (a rework cycle that pushed no actual content change — same
    # patch-id/head as the last request_changes verdict). Previously detected
    # but never consumed by anything (pr-lifecycle.md "janitor_blocked zero
    # readers" finding).
    max_no_op_rework_attempts: int = 2
    # Issue #765: orthogonal stall bound for _route_janitor_gate_failure_to_
    # rework's passive wait. max_conflict_rework_attempts/max_no_op_rework_
    # attempts only advance on a SETTLED head change (merge_conflict) or a
    # fresh non-pending detection (no_op_rework) -- both require the issue to
    # leave `rework_requested` at least once. A PR whose head simply stops
    # moving while queued (nobody dispatched a worker, or the worker's rework
    # brief was empty -- see issue #765's "relationship to the empty-findings
    # defect") never produces that signal, so it can sit in the passive
    # janitor_blocked wait forever: PR #696 spun 55 `rework_already_pushed`
    # events over 24h+ with its attempt count already AT the cap and the
    # rescue tier never even attempted, because the cap/rescue check below is
    # only reachable via progress. This bound fires independently of attempts
    # count: once the head has been observed unchanged for this long while
    # the issue is `rework_requested` (queued, nobody actively working),
    # escalate regardless of how many attempts have been burned. Only a HEAD
    # CHANGE resets the clock -- a live `dispatched` session pushing a real
    # commit resets it, but idle dispatched time (and a same-head status
    # flip, e.g. reconcile's issue_active_label_with_open_pr self-heal
    # normalizing the label away from rework_requested and back) still
    # counts toward the bound; see the call site. 240 minutes (4h)
    # matches WatchdogConfig.redispatch_window_minutes, this codebase's
    # existing convention for "how long a stuck fleet item may sit before
    # it's treated as abnormal" -- long enough to absorb normal dispatch
    # queueing latency under fleet load, short enough to bound the failure
    # mode to hours instead of the unbounded, indefinite spin this issue was
    # filed over.
    rework_stall_minutes: int = 240


@dataclass(frozen=True)
class QuotaProbeConfig:
    # Issue TBD: a blanket provider-throttle cooldown (claude_code.py's
    # _DEFAULT_QUOTA_COOLDOWN_HOURS, 24h) can badly outlive the real-world
    # constraint that caused it -- an operator can switch the ambient Claude
    # Code CLI to a different subscription account well before the cooldown
    # expires. When enabled, the supervisor periodically fires a single,
    # cheap, read-only Haiku-model CLI call (claude_code.run_quota_probe) and
    # clears the throttle early on a green result, instead of always riding
    # out the full cooldown window. False disables probing entirely -- the
    # cooldown then behaves exactly as before this feature existed.
    enabled: bool = True
    # Flat retry interval, NOT exponential backoff (deliberately distinct
    # from review_dispatch's quota_probe_interval_minutes below, which
    # doubles on each consecutive failure). The probe is cheap enough that a
    # fixed 15-minute cadence for the lifetime of the throttle is acceptable,
    # and a flat interval is simpler to reason about for "did my account
    # switch get picked up yet".
    interval_minutes: int = 15
    # Short alias form (matches ClaudeCodeConfig.model's convention), pinned
    # explicitly via --model so the probe never inherits ambient /model
    # state -- see ClaudeCodeConfig.model's comment for the outage this
    # guards against. A dedicated Haiku pin (distinct from claude_code.model)
    # keeps the probe cheap regardless of what model workers/reviewers use.
    model: str = "claude-haiku-4-5"
    # Bounded synchronous subprocess timeout. A single "reply OK" prompt
    # should return in well under a minute; this is a ceiling against a hung
    # CLI process, not an expected duration.
    timeout_seconds: int = 60
    # Deliberately trivial: the probe only needs to prove the CLI can
    # complete a session without hitting a throttle/auth signature, not
    # produce any real work.
    prompt: str = "Reply with the single word OK and nothing else."


@dataclass(frozen=True)
class ReconcilePassConfig:
    """Periodic in-loop reconcile: merge-lane-recovery plan §6-B.

    Wires ``OrchestratorApp.reconcile(fix=True)`` -- previously reachable
    only via the operator-invoked ``charlie mop-up --fix`` CLI command --
    into the fleet loop on a fixed cadence, so a state/label divergence
    (e.g. an escalation whose ``human_needed`` label write silently failed,
    per the plan's PRIMARY defect) is repaired automatically instead of only
    when an operator remembers to run mop-up. Most drift kinds are state-wins
    for labels: reconcile converges GitHub labels to match ``state.json``.
    It does rewrite ``status`` in a few narrow, machine-safe cases -- to the
    externally derived ``closed`` value, to clear a missing/invalid status
    key, to the terminal ``merged`` value on a finalized PR, and to the
    passive ``open_passive`` placeholder (most notably
    ``issue_active_label_with_open_pr``). It never rewrites ``status`` to an
    active dispatch/rework value and never rewrites an open escalated
    issue's ``status`` (D-2) -- a closed-while-escalated issue is
    finalized to ``closed`` like any other closed issue, but no escalated
    issue is ever re-entered into the machine -- so this is safe to run
    unattended on every repo, every cycle.
    """

    # Default True: the fleet has never run its own repair (baseline: zero
    # reconcile events ever recorded across the whole event history), and
    # the repair direction is provably safe (see class docstring). A knob
    # defaulted off would leave that divergence class unrepaired until an
    # operator remembered to flip it -- exactly the failure mode this
    # workstream exists to close. The knob exists for rollback, not opt-in.
    enabled: bool = True
    # detect_drift() issues two full-repo GitHub list queries (all PRs, all
    # issues) plus a GraphQL rate-limit check every time it runs -- heavier
    # than quota_probe's single cheap Haiku subprocess call, so a longer flat
    # cadence than quota_probe's 15 minutes is appropriate here. 30 minutes
    # still comfortably beats "only ever runs when an operator remembers to
    # run mop-up".
    interval_minutes: int = 30
    # Issue #947: ``agent:human-needed`` is a forced terminal state with no
    # other alerting -- an issue parked there (e.g. #894) is silently
    # invisible until an operator happens to look. detect_drift() reports any
    # OPEN issue that has carried the label at least this many days as
    # ``terminal_state_stale`` on every pass it stays that way (repeated
    # firing while true, matching the existing ``aviator_stale_blocked`` /
    # ``mergequeue_revoked`` alert-only kinds -- no dedup marker). 2 days
    # balances catching a genuinely stuck issue against not paging on every
    # escalation that clears same-day via ``charlie unescalate``.
    terminal_state_alert_days: int = 2


@dataclass(frozen=True)
class DeescalationConfig:
    """Periodic sweep that re-evaluates ``mechanical`` escalations (issue #783).

    Escalating to ``agent:human-needed`` on a process failure (dead worker,
    redispatch/rework-cycle cap, stalled worker) when the PR artifact itself
    is fine is a category error -- it was a one-way door with no automated
    re-entry. This sweep re-checks ONLY escalations recorded with
    ``reason_class == "mechanical"`` (see ``state.escalation_reason_class``);
    ``"judgment"`` escalations and any escalation with no recorded reason
    class are never auto-cleared (fail closed).

    ``max_auto_deescalations`` bounds the auto de-escalate/redispatch/
    re-escalate cycle per issue (the ``auto_deescalation_count`` field on the
    issue's state entry, never reset by this sweep -- only a human running
    ``charlie unescalate`` resets it). This is deliberately independent from
    every per-mechanism attempt cap (``max_rework_cycles``,
    ``max_auto_redispatch``, ...): this sweep never resets those counters, so
    a condition that keeps re-failing keeps re-tripping its own cap
    immediately; this counter instead bounds how many times the SWEEP itself
    may clear the human_needed door for the same issue before also leaving it
    terminal, so the outer escalate/de-escalate cycle cannot spin forever
    even if the underlying condition oscillates.
    """

    enabled: bool = True
    # Cheaper per-item than reconcile (one gh pr_view + pr_checks + pr_diff
    # per escalated-mechanical issue, no full-repo list queries), but there is
    # no urgency to re-evaluate faster than an operator would notice a fixed
    # escalation sitting idle -- 30 minutes matches reconcile_pass's cadence.
    interval_minutes: int = 30
    # Bounds the auto de-escalate -> redispatch -> re-escalate cycle per
    # issue. Once reached, the sweep stops clearing this issue's escalation
    # even if reason_class is still "mechanical" and the PR looks healthy;
    # a distinct one-time event (deescalation_cap_exhausted) makes the
    # terminal state diagnosable rather than a silently renamed one-way door.
    max_auto_deescalations: int = 2


@dataclass(frozen=True)
class ReviewDispatchConfig:
    # Issue #370: concurrent reviewer launcher for queued PRs. This is a
    # deterministic loop stage, not a provider governor; reviewers use
    # launch_claude_worker with no concurrency clamp for rate-limit reasons.
    enabled: bool = False
    # Per-PR review sidecar + log directory. MUST be distinct from
    # devin.sessions_dir so worker concurrency accounting is not poisoned by
    # reviewer processes. Empty string means "derive from runtime.state_dir"
    # (layout.reviews_dir_default) rather than a fixed literal -- see
    # paths.resolved_layout, the single place that resolves this sentinel.
    reviews_dir: str = ""
    # Local-only process bound. 0 means unlimited; raise this only if local
    # CPU/disk from concurrent reviewer worktrees becomes a visible bottleneck.
    # Default is 2 so a host that enables review_dispatch without overriding
    # this key does not run an unbounded number of local Claude Code reviewers.
    max_local_review_processes: int = 2
    # Provider-token budget slots. Limits how many reviewers can be in flight
    # simultaneously against the Claude usage budget. When a slot frees (a
    # reviewer finishes), the next poll dispatches another. 0 means unlimited.
    max_concurrent_reviews: int = 3
    # Base interval between quota-probe attempts after a reviewer launch hits
    # the usage wall. A probe is a single reviewer launch; this many minutes
    # must elapse before the next probe. Each consecutive probe that hits the
    # wall again doubles the interval (see quota_probe_max_interval_minutes)
    # instead of relaunching a real reviewer session into the wall every
    # ``quota_probe_interval_minutes`` for the duration of a live provider
    # outage (cost-spirals.md Finding 2: "No escalation backoff").
    quota_probe_interval_minutes: int = 15
    # Cap on the exponential probe backoff described above, in minutes. 240
    # (4h) keeps the floor below quota_reset_hours's default 5h window so a
    # probe is still attempted at least once before/around the window's
    # natural expiry. 0 disables the cap (backoff grows unbounded).
    quota_probe_max_interval_minutes: int = 240
    # Approximate provider usage-limit reset window in hours. When a reviewer
    # launch hits the wall, the global reviewer quota is held exhausted for at
    # least this long while probes run every ``quota_probe_interval_minutes``.
    quota_reset_hours: int = 5
    # Maximum reviewer dispatch attempts per PR before escalating to a human.
    # A dispatch attempt is counted each time a reviewer is launched; the
    # counter resets when a verdict is recorded or a new packet is generated
    # for an advanced head. Without this cap, a PR that never produces a
    # verdict (e.g. every reviewer hits the session limit) is re-dispatched
    # indefinitely, burning quota every stale-claim interval.
    max_review_dispatch_attempts: int = 3
    # Maximum agentic turns for a reviewer session. Caps token spend per
    # review by limiting how many tool-call round-trips the reviewer can make.
    # 0 means unlimited (preserves pre-existing behavior). 40 is generous for
    # a review (read diff, read tests, read a few source files, write verdict)
    # but prevents unbounded codebase exploration.
    review_max_turns: int = 40
    # Diff line count above which the review prompt includes a diff-size
    # warning and a per-file summary instead of encouraging the reviewer to
    # read the entire diff in one shot. 0 disables the threshold (always
    # include the full diff guidance). 500 lines is ~12K tokens, a reasonable
    # single-read budget; beyond that the reviewer should read file-by-file.
    diff_line_threshold: int = 500
    # Effort level pinned via --effort on reviewer session launches. Empty
    # string means fall back to claude_code.effort (the worker/reviewer
    # default). Its MEANING depends on review_effort_experiment_fraction below:
    #   - fraction == 0.0 (default, experiment disabled): review_effort, when
    #     set, applies to ALL reviewer launches unconditionally — exactly the
    #     pre-experiment behavior. This is the "just pin reviewer effort"
    #     knob with no A/B semantics.
    #   - fraction in (0.0, 1.0]: review_effort becomes the TREATMENT arm's
    #     effort, applied only to the assigned fraction of PRs (see
    #     claude_code._review_effort_arm / resolve_review_effort). The
    #     remaining PRs (control) fall back to claude_code.effort as if
    #     review_effort were unset. This lets the A/B be compared via the
    #     per-review session telemetry (review_session_metrics /
    #     review_effort_arm on record_review) without confounding from
    #     concurrent pipeline changes (time-windowed A/B was replaced by this
    #     per-PR randomized assignment for exactly that reason).
    review_effort: str = ""
    # Fraction (0.0-1.0) of PRs randomly (but deterministically, see
    # claude_code._review_effort_arm) assigned to the review_effort
    # "treatment" arm. 0.0 (default) disables the experiment: review_effort
    # applies to every review when set, same as before this field existed.
    # Assignment is a stateless hash of the PR number (+ salt below), so the
    # same PR always lands in the same arm across rework rounds/re-dispatches
    # — arm-hopping across rounds would contaminate the per-PR quality
    # signal the experiment is trying to measure.
    review_effort_experiment_fraction: float = 0.0
    # Mixed into the per-PR assignment hash alongside the PR number. Change
    # this to re-randomize arm assignment for a new experiment epoch (e.g.
    # after a config change invalidates the current cohort) without needing
    # to rename or remove the fraction field.
    review_effort_experiment_salt: str = ""


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
    # Maximum `gh run rerun` attempts per workflow run id for a required check
    # that is infra-failed (CANCELLED/INFRA_FAILURE/TIMED_OUT -- see
    # checks.classify_infra_failures, issue #841). Once every infra-failing
    # run id for a check has been retried this many times on the current
    # head, the PR escalates to a human instead of retrying forever.
    infra_rerun_attempt_cap: int = 2
    # Maximum minutes after the PR's last update (updatedAt) to wait for any
    # required check run to appear before routing an approved PR to readiness
    # rework. This catches invisible CI-never-started stalls (mergeStateStatus
    # DIRTY or a missing CI trigger). 0 disables the guard.
    readiness_no_ci_minutes: int = 15
    # Maximum minutes after the PR's last update (updatedAt) to wait before
    # querying GitHub Actions directly to distinguish "CI never created a run
    # for this head" from "CI is pending" when the janitor gate reports
    # required checks missing. Unlike readiness_no_ci_minutes (which only
    # gates the post-approval merge_ready path), this applies to any PR
    # blocked pre-review by the janitor gate. 0 disables the guard.
    ci_run_never_created_grace_minutes: int = 5
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
    state_dir: str = layout.DEFAULT_STATE_DIR
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
        # Claude Code CLI's own account-level session-limit phrasing (observed
        # 2026-07-21 verbatim as "You've hit your session limit · resets
        # 4:40pm (America/Los_Angeles)"). Distinct wording from "rate limit"/
        # "usage limit" above, so it silently fell through _classify_session_
        # failure's marker match and every downstream reap path: reviewer
        # workers that died on this message got no throttled_until cooldown
        # and were relaunched straight into the same limit every stale-claim
        # interval (job-cannon PRs #1342/#1343/#1344/#1346 stuck 5.5-20+
        # hours in a redispatch loop before this was added).
        "hit your session limit",
    )
    # Narrow marker set for the reviewer session-limit reclassification in
    # _extract_review_session_summary (issue #651/#652). Unlike
    # throttle_error_markers — which includes generic substrings like "rate
    # limit" / "usage limit" appropriate for WORKER process log tails (workers
    # don't produce analysis prose) — this list contains only the CLI's own
    # specific session-limit death message phrasing. Reviewer sessions DO
    # produce analysis prose, and reviewer launches force tee_stream_json=True
    # (claude_code.py), making log_path and events_path byte-identical — so
    # any marker matched against the log tail is also matched against the
    # parsed assistant text. Generic markers would false-positive on
    # legitimate review commentary about rate-limit/quota code (this
    # codebase's domain), silently reclassifying real review work as launch
    # failures and defeating the #583 rollback guard this reclassification
    # exists to protect. The specific "hit your session limit" phrasing is the
    # CLI's own death message, not a domain term, so it is safe to match
    # against reviewer text. Extend via config when a new session-limit
    # phrasing is observed.
    session_limit_markers: tuple[str, ...] = ("hit your session limit",)
    # Bounded retry for transient GitHub API failures (TLS blips, connection
    # resets, gateway 5xx, secondary rate limits, etc.) in GitHub.run().
    # These knobs apply fleet-wide; keep them in RuntimeConfig so GitHub stays
    # a frozen value object with no mutable state.
    gh_max_retries: int = 3
    gh_retry_base_seconds: float = 1.0
    # Wall-clock ceiling on a single `gh` invocation. Without it a hung gh
    # process blocks the orchestrator loop pass forever: the pass never
    # completes, the supervisor's staleness watchdog eventually kills the
    # child, and every PR waiting on merge_ready stalls until a human
    # intervenes (observed 2026-08-05, loop pass 0636dca635de). Retries do not
    # help a call that never returns — only a timeout converts the hang into a
    # failure the existing retry/next-pass machinery can absorb.
    gh_timeout_seconds: float = 120.0
    # Pre-emptive GraphQL rate-limit guard. Before starting quota-heavy phases
    # (mop-up sweeps, merged-PR listings), GitHub.check_graphql_rate_limit()
    # verifies ``resources.graphql.remaining`` from ``gh api rate_limit`` is at
    # least this value. Set to 0 to disable the guard.
    graphql_rate_limit_threshold: int = 1500
    # Bounded in-memory event ring for state.json. A larger cap costs only a
    # few hundred KB of JSON and preserves far more diagnostic history when a
    # single sweep emits repetitive events. Tuned via config (issue #525).
    event_ring_size: int = 2000
    # Extra safety margin added to provider-reported rate-limit reset times
    # when computing the ``throttled_until`` defer deadline. Provider reset
    # estimates are floors, not guarantees; dispatching at T+0 races the
    # provider's actual reset. Default 90 seconds.
    throttle_resume_margin_s: int = 90
    # Issue #1088: max escalated issues the label self-heal sweep will verify
    # against GitHub in a single pass. The bound is mandatory, not defensive.
    # Every subject whose ``label_error`` key is absent costs one live
    # ``issue_view`` call, and measured at the time of the fix *every* escalated
    # subject was in that arm -- 8 in charlie-work and 49 in job-cannon. Sweeping
    # all 57 in one pass would add ~57 sequential ``gh`` subprocess calls to a
    # loop pass that is shared sequentially between both repos, which is the
    # starvation mechanism of #1078. Bounding converges over a handful of passes
    # instead of one long burst; verified subjects are then free forever (their
    # ``label_error`` is None, which costs a dict lookup and no API call). Set
    # to 0 for unlimited.
    escalated_label_repair_max_per_pass: int = 10


@dataclass(frozen=True)
class DevinConfig:
    # "manual" writes a session manifest for the operator; "command" runs a
    # blocking dispatch_command per issue; "devin-shell" launches headless
    # `devin` CLI sessions non-blocking with sidecar tracking (devin_shell.py);
    # "claude-code" launches Claude Code workers in isolated git worktrees
    # (claude_code.py, configured under the claude_code section).
    adapter: str = "manual"
    # Empty string means "derive from runtime.state_dir" (layout.py's
    # session_manifest_default/session_results_default) rather than a fixed
    # literal -- see paths.resolved_layout, the single place that resolves
    # this sentinel.
    session_manifest: str = ""
    session_results: str = ""
    dispatch_command: str | tuple[str, ...] = ""
    command_timeout_seconds: int = 300
    # devin-shell adapter: sidecar JSON + per-session logs live here. Empty
    # string means "derive from runtime.state_dir" (layout.sessions_dir_default)
    # -- see paths.resolved_layout.
    sessions_dir: str = ""
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

    # Every worker/reviewer launch pins this explicitly via `--model` — see
    # claude_code._apply_model_pin. Without an explicit pin, the spawned
    # `claude` CLI subprocess falls back to whatever model an interactive
    # session on this machine last set globally (e.g. via `/model`), which
    # is never guaranteed to be available/affordable for headless fleet
    # dispatch (2026-07-22 outage: an operator session's `/model` choice of
    # a premium tier silently propagated to every reviewer launch and hit a
    # credits wall, stalling every PR review fleet-wide with zero backoff
    # signal since the error didn't match the quota-exhaustion classifier).
    model: str = "claude-sonnet-5"
    # Effort level pinned via ``--effort`` on every worker/reviewer launch —
    # see claude_code._apply_effort_pin. Empty string means no pin (the CLI
    # uses its default effort). Mirrors the model pin: prevents ambient CLI
    # state from leaking into headless fleet sessions.
    effort: str = ""
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
class ApiProviderConfig:
    """Pricing and endpoint description for one Anthropic-compatible API provider.

    ``api_key_env`` is the *name* of the environment variable that holds the key;
    the key value never appears in config. ``cached_input_usd_per_mtok`` defaults
    to ``0.0`` for providers that do not advertise a cached-input discount.
    """

    base_url: str
    api_key_env: str
    model: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cached_input_usd_per_mtok: float = 0.0


@dataclass(frozen=True)
class ApiBudgetConfig:
    """Per-session and aggregate spending caps for the api-worker adapter.

    ``max_usd_per_session`` of ``0.0`` disables per-session enforcement. The
    remaining defaults are conservative starting values for paid-API usage.
    """

    max_usd_per_session: float = 0.0
    preflight_reserve_usd: float = 1.0
    max_usd_per_day: float = 5.0
    lifetime_usd: float = 15.0

    def __post_init__(self) -> None:
        for key in (
            "max_usd_per_session",
            "preflight_reserve_usd",
            "max_usd_per_day",
            "lifetime_usd",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"api_worker.budget.{key} must be a number, got {value!r}")
            if value < 0:
                raise ConfigError(f"api_worker.budget.{key} must be >= 0, got {value}")


@dataclass(frozen=True)
class ApiWorkerConfig:
    """Registry-backed config for the api-worker adapter.

    ``enabled`` defaults to ``False`` so an absent config block is a no-op. When
    ``enabled`` is ``True``, ``provider`` must name a key in ``providers`` and the
    selected provider must have a non-empty ``api_key_env``, positive input/output
    pricing, and non-negative cached-input pricing.
    ``providers`` is exposed as an immutable view so the registry cannot be
    mutated after config load.
    """

    enabled: bool = False
    provider: str = ""
    max_concurrent_sessions: int = 1
    providers: Mapping[str, ApiProviderConfig] = field(
        default_factory=lambda: MappingProxyType({})
    )
    budget: ApiBudgetConfig = field(default_factory=ApiBudgetConfig)
    fallback_adapter: str = "devin-shell"
    worker_template: str = "worker_claude_code.md"
    rework_template: str = "rework.md"

    def __post_init__(self) -> None:
        # Normalize to an immutable mapping view once at the boundary.
        if not isinstance(self.providers, MappingProxyType):
            providers_dict: dict[str, ApiProviderConfig] = {}
            for name, cfg in self.providers.items():
                if isinstance(cfg, ApiProviderConfig):
                    providers_dict[str(name)] = cfg
                elif isinstance(cfg, dict):
                    providers_dict[str(name)] = ApiProviderConfig(**cfg)
                else:
                    raise ConfigError(
                        f"api_worker.providers[{name!r}] must be a mapping of "
                        f"ApiProviderConfig values, got {type(cfg).__name__}"
                    )
            object.__setattr__(self, "providers", MappingProxyType(providers_dict))

        if not self.enabled:
            return

        if not self.provider:
            raise ConfigError(
                "api_worker.provider must be a non-empty string when api_worker.enabled is true"
            )
        if self.provider not in self.providers:
            raise ConfigError(
                f"api_worker.provider '{self.provider}' is not a key in api_worker.providers"
            )

        active = self.providers[self.provider]
        if not active.api_key_env:
            raise ConfigError(
                f"api_worker.providers.{self.provider}.api_key_env must be a non-empty string"
            )
        for price_key in ("input_usd_per_mtok", "output_usd_per_mtok"):
            value = getattr(active, price_key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ConfigError(
                    f"api_worker.providers.{self.provider}.{price_key} must be > 0, got {value!r}"
                )
        cached_value = active.cached_input_usd_per_mtok
        if (
            not isinstance(cached_value, (int, float))
            or isinstance(cached_value, bool)
            or cached_value < 0
        ):
            raise ConfigError(
                "api_worker.providers."
                f"{self.provider}.cached_input_usd_per_mtok must be >= 0, got {cached_value!r}"
            )


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
    auto_verdict: bool = False
    # Issue #784: bounds the "content-free verdict -> forced regeneration ->
    # still content-free" cycle. Counts distinct parse-failure attempts (one
    # per genuinely new report/head, never per loop-pass re-read of a cached
    # one) per PR. Once exceeded, the PR is released from the cross-family
    # gate (recorded as a caveated "approved") instead of looping forever or
    # escalating to a human — see workflow._record_cross_family_verdicts.
    max_parse_failures: int = 2
    # Issue #1081: bounds how many times loop() will force review() to
    # regenerate an *unusable* cross-family report (a "(UNAVAILABLE)" failure
    # stub, or one carrying no head SHA) for one unchanged PR head. The bound
    # is required because regeneration runs the cross-family model
    # synchronously for up to ``timeout_seconds``; unbounded, a model that is
    # simply down burns that timeout on every pass and starves the other repo
    # in the shared sequential loop (#1078).
    #
    # This does NOT share max_parse_failures' terminal behaviour, and must not
    # be "unified" with it. That bound ends in a caveated "approved"; this one
    # ends in a human_needed escalation, because exhausting it means the head
    # was never confirmed and approving on an unconfirmed head is precisely the
    # fail-open #1079 closed.
    max_regen_attempts: int = 2


@dataclass(frozen=True)
class RescueConfig:
    """Bounded strong-model rescue tier (issue #555).

    Inserts exactly one Opus rework attempt + one cross-family (non-Claude)
    review pass between "cheap-worker cap exhausted" and escalating to a
    human. ``enabled`` defaults False so an absent config block is a no-op —
    mirrors ``CrossFamilyConfig`` (config.py:236).

    Only the three verdict-driven caps route through here
    (``max_rework_cycles``, ``max_conflict_rework_attempts``,
    ``max_no_op_rework_attempts``) — infra-driven caps (review-dispatch
    attempt cap, ``DETERMINISTIC_ESCALATION_FAILURE_KINDS``, dispatch-failed
    cap) always escalate straight to a human; a stronger model cannot fix a
    dead session or an unsafe worktree.

    The rescue rework reuses the existing claude-code rework-dispatch path
    (``_dispatch_rework_impl`` → ``adapters.dispatch_sessions`` →
    ``claude_code.launch_claude_worker``) with ``claude_code.model``
    overridden to ``worker_model`` for that one dispatch — never a parallel
    launch path. ``worker_adapter`` is currently always ``"claude-code"``;
    kept as an explicit field so a future adapter can be wired in by config
    alone, matching the spec's named knobs.

    The rescue review reuses ``cross_family.run_cross_family_review`` (the
    existing blocking, one-shot cross-family invocation) rather than a new
    polling worker session — ``reviewer_command`` empty means reuse
    ``CrossFamilyConfig.command`` with ``model`` overridden to
    ``reviewer_model``.
    """

    enabled: bool = False
    worker_adapter: str = "claude-code"
    worker_model: str = "claude-opus-4-1"
    reviewer_adapter: str = "devin"
    reviewer_model: str = "codex"
    # Empty means reuse CrossFamilyConfig.command (model still overridden to
    # reviewer_model above).
    reviewer_command: str | tuple[str, ...] = ()
    reviewer_timeout_seconds: int = 300


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
    # Issue #654: time-based escape for a dead dispatched worker whose drift
    # was surfaced by ``_detect_and_handle_orphaned_workers`` but whose PR
    # state did not qualify for auto-reset (clean exit with no push, a non-
    # request_changes decision, a head change without a review callback, ...).
    # Without this, the dispatch label holds indefinitely -- the label only
    # clears when a worker that no longer exists reports back, and there is no
    # time-based escape. After this many minutes since the drift was first
    # surfaced (``orphan_drift_at``), the issue is escalated to
    # ``agent:human-needed`` so a human can inspect the worktree for unpushed
    # commits. 0 disables the time-based escape (reverts to the pre-#654
    # hold-forever behavior). The default (60) matches the "1+ hour" window
    # in the issue title, which is trivially distinguishable from a
    # legitimately long worker session.
    dead_dispatched_reap_minutes: int = 60


@dataclass(frozen=True)
class WorktreeReclamationConfig:
    """Cadence-gated reclamation of merged-PR worker worktrees from the fleet
    pass (issue #636).

    ``clean_worktrees`` is the junction-safe, merge-gated, liveness-gated sweep
    that ``charlie worktree-clean`` runs on demand. Before this config drove a
    fleet-pass call site, reclamation never fired on the fleet's own cadence --
    worktrees for merged PRs accumulated indefinitely (77 of 81 dead on the
    host this was measured on; ``git worktree list`` became unusable as an
    operator instrument and ``du`` on the worktrees dir timed out).

    ``enabled`` defaults True so the fleet reclaims by default; set False to
    revert to operator-only ``charlie worktree-clean``. ``interval_minutes``
    gates the sweep so the per-candidate ``gh pr view`` cost is proportional to
    elapsed time, not to backlog size or loop frequency -- the sweep makes one
    live REST call per candidate worktree, so running it every pass against an
    80+ backlog would be a per-pass quota spike. The sweep itself is
    idempotent, merge-gated, liveness-gated, and fails closed on an erroring
    ``gh`` (see ``worktree.clean_worktrees``), which are the properties required
    to run it unattended.
    """

    enabled: bool = True
    interval_minutes: int = 60


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
    exempt_path_globs: tuple[str, ...] = (
        "*.md",
        "docs/**",
        "examples/**",
        ".github/workflows/**",
        "*.lock",
        "*.toml",
        "*.cfg",
        "*.ini",
    )
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
    # Empty string means "derive from runtime.state_dir"
    # (layout.notify_digest_default) rather than a fixed literal -- see
    # paths.resolved_layout, the single place that resolves this sentinel.
    file_path: str = ""


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
class MainCiReclaimConfig:
    """Per-pass reclaim of superseded, not-yet-started ``main`` CI runs (#863, #815).

    Distinct from ``RunnersConfig.cancel_superseded_main_runs`` above: that
    mechanism only fires inside this orchestrator's own successful-merge
    codepath, has no strict-ancestor check, and keeps only the single
    newest-by-``createdAt`` queued run. This section instead wires
    ``main_ci_reclaim.reclaim_superseded_main_ci_runs`` into every fleet
    loop pass regardless of merge source (Aviator, a direct ``gh pr merge``,
    or this orchestrator's own merge) -- see that module's docstring for the
    full rationale and safety invariant.

    ``enabled`` defaults True for the same reason as ``ReconcilePassConfig``:
    the repair direction is provably safe (only strict ancestors of main's
    current tip, verified via local git, and only not-yet-started runs,
    re-checked immediately before cancellation), so it is safe to run
    unattended on every pass. The knob exists for rollback, not opt-in.
    """

    enabled: bool = True
    workflow_filename: str = "ci.yml"


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
    ``max_pass_runtime_seconds``: upper bound on a single pass's wall-clock
    duration. The supervisor heartbeat freshness check uses this bound so a
    long-running pass is not mistaken for a dead supervisor (default 1800 s /
    30 min).
    ``self_deploy_failure_alarm``: consecutive ``self_deploy`` failures before
    a ``self_deploy_alarm`` events.db entry fires (default 3, mirrors
    ``AutoMergeConfig.failed_attempt_alarm``). 0 disables the alarm.
    ``zero_pass_alarm``: consecutive fleet-supervisor cycles that complete
    with zero repo passes, despite at least one repo being configured,
    before a ``supervisor_zero_pass_alarm`` events.db entry fires (default 3,
    mirrors ``self_deploy_failure_alarm``). 0 disables the alarm. A cycle
    with zero repos configured never counts toward this streak in either
    direction -- that is a configuration state, not an incident (issue #855).
    """

    poll_interval_seconds: int = 20
    full_pass_interval_seconds: int = 300
    active_cooldown_seconds: int = 30
    max_runtime_minutes: int = 0
    max_pass_runtime_seconds: int = 1800
    self_deploy_failure_alarm: int = 3
    zero_pass_alarm: int = 3


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
    quota_probe: QuotaProbeConfig = field(default_factory=QuotaProbeConfig)
    reconcile_pass: ReconcilePassConfig = field(default_factory=ReconcilePassConfig)
    deescalation: DeescalationConfig = field(default_factory=DeescalationConfig)
    auto_merge: AutoMergeConfig = field(default_factory=AutoMergeConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    devin: DevinConfig = field(default_factory=DevinConfig)
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)
    api_worker: ApiWorkerConfig = field(default_factory=ApiWorkerConfig)
    cross_family: CrossFamilyConfig = field(default_factory=CrossFamilyConfig)
    rescue: RescueConfig = field(default_factory=RescueConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    worktree_reclamation: WorktreeReclamationConfig = field(
        default_factory=WorktreeReclamationConfig
    )
    test_adequacy: TestAdequacyConfig = field(default_factory=TestAdequacyConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    runners: RunnersConfig = field(default_factory=RunnersConfig)
    main_ci_reclaim: MainCiReclaimConfig = field(default_factory=MainCiReclaimConfig)
    runner_scaling: RunnerScalingConfig = field(default_factory=RunnerScalingConfig)
    runner_allocation: RunnerAllocationConfig = field(default_factory=RunnerAllocationConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    post_mortem: PostMortemConfig = field(default_factory=PostMortemConfig)

    # Provenance, not values (issue #943). Paths of the config files that were
    # actually read to produce this value, in merge order (global layer first,
    # per-repo last). An empty tuple means *nothing was read* and every section
    # below is a dataclass default.
    #
    # Why this belongs on the value rather than only in a log line: a resolved
    # config records what a section *became*, never whether the file that
    # declares it was read, so `load_config()` with no path returns something
    # indistinguishable from a fully-configured fleet whose features happen to
    # be switched off. That ambiguity is the #590 diagnosis cost and it is what
    # `global_config.load_layered_config` already logs about but could not
    # hand to a caller.
    #
    # ``metadata={"provenance": True}`` is load-bearing: `load_config` derives
    # the set of valid YAML section names from `fields(OrchestratorConfig)`, so
    # without the marker a config file could declare `sources:` and assert its
    # own provenance -- a field whose entire purpose is to be trustworthy would
    # become the one field an untrusted input can forge. The marker is read
    # there rather than a hard-coded name so a second provenance field cannot
    # be added without inheriting the exclusion.
    #
    # ``compare=False`` because provenance is metadata about how a value was
    # obtained, not part of the value: two configs with identical sections are
    # the same config whether they came from one file, two, or none.
    sources: tuple[str, ...] = field(default=(), compare=False, metadata={"provenance": True})


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


def known_config_sections() -> frozenset[str]:
    """The section names ``load_config`` accepts at the top level.

    Derived from ``OrchestratorConfig``'s dataclass fields rather than a
    hand-maintained list, so a new section is automatically valid the moment
    its field is added. Provenance fields (see ``OrchestratorConfig.sources``)
    are excluded -- a config file must never be able to declare where it came
    from -- and that exclusion is keyed off field metadata rather than a name
    so it cannot be forgotten for a future provenance field.

    Shared with :func:`charlie_work.global_config.load_layered_config`, which
    needs the same set to reject an unknown *section name* before its
    merge drops an empty-bodied one -- see issue #962.
    """
    return frozenset(
        f.name for f in fields(OrchestratorConfig) if not f.metadata.get("provenance")
    )


def build_config_from_data(data: dict[str, Any]) -> OrchestratorConfig:
    """Validate an in-memory raw-YAML dict and build the ``OrchestratorConfig``.

    This is the shared validation core behind both config entry points:
    ``load_config`` reads a single file into a dict and calls this, and
    ``global_config.load_layered_config`` calls this directly on its merged
    dict -- no intermediate file, so the layered merge never touches disk
    (issue #704).

    ``sources`` is deliberately left at its dataclass default (``()``): a
    bare dict carries no path, so provenance is the caller's job. Both
    callers apply their own ``sources`` via ``dataclasses.replace`` on the
    value this returns.
    """
    # Section-processing below mutates the per-section sub-dicts in place
    # (e.g. coercing a YAML list to a tuple) -- that was always harmless when
    # this logic lived inline in ``load_config``, because its ``data`` was a
    # freshly-parsed dict nobody else held a reference to. Now that a caller
    # (``load_layered_config``) can hand in a dict it built and does not
    # discard, mutating it in place would be a new, surprising side effect on
    # the caller's object -- exactly what CLAUDE.md's immutability rule
    # forbids. A deep copy keeps this function's contract "reads a dict, does
    # not touch it" regardless of what future callers do with their copy.
    data = copy.deepcopy(data)
    # Validate top-level keys before processing sections.
    known_sections = known_config_sections()
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
    dispatch_staleness_minutes = dispatch_data.get("dispatch_staleness_minutes")
    if dispatch_staleness_minutes is not None and (
        isinstance(dispatch_staleness_minutes, bool)
        or not isinstance(dispatch_staleness_minutes, int)
    ):
        raise ConfigError(
            "config section 'dispatch' key 'dispatch_staleness_minutes' must be an int, "
            f"got {type(dispatch_staleness_minutes).__name__}"
        )
    if dispatch_staleness_minutes is not None and dispatch_staleness_minutes < 0:
        raise ConfigError(
            f"config section 'dispatch' key 'dispatch_staleness_minutes' must be >= 0, "
            f"got {dispatch_staleness_minutes}"
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
    # Worker-prompt comment controls (issue #872). ``DispatchConfig.__post_init__``
    # does the list->tuple coercion for every construction path; these checks exist
    # so a *config file* mistake fails loudly at load with the offending key named,
    # rather than silently degrading the prompt for every dispatched issue.
    for _seq_key in (
        "worker_prompt_comment_associations",
        "worker_prompt_excluded_comment_authors",
    ):
        _seq_value = dispatch_data.get(_seq_key)
        if _seq_value is not None:
            if not isinstance(_seq_value, list):
                raise ConfigError(
                    f"config section 'dispatch' key '{_seq_key}' must be a list of "
                    f"strings, got {type(_seq_value).__name__}"
                )
            for item in _seq_value:
                if not isinstance(item, str):
                    raise ConfigError(
                        f"config section 'dispatch' key '{_seq_key}' must be a list of "
                        f"strings, got element of type {type(item).__name__}"
                    )
            dispatch_data[_seq_key] = tuple(str(item) for item in _seq_value)
    for _int_key in ("worker_prompt_max_comments", "worker_prompt_max_comment_chars"):
        _int_value = dispatch_data.get(_int_key)
        if _int_value is not None:
            # bool is an int subclass; rejecting it explicitly (as the api_worker
            # section does) keeps `true` in YAML from silently meaning "1 comment".
            if isinstance(_int_value, bool) or not isinstance(_int_value, int):
                raise ConfigError(
                    f"config section 'dispatch' key '{_int_key}' must be an int, "
                    f"got {type(_int_value).__name__}"
                )
            if _int_value < 0:
                raise ConfigError(
                    f"config section 'dispatch' key '{_int_key}' must be >= 0, got {_int_value}"
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
    rd_max_attempts = review_dispatch_data.get("max_review_dispatch_attempts")
    if rd_max_attempts is not None and (
        isinstance(rd_max_attempts, bool) or not isinstance(rd_max_attempts, int)
    ):
        raise ConfigError(
            "config section 'review_dispatch' key 'max_review_dispatch_attempts' must be an int, "
            f"got {type(rd_max_attempts).__name__}"
        )
    if rd_max_attempts is not None and rd_max_attempts < 1:
        raise ConfigError(
            "config section 'review_dispatch' key 'max_review_dispatch_attempts' must be >= 1, "
            f"got {rd_max_attempts}"
        )
    rd_probe_max_interval = review_dispatch_data.get("quota_probe_max_interval_minutes")
    if rd_probe_max_interval is not None and (
        isinstance(rd_probe_max_interval, bool) or not isinstance(rd_probe_max_interval, int)
    ):
        raise ConfigError(
            "config section 'review_dispatch' key 'quota_probe_max_interval_minutes' must be "
            f"an int, got {type(rd_probe_max_interval).__name__}"
        )
    if rd_probe_max_interval is not None and rd_probe_max_interval < 0:
        raise ConfigError(
            "config section 'review_dispatch' key 'quota_probe_max_interval_minutes' must be "
            f">= 0, got {rd_probe_max_interval}"
        )
    rd_max_turns = review_dispatch_data.get("review_max_turns")
    if rd_max_turns is not None and (
        isinstance(rd_max_turns, bool) or not isinstance(rd_max_turns, int)
    ):
        raise ConfigError(
            "config section 'review_dispatch' key 'review_max_turns' must be an int, "
            f"got {type(rd_max_turns).__name__}"
        )
    if rd_max_turns is not None and rd_max_turns < 0:
        raise ConfigError(
            "config section 'review_dispatch' key 'review_max_turns' must be >= 0, "
            f"got {rd_max_turns}"
        )
    rd_diff_threshold = review_dispatch_data.get("diff_line_threshold")
    if rd_diff_threshold is not None and (
        isinstance(rd_diff_threshold, bool) or not isinstance(rd_diff_threshold, int)
    ):
        raise ConfigError(
            "config section 'review_dispatch' key 'diff_line_threshold' must be an int, "
            f"got {type(rd_diff_threshold).__name__}"
        )
    if rd_diff_threshold is not None and rd_diff_threshold < 0:
        raise ConfigError(
            "config section 'review_dispatch' key 'diff_line_threshold' must be >= 0, "
            f"got {rd_diff_threshold}"
        )
    rd_effort = review_dispatch_data.get("review_effort")
    if rd_effort is not None and not isinstance(rd_effort, str):
        raise ConfigError(
            "config section 'review_dispatch' key 'review_effort' must be a string, "
            f"got {type(rd_effort).__name__}"
        )
    rd_experiment_fraction = review_dispatch_data.get("review_effort_experiment_fraction")
    if rd_experiment_fraction is not None and (
        isinstance(rd_experiment_fraction, bool)
        or not isinstance(rd_experiment_fraction, (int, float))
    ):
        raise ConfigError(
            "config section 'review_dispatch' key 'review_effort_experiment_fraction' "
            f"must be a number, got {type(rd_experiment_fraction).__name__}"
        )
    if rd_experiment_fraction is not None and not (0.0 <= rd_experiment_fraction <= 1.0):
        raise ConfigError(
            "config section 'review_dispatch' key 'review_effort_experiment_fraction' "
            f"must be in [0.0, 1.0], got {rd_experiment_fraction}"
        )
    rd_experiment_salt = review_dispatch_data.get("review_effort_experiment_salt")
    if rd_experiment_salt is not None and not isinstance(rd_experiment_salt, str):
        raise ConfigError(
            "config section 'review_dispatch' key 'review_effort_experiment_salt' must be a "
            f"string, got {type(rd_experiment_salt).__name__}"
        )
    # Cross-field check: a fraction > 0.0 enables the experiment, whose
    # treatment arm IS review_effort verbatim (resolve_review_effort returns
    # it unmodified). Enabling the experiment without a treatment effort
    # would silently make "treatment" mean "no --effort pin at all" while
    # "control" still gets claude_code.effort -- a corrupted, undocumented
    # comparison that would run for the life of the experiment with no
    # warning. Fail loud at load instead.
    effective_fraction = rd_experiment_fraction if rd_experiment_fraction is not None else 0.0
    effective_effort = rd_effort if rd_effort is not None else ""
    if effective_fraction > 0.0 and not effective_effort:
        raise ConfigError(
            "config section 'review_dispatch': 'review_effort_experiment_fraction' is "
            f"{effective_fraction} but 'review_effort' is unset -- set 'review_effort' to the "
            "treatment effort string (e.g. 'high') before enabling the experiment"
        )
    review_dispatch = _build_section(ReviewDispatchConfig, "review_dispatch", review_dispatch_data)
    quota_probe_data = _section(data, "quota_probe")
    qp_enabled = quota_probe_data.get("enabled")
    if qp_enabled is not None and not isinstance(qp_enabled, bool):
        raise ConfigError(
            f"config section 'quota_probe' key 'enabled' must be a bool, "
            f"got {type(qp_enabled).__name__}"
        )
    qp_interval = quota_probe_data.get("interval_minutes")
    if qp_interval is not None and (
        isinstance(qp_interval, bool) or not isinstance(qp_interval, int)
    ):
        raise ConfigError(
            "config section 'quota_probe' key 'interval_minutes' must be an int, "
            f"got {type(qp_interval).__name__}"
        )
    if qp_interval is not None and qp_interval < 1:
        raise ConfigError(
            f"config section 'quota_probe' key 'interval_minutes' must be >= 1, got {qp_interval}"
        )
    qp_model = quota_probe_data.get("model")
    if qp_model is not None and not isinstance(qp_model, str):
        raise ConfigError(
            f"config section 'quota_probe' key 'model' must be a string, got {type(qp_model).__name__}"
        )
    if qp_model is not None and not qp_model.strip():
        raise ConfigError("config section 'quota_probe' key 'model' must not be empty")
    qp_timeout = quota_probe_data.get("timeout_seconds")
    if qp_timeout is not None and (
        isinstance(qp_timeout, bool) or not isinstance(qp_timeout, int)
    ):
        raise ConfigError(
            "config section 'quota_probe' key 'timeout_seconds' must be an int, "
            f"got {type(qp_timeout).__name__}"
        )
    if qp_timeout is not None and qp_timeout < 1:
        raise ConfigError(
            f"config section 'quota_probe' key 'timeout_seconds' must be >= 1, got {qp_timeout}"
        )
    qp_prompt = quota_probe_data.get("prompt")
    if qp_prompt is not None and not isinstance(qp_prompt, str):
        raise ConfigError(
            f"config section 'quota_probe' key 'prompt' must be a string, got {type(qp_prompt).__name__}"
        )
    if qp_prompt is not None and not qp_prompt.strip():
        raise ConfigError("config section 'quota_probe' key 'prompt' must not be empty")
    quota_probe = _build_section(QuotaProbeConfig, "quota_probe", quota_probe_data)
    reconcile_pass_data = _section(data, "reconcile_pass")
    rp_enabled = reconcile_pass_data.get("enabled")
    if rp_enabled is not None and not isinstance(rp_enabled, bool):
        raise ConfigError(
            f"config section 'reconcile_pass' key 'enabled' must be a bool, "
            f"got {type(rp_enabled).__name__}"
        )
    rp_interval = reconcile_pass_data.get("interval_minutes")
    if rp_interval is not None and (
        isinstance(rp_interval, bool) or not isinstance(rp_interval, int)
    ):
        raise ConfigError(
            "config section 'reconcile_pass' key 'interval_minutes' must be an int, "
            f"got {type(rp_interval).__name__}"
        )
    if rp_interval is not None and rp_interval < 1:
        raise ConfigError(
            f"config section 'reconcile_pass' key 'interval_minutes' must be >= 1, got {rp_interval}"
        )
    rp_alert_days = reconcile_pass_data.get("terminal_state_alert_days")
    if rp_alert_days is not None and (
        isinstance(rp_alert_days, bool) or not isinstance(rp_alert_days, int)
    ):
        raise ConfigError(
            "config section 'reconcile_pass' key 'terminal_state_alert_days' must be an int, "
            f"got {type(rp_alert_days).__name__}"
        )
    if rp_alert_days is not None and rp_alert_days < 1:
        raise ConfigError(
            "config section 'reconcile_pass' key 'terminal_state_alert_days' must be >= 1, "
            f"got {rp_alert_days}"
        )
    reconcile_pass = _build_section(ReconcilePassConfig, "reconcile_pass", reconcile_pass_data)
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
    readiness_no_ci_minutes = auto_merge_data.get("readiness_no_ci_minutes")
    if readiness_no_ci_minutes is not None:
        if isinstance(readiness_no_ci_minutes, bool) or not isinstance(
            readiness_no_ci_minutes, int
        ):
            raise ConfigError(
                "config section 'auto_merge' key 'readiness_no_ci_minutes' must be an int, "
                f"got {type(readiness_no_ci_minutes).__name__}"
            )
        if readiness_no_ci_minutes < 0:
            raise ConfigError(
                "config section 'auto_merge' key 'readiness_no_ci_minutes' must not be negative"
            )
    ci_run_never_created_grace_minutes = auto_merge_data.get("ci_run_never_created_grace_minutes")
    if ci_run_never_created_grace_minutes is not None:
        if isinstance(ci_run_never_created_grace_minutes, bool) or not isinstance(
            ci_run_never_created_grace_minutes, int
        ):
            raise ConfigError(
                "config section 'auto_merge' key 'ci_run_never_created_grace_minutes' "
                f"must be an int, got {type(ci_run_never_created_grace_minutes).__name__}"
            )
        if ci_run_never_created_grace_minutes < 0:
            raise ConfigError(
                "config section 'auto_merge' key 'ci_run_never_created_grace_minutes' "
                "must not be negative"
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
    session_limit_markers = runtime_data.get("session_limit_markers")
    if session_limit_markers is not None:
        if not isinstance(session_limit_markers, list):
            raise ConfigError(
                "config section 'runtime' key 'session_limit_markers' must be a list of "
                f"strings, got {type(session_limit_markers).__name__}"
            )
        for item in session_limit_markers:
            if not isinstance(item, str):
                raise ConfigError(
                    "config section 'runtime' key 'session_limit_markers' must be a list of "
                    f"strings, got element of type {type(item).__name__}"
                )
        runtime_data["session_limit_markers"] = tuple(session_limit_markers)
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
    gh_timeout_seconds = runtime_data.get("gh_timeout_seconds")
    if gh_timeout_seconds is not None:
        if isinstance(gh_timeout_seconds, bool) or not isinstance(
            gh_timeout_seconds, (int, float)
        ):
            raise ConfigError(
                "config section 'runtime' key 'gh_timeout_seconds' must be a number, "
                f"got {type(gh_timeout_seconds).__name__}"
            )
        # Rejected rather than silently coerced: 0/negative would mean "time out
        # instantly", turning every gh call into a failure. There is no
        # "disable" value on purpose — an unbounded gh call is the defect.
        if gh_timeout_seconds <= 0:
            raise ConfigError(
                "config section 'runtime' key 'gh_timeout_seconds' must be > 0, "
                f"got {gh_timeout_seconds}"
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
    event_ring_size = runtime_data.get("event_ring_size")
    if event_ring_size is not None:
        # bool is an int subclass, so a bare isinstance(..., int) accepts
        # `event_ring_size: true` and silently uses it as 1 -- a two-entry ring
        # that looks configured. Reject it explicitly, matching
        # escalated_label_repair_max_per_pass below.
        if not isinstance(event_ring_size, int) or isinstance(event_ring_size, bool):
            raise ConfigError(
                "config section 'runtime' key 'event_ring_size' must be an int, "
                f"got {type(event_ring_size).__name__}"
            )
        # >= 1, not >= 0: append_event truncates via events[-max_size:], and
        # -0 == 0 in Python, so max_size=0 would yield events[0:] (the FULL
        # list) — no truncation, i.e. unbounded growth, the exact failure this
        # cap exists to prevent. There is no sensible "disable" semantic for a
        # bounded ring (unlike graphql_rate_limit_threshold: 0), so reject 0.
        if event_ring_size < 1:
            raise ConfigError(
                "config section 'runtime' key 'event_ring_size' must be >= 1, "
                f"got {event_ring_size}"
            )
    throttle_resume_margin_s = runtime_data.get("throttle_resume_margin_s")
    if throttle_resume_margin_s is not None:
        if not isinstance(throttle_resume_margin_s, int):
            raise ConfigError(
                "config section 'runtime' key 'throttle_resume_margin_s' must be an int, "
                f"got {type(throttle_resume_margin_s).__name__}"
            )
        if throttle_resume_margin_s < 0:
            raise ConfigError(
                "config section 'runtime' key 'throttle_resume_margin_s' must be >= 0, "
                f"got {throttle_resume_margin_s}"
            )
    repair_cap = runtime_data.get("escalated_label_repair_max_per_pass")
    if repair_cap is not None:
        if not isinstance(repair_cap, int) or isinstance(repair_cap, bool):
            raise ConfigError(
                "config section 'runtime' key 'escalated_label_repair_max_per_pass' "
                f"must be an int, got {type(repair_cap).__name__}"
            )
        # 0 means unlimited here (matching graphql_rate_limit_threshold's
        # "0 disables the guard"), so only negatives are rejected.
        if repair_cap < 0:
            raise ConfigError(
                "config section 'runtime' key 'escalated_label_repair_max_per_pass' "
                f"must be >= 0, got {repair_cap}"
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
    effort_value = claude_code_data.get("effort")
    if effort_value is not None and not isinstance(effort_value, str):
        raise ConfigError(
            "config section 'claude_code' key 'effort' must be a string, "
            f"got {type(effort_value).__name__}"
        )
    claude_code = _build_section(ClaudeCodeConfig, "claude_code", claude_code_data)
    api_worker_data = _section(data, "api_worker")
    enabled_value = api_worker_data.get("enabled")
    if enabled_value is not None and not isinstance(enabled_value, bool):
        raise ConfigError(
            "config section 'api_worker' key 'enabled' must be a bool, "
            f"got {type(enabled_value).__name__}"
        )
    provider_value = api_worker_data.get("provider")
    if provider_value is not None and not isinstance(provider_value, str):
        raise ConfigError(
            "config section 'api_worker' key 'provider' must be a string, "
            f"got {type(provider_value).__name__}"
        )
    max_concurrent_sessions = api_worker_data.get("max_concurrent_sessions")
    if max_concurrent_sessions is not None:
        if isinstance(max_concurrent_sessions, bool) or not isinstance(
            max_concurrent_sessions, int
        ):
            raise ConfigError(
                "config section 'api_worker' key 'max_concurrent_sessions' must be an int, "
                f"got {type(max_concurrent_sessions).__name__}"
            )
        if max_concurrent_sessions < 0:
            raise ConfigError(
                "config section 'api_worker' key 'max_concurrent_sessions' must be >= 0, "
                f"got {max_concurrent_sessions}"
            )
    for str_key in ("fallback_adapter", "worker_template", "rework_template"):
        str_value = api_worker_data.get(str_key)
        if str_value is not None and not isinstance(str_value, str):
            raise ConfigError(
                f"config section 'api_worker' key '{str_key}' must be a string, "
                f"got {type(str_value).__name__}"
            )

    # Parse budget sub-section.
    budget_data = api_worker_data.get("budget")
    if budget_data is not None:
        if not isinstance(budget_data, dict):
            raise ConfigError(
                "config section 'api_worker' key 'budget' must be a mapping, "
                f"got {type(budget_data).__name__}"
            )
        budget_fields = {f.name for f in fields(ApiBudgetConfig)}
        unknown_budget_keys = sorted(set(budget_data) - budget_fields)
        if unknown_budget_keys:
            raise ConfigError(
                "config section 'api_worker' key 'budget' has unknown key(s): "
                f"{', '.join(unknown_budget_keys)} "
                f"(valid: {', '.join(sorted(budget_fields))})"
            )
        for budget_key in budget_fields:
            if budget_key in budget_data:
                budget_value = budget_data[budget_key]
                if not isinstance(budget_value, (int, float)) or isinstance(budget_value, bool):
                    raise ConfigError(
                        f"config section 'api_worker' key 'budget.{budget_key}' must be a number, "
                        f"got {type(budget_value).__name__}"
                    )
                if budget_value < 0:
                    raise ConfigError(
                        f"config section 'api_worker' key 'budget.{budget_key}' must be >= 0, "
                        f"got {budget_value}"
                    )
        api_worker_data["budget"] = ApiBudgetConfig(**budget_data)

    # Parse providers registry.
    providers_data = api_worker_data.get("providers")
    if providers_data is not None:
        if not isinstance(providers_data, dict):
            raise ConfigError(
                "config section 'api_worker' key 'providers' must be a mapping, "
                f"got {type(providers_data).__name__}"
            )
        provider_fields = {f.name for f in fields(ApiProviderConfig)}
        built_providers: dict[str, ApiProviderConfig] = {}
        for name, provider_data in providers_data.items():
            if not isinstance(provider_data, dict):
                raise ConfigError(
                    f"config section 'api_worker' key 'providers.{name}' must be a mapping, "
                    f"got {type(provider_data).__name__}"
                )
            unknown_provider_keys = sorted(set(provider_data) - provider_fields)
            if unknown_provider_keys:
                raise ConfigError(
                    f"config section 'api_worker' key 'providers.{name}' has unknown key(s): "
                    f"{', '.join(unknown_provider_keys)} "
                    f"(valid: {', '.join(sorted(provider_fields))})"
                )
            required_provider_keys = (
                "base_url",
                "api_key_env",
                "model",
                "input_usd_per_mtok",
                "output_usd_per_mtok",
            )
            for req in required_provider_keys:
                if req not in provider_data:
                    raise ConfigError(
                        f"config section 'api_worker' key 'providers.{name}' is missing "
                        f"required key '{req}'"
                    )
            for str_provider_key in ("base_url", "api_key_env", "model"):
                str_provider_value = provider_data.get(str_provider_key)
                if not isinstance(str_provider_value, str) or not str_provider_value:
                    raise ConfigError(
                        f"config section 'api_worker' key 'providers.{name}.{str_provider_key}' "
                        "must be a non-empty string"
                    )
            for price_key in (
                "input_usd_per_mtok",
                "output_usd_per_mtok",
                "cached_input_usd_per_mtok",
            ):
                if price_key in provider_data:
                    price_value = provider_data[price_key]
                    if not isinstance(price_value, (int, float)) or isinstance(price_value, bool):
                        raise ConfigError(
                            f"config section 'api_worker' key 'providers.{name}.{price_key}' "
                            f"must be a number, got {type(price_value).__name__}"
                        )
            built_providers[str(name)] = ApiProviderConfig(**provider_data)
        api_worker_data["providers"] = MappingProxyType(built_providers)

    api_worker = _build_section(ApiWorkerConfig, "api_worker", api_worker_data)
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
    rescue_data = _section(data, "rescue")
    rescue_enabled = rescue_data.get("enabled")
    if rescue_enabled is not None and not isinstance(rescue_enabled, bool):
        raise ConfigError(
            f"config section 'rescue' key 'enabled' must be a bool, "
            f"got {type(rescue_enabled).__name__}"
        )
    for rescue_str_key in (
        "worker_adapter",
        "worker_model",
        "reviewer_adapter",
        "reviewer_model",
    ):
        rescue_str_value = rescue_data.get(rescue_str_key)
        if rescue_str_value is not None and not isinstance(rescue_str_value, str):
            raise ConfigError(
                f"config section 'rescue' key '{rescue_str_key}' must be a string, "
                f"got {type(rescue_str_value).__name__}"
            )
    rescue_command = rescue_data.get("reviewer_command")
    if isinstance(rescue_command, list):
        rescue_data["reviewer_command"] = tuple(str(item) for item in rescue_command)
    rescue_command = rescue_data.get("reviewer_command")
    if rescue_command:
        _validate_command_placeholders(
            rescue_command,
            {"prompt_path", "model"},
            "rescue.reviewer_command",
        )
    rescue_timeout = rescue_data.get("reviewer_timeout_seconds")
    if rescue_timeout is not None and (
        isinstance(rescue_timeout, bool) or not isinstance(rescue_timeout, int)
    ):
        raise ConfigError(
            "config section 'rescue' key 'reviewer_timeout_seconds' must be an int, "
            f"got {type(rescue_timeout).__name__}"
        )
    if rescue_timeout is not None and rescue_timeout < 0:
        raise ConfigError(
            "config section 'rescue' key 'reviewer_timeout_seconds' must be >= 0, "
            f"got {rescue_timeout}"
        )
    rescue = _build_section(RescueConfig, "rescue", rescue_data)
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
    worktree_reclamation_data = _section(data, "worktree_reclamation")
    wr_enabled = worktree_reclamation_data.get("enabled")
    if wr_enabled is not None and not isinstance(wr_enabled, bool):
        raise ConfigError(
            "config section 'worktree_reclamation' key 'enabled' must be a bool, "
            f"got {type(wr_enabled).__name__}"
        )
    wr_interval = worktree_reclamation_data.get("interval_minutes")
    if wr_interval is not None and (
        isinstance(wr_interval, bool) or not isinstance(wr_interval, int)
    ):
        raise ConfigError(
            "config section 'worktree_reclamation' key 'interval_minutes' must be an int, "
            f"got {type(wr_interval).__name__}"
        )
    if wr_interval is not None and wr_interval < 1:
        raise ConfigError(
            "config section 'worktree_reclamation' key 'interval_minutes' must be >= 1, "
            f"got {wr_interval}"
        )
    worktree_reclamation = _build_section(
        WorktreeReclamationConfig, "worktree_reclamation", worktree_reclamation_data
    )
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
    main_ci_reclaim_data = _section(data, "main_ci_reclaim")
    mcr_enabled = main_ci_reclaim_data.get("enabled")
    if mcr_enabled is not None and not isinstance(mcr_enabled, bool):
        raise ConfigError(
            "config section 'main_ci_reclaim' key 'enabled' must be a bool, "
            f"got {type(mcr_enabled).__name__}"
        )
    mcr_workflow_filename = main_ci_reclaim_data.get("workflow_filename")
    if mcr_workflow_filename is not None and not isinstance(mcr_workflow_filename, str):
        raise ConfigError(
            "config section 'main_ci_reclaim' key 'workflow_filename' must be a string, "
            f"got {type(mcr_workflow_filename).__name__}"
        )
    main_ci_reclaim = _build_section(MainCiReclaimConfig, "main_ci_reclaim", main_ci_reclaim_data)
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
    runner_allocation_data = _section(data, "runner_allocation")
    for numeric_key in (
        "max_running_runners",
        "min_running_per_repo",
        "demand_idle_samples",
        "max_runs_scanned",
    ):
        value = runner_allocation_data.get(numeric_key)
        # bool is an int subclass; reject it so `max_running_runners: true`
        # fails loudly instead of silently allocating one slot.
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ConfigError(
                f"config section 'runner_allocation' key '{numeric_key}' must be an int, "
                f"got {type(value).__name__}"
            )
        if isinstance(value, int) and not isinstance(value, bool) and value < 0:
            raise ConfigError(
                f"config section 'runner_allocation' key '{numeric_key}' must be >= 0, got {value}"
            )
    managed_root_value = runner_allocation_data.get("managed_root")
    if managed_root_value is not None and not isinstance(managed_root_value, str):
        raise ConfigError(
            "config section 'runner_allocation' key 'managed_root' must be a string, "
            f"got {type(managed_root_value).__name__}"
        )
    allocation_enabled = runner_allocation_data.get("enabled")
    if allocation_enabled is not None and not isinstance(allocation_enabled, bool):
        raise ConfigError(
            "config section 'runner_allocation' key 'enabled' must be a bool, "
            f"got {type(allocation_enabled).__name__}"
        )
    runner_allocation = _build_section(
        RunnerAllocationConfig, "runner_allocation", runner_allocation_data
    )
    # Cross-section floor check (issue #600): runner_scaling and runner_allocation
    # both declare a "minimum runners per repo" floor on different axes -- scaling
    # keeps runners *registered* (provisioning/deregistering), allocation keeps
    # listeners *running* (starting/stopping already-configured listeners). When
    # both are enabled, an allocation floor higher than the scaling floor is
    # unsatisfiable: allocation caps each repo's target at its registered runner
    # count (runner_allocation.plan_allocation), so min_running_per_repo >
    # min_runners silently degrades to min_runners with nothing reconciling the
    # two. Reject it at load time rather than documenting the caveat. The reverse
    # (min_runners > min_running_per_repo) is a legitimate buffer -- registered
    # but parked runners that allocation promotes on demand -- and is allowed.
    if (
        runner_scaling.enabled
        and runner_allocation.enabled
        and runner_allocation.min_running_per_repo > runner_scaling.min_runners
    ):
        raise ConfigError(
            "config sections 'runner_scaling' and 'runner_allocation' are both "
            "enabled but their floors disagree: runner_allocation."
            f"min_running_per_repo={runner_allocation.min_running_per_repo} "
            f"exceeds runner_scaling.min_runners={runner_scaling.min_runners}. "
            "Allocation cannot keep more listeners running than scaling "
            "provisions; raise runner_scaling.min_runners to at least the "
            "allocation floor."
        )
    supervisor_data = _section(data, "supervisor")
    for int_key in (
        "poll_interval_seconds",
        "full_pass_interval_seconds",
        "active_cooldown_seconds",
        "max_runtime_minutes",
        "max_pass_runtime_seconds",
        "self_deploy_failure_alarm",
        "zero_pass_alarm",
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
        quota_probe=quota_probe,
        reconcile_pass=reconcile_pass,
        auto_merge=auto_merge,
        runtime=runtime,
        devin=devin,
        claude_code=claude_code,
        api_worker=api_worker,
        cross_family=cross_family,
        rescue=rescue,
        watchdog=watchdog,
        worktree_reclamation=worktree_reclamation,
        test_adequacy=test_adequacy,
        fleet=fleet,
        notify=notify,
        runners=runners,
        main_ci_reclaim=main_ci_reclaim,
        runner_scaling=runner_scaling,
        runner_allocation=runner_allocation,
        supervisor=supervisor,
        post_mortem=post_mortem,
        # ``sources`` is left at its dataclass default here -- this function
        # only ever sees a dict, never a path. ``load_config`` below (and
        # ``load_layered_config``) are the ones that know what path(s) the
        # data came from, and they attach that provenance with ``replace``.
    )


def load_config(path: Path | None = None) -> OrchestratorConfig:
    # One binding for both the read and the provenance it produces. Deriving
    # ``sources`` from a *second* ``path.exists()`` call would let the two
    # disagree if the file is created or removed between them, which is exactly
    # the class of lie this field exists to prevent.
    source_path = path if path is not None and path.exists() else None
    raw = (
        yaml.safe_load(source_path.read_text(encoding="utf-8")) if source_path is not None else {}
    )
    data = raw if isinstance(raw, dict) else {}
    return replace(
        build_config_from_data(data),
        # ``path`` as given, not ``resolve()``d: this is the string the caller
        # passed and the one the layered-config log lines print, so the two are
        # directly comparable, and resolve() can raise on paths exists() already
        # tolerated.
        sources=(str(source_path),) if source_path is not None else (),
    )
