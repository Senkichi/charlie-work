"""Types for the dead-worker sweep: facts in, requests and commits out.

The sweep is a decide -> apply -> decide-again loop (see ``decide.py``).
``decide`` is pure: it reads :class:`SweepFacts` plus an ``observed`` map of
results for requests the shell already ran, and answers with a
:class:`SweepPlan` -- at most one new *request* (an effect whose result the
decision still needs) and the *commits* (state/event/label effects whose
content is already decided) that precede it.

Requests are frozen and hashable (ints, strings and tuples only): a request's
value is its key in ``observed``. Context that is too big to hash -- issue
dicts, PR dicts, the mutable state copy -- stays in the shell and is looked up
by issue number. Results are arbitrary data. Commits are compared by equality
or position, never hashed, so their payloads may hold lists and dicts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from ..config import OrchestratorConfig

Phase = Literal["pre", "lock", "post"]


@dataclass(frozen=True)
class RepoFacts:
    """Which repo-shaped inputs the shell has (presence only, never paths)."""

    has_repo_root: bool
    has_worktrees: bool
    repo_root_is_path: bool


@dataclass(frozen=True)
class SweepFacts:
    """Everything ``decide`` may read besides ``observed``. Never mutated."""

    phase: Phase
    now: datetime
    stamp: str  # ISO-8601 timestamp the shell minted for this phase (``ports.stamp``)
    config: OrchestratorConfig
    snapshot: Mapping[str, Any]  # pre-lock state
    locked: Mapping[str, Any] | None  # state loaded under the lock (lock/post phases)
    pid_alive: Mapping[int, bool]  # per ``dispatched`` issue with a dict entry
    repo: RepoFacts
    review_available: bool  # a review callback was supplied


# ---------------------------------------------------------------- requests


@dataclass(frozen=True)
class ReadClock:
    """Refresh ``now`` (the pre-review block re-reads the clock in the original)."""


@dataclass(frozen=True)
class CollectLiveHandoff:
    """Stale live-PID handoff candidates; also resolves their fates."""


@dataclass(frozen=True)
class FetchOpenPrs:
    """Open PRs keyed by linked issue number."""


@dataclass(frozen=True)
class FetchOpenIssues:
    """Open issues by number, plus the cross-repo scope context."""


@dataclass(frozen=True)
class ResolveFate:
    issue: int
    stage: Literal["no_pr", "pushed"]
    remote_head_sha: str | None = None
    ahead_count: int | None = None


@dataclass(frozen=True)
class StripAndFlag:
    issue: int
    kind: Literal["worker_declared_blocked", "zero_artifact", "cross_repo_scope"]


@dataclass(frozen=True)
class ProbeZeroArtifact:
    issue: int


@dataclass(frozen=True)
class ProbeCrossRepoScope:
    issue: int


@dataclass(frozen=True)
class ParkOrReclaim:
    issue: int


@dataclass(frozen=True)
class ParkBackstop:
    orphans: tuple[int, ...]
    escalated: tuple[int, ...]


@dataclass(frozen=True)
class SalvagePush:
    issue: int
    branch: str


@dataclass(frozen=True)
class ProbeRemoteBranch:
    issue: int
    branch: str


@dataclass(frozen=True)
class ProbeWorktreeHead:
    issue: int
    branch: str


@dataclass(frozen=True)
class ReadReviewDecision:
    pr_number: int
    head_sha: str | None
    stage: str = ""


@dataclass(frozen=True)
class FetchPrView:
    pr_number: int


@dataclass(frozen=True)
class OpenPrForBranch:
    issue: int
    branch: str
    source: Literal["pushed_orphan", "live_handoff"]


@dataclass(frozen=True)
class AdvanceToPrOpen:
    issue: int


@dataclass(frozen=True)
class CreditDeadWorker:
    issue: int
    classify_log: bool = True


@dataclass(frozen=True)
class ReadTerminal:
    issue: int


@dataclass(frozen=True)
class ReadCompletedOutcome:
    issue: int
    pr_number: int
    live_head_sha: str
    branch: str


@dataclass(frozen=True)
class ReadBlockedOutcome:
    issue: int


@dataclass(frozen=True)
class ApplyOutcomes:
    routes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ReadAppliedHeads:
    """``APPLIED_HEADS_KEY`` as persisted after the outcome apply."""


@dataclass(frozen=True)
class Review:
    issue: int
    pr_number: int
    reason: str


@dataclass(frozen=True)
class DrainNoOp:
    routes: tuple[Any, ...]  # NoOpRoute values


Request = (
    ReadClock
    | CollectLiveHandoff
    | FetchOpenPrs
    | FetchOpenIssues
    | ResolveFate
    | StripAndFlag
    | ProbeZeroArtifact
    | ProbeCrossRepoScope
    | ParkOrReclaim
    | ParkBackstop
    | SalvagePush
    | ProbeRemoteBranch
    | ProbeWorktreeHead
    | ReadReviewDecision
    | FetchPrView
    | OpenPrForBranch
    | AdvanceToPrOpen
    | CreditDeadWorker
    | ReadTerminal
    | ReadCompletedOutcome
    | ReadBlockedOutcome
    | ApplyOutcomes
    | ReadAppliedHeads
    | Review
    | DrainNoOp
)
REQUEST_TYPES = tuple(Request.__args__)

# Requests the lock phase may issue (design section 8: lock-window discipline).
# Network I/O in the lock is limited to the two PR-open lanes the original ran
# there; everything else is a cheap local read.
LOCK_LEGAL = (
    OpenPrForBranch,
    AdvanceToPrOpen,
    CreditDeadWorker,
    ReadTerminal,
    ReadCompletedOutcome,
    ReadBlockedOutcome,
    ReadReviewDecision,
)
POST_LEGAL = (ApplyOutcomes, ReadAppliedHeads, Review, DrainNoOp)


# ----------------------------------------------------------------- results


@dataclass(frozen=True)
class LiveHandoffFound:
    candidates: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True)
class IssuesResult:
    issues_by_number: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True)
class FateResult:
    branch: str
    kind: str  # fate class name
    blocked_escalatable: bool  # Blocked fate whose outcome may reach the operator
    blocked_reason_kind: str = ""
    blocked_detail: str = ""
    throttled: bool = False  # fate is a provider-throttle death
    pushed_without_pr: bool = False
    worker_outcome: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LabelWrite:
    ok: bool
    removed_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScopeResult:
    passed: bool
    reason: str | None = None


@dataclass(frozen=True)
class BackstopResult:
    deferred: Mapping[int, str]
    reclaim_results: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True)
class SalvagePushResult:
    pushed: bool
    old_remote_sha: str | None = None
    new_remote_sha: str | None = None
    commit_count: int | None = None
    skip_reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class RemoteProbe:
    ahead_count: int | None
    ahead_error: str | None
    head_sha: str | None


@dataclass(frozen=True)
class ReviewDecisionFacts:
    decision: str | None
    stale: bool
    missing: bool
    reviewed_head_sha: str | None


@dataclass(frozen=True)
class PrOpenResult:
    pr_number: int | None
    error: str | None


@dataclass(frozen=True)
class CreditResult:
    failure_kind: str | None
    throttled_until: str | None


@dataclass(frozen=True)
class TerminalFacts:
    exit_code: Any
    duration_seconds: Any


@dataclass(frozen=True)
class ReviewResult:
    ok: bool
    routed_to_rework: bool
    closed_unmerged_converged: bool
    escalation_deferred_live_worker: bool
    is_no_op_rework: bool
    raised_error: str | None
    # Fresh state read under the drain's lock, after the callback returned.
    entry_status: str | None
    pr_reviewed_head_sha: str | None
    entry_branch: str | None


# ----------------------------------------------------------------- commits


@dataclass(frozen=True)
class Emit:
    kind: str
    payload: Mapping[str, Any]
    level: str | None = None


@dataclass(frozen=True)
class UpdateIssue:
    """Merge-update of one issue entry (ADR-0001: never a dict replace)."""

    issue: int
    set_fields: Mapping[str, Any] = field(default_factory=dict)
    clear_fields: tuple[str, ...] = ()
    require_status: str | None = None  # post-phase guard, checked on a fresh load
    require_pr_reviewed_head: str | None = None


@dataclass(frozen=True)
class Escalate:
    issue: int
    reason: str
    reason_class: str
    pr_number: int | None = None
    issue_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReportStaleEvidence:
    bucket: Literal["live_handoff", "no_pr", "swept"]


@dataclass(frozen=True)
class RoutePreReviewRework:
    issue: int
    pr_number: int
    reason: str


@dataclass(frozen=True)
class TransitionLabel:
    issue: int
    edge: Literal["escalated", "rework_requested"]
    persist_label_error: bool = False


Commit = (
    Emit | UpdateIssue | Escalate | ReportStaleEvidence | RoutePreReviewRework | TransitionLabel
)
COMMIT_TYPES = tuple(Commit.__args__)


@dataclass(frozen=True)
class SweepPlan:
    requests: tuple[Request, ...]
    commits: tuple[Commit, ...]


@dataclass(frozen=True)
class NoOpRoute:
    issue_number: int
    pr_number: int
    live_head_sha: str | None
    reason: str
    branch: str | None


@dataclass(frozen=True)
class ReviewRoute:
    issue_number: int
    pr_number: int
    reviewed_head_sha: str | None
    live_head_sha: str | None
    fingerprint: str
    reason: str


@dataclass(frozen=True)
class PreOutcome:
    """What the pre phase decided, replayed by the lock and post phases."""

    orphans: tuple[int, ...]
    no_pr_orphans: tuple[int, ...]
    pr_by_issue: Mapping[int, Mapping[str, Any]]
    details: Mapping[int, Mapping[str, Any]]  # per no-PR issue facts
    candidates: Mapping[int, Mapping[str, Any]]  # pushed-branch candidates
    escalations: Mapping[int, tuple[str, Mapping[str, Any]]]  # kind, payload
    deferred: Mapping[int, str]  # park backstop deferrals
    reclaim_results: Mapping[int, Mapping[str, Any]]
    unreviewed: Mapping[int, tuple[str, ...]]  # issue -> sorted active labels
    live_candidates: Mapping[int, Mapping[str, Any]]
    pr_already_open: Mapping[int, Mapping[str, Any]]
    salvage_events: tuple[tuple[str, Mapping[str, Any]], ...]
    heads: Mapping[int, str | None]  # observed PR head per issue (post-salvage)
    early_exit: bool
    now: datetime  # the refreshed clock the lock phase uses


@dataclass(frozen=True)
class LockOutcome:
    outcome_apply_routes: tuple[tuple[int, int], ...]
    review_routes: tuple[ReviewRoute, ...]
    no_op_routes: tuple[NoOpRoute, ...]
    reap_escalations: tuple[int, ...]
