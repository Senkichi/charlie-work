"""Types for the stalled-session lane: facts in, requests and commits out.

Same decide -> apply -> decide-again shape as the orphan sweep (see
``decide.py``), scoped to one worker at a time: ``decide_stalled`` replays a
single worker's flow against the results observed so far. Requests are frozen
and keyed by issue number (the ``WorkerView`` itself stays in the shell and in
the facts); commits are compared by equality only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..config import OrchestratorConfig
from ..worker import WorkerView

# ---------------------------------------------------------------- facts


@dataclass(frozen=True)
class StalledFacts:
    """One worker plus the clock and gate state ``decide_stalled`` may read."""

    worker: WorkerView
    config: OrchestratorConfig
    now: datetime
    dry_run: bool


# ---------------------------------------------------------------- requests


@dataclass(frozen=True)
class ReadAdapterProfile:
    """The worker's adapter profile, reduced to the two predicates the lane reads."""

    issue: int


@dataclass(frozen=True)
class ProbeHealth:
    """Real-activity probe, health classification and the next inconclusive count."""

    issue: int


@dataclass(frozen=True)
class ProbeRateLimitDefer:
    """Log-tail rate-limit signature: the defer deadline, or ``None``."""

    issue: int


@dataclass(frozen=True)
class KillTree:
    """Kill the worker's process tree (start-time verified)."""

    issue: int


@dataclass(frozen=True)
class KillOrphans:
    """Sweep and kill processes that survived the tree kill (worktree-scoped)."""

    issue: int


@dataclass(frozen=True)
class RecordFailure:
    """Classify the sidecar via the adapter profile: ``(failure_kind, throttled_until)``."""

    issue: int


@dataclass(frozen=True)
class ReadLogTail:
    """The log's last line and mtime, for the reap event payload."""

    issue: int


STALLED_REQUEST_TYPES = (
    ReadAdapterProfile,
    ProbeHealth,
    ProbeRateLimitDefer,
    KillTree,
    KillOrphans,
    RecordFailure,
    ReadLogTail,
)

# ---------------------------------------------------------------- results


@dataclass(frozen=True)
class AdapterFacts:
    over_budget: bool
    can_record_failure: bool


@dataclass(frozen=True)
class HealthProbe:
    probe: Any  # RealActivityProbe | None
    health: Any  # WorkerHealth
    next_inconclusive_count: int


@dataclass(frozen=True)
class LogTail:
    last_line: str | None
    mtime: str


# ---------------------------------------------------------------- commits


@dataclass(frozen=True)
class StampSidecar:
    """``update_worker_log_stat`` with exactly these keyword fields."""

    issue: int
    fields: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class MarkBudgetExceeded:
    """Stamp ``failure_kind="budget_exceeded"`` on the api sidecar."""

    issue: int


@dataclass(frozen=True)
class RecordRoleQuota:
    """``role_quota_ledger.record_for_session``: restrict the session's stamped
    (harness, model) chain entry until ``until`` (issue #2086)."""

    issue: int
    until: str
    reason: str
    source: str


@dataclass(frozen=True)
class RecordPostMortem:
    """``classify_and_record``: post-mortem extraction into the sidecar."""

    issue: int


@dataclass(frozen=True)
class ThrottleArm:
    until: Any
    source: str
    reason: str
    adapter_kind: str


@dataclass(frozen=True)
class FailureStamp:
    issue: int
    kind: str
    adapter_kind: str


@dataclass(frozen=True)
class StateTxn:
    """One ``state_lock`` window: arm throttle, stamp failure, append an event, save once."""

    throttle: ThrottleArm | None = None
    stamp: FailureStamp | None = None
    event_kind: str | None = None
    event_payload: tuple[tuple[str, Any], ...] = ()


STALLED_COMMIT_TYPES = (
    StampSidecar,
    MarkBudgetExceeded,
    RecordRoleQuota,
    RecordPostMortem,
    StateTxn,
)


@dataclass(frozen=True)
class StalledPlan:
    requests: tuple[Any, ...]
    commits: tuple[Any, ...]
    done: bool
    entry: dict[str, int] | None  # the ``{"issue", "pid"}`` row for the return value
