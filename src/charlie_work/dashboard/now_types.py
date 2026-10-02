"""Value objects for the dashboard "Now" view (spec section 4).

Two groups: the *input* (``SourcesRead`` and friends, what the collector hands the
model after reading fleet sources) and the *output* (``NowModel``). Every
GitHub-derived field carries ``as_of_snapshot`` so the renderer can label it
"as of snapshot" instead of presenting it as live (spec section 6).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from ..config import LabelConfig
from .sources import JsonRead, SnapshotRead


class FindingLike(Protocol):
    """Duck type of ``heartbeat_alarms.Finding`` (built on another branch).

    Declared locally so this package does not import that module.
    """

    check: str
    repo: str
    severity: str  # "ok" | "warn" | "anomaly"
    detail: str
    facts: Mapping[str, Any]


@dataclass(frozen=True)
class RepoRead:
    """One registered repo as the collector read it."""

    key: str
    repo_root: str
    snapshot: SnapshotRead
    # Live reviewers (``<state_dir>/dispatches/reviews``); None when unreadable.
    reviewers_live: int | None = None
    # ``dispatch.max_concurrent_sessions``; 0 / None mean unlimited.
    worker_cap: int | None = None
    # ``review_dispatch.max_concurrent_reviews``.
    review_cap: int | None = None
    # (issue number, when it entered its terminal escalation) from state.json
    # ``terminal_since``; supplies the age of Operator-queue rows.
    escalated_since: tuple[tuple[int, datetime], ...] = ()


@dataclass(frozen=True)
class SourcesRead:
    """Everything ``build_now_model`` needs, already read (the model does no I/O)."""

    repos: tuple[RepoRead, ...]
    labels: LabelConfig = field(default_factory=LabelConfig)
    # ``fleet.global_max_concurrent_sessions`` / ``global_max_concurrent_reviews``.
    global_worker_cap: int | None = None
    global_review_cap: int | None = None
    # Newest global ``runner_allocation`` event (``sources.latest_event`` shape:
    # ``ts`` plus decoded ``payload``); None when absent.
    runner_allocation: Mapping[str, Any] | None = None
    # Busy listeners per repo from the newest ``runner_health`` totals.
    runner_busy: Mapping[str, int] = field(default_factory=dict)
    supervisor_heartbeat: JsonRead | None = None
    # ``fleet.json``-pause block (``fleet_pause.fleet_pause_status`` shape) or None.
    pause: Mapping[str, Any] | None = None
    # Merged-in-24h count derived by the collector from events; None = not derivable.
    done_24h: int | None = None
    # Cadence of the dashboard collector; widens the stale threshold (see now_cadence).
    collector_interval_seconds: float = 30.0
    # p90 of seconds between completed loop passes over the last 24h (pooled across repos);
    # None when too few passes were observed. Drives the stale threshold (see now_cadence).
    snapshot_gap_p90_seconds: float | None = None


@dataclass(frozen=True)
class RepoFreshness:
    repo: str
    snapshot_written_at: datetime | None
    age_seconds: float | None
    stale: bool
    error: str | None


@dataclass(frozen=True)
class NeedsMeItem:
    """One exception-zone row: where, how old, why, and what to type."""

    kind: str  # operator_queue | human_needed | alarm | stale_source | paused | supervisor
    severity: str  # anomaly | action | warn
    repo: str
    age_seconds: float | None
    reason: str
    command: str | None
    as_of_snapshot: bool
    # An alternative remedy shown beneath the primary command (e.g. requeue vs verdict).
    secondary_command: str | None = None
    # Issue number the row is about (None for fleet-level rows); the final sort tie-break.
    number: int | None = None
    # The decision the operator makes (one of ``NEEDS_ME_GROUPS``); derived from ``kind``
    # in ``now_needs_me`` so a constructor cannot disagree with the grouping.
    group: str = "Exceptions"


# Needs-me groups in display order, by the decision the operator makes.
NEEDS_ME_GROUPS = ("Exceptions", "Awaiting your verdict", "Human needed", "Operator queue")


@dataclass(frozen=True)
class GroupSummary:
    """Fleet-level rollup of one Needs-me group (every group is present, count may be 0)."""

    group: str
    count: int
    oldest_age_seconds: float | None  # None when empty or no row has a known age


@dataclass(frozen=True)
class FlowStage:
    name: str  # Dispatchable | Queued | In progress | PR open | Reviewing | Needs rework
    label: str | None  # the LabelConfig value counted, None for Dispatchable
    count: int
    as_of_snapshot: bool = True


@dataclass(frozen=True)
class RepoFlow:
    """One repo's stage counts (same stage names/order as ``FlowModel.stages``)."""

    repo: str
    stages: tuple[FlowStage, ...]
    # Needs-me rows naming this repo (alarms and stale sources included).
    need_you: int


@dataclass(frozen=True)
class UnreachableReason:
    reason: str
    count: int
    examples: tuple[str, ...]  # "repo#N"
    as_of_snapshot: bool = True


@dataclass(frozen=True)
class FlowModel:
    stages: tuple[FlowStage, ...]
    # None when the collector could not derive it (not in the snapshot).
    done_24h: int | None
    not_dispatchable: tuple[UnreachableReason, ...]


@dataclass(frozen=True)
class RepoWorkers:
    repo: str
    live: int
    cap: int | None


@dataclass(frozen=True)
class RunnerRepo:
    repo: str
    capacity: int
    online: int  # running listeners (online proxy, runners recon section 4)
    busy: int | None
    parked: int
    demand: int


@dataclass(frozen=True)
class CapacityModel:
    workers_live: int
    workers_cap: int | None
    workers_by_repo: tuple[RepoWorkers, ...]
    reviewers_live: int | None  # None = no repo's live reviewers could be measured
    reviewers_cap: int | None
    reviewers_by_repo: tuple[RepoWorkers, ...]
    runners: tuple[RunnerRepo, ...]
    runners_age_seconds: float | None
    runners_stale: bool
    # Capped demand: Dispatchable work exists while the worker budget is full.
    capped_demand_now: bool
    capped_repos: tuple[str, ...]


@dataclass(frozen=True)
class NowTotals:
    """Fleet sums of the snapshot's own counters (cross-checked against fleet status)."""

    ready_issues: int
    active_issues: int
    open_linked_prs: int
    unlinked_prs: int
    live_workers: int


@dataclass(frozen=True)
class NowModel:
    generated_at: datetime
    stale_threshold_seconds: float
    freshness: tuple[RepoFreshness, ...]
    needs_me: tuple[NeedsMeItem, ...]
    flow: FlowModel
    capacity: CapacityModel
    totals: NowTotals
    repos: tuple[RepoFlow, ...] = ()
    needs_me_groups: tuple[GroupSummary, ...] = ()
