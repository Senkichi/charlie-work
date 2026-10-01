"""Frozen result types, input validation and shared helpers for the drill-down models.

Validation happens once, here, at the seam: a bad slug, number or correlation id comes back
as a ``DrillError`` value before any database is touched. Text fields hold raw (unescaped)
strings; the page renderer escapes them.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path

from ..metrics_base import open_dashboard_ro
from ..now_types import FlowStage, NeedsMeItem, RepoFreshness, RepoWorkers, RunnerRepo
from ..pages.now_fmt import valid_slug
from ..timeutil import local_iso

# ``correlation_context`` mints ``uuid4().hex[:12]``; callers may pass their own id, so accept
# any short token of id-safe characters (never a path, quote or whitespace).
_CORRELATION_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
MAX_NUMBER = 10**9


@dataclass(frozen=True)
class DrillError:
    """A drill-down that could not be produced. ``code``: invalid | not_found | unavailable."""

    code: str
    message: str


def valid_correlation_id(value: str) -> bool:
    return isinstance(value, str) and _CORRELATION_ID.fullmatch(value) is not None


def check_slug(slug: str) -> DrillError | None:
    if not isinstance(slug, str) or not valid_slug(slug):
        return DrillError("invalid", "not a repo slug (expected owner/name)")
    return None


def check_number(n: object) -> DrillError | None:
    if isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= MAX_NUMBER:
        return DrillError("invalid", "number must be a positive integer")
    return None


def local_text(ts: str | None, tz: tzinfo | None) -> str | None:
    """UTC ``ts`` as local ISO text; an unparseable value is kept verbatim, not dropped."""
    if not ts:
        return None
    try:
        return local_iso(ts, tz)
    except ValueError:
        return ts


def open_history(path: Path) -> tuple[sqlite3.Connection | None, DrillError | None]:
    conn, err = open_dashboard_ro(path)
    if conn is None:
        return None, DrillError("unavailable", f"history unavailable: {err}")
    return conn, None


def history_start(db: sqlite3.Connection, source: str) -> str | None:
    """Earliest event timestamp ``dashboard.db`` holds for ``source`` (None: nothing yet)."""
    try:
        row = db.execute("SELECT MIN(first_ts) FROM coverage WHERE source = ?", (source,))
        value = row.fetchone()
    except sqlite3.Error:
        return None
    return value[0] if value and value[0] else None


# --- repo ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopPassRow:
    correlation_id: str
    started_at: str | None
    started_local: str | None
    completed_at: str | None
    completed_local: str | None
    ok: bool | None
    elapsed_seconds: float | None
    error_count: int
    merge_count: int
    review_count: int


@dataclass(frozen=True)
class MergeRow:
    ts: str
    ts_local: str | None
    issue: int | None
    pr: int | None
    evidence: str  # event kind that first reported the merge
    approx: bool  # merged outside the orchestrator: stamped when noticed, not when it happened


@dataclass(frozen=True)
class EscalationRow:
    ts: str
    ts_local: str | None
    issue: int | None
    pr: int | None
    event_kind: str
    reason: str | None
    detail: str | None


@dataclass(frozen=True)
class RepoDrill:
    slug: str
    freshness: RepoFreshness | None
    stages: tuple[FlowStage, ...]
    needs_me: tuple[NeedsMeItem, ...]
    workers: RepoWorkers | None
    reviewers: RepoWorkers | None
    runners: RunnerRepo | None
    # History is optional: when dashboard.db is unreadable the live part still renders.
    history_error: str | None
    history_from: str | None
    passes: tuple[LoopPassRow, ...]
    merges: tuple[MergeRow, ...]
    escalations: tuple[EscalationRow, ...]


# --- issue / PR ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineEntry:
    ts: str
    ts_local: str | None
    category: str  # lifecycle | review | escalation | worker
    label: str
    detail: str | None
    event_kind: str
    approx: bool
    issue: int | None
    pr: int | None


@dataclass(frozen=True)
class StageTime:
    stage: str
    seconds: float  # accumulated across every visit
    visits: int
    open_now: bool  # the item is in this stage now; the open visit is counted up to ``now``


@dataclass(frozen=True)
class CurrentState:
    """What the status snapshot says about the item now (GitHub-derived, as of the snapshot)."""

    stage: str | None
    title: str | None
    labels: tuple[str, ...]
    pr: int | None
    review_decision: str | None
    is_draft: bool | None
    as_of: datetime | None


@dataclass(frozen=True)
class IssueDrill:
    """An issue's or a PR's lifecycle; ``kind`` says which ``number`` is."""

    kind: str  # issue | pr
    repo: str
    number: int
    issue: int | None  # the issue the timeline is about (None: a PR with no known issue)
    prs: tuple[int, ...]
    current: CurrentState | None
    timeline: tuple[TimelineEntry, ...]
    stage_times: tuple[StageTime, ...]
    lead_seconds: float | None
    approx: bool  # stage/lead times are reconstructed (no lifecycle_transition for this item)
    history_from: str | None
    known: bool  # False: neither history nor the snapshot has ever seen this number


# --- loop pass ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PassEvent:
    id: int
    ts: str
    ts_local: str | None
    level: str
    kind: str
    issue: int | None
    pr: int | None
    preview: str  # compact, truncated, control characters removed; the renderer escapes it


@dataclass(frozen=True)
class PassDrill:
    repo: str
    correlation_id: str
    events: tuple[PassEvent, ...]
    truncated: bool  # more events matched than ``MAX_PASS_EVENTS``
    level_counts: tuple[tuple[str, int], ...]
    started_at: str | None
    completed_at: str | None
    ok: bool | None
    elapsed_seconds: float | None
