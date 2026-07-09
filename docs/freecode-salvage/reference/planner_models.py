"""Planner domain types — frozen value objects.

Single source of truth for planner row shapes (D-05). `freecode.state.store`
imports from THIS module; this module MUST NOT import anything from
`freecode.state.*` (the inverted-leaf direction is documented and enforced
by `tests/test_planner_capacity_choke_guard.py` for `capacity.py`; an
equivalent assertion lives in `tests/test_planner_models.py`).

Conventions:
- Every dataclass is `@dataclass(frozen=True)` (D-05 immutability).
- Timestamps are stored as `str` (ISO 8601 UTC) per D-07; never `datetime`.
- Compound list fields are typed as `tuple[str, ...]` per D-06 / RESEARCH
  Pattern 3 — they persist as JSON in `*_json TEXT NOT NULL DEFAULT '[]'`
  columns, but in-memory must be hashable/immutable.
- Pydantic types do NOT belong here. Pydantic models for `limits.yaml`
  validation land in `planner/limits.py` (Plan 06-05).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReservationStatus(str, Enum):
    HELD = "held"
    RUNNING = "running"
    CONSUMED = "consumed"
    RELEASED = "released"
    EXPIRED = "expired"
    FAILED_QUOTA = "failed_quota"


@dataclass(frozen=True)
class WorkSource:
    id: str
    kind: str  # e.g., "markdown", "github_issue_json"
    uri: str  # source URI / file path / issue URL
    ingested_at: str  # ISO 8601 UTC TEXT (D-07)
    metadata_json: str  # JSON-serialized dict (D-06)


@dataclass(frozen=True)
class WorkItem:
    id: str
    source_id: str
    title: str
    body_excerpt: str
    capability_tags: tuple[str, ...]  # noqa: E501 -- JSON list on disk; tuple in-memory (D-06 + RESEARCH Pattern 3)
    metadata_json: str
    created_at: str


@dataclass(frozen=True)
class WorkChunk:
    id: str
    work_item_id: str
    sequence: int
    summary: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    capability_tags: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    status: str  # "pending" | "scheduled" | "started" | "completed" | "blocked"
    blocked_reason: str | None
    blocked_until: str | None  # ISO 8601 UTC soft-block (Phase 9 m006)
    metadata_json: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CapacityModel:
    provider_id: str
    model_id: str
    dimension: str  # "requests" | "tokens"
    window_semantics: str  # "per_minute" | "per_hour" | "per_day"
    window_seconds: int
    limit_value: float  # PROSE-stated limit — NEVER rewritten by tighten_after_429 (D-13)
    safety_margin: float  # 0.0..0.5 (D-14 cap)
    confidence: str  # Confidence enum value
    source: str  # "seed" | "runtime"
    citation: str
    last_adjustment_reason: str | None
    updated_at: str


@dataclass(frozen=True)
class QuotaReservation:
    id: str
    provider_id: str
    model_id: str
    dimension: str
    window_semantics: str
    window_start: str  # ISO TEXT, calendar-aligned per window_for_timestamp
    window_end: str
    amount: float
    status: str  # ReservationStatus value
    work_chunk_id: str | None
    attempt_id: str | None
    created_at: str
    updated_at: str
    expires_at: str


@dataclass(frozen=True)
class ExecutionAttempt:
    id: str
    chunk_id: str
    planner_session_id: str
    provider_id: str
    model_id: str
    reservation_id: str | None
    status: str  # "started" | "succeeded" | "failed" | "failed_quota" | "released"
    exit_code: int | None
    quota_feedback_json: str | None  # populated by Phase 9 learning loop
    started_at: str
    finished_at: str | None


@dataclass(frozen=True)
class ScheduleDecision:
    chunk_id: str
    provider_id: str | None
    model_id: str | None
    reason: str  # noqa: E501 -- enumerated literal set kept on one line for grep: scheduled | no_capacity | cooldown | ack_missing | dependency_unresolved | checkpoint | context_limit
    capability_fit: float
    quota_waste: float
    decided_at: str
