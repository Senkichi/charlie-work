"""Drill-down read models (spec section 4): repo, issue, PR and loop pass.

Each entry point returns a frozen result or a ``DrillError`` value, never raises on bad
input or a missing/corrupt source. History comes from ``dashboard.db`` only; the loop-pass
drill-down is the single exception and reads one registered repo's ``events.db`` read-only,
filtered by correlation id.
"""

from __future__ import annotations

from ..pages.now_fmt import valid_correlation_id
from .loop_pass import pass_drill
from .repo import repo_drill
from .timeline import issue_drill, pr_drill
from .types import (
    DrillError,
    EscalationRow,
    IssueDrill,
    LoopPassRow,
    MergeRow,
    PassDrill,
    PassEvent,
    RepoDrill,
    StageSpan,
    StageTime,
    TimelineEntry,
)

__all__ = [
    "DrillError",
    "EscalationRow",
    "IssueDrill",
    "LoopPassRow",
    "MergeRow",
    "PassDrill",
    "PassEvent",
    "RepoDrill",
    "StageSpan",
    "StageTime",
    "TimelineEntry",
    "issue_drill",
    "pass_drill",
    "pr_drill",
    "repo_drill",
    "valid_correlation_id",
]
