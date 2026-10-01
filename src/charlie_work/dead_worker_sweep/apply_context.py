"""Mutable scratch the apply shell threads through one sweep.

Requests are frozen and hashable, so anything too big to hash (issue dicts, PR
dicts, the working state copy, fates) lives here, keyed by issue number. This is
shell-only bookkeeping: ``decide`` never sees it (it gets ``SweepFacts`` and the
``observed`` results). The dataclass is deliberately mutable, like ``LockAcc``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import OrchestratorConfig
from ..write_gate import WriteGate
from .ports import SweepPorts

StaleBuckets = dict[str, dict[int, list[Any]]]


@dataclass
class SweepContext:
    sessions_dir: Path
    state_file: Path
    config: OrchestratorConfig
    gh: Any
    write_gate: WriteGate
    ports: SweepPorts
    review_callback: Callable[[int], Any] | None
    record_review_callback: Callable[..., Any] | None
    enrich_checks_callback: Callable[..., list[dict[str, Any]]] | None
    fleet_dir_override: str | None
    repo_root: Any
    worktrees_dir: Path | None
    state: dict[str, Any]
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    stamp: str = ""
    pid_alive: dict[int, bool] = field(default_factory=dict)
    issues: dict[int, dict[str, Any]] | None = None
    pr_by_issue: dict[int, dict[str, Any]] = field(default_factory=dict)
    pr_views: dict[int, Any] = field(default_factory=dict)
    fates: dict[int, Any] = field(default_factory=dict)
    worker_outcomes: dict[int, dict[str, Any] | None] = field(default_factory=dict)
    branches: dict[int, str] = field(default_factory=dict)
    reclaim_results: dict[int, dict[str, Any]] = field(default_factory=dict)
    park_verdicts: dict[int, Any] = field(default_factory=dict)
    local_park_deferred: dict[int, str] = field(default_factory=dict)
    live_candidates: dict[int, dict[str, Any]] = field(default_factory=dict)
    escalations: dict[str, dict[int, dict[str, Any]]] = field(
        default_factory=lambda: {
            "worker_declared_blocked": {},
            "zero_artifact": {},
            "cross_repo_scope": {},
            "verified_no_changes": {},
        }
    )
    review_prs: dict[int, int] = field(default_factory=dict)
    buckets: StaleBuckets = field(
        default_factory=lambda: {"live_handoff": {}, "no_pr": {}, "swept": {}}
    )
    scope_context: tuple[frozenset[str], str] | None = None

    def issue_entry(self, number: int) -> dict[str, Any]:
        entry = (self.state.get("issues") or {}).get(str(number), {})
        return entry if isinstance(entry, dict) else {}

    def collector(self, bucket: str) -> Callable[[Any], None]:
        from ..worker_fate import collect_fate

        target = self.buckets[bucket]
        return lambda fate: collect_fate(target, fate)
