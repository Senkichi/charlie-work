"""Issue #1243 per-issue redispatch cap for the no-open-PR orphan lane (pure).

Counts dead *dispatches* (not sweep passes) in a windowed timestamp list on the
entry and compares the branch head fingerprint across attempts: a moving head is
progress (the salvage path's job), an unchanged one counts toward the cap.
Provider-throttle deaths never count (#1917).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..worker_fate import persisted_failure
from .decide_common import orphan_head_fingerprint, windowed_orphan_redispatch_at


@dataclass(frozen=True)
class RedispatchVerdict:
    exceeded: bool  # cap exceeded with no progress: escalate
    redispatch_count: int
    orphan_redispatch_at: tuple[str, ...]
    current_head: str
    dispatch_identity: str


def decide_redispatch_cap(
    entry: Mapping[str, Any],
    *,
    head_details: Mapping[str, Any],
    window_minutes: int,
    max_auto_redispatch: int,
    stamp: str,
    now: datetime,
) -> RedispatchVerdict:
    current_head = orphan_head_fingerprint(
        head_details.get("remote_head_sha"), head_details.get("local_head_sha")
    )
    prior_head = entry.get("orphan_redispatch_head_sha")
    head_changed = prior_head is not None and current_head != prior_head
    first_observation = prior_head is None
    dispatch_identity = (
        f"{entry.get('dispatched_at') or 'none'}:{entry.get('worker_pid') or 'none'}"
    )
    prior_dispatch = entry.get("orphan_redispatch_counted_dispatch")
    history = windowed_orphan_redispatch_at(entry, window_minutes=window_minutes, now=now)
    throttled_death = persisted_failure(entry).is_throttle
    if head_changed or first_observation:
        history = [] if throttled_death else [stamp]
    elif dispatch_identity != prior_dispatch and not throttled_death:
        history = history + [stamp]
    count = len(history)
    return RedispatchVerdict(
        exceeded=count > max_auto_redispatch and not head_changed,
        redispatch_count=count,
        orphan_redispatch_at=tuple(history),
        current_head=current_head,
        dispatch_identity=dispatch_identity,
    )
