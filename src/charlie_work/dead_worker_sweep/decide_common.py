"""Shared pure helpers for the decide modules: drafts, the flow driver, predicates.

Nothing here reads a clock, the filesystem, GitHub or ``state.json``. The
predicates are pure copies of helpers that live next to impure code in
``dead_worker_reap``/``dispatch_selection`` (those read ``datetime.now``); the
copies take ``now`` as an argument so a decision replays identically.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Generator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ..iso_timestamp import parse_iso_timestamp
from ..worker_fate import persisted_failure
from .model import COMMIT_TYPES, Emit, UpdateIssue

Flow = Generator[Any, Any, Any]

_MISSING = object()


def run_flow(
    gen: Flow,
    observed: Mapping[Any, Any],
    commit_types: tuple[type, ...] = COMMIT_TYPES,
) -> tuple[Any, tuple[Any, ...], Any]:
    """Drive one phase flow against ``observed``.

    Commits are recorded (``None`` is sent back); a request already in
    ``observed`` has its result sent back; the first unobserved request ends
    the round. ``commit_types`` names which yielded values are commits (the
    stalled lane has its own set). Returns ``(return_value, commits, pending_request)``;
    ``pending_request`` is ``None`` when the flow ran to completion.
    """
    commits: list[Any] = []
    try:
        item = next(gen)
        while True:
            if isinstance(item, commit_types):
                commits.append(item)
                item = gen.send(None)
            elif item in observed:
                item = gen.send(observed[item])
            else:
                return None, tuple(commits), item
    except StopIteration as stop:
        return stop.value, tuple(commits), None


class Draft:
    """A private, mutable copy of one issue entry that records what changed.

    The flows port the original in-place ``entry[...] = ...`` logic unchanged
    against ``draft.work``; ``take()`` turns the accumulated difference into one
    :class:`UpdateIssue` commit. The snapshot entry is never touched.
    """

    __slots__ = ("_base", "_dead", "issue", "work")

    def __init__(self, issue: int, entry: Mapping[str, Any]) -> None:
        self.issue = issue
        self.work: dict[str, Any] = copy.deepcopy(dict(entry))
        self._base: dict[str, Any] = copy.deepcopy(self.work)
        self._dead = False

    def invalidate(self) -> None:
        """Drop the draft: the live entry was replaced (``Escalate``), so later
        writes must target the new entry through their own ``UpdateIssue``."""
        self._dead = True

    def take(self) -> UpdateIssue | None:
        if self._dead:
            return None
        diff = {
            key: copy.deepcopy(value)
            for key, value in self.work.items()
            if self._base.get(key, _MISSING) != value
        }
        self._base = copy.deepcopy(self.work)
        return UpdateIssue(self.issue, diff) if diff else None


def flush(draft: Draft) -> Generator[Any, Any, None]:
    """Yield the draft's pending change (if any) as an ``UpdateIssue`` commit."""
    update = draft.take()
    if update is not None:
        yield update


def emit(
    kind: str, payload: Mapping[str, Any], level: str | None = None, *, audit_only: bool = False
) -> Emit:
    return Emit(kind=kind, payload=dict(payload), level=level, audit_only=audit_only)


def events_to_emits(events: list[tuple[str, dict[str, Any]]]) -> tuple[Emit, ...]:
    return tuple(emit(kind, payload) for kind, payload in events)


def drift_fingerprint(**parts: Any) -> str:
    """Stable fingerprint for an orphaned-worker drift finding."""
    return json.dumps(parts, sort_keys=True, default=str)


def label_names(item: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for label in item.get("labels") or []:
        if isinstance(label, dict) and label.get("name"):
            names.add(str(label["name"]))
        elif isinstance(label, str):
            names.add(label)
    return names


def orphan_head_fingerprint(remote_sha: str | None, local_sha: str | None) -> str:
    return f"{remote_sha or 'none'}:{local_sha or 'none'}"


def session_failed_relabeled_payload(
    *,
    issue_number: int,
    reason: str,
    failure_kind: str | None = None,
    removed_labels: Any = (),
    added_ready: bool = False,
    label_write_ok: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "issue_number": issue_number,
        "reason": reason,
        "removed_labels": sorted(removed_labels),
        "added_ready": added_ready,
        "label_write_ok": label_write_ok,
    }
    if failure_kind is not None:
        payload["failure_kind"] = failure_kind
    if extra:
        payload.update(extra)
    return payload


def reap_due(
    entry: Mapping[str, Any],
    *,
    throttled_until: Any,
    reap_minutes: float,
    now: datetime,
) -> bool:
    """Pure copy of ``dead_dispatched_timer.dead_dispatched_reap_due``.

    ``throttled_until`` replaces ``state["throttled_until"]`` so the caller can
    track credits made earlier in the same sweep.
    """
    orphan_drift_at = entry.get("orphan_drift_at")
    drift_dt = parse_iso_timestamp(orphan_drift_at) if orphan_drift_at else None
    if drift_dt is None or reap_minutes <= 0:
        return False
    grace_anchor = drift_dt
    if persisted_failure(entry).is_throttle:
        throttled_dt = parse_iso_timestamp(throttled_until)
        if throttled_dt is not None:
            if throttled_dt > now:
                return False
            grace_anchor = max(drift_dt, throttled_dt)
    return (now - grace_anchor).total_seconds() / 60 >= reap_minutes


def is_pr_updated_at_older_than(pr: Mapping[str, Any], now: datetime, minutes: int) -> bool:
    updated_at = pr.get("updatedAt")
    if not updated_at:
        return False
    try:
        updated = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return (now - updated).total_seconds() > minutes * 60


def is_pre_review_rework_candidate(
    pr: Mapping[str, Any], stale_minutes: int, now: datetime
) -> tuple[bool, str]:
    """Pure copy of ``dead_worker_reap._is_pre_review_rework_candidate``."""
    if str(pr.get("mergeable") or "").upper() == "CONFLICTING":
        return True, "merge_conflict"
    if str(pr.get("mergeStateStatus") or "").upper() == "DIRTY":
        return True, "rework_branch_conflict"
    if stale_minutes <= 0:
        return False, ""
    if pr.get("statusCheckRollup"):
        return False, ""
    if is_pr_updated_at_older_than(pr, now, stale_minutes):
        return True, "stale_empty_checks"
    return False, ""


def windowed_orphan_redispatch_at(
    entry: Mapping[str, Any], *, window_minutes: int, now: datetime
) -> list[str]:
    """Pure copy of ``dispatch_selection._windowed_orphan_redispatch_at`` (``now`` passed in)."""
    raw = entry.get("orphan_redispatch_at")
    if not isinstance(raw, list):
        return []
    window_start = now - timedelta(minutes=window_minutes)
    result: list[str] = []
    for stamp in raw:
        if not isinstance(stamp, str):
            continue
        try:
            if datetime.fromisoformat(stamp.replace("Z", "+00:00")) >= window_start:
                result.append(stamp)
        except (ValueError, AttributeError, TypeError):
            continue
    return result


@dataclass
class LockAcc:
    """Per-replay scratch for the lock flow (rebuilt from nothing on every ``decide``)."""

    throttled_until: Any
    outcome_apply_routes: list[tuple[int, int]] = field(default_factory=list)
    review_routes: list[Any] = field(default_factory=list)
    no_op_routes: list[Any] = field(default_factory=list)
    reap_escalations: list[int] = field(default_factory=list)
