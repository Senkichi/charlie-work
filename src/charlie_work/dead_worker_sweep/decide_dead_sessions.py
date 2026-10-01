"""The dead-session lane's pure decisions.

The lane (``dead_sessions``) walks every session sidecar, reaps the dead ones and
reclaims their issues. Everything it *decides* lives here as small total
functions over already-read facts; the shell reads GitHub, state and the
worktree, calls these, and applies the answer in the original effect order.
Nothing here reads a clock, the filesystem, GitHub or state: ``now`` is the
pass's own sample, handed in.

Decisions:

* ``escalation_class``: is a failure kind a deterministic one (escalate on the
  first occurrence, never redispatch), and which ``reason_class`` does it carry
  (issue #783 mechanical vs issue #807 judgment).
* ``launch_failure_redispatch_at`` / ``launch_failure_escalates``: the
  launch-failed branch's bookkeeping (issue #266).
* ``dead_fallback_kind``: the fallback failure kind when the sidecar carries no
  classification (issue #261).
* ``wants_unsafe_salvage``: issue #1130's "salvage before escalating a
  ``worktree_unsafe`` launch failure" gate.
* ``redispatch_verdict``: the no-open-PR relabel branch's redispatch cap
  (issues #165, #261, #807, #1684).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..config import (
    DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS,
)
from ..throttle_signatures import is_provider_throttle_failure
from ..worktree import WORKTREE_UNSAFE_KINDS

REDISPATCH_CAP_REASON = "redispatch_cap_exceeded"


@dataclass(frozen=True)
class EscalationClass:
    """Whether a failure kind escalates on first sight, and as which reason class."""

    immediate: bool
    reason_class: str  # "judgment" | "mechanical"


def escalation_class(failure_kind: str | None) -> EscalationClass:
    """Classify ``failure_kind`` against the two deterministic-escalation sets.

    Issue #783: a deterministic launch/process failure is mechanical. Issue
    #807: a deterministic judgment failure (genuine local commits) is judgment,
    and wins when a kind is in both sets.
    """
    terminal = failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
    judgment = failure_kind in DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS
    return EscalationClass(
        immediate=terminal or judgment,
        reason_class="judgment" if judgment else "mechanical",
    )


def stamp(now: datetime) -> str:
    """The ``redispatch_at`` timestamp spelling (UTC ``Z`` suffix)."""
    return now.isoformat().replace("+00:00", "Z")


def launch_failure_escalates(failure_kind: str | None, *, has_open_pr: bool) -> bool:
    """A launch failure escalates when its kind is deterministic and no PR is open."""
    return escalation_class(failure_kind).immediate and not has_open_pr


def launch_failure_redispatch_at(windowed: Sequence[str], now: datetime) -> tuple[str, ...]:
    """The launch-failure escalation always counts itself as a redispatch event."""
    return (*windowed, stamp(now))


# Issue #2096: a headless `claude -p` worker that backgrounds its suite ends its
# turn within minutes, which ends the session and kills the background job.
BACKGROUND_EXIT_MAX_SECONDS: float = 600.0
BACKGROUND_EXIT_FAILURE_KIND = "worker_exited_with_background_work"


def exited_with_background_work(
    terminal: Mapping[str, Any] | None,
    *,
    adapter_kind: str,
    dirty: bool,
    ahead_count: int,
) -> bool:
    """A claude-code worker that exited 0 quickly, leaving a dirty tree and no commit."""
    if adapter_kind != "claude-code" or terminal is None:
        return False
    duration = terminal.get("duration_seconds")
    return bool(
        terminal.get("exit_code") == 0
        and isinstance(duration, int | float)
        and duration < BACKGROUND_EXIT_MAX_SECONDS
        and dirty
        and ahead_count == 0
    )


def dead_fallback_kind(
    *, is_completed: bool, worktree_unknown: bool, background_exit: bool = False
) -> str | None:
    """Fallback failure kind for a dead session whose log classifies as nothing.

    A completed-but-unpublished worktree is ``unpublished_work``; a fast clean
    exit that left uncommitted work behind is ``worker_exited_with_background_work``
    (issue #2096); otherwise a readable worktree is ``stalled`` and an
    unreadable one has no fallback.
    """
    if is_completed:
        return "unpublished_work"
    if background_exit and not worktree_unknown:
        return BACKGROUND_EXIT_FAILURE_KIND
    return None if worktree_unknown else "stalled"


def wants_unsafe_salvage(
    failure_kind: str | None,
    *,
    has_repo_root: bool,
    branch: str,
    active_labels: Collection[str],
) -> bool:
    """Issue #1130: try salvage before escalating a ``worktree_unsafe`` launch failure."""
    return bool(
        failure_kind in WORKTREE_UNSAFE_KINDS and has_repo_root and branch and active_labels
    )


@dataclass(frozen=True)
class RedispatchVerdict:
    """The no-open-PR relabel branch's outcome for one dead session."""

    redispatch_at: tuple[str, ...]
    escalate: bool
    reason: str | None  # set when ``escalate``
    reason_class: str


def redispatch_verdict(
    windowed: Sequence[str],
    failure_kind: str | None,
    *,
    now: datetime,
    max_auto_redispatch: int,
) -> RedispatchVerdict:
    """Count this relabel as a redispatch event and decide whether the cap is blown.

    Issue #1684: a provider-throttle death is a global condition, not a
    worker-quality signal, so it does not consume the cap. Issue #261/#807: a
    deterministic failure bypasses the cap and escalates on first occurrence.
    """
    redispatch_at = tuple(windowed)
    if not is_provider_throttle_failure(failure_kind):
        redispatch_at = (*redispatch_at, stamp(now))
    cls = escalation_class(failure_kind)
    escalate = cls.immediate or len(redispatch_at) > max_auto_redispatch
    reason = None
    if escalate:
        reason = (
            failure_kind if cls.immediate and failure_kind is not None else REDISPATCH_CAP_REASON
        )
    return RedispatchVerdict(
        redispatch_at=redispatch_at,
        escalate=escalate,
        reason=reason,
        reason_class=cls.reason_class,
    )
