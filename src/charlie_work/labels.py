"""Single owner of every GitHub-label edge in the issue lifecycle.

Every add/remove pair lives here as a named transition; workflow code names
the event and never touches individual labels. This is the single point of
enforcement for label-state consistency — scattering add/remove calls across
the workflow was how stalled label states happened in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import LabelConfig
from .github import GitHub


class TransitionOutcome(Enum):
    """Result of a label transition operation."""

    APPLIED = "applied"  # All adds and removes succeeded
    PARTIAL_FAILURE = "partial_failure"  # At least one add or remove failed
    NOTHING_CHANGED = "nothing_changed"  # No labels to add or remove


@dataclass(frozen=True)
class TransitionResult:
    """Detailed result of a label transition operation."""

    outcome: TransitionOutcome
    add_failures: list[tuple[int, str]]  # (issue_number, label) pairs that failed to add
    remove_failures: list[tuple[int, str]]  # (issue_number, label) pairs that failed to remove


def _edges(labels: LabelConfig) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    return {
        # manifest written, worker not yet independently confirmed
        "queued": ((labels.queued,), ()),
        # worker launch confirmed (subprocess ok / independent evidence)
        "dispatched": ((labels.in_progress,), (labels.queued,)),
        # Re-review is a fresh cycle: clear needs_rework so repeated loop()
        # passes don't permanently stack reviewing on top of needs_rework.
        "review_started": ((labels.pr_open, labels.reviewing), (labels.needs_rework,)),
        "rework_requested": ((labels.needs_rework,), (labels.reviewing,)),
        # rework worker launched for non-manual adapters
        "rework_dispatched": ((labels.in_progress,), (labels.needs_rework,)),
        # reviewer approved; waiting on merge. pr_open (kept) without reviewing
        # is the "approved, not yet merged" state — distinct from under-review.
        "review_approved": ((), (labels.reviewing, labels.needs_rework)),
        # rework cap exhausted or reviewer blocked — a human decision is needed
        "escalated": ((labels.human_needed,), (labels.reviewing,)),
        "blocked": ((labels.human_needed,), ()),
        "merged": ((labels.done,), tuple(sorted(labels.active))),
    }


def transition(gh: GitHub, labels: LabelConfig, issue_number: int, event: str) -> TransitionResult:
    add, remove = _edges(labels)[event]
    add_failures: list[tuple[int, str]] = []
    remove_failures: list[tuple[int, str]] = []

    for label in add:
        if not gh.add_issue_label(issue_number, label):
            add_failures.append((issue_number, label))

    for label in remove:
        if not gh.remove_issue_label(issue_number, label):
            remove_failures.append((issue_number, label))

    if not add and not remove:
        return TransitionResult(TransitionOutcome.NOTHING_CHANGED, [], [])
    if add_failures or remove_failures:
        return TransitionResult(TransitionOutcome.PARTIAL_FAILURE, add_failures, remove_failures)
    return TransitionResult(TransitionOutcome.APPLIED, [], [])
