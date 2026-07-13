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
    # Helper to compute removal set: all workflow labels except the ones being added
    # This ensures label transitions are exclusive (single-state by design)
    def _compute_remove(add_labels: tuple[str, ...]) -> tuple[str, ...]:
        if not add_labels:
            return ()
        to_remove = labels.workflow_labels - set(add_labels)
        return tuple(sorted(to_remove))

    return {
        # manifest written, worker not yet independently confirmed
        "queued": ((labels.queued,), _compute_remove((labels.queued,))),
        # worker launch confirmed (subprocess ok / independent evidence)
        "dispatched": ((labels.in_progress,), _compute_remove((labels.in_progress,))),
        # Re-review is a fresh cycle: clear needs_rework so repeated loop()
        # passes don't permanently stack reviewing on top of needs_rework.
        "review_started": (
            (labels.pr_open, labels.reviewing),
            _compute_remove((labels.pr_open, labels.reviewing)),
        ),
        "rework_requested": ((labels.needs_rework,), _compute_remove((labels.needs_rework,))),
        # rework worker launched for non-manual adapters
        "rework_dispatched": ((labels.in_progress,), _compute_remove((labels.in_progress,))),
        # reviewer approved; waiting for merge. pr_open (kept) without reviewing
        # is the "approved, not yet merged" state — distinct from under-review.
        "review_approved": ((labels.pr_open,), _compute_remove((labels.pr_open,))),
        # rework cap exhausted or reviewer blocked — a human decision is needed
        "escalated": ((labels.human_needed,), _compute_remove((labels.human_needed,))),
        "blocked": ((labels.human_needed,), _compute_remove((labels.human_needed,))),
        "merged": ((labels.done,), _compute_remove((labels.done,))),
        # redispatch cap exhausted — a human decision is needed
        "redispatch_escalated": ((labels.human_needed,), _compute_remove((labels.human_needed,))),
        # Issue #203: a merged PR only *mentions* the issue in free text, with
        # no hijack-safe branch/closing-keyword binding. That never authorizes
        # a close — flag it for a human decision instead, same label as any
        # other human-needed escalation.
        "merged_pr_mention_flagged": (
            (labels.human_needed,),
            _compute_remove((labels.human_needed,)),
        ),
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
