"""Single owner of every GitHub-label edge in the issue lifecycle.

Every add/remove pair lives here as a named transition; workflow code names
the event and never touches individual labels. This is the single point of
enforcement for label-state consistency — scattering add/remove calls across
the workflow was how stalled label states happened in production.
"""

from __future__ import annotations

from .config import LabelConfig
from .github import GitHub


def _edges(labels: LabelConfig) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    return {
        # manifest written, worker not yet independently confirmed
        "queued": ((labels.queued,), ()),
        # worker launch confirmed (subprocess ok / independent evidence)
        "dispatched": ((labels.in_progress,), (labels.queued,)),
        "review_started": ((labels.pr_open, labels.reviewing), ()),
        "rework_requested": ((labels.needs_rework,), (labels.reviewing,)),
        # rework cap exhausted or reviewer blocked — a human decision is needed
        "escalated": ((labels.human_needed,), (labels.reviewing,)),
        "blocked": ((labels.human_needed,), ()),
        "merged": ((labels.done,), tuple(sorted(labels.active))),
    }


def transition(gh: GitHub, labels: LabelConfig, issue_number: int, event: str) -> None:
    add, remove = _edges(labels)[event]
    for label in add:
        gh.add_issue_label(issue_number, label)
    for label in remove:
        gh.remove_issue_label(issue_number, label)
