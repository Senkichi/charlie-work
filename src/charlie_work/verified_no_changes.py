"""The ``verified_no_changes`` worker outcome (issue #2185).

A verification-only ticket ("Files: Modify: (none expected)") has a correct
zero-diff completion: the worker runs the checks, everything passes, and it
rightly makes no commit. ``.worker-outcome.json`` offered only ``blocked`` as a
non-PR terminal outcome, so that completion was indistinguishable from a worker
that died before committing and flowed into the orphan sweep's no-PR lane
(``session_failed_relabeled`` -> redispatch -> another worker session
re-verifying the same thing).

This module owns the outcome's wire literal and the pure helpers around it. The
worker brief (``prompts/worker_sections/verified_no_changes_outcome.md``) and the
sweep consumer both derive from :data:`VERIFIED_NO_CHANGES_OUTCOME`;
``tests/test_verified_no_changes_outcome.py`` pins the prompt text to it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: ``.worker-outcome.json`` ``outcome`` value: task complete, verified, no change needed.
VERIFIED_NO_CHANGES_OUTCOME = "verified_no_changes"

#: Why a ``verified_no_changes`` claim was refused (the sweep falls through to
#: today's handling). Carried on the ``worker_verified_no_changes_ignored`` event.
REFUSED_ESCALATED = "escalated"
REFUSED_DRY_RUN = "dry_run"
REFUSED_NO_WORKTREE = "worktree_unavailable"
REFUSED_CLOSE_FAILED = "close_failed"


def is_verified_no_changes(raw: Mapping[str, Any] | None) -> bool:
    """True when a raw ``.worker-outcome.json`` dict declares the success outcome."""
    return isinstance(raw, Mapping) and raw.get("outcome") == VERIFIED_NO_CHANGES_OUTCOME


def verified_detail(raw: Mapping[str, Any] | None) -> str:
    """The worker's evidence string (commands run + results), ``""`` when absent."""
    if not isinstance(raw, Mapping):
        return ""
    return str(raw.get("detail") or "").strip()


def resolution_comment(detail: str) -> str:
    """The issue comment (the local backend's ``resolution``) recording why it closed."""
    evidence = detail or "(the worker supplied no detail)"
    return (
        "Closed by the orchestrator: the worker declared `verified_no_changes` -- the "
        "task is complete and verified, and its worktree holds no commits ahead of the "
        "base and no source changes.\n\n"
        f"Worker evidence:\n\n{evidence}\n"
    )
