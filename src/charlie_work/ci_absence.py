"""Terminally-absent required-check classification (issue #1681).

Pure value object + predicate shared by ``_detect_ci_absence`` (the detector)
and ``review()`` (its consumer); kept out of ``orchestration/reap_dispatch``
because every top-level ``def`` there is installed onto ``OrchestratorApp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CiAbsence:
    """A required check that will never arrive for ``head_sha`` (issue #1681).

    ``kind`` is ``never_created`` (zero workflow run objects) or
    ``workflow_no_jobs`` (runs exist, all completed, none can report the
    missing checks -- e.g. GitHub rejected the workflow file).
    """

    kind: str
    head_sha: str


def runs_terminally_without_jobs(runs: list[dict[str, Any]]) -> bool:
    """True when every run is ``completed`` and none produced jobs.

    ``actions/runs`` run objects do not embed ``jobs``, so a run with an
    explicit ``jobs`` list is judged by it (empty == no jobs); otherwise the
    rejected-workflow signature is a ``completed`` run whose conclusion is a
    failure (GitHub's ``failure`` / ``startup_failure`` for an invalid file).
    Any run still queued/in_progress keeps the head pending.
    """
    if not runs:
        return False
    for run in runs:
        if not isinstance(run, dict) or run.get("status") != "completed":
            return False
        jobs = run.get("jobs")
        if isinstance(jobs, list):
            if jobs:
                return False
        elif run.get("conclusion") not in ("failure", "startup_failure"):
            return False
    return True
