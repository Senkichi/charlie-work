"""Permanent ``gh run rerun`` refusals for the infra-rerun driver (issue #2445).

GitHub refuses to re-run a workflow run created more than a month ago (HTTP 403
"... created over a month ago"). Unlike the "already running" refusal (#1936)
this never clears, yet the driver re-requested it on every pass with no backoff
and no escalation -- ~1 REST point per pass per PR against a 5,000/h budget.

A permanent refusal is recorded per run id under the PR's
``infra_rerun_refused`` state key (``{head_sha: [run_id, ...]}``), the run id is
never re-requested, and the PR is escalated once. Anything not matching a known
permanent phrase stays a transient error with the existing per-pass behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.state import load_state_locked

# Reason recorded on the escalation. The ``infra_rerun_escalated`` event kind
# is reused (its reason_class is already "mechanical"); this string
# distinguishes the refusal from a cap exhaustion in ``escalation_reasons_seen``.
REFUSED_ESCALATION_REASON = "infra_rerun_refused"

# Lower-cased substrings of ``gh run rerun`` errors GitHub will never lift.
_PERMANENT_REFUSAL_PHRASES: tuple[str, ...] = ("created over a month ago",)


def is_rerun_already_running(error: str) -> bool:
    """True if a ``gh run rerun`` error means the run is still in progress.

    GitHub refuses to rerun a workflow while any job in the run is still
    running. The CLI surfaces this with the phrase ``already running`` in the
    error, e.g. ``run <id> cannot be rerun; This workflow is already running``
    or ``The workflow run containing this job is already running``. Other
    refusal reasons (expired run, broken workflow file, missing permissions,
    ...) do not contain this phrase and should not be treated as retryable.

    This is a retry-later race, not a failure: it clears by itself once the run
    finishes. It is the single classifier shared by the infra-rerun driver, the
    closing-keyword autofix and the aggregate-job heal lane (issue #2666).
    """
    return "already running" in error.lower()


def is_permanent_rerun_refusal(error: str) -> bool:
    """True if a ``gh run rerun`` error is a refusal that retrying cannot fix."""
    lowered = error.lower()
    return any(phrase in lowered for phrase in _PERMANENT_REFUSAL_PHRASES)


def _pr_record(state_file: Path, pr_number: int) -> dict[str, Any]:
    record = load_state_locked(state_file).get("prs", {}).get(str(pr_number), {})
    return record if isinstance(record, dict) else {}


def load_refused_run_ids(state_file: Path, pr_number: int, head_key: str) -> set[int]:
    """Run ids already permanently refused on this PR head (empty if none)."""
    raw = _pr_record(state_file, pr_number).get("infra_rerun_refused") or {}
    ids: set[int] = set()
    if isinstance(raw, dict):
        for raw_id in raw.get(head_key) or []:
            try:
                ids.add(int(raw_id))
            except (TypeError, ValueError):
                continue
    return ids


def refusal_is_sole_remaining_work(
    state_file: Path,
    pr_number: int,
    *,
    dispatched: bool,
    deferred: bool,
    errors: list[str],
) -> bool:
    """Escalate-once gate: a refusal is all that is left and not yet escalated.

    Nothing dispatched or deferred this pass, every error is itself a permanent
    refusal (a transient error keeps today's retry-next-pass behaviour), and the
    PR's ``escalation_reasons_seen`` does not already hold the refusal reason.
    """
    if dispatched or deferred or not all(is_permanent_rerun_refusal(e) for e in errors):
        return False
    return REFUSED_ESCALATION_REASON not in (
        _pr_record(state_file, pr_number).get("escalation_reasons_seen") or []
    )


def refused_state_patch(head_key: str, refused: set[int]) -> dict[str, Any]:
    """PR-state fragment persisting the refused ids (empty when none)."""
    return {"infra_rerun_refused": {head_key: sorted(refused)}} if refused else {}
