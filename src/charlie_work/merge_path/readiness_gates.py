"""Stage-3 gate predicates: cross-PR revert verdict and the human-merge fold-in.

Pure. Split out of ``decide.py`` (800-line cap). ``decide_readiness`` and the
live shell share these so the preview and the live path cannot drift.
"""

from __future__ import annotations

from dataclasses import replace

from .model import MergePathConfig, Readiness, RevertStatus, RevertVerdict
from .rules import REWORK_ALREADY_ROUTED_STATUSES


def decide_revert(
    *,
    approved: bool,
    sync_failed: bool,
    revert: RevertStatus,
    issue_number: int | None,
    issue_status: str | None,
) -> RevertVerdict:
    """The cross-PR revert gate: block, and whether to route the bound issue to rework.

    Needs no check data, so live applies the route before it reads ``pr_checks``
    (legacy order). DETECTED and UNDETERMINED both fail the sync gate closed;
    only DETECTED on a bound issue not already routed requests rework.
    """
    detected = False
    undetermined = False
    route = False
    if approved and not sync_failed:
        if revert is RevertStatus.DETECTED:
            sync_failed = True
            detected = True
            route = issue_number is not None and issue_status not in REWORK_ALREADY_ROUTED_STATUSES
        elif revert is RevertStatus.UNDETERMINED:
            # Fail closed, never route.
            sync_failed = True
            undetermined = True
    return RevertVerdict(sync_failed, detected, undetermined, route)


def deescalates(
    cfg: MergePathConfig,
    issue_number: int | None,
    human_merge_hold: bool,
    human_merge_check_unavailable: bool,
    issue_status: str | None,
    issue_reason_class: str | None,
) -> bool:
    return bool(
        deescalation_read_needed(
            cfg, issue_number, human_merge_hold, human_merge_check_unavailable
        )
        and issue_status == "escalated"
        and issue_reason_class == "policy"
    )


def with_human_merge(
    readiness: Readiness,
    cfg: MergePathConfig,
    human_merge: tuple[bool, bool],
    issue_status: str | None,
    issue_reason_class: str | None,
) -> Readiness:
    """Fold the human-merge read into a readiness decided before it was taken.

    Live reads the human-merge labels only after the stall and infra exits
    (legacy order), so it decides those first and folds the read in here.
    """
    hold, unavailable = human_merge
    return replace(
        readiness,
        human_merge_hold=hold,
        human_merge_check_unavailable=unavailable,
        deescalate=deescalates(
            cfg, readiness.issue_number, hold, unavailable, issue_status, issue_reason_class
        ),
    )


def deescalation_read_needed(
    cfg: MergePathConfig,
    issue_number: int | None,
    human_merge_hold: bool,
    human_merge_check_unavailable: bool,
) -> bool:
    """The issue's escalation entry is consumed: a policy escalation could be lifted.

    Only a bound issue under configured human-merge labels whose label has
    verifiably gone can be de-escalated (issue #1598); ``decide_readiness`` and
    the live gather share this predicate.
    """
    return bool(
        cfg.human_merge_labels
        and issue_number is not None
        and not human_merge_hold
        and not human_merge_check_unavailable
    )
