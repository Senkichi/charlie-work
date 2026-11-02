"""Whether the same-head packet skip may keep serving the packet's PR body (#1983, #2826).

A free function over the app (duck-typed, like ``mergequeue_stall.alarm_best_effort``),
not an ``OrchestratorApp`` member: the member surface is pinned
(``test_orchestratorapp_member_surface_conserved_lexical_plus_installed``), and
``reap_loop.py`` sits exactly at the file-size cap.
"""

from __future__ import annotations

from typing import Any

from .no_op_rework_body import _request_changes_body_drifted
from .rework_outcome import applied_body_outdates_packet


def packet_body_current_for_skip(
    app: Any, state: Any, pr_number: int, pr: dict, resolved_decision: Any
) -> bool:
    """True if the same-head packet skip may keep serving the packet's PR body.

    Two independent reasons the packet's body is not the one a reviewer must see:

    - #1983: a ``request_changes`` verdict whose body drifted re-reviews at the same
      head, so the packet must carry the live body.
    - #2826: a PR with no verdict at its head (missing, stale or pending) whose body a
      rework outcome apply rewrote AFTER the packet was built. Keyed on the apply's
      own record (``applied_body_outdates_packet``), never on "the body changed", so a
      body nobody applied through the orchestrator rebuilds nothing.
      ``watchdog.apply_rework_outcome_on_every_route`` off restores the #1983-only rule.
    """
    if _request_changes_body_drifted(app._review_decision(pr_number), pr):
        if not app._packet_body_current(pr_number, pr):
            return False
    if not app.config.watchdog.apply_rework_outcome_on_every_route or "body" not in pr:
        return True
    if not (
        resolved_decision.missing
        or resolved_decision.stale
        or resolved_decision.decision == "pending"
    ):
        return True
    return not applied_body_outdates_packet(
        state, pr_number, pr.get("body"), app._read_packet_body(pr_number)
    )
