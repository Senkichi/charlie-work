"""Whether the same-head packet skip may keep serving the packet's PR body (#1983, #2826, #3182).

A free function over the app (duck-typed, like ``mergequeue_stall.alarm_best_effort``),
not an ``OrchestratorApp`` member: the member surface is pinned
(``test_orchestratorapp_member_surface_conserved_lexical_plus_installed``), and
``reap_loop.py`` sits exactly at the file-size cap.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .instrumentation import log_event
from .iso_timestamp import parse_iso_timestamp
from .no_op_rework_body import _request_changes_body_drifted
from .rework_outcome import applied_body_outdates_packet
from .unescalate_reset_fields import UNESCALATE_VERDICT_VOIDED_AT_KEY


def _unescalate_stale_packet_body(app: Any, state: Any, pr_number: int, pr: dict) -> bool:
    """True when a verdict-voiding unescalation postdates this packet and the body drifted (#3182).

    The trigger is the recorded unescalation, never "the body changed":
    ``unescalate()`` stamps ``unescalate_verdict_voided_at`` on the PR record at the
    moment it voids a still-valid verdict, and the packet builders stamp
    ``packet_built_at`` into ``pr.json``. A packet built before that stamp cannot
    carry any operator decision written into the PR body since -- the resolution
    the voided verdict was waiting on -- so the same-head skip must not serve it
    when the live body hash differs from the packet body's. A packet that
    postdates the void was already rebuilt (or rendered) after it, so drift
    against it is an unrelated edit and #2826's rule governs: it rebuilds
    nothing.
    """
    pr_record = state.get("prs", {}).get(str(pr_number)) if isinstance(state, Mapping) else None
    if not isinstance(pr_record, Mapping):
        return False
    voided_at_raw = pr_record.get(UNESCALATE_VERDICT_VOIDED_AT_KEY)
    voided_at = parse_iso_timestamp(voided_at_raw)
    if voided_at is None:
        return False
    packet_built_at = app._read_packet_built_at(pr_number)
    # Both stamps are second-resolution (``format_utc``), so equality cannot
    # prove the packet postdates the void -- only a provably-newer packet is
    # exempt. A same-second pair fails safe toward one rebuild; the rebuilt
    # packet's body matches the live body, so it converges after one pass.
    if packet_built_at is not None and packet_built_at > voided_at:
        return False
    if app._packet_body_current(pr_number, pr):
        return False
    log_event(
        app.paths.state_file,
        "unescalate_packet_body_stale",
        {
            "pr_number": pr_number,
            "issue_number": pr_record.get("issue_number"),
            UNESCALATE_VERDICT_VOIDED_AT_KEY: voided_at_raw,
            "packet_built_at": packet_built_at.isoformat()
            if packet_built_at is not None
            else None,
        },
        repo=app.repo_root.name,
    )
    return True


def packet_body_current_for_skip(
    app: Any, state: Any, pr_number: int, pr: dict, resolved_decision: Any
) -> bool:
    """True if the same-head packet skip may keep serving the packet's PR body.

    Three independent reasons the packet's body is not the one a reviewer must see:

    - #1983: a ``request_changes`` verdict whose body drifted re-reviews at the same
      head, so the packet must carry the live body.
    - #3182: an operator ``charlie unescalate`` voided the verdict AFTER the packet
      was built and the live body drifted from the packet body -- the operator's
      decision written into the PR body must reach the fresh review, not the stale
      packet's copy.
    - #2826: a PR with no verdict at its head (missing, stale or pending) whose body a
      rework outcome apply rewrote AFTER the packet was built. Keyed on the apply's
      own record (``applied_body_outdates_packet``), never on "the body changed", so a
      body nobody applied through the orchestrator rebuilds nothing.
      ``watchdog.apply_rework_outcome_on_every_route`` off restores the #1983-only rule.
    """
    if _request_changes_body_drifted(app._review_decision(pr_number), pr):
        if not app._packet_body_current(pr_number, pr):
            return False
    if _unescalate_stale_packet_body(app, state, pr_number, pr):
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
