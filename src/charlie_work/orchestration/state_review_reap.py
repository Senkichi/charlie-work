"""Standalone review-claim reap delegates for ``OrchestratorApp`` (issue #1874).

Extracted out of ``state_operator_commands.py`` in this same PR's rework round:
the new ``reap_reviews`` / ``_open_review_claims`` pair pushed that module past
its 1000-line ``file_size_ratchet_baseline`` mark. Per the established facade
re-export pattern (``workflow_delegation``, used across every submodule of this
package), the implementation lives here as plain top-level ``def``s; the
installer discovers this module the same way it discovers its siblings and
re-attaches each function onto ``OrchestratorApp`` as a class attribute, so
``app.reap_reviews(...)`` and ``self._open_review_claims(...)`` keep working
with no change at any call site (``cli.py``, the test suite).

Names reached through ``_wf.`` (module-object seam, design Section 3.1 rule 2,
#1627): ``charlie_work.workflow``'s ``CommandResult`` and the Tier-D names the
suite patches on it (``is_claim_stale``, ``is_pid_alive``, ``load_state_locked``)
-- the same convention ``state_operator_commands.py`` documents for its own
sibling delegates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from charlie_work import layout
from charlie_work.state import _REVIEW_STALE_CLAIM_TIMEOUT_MINUTES
import charlie_work.workflow as _wf


def _open_review_claims(self, now: datetime) -> list[dict[str, Any]]:
    """Read-only scan of open review-dispatch claims for ``reap_reviews``.

    Reports each PR whose ``review_dispatch_status`` is ``pending`` or
    ``dispatched`` with the liveness/staleness facts the stalled-review sweep
    applies, so ``--dry-run`` can show which claims a reap would act on
    without mutating state, checkouts, or sidecars.

    ``would_reap`` mirrors the sweep's state-level predicates in
    ``_detect_and_handle_stalled_reviews``: a ``pending`` claim reaps once
    ``review_dispatch_pending_at`` is past the stale-claim timeout; a
    ``dispatched`` claim reaps once its reviewer pid is dead AND
    ``review_dispatched_at`` is stale. The sweep's sidecar-driven branch can
    additionally reap on the sidecar's own ``started_at`` clock — this scan
    is the claim-level view, not a prediction of every branch outcome.

    ``is_pid_alive``/``is_claim_stale``/``load_state_locked`` are reached
    through ``_wf`` because the suite patches them on ``charlie_work.workflow``
    (the Tier-D convention documented in this module's sibling delegates).
    """
    state = _wf.load_state_locked(self.paths.state_file)
    claims: list[dict[str, Any]] = []
    for pr_key, entry in (state.get("prs") or {}).items():
        if not isinstance(entry, dict):
            continue
        status = entry.get("review_dispatch_status")
        pr_number: Any = int(pr_key) if str(pr_key).isdigit() else pr_key
        if status == "review_dispatch_pending":
            claim_at = entry.get("review_dispatch_pending_at")
            stale = _wf.is_claim_stale(
                claim_at,
                timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES,
                now=now,
            )
            claims.append(
                {
                    "pr": pr_number,
                    "status": "pending",
                    "claim_at": claim_at,
                    "stale": stale,
                    "would_reap": stale,
                }
            )
        elif status == "review_dispatch_dispatched":
            pid = entry.get("reviewer_pid")
            pid_alive = pid is not None and _wf.is_pid_alive(
                pid, entry.get("reviewer_process_start_time")
            )
            claim_at = entry.get("review_dispatched_at")
            stale = _wf.is_claim_stale(
                claim_at,
                timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES,
                now=now,
            )
            claims.append(
                {
                    "pr": pr_number,
                    "status": "dispatched",
                    "reviewer_pid": pid,
                    "pid_alive": pid_alive,
                    "claim_at": claim_at,
                    "stale": stale,
                    "would_reap": stale and not pid_alive,
                }
            )
    return claims


def reap_reviews(self, limit: int | None = None) -> _wf.CommandResult:
    """Force-reap dead review claims without waiting for a loop pass (issue #1874).

    The dead-reviewer reap normally runs inside ``dispatch_reviews`` at the
    top of each ``loop()`` pass. When no pass runs — a supervisor wedged
    mid-pass, or a sibling repo's lane overrunning the serial fleet pass — a
    dead reviewer's claim sits open past every stale threshold: the
    heartbeat flags it (``review-liveness`` anomaly, 45-minute reporting
    threshold) but nothing acts, because no code path outside a pass reached
    the sweep. This command is that path.

    Two modes, keyed on this repo's supervisor lock:

    - **Lock free** — acquire it (the ``bash-rats --once`` pattern) and run
      ``dispatch_reviews``, which performs the full reap block and then
      re-dispatches freed claims in the same invocation. The lock closes the
      governor read→launch double-dispatch window against a live supervisor.
    - **Lock held** — a live or wedged supervisor owns this repo's lane.
      Run only the reap block (``_run_review_reap_sweeps``): the sweeps
      never launch a reviewer, so the double-dispatch window the lock exists
      to close cannot arise, and reaping must not wait on the wedge the
      operator is investigating. Freed claims re-dispatch on the next pass
      that runs (or once the wedge is killed).

    ``--dry-run`` never reaches the sweeps: ``reap_sidecar`` /
    ``remove_review_checkout`` delete outside the write gate, so the preview
    is a read-only claim scan (``_open_review_claims``) reporting liveness
    and staleness against the sweep's own predicates.
    """
    resolved_now = datetime.now(UTC)

    if self.dry_run:
        claims = self._open_review_claims(resolved_now)
        reapable = [c["pr"] for c in claims if c["would_reap"]]
        return _wf.CommandResult(
            True,
            f"dry-run: {len(reapable)} of {len(claims)} open review claim(s) would be reaped",
            {
                "dry_run": True,
                "claims": claims,
                "reapable": reapable,
            },
        )

    # Lazy import, same as cli.py's bash-rats handler: supervise is a heavy
    # module and this is the only call site in the module that needs it.
    from charlie_work.supervise import try_acquire_supervisor_lock

    lock = try_acquire_supervisor_lock(layout.supervisor_lock_path(self.paths.root))
    if lock is not None:
        try:
            result = self.dispatch_reviews(limit, now=resolved_now)
        finally:
            lock.release()
        # ``kind=`` by keyword: the event-kind scanner reads positional arg 1
        # as the kind (the ``log_event(state_path, kind, ...)`` shape), so a
        # positional kind here would leave the payload dict unresolved.
        self.write_gate.log_event(
            kind="review_reap_invoked",
            payload={
                "mode": "reap_and_dispatch",
                "launched_count": result.data.get("launched_count"),
            },
        )
        data = dict(result.data)
        data["mode"] = "reap_and_dispatch"
        return _wf.CommandResult(result.ok, result.message, data)

    # Lock held: a supervisor owns this repo's lane — possibly the wedged
    # pass this command exists to route around. Reap anyway; the sweeps
    # never launch a reviewer, so the lock's double-dispatch window cannot
    # open, and every write inside the sweeps is state_lock-serialized and
    # merge-on-write safe against a concurrent pass (issue #594).
    sweep = self._run_review_reap_sweeps(resolved_now)
    verdict_result = sweep["verdict_result"]
    stalled = sweep["stalled"]
    # Same keyword-kind shape as the lock-free branch above — see that call
    # site for why ``kind`` must not be positional.
    self.write_gate.log_event(
        kind="review_reap_invoked",
        payload={
            "mode": "reap_only",
            "dispatch_skipped": "supervisor_lock_held",
            "stalled_count": len(stalled),
            "reaped_prs": [entry.get("pr") for entry in stalled],
        },
    )
    return _wf.CommandResult(
        True,
        f"reaped {len(stalled)} stalled review claim(s); dispatch skipped "
        "(supervisor lock held — freed claims re-dispatch on the next pass)",
        {
            "mode": "reap_only",
            "dispatch_skipped": "supervisor_lock_held",
            "stalled": stalled,
            "recorded_verdicts": verdict_result.get("recorded", []),
            "missed_verdicts": verdict_result.get("missed", []),
            "reconciled_verdicts": sweep["reconciled_verdicts"],
            "reaped_checkouts": sweep["reaped_checkouts"],
            "orphaned_checkouts": sweep["orphaned_checkouts"],
        },
    )
