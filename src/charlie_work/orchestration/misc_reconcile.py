"""Reconcile-loop delegates moved out of ``OrchestratorApp``.

Track 2 Phase B, L03 batch 3 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each top-level ``def``
unwrapped onto ``OrchestratorApp``.

Workflow-defined names are reached through ``_wf.``: the ``CommandResult`` class,
the ``_state_lock_busy_result`` helper, the Tier D ``state_lock`` primitive
(patched on ``charlie_work.workflow`` by the suite), and ``layout`` -- the
``from charlie_work import layout`` module import, bound via ``_wf.layout`` so the
bare read never resolves to the L09 ``layout`` PROPERTY on ``OrchestratorApp``
(LEGB resolves the free name to the module import, not the class property; the
``_wf`` qualification makes that explicit). The intra-module ``reconcile ->
self._reconcile_locked`` call and the ``self._record_event`` sibling call stay
``self.`` calls.
"""

from __future__ import annotations

import charlie_work.workflow as _wf
from dataclasses import asdict

from charlie_work.file_lock import try_acquire_byte_range_lock
from charlie_work.github import GraphQLBudgetError
from charlie_work.reconcile import (
    DriftItem,
    apply_fixes as apply_drift_fixes,
    detect_aviator_stale_blocked,
    detect_drift,
    detect_mergequeue_not_approved,
    detect_mergequeue_wedged,
)
from charlie_work.state import StateLockBusy


def reconcile(
    self,
    *,
    fix: bool = False,
    skip_dead_session_sweep: bool = False,
    dry_run: bool | None = None,
) -> _wf.CommandResult:
    """Detect (and optionally repair) drift between GitHub reality and the
    orchestrator's labels/state — e.g. a PR merged by hand outside
    merge-ready leaving `agent:in-progress` stale forever. Read-only unless
    ``fix`` is passed.

    ``mop-up --fix`` is a state writer and must be mutually exclusive with
    a supervised ``bash-rats``/fleet pass on the same repo. It acquires the
    same ``supervisor.lock`` used by fleet/supervise. If either the
    supervisor lock or the state lock is held, the call returns a skipped
    value result rather than blocking or writing unlocked.

    This is the public entry point: acquire ``supervisor.lock`` (only
    when ``fix``, matching today's behavior), then delegate everything
    else to ``_reconcile_locked``, which requires the lock already be
    held. Do not inline the body back here — the periodic in-loop caller
    (``_maybe_reconcile_drift``) already holds ``supervisor.lock`` via
    its own caller (``loop()``) and must call ``_reconcile_locked``
    directly; re-entering this method would always fail to reacquire the
    same non-reentrant lock and silently no-op (merge-lane-recovery D-8a).
    """
    if dry_run is None:
        dry_run = getattr(self, "dry_run", False)
    supervisor_lock = None
    if fix and not dry_run:
        supervisor_lock = try_acquire_byte_range_lock(
            _wf.layout.supervisor_lock_path(self.paths.root)
        )
        if supervisor_lock is None:
            return _wf.CommandResult(
                True,
                "reconcile deferred: supervisor lock held",
                {"pass_skipped": True, "reason": "supervisor_lock_held"},
            )
    try:
        return self._reconcile_locked(
            fix=fix,
            skip_dead_session_sweep=skip_dead_session_sweep,
            dry_run=dry_run,
        )
    finally:
        if supervisor_lock is not None:
            supervisor_lock.release()


def _reconcile_locked(
    self,
    *,
    fix: bool = False,
    skip_dead_session_sweep: bool = False,
    dry_run: bool = False,
) -> _wf.CommandResult:
    """Run drift detection (and optional repair) against GitHub/state.

    Precondition: the caller MUST already hold ``supervisor.lock`` — this
    method never acquires it itself. ``reconcile()`` is the only method
    that acquires the lock (and only when ``fix``); it delegates here
    immediately afterward. The periodic in-loop caller
    (``_maybe_reconcile_drift``) calls this directly, bypassing
    ``reconcile()``'s lock-acquisition entirely, because ``loop()``'s own
    caller already holds ``supervisor.lock`` for the whole pass —
    acquiring it a second time on the same non-reentrant byte-range lock
    would always fail and silently no-op (merge-lane-recovery D-8a).

    Extracted verbatim from ``reconcile()``'s former body — logic here is
    unchanged from before the split, including the GraphQL rate-limit
    deferral, so ``charlie mop-up --fix`` (which still goes through
    ``reconcile()``) is byte-for-byte unchanged in behaviour.
    """
    try:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            threshold = self.config.runtime.graphql_rate_limit_threshold
            sufficient, remaining, reset_at = self.gh.check_graphql_rate_limit(threshold)
            if not sufficient:
                if not dry_run:
                    state = _wf.append_event(
                        state,
                        "graphql_rate_limit_deferred",
                        {
                            "remaining": remaining,
                            "reset": reset_at,
                            "threshold": threshold,
                            "phase": "reconcile",
                        },
                        state_path=self.paths.state_file,
                    )
                    _wf.save_state(self.paths.state_file, state)
                return _wf.CommandResult(
                    True,
                    "reconcile deferred: GraphQL rate limit below threshold",
                    {
                        "deferred": True,
                        "deferred_reason": "graphql_rate_limit",
                        "graphql_remaining": remaining,
                        "graphql_reset": reset_at,
                        "graphql_threshold": threshold,
                    },
                )
            new_state = state
            try:
                drift = (
                    detect_drift(
                        self.gh,
                        state,
                        self.config,
                        repo_root=self.repo_root,
                        skip_dead_session_sweep=skip_dead_session_sweep,
                        state_path=self.paths.state_file,
                    )
                    + detect_aviator_stale_blocked(self.gh, self.config, repo_root=self.repo_root)
                    + detect_mergequeue_not_approved(
                        self.gh, self.config, repo_root=self.repo_root
                    )
                    + detect_mergequeue_wedged(
                        self.gh, self.config, state, repo_root=self.repo_root
                    )
                )
                fixed = False
                post_fix_drift: list[DriftItem] = []
                if fix and not dry_run and drift:
                    new_state = apply_drift_fixes(
                        self.gh,
                        state,
                        drift,
                        self.config,
                        repo_root=self.repo_root,
                        state_path=self.paths.state_file,
                    )
                    _wf.save_state(self.paths.state_file, new_state)
                    # Post-#134: transition() returns TransitionResult with PARTIAL_FAILURE
                    # for failed adds/removes, and apply_fixes records the outcome in the
                    # reconcile event. Re-detect against the new state to verify the repairs
                    # actually landed before reporting success.
                    post_fix_drift = (
                        detect_drift(
                            self.gh,
                            new_state,
                            self.config,
                            repo_root=self.repo_root,
                            skip_dead_session_sweep=skip_dead_session_sweep,
                            state_path=self.paths.state_file,
                        )
                        + detect_aviator_stale_blocked(
                            self.gh, self.config, repo_root=self.repo_root
                        )
                        + detect_mergequeue_not_approved(
                            self.gh, self.config, repo_root=self.repo_root
                        )
                        + detect_mergequeue_wedged(
                            self.gh, self.config, new_state, repo_root=self.repo_root
                        )
                    )
                    fixed = len(post_fix_drift) == 0
                elif fix and dry_run and drift:
                    post_fix_drift = drift
            except GraphQLBudgetError as exc:
                # Defensive: detect_drift re-checks the budget and may raise.
                if not dry_run:
                    new_state = self._record_event(
                        new_state,
                        "reconcile_pass_deferred",
                        {
                            "remaining": exc.remaining,
                            "reset": exc.reset_at,
                            "threshold": exc.threshold,
                            "fix": fix,
                            "deferred_reason": "graphql_rate_limit",
                        },
                    )
                    _wf.save_state(self.paths.state_file, new_state)
                return _wf.CommandResult(
                    True,
                    "reconcile deferred: GraphQL rate limit below threshold",
                    {
                        "deferred": True,
                        "deferred_reason": "graphql_rate_limit",
                        "graphql_remaining": exc.remaining,
                        "graphql_reset": exc.reset_at,
                        "graphql_threshold": exc.threshold,
                        "reconcile_pass_event_recorded": not dry_run,
                    },
                )
        message = f"found {len(drift)} drift item(s)"
        if fixed:
            message += " — fixed"
        elif drift:
            if dry_run and fix:
                message += " (dry-run; no changes applied)"
            elif fix and post_fix_drift:
                message += f" — partially fixed — {len(post_fix_drift)} item(s) remain"
            else:
                message += " (read-only; pass --fix to repair)"
        # ok=False when drift is present and not fixed: scripts and CI can gate
        # on exit code to detect unresolved drift, matching how `doctor` gates.
        ok = not drift or fixed or (dry_run and fix)
        return _wf.CommandResult(
            ok,
            message,
            {
                "drift": [asdict(item) for item in drift],
                "fixed": fixed,
                "drift_before": len(drift),
                "drift_after": len(post_fix_drift),
                "remaining_drift": [asdict(item) for item in post_fix_drift],
            },
        )
    except StateLockBusy:
        return _wf._state_lock_busy_result("reconcile deferred: state lock held")
