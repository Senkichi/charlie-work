"""Dead-worker-with-PR classification for the orphaned-worker sweep.

Extracted from ``workflow._detect_and_handle_orphaned_workers`` during the
issue #1911 rework: ``workflow.py`` sits over its file-size ratchet mark,
and the completed-outcome recovery this module contains was the growth the
reviewer asked to move. The whole "dead dispatched worker that still has an
open PR" classification moved here -- verbatim except that loop-level
``continue`` statements are early ``return``s and the sweep's closed-over
locals are explicit parameters -- so the sweep's ``for`` loop shell and the
no-open-PR salvage branch stay in ``workflow.py``. Issue #1917 later moved
the #654 timed ``dead_dispatched_reap_minutes`` escalation backstop here
too (``maybe_reap_dead_dispatched_worker``) for the same ratchet reason.

Called once per dead-PID ``dispatched`` issue that still has a linked open
PR, inside the sweep's ``state_lock`` (the same lock window the code ran
under before the move). Every finding either mutates ``entry`` and appends
to ``sweep_events`` or collects a post-lock route -- ``review_routes`` for a
head-advanced verdict re-review (and, since issue #1915, for a completed
worker outcome once its apply pass has landed), ``outcome_apply_routes`` for
a fresh, on-target ``.worker-outcome.json`` whose PR edits the orchestrator
applies through the #1877 seam in
``rework_outcome.apply_collected_rework_outcomes``.

Workflow-module names the moved code resolved through ``workflow``'s module
namespace (``utc_now``, ``_parse_iso_timestamp``) are reached through a
function-local ``import charlie_work.workflow as _wf`` -- a top-level import
would cycle (``workflow`` imports this module), and the module-object seam
keeps suite patches on ``charlie_work.workflow.<name>`` interceptable. Every
other free name is imported directly from its defining module.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .dispatch_selection import _credit_worker_death
from .orphaned_worker_review_drain import OrphanedWorkerReviewRoute
from .process_utils import find_worker_terminal_status
from .review_decision import review_decision
from .rework_outcome import (
    APPLIED_HEADS_KEY,
    fresh_completed_worker_outcome,
)
from .state import PASSIVE_OPEN_STATUS
from .throttle_signatures import is_provider_throttle_failure
from .worktree import worktree_path_for_branch

if TYPE_CHECKING:
    from .config import OrchestratorConfig
    from .github import GitHubLike


def maybe_reap_dead_dispatched_worker(
    *,
    state: dict[str, Any],
    entry: dict[str, Any],
    issue_number: int,
    sessions_dir: Path,
    pr_data: dict[str, Any] | None,
    dead_dispatched_reap_minutes: float,
    now: datetime,
    sweep_events: list[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, Any], bool]:
    """Timed dead-dispatched backstop (issue #654), run per dead-PID entry.

    Issue #654: time-based escape for a dead dispatched worker whose drift
    was already surfaced on a prior pass (``orphan_drift_at`` is set) but
    whose PR state did not qualify for auto-reset -- a clean exit with no
    push (issue #773's ``dead_worker_clean_exit_no_op`` branch), a
    non-request_changes decision, a head change without a review callback,
    or a PR-create failure on a pushed branch. In all of these the specific
    sub-branches in the caller emit drift once, set ``orphan_drift_at``,
    then on every subsequent pass the fingerprint match short-circuits --
    so the dispatch label (``agent:in-progress``) holds indefinitely. The
    label is the one-writer-per-branch mutex, so no re-dispatch can proceed
    on that branch until a worker that no longer exists reports back.
    After ``dead_dispatched_reap_minutes`` since the drift was first
    surfaced, escalate to ``agent:human-needed`` so a human can inspect the
    worktree for unpushed commits and decide whether to salvage or
    re-dispatch. This runs BEFORE the specific sub-branches so it is a pure
    backstop: on the first pass ``orphan_drift_at`` is not yet set and the
    specific sub-branch runs normally (either resetting immediately or
    emitting the first drift). Only issues that already have drift recorded
    and have exceeded the grace period are escalated here. 0 disables the
    escape (pre-#654 hold-forever).

    Issue #1917: a death classified as a provider throttle is a fleet-wide
    condition, not a worker-quality signal, so while the provider cooldown
    the classifier armed is still active (``throttled_until`` in the
    future) it never escalates through this timed backstop -- the same
    #1684 exemption the rework lanes apply to their caps. The entry falls
    through to the caller's normal handling, which returns the issue to
    the dispatchable pool; the dispatch governor's provider_throttled
    deferral holds the actual re-dispatch until ``throttled_until`` passes.
    The classification is read from ``dead_worker_failure_kind``, stamped
    on the entry by the stall/dead reap lanes before the sidecar is
    reaped.

    The exemption is bounded by the throttle window it exists to ride
    out. Once ``throttled_until`` has passed -- or no window was ever
    stamped -- the #654 backstop resumes: in the PR-linked case the
    sub-branches only emit drift once and then short-circuit on the
    fingerprint (a no-op clean exit, a non-request_changes decision, a
    head change without a review callback), so a stamp that held forever
    would reintroduce exactly the wedge this backstop fixes. An expired
    or unparseable window fails closed to the normal reap timer.

    Returns the (possibly replaced) ``state`` mapping -- the escalation
    helpers rebuild it -- and ``True`` when the entry was escalated, so the
    caller appends to ``reap_escalations`` and moves to the next issue.
    """

    # Deferred: workflow.py imports this module top-level, so a top-level
    # import here would cycle. Attribute access through the module object
    # also keeps suite patches on ``charlie_work.workflow.<name>`` live.
    import charlie_work.workflow as _wf

    orphan_drift_at = entry.get("orphan_drift_at")
    if orphan_drift_at is None or dead_dispatched_reap_minutes <= 0:
        return state, False
    if is_provider_throttle_failure(entry.get("dead_worker_failure_kind")):
        throttled_until_dt = _wf._parse_iso_timestamp(state.get("throttled_until"))
        if throttled_until_dt is not None and throttled_until_dt > now:
            return state, False
    drift_dt = _wf._parse_iso_timestamp(orphan_drift_at)
    if drift_dt is None or (now - drift_dt).total_seconds() / 60 < dead_dispatched_reap_minutes:
        return state, False
    pr_number = int(pr_data["number"]) if pr_data else None
    terminal = find_worker_terminal_status(sessions_dir, issue_number)
    terminal_exit_code = terminal.get("exit_code") if terminal else None
    state = _wf._escalate_issue(
        state,
        issue_number,
        reason="dead_dispatched_worker_reap",
        reason_class="mechanical",
        pr_number=pr_number,
        issue_extra={
            "dispatched_at": None,
            "orphan_drift_fingerprint": None,
            "orphan_drift_at": None,
        },
    )
    sweep_events.append(
        (
            "dead_dispatched_worker_reaped",
            {
                "issue_number": issue_number,
                "pr_number": pr_number,
                "previous_status": "dispatched",
                "reason": "dead_dispatched_worker_reap",
                "orphan_drift_at": orphan_drift_at,
                "reap_minutes": dead_dispatched_reap_minutes,
                "exit_code": terminal_exit_code,
            },
        )
    )
    return state, True


def handle_dead_worker_completed_outcome(
    *,
    state: dict[str, Any],
    sweep_events: list[tuple[str, dict[str, Any]]],
    entry: dict[str, Any],
    issue_number: int,
    pr_number: int,
    pr_data: dict[str, Any],
    reviewed_head_sha: str | None,
    live_head_sha: str | None,
    terminal_pid: Any,
    terminal_exit_code: int | None,
    terminal_duration_seconds: Any,
    repo_root: Any,
    worktrees_dir: Path | None,
    outcome_apply_routes: list[tuple[int, int]],
    review_routes: list[OrphanedWorkerReviewRoute],
    review_callback: Callable[[int], Any] | None,
    drift_fingerprint: Callable[..., str],
    extra_payload: dict[str, Any] | None = None,
) -> bool:
    """Recover a dead worker that provably completed its handoff (#1911).

    Returns ``True`` only when there is no terminal record
    (``terminal_exit_code is None``) AND the worktree holds a fresh,
    on-target ``.worker-outcome.json`` -- written after this dispatch's
    ``dispatched_at`` (a previous session's leftover does not count),
    reporting ``push_succeeded``, and pinning ``head_sha`` to the live
    head. In that case the outcome is queued for the post-lock #1877
    apply pass (idempotent: ``APPLIED_HEADS_KEY`` dedups on the reported
    head, so a transient failure retries next pass while an
    already-applied outcome is never re-queued) and -- when a review
    callback is available -- the issue is also queued for the post-lock
    ``review_routes`` drain (issue #1915): the drain re-checks that the
    outcome applied, then calls ``review()`` and flips
    ``dispatched`` -> ``reviewing`` on a fresh packet or, when review
    cannot produce one, returns the issue to ``rework_requested`` -- still
    without a ``worker_death_at`` credit. False-crediting a worker death
    here while the issue sat ``dispatched`` forever was the two-step
    failure that drove the swole#198 0-commit
    ``no_op_rework_attempts_cap_exceeded`` loop. The
    drift fingerprint is deliberately NOT marked on the routed path: the
    drain is what resolves the finding, so an unapplied/skipped route must
    re-collect cleanly on the next pass. Without a review callback the
    finding surfaces once via the same fingerprinted drift the clean-exit
    (#773) branch uses, and ``orphan_drift_at`` arms the #654 time-based
    reap backstop either way so a route that never resolves still
    converges.

    Every negative answer (a recorded exit code -- zero handled by the
    caller's own branch, non-zero being a confirmed crash -- a missing
    branch/worktree/timestamp, a stale or off-target outcome) returns
    ``False`` and leaves the caller's worker-death path untouched.
    """

    # Deferred: workflow.py imports this module top-level, so a top-level
    # import here would cycle. Attribute access through the module object
    # also keeps suite patches on ``charlie_work.workflow.<name>`` live.
    import charlie_work.workflow as _wf

    if terminal_exit_code is not None or not live_head_sha:
        return False
    branch = pr_data.get("headRefName") or entry.get("branch_name")
    if not branch or not isinstance(repo_root, Path) or worktrees_dir is None:
        return False
    outcome = fresh_completed_worker_outcome(
        worktree_path_for_branch(repo_root, branch, worktrees_dir),
        live_head_sha=live_head_sha,
        dispatched_at=_wf._parse_iso_timestamp(entry.get("dispatched_at")),
    )
    if outcome is None:
        return False
    outcome_head_sha = outcome.get("head_sha")
    applied_heads = state.get(APPLIED_HEADS_KEY, {})
    if not (
        isinstance(applied_heads, dict)
        and applied_heads.get(str(issue_number)) == outcome_head_sha
    ):
        outcome_apply_routes.append((issue_number, pr_number))
    fingerprint = drift_fingerprint(
        reason="dead_worker_completed_outcome",
        reviewed_head_sha=reviewed_head_sha,
    )
    if entry.get("orphan_drift_fingerprint") == fingerprint:
        return True
    if review_callback is not None:
        # Issue #1915: applying the outcome is only half the recovery -- the
        # issue must also leave ``dispatched`` or it dead-ends until the
        # 60-minute ``dead_dispatched_worker_reap`` backstop escalates it
        # (swole#198/PR#348). Queue a review route like the head-advanced
        # branch below: post-lock, once the outcome-apply drain has run,
        # ``review()`` either produces a fresh packet (the drain flips the
        # issue to ``reviewing``) or cannot (the drain returns it to
        # ``rework_requested``, still without a death credit, so the
        # ordinary dispatch loop owns the still-outstanding rework). The
        # drift fingerprint is deliberately NOT marked here -- the drain is
        # what resolves the finding, so a route whose apply has not landed
        # yet must re-collect cleanly next pass -- but ``orphan_drift_at``
        # still arms so the #654 backstop stays the terminal for a route
        # that never resolves (e.g. an apply that can never succeed).
        if entry.get("orphan_drift_at") is None:
            entry["orphan_drift_at"] = _wf.utc_now()
        review_routes.append(
            OrphanedWorkerReviewRoute(
                issue_number=issue_number,
                pr_number=pr_number,
                reviewed_head_sha=reviewed_head_sha,
                live_head_sha=live_head_sha,
                fingerprint=fingerprint,
                reason="dead_worker_completed_outcome",
            )
        )
        return True
    entry["orphan_drift_fingerprint"] = fingerprint
    entry["orphan_drift_at"] = _wf.utc_now()
    sweep_events.append(
        (
            "orphaned_worker_drift",
            {
                "issue_number": issue_number,
                "pr_number": pr_number,
                "previous_status": "dispatched",
                "reason": "dead_worker_completed_outcome",
                "pid": terminal_pid,
                "exit_code": terminal_exit_code,
                "duration_seconds": terminal_duration_seconds,
                "worker_outcome_head_sha": outcome_head_sha,
                **(extra_payload or {}),
            },
        )
    )
    return True


def handle_dead_worker_with_pr(
    *,
    state: dict[str, Any],
    sweep_events: list[tuple[str, dict[str, Any]]],
    entry: dict[str, Any],
    issue_number: int,
    pr_data: dict[str, Any],
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    gh: GitHubLike,
    review_callback: Callable[[int], Any] | None,
    repo_root: Any,
    worktrees_dir: Path | None,
    review_routes: list[OrphanedWorkerReviewRoute],
    outcome_apply_routes: list[tuple[int, int]],
    pr_orphan_unreviewed_details: dict[int, dict[str, Any]],
    drift_fingerprint: Callable[..., str],
) -> None:
    """Classify one dead dispatched worker that still has an open PR.

    Verbatim move of the ``if pr_data:`` arm of
    ``_detect_and_handle_orphaned_workers``'s per-issue loop: resolves the
    last review decision and the durable terminal-status probe, then routes
    to exactly one of -- clean-exit no-op drift (#773), completed-outcome
    recovery (#1911, via :func:`handle_dead_worker_completed_outcome`),
    worker-death reset to ``rework_requested`` (#1134), head-advanced review
    routing (#339), ``pr-open`` label advancement for an unverdicted PR
    (#1128), or the shared ``dead_worker_unsafe_to_auto_reset`` drift
    fallback. Every ``return`` below corresponds to a ``continue`` in the
    original loop body; paths that fall off the end rely on the caller's
    loop tail writing ``entry`` back into ``state``, as before.
    """

    # Deferred: workflow.py imports this module top-level, so a top-level
    # import here would cycle. Attribute access through the module object
    # also keeps suite patches on ``charlie_work.workflow.<name>`` live.
    import charlie_work.workflow as _wf

    pr_number = int(pr_data["number"])
    pr_state = state.get("prs", {}).get(str(pr_number), {})
    live_head_sha = pr_data.get("headRefOid")
    # Issue #1362 Stage 1: read the last review decision through
    # the single file-first reader (flat file, falling back to
    # the highest archived round) instead of state.json's
    # decision/reviewed_head_sha fields, which can lag a
    # concurrent record_review/void -- the #1340 divergence
    # class AC1 exists to eliminate.
    resolved_decision = review_decision(
        state_file.parent / "prs" / f"pr-{pr_number}", None, live_head_sha
    )
    last_decision = resolved_decision.decision
    reviewed_head_sha = resolved_decision.reviewed_head_sha

    # Issue #773: measurement-first payload enrichment. A dead PID
    # alone cannot distinguish a worker that crashed from one that
    # exited 0 having pushed nothing -- both present identically
    # to `_worker_pid_alive`. `find_worker_terminal_status` reads
    # the durable record `start_terminal_status_watcher`
    # (process_utils.py) writes at the moment a claude-code worker
    # actually exits; it returns None for legacy sessions, sessions
    # from adapters that don't write one (e.g. devin-shell), or
    # any session whose watcher never got to run (e.g. orchestrator
    # restart mid-session). `terminal_exit_code` is deliberately
    # left as None in all of those cases rather than guessed at --
    # every event below records it as-is so the two populations
    # (confirmed clean exit vs. everything else) are queryable
    # retrospectively even before they're fully separable.
    terminal = find_worker_terminal_status(sessions_dir, issue_number)
    terminal_pid = entry.get("worker_pid")
    terminal_exit_code = terminal.get("exit_code") if terminal else None
    terminal_duration_seconds = terminal.get("duration_seconds") if terminal else None

    if last_decision == "request_changes" and reviewed_head_sha and live_head_sha:
        if reviewed_head_sha == live_head_sha:
            if terminal_exit_code == 0:
                # The worker exited cleanly (exit code 0) rather
                # than crashing -- e.g. it was handed an empty
                # rework brief with nothing left to act on. Do NOT
                # auto-reset to rework_requested: that would spend
                # one of max_auto_redispatch's attempts on a
                # worker that never had anything to change,
                # eventually escalating a benign no-op to
                # agent:human-needed (issue #773). Surface it once
                # instead, via the same fingerprinted
                # surface-once convergence the other drift
                # branches below already use, so a human/janitor
                # can decide whether the review itself needs
                # revisiting -- retrying a dispatch that already
                # proved it produces no change on this exact head
                # would just repeat the no-op.
                fingerprint = drift_fingerprint(
                    reason="dead_worker_clean_exit_no_op",
                    reviewed_head_sha=reviewed_head_sha,
                )
                if entry.get("orphan_drift_fingerprint") == fingerprint:
                    state["issues"][str(issue_number)] = entry
                    return
                entry["orphan_drift_fingerprint"] = fingerprint
                entry["orphan_drift_at"] = _wf.utc_now()
                sweep_events.append(
                    (
                        "orphaned_worker_drift",
                        {
                            "issue_number": issue_number,
                            "pr_number": pr_number,
                            "previous_status": "dispatched",
                            "reason": "dead_worker_clean_exit_no_op",
                            "pid": terminal_pid,
                            "exit_code": terminal_exit_code,
                            "duration_seconds": terminal_duration_seconds,
                        },
                    )
                )
            elif not handle_dead_worker_completed_outcome(
                state=state,
                sweep_events=sweep_events,
                entry=entry,
                issue_number=issue_number,
                pr_number=pr_number,
                pr_data=pr_data,
                reviewed_head_sha=reviewed_head_sha,
                live_head_sha=live_head_sha,
                terminal_pid=terminal_pid,
                terminal_exit_code=terminal_exit_code,
                terminal_duration_seconds=terminal_duration_seconds,
                repo_root=repo_root,
                worktrees_dir=worktrees_dir,
                outcome_apply_routes=outcome_apply_routes,
                review_routes=review_routes,
                review_callback=review_callback,
                drift_fingerprint=drift_fingerprint,
            ):
                # No terminal record and no fresh on-target
                # outcome file, or a non-zero exit code (a
                # recorded crash never counts as a completed
                # outcome -- issue #1911): unchanged from
                # pre-#773 behavior -- safe to reset
                # to rework_requested (PR head unchanged since
                # request_changes).
                entry["status"] = "rework_requested"
                entry["dispatched_at"] = None
                # Issue #1134: record this as a worker death, not
                # a no-op.  A death redispatch must not count
                # against the no-op rework cap — the worker may
                # have completed its work but died before pushing
                # (salvageable stranded commits).  A separate
                # death counter with its own escalation reason
                # (worker_death_loop) lets the operator triage
                # "check the worktree" vs. "worker is spinning."
                # Issue #1917: skip the credit entirely for a
                # provider-throttle-classified death — the same
                # #1684 exemption the rework lanes apply, so the
                # death can never inflate ``worker_death_at`` for a
                # later, genuinely different death's cap check.
                death_ts = _wf.utc_now()
                if not is_provider_throttle_failure(entry.get("dead_worker_failure_kind")):
                    entry["worker_death_at"] = _credit_worker_death(
                        entry,
                        at=death_ts,
                    )
                sweep_events.append(
                    (
                        "orphaned_worker_recovered",
                        {
                            "issue_number": issue_number,
                            "pr_number": pr_number,
                            "previous_status": "dispatched",
                            "new_status": "rework_requested",
                            "reason": "dead_worker_with_request_changes",
                            "pid": terminal_pid,
                            "exit_code": terminal_exit_code,
                            "duration_seconds": terminal_duration_seconds,
                            "worker_death_at": death_ts,
                        },
                    )
                )
        else:
            # PR head has changed - route to review if possible,
            # otherwise surface as a drift finding (once per fingerprint).
            fingerprint = drift_fingerprint(
                reason="dead_worker_with_head_change",
                reviewed_head_sha=reviewed_head_sha,
                live_head_sha=live_head_sha,
            )
            if entry.get("orphan_drift_fingerprint") == fingerprint:
                # Already handled/failed for this exact head advance;
                # don't re-emit or retry.
                state["issues"][str(issue_number)] = entry
                return
            if review_callback is not None:
                review_routes.append(
                    OrphanedWorkerReviewRoute(
                        issue_number=issue_number,
                        pr_number=pr_number,
                        reviewed_head_sha=reviewed_head_sha,
                        live_head_sha=live_head_sha,
                        fingerprint=fingerprint,
                        reason="dead_worker_with_head_change",
                    )
                )
            else:
                entry["orphan_drift_fingerprint"] = fingerprint
                entry["orphan_drift_at"] = _wf.utc_now()
                sweep_events.append(
                    (
                        "orphaned_worker_drift",
                        {
                            "issue_number": issue_number,
                            "pr_number": pr_number,
                            "previous_status": "dispatched",
                            "last_decision": last_decision,
                            "reviewed_head_sha": reviewed_head_sha,
                            "live_head_sha": live_head_sha,
                            "reason": "dead_worker_with_head_change",
                            "pid": terminal_pid,
                            "exit_code": terminal_exit_code,
                            "duration_seconds": terminal_duration_seconds,
                        },
                    )
                )
    else:
        # Not a simple request_changes case.
        # Issue #1109: a dead worker on an approved PR is not
        # unclassifiable when the post-approval rework lane
        # (#674 -> PR #685, plus the merge-conflict and no-op
        # rework lanes that share ``_route_to_rework``) dispatched
        # it. Those lanes set the PR state status to
        # ``rework_requested`` while preserving
        # ``decision="approved"``, and the worker is dispatched to
        # fix CI/a conflict without re-litigating the review. If
        # that worker dies before pushing (head unchanged since
        # review), the issue previously wedged in ``dispatched``
        # forever because this sweep refused to auto-reset on a
        # non-``request_changes`` decision -- no redispatch, no
        # cap consumption, invisible to every downstream lane.
        # Treat ``decision="approved"`` + PR-state
        # ``rework_requested`` + head unchanged as safe to
        # auto-reset, mirroring the request_changes branch above
        # (including the #773 clean-exit-no-op sub-case so a
        # benign exit-0 worker does not burn redispatch attempts).
        # ``dead_worker_unsafe_to_auto_reset`` is kept only for
        # genuinely unclassifiable decisions -- an approved PR
        # whose PR state does not carry ``rework_requested`` has
        # no evidence a post-approval rework lane dispatched this
        # worker, so auto-resetting would be a guess.
        #
        # Issue #1128: when the dead worker has an OPEN PR with no
        # review verdict yet (``last_decision`` is null/absent),
        # the "unsafe to auto-reset" judgment stays -- the PR
        # carries the work, so re-dispatching would duplicate it --
        # but leaving the issue on ``agent:in-progress`` makes the
        # state machine assert a live worker the reconciler just
        # confirmed dead, and review dispatch (which keys off
        # ``agent:pr-open``) never sees the salvage PR. Transition
        # to ``pr-open`` via the same LabelConfig-driven swap the
        # ``orphaned_worker_opened_pr`` lane uses. On label write
        # failure, fall through to the conservative drift path so
        # the next pass re-attempts rather than resetting the
        # worker.
        pr_state_status = pr_state.get("status")
        if (
            last_decision == "approved"
            and pr_state_status == "rework_requested"
            and reviewed_head_sha
            and live_head_sha
            and reviewed_head_sha == live_head_sha
        ):
            if terminal_exit_code == 0:
                # Clean exit with no push -- same #773 rationale
                # as the request_changes branch: do not spend a
                # redispatch attempt on a worker that produced no
                # change on this exact head.
                fingerprint = drift_fingerprint(
                    reason="dead_worker_clean_exit_no_op",
                    reviewed_head_sha=reviewed_head_sha,
                )
                if entry.get("orphan_drift_fingerprint") == fingerprint:
                    state["issues"][str(issue_number)] = entry
                    return
                entry["orphan_drift_fingerprint"] = fingerprint
                entry["orphan_drift_at"] = _wf.utc_now()
                sweep_events.append(
                    (
                        "orphaned_worker_drift",
                        {
                            "issue_number": issue_number,
                            "pr_number": pr_number,
                            "previous_status": "dispatched",
                            "reason": "dead_worker_clean_exit_no_op",
                            "decision": "approved",
                            "pr_state_status": pr_state_status,
                            "pid": terminal_pid,
                            "exit_code": terminal_exit_code,
                            "duration_seconds": terminal_duration_seconds,
                        },
                    )
                )
            elif not handle_dead_worker_completed_outcome(
                state=state,
                sweep_events=sweep_events,
                entry=entry,
                issue_number=issue_number,
                pr_number=pr_number,
                pr_data=pr_data,
                reviewed_head_sha=reviewed_head_sha,
                live_head_sha=live_head_sha,
                terminal_pid=terminal_pid,
                terminal_exit_code=terminal_exit_code,
                terminal_duration_seconds=terminal_duration_seconds,
                repo_root=repo_root,
                worktrees_dir=worktrees_dir,
                outcome_apply_routes=outcome_apply_routes,
                review_routes=review_routes,
                review_callback=review_callback,
                drift_fingerprint=drift_fingerprint,
                extra_payload={
                    "decision": "approved",
                    "pr_state_status": pr_state_status,
                },
            ):
                # No terminal record and no fresh on-target
                # outcome file, or a non-zero exit code
                # (issue #1911): safe to reset to rework_requested
                # (PR head
                # unchanged since the approved review, and the
                # post-approval rework lane dispatched this
                # worker). Records this as a worker death with a
                # distinct reason so the death counter (issue
                # #1134) and the redispatch cap (issue #165) apply
                # exactly as they do for request_changes.
                entry["status"] = "rework_requested"
                entry["dispatched_at"] = None
                death_ts = _wf.utc_now()
                # Issue #1917: provider-throttle deaths are not credited —
                # same #1684 exemption as the request_changes branch above.
                if not is_provider_throttle_failure(entry.get("dead_worker_failure_kind")):
                    entry["worker_death_at"] = _credit_worker_death(
                        entry,
                        at=death_ts,
                    )
                sweep_events.append(
                    (
                        "orphaned_worker_recovered",
                        {
                            "issue_number": issue_number,
                            "pr_number": pr_number,
                            "previous_status": "dispatched",
                            "new_status": "rework_requested",
                            "reason": "dead_worker_with_approved_rework",
                            "decision": "approved",
                            "pr_state_status": pr_state_status,
                            "pid": terminal_pid,
                            "exit_code": terminal_exit_code,
                            "duration_seconds": terminal_duration_seconds,
                            "worker_death_at": death_ts,
                        },
                    )
                )
        else:
            # Not the #1109 approved+rework_requested classified
            # case. This branch covers two populations that share
            # one fingerprinted drift fallback below:
            #   (a) #1128: ``last_decision`` is None or "pending"
            #       (open PR, no terminal review verdict yet) --
            #       try advancing to ``pr-open``; on label-write
            #       failure or missing details, fall through to
            #       the shared drift. Issue #1362 Stage 1: this
            #       must match ``_has_no_review_verdict_yet``'s
            #       predicate above (``.missing or .decision ==
            #       "pending"``) or a pending-packet PR would be
            #       precomputed into ``pr_orphan_unreviewed_details``
            #       but never consulted here, re-stranding the
            #       issue on ``agent:in-progress``.
            #   (b) genuinely unclassifiable decisions -- fall
            #       through to the shared drift directly.
            if last_decision is None or last_decision == "pending":
                details = pr_orphan_unreviewed_details.get(issue_number)
                if details is not None:
                    active_labels = details["active_labels"]
                    issue_labels = details["issue_labels"]
                    label_write_ok = True
                    for label in sorted(active_labels):
                        if not gh.remove_issue_label(issue_number, label):
                            label_write_ok = False
                    if config.labels.pr_open not in issue_labels:
                        if not gh.add_issue_label(issue_number, config.labels.pr_open):
                            label_write_ok = False
                    if label_write_ok:
                        entry["status"] = PASSIVE_OPEN_STATUS
                        entry["dispatched_at"] = None
                        # Clear any prior drift fingerprint so a
                        # later regression on this issue re-surfaces.
                        entry["orphan_drift_fingerprint"] = None
                        entry["orphan_drift_at"] = None
                        # Issue #1134's death-crediting invariant
                        # applies here too: this worker died just
                        # as much as the request_changes/approved
                        # branches above, and its issue can still
                        # return to ``rework_requested`` once the
                        # PR is eventually reviewed (a salvage-
                        # opened PR with no verdict yet is exactly
                        # this case). Before this fix, this was
                        # the one dead-worker branch that never
                        # touched ``worker_death_at`` -- a
                        # redispatch caused by THIS death read
                        # back as an uncredited no-op on every
                        # later no-op-rework-cap check.
                        # Issue #1917: provider-throttle deaths are
                        # not credited — same #1684 exemption as the
                        # branches above.
                        death_ts = _wf.utc_now()
                        if not is_provider_throttle_failure(entry.get("dead_worker_failure_kind")):
                            entry["worker_death_at"] = _credit_worker_death(
                                entry,
                                at=death_ts,
                            )
                        sweep_events.append(
                            (
                                "orphaned_worker_advanced_to_pr_open",
                                {
                                    "issue_number": issue_number,
                                    "pr_number": pr_number,
                                    "previous_status": "dispatched",
                                    "new_status": PASSIVE_OPEN_STATUS,
                                    "reason": "dead_worker_unsafe_to_auto_reset_open_unreviewed_pr",
                                    "removed_labels": sorted(active_labels),
                                    "pid": terminal_pid,
                                    "exit_code": terminal_exit_code,
                                    "duration_seconds": terminal_duration_seconds,
                                    "label_write_ok": True,
                                    "worker_death_at": death_ts,
                                },
                            )
                        )
                        state["issues"][str(issue_number)] = entry
                        return
                    # Label write failed -- fall through to the
                    # fingerprinted drift path so the next pass
                    # re-attempts the transition (the drift
                    # fingerprint gates only re-emission of the
                    # diagnostic, not the transition retry above,
                    # which runs first on every pass).
            # Genuinely unclassifiable decision, or #1128 label-
            # write failure -- surface as drift once. One shared
            # fingerprinted fallback for both lanes (#1109 keeps
            # its own classified branch above; this covers
            # everything else).
            fingerprint = drift_fingerprint(
                reason="dead_worker_unsafe_to_auto_reset",
                last_decision=last_decision or "",
                pr_number=pr_number,
            )
            if entry.get("orphan_drift_fingerprint") != fingerprint:
                entry["orphan_drift_fingerprint"] = fingerprint
                entry["orphan_drift_at"] = _wf.utc_now()
                sweep_events.append(
                    (
                        "orphaned_worker_drift",
                        {
                            "issue_number": issue_number,
                            "pr_number": pr_number,
                            "previous_status": "dispatched",
                            "last_decision": last_decision,
                            "reason": "dead_worker_unsafe_to_auto_reset",
                            "pid": terminal_pid,
                            "exit_code": terminal_exit_code,
                            "duration_seconds": terminal_duration_seconds,
                        },
                    )
                )
