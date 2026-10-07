"""Rework routing for dead workers that still own an open PR.

Moved verbatim from the retired ``dead_worker_reap`` module: restoring
``rework_requested`` on a dead rework session (issue #295) and routing a dead
worker's PR into the pre-review rework lane.
"""

from __future__ import annotations


from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import (
    DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS,
    OrchestratorConfig,
)
from ..dispatch_selection import (
    _credit_worker_death,
    _windowed_redispatch_at,
    _windowed_worker_death_at,
)
from ..escalation import _escalate_issue, _escalation_edge
from ..github import (
    GitHubLike,
)
from ..host import current as _host_current
from ..labels import TransitionOutcome
from ..no_op_checkpoint import _paired_death_count
from ..review_decision import review_decision
from ..rework_attempt_exemption import exempt_provider_throttle_rework_death
from ..rework_prompts import _write_rework_prompt
from ..state import (
    load_state,
    state_lock,
)
from ..throttle_signatures import is_provider_throttle_failure
from ..worker import WorkerView
from ..worktree import (
    SalvagePushResult,
    salvage_push_stranded_commits,
    worktree_ahead_of_sha,
)
from ..write_gate import WriteGate, require_write_gate
from .effects_sessions import _is_startup_death, _worker_death_bounded_runtime_seconds


def _rework_pr_for_worker(
    open_prs_by_issue: dict[int, list[dict[str, Any]]],
    worker: WorkerView,
) -> dict[str, Any] | None:
    """Return the most likely open PR for a dead/launch-failed rework worker.

    Prefer the PR whose ``headRefName`` matches the worker's branch, falling back
    to the lowest PR number for the issue.
    """
    prs = open_prs_by_issue.get(worker.issue_number, [])
    if not prs:
        return None
    for pr in prs:
        if pr.get("headRefName") == worker.branch:
            return pr
    return min(prs, key=lambda pr: int(pr["number"]))


def _reap_restore_rework_requested(
    state_file: Path,
    gh: GitHubLike,
    config: OrchestratorConfig,
    open_prs_by_issue: dict[int, list[dict[str, Any]]],
    worker: WorkerView,
    failure_kind: str | None = None,
    *,
    repo_root: Path | None = None,
    write_gate: WriteGate,
) -> None:
    """Restore a dead/launch-failed rework worker to ``rework_requested``, or
    escalate it to a human when the redispatch cap is exhausted or the
    failure is deterministic.

    Called when a reaped session has an open linked PR. The PR must have a
    ``request_changes`` verdict that is still LIVE for the current head
    (``reviewed_head_sha == live_head_sha``) — ``has_request_changes`` is the
    single source of truth here. A ``rework-prompt.md`` on disk is only ever
    a supplement to that signal, never an independent trigger (issue #315
    finding 1): the prompt file is written once per PR by
    ``_write_rework_prompt`` and is never deleted, so by itself it cannot
    distinguish "still awaiting this exact rework cycle" from "stale leftover
    from an earlier cycle whose head has since been approved or superseded".
    ``has_request_changes`` already re-derives from the PR's current review
    record on every call, so gating on it makes the whole check
    self-invalidating the moment the head advances or gets approved — the
    same property the prompt-existence check was missing.

    Issue #295: this is the rework counterpart to the no-open-PR relabel path.
    It preserves the liveness fingerprint (``worker_pid`` /
    ``worker_process_start_time``) for the recovery probe, matching the
    no-open-PR dead-session escalation branch (issue #282).

    Issue #315 finding 2: this lane must consult the same redispatch-cap and
    deterministic-failure-kind escalation rules the no-open-PR lanes already
    enforce (~line 936 and ~line 1138) — otherwise a rework worker that dies
    at launch every time (or whose failure_kind is confirmed-deterministic,
    e.g. ``worktree_unsafe``) loops rework_requested forever instead of
    escalating to a human. Rework workers always have an open PR, so they
    never reach those lanes' checks; the equivalent must live here.

    Issue #1134: a worker that dies before pushing leaves the PR head
    unchanged but is NOT a no-op — the worker may have completed its work
    and died mid-push with salvageable stranded commits.  This lane records
    each non-terminal death in ``worker_death_at`` (parallel to the orphan
    sweep) and separates the death count from the no-op count in the cap
    check.  A death-loop escalates with ``worker_death_loop`` (triage:
    "check the worktree for stranded work") instead of
    ``redispatch_cap_exceeded`` (triage: "worker is spinning"), and
    includes ``stranded_commits`` in the escalation payload.

    Issue #1239: the rework lane had the stranded-commit detector (#1134)
    but not the remediation the fresh-dispatch lane has (#1248).  Before
    counting a death, salvage-push stranded commits via the same
    sanctioned-git path (``salvage_push_stranded_commits``: ls-remote
    before pushing, never force-push, fast-forward only).  When the push
    succeeds the death is NOT recorded and the issue resets to
    ``rework_requested``; the next ``dispatch_rework`` pass sees the PR
    head has moved past the ``request_changes`` verdict and routes to
    review (packet regeneration supersedes the verdict).  Only a death
    that produced NO pushable commit counts toward ``worker_death_loop``;
    the genuinely-empty death-loop escalation is unchanged.  ``repo_root``
    gates the salvage attempt (``None`` for callers without a checkout,
    e.g. unit tests of the cap logic itself).
    """
    write_gate = require_write_gate(write_gate)
    pr_data = _rework_pr_for_worker(open_prs_by_issue, worker)
    if pr_data is None:
        return

    pr_number = int(pr_data["number"])
    live_head_sha = pr_data.get("headRefOid")
    prs_dir = state_file.parent / "prs"
    rework_prompt_path = prs_dir / f"pr-{pr_number}" / "rework-prompt.md"

    # Issue #1239 round-2: workers are discovered from sidecar files,
    # decoupled from state.json, so by the time we reach here the issue's
    # status may have already moved off ``dispatched`` (e.g. a concurrent
    # loop pass re-dispatched, escalated, or the issue was closed).  Re-check
    # under the state lock BEFORE any network push — a salvage push to the
    # shared origin remote for an issue that is no longer dispatched would be
    # an unaudited side effect with no event trail if it succeeded.  This is
    # a short, separate state_lock scope so the network git push below stays
    # outside any lock; the same precondition is re-checked inside the
    # success branch's state_lock (line ~3554) and the death-recording
    # block's state_lock (line ~3593) — both still load-bearing because the
    # status can move again between this check and those scopes.
    with state_lock(state_file):
        state = load_state(state_file)
        entry = state["issues"].get(str(worker.issue_number), {})
        if not isinstance(entry, dict) or entry.get("status") != "dispatched":
            return

    # Issue #1362 Stage 1: read through the single review-decision reader
    # instead of state.json's ``decision``/``reviewed_head_sha``.  Computed
    # outside the state lock: it reads only the review-decision file and the
    # PR snapshot's ``headRefOid`` (``live_head_sha``), neither of which
    # depends on state.json.
    resolved_decision = review_decision(prs_dir / f"pr-{pr_number}", None, live_head_sha)
    has_request_changes = (
        resolved_decision.decision == "request_changes" and not resolved_decision.stale
    )
    if not has_request_changes:
        return

    # Issue #1239: salvage-push stranded commits BEFORE counting a death.
    # Network I/O (git push) — kept outside the state lock, matching the
    # fresh-dispatch salvage lane (~line 2168).  ``salvage_push_stranded_commits``
    # is the sanctioned-git path: ls-remote before pushing (never trust the
    # sidecar's ``push_succeeded``), never force-push, fast-forward only.
    salvage_result: SalvagePushResult | None = None
    if repo_root is not None and live_head_sha and worker.branch and worker.worktree_path:
        salvage_result = salvage_push_stranded_commits(
            repo_root,
            worker.branch,
            Path(worker.worktree_path),
            base_ref=config.dispatch.base_ref,
            dry_run=write_gate.dry_run,
        )

    # Issue #1239: when stranded commits were published, reset to
    # rework_requested WITHOUT recording a death and return — the PR head
    # moved past the request_changes verdict, so the next dispatch_rework
    # pass routes to review (packet regeneration supersedes the verdict)
    # instead of re-dispatching into the same tail-death.  A death that
    # produced a pushable commit is a completion with a failed last step,
    # not a failed attempt.  The label transition mirrors the non-salvage
    # restore path below (edge="rework_requested").
    if salvage_result is not None and salvage_result.pushed:
        with state_lock(state_file):
            state = load_state(state_file)
            entry = state["issues"].get(str(worker.issue_number), {})
            if not isinstance(entry, dict) or entry.get("status") != "dispatched":
                return
            entry["status"] = "rework_requested"
            entry["dispatched_at"] = None
            state["issues"][str(worker.issue_number)] = entry
            state = write_gate.append_event(
                state,
                "rework_stranded_commits_salvaged",
                {
                    "issue_number": worker.issue_number,
                    "pr_number": pr_number,
                    "previous_status": "dispatched",
                    "new_status": "rework_requested",
                    "reason": "dead_rework_worker_salvaged",
                    "failure_kind": failure_kind,
                    "commit_count": salvage_result.commit_count,
                    "old_remote_sha": salvage_result.old_remote_sha,
                    "new_remote_sha": salvage_result.new_remote_sha,
                },
            )
            write_gate.save_state(state)
        result = write_gate.transition(gh, config.labels, worker.issue_number, "rework_requested")
        if result.outcome != TransitionOutcome.APPLIED:
            with state_lock(state_file):
                state = load_state(state_file)
                entry = state["issues"].get(str(worker.issue_number), {})
                entry["label_error"] = {
                    "edge": "rework_requested",
                    "outcome": result.outcome.value,
                    "add_failures": result.add_failures,
                    "remove_failures": result.remove_failures,
                }
                state["issues"][str(worker.issue_number)] = entry
                write_gate.save_state(state)
        return

    with state_lock(state_file):
        state = load_state(state_file)
        entry = state["issues"].get(str(worker.issue_number), {})
        if not isinstance(entry, dict) or entry.get("status") != "dispatched":
            return

        # Diagnostic only (issue #315 finding 1) — never gates the restore by
        # itself; see the docstring above.
        has_rework_prompt = rework_prompt_path.exists()

        # Issue #1684: a provider-throttle-classified death (the kinds that
        # arm ``throttled_until`` — quota_exhausted, rate_limited,
        # provider_auth) is a global provider condition, not a worker-quality
        # signal, so it must not consume either cap below. The fleet-wide
        # cooldown is already armed by the classifier; the rework is
        # restored below and re-dispatches once the window opens.
        provider_throttled = is_provider_throttle_failure(failure_kind)

        # Issue #315 finding 2: same window-filtered redispatch_at bookkeeping
        # the sibling lanes use (~line 950-961, ~4186-4194), so the cap below
        # is actually consulted instead of silently never growing.
        redispatch_at = _windowed_redispatch_at(
            entry, window_minutes=config.watchdog.redispatch_window_minutes
        )
        if not provider_throttled:
            redispatch_at = redispatch_at + [
                _host_current().clock.now().isoformat().replace("+00:00", "Z")
            ]

        terminal_failure = failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
        # Issue #807: a deterministic judgment failure (e.g. genuine local
        # commits on the worktree branch) escalates immediately like a
        # terminal_failure but as ``reason_class="judgment"`` so the
        # de-escalation sweep never auto-clears it.
        deterministic_judgment = failure_kind in DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS
        immediate_escalation = terminal_failure or deterministic_judgment

        # Issue #1134: a worker that died before pushing leaves the PR head
        # unchanged, but that is NOT a no-op — the worker may have completed
        # its work and died mid-push with salvageable stranded commits.
        # Record this death in worker_death_at through the same
        # ``_credit_worker_death`` helper the orphan sweep uses (single
        # point of enforcement for the append), and separate the death
        # count from the no-op count in the cap check below.  A death-loop
        # escalates with worker_death_loop (triage: "check the worktree for
        # stranded work") instead of redispatch_cap_exceeded (triage:
        # "worker is spinning").
        if not immediate_escalation and not provider_throttled:
            worker_death_at = _credit_worker_death(entry, kind=failure_kind)
        else:
            worker_death_at = _windowed_worker_death_at(
                entry, window_minutes=config.watchdog.redispatch_window_minutes
            )

        no_op_count = max(0, len(redispatch_at) - len(worker_death_at))
        death_count = len(worker_death_at)
        # Issue #1784 finding 3: paired against redispatch_at so a death
        # credited without a matching redispatch (the orphan sweep's
        # advance-to-pr-open lane, on the ORIGINAL implementer dispatch)
        # cannot escalate a death-loop before this many redispatches
        # actually happened. See ``_paired_death_count``'s docstring.
        # Issue #1993: not crediting a throttle death (above) is not enough
        # -- deaths already in the window would still trip either cap on
        # it (2026-09-29: #1971 escalated worker_death_loop at count 3 on a
        # ``rate_limited`` death). A throttle death is never the one that
        # escalates; the next non-throttle death re-evaluates both caps.
        death_loop = (
            not immediate_escalation
            and not provider_throttled
            and _paired_death_count(redispatch_at=redispatch_at, worker_death_at=worker_death_at)
            > config.watchdog.max_auto_redispatch
        )
        no_op_loop = (
            not immediate_escalation
            and not provider_throttled
            and not death_loop
            and no_op_count > config.watchdog.max_auto_redispatch
        )
        should_escalate = immediate_escalation or death_loop or no_op_loop

        if should_escalate:
            if immediate_escalation:
                reason = failure_kind
            elif death_loop:
                reason = "worker_death_loop"
            else:
                reason = "redispatch_cap_exceeded"
            reason_class = "judgment" if deterministic_judgment else "mechanical"
            # Preserve worker_pid/worker_process_start_time (issue #282): the
            # recovery probe still needs the fingerprint even after escalation.
            issue_extra: dict[str, Any] = {
                "redispatch_at": redispatch_at,
                "dispatched_at": None,
            }
            event_payload: dict[str, Any] = {
                "issue_number": worker.issue_number,
                "pr_number": pr_number,
                "failure_kind": failure_kind,
                "previous_status": "dispatched",
                "reason": "dead_rework_session_escalated",
                "redispatch_count": len(redispatch_at),
            }
            if not immediate_escalation:
                # Persist the death record regardless of which cap fired —
                # the death still happened, and the consumption side
                # (_dispatch_rework_impl) reads it from state.
                issue_extra["worker_death_at"] = worker_death_at
            if death_loop:
                event_payload["reason"] = "worker_death_loop"
                event_payload["worker_death_count"] = death_count
                # Issue #1134: probe the worktree for stranded commits —
                # work the worker completed but died before pushing.
                if live_head_sha and worker.worktree_path:
                    stranded, _err = worktree_ahead_of_sha(
                        Path(worker.worktree_path), live_head_sha
                    )
                    if stranded is not None:
                        issue_extra["stranded_commits"] = stranded
                        event_payload["stranded_commits"] = stranded
            state = _escalate_issue(
                state,
                worker.issue_number,
                reason=reason,
                reason_class=reason_class,
                issue_extra=issue_extra,
            )
            state = write_gate.append_event(
                state,
                "session_failed_escalated",
                event_payload,
            )
            write_gate.save_state(state)
        else:
            dispatched_at = entry.get("dispatched_at")
            entry["status"] = "rework_requested"
            entry["dispatched_at"] = None
            entry["redispatch_at"] = redispatch_at
            if not immediate_escalation:
                entry["worker_death_at"] = worker_death_at
            # Preserve worker_pid (issues #165, #282, #295)
            state["issues"][str(worker.issue_number)] = entry
            # Issue #1106: record the failure_kind and startup-death
            # classification in the PR state so the janitor gate's
            # _route_janitor_gate_failure_to_rework can skip the cap
            # increment when the session died at CLI startup (before
            # the worker's first tool action) instead of miscounting
            # it as a no-op/conflict rework attempt.
            startup_death = _is_startup_death(
                failure_kind, _worker_death_bounded_runtime_seconds(worker)
            )
            state["prs"][str(pr_number)] = {
                **state.get("prs", {}).get(str(pr_number), {}),
                "last_rework_failure_kind": failure_kind,
                "last_rework_was_startup_death": startup_death,
            }
            # Issue #2282: a provider-throttle death refunds its dispatch's
            # redispatch_at stamp and sets the PR exemption flag (single
            # point of enforcement, see ``rework_attempt_exemption``).
            throttle_exempt = exempt_provider_throttle_rework_death(
                state,
                worker.issue_number,
                entry,
                failure_kind,
                dispatched_at=dispatched_at,
                pr_number=pr_number,
                source="dead_rework_session_restore",
                write_gate=write_gate,
            )
            state = write_gate.append_event(
                state,
                "rework_requeued",
                {
                    "issue_number": worker.issue_number,
                    "pr_number": pr_number,
                    "failure_kind": failure_kind,
                    "previous_status": "dispatched",
                    "reason": "dead_rework_session_recovered",
                    "has_request_changes": has_request_changes,
                    "has_rework_prompt": has_rework_prompt,
                    "startup_death": startup_death,
                    "provider_throttle_exempt": throttle_exempt,
                },
            )
            write_gate.save_state(state)

    # Transition labels: escalate (operator_queue for mechanical reasons,
    # human_needed reserved for judgment), or rework_requested (needs_rework),
    # removing the stale in_progress label from the failed launch.
    # Issue #807: the edge must follow reason_class so a deterministic judgment
    # failure (genuine local commits) lands agent:human-needed, not
    # agent:operator-queue. reason_class is only assigned inside the
    # should_escalate branch above, and this ternary only reads it when
    # should_escalate is true, so it is always bound on this access.
    edge = (
        _escalation_edge("redispatch_escalated", reason_class)
        if should_escalate
        else "rework_requested"
    )
    result = write_gate.transition(gh, config.labels, worker.issue_number, edge)
    if result.outcome != TransitionOutcome.APPLIED:
        with state_lock(state_file):
            state = load_state(state_file)
            entry = state["issues"].get(str(worker.issue_number), {})
            entry["label_error"] = {
                "edge": edge,
                "outcome": result.outcome.value,
                "add_failures": result.add_failures,
                "remove_failures": result.remove_failures,
            }
            state["issues"][str(worker.issue_number)] = entry
            write_gate.save_state(state)


def _is_pr_updated_at_older_than(
    pr: dict[str, Any],
    now: datetime,
    minutes: int,
) -> bool:
    """Return True when ``pr["updatedAt"]`` is more than ``minutes`` old.

    Parses ISO-8601 timestamps with an optional ``Z`` suffix, normalizes
    naive datetimes to UTC, and tolerates missing or malformed values.
    """
    updated_at = pr.get("updatedAt")
    if not updated_at:
        return False
    try:
        updated = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return (now - updated).total_seconds() > minutes * 60


def _is_pre_review_rework_candidate(
    pr: dict[str, Any],
    config: OrchestratorConfig,
    now: datetime,
) -> tuple[bool, str]:
    """Detect PRs that are stuck before review and need a rework cycle.

    Returns ``(True, reason)`` when either:

    * ``mergeable`` is ``CONFLICTING`` — the branch cannot be merged and CI
      will not run because GitHub cannot build a merge ref; or
    * ``mergeStateStatus`` is ``DIRTY`` — the rework branch conflicts with the
      base, so the merge ref cannot be built and no ``pull_request`` CI can run; or
    * ``statusCheckRollup`` is empty and the PR's ``updatedAt`` is older than
      ``watchdog.pre_review_rework_stale_minutes`` — the worker opened a PR
      and then died before any checks were created.
    """
    mergeable = str(pr.get("mergeable") or "").upper()
    if mergeable == "CONFLICTING":
        return True, "merge_conflict"

    merge_state = str(pr.get("mergeStateStatus") or "").upper()
    if merge_state == "DIRTY":
        return True, "rework_branch_conflict"

    stale_minutes = config.watchdog.pre_review_rework_stale_minutes
    if stale_minutes <= 0:
        return False, ""

    # Issue #2443: an absent key (``pr_list`` rows carry no rollup) means the
    # checks are unknown, not empty -- never ``stale_empty_checks``.
    status_rollup = pr.get("statusCheckRollup")
    if "statusCheckRollup" not in pr or status_rollup:
        return False, ""

    if _is_pr_updated_at_older_than(pr, now, stale_minutes):
        return True, "stale_empty_checks"

    return False, ""


def _route_dead_worker_to_pre_review_rework(
    state_file: Path,
    gh: GitHubLike,
    config: OrchestratorConfig,
    pr: dict[str, Any],
    issue_number: int,
    reason: str,
    *,
    failure_kind: str | None = None,
    write_gate: WriteGate,
) -> dict[str, Any] | None:
    """Route a dead worker's stuck pre-review PR to the rework pipeline.

    Writes a rebase-onto-main brief, transitions the issue to ``needs_rework``,
    and updates state.json to ``rework_requested``. Idempotent: if the issue is
    already ``rework_requested`` or ``escalated``, this is a no-op.

    Enforces ``watchdog.max_auto_redispatch`` and escalates deterministic
    failures immediately, mirroring the existing redispatch-escalation logic.
    """
    write_gate = require_write_gate(write_gate)
    pr_number = int(pr["number"])
    if reason == "merge_conflict":
        summary = (
            "The PR branch has a merge conflict with the base branch. "
            "Rebase the branch onto the current base branch, resolve the conflicts, "
            "and push. The code changes are already approved; do not re-litigate the review."
        )
    elif reason == "rework_branch_conflict":
        summary = (
            "The rework branch conflicts with the current base branch; GitHub cannot "
            "build the merge ref, so no pull_request CI will run. Resolve the conflicts "
            "manually and push."
        )
        if failure_kind is None:
            failure_kind = "rework_branch_conflict"
    else:
        summary = (
            "The PR was opened but no CI checks have been created after the stale threshold. "
            "Rebase the branch onto the current base branch and push to trigger a fresh CI run. "
            "The existing changes are pre-approved; do not re-litigate the review."
        )

    with state_lock(state_file):
        state = load_state(state_file)
        state.setdefault("issues", {})
        state.setdefault("prs", {})
        entry = state["issues"].get(str(issue_number), {})
        if not isinstance(entry, dict):
            entry = {}
        current_status = entry.get("status")
        if current_status in ("rework_requested", "escalated"):
            return None

        redispatch_at = _windowed_redispatch_at(
            entry, window_minutes=config.watchdog.redispatch_window_minutes
        )
        # Issue #1684: a provider-throttle-classified death (the kinds that
        # arm ``throttled_until``) is a global provider condition, not a
        # worker-quality signal — it must not consume the redispatch cap.
        if not is_provider_throttle_failure(failure_kind):
            redispatch_at = redispatch_at + [
                _host_current().clock.now().isoformat().replace("+00:00", "Z")
            ]

        terminal_failure = failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
        # Issue #807: a deterministic judgment failure escalates immediately
        # but as ``reason_class="judgment"`` so the de-escalation sweep
        # never auto-clears it.
        deterministic_judgment = failure_kind in DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS
        immediate_escalation = terminal_failure or deterministic_judgment
        if immediate_escalation or len(redispatch_at) > config.watchdog.max_auto_redispatch:
            # Issue #783: merge conflict / rework-branch conflict / stale-CI
            # redispatch cap are all process failures, not judgment calls.
            # Issue #807: a deterministic judgment failure (genuine local
            # commits) is a judgment call, not a process failure.
            reason_class = "judgment" if deterministic_judgment else "mechanical"
            state = _escalate_issue(
                state,
                issue_number,
                reason=(failure_kind if immediate_escalation else "redispatch_cap_exceeded"),
                reason_class=reason_class,
                issue_extra={
                    "redispatch_at": redispatch_at,
                    "pre_review_rework_reason": reason,
                },
            )
            write_gate.save_state(state)
            edge = _escalation_edge("redispatch_escalated", reason_class)
            result = write_gate.transition(gh, config.labels, issue_number, edge)
            if result.outcome != TransitionOutcome.APPLIED:
                entry = state["issues"].get(str(issue_number), {})
                if isinstance(entry, dict):
                    entry = {
                        **entry,
                        "label_error": {
                            "edge": edge,
                            "outcome": result.outcome.value,
                            "add_failures": result.add_failures,
                            "remove_failures": result.remove_failures,
                        },
                    }
                    state["issues"][str(issue_number)] = entry
                    write_gate.save_state(state)
            return {
                "issue_number": issue_number,
                "pr_number": pr_number,
                "reason": reason,
                "escalated": True,
                "escalation_reason": state["issues"][str(issue_number)]["escalation_reason"],
            }

        repo_root = getattr(gh, "repo_root", None)
        _write_rework_prompt(state_file, pr, issue_number, summary, config, repo_root=repo_root)
        entry = {
            **entry,
            "number": issue_number,
            "status": "rework_requested",
            "dispatched_at": None,
            "pre_review_rework_reason": reason,
        }
        state["issues"][str(issue_number)] = entry
        state["prs"][str(pr_number)] = {
            **state["prs"].get(str(pr_number), {}),
            "number": pr_number,
            "issue_number": issue_number,
            "status": "rework_requested",
        }
        state = write_gate.append_event(
            state,
            # event-consumer: audit-only -- records a rework routing decision already
            # applied inline above (status set to rework_requested); no downstream consumer needed
            "pre_review_rework_routed",
            {
                "issue_number": issue_number,
                "pr_number": pr_number,
                "reason": reason,
                "failure_kind": failure_kind,
            },
        )
        write_gate.save_state(state)

    result = write_gate.transition(gh, config.labels, issue_number, "rework_requested")
    label_error = None
    if result.outcome != TransitionOutcome.APPLIED:
        label_error = {
            "edge": "rework_requested",
            "outcome": result.outcome.value,
            "add_failures": result.add_failures,
            "remove_failures": result.remove_failures,
        }
        with state_lock(state_file):
            state = load_state(state_file)
            entry = state["issues"].get(str(issue_number), {})
            if isinstance(entry, dict):
                entry = {**entry, "label_error": label_error}
                state["issues"][str(issue_number)] = entry
                write_gate.save_state(state)

    return {
        "issue_number": issue_number,
        "pr_number": pr_number,
        "reason": reason,
        "label_error": label_error,
    }
