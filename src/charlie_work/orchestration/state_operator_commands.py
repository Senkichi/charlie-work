"""Operator-command delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 4 (issue #1647, parent #1632, umbrella
#1582). The public operator commands ``claim``, ``merge_authorize``,
``unescalate`` and ``ack_unauthorized_merge`` -- invoked from ``cli.py`` as
``app.<name>(...)`` -- moved verbatim from ``OrchestratorApp`` in
``charlie_work.workflow``; the ``workflow_delegation`` installer re-attaches
each ``def`` onto the class.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
import json

from charlie_work import layout
from charlie_work.github import GitHubError
from charlie_work.labels import TransitionOutcome
from charlie_work.review_decision import record_decision
from charlie_work.state import (
    PASSIVE_OPEN_STATUS,
    SINK_STATUSES,
    _REVIEW_STALE_CLAIM_TIMEOUT_MINUTES,
)
from charlie_work.worktree import (
    OPERATOR_MARKER_KIND,
    OPERATOR_MARKER_SESSION_ID,
    WORKTREE_UNSAFE_KINDS,
)
import charlie_work.workflow as _wf


def claim(self, issue_number: int, release: bool = False) -> _wf.CommandResult:
    """Record or release an operator claim on an issue.

    A claimed issue is excluded from fresh dispatch and rework dispatch
    regardless of its labels. When a worktree for the issue already
    exists, a ``.charlie-writer.json`` marker is written or removed so the
    protection is mutual: the orchestrator refuses to dispatch into a
    worktree with a live foreign writer marker.
    """
    try:
        issue = self.gh.issue_view(issue_number)
    except GitHubError as exc:
        return _wf.CommandResult(
            False, f"issue #{issue_number} not found: {exc}", {"issue_number": issue_number}
        )

    branch_name = self._branch_name(issue)
    worktree_path = _wf.worktree_path_for_branch(
        self.repo_root, branch_name, self._layout.worktrees
    )

    marker_written = False
    if not release and worktree_path.is_dir():
        # Operator markers intentionally do not encode the CLI's transient
        # PID; liveness is keyed off operator_claimed_at in state.json.
        try:
            _wf.write_worktree_marker(
                worktree_path,
                0,
                OPERATOR_MARKER_SESSION_ID,
                kind=OPERATOR_MARKER_KIND,
            )
            marker_written = True
        except OSError:
            # Marker write failure is not fatal, but record it in the result.
            pass
    elif release:
        # Best-effort marker removal; state release is what matters. Only
        # remove a marker that belongs to this operator claim, never a
        # worker marker from an active session.
        try:
            _wf.remove_worktree_marker(worktree_path, session_id=OPERATOR_MARKER_SESSION_ID)
        except OSError:
            pass

    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        if release:
            state = _wf.release_operator_claimed(state, issue_number)
        else:
            state = _wf.set_operator_claimed(state, issue_number)
        state = self._record_event(
            state,
            "operator_claim_released" if release else "operator_claim",
            {
                "issue_number": issue_number,
                "branch_name": branch_name,
                "worktree_path": str(worktree_path),
                "marker_written": marker_written,
            },
        )
        _wf.save_state(self.paths.state_file, state)

    message = (
        f"operator claim released for issue #{issue_number}"
        if release
        else f"operator claim recorded for issue #{issue_number}"
    )
    return _wf.CommandResult(
        True,
        message,
        {
            "issue_number": issue_number,
            "branch_name": branch_name,
            "worktree_path": str(worktree_path),
            "released": release,
            "marker_written": marker_written,
        },
    )


def merge_authorize(
    self,
    pr_number: int,
    reason: str,
    *,
    by: str | None = None,
    sha: str | None = None,
) -> _wf.CommandResult:
    """Record an operator's explicit authorization to merge a worker PR (issue #934).

    The unauthorized-merge tripwire (#673) and the ``merge-check`` preflight
    (#894) both infer authorization from ``decision == "approved"`` and
    ``reviewed_head_sha == live_head_sha``. An operator who legitimately
    adjudicates a PR whose recorded decision is stale, absent, or pending —
    and merges it — has no way to record that adjudication, so every
    legitimate operator merge becomes a tripwire finding that pins
    ``ok=False`` on every subsequent pass until someone writes a
    retrospective ack. The authorized path and the unauthorized path are
    indistinguishable by construction.

    This writes an ``authorized_override`` into the PR's
    ``review-decision.json``: ``{by, reason, authorized_sha,
    authorized_at}``. The tripwire and ``merge-check`` treat an override
    whose ``authorized_sha`` matches the live head as explicit
    authorization (via ``_authorized_override_matches``), so the control
    reads a **recorded** authorization rather than inferring one.

    Properties (from #934):

    - **Does not weaken the control.** This adds a way to *record*
      authorization; it must not add a way to *skip* the check. An
      unrecorded merge is still a finding. The override is a field in the
      decision record, not a bypass — both ``_detect_unauthorized_merges``
      and ``merge_check`` still run their full logic; they just gain an
      additional authorization source.
    - **The reason stays mandatory**, matching ``tripwire ack`` — "a
      tripwire that can be silenced silently is no control" applies at
      least as strongly before the merge as after it.
    - **Bind to the SHA.** Two of the four Class B cases (#802, #804) were
      flagged specifically because a rebase moved the head after the
      decision was recorded. The override names the exact SHA it
      authorizes; a rebase after authorization moves the head and
      invalidates the override, exactly as it invalidates an approved
      decision.

    The override is merge-updated into the existing decision record (not a
    fresh file), so the reviewer's original verdict is preserved alongside
    the operator's authorization. If no decision file exists, one is
    created with just the override — the reviewer verdict is absent, and
    the override is the authorization.
    """
    if not reason.strip():
        return _wf.CommandResult(
            False,
            "a non-empty --reason is required to record a merge authorization "
            "(a tripwire that can be silenced silently is no control)",
            {"pr": pr_number},
        )

    pr = self.gh.pr_view(pr_number)
    if not isinstance(pr, dict) or not pr:
        return _wf.CommandResult(
            False,
            f"PR #{pr_number}: cannot read PR from GitHub — refusing to authorize",
            {"pr": pr_number, "authorized": False, "reason": "pr_unreadable"},
        )

    authorized_sha = sha if sha is not None else pr.get("headRefOid")
    if not authorized_sha or not isinstance(authorized_sha, str):
        return _wf.CommandResult(
            False,
            f"PR #{pr_number}: no head SHA to bind the authorization to "
            "— refusing to record an unbound override",
            {"pr": pr_number, "authorized": False, "reason": "no_head_sha"},
        )

    decision_path = self.paths.prs / f"pr-{pr_number}" / "review-decision.json"
    override_payload = {
        "by": by,
        "reason": reason,
        "authorized_sha": authorized_sha,
        "authorized_at": _wf.utc_now(),
    }

    with _wf.state_lock(self.paths.state_file):
        # Merge-update the override into the existing decision record. If
        # the file is absent or unparseable, start from an empty base — the
        # override is the authorization, and the reviewer verdict (if any)
        # is preserved when it exists. The read is inside the ``state_lock``
        # (which ``record_review`` also holds) so the read-modify-write is
        # genuinely atomic against concurrent writers — a TOCTOU where
        # ``record_review`` overwrites the file between this read and the
        # write below would silently discard the override (issue #934
        # review finding).
        if decision_path.exists():
            try:
                with decision_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if not isinstance(existing, dict):
                    existing = {}
            except (OSError, json.JSONDecodeError):
                existing = {}
        else:
            existing = {}
        updated = {**existing, "authorized_override": override_payload}
        # Single writer (issue #1362 Stage 2): ``archive_round=False`` --
        # this patch only adds ``authorized_override``, not a new
        # reviewer verdict, so it must never mint a round itself. Before
        # the F1 fix above, the packet-build placeholder guaranteed a
        # round-1 always existed by the time an override could land, so
        # this patch merely deduped onto that round; now that the
        # placeholder no longer archives, a PR with no reviewer verdict
        # yet has zero archived rounds, and without this flag the
        # override patch would become the round-1 minter -- a phantom
        # "reviewer round" containing only an operator override.
        # head_sha=None: any ``reviewed_head_sha`` already on ``existing``
        # passes through unchanged.
        record_decision(decision_path.parent, updated, None, archive_round=False)
        state = _wf.load_state(self.paths.state_file)
        state = self._record_event(
            state,
            "merge_authorized",
            {
                "pr": pr_number,
                "by": by,
                "reason": reason,
                "authorized_sha": authorized_sha,
            },
        )
        _wf.save_state(self.paths.state_file, state)

    return _wf.CommandResult(
        True,
        f"PR #{pr_number}: recorded operator authorization to merge at head "
        f"{authorized_sha} (by {by or 'unknown'})",
        {
            "pr": pr_number,
            "authorized": True,
            "authorized_sha": authorized_sha,
            "authorized_by": by,
            "reason": reason,
            "decision_path": str(decision_path),
            "state_file": str(self.paths.state_file),
        },
    )


def unescalate(
    self,
    pr_number: int | None = None,
    issue_number: int | None = None,
    *,
    dry_run: bool = False,
) -> _wf.CommandResult:
    """Operator re-arm for a PR/issue parked in the human/operator sink.

    Escalation is deliberately terminal for every automated path (review()
    and record_review() both hard-stop on it); until this command existed
    the only recovery was hand-editing state.json and labels, which is
    exactly how the status/label desyncs this repair sweep keeps finding
    were produced. This is the sanctioned door back into the pipeline:

    - PR merged/closed on GitHub: normalize the record to that terminal
      state (finalization/reconcile handle the rest); no label changes.
    - PR open: reset status to the passive pr-open state, zero every
      attempt counter and frozen janitor/review cache, and apply the
      ``unescalated_pr_open`` label edge so the next pass re-reviews it
      from scratch. If the PR also carries a terminal review verdict
      (approved/request_changes/blocked) still valid at the live head --
      the defining shape of a "blocked" record, and reachable for
      "escalated" too -- that verdict is voided back to a "pending" stub
      (issue #1765 finding 1) the same way ``review()``'s own stale-head
      void does: archive it first, then overwrite with a head-stamped
      pending placeholder. Without this, ``review_queue()`` treats the
      still-valid verdict as "nothing to do" and skips the PR forever, so
      resetting the status to passive-open would report changed=True while
      leaving the PR permanently unreachable -- the alerting label is gone
      but nothing re-queues it, an invisible park replacing a visible one.
    - Issue with no live PR: drop the issue back to the never-dispatched
      baseline and strip workflow labels (``unescalated_requeued``) so
      dispatch treats it as fresh.
    - Issue carrying a merged-PR mention flag: the flag path
      (``dispatch_merged_pr_mention_flagged``) deliberately leaves
      ``status`` untouched, so a mention-flagged issue parks in the
      ``agent:human-needed`` sink WITHOUT a sink status — invisible to
      ``SINK_STATUSES`` and to this command's pre-#1803 stuck predicate.
      A flag recorded in a prior pass and not yet re-armed counts as
      stuck here; re-arming stamps ``mention_rearmed_at`` (the same
      durable marker the dispatch-side label-removal detection writes)
      so the mention-only dispatch exclusion lifts on the next pass.

    "Stuck" is derived from ``state.SINK_STATUSES`` (currently "escalated"
    and "blocked" -- both map to the same ``agent:human-needed`` label edge
    in ``labels.py``) plus the PR-only "janitor_blocked" status, rather than
    a second hardcoded list here: issue #1765 found that "blocked" (added by
    #1642's human-decision-marker verdict path) had been left out of this
    predicate, so ``unescalate`` silently no-op'd (ok=True) on exactly the
    records an operator was trying to re-arm. Deriving from the shared set
    means a future sink status cannot be missed by this command alone.

    Idempotent: a record whose status is outside the sink (and, for a PR,
    not "janitor_blocked") is a no-op (ok=True) -- this is the legitimate,
    tested "nothing to do" case, distinct from the #1765 bug above. ``dry_run``
    computes and reports the full transition map without touching state,
    labels, or events.
    """
    if pr_number is None and issue_number is None:
        return _wf.CommandResult(False, "unescalate requires --pr and/or --issue", {})

    state = _wf.load_state_locked(self.paths.state_file)

    # Resolve the PR/issue pair from whichever side was given.
    if pr_number is None and issue_number is not None:
        open_pr_numbers = sorted(
            int(k)
            for k, v in state.get("prs", {}).items()
            if isinstance(v, dict)
            and v.get("issue_number") == issue_number
            and v.get("status") not in ("merged", "closed")
            and k.isdigit()
        )
        if open_pr_numbers:
            pr_number = open_pr_numbers[0]
    pr_state = state.get("prs", {}).get(str(pr_number), {}) if pr_number is not None else {}
    if issue_number is None and pr_number is not None:
        issue_number = pr_state.get("issue_number")
    issue_state = (
        state.get("issues", {}).get(str(issue_number), {}) if issue_number is not None else {}
    )

    pr_status = pr_state.get("status")
    pr_stuck = pr_status in SINK_STATUSES or pr_status == "janitor_blocked"
    issue_stuck = issue_state.get("status") in SINK_STATUSES
    # Issue #1803: the merged-PR mention flag parks an issue in the same
    # human sink (agent:human-needed) without writing a sink status, so
    # SINK_STATUSES alone reports "nothing to unescalate" on exactly the
    # records this command exists to re-arm (the operator-visible form of
    # the jobcannon #377/#391 recovery path). Flagged in a prior pass and
    # not already re-armed is the stuck condition; an issue already
    # carrying mention_rearmed_at has nothing left to lift.
    mention_flagged = (
        isinstance(issue_state, dict)
        and bool(issue_state.get("merged_pr_mention_flagged_at"))
        and not issue_state.get("mention_rearmed_at")
    )
    if not pr_stuck and not issue_stuck and not mention_flagged:
        return _wf.CommandResult(
            True,
            f"nothing to unescalate (pr={pr_number} status="
            f"{pr_state.get('status')!r}, issue={issue_number} status="
            f"{issue_state.get('status')!r}, mention_flagged={mention_flagged})",
            {"pr": pr_number, "issue": issue_number, "changed": False},
        )

    # Ground truth decides the re-entry point.
    live_pr = self.gh.pr_view(pr_number) if pr_number is not None else {}
    live_pr_state = str((live_pr or {}).get("state") or "").upper()

    # Issue #214 precedent (reconcile's live_session_issue_numbers guard):
    # a verifiably live worker session means nothing here is stuck, it is
    # IN USE -- refuse entirely rather than re-arm around a running
    # process. Popping issue-side worker_pid/dispatched_at would blind
    # orphan-worker detection; resetting the PR side would zero the
    # conflict/no-op attempt counters for a rework cycle still in flight
    # (defeating the caps) and flip the PR to the passive reviewing
    # status, inviting a concurrent review() against the worker's
    # in-progress push. PR "janitor_blocked" + issue "dispatched" is the
    # NORMAL mid-rework steady state, not a wedge.
    #
    # Issue #625: "live" is no longer just "is the PID alive?". Both the
    # sidecar-based and state.json-based checks route through one
    # predicate (``issue_worker_liveness``) that bounds the state-side
    # check with the watchdog's stall standard -- an alive-but-silent
    # session (no real activity for > stall_minutes, or past the
    # wall-clock deadline with an inconclusive probe) is wedged, not
    # live, and unescalate may proceed. The refusal carries session age
    # and last-activity diagnostics so an operator can tell a wedged
    # worker from a working one.
    if issue_number is not None:
        from datetime import UTC

        from charlie_work.worker import issue_worker_liveness

        sessions_dir = self._layout.sessions_dir
        verdict = issue_worker_liveness(
            issue_number, issue_state, sessions_dir, self.config, datetime.now(UTC)
        )
    else:
        verdict = None
    if verdict is not None and verdict.live:
        return _wf.CommandResult(
            True,
            f"issue #{issue_number} has a live worker session; nothing to "
            f"unescalate (pr={pr_number} left untouched) -- {verdict.reason}",
            {
                "pr": pr_number,
                "issue": issue_number,
                "issue_worker_alive": True,
                "issue_worker_last_activity_at": verdict.last_activity_at,
                "issue_worker_last_activity_source": verdict.last_activity_source,
                "issue_worker_session_started_at": verdict.session_started_at,
                "issue_worker_pid": verdict.pid,
                "issue_worker_source": verdict.source,
                "changed": False,
            },
        )

    # Issue #849: a ``worktree_unsafe`` escalation is caused by bytes on
    # disk, not by a label or a PR state. Clearing the label without
    # inspecting the worktree reports success for an operation that
    # changed nothing causal — the next rework dispatch reproduces the
    # escalation deterministically. Re-run the safety check and refuse to
    # clear while the worktree still fails it.
    # Issue #807: ``worktree_unsafe`` is split into
    # ``worktree_unsafe_shim_dirt`` and ``worktree_unsafe_local_commits``;
    # both are covered by ``WORKTREE_UNSAFE_KINDS`` so the safety re-check
    # fires for either kind.
    if (
        issue_number is not None
        and issue_stuck
        and issue_state.get("escalation_reason") in WORKTREE_UNSAFE_KINDS
    ):
        unsafe_reason = self._worktree_still_unsafe(issue_number, state)
        if unsafe_reason:
            return _wf.CommandResult(
                True,
                f"issue #{issue_number} escalated as worktree_unsafe; "
                f"worktree is still unsafe ({unsafe_reason}) — clearing "
                f"the label would change nothing causal. Remove or "
                f"commit the worktree work before re-arming.",
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    "worktree_still_unsafe": True,
                    "worktree_unsafe_reason": unsafe_reason,
                    "changed": False,
                },
            )

    # Compute the TRANSFORMATION from the pre-fetch snapshot (for dry_run
    # reporting), then re-apply it to freshly-loaded entries inside the
    # write lock below -- gh.pr_view has real latency, and writing this
    # snapshot's dicts wholesale would clobber any field a concurrent
    # writer (e.g. a reconcile pass) touched in the meantime.
    transitions: dict[str, list[Any]] = {}
    label_edge: str | None = None

    pr_status_target: str | None = None
    if pr_number is not None and pr_stuck:
        if live_pr_state == "MERGED":
            pr_status_target = "merged"
        elif live_pr_state == "CLOSED":
            pr_status_target = "closed"
        else:
            pr_status_target = PASSIVE_OPEN_STATUS

    # Issue #1765: resetting the PR to the passive-open state is meant to
    # make review_queue() reachable again, but review_queue() skips any PR
    # whose recorded verdict is a terminal decision (approved/
    # request_changes/blocked) still valid at the live head -- exactly the
    # property a "blocked" verdict pinned to an unmoved head has by
    # construction (that is the defining shape of the state this branch
    # widened ``pr_stuck`` to cover). Left alone, unescalate reports
    # changed=True, strips the alerting ``agent:human-needed`` label, and
    # leaves the PR permanently unreachable except via a second, explicit
    # ``charlie verdict`` or ``why-charlie-hate --force-rereview`` call --
    # an invisible park replacing a visible one, the same rc=0-silent-no-op
    # class this command exists to fix, moved one hop downstream. Detected
    # here (pre-lock, alongside the other network reads) via the same
    # predicate ``review_verdict_guard`` uses to refuse a destructive
    # re-review; the identity is re-checked inside the write lock below
    # before anything is actually voided, so a verdict that changed
    # underneath this call (e.g. a concurrent ``record_review``) is left
    # alone rather than clobbered with a stale "pending" stub.
    #
    # A still-valid "approved" verdict is deliberately excluded from voiding
    # here (unlike ``review_verdict_guard``, which must protect all three
    # decision values from a destructive manual re-review). "blocked"/
    # "request_changes" are terminal NEGATIVE outcomes -- the verdict itself
    # is why review_queue() has nothing to do, so it must be voided to make
    # the PR reachable again. "approved" means review has nothing left to
    # do BY DESIGN; when an approved PR is stuck it is stuck for an
    # unrelated reason (e.g. a conflict-rework-attempts cap), and
    # merge_ready()'s own conflict-detection/rework-dispatch lane requires
    # ``approved`` to be true to run at all. Voiding it here would silently
    # disable that lane on every subsequent merge_ready() pass until a full
    # fresh review completes -- the conflict-cap re-arm this command exists
    # to restore (issue #776 follow-up) would be dead on arrival.
    still_valid_verdict = (
        self._still_valid_recorded_verdict(pr_number, live_pr.get("headRefOid"))
        if pr_status_target == PASSIVE_OPEN_STATUS
        else None
    )
    if still_valid_verdict is not None and still_valid_verdict[0].get("decision") == "approved":
        still_valid_verdict = None

    def _apply_pr_reset(entry: dict[str, Any]) -> dict[str, Any]:
        updated = dict(entry)
        updated["status"] = pr_status_target
        if pr_status_target == PASSIVE_OPEN_STATUS:
            updated["review_dispatch_attempt_count"] = 0
            updated["request_changes_count"] = 0
            for field_name in self._UNESCALATE_PR_RESET_FIELDS:
                if field_name in ("review_dispatch_attempt_count", "request_changes_count"):
                    continue
                updated.pop(field_name, None)
        elif pr_status_target == "merged":
            # Issue #747: stamp ``merged_at`` only on a genuine non-merged
            # -> merged transition (the same pattern as the other five
            # merged_at sites) so an unescalate that re-observes a PR
            # already recorded as merged does not back-date the original
            # observation time. ``entry`` is the pre-reset entry, so its
            # ``status`` is the prior state, not the target just assigned.
            if entry.get("status") != "merged":
                updated["merged_at"] = _wf.utc_now()
        return updated

    issue_status_action: str = "leave"
    if issue_number is not None:
        if live_pr_state == "OPEN" and pr_number is not None:
            issue_status_action = "passive"
            label_edge = "unescalated_pr_open"
        elif live_pr_state in ("MERGED", "CLOSED"):
            # Terminal PR on GitHub. If the issue is still escalated and
            # no other open PR references it, drop the issue to baseline
            # in the same call — reconcile deliberately never rewrites an
            # open escalated issue's status (D-2), so leaving it would
            # require a second identical ``unescalate --issue N`` call to
            # take the no-live-PR path (issue #1391). When another open PR
            # exists, or the issue is not stuck, leave the issue to
            # finalization/reconcile as before.
            if (issue_stuck or mention_flagged) and not _wf._has_other_open_pr(
                state, issue_number, pr_number
            ):
                issue_status_action = "drop"
                label_edge = "unescalated_requeued"
            else:
                label_edge = None
        elif issue_stuck or mention_flagged:
            # No live PR at all — back to the never-dispatched baseline
            # (a status literal no dispatch selector reads would just
            # recreate the orphan gap reconcile now repairs). For a
            # mention-flagged issue "drop" is a no-op on ``status`` (the
            # flag path never writes one); what matters is the
            # ``unescalated_requeued`` edge stripping agent:human-needed
            # and the mention_rearmed_at stamp below.
            issue_status_action = "drop"
            label_edge = "unescalated_requeued"

    def _apply_issue_reset(entry: dict[str, Any]) -> dict[str, Any]:
        updated = dict(entry)
        if issue_status_action == "passive":
            updated["status"] = PASSIVE_OPEN_STATUS
        elif issue_status_action == "drop":
            updated.pop("status", None)
        for field_name in self._UNESCALATE_ISSUE_RESET_FIELDS:
            updated.pop(field_name, None)
        return updated

    if pr_number is not None and pr_stuck:
        snapshot_new_pr = _apply_pr_reset(pr_state)
        if snapshot_new_pr.get("status") != pr_state.get("status"):
            transitions["pr.status"] = [pr_state.get("status"), snapshot_new_pr["status"]]
    if still_valid_verdict is not None:
        transitions["pr.review_decision"] = [still_valid_verdict[0].get("decision"), "pending"]
    if issue_number is not None:
        snapshot_new_issue = _apply_issue_reset(issue_state)
        if snapshot_new_issue.get("status") != issue_state.get("status"):
            transitions["issue.status"] = [
                issue_state.get("status"),
                snapshot_new_issue.get("status"),
            ]

    # Issue #1602: capture the in-window blocked-environment count from
    # the pre-lock snapshot so the ``unescalated`` event records what the
    # release cleared. ``blocked_environment_at`` is popped by
    # ``_apply_issue_reset`` (via ``UNESCALATE_ISSUE_RESET_FIELDS``), so
    # without this the audit trail would not show why a subsequent
    # rework pass dispatched instead of re-escalating.
    prior_blocked_environment_count = (
        len(
            _wf._windowed_blocked_environment_at(
                issue_state,
                window_minutes=self.config.watchdog.redispatch_window_minutes,
            )
        )
        if isinstance(issue_state, dict)
        else 0
    )

    if dry_run:
        return _wf.CommandResult(
            True,
            f"dry-run: would unescalate pr={pr_number} issue={issue_number} "
            f"(label edge: {label_edge})",
            {
                "pr": pr_number,
                "issue": issue_number,
                "transitions": transitions,
                "label_edge": label_edge,
                "mention_rearmed": mention_flagged,
                "blocked_environment_at_reset": prior_blocked_environment_count > 0,
                "blocked_environment_at_prior_count": prior_blocked_environment_count,
                "verdict_would_be_voided": still_valid_verdict is not None,
                "changed": False,
            },
        )

    verdict_voided = False
    mention_rearmed = False
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        if pr_number is not None and pr_stuck:
            fresh_pr = state["prs"].get(str(pr_number), {})
            state["prs"][str(pr_number)] = {
                **_apply_pr_reset(fresh_pr if isinstance(fresh_pr, dict) else {}),
                "number": pr_number,
            }
        if still_valid_verdict is not None:
            voided_decision, _reason = still_valid_verdict
            # Compare-and-swap against the pre-lock snapshot (same guard
            # ``_update_approval_head`` applies before its own record_decision
            # call): a concurrent record_review landing a fresh verdict
            # between the pre-lock read and this write must win, not be
            # clobbered by a stale "pending" stub for a decision that no
            # longer exists.
            live_decision = self._review_decision(pr_number)
            if live_decision.get("decision") == voided_decision.get(
                "decision"
            ) and live_decision.get("reviewed_head_sha") == voided_decision.get(
                "reviewed_head_sha"
            ):
                pr_dir = self.paths.prs / f"pr-{pr_number}"
                # Issue #1695 precedent (review()'s own stale-head void):
                # preserve the verdict being voided in the rounds archive
                # before the pending stub overwrites the flat file. A
                # carried-forward re-pin lives ONLY in the flat file
                # (``_update_approval_head`` writes ``archive_round=False``),
                # so overwriting it without archiving first would destroy
                # the sole copy. ``head_sha=None`` leaves the payload's own
                # ``reviewed_head_sha`` untouched -- re-stamping it here
                # would fabricate a verdict pin that was never recorded.
                # Note: state.json's ``prs[N].decision``/``reviewed_head_sha``
                # cache is deliberately NOT mirrored here -- only six
                # sanctioned sites may write those keys
                # (test_state_decision_cache_enforcement.py), and this write
                # is not one of them. The next loop() pass's
                # ``_refresh_pr_decision_cache`` boundary refresh resyncs it
                # from this file, the same declared-cache path every other
                # untracked-PR pass already relies on.
                record_decision(
                    pr_dir,
                    {
                        **voided_decision,
                        "verdict_provenance": voided_decision.get("verdict_provenance"),
                    },
                    None,
                    archive_round=True,
                )
                record_decision(
                    pr_dir,
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "decision": "pending",
                        "summary": "",
                        "required_changes": [],
                        "reviewed_at": None,
                        "verdict_provenance": None,
                    },
                    live_pr.get("headRefOid"),
                    archive_round=False,
                )
                verdict_voided = True
        if issue_number is not None:
            fresh_issue = state["issues"].get(str(issue_number), {})
            state["issues"][str(issue_number)] = {
                **_apply_issue_reset(fresh_issue if isinstance(fresh_issue, dict) else {}),
                "number": issue_number,
            }
            # Issue #1803: a mention-flagged issue needs the durable
            # re-arm marker, not just the label edge — the mention-only
            # dispatch exclusion lifts on mention_rearmed_at (or on the
            # human-needed removal the edge performs, but the stamp makes
            # the re-arm observable and immune to a stale label snapshot).
            # Re-checked on the fresh entry so a marker stamped between
            # the pre-lock read and this write is left alone. Routed
            # through _stamp_mention_rearm so the write goes through the
            # write gate and emits the same
            # dispatch_merged_pr_mention_rearmed event the dispatch-side
            # detection emits — operators see one re-arm event shape
            # regardless of which door performed it.
            fresh_mention_flagged = (
                isinstance(fresh_issue, dict)
                and bool(fresh_issue.get("merged_pr_mention_flagged_at"))
                and not fresh_issue.get("mention_rearmed_at")
            )
            if fresh_mention_flagged:
                state = self._stamp_mention_rearm(state, [issue_number])
                mention_rearmed = True
        state = self._record_event(
            state,
            "unescalate",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "transitions": transitions,
                "label_edge": label_edge,
                "mention_rearmed": mention_rearmed,
                "blocked_environment_at_reset": prior_blocked_environment_count > 0,
                "blocked_environment_at_prior_count": prior_blocked_environment_count,
                "verdict_voided": verdict_voided,
            },
        )
        _wf.save_state(self.paths.state_file, state)

    label_error = None
    if label_edge is not None and issue_number is not None:
        result = _wf.transition(self.gh, self.config.labels, int(issue_number), label_edge)
        if result.outcome != TransitionOutcome.APPLIED:
            label_error = {
                "edge": label_edge,
                "outcome": result.outcome.value,
                "add_failures": result.add_failures,
                "remove_failures": result.remove_failures,
            }
            with _wf.state_lock(self.paths.state_file):
                state = _wf.load_state(self.paths.state_file)
                entry = state["issues"].get(str(issue_number), {})
                state["issues"][str(issue_number)] = {
                    **(entry if isinstance(entry, dict) else {}),
                    "number": issue_number,
                    "label_error": label_error,
                }
                _wf.save_state(self.paths.state_file, state)

    summary = ", ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in transitions.items())
    message = f"unescalated pr={pr_number} issue={issue_number}"
    if summary:
        message += f" ({summary})"
    if mention_rearmed:
        message += " (mention flag re-armed)"
    if label_error:
        message += f" (label update failed: {label_error['outcome']})"
    return _wf.CommandResult(
        True,
        message,
        {
            "pr": pr_number,
            "issue": issue_number,
            "transitions": transitions,
            "label_edge": label_edge,
            "label_error": label_error,
            "mention_rearmed": mention_rearmed,
            "verdict_voided": verdict_voided,
            "changed": True,
        },
    )


def ack_unauthorized_merge(
    self, pr_number: int, reason: str, *, by: str | None = None
) -> _wf.CommandResult:
    """Acknowledge a post-arming unauthorized-merge finding so it stops pinning ok=False.

    The #502 tripwire's pre-arming baseline (``_apply_unauthorized_merge_baseline``)
    suppresses history once. It has no equivalent for *post-arming* findings, so a
    single confirmed, already-actioned bypass is re-detected and re-appended to
    ``loop()``'s ``errors`` bucket on every pass for as long as the merged PR stays
    inside ``merged_pr_list()``'s 500-PR REST window — pinning ``ok=False`` forever
    and drowning any new signal (issue #673).

    This records an explicit acknowledgment in ``state.json`` under
    ``unauthorized_merge_acknowledged`` (a ``{pr_number: {acknowledged_at, reason,
    by}}`` map). The tripwire filters acknowledged PRs out of its reported
    candidates the same way it filters pre-arming history, so the finding keeps its
    bite until acked and then goes quiet, freeing ``ok=False`` / ``errors`` to mean
    "there is something new to look at" again.

    Acknowledgment is never automatic — that would defeat the tripwire. It requires
    this explicit action (exposed as ``charlie tripwire ack``), mirroring how
    ``agent:human-needed`` requires explicit human action to clear. A non-empty
    ``reason`` is mandatory: a tripwire that can be silenced silently is no control.
    Re-acking the same PR updates the record (new reason/by/timestamp) rather than
    duplicating or refusing, so a finding's triage state can be corrected.
    """
    if not reason.strip():
        return _wf.CommandResult(
            False,
            "a non-empty --reason is required to acknowledge an unauthorized-merge "
            "finding (a tripwire that can be silenced silently is no control)",
            {"pr": pr_number},
        )

    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        acks = state.get(_wf.UNAUTHORIZED_MERGE_ACK_KEY)
        if not isinstance(acks, dict):
            acks = {}
        acks[str(pr_number)] = {
            "acknowledged_at": _wf.utc_now(),
            "reason": reason,
            "by": by,
        }
        state[_wf.UNAUTHORIZED_MERGE_ACK_KEY] = acks
        # Drop this PR's first-detection record (issue #933). That map exists
        # solely to make `unauthorized_merge_detected` fire once per finding
        # instead of once per pass; an ack ends the finding, so the entry has
        # no remaining job. Clearing it here is what makes the map mean
        # "announced findings still open" rather than "every PR ever found",
        # which keeps it bounded and — the reason this is code and not a
        # comment — makes re-detection after an ack is *withdrawn* announce
        # again. There is no revoke command today; the one revocation on
        # record (2026-08-02, the #895 misrouted acks) was an operator edit.
        # Were the entry left behind, whoever adds that command would inherit
        # a finding that silently re-pins ok=False with no fresh event.
        detected = state.get(_wf.UNAUTHORIZED_MERGE_DETECTED_KEY)
        if isinstance(detected, dict) and str(pr_number) in detected:
            state[_wf.UNAUTHORIZED_MERGE_DETECTED_KEY] = {
                key: value for key, value in detected.items() if key != str(pr_number)
            }
        state = self._record_event(
            state,
            "unauthorized_merge_acknowledged",
            {"pr": pr_number, "reason": reason, "by": by},
        )
        _wf.save_state(self.paths.state_file, state)

    # Echo the resolved state file (issue #895). An ack is a write to a
    # security control's audit record, and the operator's only evidence of
    # *which repo* received it was previously the exit code. Naming the
    # path makes a misrouted ack visible in the output that reports success.
    return _wf.CommandResult(
        True,
        f"acknowledged unauthorized-merge finding for PR #{pr_number} "
        f"in {self.paths.state_file}; it will no longer pin ok=False",
        {
            "pr": pr_number,
            "reason": reason,
            "by": by,
            "state_file": str(self.paths.state_file),
        },
    )


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
